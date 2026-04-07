# Step 03 — Request Entry, Exit, and Scheduling

> **Files covered** (in reading order):
> ```
> vllm/v1/engine/input_processor.py   444 L  ← entry gate: raw prompt → EngineCoreRequest
> vllm/v1/engine/output_processor.py  807 L  ← exit gate:  token IDs  → RequestOutput
> vllm/v1/core/sched/scheduler.py    2305 L  ← core:       batch scheduling
> ```
>
> **Roadmap position**: the complete CPU-side request lifetime.
> ```
> User
>  │
>  ▼  InputProcessor.process_inputs()          ← §1
>  │  EngineCoreRequest
>  ▼  EngineCore → Scheduler.add_request()     ← §3–8
>  │  SchedulerOutput
>  ▼  execute_model()
>  │  EngineCoreOutputs  (raw token IDs)
>  ▼  OutputProcessor.process_outputs()        ← §2
>  │  RequestOutput  (decoded text)
> User
> ```

---

## 1. InputProcessor (`vllm/v1/engine/input_processor.py`)

```
LLMEngine.add_request()
│
▼
InputProcessor.process_inputs()   ← 你現在要學的 ⭐
│
▼
EngineCoreRequest（含 prompt_token_ids / mm_features / sampling_params ...）
```

### Where it sits

`InputProcessor` lives in the **frontend process** (same as `LLMEngine`). It never crosses the IPC boundary. Its job: transform a raw user request into a fully-validated `EngineCoreRequest` the scheduler can consume.

`InputProcessor` 是 **LLMEngine.add_request()** 收到用戶輸入後的**第一道真正處理關卡**。  
它負責把「人類可讀的 prompt（文字 / tokens / 多模態）」轉換成「引擎內部可直接使用的 `EngineCoreRequest`」。

**核心職責（對照 LLMEngine 筆記）**：
- 參數驗證（SamplingParams / PoolingParams）
- LoRA 相容性檢查
- Tokenization + Multimodal 預處理（委託給 `InputPreprocessor`）
- Encoder-Decoder 模型特殊處理
- Prompt 長度 / 詞彙表檢查
- 多模態特徵打包（`MultiModalFeatureSpec`）
- 產生內部唯一 request ID（加隨機後綴）
- 最終打包成 `EngineCoreRequest`

### `process_inputs()` — the full pipeline (line 195)

```
process_inputs(request_id, prompt, params, ...)
│
├─ 1. _validate_params()
│      SamplingParams.verify() — max_tokens bounds, temperature, structured output config
│      PoolingParams.verify()  — task type supported by this model?
│
├─ 2. _validate_lora()
│      LoRA enabled? tokenizer compatibility warning
│
├─ 3. Tokenize  (only if raw text/tokens passed; new-style callers pass pre-rendered EngineInput)
│      InputPreprocessor.preprocess(prompt, tokenization_kwargs)
│      handles: str, list[int], TextPrompt, TokensPrompt, multimodal dicts
│
├─ 4. split_enc_dec_input()
│      encoder-decoder models: split into separate encoder / decoder inputs
│
├─ 5. _validate_model_inputs()
│      _validate_prompt_len():
│        raise if prompt_len == 0
│        raise if prompt_len > max_model_len
│        raise if prompt_len == max_model_len  (no room for even one output token)
│      token ID range check:
│        raise if any token_id > max(tokenizer.max_token_id, vocab_size - 1)
│
├─ 6. Multimodal packaging  (only if mm inputs present)
│      argsort_mm_positions() — sort images/audio by their position in the sequence
│      build list[MultiModalFeatureSpec]:
│        { modality, data, mm_hash, identifier, mm_position }
│
├─ 7. SamplingParams finalisation
│      sampling_params.update_from_generation_config(generation_config_fields, eos_token_id)
│        fills in default stop tokens, repetition penalties from the model's generation_config
│      sampling_params.update_from_tokenizer(tokenizer)
│        merges EOS / BOS token IDs from the tokenizer vocabulary
│      if max_tokens is None:
│        max_tokens = max_model_len - prompt_len   ← auto-fill
│
└─ 8. Build and return EngineCoreRequest(
          request_id, prompt_token_ids, prompt_embeds,
          mm_features, sampling_params, pooling_params,
          arrival_time, lora_request, cache_salt, priority, ...)
```

### Request ID randomisation (line 175)

```python
@staticmethod
def assign_request_id(request: EngineCoreRequest):
    request.external_req_id = request.request_id       # preserve user-supplied ID
    request.request_id = f"{request.external_req_id}-{random_uuid():.8}"
```

Why: two clients might accidentally reuse the same `request_id`. Appending 8 random chars guarantees internal uniqueness. The original ID is saved in `external_req_id` and is what the user sees in `RequestOutput.request_id`.

### What `EngineCoreRequest` carries into the scheduler

```python
@dataclass
class EngineCoreRequest:
    request_id:         str                       # internal (randomised)
    external_req_id:    str | None                # original user-supplied ID
    prompt_token_ids:   list[int] | None          # None when using prompt embeddings
    prompt_embeds:      torch.Tensor | None       # alternative to token IDs
    mm_features:        list[MultiModalFeatureSpec] | None
    sampling_params:    SamplingParams | None
    pooling_params:     PoolingParams | None
    arrival_time:       float
    lora_request:       LoRARequest | None
    cache_salt:         str | None                # per-request prefix cache key override
    priority:           int                       # for PriorityRequestQueue
    data_parallel_rank: int | None
```

The scheduler's `Request` object is built from this inside `EngineCore.add_request()`.
```python
request = self.input_processor.process_inputs(...)   # ← 關鍵呼叫

self.output_processor.add_request(...)
self.engine_core.add_request(request)                # ← 送進 EngineCore
```

### 架構洞察
#### 三大子系統在 Input 端的協作
```text
用戶 prompt
    │
    ▼
InputPreprocessor   ← 真正做 tokenize / multimodal processing
    │
    ▼
InputProcessor      ← 驗證 + 打包 + mm_feature 排序 + ID 處理
    │
    ▼
EngineCoreRequest   ← 交給 EngineCoreClient.add_request()
```

### 設計原則
* Fail Fast：所有能提前驗證的錯誤都在這裡解決
* 單一責任：InputProcessor 只負責「輸入轉內部格式」，不負責 scheduling 或 inference
* 可擴展：InputPreprocessor 是 renderer 層的抽象，未來支援新模態只需改 renderer

---

## 2. OutputProcessor (`vllm/v1/engine/output_processor.py`)
```
EngineCore.get_output() → EngineCoreOutputs（raw tokens）
│
▼
OutputProcessor.process_outputs()   ← 你現在要學的 ⭐
│
▼
RequestOutput / PoolingRequestOutput（含 text、finish_reason、logprobs）
```
### Where it sits

`OutputProcessor` also lives in the **frontend process**. It receives raw token IDs from `EngineCore` and converts them into decoded text + metadata.

`OutputProcessor` 是 **LLMEngine.step()** 從 EngineCore 拿到原始輸出後的**最後一道處理關卡**。  
它負責把「引擎內部的 raw token IDs + finish reason」轉換成「用戶最終看到的 `RequestOutput` / `PoolingRequestOutput`」。

**核心職責**：
- 維護每個 request 的狀態（`RequestState`）
- Incremental detokenization（邊生成邊解碼）
- Stop-string / stop-token 檢測
- Streaming 控制（`DELTA` vs `FINAL_ONLY` + `stream_interval`）
- Logprobs 處理
- Parallel sampling（n > 1）的 parent/child 聚合
- 請求完成後的 abort 回報 + 統計更新
- 支援 streaming input（prompt 邊生成邊更新）


**設計觀察**：  
跟 `InputProcessor` 一樣，這層也是**薄協調 + 後處理**。真正重的 detokenize 工作委託給 `IncrementalDetokenizer` 和 `LogprobsProcessor`。所有「輸出端」的複雜邏輯（stop string、streaming interval、parent aggregation）都在這裡完成。

**調用鏈全景圖**
```text
LLMEngine.step()
│
├── 1️⃣ outputs = self.engine_core.get_output()
│
├── 2️⃣ processed = self.output_processor.process_outputs(
│       outputs.outputs,
│       engine_core_timestamp=...,
│       iteration_stats=...)
│       ← 這裡！
│
├── 3️⃣ self.engine_core.abort_requests(processed.reqs_to_abort)
│
└── return processed.request_outputs
```

**OutputProcessor 內部主要流程**
```text
process_outputs(engine_core_outputs)
│
├── for 每個 EngineCoreOutput：
│       ├── 取出 RequestState（或略過已完成）
│       ├── 更新 stats
│       ├── detokenizer.update(new_token_ids)
│       ├── 檢查 stop_string / finish_reason
│       ├── 決定是否要 emit output（考慮 stream_interval）
│       └── 呼叫 make_request_output() → 產生 RequestOutput
│
├── 收集 reqs_to_abort（stop string 觸發的）
└── 返回 OutputProcessorOutput(request_outputs, reqs_to_abort)
```


### Key state: `RequestState`

For every in-flight request, `OutputProcessor` keeps a `RequestState` (line 129):

```python
class RequestState:
    request_id:         str           # internal ID
    external_req_id:    str           # user-supplied ID (used in RequestOutput)
    prompt:             str | None    # original prompt text (for RequestOutput)
    prompt_token_ids:   list[int]

    detokenizer:        IncrementalDetokenizer | None   # token IDs → text
    logprobs_processor: LogprobsProcessor | None

    output_kind:        RequestOutputKind   # DELTA or FINAL_ONLY
    is_prefilling:      bool                # True until first output token
    queue: RequestOutputCollector | None    # non-None in AsyncLLM mode
    stream_interval:    int                 # emit every N tokens (default 1)
```

`RequestState` is created in `add_request()` (line 508) and removed in `_finish_request()`.

### `IncrementalDetokenizer` — why "incremental"?

You cannot call `tokenizer.decode([tok])` on each token independently. Some tokens only decode correctly in context — multi-byte UTF-8 sequences, BPE merges that span token boundaries. The detokenizer accumulates all output tokens and uses the tokenizer's incremental decoding logic to emit only the safely-decodable prefix.

```python
stop_string = detokenizer.update(new_token_ids, finished=False)
# → None if no stop string matched yet
# → "stop phrase" if a stop string was found in the decoded text

text_chunk = detokenizer.get_next_output_text(finished, delta=True)
# delta=True  → only new text since last call  (streaming)
# delta=False → full accumulated text so far   (batch)
```

### `process_outputs()` — the main loop (line 572, 被 LLMEngine.step() 直接呼叫)

這是整個 class 的心臟（被 LLMEngine.step() 直接呼叫）：

主要步驟：

* 遍歷每個 EngineCoreOutput
  * 取得對應 RequestState
  * 更新 iteration stats
  * req_state.detokenizer.update(new_token_ids, is_stop=...)
  * 偵測 stop string → 設定 finish_reason = STOP
  * 呼叫 req_state.make_request_output(...) 產生輸出物件
  * 如果需要 abort（stop string 觸發）→ 加入 reqs_to_abort
  * 處理 streaming input update（如果有 pending chunk）
  * 最後返回 OutputProcessorOutput

### The stop-string abort loop

Stop-string detection lives in `OutputProcessor` (frontend), **not** in `EngineCore`. Reason: detecting a stop string requires decoded text, which only exists in the frontend. The scheduler only sees token IDs.

Consequence: there is a one-step lag. EngineCore keeps scheduling a request until `OutputProcessor` puts it in `reqs_to_abort`, which then calls `scheduler.finish_requests(reqs_to_abort, FINISHED_ABORTED)`.

### `DELTA` vs `FINAL_ONLY`

| `output_kind` | Behaviour | Used by |
|---|---|---|
| `DELTA` | emit partial text every step (respecting `stream_interval`) | streaming APIs |
| `FINAL_ONLY` | `make_request_output()` returns `None` until finish; only one complete output | `LLM.generate()` |

### `RequestOutputCollector` — backpressure handling (line 45)

In `AsyncLLM` mode each request has a `RequestOutputCollector` queue. If the producer (OutputProcessor) gets ahead of the consumer (`generate()` coroutine), outputs are **merged** in DELTA mode — no tokens are lost under backpressure.

### `external_req_id` vs `request_id`

| | `request_id` | `external_req_id` |
|---|---|---|
| Created by | `InputProcessor.assign_request_id()` | user-supplied, saved at that point |
| Format | `"user-id-a3f8b2c1"` | `"user-id"` |
| Used in | scheduler, EngineCore, IPC | `RequestOutput.request_id` (what user sees) |

---

## 3. The One Invariant That Explains the Scheduler

The docstring at the top of `schedule()` (line 342):

```
There is no "decoding phase" nor "prefill phase" in the scheduler.
Each request just has num_computed_tokens and num_tokens_with_spec.
At each step the scheduler tries to assign tokens so that each request's
num_computed_tokens can catch up its num_tokens_with_spec.
```

| Field | Meaning |
|---|---|
| `num_computed_tokens` | tokens already processed (KV computed) |
| `num_tokens_with_spec` | tokens that *should* be processed (prompt + output so far + spec drafts) |

```
num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens
num_new_tokens = min(num_new_tokens, token_budget)
```

This covers every mode:
- **First prefill**: `num_computed_tokens = 0` → `num_new_tokens = len(prompt)`
- **Chunked prefill**: `num_new_tokens` clamped to budget; resumes next step
- **Decode**: `num_new_tokens = 1`
- **Speculative decode**: spec tokens inflate `num_tokens_with_spec`

---

## 4. Scheduler Package Layout

```
vllm/v1/core/sched/
├── scheduler.py        2305 L  ← Scheduler class
├── output.py            261 L  ← SchedulerOutput, NewRequestData, CachedRequestData
├── request_queue.py     208 L  ← FCFSRequestQueue, PriorityRequestQueue
├── interface.py         243 L  ← SchedulerInterface (abstract base)
└── async_scheduler.py    60 L  ← thin async wrapper

vllm/v1/core/
├── kv_cache_manager.py   ← block allocator (called by scheduler)
└── kv_cache_utils.py     ← block hashing, prefix matching
```

---

## 5. The Three Queues

```
self.waiting          RequestQueue   WAITING or PREEMPTED requests
self.skipped_waiting  RequestQueue   blocked on async deps (remote KV, grammar init…)
self.running          list[Request]  currently in-flight; ordered by arrival time
```

`_enqueue_waiting_request()` routes to `skipped_waiting` for blocked statuses, otherwise to `waiting`.

Global constraints:
- `max_num_running_reqs` — max concurrent sequences (e.g. 256)
- `max_num_scheduled_tokens` — max tokens per step (e.g. 8192)

---

## 6. Scheduler Data Structures (`output.py`)

### `SchedulerOutput` — message sent to the GPU worker each step

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs:           list[NewRequestData]   # first appearance: full payload
    scheduled_cached_reqs:        CachedRequestData       # repeats: diff only
    num_scheduled_tokens:         dict[str, int]          # {req_id: num_tokens}
    total_num_scheduled_tokens:   int
    scheduled_spec_decode_tokens: dict[str, list[int]]
    scheduled_encoder_inputs:     dict[str, list[int]]
    num_common_prefix_blocks:     list[int]               # for cascade attention
    finished_req_ids:             set[str]                # workers: drop cached state
    free_encoder_mm_hashes:       list[str]
    preempted_req_ids:            set[str] | None         # v2 model runner only
    new_block_ids_to_zero:        list[int] | None        # zero fresh GPU memory
```

### `NewRequestData` vs `CachedRequestData` — the bandwidth optimization

Workers maintain a persistent `InputBatch` on the GPU side. `NewRequestData` ships the **full payload** once:

```
req_id, prompt_token_ids, mm_features, sampling_params,
pooling_params, block_ids (full list), num_computed_tokens, lora_request
```

Every subsequent step, only the **delta** travels in `CachedRequestData`:

```
req_ids[]             — which requests
new_block_ids[]       — only newly allocated blocks
num_computed_tokens[] — where to slice the token buffer
num_output_tokens[]   — for tracking
resumed_req_ids (set) — preempted-and-resumed: replace block list, don't append
new_token_ids[]       — only used in pipeline-parallel mode
```

A 4096-token prompt (16 KB of ints) crosses the IPC pipe **exactly once**. All subsequent decode steps ship only a handful of scalars.

### `FCFSRequestQueue` vs `PriorityRequestQueue`

| | FCFS | Priority |
|---|---|---|
| Backing store | `deque` | `list` (min-heap via `heapq`) |
| Order key | arrival time | `(priority, arrival_time)` |
| `add_request` | `appendright` O(1) | `heappush` O(log n) |
| `pop_request` | `popleft` O(1) | `heappop` O(log n) |
| `prepend_request` | `appendleft` — to **front** | `heappush` — priority rules, no front |

The asymmetry matters for preemption: in FCFS a preempted request goes back to the front of `waiting` (FCFS order preserved). In Priority mode it just re-heaps.

---

## 7. Request Lifecycle (Scheduler side)

```
add_request()                   finish_requests() / update_from_output()
     │                                    │
     ▼                                    ▼
  WAITING ──── schedule() ──────► RUNNING ──── update_from_output() ──► FINISHED_*
     ▲              │
     │         (OOM / preempt)
     └─── PREEMPTED ◄── _preempt_request()
               (num_computed_tokens = 0)
```

### `add_request()` (line 1726)

- New request: `_enqueue_waiting_request()` + register in `self.requests` dict.
- No KV allocation here. Prefix-cache lookup happens in `schedule()` at promotion time.

### `finish_requests()` (line 1748)

Two-pass for efficiency:
1. Batch-collect running vs. waiting targets into sets.
2. `request.status = finished_status` → `_free_request()` → `kv_cache_manager.free()`.

`finished_req_ids` is added to here but not cleared until `_update_after_schedule()`, so it travels in the next `SchedulerOutput`.

### `_preempt_request()` (line 949)

```python
kv_cache_manager.free(request)        # all KV blocks returned to pool immediately
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0        # full reset — no swap, must recompute
request.num_preemptions += 1
self.waiting.prepend_request(request)  # back to front of queue
```

vLLM v1 has **no swap-to-CPU**. Preemption means losing all KV state and re-prefilling from scratch.

---

## 8. `schedule()` — The Algorithm (lines 341–942)

### Phase 1: RUNNING requests (lines 378–545)

```python
for request in self.running:
    num_new_tokens = (
        request.num_tokens_with_spec
        + request.num_output_placeholders
        - request.num_computed_tokens
    )
    num_new_tokens = min(num_new_tokens, token_budget)

    while True:
        new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
        if new_blocks is not None:
            break   # success
        # OOM → preempt
        if FCFS:   preempted = self.running.pop()       # last-in = most recently admitted
        else:      preempted = max(self.running, ...)   # lowest-priority
        _preempt_request(preempted, ...)
        if preempted == request:
            break   # preempted ourselves; give up
```

### Phase 2: WAITING requests (lines 556–843)

Only entered if no preemptions occurred this step.

```python
while (self.waiting or self.skipped_waiting) and token_budget > 0:
    if len(self.running) == max_num_running_reqs:
        break

    request = peek from queue

    # (1) Prefix cache lookup
    new_computed_blocks, num_cached = kv_cache_manager.get_computed_blocks(request)

    # (2) Tokens to schedule
    num_new_tokens = request.num_tokens - num_cached
    if not enable_chunked_prefill and num_new_tokens > token_budget:
        break   # can't fit; stop

    num_new_tokens = min(num_new_tokens, token_budget)

    # (3) Allocate KV blocks
    new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens,
                                                  new_computed_blocks=...)
    if new_blocks is None:
        break   # OOM

    # (4) Promote
    self.running.append(request)
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = num_cached   # starts at cache-hit offset!
    token_budget -= num_new_tokens
```

### Phase 3: Build `SchedulerOutput` + timing trick (lines 868–998)

```python
scheduler_output = SchedulerOutput(
    scheduled_new_reqs    = [NewRequestData(full payload) for new requests],
    scheduled_cached_reqs = CachedRequestData(diffs for previously seen requests),
    finished_req_ids      = self.finished_req_ids,  # freed since last step
    ...
)
_update_after_schedule(scheduler_output)
# ↑ advances num_computed_tokens AFTER building output
# so output describes the pre-step state (worker needs it to slice inputs),
# but scheduler state is already updated for the next call
```

---

## 9. `update_from_output()` — The Feedback Loop (line 1295)

Called right after `execute_model()` returns.

```python
for req_id, _ in num_scheduled_tokens.items():
    generated_token_ids = sampled_token_ids[req_index]  # [] during prefill

    # Speculative decode: roll back rejected draft tokens
    if scheduled_spec_token_ids and generated_token_ids:
        num_rejected = num_draft_tokens - (len(generated_token_ids) - 1)
        request.num_computed_tokens -= num_rejected

    new_token_ids, stopped = _update_request_with_output(request, generated_token_ids)

    if stopped:
        _handle_stopped_request(request)  # re-enqueue if streaming session
        _free_request(request)            # release KV blocks

    outputs[request.client_index].append(EngineCoreOutput(req_id, new_token_ids, ...))
```

**During prefill**: `generated_token_ids = []`. No output emitted. Request stays in `running`.  
**During decode**: `generated_token_ids = [token_id]`. `EngineCoreOutput` sent back to frontend → `OutputProcessor`.

---

## 10. Prefix Caching

```
Prompt: [tok_0 ... tok_1023 | tok_1024 ... tok_4095]
         ←── cached ────────→ ←── needs compute ──→
         num_computed_tokens = 1024   num_new_tokens = 3072
```

On first promotion from `waiting`, `get_computed_blocks()` finds matching cached blocks. `request.num_computed_tokens` is set to the cached count before entering `running`. The request only pays compute cost for the uncached suffix.

Full cache hit → `num_new_tokens = 1` → first step is already a decode step.

---

## 11. Five Key Questions

**Q1: Request too long for one batch?**  
`enable_chunked_prefill=True` (default): `num_new_tokens` clamped to budget. Partial prefill happens; request stays in `running` and continues next step. `enable_chunked_prefill=False`: request stays in `waiting` until the full prompt fits in one budget.

**Q2: GPU memory full?**  
Preemption fires when `allocate_slots()` returns `None`. FCFS: evict last element of `running`. Priority: evict lowest-priority. All KV blocks freed, `num_computed_tokens = 0`, full recompute on reschedule.

**Q3: Why `NewRequestData` AND `CachedRequestData`?**  
Workers cache full request state in a persistent GPU-side `InputBatch`. Full payload crosses the IPC pipe once (`NewRequestData`). All subsequent steps ship only the diff (`CachedRequestData`).

**Q4: Prefill vs. decode distinction in the scheduler?**  
None. The scheduler only sees `num_computed_tokens` vs `num_tokens_with_spec`.

**Q5: How does prefix caching change `num_computed_tokens`?**  
`get_computed_blocks()` returns cached prefix blocks and their token count. `num_computed_tokens` is set to that count before the request enters `running`. Only the uncached suffix is scheduled.

---

## 12. The Bigger Picture

```
Step 01 ✅  entrypoints/llm.py           User API (LLM class)
Step 02 ✅  v1/engine/                    LLMEngine façade + EngineCore + IPC
Step 03 ✅  v1/engine/input_processor.py  Entry gate   ← THIS STEP
            v1/engine/output_processor.py Exit gate    ← THIS STEP
            v1/core/sched/scheduler.py    Batch scheduling ← THIS STEP
Step 04     v1/worker/gpu_worker.py       GPU execution, InputBatch
Step 05     v1/attention/ + KV cache      PagedAttention, block tables
Step 06     v1/sample/                    Token sampling
Step 07     entrypoints/openai/           Production API server
```

---

## 13. The Complete Request Journey

```
User: LLM.generate("Tell me a joke")
│
│ ── frontend process ──────────────────────────────────────────────────────
│
▼ InputProcessor.process_inputs()                              §1
  ├─ tokenize "Tell me a joke" → [15, 7032, 263, 27236]
  ├─ validate length (4 < max_model_len=4096 ✓)
  ├─ sampling_params: fill max_tokens = 4096 - 4 = 4092
  ├─ assign_request_id: "req-42" → "req-42-a3f8b2c1"
  └─ → EngineCoreRequest(request_id="req-42-a3f8b2c1",
                          external_req_id="req-42",
                          prompt_token_ids=[15, 7032, 263, 27236])

▼ OutputProcessor.add_request()                                §2
  └─ creates RequestState with IncrementalDetokenizer
     (detokenizer accumulates token IDs; decodes text incrementally)

│ ── IPC boundary (socket / shared memory) ─────────────────────────────────
│
▼ EngineCore.add_request() → Scheduler.add_request()          §7
  └─ Request born here; pushed to self.waiting

▼ Scheduler.schedule()                                         §8
  ├─ Phase 1 (running): nothing yet
  ├─ Phase 2 (waiting): prefix cache miss → num_computed_tokens=0
  │    allocate 4 KV slots; promote to running
  ├─ NewRequestData(prompt_token_ids=[15,7032,263,27236], blocks=[0,1,2,3])
  └─ → SchedulerOutput

▼ execute_model(SchedulerOutput)
  └─ GPU forward pass on 4 prompt tokens → samples token 7704 ("Why")

▼ Scheduler.update_from_output()                               §9
  └─ EngineCoreOutput(req_id="req-42-a3f8b2c1", new_token_ids=[7704])

│ ── IPC boundary ───────────────────────────────────────────────────────────
│
▼ OutputProcessor.process_outputs()                            §2
  ├─ detokenizer.update([7704]) → text=" Why", no stop string
  ├─ output_kind=DELTA → emit RequestOutput(delta_text=" Why")
  └─ request stays open; repeat for next token

  ... (more decode steps: [263]→" did", [12]→" the", ...)

  ├─ detokenizer.update([17]) → text=" Why did the chicken cross the road?"
  │   stop_string "?" matched!
  │   finish_reason = STOP
  │   reqs_to_abort = ["req-42-a3f8b2c1"]   ← signal scheduler
  └─ RequestOutput(request_id="req-42",      ← external_req_id restored
                   text="Why did the chicken cross the road?",
                   finish_reason="stop",
                   finished=True)

▼ Scheduler.finish_requests(["req-42-a3f8b2c1"], FINISHED_ABORTED)
  └─ KV blocks freed; request removed from running

User sees: RequestOutput with decoded text, finished=True
```
