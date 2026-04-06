# Step 03 — vLLM Scheduler (`vllm/v1/core/sched/`)

> **Roadmap position**: CPU side, batch-decision layer.
> Sits between `EngineCore` (the orchestrator, Step 02) and the GPU worker (Step 04).
>
> ```
> LLMEngine  →  EngineCoreClient  →  EngineCore.step()
>                                         │
>                                    scheduler.schedule()   ← YOU ARE HERE
>                                         │
>                                    execute_model()
>                                         │
>                                    scheduler.update_from_output()
> ```

---

## 1. The One Invariant That Explains Everything

The docstring at the top of `schedule()` (line 342) is the key to the whole file:

```
There is no "decoding phase" nor "prefill phase" in the scheduler.
Each request just has num_computed_tokens and num_tokens_with_spec.
At each step the scheduler tries to assign tokens so that each request's
num_computed_tokens can catch up its num_tokens_with_spec.
```

Every token the model processes is described by two numbers on the request object:

| Field | Meaning |
|---|---|
| `num_computed_tokens` | How many tokens have already been processed (KV computed) |
| `num_tokens_with_spec` | How many tokens *should* be processed (prompt + output so far + any speculative draft tokens) |

The scheduler's only job each step: pick a set of requests and decide how many new tokens each one gets (`num_new_tokens`), bounded by a global token budget.

```
num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens
```

Then clamp: `num_new_tokens = min(num_new_tokens, token_budget)`.

This one formula covers:
- **First prefill step**: `num_computed_tokens = 0`, so `num_new_tokens = len(prompt)`
- **Chunked prefill**: `num_new_tokens` gets clamped to `token_budget`; next step it continues from where it left off
- **Decode step**: `num_computed_tokens = len(prompt) + k` output tokens already done; `num_tokens_with_spec = len(prompt) + k + 1`; so `num_new_tokens = 1`
- **Speculative decode**: spec draft tokens inflate `num_tokens_with_spec`; the scheduler allocates KV for them too

---

## 2. Package Layout

```
vllm/v1/core/sched/
├── scheduler.py        2305 L  ← the implementation (Scheduler class)
├── output.py            261 L  ← SchedulerOutput, NewRequestData, CachedRequestData
├── request_queue.py     208 L  ← FCFSRequestQueue, PriorityRequestQueue
├── interface.py         243 L  ← SchedulerInterface (abstract base)
└── async_scheduler.py    60 L  ← thin async wrapper (skip for now)

vllm/v1/core/
├── kv_cache_manager.py          ← block allocator (called by scheduler)
└── kv_cache_utils.py            ← block hashing, prefix matching
```

---

## 3. The Three Queues

At any moment, a request lives in exactly one of:

```
self.waiting          RequestQueue   normal queue; status WAITING or PREEMPTED
self.skipped_waiting  RequestQueue   blocked on async deps (remote KV, grammar init…)
self.running          list[Request]  currently in-flight; ordered by arrival time
```

`_enqueue_waiting_request()` routes to `skipped_waiting` for blocked statuses
(`WAITING_FOR_REMOTE_KVS`, `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`, etc.),
otherwise to `waiting`.

The global constraints that bound how full `running` can get:
- `max_num_running_reqs` — max concurrent sequences (e.g. 256)
- `max_num_scheduled_tokens` — max tokens per step across all requests (e.g. 8192)

---

## 4. Data Structures (`output.py`)

### `SchedulerOutput` — the message sent to the GPU worker each step

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs:    list[NewRequestData]   # first appearance: full payload
    scheduled_cached_reqs: CachedRequestData       # repeat appearances: diff only

    num_scheduled_tokens:        dict[str, int]   # {req_id: num_tokens}
    total_num_scheduled_tokens:  int
    scheduled_spec_decode_tokens: dict[str, list[int]]
    scheduled_encoder_inputs:    dict[str, list[int]]
    num_common_prefix_blocks:    list[int]         # for cascade attention

    finished_req_ids:     set[str]   # tell workers to drop cached state for these
    free_encoder_mm_hashes: list[str]
    preempted_req_ids:    set[str] | None   # v2 model runner only
    kv_connector_metadata: ...
    new_block_ids_to_zero: list[int] | None   # fresh blocks that need GPU memory zeroing
```

### `NewRequestData` vs `CachedRequestData` — the bandwidth optimization

Workers maintain a persistent `InputBatch` on the GPU side. The first time a request appears in a batch, `NewRequestData` ships the **full payload**:

```
req_id, prompt_token_ids, mm_features, sampling_params,
pooling_params, block_ids (full list), num_computed_tokens, lora_request
```

Every subsequent step, the worker already has all of that cached. Only the **delta** is sent via `CachedRequestData`:

```
req_ids[]                — which requests
new_block_ids[]          — only newly allocated blocks (not the full list)
num_computed_tokens[]    — so worker knows where to slice the token buffer
num_output_tokens[]      — for tracking
resumed_req_ids (set)    — preempted-and-resumed: replace block list, don't append
new_token_ids[]          — only used in pipeline-parallel mode
```

**Why this matters**: a 4096-token prompt's token ID list (16 KB of ints) is serialized across a process boundary exactly once. Decode steps that follow only ship a handful of scalars per request.

### `FCFSRequestQueue` vs `PriorityRequestQueue`

| | FCFS | Priority |
|---|---|---|
| Backing store | `deque` | `list` (min-heap via `heapq`) |
| Order key | arrival time | `(priority, arrival_time)` |
| `add_request` | `appendright` — O(1) | `heappush` — O(log n) |
| `pop_request` | `popleft` — O(1) | `heappop` — O(log n) |
| `prepend_request` | `appendleft` — to front | `heappush` — priority still rules |
| iteration | left-to-right | successive `heappop` on a copy |

The `prepend_request` asymmetry matters for preemption: in FCFS a preempted request goes back to the **front** of `waiting`, preserving relative FCFS order. In Priority mode there is no front; the request just re-heaps normally.

---

## 5. Request Lifecycle

```
add_request()                   finish_requests() / update_from_output()
     │                                    │
     ▼                                    ▼
  WAITING ──── schedule() ──────► RUNNING ──── update_from_output() ──► FINISHED_*
     ▲              │                  │
     │         (OOM / preempt)         │
     └─── PREEMPTED ◄──────────────────┘
               (num_computed_tokens = 0)
```

### `add_request()` (line 1726)

- New request: `_enqueue_waiting_request(request)` + register in `self.requests` dict.
- Streaming-input update (same ID): queue the next chunk or start it immediately.
- No KV allocation here. Prefix-cache lookup happens in `schedule()` at promotion time.

### `finish_requests()` (line 1748)

Two-pass:
1. Batch-collect running vs. waiting targets to avoid repeated O(n) removals.
2. `request.status = finished_status` → `_free_request()`.

`_free_request()`:
- Adds `req_id` to `self.finished_req_ids` — this set travels in the *next* `SchedulerOutput.finished_req_ids` so workers drop their cached state.
- Calls `_free_blocks()` → `kv_cache_manager.free(request)` — KV blocks return to pool immediately.

`finished_req_ids` is **not cleared** in `_free_request()`. It is cleared in `_update_after_schedule()` *after* it has been packaged into the `SchedulerOutput`. This is the delayed-flush design.

### `_preempt_request()` (line 949)

```python
kv_cache_manager.free(request)        # return all KV blocks immediately to pool
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0       # ← full reset; must recompute from scratch
request.num_preemptions += 1
self.waiting.prepend_request(request) # back to front of queue
```

vLLM v1 does **not** support swap-to-CPU. Preemption means the request loses all its KV state and must re-prefill completely when rescheduled.

---

## 6. `schedule()` — The Algorithm (lines 341–942)

### Phase 1: RUNNING requests (lines 378–545)

```python
for request in self.running:
    num_new_tokens = (
        request.num_tokens_with_spec
        + request.num_output_placeholders
        - request.num_computed_tokens
    )
    num_new_tokens = min(num_new_tokens, token_budget)

    # Inner loop: allocate KV blocks, preempting others if OOM
    while True:
        new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
        if new_blocks is not None:
            break   # success
        # OOM: preempt
        if FCFS:
            preempted = self.running.pop()          # last-in (most recently added)
        else:
            preempted = max(self.running, ...)      # lowest-priority
        _preempt_request(preempted, ...)
        if preempted == request:
            break   # preempted ourselves; can't schedule
```

FCFS preemption removes the **last** element of `self.running`. Since `running` is ordered by arrival time, the last element is the most recently promoted request — "last in, first out."

### Phase 2: WAITING requests (lines 556–843)

Only entered if no preemptions occurred and scheduler is not paused.

```python
while (self.waiting or self.skipped_waiting) and token_budget > 0:
    if len(self.running) == max_num_running_reqs:
        break

    request = next from waiting / skipped_waiting queue

    # (1) Prefix cache lookup — only on first scheduling attempt
    if request.num_computed_tokens == 0:
        new_computed_blocks, num_local_cached = kv_cache_manager.get_computed_blocks(request)
        # num_local_cached = how many prefix tokens are already in cache
    
    # (2) Tokens to schedule
    num_new_tokens = request.num_tokens - num_computed_tokens
    if not enable_chunked_prefill and num_new_tokens > token_budget:
        break       # can't schedule this request without chunking; stop here

    num_new_tokens = min(num_new_tokens, token_budget)  # chunked prefill: partial OK

    # (3) Allocate KV blocks
    new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens,
                                                  new_computed_blocks=new_computed_blocks, ...)
    if new_blocks is None:
        break   # OOM from the waiting side; stop entirely

    # (4) Promote to running
    self.running.append(request)
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = num_computed_tokens   # starts at cache-hit offset
    num_scheduled_tokens[request.request_id] = num_new_tokens
    token_budget -= num_new_tokens
```

### Phase 3: Build `SchedulerOutput` (lines 868–942)

```python
new_reqs_data   = [NewRequestData.from_request(req, block_ids) for req in scheduled_new_reqs]
cached_reqs_data = _make_cached_request_data(running_reqs, resumed_reqs, ...)

scheduler_output = SchedulerOutput(
    scheduled_new_reqs=new_reqs_data,
    scheduled_cached_reqs=cached_reqs_data,
    num_scheduled_tokens=num_scheduled_tokens,
    finished_req_ids=self.finished_req_ids,   # from _free_request() since last step
    ...
)
_update_after_schedule(scheduler_output)   # advance num_computed_tokens
return scheduler_output
```

### `_update_after_schedule()` — the timing trick (line 971)

`num_computed_tokens` is advanced **after** building `SchedulerOutput`, not before.

Why:
- The output must describe the *pre-step* state so the worker can reconstruct `[num_computed_tokens : num_computed_tokens + num_scheduled_tokens]` as the input slice.
- But we want the scheduler's internal state ready for the *next* scheduling call immediately.

So the scheduler builds the output with the old `num_computed_tokens`, then increments it:
```python
request.num_computed_tokens += num_scheduled_token
```

---

## 7. `update_from_output()` — The Feedback Loop (line 1295)

Called right after `execute_model()` returns. Core loop:

```python
for req_id, _ in scheduler_output.num_scheduled_tokens.items():
    generated_token_ids = sampled_token_ids[req_index]  # [] during prefill

    # Speculative decode: undo rejected draft tokens
    if scheduled_spec_token_ids and generated_token_ids:
        num_rejected = num_draft_tokens - (len(generated_token_ids) - 1)
        request.num_computed_tokens -= num_rejected

    # Append tokens, check stop conditions
    new_token_ids, stopped = _update_request_with_output(request, generated_token_ids)

    if stopped:
        _handle_stopped_request(request)  # re-enqueue if streaming session
        _free_request(request)            # free KV blocks

    # Emit EngineCoreOutput for this request
    outputs[request.client_index].append(EngineCoreOutput(req_id, new_token_ids, ...))
```

**During prefill**: `generated_token_ids = []` (model doesn't sample mid-prefill). No `EngineCoreOutput` is emitted. The request stays in `running` and gets more tokens next step.

**During decode**: `generated_token_ids = [token_id]` (one new token). `EngineCoreOutput` is emitted and streamed back toward the client.

---

## 8. Prefix Caching — How It Changes `num_computed_tokens`

When a waiting request is promoted, `get_computed_blocks()` returns any already-cached prefix:

```
Prompt: [tok_0 tok_1 ... tok_1023 | tok_1024 ... tok_4095]
                                   ↑
         ←── already cached ──────→ ←── needs compute ──→
         num_computed_tokens = 1024   num_new_tokens = 3072
```

The request enters `running` with `num_computed_tokens = 1024`. It only needs to process 3072 tokens, not 4096. For a **full hit** (entire prompt cached): `num_new_tokens = 1`, and the very first step is a decode step.

`request.num_cached_tokens` is set once to record the hit count for metrics.

---

## 9. Five Key Questions — Quick Answers

**Q1: Request too long to fit in one batch?**
When `enable_chunked_prefill=True` (the default), `num_new_tokens` is clamped to `token_budget`. The request is promoted to `running` with a partial count. Next step, Phase 1 continues it from `num_computed_tokens`. Repeats until `num_computed_tokens == num_tokens_with_spec`.
When `enable_chunked_prefill=False`, the request waits in `waiting` until the batch has enough budget for the full prompt.

**Q2: GPU memory full?**
Preemption fires in Phase 1 when `allocate_slots()` returns `None`.
- FCFS: pop the **last** element of `self.running` (most recently admitted request).
- Priority: find the request with the highest `(priority, arrival_time)` (lowest scheduling priority).
All KV blocks freed immediately. `num_computed_tokens = 0`. Full recompute when rescheduled.

**Q3: Why `NewRequestData` AND `CachedRequestData`?**
Workers cache full request state in a persistent GPU-side `InputBatch`. `NewRequestData` ships the full payload exactly once. `CachedRequestData` ships only the diff (new block IDs, `num_computed_tokens`, output token count) on every subsequent step. Saves serializing multi-KB prompt token lists every decode step.

**Q4: Prefill vs. decode distinction in the scheduler?**
**None.** The scheduler only sees `num_computed_tokens` vs `num_tokens_with_spec`. The field `request.is_prefill_chunk` exists for structured-output logic, but the scheduling algorithm itself does not branch on it.

**Q5: How does prefix caching change `num_computed_tokens`?**
On first promotion from `waiting`, `get_computed_blocks()` returns cached blocks covering the matched prefix. `request.num_computed_tokens` is set to the cached count before the request enters `running`. The request only pays compute cost for the uncached suffix.

---

## 10. The Bigger Picture

```
Step 01 ✅  entrypoints/llm.py          User API (LLM class)
Step 02 ✅  v1/engine/                   LLMEngine façade + EngineCore + IPC
Step 03 ✅  v1/core/sched/scheduler.py   Batch scheduling ← THIS STEP
Step 04     v1/worker/gpu_worker.py      GPU execution, InputBatch
Step 05     v1/attention/ + KV cache     PagedAttention, block tables
Step 06     v1/sample/                   Token sampling
Step 07     entrypoints/openai/          Production API server
```

After Step 03 you understand the complete **CPU-side decision-making** loop. Steps 04–05 are the GPU side: how `SchedulerOutput` is consumed, how `InputBatch` maintains persistent state, and how block IDs map to physical GPU memory in PagedAttention.
