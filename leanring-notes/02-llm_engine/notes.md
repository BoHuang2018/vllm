# 02-llm_engine 學習筆記

> 源碼文件：`vllm/v1/engine/llm_engine.py`
> 核心類：`class LLMEngine`

## 1. 總覽

`LLMEngine` 是 vLLM v1 的**核心協調器**。它本身不做推理，而是把工作分配給三個內部組件：
| 組件 | 職責 | 資料流方向 |
|--------------------|--------------------------------------------------------------------|-----------|
| `InputProcessor`   | 把用戶的 prompt → `EngineCoreRequest` (tokenize、validation)        | 輸入 → 引擎 |
| `EngineCoreClient` | 排程、推理、GPU 計算 | 引擎核心 |
| `OutputProcessor`  | 把 `EngineCoreOutputs` → `RequestOutput` (detokenize、stop-string) | 引擎 → 輸出 |

**設計觀察：** 跟 `01-entrypoints` 中的 `LLM` 類一樣，`LLMEngine` 也是一個**協調層**，
真正的推理工作在 `EngineCoreClient` 裡面。

```
用戶代碼                     LLM (薄包裝)              LLMEngine (協調器)
─────────                    ──────────              ──────────────────
llm.generate(prompts) ──→ _add_request()      ──→    add_request()
                          _run_engine()       ──→    step() (while loop)
```

## 2. 調用鏈全景圖

### 2.1 The Big Picture (with real class names)
```
User Code
    │
    ▼
LLM  (entrypoints/llm.py)
    │  thin wrapper, provides generate() API
    ▼
LLMEngine  (v1/engine/llm_engine.py)
    ├── InputProcessor          prompt str → EngineCoreRequest
    ├── OutputProcessor         new_token_ids → RequestOutput (with text)
    └── EngineCoreClient        IPC bridge (location-transparent abstraction)
              │
    ┌─────────┴─────────────────────────────────────────────┐
    │       ZMQ (multiprocess) / direct call (in-process)   │
    └─────────┬─────────────────────────────────────────────┘
              ▼
EngineCore  (v1/engine/core.py)                    ← GPU side
    ├── Scheduler  (v1/core/sched/scheduler.py)    manages request queue + KV blocks
    ├── model_executor  (v1/executor/)             manages GPU Workers
    │       └── Worker per GPU  (v1/worker/)       runs actual forward pass
    └── StructuredOutputManager                    JSON schema / regex constraints
```


### 2.2 From LLM to LLMEngine

```
用戶調用 llm.generate(prompts, params)
    │
    ▼
LLM._add_request(prompt, params)
    │
    ▼
LLMEngine.add_request(request_id, prompt, params)          [llm_engine.py:216]
    │
    ├── InputProcessor.process_inputs()                    [input_processor.py:195]
    │       ├── 驗證 params (SamplingParams / PoolingParams)
    │       ├── 驗證 LoRA 請求
    │       ├── 執行 tokenization → prompt_token_ids
    │       ├── 處理多模態輸入 (MultiModalFeatureSpec)
    │       └── 返回 EngineCoreRequest
    │
    ├── OutputProcessor.add_request()                      [output_processor.py:508]
    │       └── 建立 RequestState (持有 detokenizer, logprobs processor)
    │
    └── EngineCoreClient.add_request(EngineCoreRequest)    [core_client.py]
            └── → 傳給 EngineCore（in-process 直接調用 / 跨進程用 ZMQ）
 
LLM._run_engine() while loop:
    │
    ▼
LLMEngine.step()                                           [llm_engine.py:294]
    │
    ├── Phase 1: engine_core.get_output()                  → 從 EngineCore 拿回 token IDs
    ├── Phase 2: output_processor.process_outputs()        → token IDs → 文字
    ├── Phase 3: engine_core.abort_requests(reqs_to_abort) → 清理停止詞觸發的請求
    └── Phase 4: logger_manager.record() (if log_stats)    → 記錄統計
```

### 2.3 In LLMEngine

```
用戶代碼 (直接用 LLMEngine 的情境):
    engine_args = EngineArgs(model="facebook/opt-125m")
    engine = LLMEngine.from_engine_args(engine_args)
    engine.add_request("req-0", "Hello world", SamplingParams(...))
    while engine.has_unfinished_requests():
        outputs = engine.step()

內部調用鏈:

LLMEngine.from_engine_args(engine_args)                    # line ~159
    │
    ├── vllm_config = engine_args.create_engine_config()   # EngineArgs → VllmConfig
    ├── executor_class = Executor.get_class(vllm_config)   # 選擇執行後端
    │
    └── cls(vllm_config, executor_class, ...)              # 調用 __init__
            │
            ├── self.renderer = renderer_from_config(...)
            │       → chat template 渲染 + multimodal 處理
            │
            ├── self.io_processor = get_io_processor(...)
            │       → IO 處理器插件 (可自定義)
            │
            ├── self.input_processor = InputProcessor(vllm_config, renderer)
            │       → 負責: EngineInput → EngineCoreRequest
            │
            ├── self.output_processor = OutputProcessor(tokenizer, ...)
            │       → 負責: EngineCoreOutputs → RequestOutput
            │
            ├── self.engine_core = EngineCoreClient.make_client(
            │       multiprocess_mode=..., asyncio_mode=False, ...)
            │       → 負責: 排程 + 推理（可跨進程）
            │
            └── (optional) self.logger_manager = StatLoggerManager(...)

LLMEngine.add_request(request_id, prompt, params, ...)    # line ~216
    │
    ├── 型別校驗: request_id 必須是 str
    │
    ├── if isinstance(prompt, EngineCoreRequest):
    │       → 舊相容路徑 (deprecated, 將在 v0.18 移除)
    │       → 直接使用，但會檢查 request_id 是否匹配
    │
    └── else:  ← 正常路徑
            │
            ├── request = self.input_processor.process_inputs(
            │       request_id, prompt, params,
            │       supported_tasks=self.get_supported_tasks(),
            │       arrival_time, lora_request, ...)
            │       → tokenization、validation、LoRA 處理
            │
            ├── prompt_text, _, _ = extract_prompt_components(...)
            │       → 提取原始 prompt 文字 (用於後續輸出顯示)
            │
            ├── self.input_processor.assign_request_id(request)
            │       → 確保 request 有唯一 ID
            │
            ├── params = request.params
            │       → 使用 process_inputs() 中可能修改過的參數副本
            │
            ├── n = params.n if SamplingParams else 1
            │
            ├── if n == 1:                                     ← 常見路徑
            │       ├── self.output_processor.add_request(request, prompt_text, None, 0)
            │       │       → 在 OutputProcessor 中註冊 RequestState
            │       ├── self.engine_core.add_request(request)
            │       │       → 送進 EngineCore 排隊等推理
            │       └── return req_id
            │
            └── if n > 1:                                      ← parallel sampling
                    ├── parent_req = ParentRequest(request)
                    └── for idx in range(n):
                            ├── request_id, child_params = parent_req.get_child_info(idx)
                            ├── child_request = copy(request) if idx < n-1 else request
                            │       → 最後一個 child 直接複用原始 request (避免多餘 copy)
                            ├── child_request.request_id = request_id
                            ├── child_request.sampling_params = child_params
                            ├── self.output_processor.add_request(
                            │       child_request, prompt_text, parent_req, idx)
                            └── self.engine_core.add_request(child_request)

LLMEngine.step()                                           # line ~294 ⭐ 核心
    │
    ├── if self.should_execute_dummy_batch:                 # Data Parallel 同步用
    │       → self.engine_core.execute_dummy_batch()
    │       → return []    ← 不產生實際輸出
    │
    ├── 1️⃣ outputs = self.engine_core.get_output()
    │       → 從 EngineCore 取得本輪推理的原始輸出 (EngineCoreOutputs)
    │
    ├── 2️⃣ processed_outputs = self.output_processor.process_outputs(
    │       outputs.outputs,
    │       engine_core_timestamp=outputs.timestamp,
    │       iteration_stats=iteration_stats)
    │       → detokenization、stop-string 檢測、streaming 處理
    │       → 同時更新 scheduler stats
    │
    ├── 3️⃣ self.engine_core.abort_requests(processed_outputs.reqs_to_abort)
    │       → 中止因 stop string 而結束的請求
    │       → 注意：stop string 的檢測在 OutputProcessor 而非 EngineCore
    │
    ├── 4️⃣ if log_stats and outputs 非空:
    │        ├── self.logger_manager.record(scheduler_stats, iteration_stats, ...)
    │        ├── self.do_log_stats_with_interval()
    └── return processed_outputs.request_outputs   # → List[RequestOutput]

LLMEngine.has_unfinished_requests()                        # line ~170
    │
    ├── has_unfinished = self.output_processor.has_unfinished_requests()
    │       → 問 OutputProcessor 還有沒有未完成的 RequestState
    │
    ├── if self.dp_group is None:    ← 非 Data Parallel (常見)
    │       → return has_unfinished or self.engine_core.dp_engines_running()
    │
    └── else:                        ← Data Parallel 模式
            → return self.has_unfinished_requests_dp(has_unfinished)
              → 跨 DP group 聚合，只要任一 rank 有未完成就繼續
```

## 3. 關鍵方法解析

### 3.1 `__init__()` — 建立三大組件

**最重要的 4 行：**

```python
# 0) Set up renderer
self.renderer = renderer_from_config(self.vllm_config)
# 1) 把字符串/Token prompt → EngineCoreRequest
self.input_processor = InputProcessor(self.vllm_config, renderer)
# 2) 把 EngineCoreOutput (token IDs) → RequestOutput (文字)
self.output_processor = OutputProcessor(renderer.tokenizer, ...)
# 3) IPC 橋：聯繫 EngineCore（本進程或後台進程）
self.engine_core = EngineCoreClient.make_client(multiprocess_mode=..., asyncio_mode=False, ...)
```

**設計觀察：**

- `EngineCoreClient.make_client()` 根據 `multiprocess_mode` 決定是
  **同進程** (直接調用) 還是**多進程** (IPC 通訊)
- `asyncio_mode=False` → 這是給同步 `LLM` 類用的；
  異步的 `AsyncLLMEngine` 會傳 `True`

### 3.2 `from_engine_args()` — 工廠方法

```python
vllm_config = engine_args.create_engine_config(usage_context)  # EngineArgs → VllmConfig
executor_class = Executor.get_class(vllm_config)  # 選後端
return cls(vllm_config, executor_class, ...)  # 調 __init__
```

**設計觀察：**

- 另外還有 `from_vllm_config()` 工廠方法，跳過 EngineArgs 直接用 VllmConfig
- 兩層工廠 = 兩種入口：`EngineArgs`（CLI/用戶友好）vs `VllmConfig`（程式化）

### 3.3 `add_request()` — 請求進入引擎

**核心流程：preprocess → register → enqueue**

```python
# 1. preprocess
request = self.input_processor.process_inputs(request_id, prompt, params, ...)

# 2. register (在 OutputProcessor 登記，準備接收輸出)
self.output_processor.add_request(request, prompt_text, None, 0)

# 3. enqueue (送進 EngineCore 排隊)
self.engine_core.add_request(request)
```

**n > 1 (parallel sampling) 的設計：**

- 一個 parent request 展開成 n 個 child requests
- 每個 child 有獨立的 `request_id` 和 `sampling_params`
- 最後一個 child 複用原始 request 對象 (節省一次 copy)

### 3.4 `step()` ⭐ — 引擎主循環的一步

**四步走：get → process → abort → log**

| 步驟                  | 代碼                                           | 說明                                    |
|---------------------|----------------------------------------------|---------------------------------------|
| 1️⃣ get_output      | `self.engine_core.get_output()`              | 取回 EngineCore 的計算結果（包含 raw token IDs） |
| 2️⃣ process_outputs | `self.output_processor.process_outputs(...)` | detokenize token IDs → 文字，檢查停止詞       |
| 3️⃣ abort_requests  | `self.engine_core.abort_requests(...)`       | 把被停止詞終止的請求通知 EngineCore 取消            |
| 4️⃣ log_stats       | `self.logger_manager.record(...)`            | 統計日誌                                  |

**重點：Phase 1 的 `get_output()` 在多進程模式下是阻塞的** —— 它等待 EngineCore 完成一輪推理並把結果通過 ZMQ 傳回來。

**關鍵洞察：stop string 的處理**

- EngineCore 不知道 stop string → 它只管生成 token
- OutputProcessor 在 detokenize 時檢測到 stop string
- 然後通過 `step()` 的第 3 步回報給 EngineCore 中止該請求
- 這是一���跨組件的協作流程

**關鍵一句話：** `LLMEngine.step()` **不調用 GPU**。它只是收集 EngineCore 已經算好的結果。

### 3.5 `has_unfinished_requests()` — 循環控制

```python
def has_unfinished_requests(self) -> bool:
    has_unfinished = self.output_processor.has_unfinished_requests()
    ...
```

**注意：** 由 `OutputProcessor` 而非 `EngineCore` 判斷。
因為 OutputProcessor 追蹤所有 `RequestState`，它知道哪些請求還在進行中。

### 4. EngineCoreClient — IPC 橋
 
`make_client()` 根據模式返回不同實現：
 
| 模式 | 類 | 使用場景 |
|---|---|---|
| `multiprocess=False` | `InprocClient` | 默認 / 調試，同進程直接調用 |
| `multiprocess=True, asyncio=False` | `SyncMPClient` | `LLM` 類（sync 模式） |
| `multiprocess=True, asyncio=True` | `AsyncMPClient` | `AsyncLLM` / OpenAI API server |
 
### InprocClient（最簡單，適合理解）
 
```python
class InprocClient(EngineCoreClient):
    def __init__(self, *args, **kwargs):
        self.engine_core = EngineCore(*args, **kwargs)   # 直接在本進程建立
 
    def get_output(self) -> EngineCoreOutputs:
        outputs, model_executed = self.engine_core.step_fn()   # 直接調用！
        self.engine_core.post_step(model_executed)
        return outputs.get(0) or EngineCoreOutputs()
 
    def add_request(self, request):
        req, wave = self.engine_core.preprocess_add_request(request)
        self.engine_core.add_request(req, wave)
```
 
**關鍵發現：** 在 `InprocClient` 模式下，`LLMEngine.step()` → `engine_core.get_output()` → `InprocClient.get_output()` → `EngineCore.step_fn()` —— **GPU 推理就發生在這裡！**
 
### SyncMPClient（多進程模式）
 
- EngineCore 運行在獨立的 **後台進程**（`EngineCoreProc.run_engine_core()`）
- 通訊方式：**ZMQ 套接字** + **msgpack 序列化**
- `add_request()` → 序列化 → ZMQ PUSH → 後台進程接收
- `get_output()` → ZMQ PULL → 反序列化 → `EngineCoreOutputs`
 
---

## 4. 架構洞察

### 三大組件的資料流

```
prompt (str/tokens)
    │
    ▼
┌──────────────────┐
│  InputProcessor  │  process_inputs()
│  "翻譯官"         │  prompt → EngineCoreRequest
└────────┬─────────┘
         │ EngineCoreRequest
         ▼
┌──────────────────┐
│  EngineCoreClient│  add_request() / get_output()
│  "引擎核心"        │  scheduling + inference + GPU
└────────┬─────────┘
         │ EngineCoreOutputs
         ▼
┌──────────────────┐
│  OutputProcessor │  process_outputs()
│  "翻譯官 (反向)"   │  raw tokens → RequestOutput (text)
└──────────────────┘
         │
         ▼
    RequestOutput (用戶拿到的結果)
```

### `LLMEngine` 自己做了什麼？

幾乎什麼都不做！它只是：

1. **組裝** — 在 `__init__` 中建立三個組件
2. **協調** — 在 `add_request()` 和 `step()` 中按順序調用三個組件
3. **膠水邏輯** — stop string 回報、Data Parallel 同步、日誌統計

這與 `01-entrypoints` 的 `LLM` 類完全一致：**每一層都是薄包裝，真正的工作往下傳遞。**

```
LLM              → 薄包裝 → LLMEngine
LLMEngine        → 薄協調 → InputProcessor + EngineCoreClient + OutputProcessor
EngineCoreClient → ???    → (下一步要學的)
```

## 5. 次要功能

| 方法                                   | 用途                   |
|--------------------------------------|----------------------|
| `sleep()` / `wake_up()`              | 暫停/喚醒引擎 (節省 GPU 資源)  |
| `add_lora()` / `remove_lora()`       | 動態載入/卸載 LoRA adapter |
| `reset_prefix_cache()`               | 清除 KV cache 前綴       |
| `reset_mm_cache()`                   | 清除 multimodal cache  |
| `collective_rpc()` / `apply_model()` | 分佈式 worker 通訊        |
| `start_profile()` / `stop_profile()` | 效能分析                 |

這些都直接委託給 `self.engine_core`，LLMEngine 只做轉發。

## 6. 實驗建議

在 `vllm/v1/engine/llm_engine.py` 中暫時加入 print，用你的 demo 腳本跑一次：

```python
# 在 add_request() 開頭加:
print(f"🔵 [LLMEngine.add_request] request_id={request_id}, prompt_type={type(prompt)}")

# 在 step() 的每個階段加:
print(f"🟢 [LLMEngine.step] 1. engine_core.get_output() → {len(outputs.outputs)} outputs")
print(f"🟢 [LLMEngine.step] 2. process_outputs → {len(processed_outputs.request_outputs)} results")
print(f"🟢 [LLMEngine.step] 3. abort_requests → {processed_outputs.reqs_to_abort}")
```

**觀察重點：**

- `step()` 被調用幾次？每次返回幾個 output？
- 哪次 `step()` 才開始有 finished output？
- `reqs_to_abort` 什麼時候非空？

## 7. 下一步

根據深入程度，有三條路可選：

| 方向                  | 文件                                                 | 學什麼                                              |
|---------------------|----------------------------------------------------|--------------------------------------------------|
| **InputProcessor**  | `vllm/v1/engine/input_processor.py`                | `process_inputs()` 如何 tokenize、處理 multimodal     |
| **OutputProcessor** | `vllm/v1/engine/output_processor.py`               | `process_outputs()` 如何 detokenize、檢測 stop string |
| **EngineCore** ⭐    | `vllm/v1/engine/core_client.py` → `engine_core.py` | 真正的排程與推理邏輯                                       |

建議順序：`InputProcessor` → `OutputProcessor` → `EngineCore`
（先理解進出兩端，再攻核心）