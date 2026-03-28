# 01-entrypoints learning-notes

> 源碼文件：`vllm/entrypoints/llm.py` (2054 行)
> 核心類：`class LLM` (line 111)

## 1. 總覽

`LLM` 類是 vLLM **離線推理**的用戶入口。用戶只需要和這個類互動，它內部負責：
1. 解析參數 → 建立 `EngineArgs`
2. 用 `EngineArgs` 建立 `LLMEngine`（來自 `vllm/v1/engine/llm_engine.py`）
3. 把用戶的 prompt 送進引擎
4. 執行引擎主循環，收集結果

## 2. 調用鏈全景圖

```
用戶代碼:
    llm = LLM(model="facebook/opt-125m")
    output = llm.generate(["Hello world"], SamplingParams(...))

內部調用鏈:

LLM.__init__(model, **kwargs)                    # line 217
    │
    ├── EngineArgs(model=..., **kwargs)           # line 342 — 收集所有參數
    │
    └── LLMEngine.from_engine_args(engine_args)   # line 382 — 建立引擎
            │
            └── self.llm_engine = LLMEngine(...)  # 這是 v1 新引擎！

LLM.generate(prompts, sampling_params)            # line 440
    │
    └── _run_completion(prompts, params, ...)      # line 1805
            │
            ├── _add_completion_requests(...)       # line 1774
            │       │
            │       └── _render_and_add_requests()  # line 1922
            │               │
            │               └── for each prompt:
            │                       _add_request()  # line 1935
            │                           │
            │                           └── self.llm_engine.add_request(
            │                                   request_id, prompt, params)
            │
            └── _run_engine(output_type, use_tqdm)  # line 1956
                    │
                    └── while has_unfinished_requests():
                            step_outputs = self.llm_engine.step()  ← 核心！
                            收集 finished outputs
                    │
                    └── return sorted(outputs, key=request_id)
```

## 3. 關鍵方法解析

### 3.1 `__init__()` (line 217-385)

**做了什麼：**
- 接收大量配置參數（模型名、dtype、GPU 記憶體、量化方式等）
- 處理一些配置轉換（compilation_config, structured_outputs 等）
- 把所有參數打包成 `EngineArgs` 對象
- 調用 `LLMEngine.from_engine_args()` 建立引擎

**關鍵代碼片段：**
```python
# line 342-383
engine_args = EngineArgs(model=model, ...)
self.llm_engine = LLMEngine.from_engine_args(
    engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
)
```

**設計觀察：**
- `LLM` 類本身不做推理，它是一個薄薄的包裝層
- 真正的工作都委託給 `self.llm_engine`（即 `vllm/v1/engine/llm_engine.py` 的 `LLMEngine`）

### 3.2 `generate()` (line 440-501)

**做了什麼：**
- 校驗 runner_type 必須是 "generate"
- 如果沒傳 sampling_params，用默認值
- 調用 `_run_completion()` 做實際工作

**就這麼簡單！** `generate()` 本身只有 ~10 行有效代碼。

### 3.3 `_run_completion()` (line 1805-1827)

**做了什麼：兩步走**
```python
def _run_completion(self, prompts, params, output_type, ...):
    # Step 1: 把所有請求加入引擎隊列
    self._add_completion_requests(prompts, params, ...)

    # Step 2: 跑引擎主循環直到所有請求完成
    return self._run_engine(use_tqdm=use_tqdm, output_type=output_type)
```

### 3.4 `_add_request()` (line 1935-1955)

**做了什麼：**
- 生成唯一 request_id
- 調用 `self.llm_engine.add_request()` 把請求送進引擎

```python
def _add_request(self, prompt, params, ...):
    request_id = str(next(self.request_counter))
    return self.llm_engine.add_request(request_id, prompt, params, ...)
```

### 3.5 `_run_engine()` (line 1956-2010) ⭐ 最重要

**這是整個離線推理的主循環：**

```python
def _run_engine(self, output_type, use_tqdm=True):
    outputs = []
    while self.llm_engine.has_unfinished_requests():
        step_outputs = self.llm_engine.step()     # ← 每次推進一步
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
    return sorted(outputs, key=lambda x: int(x.request_id))
```

**核心邏輯：**
1. 不斷調用 `self.llm_engine.step()` — 每次執行一輪推理
2. 收集已完成的輸出
3. 按 request_id 排序後返回（因為不同請求可能在不同時間完成）

## 4. 架構洞察

### LLM 類的角色 = "薄包裝層"

```
┌───────────────────────────────────────────────┐
│  LLM (entrypoints/llm.py)                     │
│  ┌────────────────────────────────────────┐   │
│  │  職責：                                 │   │
│  │  • 參數校驗和轉換                        │   │
│  │  • 用戶友好的 API (generate/chat/...)    │   │
│  │  • tqdm 進度條                          │   │
│  │  • 結果排序和返回                        │    │
│  └────────────────────────────────────────┘   │
│                    │                          │
│                    ▼                          │
│  ┌────────────────────────────────────────┐   │
│  │  self.llm_engine (LLMEngine)           │   │
│  │  真正的推理引擎，所有重活都在這裡            │   │
│  │  • add_request()                       │   │
│  │  • step()                              │   │
│  │  • has_unfinished_requests()           │   │
│  └────────────────────────────────────────┘   │
└───────────────────────────────────────────────┘
```

### 兩種使用模式

| 模式 | 方法 | 說明 |
|---|---|---|
| **同步一次性** | `generate()` | 加入請求 + 跑引擎 + 返回結果，一步到位 |
| **異步排隊** | `enqueue()` + `wait_for_completion()` | 先加請求，之後再跑引擎 |

## 5. 下一步

→ 進入 `vllm/v1/engine/llm_engine.py`，理解：
- `LLMEngine.from_engine_args()` 怎麼建立引擎
- `add_request()` 內部做了什麼
- `step()` 一次推進的邏輯是什麼