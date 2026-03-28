# 01 - Entrypoints 待解答問題

## ✅ 已解答
- [x] `LLM` 類的核心職責是什麼？
  → 薄包裝層，參數校驗 + API 包裝，真正的工作委託給 `LLMEngine`

- [x] `generate()` 的調用鏈是什麼？
  → `generate()` → `_run_completion()` → `_add_completion_requests()` + `_run_engine()`

- [x] 為什麼 `_run_completion()` 不接收 `_add_completion_requests()` 的返回值？
  → **有意設計。** `_add_completion_requests()` 返回的 `request_ids` 只在 `enqueue()` 模式下才需要（返回給用戶追蹤）。
  在 `_run_completion()` 中，緊接著就調用 `_run_engine()`，引擎內部通過 `has_unfinished_requests()` 自行追蹤所有請求，不需要外部傳入 request_ids。

- [x] 為什麼要把 `enqueue()` 和 `wait_for_completion()` 分開？
  → 提供兩種使用模式：
  - **同步模式**: `generate()` = 加請求 + 跑引擎，一步完成
  - **異步模式**: `enqueue()` 返回 request_ids → 稍後 `wait_for_completion()` 取結果

## 🔍 待深入（進入 02-engine 後解答）
- [ ] `LLMEngine.from_engine_args()` 內部建立了哪些組件？
- [ ] `LLMEngine.add_request()` 怎麼處理 prompt？tokenize 在哪一步發生？
- [ ] `LLMEngine.step()` 一次 step 具體做了什麼？是生成一個 token 還是多個？
- [ ] `has_unfinished_requests()` 怎麼判斷一個請求是否完成？

## 💡 設計問題（持續思考）
- [ ] `_run_engine` 為什麼要按 request_id 排序？什麼情況下順序會亂？
- [ ] `generate()` vs `chat()` 的差異在哪？
  → 提示：chat 走的是 `_run_chat()` → `_render_and_run_requests()`，多了模板渲染