"""
Necessary environment variables to run this file successfully:
TORCHDYNAMO_DISABLE=1;
VLLM_HOST_IP=127.0.0.1;

02-llm_engine Demo: 直接使用 LLMEngine，觀察三大組件的協作

目標：
  - 跳過 LLM 包裝層，直接操作 LLMEngine
  - 觀察 add_request() 和 step() 的行為
  - 理解 InputProcessor / EngineCoreClient / OutputProcessor 的協作

與 01-entrypoints demo 的區別：
  01 → 用 LLM.generate()，一步到位，看不到內部細節
  02 → 用 LLMEngine 直接操作，手動控制 add_request + step 循環
"""

from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine


def main():
    # ============================================
    # Step 1: 建立 LLMEngine（跳過 LLM 包���層）
    # ============================================
    # 在 01 中我們用的是：
    #   llm = LLM(model=...)  → 內部調 LLMEngine.from_engine_args()
    #
    # 現在我們直接做 LLM 內部做的事：
    #   EngineArgs → LLMEngine.from_engine_args()
    #
    # 📌 觀察: from_engine_args 內部做了三件事：
    #   1. engine_args.create_engine_config() → VllmConfig
    #   2. Executor.get_class(vllm_config) → executor_class
    #   3. LLMEngine(vllm_config, executor_class, ...) → 建立三大組件

    print("=" * 60)
    print("Step 1: 建立 LLMEngine")
    print("=" * 60)

    engine_args = EngineArgs(
        model="facebook/opt-125m",
        dtype="float32",
        enforce_eager=True,
    )

    engine = LLMEngine.from_engine_args(engine_args)

    # 🔍 觀察三大組件是否已建立
    print(f"\n✅ LLMEngine 建立完成!")
    print(f"   InputProcessor:  {type(engine.input_processor).__name__}")
    print(f"   OutputProcessor: {type(engine.output_processor).__name__}")
    print(f"   EngineCoreClient: {type(engine.engine_core).__name__}")
    print(f"   Tokenizer:       {type(engine.get_tokenizer()).__name__}")

    # ============================================
    # Step 2: 手動 add_request()
    # ============================================
    # 在 01 中，LLM.generate() 內部會調用:
    #   self.llm_engine.add_request(request_id, prompt, params)
    #
    # 現在我們自己做這件事
    #
    # 📌 觀察 add_request 內部的三步:
    #   1. input_processor.process_inputs() → tokenize
    #   2. output_processor.add_request()   → 註冊 RequestState
    #   3. engine_core.add_request()        → 送進排程器

    print("\n" + "=" * 60)
    print("Step 2: 手動 add_request()")
    print("=" * 60)

    prompts = [
        "The capital of France is",
        "Machine learning is",
        "The meaning of life is",
    ]

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=30,
    )

    for i, prompt in enumerate(prompts):
        request_id = f"req-{i}"
        engine.add_request(
            request_id=request_id,
            prompt=prompt,
            params=sampling_params,
        )
        print(f"   ✅ Added request: id={request_id}, prompt={prompt!r}")

    # 🔍 確認請求已註冊
    print(f"\n   Unfinished requests: {engine.get_num_unfinished_requests()}")
    print(f"   has_unfinished_requests(): {engine.has_unfinished_requests()}")

    # ============================================
    # Step 3: 手動 step() 循環
    # ============================================
    # 在 01 中，LLM._run_engine() 內部跑的就是這個 while 循環
    # 現在我們自己控制，並且可以觀察每一步的細節
    #
    # 📌 觀察 step() 內部的四步:
    #   1️⃣ engine_core.get_output()        → 取原始輸出
    #   2️⃣ output_processor.process_outputs() → detokenize
    #   3️⃣ engine_core.abort_requests()    → 中止因 stop string 結束的
    #   4️⃣ logger_manager.record()         → 統計

    print("\n" + "=" * 60)
    print("Step 3: 手動 step() 循環")
    print("=" * 60)

    iteration = 0
    all_outputs = []

    while engine.has_unfinished_requests():
        iteration += 1

        # 這就是 LLMEngine.step() — 每次推進一步推理
        step_outputs = engine.step()

        # 🔍 觀察每次 step 的返回值
        finished_in_this_step = [o for o in step_outputs if o.finished]
        in_progress = [o for o in step_outputs if not o.finished]

        if step_outputs:
            print(f"\n--- Iteration {iteration} ---")
            print(f"   step() returned {len(step_outputs)} output(s)")
            print(f"   Finished: {len(finished_in_this_step)}, "
                  f"In progress: {len(in_progress)}")
            print(f"   Remaining unfinished: "
                  f"{engine.get_num_unfinished_requests()}")

        # 收集已完成的輸出
        for output in step_outputs:
            if output.finished:
                all_outputs.append(output)
                print(f"   🎉 Request {output.request_id} FINISHED!")

    print(f"\n   Total iterations: {iteration}")

    # ============================================
    # Step 4: 觀察輸出結構
    # ============================================
    # RequestOutput 是最終用戶拿到的對象
    # 由 OutputProcessor 從 EngineCoreOutputs detokenize 而來

    print("\n" + "=" * 60)
    print("Step 4: 觀察輸出結構 (RequestOutput)")
    print("=" * 60)

    # 按 request_id 排序（跟 LLM._run_engine 做的一樣）
    all_outputs.sort(key=lambda x: x.request_id)

    for output in all_outputs:
        print(f"\n{'─' * 50}")
        print(f"Request ID:     {output.request_id}")
        print(f"Prompt:         {output.prompt!r}")
        print(f"Finished:       {output.finished}")
        print(f"Output text:    {output.outputs[0].text!r}")
        print(f"Finish reason:  {output.outputs[0].finish_reason}")
        print(f"Token count:    {len(output.outputs[0].token_ids)}")

    # ============================================
    # Step 5: 對比 — n=1 vs n>1 (parallel sampling)
    # ============================================
    # 在 notes 中我們學到: n>1 時，add_request 會展開成多個 child requests
    # 讓我們實際觀察這個行為

    print("\n" + "=" * 60)
    print("Step 5: parallel sampling (n=3)")
    print("=" * 60)

    parallel_params = SamplingParams(
        temperature=0.9,
        top_p=0.95,
        max_tokens=20,
        n=3,  # ← 要求 3 個不同的輸出
    )

    req_id = engine.add_request(
        request_id="parallel-0",
        prompt="The best programming language is",
        params=parallel_params,
    )
    print(f"   Added request with n=3, returned req_id={req_id!r}")
    print(f"   Unfinished requests: {engine.get_num_unfinished_requests()}")
    # 🔍 觀察: unfinished 應該是 3 (展開成了 3 個 child requests)

    parallel_outputs = []
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for o in step_outputs:
            if o.finished:
                parallel_outputs.append(o)

    print(f"\n   Got {len(parallel_outputs)} finished outputs:")
    for output in parallel_outputs:
        print(f"   ID: {output.request_id}")
        for i, out in enumerate(output.outputs):
            print(f"      Output {i}: {out.text!r}")


# ============================================
# 🤔 觀察完後思考這些問題：
# ============================================
# Q1: step() 循環跑了多少次？每次返回多少個 output？
#     → 這取決於 max_tokens 和模型生成速度
#
# Q2: finished output 是在哪次 step 出現的？
#     → 觀察 iteration 數字和 "🎉 FINISHED" 的時機
#
# Q3: n=3 的 parallel sampling 中，unfinished_requests 是 1 還是 3？
#     → 應該是 3，因為 add_request 展開成了 3 個 child
#
# Q4: 比較這個 demo 和 01 的 demo：
#     → 01 中 llm.generate() 一��完成的事，
#        在這裡被拆成了 add_request + while step() 循環
#     → LLM 類只是把這些步驟包裝起來而已
#
# Q5: 如果你想實現 streaming（邊生成邊輸出），該怎麼改？
#     → 提示：不要等 output.finished，
#        而是每次 step() 都看 output.outputs[0].text


if __name__ == "__main__":
    main()