"""
01-entrypoints Demo: 追蹤 LLM.generate() 的調用鏈

使用方式 (在 PyCharm 中)：
1. 打開 vllm 項目
2. 在以下標記的地方設 breakpoint
3. Debug 模式運行此腳本
4. 逐步跟蹤，觀察調用順序

注意：需要下載模型，首次運行會比較慢
如果 MacBook 記憶體不夠，可以用更小的模型
"""

from vllm import LLM, SamplingParams


def main():
    """
    Note: Necessary environment variables to run this file successfully:
        TORCHDYNAMO_DISABLE=1;
        VLLM_HOST_IP=127.0.0.1;
    """
    # ============================================
    print("Step 1: 建立 LLM 實例")
    # ============================================
    # 📌 在 PyCharm 中打開 vllm/entrypoints/llm.py
    #    在 line 382 設 breakpoint:
    #    self.llm_engine = LLMEngine.from_engine_args(...)
    #
    # 觀察：EngineArgs 包含了哪些配置？
    llm = LLM(
        model="facebook/opt-125m",  # 很小的模型，適合本地測試
        # 如果記憶體不夠，試試：
        # model="sshleifer/tiny-gpt2",
        dtype="float32",            # CPU 模式用 float32
        enforce_eager=True,         # 關閉 CUDA graph，適合 debug
    )

    # ============================================
    print("Step 2: 設定取樣參數")
    # ============================================
    sampling_params = SamplingParams(
        # temperature=0.8,
        top_p=0.95,
        # max_tokens=30,
        temperature=0.0,  # greedy — easiest to reason about for a demo
        max_tokens=50,
        stop=["\n\n"],  # stop at paragraph break
    )

    # ============================================
    print("Step 3: 調用 generate()")
    # ============================================
    # 📌 設 breakpoint 的好地方：
    #    1. vllm/entrypoints/llm.py line 443  → generate() 入口
    #    2. vllm/entrypoints/llm.py line 1833 → _run_completion()
    #    3. vllm/entrypoints/llm.py line 1963 → _add_request() 送進引擎
    #    4. vllm/entrypoints/llm.py line 1984 → _run_engine() 主循環
    #    5. 關注 self.llm_engine.step() 的返回值

    prompts = [
        "The capital of France is",
        "Machine learning is",
    ]

    outputs = llm.generate(prompts, sampling_params)

    # ============================================
    print("Step 4: 觀察輸出結構")
    # ============================================
    for output in outputs:
        print(f"\n{'='*50}")
        print(f"Request ID: {output.request_id}")
        print(f"Prompt: {output.prompt!r}")
        print(f"Prompt tokens: {output.prompt_token_ids[:10]}...")  # 前10個
        print(f"Number of outputs: {len(output.outputs)}")
        for i, out in enumerate(output.outputs):
            print(f"  Output {i}:")
            print(f"    Text: {out.text!r}")
            print(f"    Token IDs: {out.token_ids[:10]}...")
            print(f"    Finish reason: {out.finish_reason}")

# ============================================
# 🤔 觀察完後思考這些問題：
# ============================================
# Q1: LLMEngine 是什麼時候被建立的？裡面包含哪些組件？
# Q2: _run_engine 的 while 循環跑了幾次？每次 step() 返回什麼？
# Q3: 兩個 prompt 是怎麼被 batch 在一起的？
# Q4: output.request_id 是遞增的嗎？排序的意義是什麼？


if __name__ == "__main__":
    main()