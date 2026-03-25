# =============================================================================
# vLLM — LLM Class Lifecycle Demo
# MacBook Pro (CPU / Apple Silicon Metal backend)
#
# Run with:  python examples/demo_llm_lifecycle.py
# Interpreter: /Users/bohuang/vllm/.venv/bin/python
# =============================================================================
#
# LIFECYCLE OVERVIEW
# ==================
# 1. LLM.__init__()            — load & validate config, init engine
# 2. InputProcessor            — tokenize / encode the prompt
# 3. Scheduler                 — schedule the request into a batch
# 4. ModelRunner.forward()     — run the PyTorch forward pass (CPU)
# 5. OutputProcessor           — decode token IDs → text
# 6. RequestOutput             — returned to the caller
#
# =============================================================================

from vllm import LLM, SamplingParams

# -----------------------------------------------------------------------------
# STEP 1 — Create the LLM object
# -----------------------------------------------------------------------------
# What happens under the hood:
#   • Reads VllmConfig (model, parallel, cache, scheduler configs)
#   • Detects platform → CpuPlatform on macOS
#   • Initialises EngineCore (in the same process for CPU/single-device)
#   • Loads model weights from HuggingFace (cached in ~/.cache/huggingface)
#   • Allocates the KV-cache block pool (PagedAttention)
#
# We use GPT-2 (small, ~500 MB) so the download is fast.
# Set max_model_len small to keep memory usage low on a laptop.


def main():
    print("\n=== STEP 1: Initialising LLM ===")
    llm = LLM(
        model="gpt2",          # 117M-parameter model — fast on CPU
        max_model_len=256,     # limit context window for demo purposes
        dtype="float32",       # CPU doesn't support float16 on macOS
        enforce_eager=True,    # skip CUDA graph capture (not needed on CPU)
    )
    print("LLM ready.\n")


    # -----------------------------------------------------------------------------
    # STEP 2 — Define sampling parameters
    # -----------------------------------------------------------------------------
    # SamplingParams controls how tokens are sampled at each step:
    #   temperature  : 0.0 = greedy (deterministic); >0 = random sampling
    #   max_tokens   : stop after this many new tokens
    #   stop         : stop if any of these strings is generated

    print("=== STEP 2: SamplingParams ===")
    params = SamplingParams(
        temperature=0.0,   # greedy — easiest to reason about for a demo
        max_tokens=50,
        stop=["\n\n"],     # stop at paragraph break
    )
    print(f"  temperature={params.temperature}, max_tokens={params.max_tokens}\n")


    # -----------------------------------------------------------------------------
    # STEP 3 — Submit prompts and run generation
    # -----------------------------------------------------------------------------
    # What happens under the hood:
    #   • LLM.generate() calls InputProcessor → tokenises each prompt
    #   • Each tokenised prompt becomes an EngineCoreRequest
    #   • EngineCore.step() loop:
    #       a. Scheduler selects the batch and allocates KV-cache blocks
    #       b. Executor.execute_model() sends the batch to the Worker
    #       c. Worker.GPUModelRunner (or CpuModelRunner) runs the forward pass
    #       d. Logits → SamplingParams → next token
    #       e. Repeat until max_tokens or stop condition
    #   • OutputProcessor detokenises the generated token IDs
    #   • Returns a list of RequestOutput objects

    prompts = [
        "The capital of France is",
        "In machine learning, a transformer model is",
        "vLLM is a fast inference engine because",
    ]

    print("=== STEP 3: Running generation ===")
    outputs = llm.generate(prompts, params)


    # -----------------------------------------------------------------------------
    # STEP 4 — Inspect the output
    # -----------------------------------------------------------------------------
    # RequestOutput fields:
    #   request_id        : unique ID assigned by the engine
    #   prompt            : original text
    #   prompt_token_ids  : tokenised prompt IDs
    #   outputs           : list of CompletionOutput (one per beam / sample)
    #     .text           : generated text (decoded)
    #     .token_ids      : raw generated token IDs
    #     .finish_reason  : "stop" | "length" | "abort"
    #     .logprobs       : per-token log-probabilities (if requested)

    print("\n=== STEP 4: Results ===\n")
    for output in outputs:
        print(f"Request ID   : {output.request_id}")
        print(f"Prompt       : {output.prompt!r}")
        print(f"Prompt tokens: {output.prompt_token_ids}")
        completion = output.outputs[0]           # first (and only) beam
        print(f"Generated    : {completion.text!r}")
        print(f"Token IDs    : {completion.token_ids}")
        print(f"Finish reason: {completion.finish_reason}")
        print("-" * 60)


if __name__ == "__main__":
    main()