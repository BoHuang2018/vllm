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

    # But, parallel_outputs got only one inference result.
    # Looks like array appending is not a proper way to collect results in this case.
    # We will know the reason when we dig more in the repo.
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

""" Running record
/Users/bohuang/vllm/.venv/bin/python /Users/bohuang/vllm/leanring-notes/02-llm_engine/demo.py 
INFO 03-30 22:27:45 [__init__.py:44] Available plugins for group vllm.platform_plugins:
INFO 03-30 22:27:45 [__init__.py:46] - metal -> vllm_metal:register
INFO 03-30 22:27:45 [__init__.py:49] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
============================================================
Step 1: 建立 LLMEngine
============================================================
INFO 03-30 22:28:09 [model.py:549] Resolved architecture: OPTForCausalLM
INFO 03-30 22:28:09 [model.py:2010] Upcasting torch.float16 to torch.float32.
INFO 03-30 22:28:09 [model.py:1678] Using max model len 2048
WARNING 03-30 22:28:09 [cpu.py:136] VLLM_CPU_KVCACHE_SPACE not set. Using 16.0 GiB for KV cache.
INFO 03-30 22:28:09 [scheduler.py:238] Chunked prefill is enabled with max_num_batched_tokens=2048.
INFO 03-30 22:28:09 [vllm.py:786] Asynchronous scheduling is enabled.
WARNING 03-30 22:28:09 [vllm.py:844] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
WARNING 03-30 22:28:09 [vllm.py:855] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
INFO 03-30 22:28:10 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
INFO 03-30 22:28:10 [compilation.py:290] Enabled custom fusions: norm_quant, act_quant
INFO 03-30 22:28:12 [__init__.py:44] Available plugins for group vllm.platform_plugins:
INFO 03-30 22:28:12 [__init__.py:46] - metal -> vllm_metal:register
INFO 03-30 22:28:12 [__init__.py:49] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
(EngineCore pid=11747) INFO 03-30 22:28:15 [core.py:105] Initializing a V1 LLM engine (v0.1.dev14592+gb7332b058) with config: model='facebook/opt-125m', speculative_config=None, tokenizer='facebook/opt-125m', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float32, max_seq_len=2048, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=True, quantization=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cpu, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=facebook/opt-125m, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['all'], 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_images_per_batch': 0, 'compile_sizes': None, 'compile_ranges_endpoints': [2048], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': None, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}
(EngineCore pid=11747) INFO 03-30 22:28:15 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
(EngineCore pid=11747) INFO 03-30 22:28:16 [cpu_worker.py:109] Warning: NUMA is not enabled in this build. `init_cpu_threads_env` has no effect to setup thread affinity.
(EngineCore pid=11747) INFO 03-30 22:28:16 [parallel_state.py:1400] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://127.0.0.1:57102 backend=gloo
[W330 22:28:16.481362000 ProcessGroupGloo.cpp:542] Warning: Unable to resolve hostname to a (local) address. Using the loopback address as fallback. Manually set the network interface to bind to with GLOO_SOCKET_IFNAME. (function operator())
(EngineCore pid=11747) INFO 03-30 22:28:16 [parallel_state.py:1716] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(EngineCore pid=11747) INFO 03-30 22:28:16 [cpu_model_runner.py:71] Starting to load model facebook/opt-125m...
Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  8.51it/s]
Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  8.50it/s]
(EngineCore pid=11747) 
(EngineCore pid=11747) INFO 03-30 22:28:17 [default_loader.py:384] Loading weights took 0.12 seconds
(EngineCore pid=11747) INFO 03-30 22:28:17 [kv_cache_utils.py:1319] GPU KV cache size: 232,960 tokens
(EngineCore pid=11747) INFO 03-30 22:28:17 [kv_cache_utils.py:1324] Maximum concurrency for 2,048 tokens per request: 113.75x
(EngineCore pid=11747) INFO 03-30 22:28:19 [cpu_model_runner.py:82] Warming up model for the compilation...
(EngineCore pid=11747) INFO 03-30 22:28:19 [cpu_model_runner.py:92] Warming up done.
(EngineCore pid=11747) INFO 03-30 22:28:19 [core.py:283] init engine (profile, create kv cache, warmup model) took 2.06 seconds
(EngineCore pid=11747) INFO 03-30 22:28:20 [vllm.py:786] Asynchronous scheduling is disabled.
(EngineCore pid=11747) WARNING 03-30 22:28:20 [vllm.py:844] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
(EngineCore pid=11747) WARNING 03-30 22:28:20 [vllm.py:855] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(EngineCore pid=11747) WARNING 03-30 22:28:20 [cpu.py:136] VLLM_CPU_KVCACHE_SPACE not set. Using 16.0 GiB for KV cache.
(EngineCore pid=11747) INFO 03-30 22:28:20 [compilation.py:290] Enabled custom fusions: norm_quant, act_quant

✅ LLMEngine 建立完成!
   InputProcessor:  InputProcessor
   OutputProcessor: OutputProcessor
   EngineCoreClient: SyncMPClient
   Tokenizer:       CachedGPT2TokenizerFast

============================================================
Step 2: 手動 add_request()
============================================================
🔵 [LLMEngine.add_request] request_id=req-0, prompt_type=<class 'str'>
WARNING 03-30 22:28:20 [input_processor.py:235] Passing raw prompts to InputProcessor is deprecated and will be removed in v0.18. You should instead pass the outputs of Renderer.render_cmpl() or Renderer.render_chat().
   ✅ Added request: id=req-0, prompt='The capital of France is'
🔵 [LLMEngine.add_request] request_id=req-1, prompt_type=<class 'str'>
   ✅ Added request: id=req-1, prompt='Machine learning is'
🔵 [LLMEngine.add_request] request_id=req-2, prompt_type=<class 'str'>
   ✅ Added request: id=req-2, prompt='The meaning of life is'

   Unfinished requests: 3
   has_unfinished_requests(): True

============================================================
Step 3: 手動 step() 循環
============================================================
🟢 [LLMEngine.step] 1. engine_core.get_output() → 1 outputs
🟢 [LLMEngine.step] 2. process_outputs → 1 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 1 ---
   step() returned 1 output(s)
   Finished: 0, In progress: 1
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 2 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 3 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 4 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 5 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 6 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 7 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 8 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 9 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 10 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 11 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 12 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 13 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 14 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 15 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 16 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 17 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 18 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 19 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 20 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 21 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 22 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 23 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 24 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 25 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 26 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 27 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 28 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 29 ---
   step() returned 3 output(s)
   Finished: 0, In progress: 3
   Remaining unfinished: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 30 ---
   step() returned 3 output(s)
   Finished: 1, In progress: 2
   Remaining unfinished: 2
   🎉 Request req-0 FINISHED!
🟢 [LLMEngine.step] 1. engine_core.get_output() → 2 outputs
🟢 [LLMEngine.step] 2. process_outputs → 2 results
🟢 [LLMEngine.step] 3. abort_requests → []

--- Iteration 31 ---
   step() returned 2 output(s)
   Finished: 2, In progress: 0
   Remaining unfinished: 0
   🎉 Request req-1 FINISHED!
   🎉 Request req-2 FINISHED!

   Total iterations: 31

============================================================
Step 4: 觀察輸出結構 (RequestOutput)
============================================================

──────────────────────────────────────────────────
Request ID:     req-0
Prompt:         'The capital of France is'
Finished:       True
Output text:    'TheTheTheTheTheTheTheTheTheTheSemTheNumberThePhoneTheStreetTheTheUpTheTTheStreetTheTimesThe"TheS'
Finish reason:  length
Token count:    30

──────────────────────────────────────────────────
Request ID:     req-1
Prompt:         'Machine learning is'
Finished:       True
Output text:    'MachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachineMachine'
Finish reason:  length
Token count:    30

──────────────────────────────────────────────────
Request ID:     req-2
Prompt:         'The meaning of life is'
Finished:       True
Output text:    'TheTheTheTheTheTheTheTheTheTheTheUniversityTheTheInTheTheRedTheTheFianTheTheTheWindowTheSherTheTy'
Finish reason:  length
Token count:    30

============================================================
Step 5: parallel sampling (n=3)
============================================================
🔵 [LLMEngine.add_request] request_id=parallel-0, prompt_type=<class 'str'>
   Added request with n=3, returned req_id='parallel-0-bf2f9394'
   Unfinished requests: 3
🟢 [LLMEngine.step] 1. engine_core.get_output() → 0 outputs
🟢 [LLMEngine.step] 2. process_outputs → 0 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 1 outputs
🟢 [LLMEngine.step] 2. process_outputs → 1 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 3 outputs
🟢 [LLMEngine.step] 2. process_outputs → 3 results
🟢 [LLMEngine.step] 3. abort_requests → []
🟢 [LLMEngine.step] 1. engine_core.get_output() → 2 outputs
🟢 [LLMEngine.step] 2. process_outputs → 2 results
🟢 [LLMEngine.step] 3. abort_requests → []

   Got 1 finished outputs:
   ID: parallel-0
      Output 0: 'TheTheTheTheThatTheTheTheTheTheThePhotoByTheSpeTheGlass,TheToday'
(EngineCore pid=11747) INFO 03-30 22:28:22 [core.py:1210] Shutdown initiated (timeout=0)
(EngineCore pid=11747) INFO 03-30 22:28:22 [core.py:1233] Shutdown complete

Process finished with exit code 0

"""

