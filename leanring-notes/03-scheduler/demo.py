"""
demo.py — Standalone simulation of the vLLM v1 Scheduler

Illustrates the core scheduling concepts WITHOUT requiring a GPU or imports from vllm.

Topics covered:
  1. The unified num_computed_tokens / num_tokens_with_spec model
  2. Chunked prefill: long prompts spread across multiple steps
  3. Preemption: KV cache OOM → evict last-admitted running request
  4. Prefix caching: num_computed_tokens starts ahead of 0
  5. NewRequestData vs CachedRequestData output split

Run with:  python study/03-scheduler/demo.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Colours for terminal output
# ─────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
GREY   = "\033[90m"

def section(title: str) -> None:
    width = 68
    print(f"\n{BOLD}{'─' * width}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * width}{RESET}")

def step_header(step: int) -> None:
    print(f"\n{CYAN}  ── step {step} ──{RESET}")

def info(msg: str)   -> None: print(f"    {GREY}{msg}{RESET}")
def ok(msg: str)     -> None: print(f"    {GREEN}✓  {msg}{RESET}")
def warn(msg: str)   -> None: print(f"    {YELLOW}⚠  {msg}{RESET}")
def bad(msg: str)    -> None: print(f"    {RED}✗  {msg}{RESET}")
def label(msg: str)  -> None: print(f"    {BOLD}{msg}{RESET}")


# ─────────────────────────────────────────────
# Core data model
# ─────────────────────────────────────────────

class Request:
    """
    Mirrors the fields on vllm.v1.request.Request that the scheduler actually
    uses to make decisions.

    Key invariant (from scheduler.py line 342–351):
      num_computed_tokens must catch up to num_tokens_with_spec.
    """

    def __init__(
        self,
        req_id: str,
        prompt_len: int,
        max_output_tokens: int = 3,
        prefix_cache_hit: int = 0,   # how many prompt tokens are already cached
    ):
        self.req_id = req_id
        self.prompt_len = prompt_len
        self.max_output_tokens = max_output_tokens

        # --- scheduler-visible state ---
        self.num_computed_tokens: int = 0
        self.num_output_tokens: int = 0       # tokens produced so far
        self.status: str = "WAITING"
        self.num_preemptions: int = 0
        self._prefix_cache_hit: int = prefix_cache_hit

    # num_tokens_with_spec = prompt + output tokens accumulated
    @property
    def num_tokens_with_spec(self) -> int:
        return self.prompt_len + self.num_output_tokens

    # Still has tokens to process?
    @property
    def has_pending_tokens(self) -> bool:
        return self.num_computed_tokens < self.num_tokens_with_spec

    # During prefill: generated tokens = [] (model does not sample mid-prefill)
    @property
    def is_prefilling(self) -> bool:
        return self.num_computed_tokens < self.prompt_len

    def __repr__(self) -> str:
        return (
            f"Request({self.req_id!r}, "
            f"computed={self.num_computed_tokens}/{self.num_tokens_with_spec}, "
            f"out={self.num_output_tokens}/{self.max_output_tokens}, "
            f"status={self.status})"
        )


# ─────────────────────────────────────────────
# Toy KV-cache manager
# ─────────────────────────────────────────────

class KVCacheManager:
    """
    Extremely simplified block allocator.

    Real vLLM uses fixed-size blocks (e.g. 16 tokens per block).
    Here each token costs exactly 1 "slot" to keep the maths obvious.
    """

    def __init__(self, total_slots: int):
        self.total_slots = total_slots
        self._used: dict[str, int] = {}   # req_id → slots currently held

    @property
    def free_slots(self) -> int:
        return self.total_slots - sum(self._used.values())

    def get_computed_blocks(self, request: Request) -> int:
        """
        Prefix-cache lookup.
        Returns the number of already-cached tokens for this request.
        (In the real impl this hashes prompt blocks and checks the cache table.)
        """
        return request._prefix_cache_hit

    def allocate_slots(self, req_id: str, num_new_tokens: int) -> bool:
        """
        Try to allocate `num_new_tokens` additional slots.
        Returns True on success, False if OOM.
        """
        if num_new_tokens <= self.free_slots:
            self._used[req_id] = self._used.get(req_id, 0) + num_new_tokens
            return True
        return False

    def free(self, req_id: str) -> None:
        self._used.pop(req_id, None)

    def status_line(self) -> str:
        used = sum(self._used.values())
        bar_len = 30
        filled = int(bar_len * used / self.total_slots)
        bar = "█" * filled + "░" * (bar_len - filled)
        return f"KV [{bar}] {used}/{self.total_slots} slots used"


# ─────────────────────────────────────────────
# Scheduler output structures
# ─────────────────────────────────────────────

@dataclass
class NewRequestData:
    """
    Sent to the worker the FIRST time a request appears in a batch.
    Contains the full payload (prompt token IDs, sampling params, block IDs…).

    The worker caches this in its persistent InputBatch.
    """
    req_id: str
    prompt_len: int
    num_computed_tokens: int   # initial offset (>0 if prefix cache hit)
    block_ids: list[int]       # which KV slots are allocated

    def summary(self) -> str:
        return (
            f"NewRequestData(req={self.req_id!r}, "
            f"prompt_len={self.prompt_len}, "
            f"cached_prefix={self.num_computed_tokens}, "
            f"blocks={self.block_ids})"
        )


@dataclass
class CachedRequestData:
    """
    Sent for EVERY SUBSEQUENT step for a request that the worker already knows.
    Only ships the diff: new block IDs + updated num_computed_tokens.

    Bandwidth saving: a 4096-token prompt's token ID list is serialized ONCE.
    After that, only a few scalars cross the IPC pipe per decode step.
    """
    req_ids: list[str]
    new_block_ids: list[list[int]]    # only newly allocated blocks
    num_computed_tokens: list[int]

    def summary(self) -> str:
        entries = ", ".join(
            f"{rid}(computed={ct}, new_blocks={nb})"
            for rid, ct, nb in zip(
                self.req_ids, self.num_computed_tokens, self.new_block_ids
            )
        )
        return f"CachedRequestData([{entries}])"


@dataclass
class SchedulerOutput:
    """
    The message emitted each step. Mirrors vllm.v1.core.sched.output.SchedulerOutput.
    """
    scheduled_new_reqs:    list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData

    # {req_id: num_tokens_to_process_this_step}
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int

    preempted_req_ids: list[str] = field(default_factory=list)
    finished_req_ids:  set[str]  = field(default_factory=set)

    def print(self, kv: KVCacheManager) -> None:
        print(f"    {GREY}{kv.status_line()}{RESET}")
        if self.preempted_req_ids:
            warn(f"PREEMPTED: {self.preempted_req_ids}")
        if self.finished_req_ids:
            ok(f"FINISHED:  {sorted(self.finished_req_ids)}")
        if self.scheduled_new_reqs:
            for d in self.scheduled_new_reqs:
                ok(f"NEW    → {d.summary()}")
        if self.scheduled_cached_reqs.req_ids:
            info(f"CACHED → {self.scheduled_cached_reqs.summary()}")
        tokens = [f"{rid}:{n}" for rid, n in self.num_scheduled_tokens.items()]
        label(f"tokens: [{', '.join(tokens)}]  total={self.total_num_scheduled_tokens}")


# ─────────────────────────────────────────────
# The Scheduler
# ─────────────────────────────────────────────

class Scheduler:
    """
    Faithful (but simplified) reproduction of the vLLM v1 Scheduler loop.

    Two main constraints per step:
      - token_budget  : max total tokens across all scheduled requests
      - max_running   : max number of concurrently active requests
    """

    def __init__(
        self,
        token_budget: int,
        max_running: int,
        kv_cache: KVCacheManager,
        enable_chunked_prefill: bool = True,
    ):
        self.token_budget = token_budget
        self.max_running = max_running
        self.kv = kv_cache
        self.enable_chunked_prefill = enable_chunked_prefill

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

        # Worker-side cache: set of req_ids whose full payload has been sent.
        # Mirrors the persistent InputBatch in the real GPU worker.
        self._worker_known: set[str] = set()

        # Finished between last step and this step; ships in next SchedulerOutput.
        self._finished_req_ids: set[str] = set()

    def add_request(self, req: Request) -> None:
        """
        Matches scheduler.py:1726.
        Just enqueues — no KV allocation here.
        """
        self.waiting.append(req)

    # ── schedule() ────────────────────────────────────────────────────────────

    def schedule(self) -> SchedulerOutput:
        """
        Core algorithm. Two phases:

          Phase 1 — RUNNING requests: compute num_new_tokens, allocate KV, preempt if OOM.
          Phase 2 — WAITING requests: prefix-cache lookup, promote if budget allows.

        Then build SchedulerOutput (new vs cached split).
        Finally, _update_after_schedule() advances num_computed_tokens.
        """
        budget = self.token_budget
        preempted: list[str] = []

        scheduled_new_req_objs:    list[Request] = []
        scheduled_cached_req_objs: list[Request] = []
        num_scheduled_tokens:      dict[str, int] = {}
        new_block_ids_per_req:     dict[str, list[int]] = {}  # req_id → new blocks this step

        # ── Phase 1: RUNNING requests ─────────────────────────────────────────
        req_index = 0
        while req_index < len(self.running) and budget > 0:
            req = self.running[req_index]

            # The unified formula (scheduler.py line 397-401):
            num_new = req.num_tokens_with_spec - req.num_computed_tokens
            num_new = min(num_new, budget)

            if num_new == 0:
                req_index += 1
                continue

            # Allocate KV slots. Preempt if OOM.
            while True:
                if self.kv.allocate_slots(req.req_id, num_new):
                    break  # success

                # OOM — preempt last-admitted running request (FCFS policy)
                if not self.running:
                    break
                evict = self.running.pop()       # last element (most recently admitted)
                self.kv.free(evict.req_id)
                evict.status = "PREEMPTED"
                evict.num_computed_tokens = 0    # ← full reset, must recompute
                evict.num_preemptions += 1
                self.waiting.appendleft(evict)   # prepend — back to front of queue
                preempted.append(evict.req_id)

                if evict is req:
                    break  # preempted ourselves; give up on this request

            if not self.kv.allocate_slots.__doc__ or req.req_id not in {
                r.req_id for r in self.running
            }:
                # If we preempted ourselves, req is no longer in running; skip.
                if req not in self.running:
                    continue

            num_scheduled_tokens[req.req_id] = num_new
            new_block_ids_per_req[req.req_id] = list(range(num_new))  # mock block IDs
            scheduled_cached_req_objs.append(req)
            budget -= num_new
            req_index += 1

        # ── Phase 2: WAITING requests ─────────────────────────────────────────
        if not preempted:  # only schedule new work if no preemptions this step
            while self.waiting and budget > 0:
                if len(self.running) >= self.max_running:
                    break

                req = self.waiting[0]   # peek

                # (a) Prefix-cache lookup (scheduler.py line 603-636)
                num_cached = self.kv.get_computed_blocks(req)   # 0 unless prefix hit
                num_new = req.prompt_len - num_cached

                # (b) Chunked-prefill gate (scheduler.py line 665-671)
                if not self.enable_chunked_prefill and num_new > budget:
                    break   # can't fit full prompt; stop scheduling waiting requests

                num_new = min(num_new, budget)

                # (c) Allocate KV slots
                if not self.kv.allocate_slots(req.req_id, num_new):
                    break   # OOM from the waiting side; stop entirely

                # (d) Promote to running
                self.waiting.popleft()
                self.running.append(req)
                req.status = "RUNNING"
                req.num_computed_tokens = num_cached   # start at cache-hit offset!

                num_scheduled_tokens[req.req_id] = num_new
                new_block_ids_per_req[req.req_id] = list(range(num_new))
                scheduled_new_req_objs.append(req)
                budget -= num_new

        # ── Build SchedulerOutput (new vs cached split) ───────────────────────

        # NewRequestData: full payload for first-time requests.
        new_reqs_data = [
            NewRequestData(
                req_id=r.req_id,
                prompt_len=r.prompt_len,
                num_computed_tokens=r.num_computed_tokens,
                block_ids=new_block_ids_per_req.get(r.req_id, []),
            )
            for r in scheduled_new_req_objs
        ]

        # CachedRequestData: diff only for requests the worker already knows.
        cached_reqs_data = CachedRequestData(
            req_ids=[r.req_id for r in scheduled_cached_req_objs],
            new_block_ids=[new_block_ids_per_req.get(r.req_id, []) for r in scheduled_cached_req_objs],
            num_computed_tokens=[r.num_computed_tokens for r in scheduled_cached_req_objs],
        )

        total = sum(num_scheduled_tokens.values())

        output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total,
            preempted_req_ids=preempted,
            finished_req_ids=self._finished_req_ids.copy(),
        )

        # Mark all first-time requests as "known" by the worker.
        for r in scheduled_new_req_objs:
            self._worker_known.add(r.req_id)

        # _update_after_schedule(): advance num_computed_tokens AFTER building output.
        # (So the output describes pre-step state; state is ready for next step.)
        for req_id, n in num_scheduled_tokens.items():
            req_obj = next(r for r in self.running if r.req_id == req_id)
            req_obj.num_computed_tokens += n

        self._finished_req_ids = set()
        return output

    # ── update_from_output() ──────────────────────────────────────────────────

    def update_from_output(self, output: SchedulerOutput) -> list[str]:
        """
        Simulates the feedback after the GPU runs.

        In real vLLM each request gets sampled_token_ids from the model:
          - [] during prefill (no sampling mid-prefill)
          - [token_id] during decode

        Here we just advance the counter and check for done.
        Returns list of finished req_ids this step.
        """
        finished_this_step: list[str] = []

        for req_id in output.num_scheduled_tokens:
            req = next((r for r in self.running if r.req_id == req_id), None)
            if req is None:
                continue

            # During decode (all prompt tokens computed): advance output counter.
            if not req.is_prefilling:
                req.num_output_tokens += 1   # one token sampled

            # Check done.
            if req.num_output_tokens >= req.max_output_tokens:
                req.status = "DONE"
                self.running = [r for r in self.running if r.req_id != req_id]
                self.kv.free(req_id)
                self._finished_req_ids.add(req_id)
                finished_this_step.append(req_id)

        return finished_this_step


# ─────────────────────────────────────────────
# Helper: run until all done
# ─────────────────────────────────────────────

def run_to_completion(
    scheduler: Scheduler,
    requests: list[Request],
    max_steps: int = 30,
) -> None:
    for req in requests:
        scheduler.add_request(req)

    for step in range(1, max_steps + 1):
        if not scheduler.running and not scheduler.waiting:
            break
        step_header(step)
        out = scheduler.schedule()
        out.print(scheduler.kv)
        finished = scheduler.update_from_output(out)
        if finished:
            ok(f"completed: {finished}")


# ─────────────────────────────────────────────
# SCENARIO 1 — Normal decode
# ─────────────────────────────────────────────

def scenario_normal_decode() -> None:
    section("SCENARIO 1 — Normal decode (two short requests, interleaved)")
    print("""
  Two requests with 4-token prompts, each generating 3 output tokens.
  Token budget = 16, max_running = 4, KV slots = 32.

  Expected behaviour:
    Step 1: Both requests promoted from waiting → NewRequestData for each.
            4 tokens each scheduled (full prefill, within budget).
    Step 2: Both requests decoding, 1 token each → CachedRequestData (diff only).
    Steps 3–4: Continues until both hit max_output_tokens.
""")

    kv = KVCacheManager(total_slots=32)
    sched = Scheduler(token_budget=16, max_running=4, kv_cache=kv)

    run_to_completion(sched, [
        Request("r1", prompt_len=4, max_output_tokens=3),
        Request("r2", prompt_len=4, max_output_tokens=3),
    ])


# ─────────────────────────────────────────────
# SCENARIO 2 — Chunked prefill
# ─────────────────────────────────────────────

def scenario_chunked_prefill() -> None:
    section("SCENARIO 2 — Chunked prefill (long prompt, tight token budget)")
    print("""
  One request with a 20-token prompt, token budget = 8.

  Expected behaviour:
    Step 1: r1 promoted from waiting, 8 tokens scheduled (capped by budget).
            r1 enters running with num_computed_tokens=0, gets 8.
            After _update_after_schedule: num_computed_tokens=8.
    Step 2: r1 still in running (it's prefilling), 8 more tokens.
            num_computed_tokens advances to 16.
    Step 3: r1 prefill complete (last 4 tokens), then decode begins.
    Steps 4–6: 1 decode token per step until max_output_tokens=3.
""")

    kv = KVCacheManager(total_slots=32)
    sched = Scheduler(token_budget=8, max_running=4, kv_cache=kv)

    run_to_completion(sched, [
        Request("r1", prompt_len=20, max_output_tokens=3),
    ])


# ─────────────────────────────────────────────
# SCENARIO 3 — Preemption
# ─────────────────────────────────────────────

def scenario_preemption() -> None:
    section("SCENARIO 3 — Preemption (KV cache pressure)")
    print("""
  KV cache has only 12 slots. Three requests, each needing 5-token prompts.

  Expected behaviour:
    Step 1: r1 and r2 both fit (5+5=10 slots used). r3 waits (only 2 free).
    Step 2: r1 is decoding (1 new token → needs 1 more slot → total 11).
            r2 is decoding (1 new slot). Only 12 total.
    Step 3 onwards: when a decode slot allocation fails, last running req
            (r2) is PREEMPTED: blocks freed, num_computed_tokens=0,
            pushed to front of waiting. r1 can continue.
    Later: r2 re-enters and re-prefills from scratch.
""")

    kv = KVCacheManager(total_slots=12)
    sched = Scheduler(token_budget=16, max_running=4, kv_cache=kv)

    run_to_completion(sched, [
        Request("r1", prompt_len=5, max_output_tokens=4),
        Request("r2", prompt_len=5, max_output_tokens=4),
        Request("r3", prompt_len=5, max_output_tokens=2),
    ])


# ─────────────────────────────────────────────
# SCENARIO 4 — Prefix cache hit
# ─────────────────────────────────────────────

def scenario_prefix_cache() -> None:
    section("SCENARIO 4 — Prefix cache hit")
    print("""
  Two requests share the same system prompt (10 tokens).
  r1 is scheduled first and caches those 10 tokens.
  r2 arrives with prefix_cache_hit=10: its num_computed_tokens starts at 10.

  Expected behaviour:
    r1: promoted normally, 10-token prefill, then decode.
    r2: when promoted, kv_cache_manager.get_computed_blocks() returns 10.
        num_computed_tokens = 10 before r2 even starts.
        Only the 5 new tokens (prompt_len=15, 15-10=5) need to be computed.
        r2 reaches decode much faster.
""")

    kv = KVCacheManager(total_slots=64)
    sched = Scheduler(token_budget=20, max_running=4, kv_cache=kv)

    run_to_completion(sched, [
        Request("r1",          prompt_len=10, max_output_tokens=3, prefix_cache_hit=0),
        Request("r2_cached",   prompt_len=15, max_output_tokens=3, prefix_cache_hit=10),
    ])


# ─────────────────────────────────────────────
# SCENARIO 5 — Chunked prefill disabled
# ─────────────────────────────────────────────

def scenario_no_chunked_prefill() -> None:
    section("SCENARIO 5 — Chunked prefill DISABLED")
    print("""
  Same as Scenario 2 (20-token prompt, budget=8) but chunked_prefill=False.

  Expected behaviour:
    The waiting-queue loop hits the gate at scheduler.py line 665-671:
      if not enable_chunked_prefill and num_new_tokens > token_budget: break
    r1 cannot be promoted until the budget is enough for its full prompt (20).
    Since budget is always 8, r1 stays in waiting forever unless budget grows.

    (In practice, vLLM would use a larger budget or enable chunked prefill.)
""")

    kv = KVCacheManager(total_slots=64)
    sched = Scheduler(
        token_budget=8,
        max_running=4,
        kv_cache=kv,
        enable_chunked_prefill=False,   # ← disabled
    )

    # Only run a few steps to show the request stays stuck
    req = Request("r1", prompt_len=20, max_output_tokens=3)
    sched.add_request(req)
    for step in range(1, 6):
        step_header(step)
        out = sched.schedule()
        out.print(kv)
        sched.update_from_output(out)
        if not sched.running and not sched.waiting:
            break

    if sched.waiting:
        warn(f"r1 is STILL in waiting after 5 steps (budget {sched.token_budget} < prompt_len {req.prompt_len})")
        info("→ To fix: enable chunked_prefill=True, or increase max_num_scheduled_tokens")


# ─────────────────────────────────────────────
# SCENARIO 6 — NewRequestData vs CachedRequestData over 4 steps
# ─────────────────────────────────────────────

def scenario_new_vs_cached() -> None:
    section("SCENARIO 6 — NewRequestData vs CachedRequestData over multiple steps")
    print("""
  One request, 3-token prompt, 4 output tokens.
  Shows explicitly which output structure is used each step:

    Step 1 (prefill):  NewRequestData  — full payload shipped to worker for the first time.
    Steps 2–5 (decode): CachedRequestData — diff only (new_block_ids + num_computed_tokens).

  This is the bandwidth optimization: prompt token IDs are serialized exactly once.
""")

    kv = KVCacheManager(total_slots=32)
    sched = Scheduler(token_budget=16, max_running=2, kv_cache=kv)

    req = Request("r1", prompt_len=3, max_output_tokens=4)
    sched.add_request(req)

    for step in range(1, 8):
        if not sched.running and not sched.waiting:
            break
        step_header(step)
        out = sched.schedule()

        if out.scheduled_new_reqs:
            print(f"    {GREEN}→ Worker receives NewRequestData (first time!){RESET}")
            for d in out.scheduled_new_reqs:
                info(d.summary())
        elif out.scheduled_cached_reqs.req_ids:
            print(f"    {GREY}→ Worker receives CachedRequestData (diff only){RESET}")
            info(out.scheduled_cached_reqs.summary())

        sched.update_from_output(out)
        if out.finished_req_ids:
            ok(f"finished: {sorted(out.finished_req_ids)}")


# ─────────────────────────────────────────────
# SCENARIO 7 — InputProcessor pipeline
# ─────────────────────────────────────────────

import uuid as _uuid


def scenario_input_processor() -> None:
    section("SCENARIO 7 — InputProcessor pipeline")
    print("""
  Simulates what InputProcessor.process_inputs() does before a request
  ever reaches the scheduler.

  Stages (from input_processor.py line 195):
    1. Validate SamplingParams / PoolingParams
    2. Tokenize the prompt (if raw text)
    3. Validate prompt length against max_model_len
    4. Validate token IDs against vocab_size
    5. Finalise SamplingParams (merge generation config, EOS tokens)
    6. Assign internal request ID (external_id + 8 random chars)
    7. Build EngineCoreRequest

  Shows three cases:
    (a) Happy path
    (b) Prompt too long → ValueError before the request ever reaches scheduler
    (c) max_tokens=None → auto-filled as max_model_len - prompt_len
""")

    MAX_MODEL_LEN = 32
    VOCAB_SIZE    = 1000

    class MockSamplingParams:
        def __init__(self, max_tokens=None, stop=None, temperature=1.0):
            self.max_tokens  = max_tokens
            self.stop        = stop or []
            self.temperature = temperature

        def verify(self, max_model_len: int) -> None:
            if self.temperature < 0:
                raise ValueError(f"temperature must be >= 0, got {self.temperature}")
            if self.max_tokens is not None and self.max_tokens <= 0:
                raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")

        def fill_max_tokens(self, prompt_len: int, max_model_len: int) -> None:
            if self.max_tokens is None:
                self.max_tokens = max_model_len - prompt_len

        def __repr__(self):
            return (f"SamplingParams(max_tokens={self.max_tokens}, "
                    f"stop={self.stop}, temperature={self.temperature})")

    class MockInputProcessor:
        """Mirrors the key steps in InputProcessor.process_inputs()."""

        def __init__(self, max_model_len: int, vocab_size: int):
            self.max_model_len = max_model_len
            self.vocab_size    = vocab_size
            # Simulate a simple vocabulary: each word maps to a fixed token ID
            self._fake_vocab = {w: i for i, w in enumerate(
                "the quick brown fox jumps over lazy dog tell me joke why did "
                "chicken cross road to get other side".split()
            )}

        def _tokenize(self, text: str) -> list[int]:
            """Extremely simplified tokenizer: split on spaces."""
            return [self._fake_vocab.get(w.lower(), 999) for w in text.split()]

        def _validate_params(self, params: MockSamplingParams) -> None:
            params.verify(self.max_model_len)

        def _validate_length(self, prompt_token_ids: list[int]) -> None:
            n = len(prompt_token_ids)
            if n == 0:
                raise ValueError("Prompt cannot be empty")
            if n >= self.max_model_len:
                raise ValueError(
                    f"Prompt length {n} >= max_model_len {self.max_model_len}. "
                    f"No room for output tokens."
                )

        def _validate_token_ids(self, prompt_token_ids: list[int]) -> None:
            bad = [t for t in prompt_token_ids if t >= self.vocab_size]
            if bad:
                raise ValueError(f"Token IDs out of vocabulary: {bad}")

        @staticmethod
        def _assign_request_id(external_req_id: str) -> str:
            """
            Mirrors InputProcessor.assign_request_id() (line 175).
            Appends 8 random chars to ensure uniqueness even if the
            user sends duplicate request IDs.
            """
            suffix = str(_uuid.uuid4()).replace("-", "")[:8]
            return f"{external_req_id}-{suffix}"

        def process_inputs(
            self,
            request_id: str,
            prompt: str,
            params: MockSamplingParams,
        ) -> dict:
            """Returns a dict mimicking EngineCoreRequest fields."""
            # 1. Validate params
            self._validate_params(params)

            # 2. Tokenize
            prompt_token_ids = self._tokenize(prompt)
            info(f"tokenized '{prompt}' → {prompt_token_ids}")

            # 3. Validate length
            self._validate_length(prompt_token_ids)

            # 4. Validate token IDs
            self._validate_token_ids(prompt_token_ids)

            # 5. Finalise SamplingParams
            params.fill_max_tokens(len(prompt_token_ids), self.max_model_len)
            info(f"sampling_params after finalisation: {params}")

            # 6. Assign internal ID
            internal_id = self._assign_request_id(request_id)
            info(f"external_req_id={request_id!r}  →  request_id={internal_id!r}")

            # 7. Build EngineCoreRequest (as dict for simplicity)
            return dict(
                request_id       = internal_id,
                external_req_id  = request_id,
                prompt_token_ids = prompt_token_ids,
                sampling_params  = params,
            )

    proc = MockInputProcessor(max_model_len=MAX_MODEL_LEN, vocab_size=VOCAB_SIZE)

    # ── (a) Happy path ────────────────────────────────────────────────────────
    step_header("(a) happy path")
    try:
        req = proc.process_inputs("req-42", "Tell me a joke", MockSamplingParams(max_tokens=10))
        ok(f"EngineCoreRequest built: {req}")
    except ValueError as e:
        bad(f"Unexpected error: {e}")

    # ── (b) Prompt too long ───────────────────────────────────────────────────
    step_header("(b) prompt too long (35 tokens > max_model_len=32)")
    long_prompt = " ".join(["the"] * 35)
    try:
        proc.process_inputs("req-43", long_prompt, MockSamplingParams(max_tokens=5))
        bad("Should have raised!")
    except ValueError as e:
        ok(f"Correctly rejected: {e}")

    # ── (c) max_tokens=None auto-filled ──────────────────────────────────────
    step_header("(c) max_tokens=None → auto-filled")
    try:
        params = MockSamplingParams(max_tokens=None)
        info(f"sampling_params before: {params}")
        req = proc.process_inputs("req-44", "Tell me a joke", params)
        ok(f"max_tokens filled to: {req['sampling_params'].max_tokens}  "
           f"(= {MAX_MODEL_LEN} - {len(req['prompt_token_ids'])})")
    except ValueError as e:
        bad(f"Unexpected error: {e}")


# ─────────────────────────────────────────────
# SCENARIO 8 — OutputProcessor: detokenization and stop strings
# ─────────────────────────────────────────────

def scenario_output_processor() -> None:
    section("SCENARIO 8 — OutputProcessor: detokenization and stop strings")
    print("""
  Simulates what OutputProcessor.process_outputs() does after each
  GPU step returns token IDs.

  Key points:
    • IncrementalDetokenizer accumulates token IDs and decodes them
      incrementally (can't decode single tokens independently due to
      multi-byte / BPE boundary issues).
    • Stop-string detection happens HERE (in the frontend), not in EngineCore.
    • When a stop string is found, the request is added to reqs_to_abort —
      the scheduler is then told to stop scheduling it.
    • DELTA mode: emit partial text each step (streaming).
    • FINAL_ONLY mode: emit nothing until finish (batch).

  Shows three cases:
    (a) DELTA mode — text emitted incrementally, then natural end
    (b) FINAL_ONLY mode — nothing emitted until finish
    (c) Stop string detected mid-stream → reqs_to_abort
""")

    # ── Minimal incremental detokenizer simulation ────────────────────────────

    # A fake "vocabulary": token ID → word fragment
    FAKE_VOCAB = {
        10: "Why",  11: " did",  12: " the",  13: " chicken",
        14: " cross", 15: " the",  16: " road", 17: "?",
        18: "\nTo",  19: " get",  20: " to",   21: " the",
        22: " other", 23: " side", 24: "!",
        25: "\n",    26: "END",   # 26 = EOS
    }
    EOS_TOKEN_ID = 26

    class SimpleDetokenizer:
        """
        Mirrors IncrementalDetokenizer behaviour:
          - accumulates output tokens
          - checks for stop strings in the decoded text
          - supports delta (streaming) vs full output
        """
        def __init__(self, stop: list[str]):
            self.stop           = stop
            self.output_tokens: list[int] = []
            self._text_so_far   = ""
            self._last_sent_len = 0   # for delta mode

        def update(self, new_token_ids: list[int]) -> str | None:
            """
            Append tokens, return stop_string if matched, else None.
            Mirrors IncrementalDetokenizer.update().
            """
            for tok in new_token_ids:
                self.output_tokens.append(tok)
                fragment = FAKE_VOCAB.get(tok, f"<{tok}>")
                self._text_so_far += fragment
                for s in self.stop:
                    if s in self._text_so_far:
                        return s   # stop string found
            return None

        def get_delta_text(self) -> str:
            """New text since last call (DELTA mode)."""
            new_text = self._text_so_far[self._last_sent_len:]
            self._last_sent_len = len(self._text_so_far)
            return new_text

        @property
        def full_text(self) -> str:
            return self._text_so_far

    class SimpleOutputProcessor:
        """
        Mirrors OutputProcessor.process_outputs() core loop.
        Tracks request states; each state has a detokenizer.
        """

        def __init__(self):
            # req_id → (detokenizer, output_kind, external_req_id)
            self.states: dict[str, tuple[SimpleDetokenizer, str, str]] = {}

        def add_request(
            self,
            internal_id: str,
            external_id: str,
            output_kind: str,   # "DELTA" or "FINAL_ONLY"
            stop: list[str],
        ) -> None:
            self.states[internal_id] = (SimpleDetokenizer(stop), output_kind, external_id)

        def process_outputs(
            self, outputs: list[tuple[str, list[int], bool]]
        ) -> tuple[list[dict], list[str]]:
            """
            outputs: list of (req_id, new_token_ids, engine_said_finished)
            Returns: (request_outputs, reqs_to_abort)
            """
            request_outputs: list[dict] = []
            reqs_to_abort:   list[str]  = []

            for req_id, new_token_ids, engine_finished in outputs:
                if req_id not in self.states:
                    continue  # already aborted
                detok, output_kind, external_id = self.states[req_id]

                # 1. Detokenize; check for stop string
                stop_string = detok.update(new_token_ids)
                finish_reason = None
                if engine_finished:
                    finish_reason = "length"   # max_tokens reached
                if stop_string:
                    finish_reason = "stop"

                # 2. Build output based on mode
                if output_kind == "DELTA":
                    delta = detok.get_delta_text()
                    if delta or finish_reason:
                        request_outputs.append(dict(
                            request_id  = external_id,  # ← user sees external ID
                            delta_text  = delta,
                            finish_reason = finish_reason,
                        ))
                elif output_kind == "FINAL_ONLY":
                    if finish_reason:  # only emit on finish
                        request_outputs.append(dict(
                            request_id  = external_id,
                            full_text   = detok.full_text,
                            finish_reason = finish_reason,
                        ))

                # 3. Clean up finished requests; signal abort if needed
                if finish_reason:
                    del self.states[req_id]
                    if finish_reason == "stop" and not engine_finished:
                        # Detokenizer found stop string BEFORE EngineCore did.
                        # Must tell scheduler: stop scheduling this request.
                        reqs_to_abort.append(req_id)
                        info(f"reqs_to_abort += [{req_id!r}]  "
                             f"(stop string '{stop_string}' detected by detokenizer)")

            return request_outputs, reqs_to_abort

    # ── (a) DELTA mode, natural end (EOS token) ───────────────────────────────
    step_header("(a) DELTA mode — incremental streaming, natural end")

    op = SimpleOutputProcessor()
    op.add_request("req-42-a3f8b2c1", "req-42", "DELTA", stop=[])

    # Simulate 5 decode steps: token IDs from the "chicken" joke
    steps = [[10], [11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21, 22, 23, 24], [26]]
    for i, token_ids in enumerate(steps, 1):
        engine_done = (EOS_TOKEN_ID in token_ids)
        outs, aborts = op.process_outputs([("req-42-a3f8b2c1", token_ids, engine_done)])
        for o in outs:
            marker = ok if o.get("finish_reason") else info
            marker(f"step {i}: delta={o.get('delta_text')!r}  finish={o.get('finish_reason')}")

    # ── (b) FINAL_ONLY mode — nothing until done ──────────────────────────────
    step_header("(b) FINAL_ONLY mode — silent until finish")

    op2 = SimpleOutputProcessor()
    op2.add_request("req-43-b9d7e1f2", "req-43", "FINAL_ONLY", stop=[])

    steps2 = [[10], [11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21, 22, 23, 24], [26]]
    for i, token_ids in enumerate(steps2, 1):
        engine_done = (EOS_TOKEN_ID in token_ids)
        outs, _ = op2.process_outputs([("req-43-b9d7e1f2", token_ids, engine_done)])
        if outs:
            ok(f"step {i}: FINAL output emitted → text={outs[0].get('full_text')!r}  "
               f"finish={outs[0].get('finish_reason')}")
        else:
            info(f"step {i}: no output (FINAL_ONLY, not done yet)")

    # ── (c) Stop string detected mid-stream ───────────────────────────────────
    step_header("(c) Stop string '?' detected — triggers reqs_to_abort")

    op3 = SimpleOutputProcessor()
    op3.add_request("req-44-c2a1b3d4", "req-44", "DELTA", stop=["?"])

    # Scheduler is still running, will keep sending tokens until told to stop.
    # After '?' appears, detokenizer catches it; engine has NOT finished yet.
    steps3 = [[10], [11, 12, 13], [14, 15, 16, 17]]  # ends with '?'
    for i, token_ids in enumerate(steps3, 1):
        outs, aborts = op3.process_outputs([("req-44-c2a1b3d4", token_ids, False)])
        for o in outs:
            if o.get("finish_reason"):
                warn(f"step {i}: stop string triggered → text={o.get('delta_text')!r}  "
                     f"finish={o.get('finish_reason')}")
            else:
                info(f"step {i}: delta={o.get('delta_text')!r}")
        if aborts:
            bad(f"  → reqs_to_abort sent to scheduler: {aborts}")
            info("  → scheduler.finish_requests(reqs_to_abort, FINISHED_ABORTED)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    scenario_normal_decode()
    scenario_chunked_prefill()
    scenario_preemption()
    scenario_prefix_cache()
    scenario_no_chunked_prefill()
    scenario_new_vs_cached()
    scenario_input_processor()
    scenario_output_processor()

    print(f"\n{BOLD}{'─' * 68}{RESET}")
    print(f"{BOLD}  All scenarios complete.{RESET}")
    print(f"{BOLD}{'─' * 68}{RESET}\n")
