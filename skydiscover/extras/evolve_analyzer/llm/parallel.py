# Vendored from clear_eval/pipeline/inference_utils/llm_client.py — parallel execution section
# (CLEAR commit 740bb0c49d782d2e49e9aa3fddabf8378ba88554, 2026-04-16).
# Maintained independently within evolve-analyzer.
# To incorporate upstream improvements, diff manually against the CLEAR source.

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from skydiscover.extras.evolve_analyzer.llm.client import ParallelResult, _get_or_create_event_loop

logger = logging.getLogger(__name__)


def run_async(coro):
    """Run an async coroutine on the shared event loop."""
    loop = _get_or_create_event_loop()
    return loop.run_until_complete(coro)


def _run_threaded(
    func: Callable,
    inputs: List[Any],
    max_workers: int = 10,
    task_timeout: float = 300,
    error_prefix: str = "Error",
    progress_desc: str = "Processing",
) -> List[ParallelResult]:
    if not inputs:
        return []

    if len(inputs) == 1:
        item = inputs[0]
        try:
            result = func(*item) if isinstance(item, tuple) else func(item)
            return [ParallelResult(is_success=True, result=result)]
        except Exception as e:
            return [ParallelResult(is_success=False, error=f"{error_prefix}: {e}")]

    results: List[ParallelResult] = [None] * len(inputs)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers) as executor:
        future_to_idx = {}
        for i, item in enumerate(inputs):
            future = executor.submit(func, *item) if isinstance(item, tuple) else executor.submit(func, item)
            future_to_idx[future] = i

        for future in tqdm(as_completed(future_to_idx), total=len(inputs), desc=progress_desc):
            idx = future_to_idx[future]
            try:
                results[idx] = ParallelResult(is_success=True, result=future.result(timeout=task_timeout))
            except Exception as e:
                logger.error(f"Task {idx} failed: {e}")
                results[idx] = ParallelResult(is_success=False, error=f"{error_prefix} item {idx}: {e}")

    return results


async def _run_async(
    func: Callable,
    inputs: List[Any],
    max_workers: int = 10,
    task_timeout: float = 300,
    error_prefix: str = "Error",
    progress_desc: str = "Processing",
) -> List[ParallelResult]:
    if not inputs:
        return []

    semaphore = asyncio.Semaphore(max_workers)

    async def limited_call(idx: int, item) -> ParallelResult:
        async with semaphore:
            try:
                coro = func(*item) if isinstance(item, tuple) else func(item)
                result = await asyncio.wait_for(coro, timeout=task_timeout)
                return ParallelResult(is_success=True, result=result)
            except asyncio.TimeoutError:
                return ParallelResult(is_success=False, error=f"{error_prefix} item {idx}: Timeout")
            except Exception as e:
                logger.error(f"Task {idx} failed: {e}")
                return ParallelResult(is_success=False, error=f"{error_prefix} item {idx}: {e}")

    tasks = [limited_call(i, item) for i, item in enumerate(inputs)]
    return await tqdm_asyncio.gather(*tasks, desc=progress_desc)


def run_parallel(
    func: Callable,
    inputs: List[Any],
    use_async: bool = False,
    max_workers: int = 10,
    task_timeout: float = 300,
    error_prefix: str = "Error",
    progress_desc: str = "Processing",
) -> List[ParallelResult]:
    """
    Run func over inputs in parallel.

    Each element of inputs is either a single value or a tuple of positional args.
    use_async=True requires func to be an async function.
    Returns results in the same order as inputs.
    """
    if use_async:
        return run_async(
            _run_async(func, inputs, max_workers, task_timeout, error_prefix, progress_desc)
        )
    return _run_threaded(func, inputs, max_workers, task_timeout, error_prefix, progress_desc)
