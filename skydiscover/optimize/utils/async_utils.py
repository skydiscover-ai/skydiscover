"""
Async utilities for SkyDiscover
"""

import asyncio
import logging
from typing import Any, Callable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class TaskPool:
    """
    A simple task pool for managing and limiting concurrent tasks
    """

    def __init__(self, max_concurrency: int = 10):
        self.max_concurrency = max_concurrency
        self._semaphore: Optional[asyncio.Semaphore] = None
        self.tasks: List[asyncio.Task] = []

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Lazy-initialize the semaphore when first needed."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    async def run(self, coro: Callable, *args: Any, **kwargs: Any) -> Any:
        """Run a single coroutine function under the concurrency semaphore."""
        async with self.semaphore:
            return await coro(*args, **kwargs)

    def create_task(self, coro: Callable, *args: Any, **kwargs: Any) -> asyncio.Task:
        """Create, track, and return an ``asyncio.Task`` bounded by the pool."""
        task = asyncio.create_task(self.run(coro, *args, **kwargs))
        self.tasks.append(task)
        task.add_done_callback(lambda t: self.tasks.remove(t))
        return task

    async def gather(
        self,
        coros: Sequence[Callable],
        args_list: Sequence[Tuple[Any, ...]] = (),
        kwargs_list: Sequence[dict] = (),
        return_exceptions: bool = False,
    ) -> List[Any]:
        """Run *coros* concurrently (bounded by the semaphore), return results in order."""
        n = len(coros)
        _args = args_list if args_list else [() for _ in range(n)]
        _kwargs = kwargs_list if kwargs_list else [{} for _ in range(n)]

        if len(_args) != n:
            raise ValueError(f"args_list length ({len(_args)}) must match coros length ({n})")
        if len(_kwargs) != n:
            raise ValueError(f"kwargs_list length ({len(_kwargs)}) must match coros length ({n})")

        tasks = [
            self.create_task(coro, *args, **kwargs)
            for coro, args, kwargs in zip(coros, _args, _kwargs)
        ]
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
