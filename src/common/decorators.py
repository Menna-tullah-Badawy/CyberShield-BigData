"""
CyberShield-BigData: Performance & Monitoring Decorators
========================================================

Decorators for measuring execution time and monitoring
pipeline operations.
"""

import time
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from src.common.logger import get_logger


logger = get_logger("PerformanceDecorator")


F = TypeVar("F", bound=Callable[..., Any])


def measure_performance(func: F) -> F:
    """
    Measure and log the execution time of a function.

    The decorator:
    - Logs when execution starts.
    - Measures execution duration.
    - Logs successful completion.
    - Logs exceptions.
    - Re-raises the original exception without modifying it.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:

        func_name = func.__name__

        logger.info(
            f"⏳ Starting execution for: [{func_name}]..."
        )

        start_time = time.perf_counter()

        try:

            result = func(
                *args,
                **kwargs
            )

            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            logger.info(
                f"⏱️ Finished [{func_name}] "
                f"successfully in: "
                f"{elapsed_time:.4f}s"
            )

            return result

        except Exception:

            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            logger.exception(
                f"❌ Error occurred during "
                f"[{func_name}] after "
                f"{elapsed_time:.4f}s"
            )

            # IMPORTANT:
            # Re-raise the original exception while
            # preserving the original traceback.
            raise

    return cast(F, wrapper)


# ============================================================
# Aliases
# ============================================================

time_execution = measure_performance

log_execution_time = measure_performance

performance_decorator = measure_performance

track_performance = measure_performance