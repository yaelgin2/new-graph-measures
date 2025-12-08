# pylint: disable=protected-access

"""
Timing decorator for feature calculators.

Provides the `time_log` decorator to log the start and finish of
feature computation methods, including elapsed time.
"""
from datetime import datetime
from functools import wraps


def time_log(func):
    """
    Decorator to log the start and end times of a method call.

    Logs messages using the `_logger` attribute of the instance
    and includes the method/feature name from `_print_name`.

    Args:
        func (Callable): The method to wrap.

    Returns:
        Callable: Wrapped method with logging of start, finish, and elapsed time.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = datetime.now()
        if hasattr(self, "_logger") and hasattr(self, "_print_name"):
            self._logger.debug(f"Start {self._print_name}")
        result = func(self, *args, **kwargs)
        end_time = datetime.now()
        elapsed = end_time - start_time
        if hasattr(self, "_logger") and hasattr(self, "_print_name"):
            self._logger.debug(f"Finish {self._print_name} in {elapsed}")
        return result
    return wrapper
