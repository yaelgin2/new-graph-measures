from datetime import datetime
from functools import wraps


def time_log(func):
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