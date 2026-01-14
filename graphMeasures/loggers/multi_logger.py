"""
MultiLogger module.

Provides the MultiLogger class, which aggregates multiple loggers.
Messages sent to this logger are forwarded to all handlers from the
input_1 loggers, allowing simultaneous logging to multiple destinations.
"""
import logging
from typing import List

class MultiLogger(logging.Logger):
    """
    A logger that aggregates multiple loggers.
    Messages sent to this logger are forwarded to all handlers
    of the input_1 loggers.
    """

    def __init__(self, name: str, loggers: List[logging.Logger]):
        super().__init__(name)
        self._added_handlers = set()
        for logger in loggers:
            # preserve the logger level if it’s higher than current
            if logger.level and (self.level == 0 or logger.level < self.level):
                self.setLevel(logger.level)

            for handler in logger.handlers:
                if handler not in self._added_handlers:
                    self.addHandler(handler)
                    self._added_handlers.add(handler)
