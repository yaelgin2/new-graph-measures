"""
EmptyLogger module.

Provides the EmptyLogger class, a no-op logger that ignores all messages.
Useful as a placeholder logger when logging is optional or needs to be
disabled without changing code logic.
"""
import logging

from .base_logger import BaseLogger

# pylint: disable=too-few-public-methods
class EmptyLogger(BaseLogger):
    """
    Logger that ignores all messages.

    Acts as a no-op or placeholder logger. Useful when a logger is required
    by an interface or class but you do not want any actual logging output.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.addHandler(logging.NullHandler())
        self._initialize_handler()
