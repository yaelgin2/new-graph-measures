"""
PrintLogger module.

Provides the PrintLogger class, a logger that outputs messages to the
console (stdout). Automatically attaches a StreamHandler for convenience.
"""
import logging
import sys

from .base_logger import BaseLogger

# pylint: disable=too-few-public-methods
class PrintLogger(BaseLogger):
    """
    Logger that outputs messages to the console (stdout).

    Automatically adds a StreamHandler to stdout, allowing quick and easy
    logging to the console without requiring additional setup. Ideal for
    debugging or interactive use.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.addHandler(logging.StreamHandler(stream=sys.stdout))  # create console handler
        self._initialize_handler()
