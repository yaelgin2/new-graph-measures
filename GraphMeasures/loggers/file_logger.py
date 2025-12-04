"""
FileLogger module.

Provides the FileLogger class, a logger that writes messages to a file.
Automatically attaches a FileHandler and supports customizable formatting
and logging levels.
"""
import logging
import os
import time

from .base_logger import BaseLogger


class FileLogger(BaseLogger):
    """
    Logger that writes messages to a file.

    Automatically adds a FileHandler to write log messages to the specified
    file path. Supports custom logging levels and formatting. Useful for
    persistent logging and record-keeping.
    """
    # pylint: disable=too-many-arguments
    def __init__(self, filename, *args, ext="log", path="logs", add_timestamp=False,
                 should_overwrite=True, **kwargs):
        super().__init__(*args, **kwargs)

        if path == "":
            raise ValueError("Path cannot be empty.")
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except OSError:
                return

        filename = os.path.join(path, filename)
        if add_timestamp:
            filename += time.strftime("-%Y-%m-%d-%H%M%S")

        mode = 'w' if should_overwrite else 'a'

        self.addHandler(logging.FileHandler(f"{filename}.{ext}", mode=mode))
        self._initialize_handler()
