"""
BaseLogger module.

Provides the BaseLogger class, a customizable logger that allows setting
log level, formatting, and handler initialization. Serves as the base
class for other logger implementations.
"""

import logging

from .constants import DEFAULT_LOG_FORMAT


class BaseLogger(logging.getLoggerClass()):
    """
    Base logger class with customizable format and level.

    Attributes:
        formatter (logging.Formatter): Formatter used for all handlers.
         can configure level=logging.[CRITICAL(50), FATAL(CRITICAL), ERROR(40),
         WARNING(30), WARN(WARNING), INFO(20), DEBUG(10), NOTSET(0 - default)]
         for various log_format options see help(logging.Formatter)
    """
    def __init__(self, name=None, level=logging.NOTSET, log_format=None):
        if log_format is None:
            log_format = DEFAULT_LOG_FORMAT
            if name is None:
                log_format = log_format[:1] + log_format[2:]
            log_format = " - ".join(log_format)

        if name is None:
            name = type(self).__name__
        super().__init__(name, level=level)

        # create formatter
        self.formatter = logging.Formatter(log_format)

    def _set_format(self, *args, **kwargs):
        self.formatter = logging.Formatter(*args, **kwargs)
        list(map(lambda handler: handler.setFormatter(self.formatter), self.handlers))

    def _initialize_handler(self):
        list(map(lambda handler: handler.setLevel(self.level), self.handlers))

        # attach formatter to handlers
        list(map(lambda handler: handler.setFormatter(self.formatter), self.handlers))

    def close(self):
        """
        Shutdown the logging system and release all handler resources.

        This method ensures that all logging handlers associated with this
        logger are properly closed and references are cleaned up. After
        calling this method, the logger should not be used.
        """
        logging.shutdown([x.__weakref__ for x in self.handlers])
