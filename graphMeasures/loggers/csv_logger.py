"""
CSVLogger module.

Provides the CSVLogger class, which logs messages in CSV format.
Useful for structured logging and later analysis in spreadsheet or
data processing tools.
"""
from .file_logger import FileLogger


class CSVLogger(FileLogger):
    """
    Logger that outputs messages to a CSV file.

    Specialized logger that formats each log record as a CSV row. Useful
    for structured logging where logs need to be parsed or analyzed later
    in spreadsheet or data processing tools.
    """
    def __init__(self, *args, **kwargs):
        if "ext" not in kwargs:
            kwargs["ext"] = "csv"
        kwargs["log_format"] = "%(message)s"
        self._delimiter = kwargs.pop("delimiter", ",")
        self._other_del = kwargs.pop("other_delimiter", "-")
        super().__init__(*args, **kwargs)

    def space(self, num_spaces=1):
        """
        Insert blank lines in the log output.

        Args:
            num_spaces (int): Number of blank lines to insert. Default is 1.

        This method logs newline characters to create spacing in the log
        for better readability.
        """
        self.info("\n" * num_spaces)

    def info(self, msg, *args, **kwargs):
        """
        Log a message at INFO level with CSV-specific formatting.

        Args:
            msg: The main log message.
            *args: Additional arguments to include in the log.
            **kwargs: Extra keyword arguments passed to the base logger.

        Replaces occurrences of the logger's delimiter in each argument
        and joins them before passing to the parent logger's `info`.
        """
        args = [arg.replace(self._delimiter, self._other_del).replace(" ", "")
                if self._delimiter in arg else arg for arg in map(str, args)]
        super().info(self._delimiter.join(args))
