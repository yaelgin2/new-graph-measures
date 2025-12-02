class GraphMeasuresException(Exception):
    """
    Base exception for all user-facing errors in the GraphMeasures system.

    Attributes:
        message (str): Human-readable description of the error.
        code (int): Error code identifying the type of error.
    """

    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self):
        if self.code is not None:
            return f"[{self.code}] GraphMeasuresError: {self.message}"
        return f"GraphMeasuresError: {self.message}"