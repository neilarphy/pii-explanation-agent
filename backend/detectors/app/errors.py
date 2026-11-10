"""Safe errors: exception messages never contain input or library diagnostics."""

from typing import Literal

Stage = Literal["validation", "admission", "model", "regex", "merge", "startup"]


class ServiceError(Exception):
    def __init__(self, code: str, status: int, stage: Stage, completed=()):
        super().__init__(code)
        self.code = code
        self.status = status
        self.stage = stage
        self.completed = list(completed)


class AdapterContractError(ServiceError):
    def __init__(self):
        super().__init__("ADAPTER_CONTRACT_ERROR", 500, "model")


class TraceLimitError(ServiceError):
    def __init__(self, stage: Stage):
        super().__init__("TRACE_LIMIT_EXCEEDED", 422, stage)


class ConfigurationError(Exception):
    """Only static configuration identifiers belong in the message."""
