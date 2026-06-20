from __future__ import annotations

from typing import Any, Dict, Optional


class MlxvmError(Exception):
    """Expected operational error with a stable code and exit status."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "mlxvm_error",
        exit_code: int = 1,
        hint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.exit_code = exit_code
        self.hint = hint
        self.details = details or {}


class ConfigurationError(MlxvmError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, code="configuration_error", exit_code=2, **kwargs)


class ModelNotFoundError(MlxvmError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, code="model_not_found", exit_code=3, **kwargs)


class DependencyError(MlxvmError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, code="dependency_error", exit_code=4, **kwargs)


class NetworkError(MlxvmError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, code="network_error", exit_code=5, **kwargs)


class LockTimeoutError(MlxvmError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, code="lock_timeout", exit_code=6, **kwargs)


class SafetyError(MlxvmError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, code="safety_error", exit_code=7, **kwargs)


class RuntimeFailure(MlxvmError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, code="runtime_failure", exit_code=8, **kwargs)
