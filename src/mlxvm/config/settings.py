from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from mlxvm.config.project import ConfigError, _toml_load

GENERATION_KEYS = {
    "temperature",
    "top_p",
    "min_p",
    "top_k",
    "max_tokens",
    "max_kv_size",
    "system_prompt",
    "seed",
}


@dataclass(frozen=True)
class Settings:
    offline: bool = False
    trust_remote_code: bool = False
    generation: Dict[str, Any] = field(default_factory=dict)
    profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def generation_for(
        self, profile: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        result = dict(self.generation)
        if profile:
            if profile not in self.profiles:
                raise ConfigError(f"generation profile '{profile}' does not exist")
            result.update(self.profiles[profile])
        if overrides:
            result.update(_validate_generation(overrides, "project generation"))
        return result


def _validate_generation(value: Any, location: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{location} must be a TOML table")
    unknown = set(value) - GENERATION_KEYS
    if unknown:
        raise ConfigError(f"{location} contains unsupported keys: {', '.join(sorted(unknown))}")
    numeric = {
        "temperature": (int, float),
        "top_p": (int, float),
        "min_p": (int, float),
        "top_k": (int,),
        "max_tokens": (int,),
        "max_kv_size": (int,),
        "seed": (int,),
    }
    for key, types in numeric.items():
        if key in value and (isinstance(value[key], bool) or not isinstance(value[key], types)):
            raise ConfigError(f"{location}.{key} has an invalid type")
    if "system_prompt" in value and not isinstance(value["system_prompt"], str):
        raise ConfigError(f"{location}.system_prompt must be a string")
    if "temperature" in value and value["temperature"] < 0:
        raise ConfigError(f"{location}.temperature must be at least 0")
    for key in ("top_p", "min_p"):
        if key in value and not 0 <= value[key] <= 1:
            raise ConfigError(f"{location}.{key} must be between 0 and 1")
    if "top_k" in value and value["top_k"] < 0:
        raise ConfigError(f"{location}.top_k must be at least 0")
    for key in ("max_tokens", "max_kv_size"):
        if key in value and value[key] <= 0:
            raise ConfigError(f"{location}.{key} must be greater than 0")
    return dict(value)


def validate_generation(value: Dict[str, Any], location: str = "generation") -> Dict[str, Any]:
    return _validate_generation(value, location)


def load_settings(path: Path) -> Settings:
    if not path.exists():
        return Settings()
    data = _toml_load(path)
    offline = data.get("offline", False)
    trust = data.get("trust_remote_code", False)
    if not isinstance(offline, bool) or not isinstance(trust, bool):
        raise ConfigError(f"{path}: offline and trust_remote_code must be booleans")
    generation = _validate_generation(data.get("generation", {}), "generation")
    raw_profiles = data.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        raise ConfigError(f"{path}: profiles must be a TOML table")
    profiles = {
        name: _validate_generation(profile, f"profiles.{name}")
        for name, profile in raw_profiles.items()
    }
    return Settings(offline, trust, generation, profiles)
