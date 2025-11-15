"""Model registry + adapter base classes for the refactored SURGE core."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Type, TypeVar


class AdapterNotFittedError(RuntimeError):
    """Raised when prediction is attempted before calling `fit`."""


class AdapterSerializationError(RuntimeError):
    """Raised when save/load fail or are not implemented."""


class BaseModelAdapter(ABC):
    """
    Unified interface for any backend plugged into the SURGE registry.

    Sub-classes can wrap scikit-learn, PyTorch, GPflow, or any custom estimator as
    long as they implement the methods below. The base implementation intentionally
    keeps functionality lightweight—the workflow layer will orchestrate scaling,
    splitting, and metrics so adapters simply focus on `fit`/`predict`.
    """

    name: str = "base"
    backend: str = "generic"
    supports_uq: bool = False
    supports_serialization: bool = False

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = dict(params)
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def fit(self, X, y) -> "BaseModelAdapter":
        """Train the underlying estimator."""

    @abstractmethod
    def predict(self, X):
        """Deterministic predictions."""

    def predict_with_uncertainty(self, X) -> Mapping[str, Any]:
        """
        Return predictive mean and variance if the backend supports UQ.

        Sub-classes should override this to provide consistent keys:
        `{'mean': ndarray, 'variance': ndarray}`.
        """
        raise NotImplementedError(f"{self.name} adapter does not expose uncertainty.")

    def save(self, path: Path) -> None:
        if not self.supports_serialization:
            raise AdapterSerializationError(
                f"{self.name} adapter does not implement save/load."
            )
        raise NotImplementedError

    def load(self, path: Path) -> "BaseModelAdapter":
        if not self.supports_serialization:
            raise AdapterSerializationError(
                f"{self.name} adapter does not implement save/load."
            )
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise AdapterNotFittedError(
                f"{self.name} adapter must be fitted before calling predict()."
            )

    def mark_fitted(self) -> None:
        self._is_fitted = True

    # Back-compat convenience
    def get_backend_name(self) -> str:
        return self.backend

    def get_model_name(self) -> str:
        return self.name


@dataclass(slots=True)
class RegistryEntry:
    """Metadata tracked for each registered adapter."""

    key: str
    adapter_cls: Type[BaseModelAdapter]
    backend: str
    name: str
    description: str = ""
    tags: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()
    default_params: Dict[str, Any] = field(default_factory=dict)

    def instantiate(self, **override_params: Any) -> BaseModelAdapter:
        params = dict(self.default_params)
        params.update(override_params)
        return self.adapter_cls(**params)


TRegistry = TypeVar("TRegistry", bound="ModelRegistry")


class ModelRegistry:
    """
    Lightweight registry responsible for discovering and instantiating adapters.

    Legacy SURGE exposed a similar concept in `surge.model.registry`. The new
    implementation stores richer metadata (backend, description, tags) so workflow
    components can surface capability information in status bars and reports.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, RegistryEntry] = {}
        self._aliases: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(
        self,
        adapter_cls: Type[BaseModelAdapter],
        *,
        key: Optional[str] = None,
        name: Optional[str] = None,
        backend: Optional[str] = None,
        description: str = "",
        tags: Iterable[str] | None = None,
        aliases: Iterable[str] | None = None,
        default_params: Optional[Mapping[str, Any]] = None,
    ) -> RegistryEntry:
        resolved_key = key or adapter_cls.__name__
        resolved_name = name or getattr(adapter_cls, "name", adapter_cls.__name__)
        resolved_backend = backend or getattr(adapter_cls, "backend", "generic")
        normalized_aliases = tuple(dict.fromkeys(list(aliases or ())))

        entry = RegistryEntry(
            key=resolved_key,
            adapter_cls=adapter_cls,
            backend=resolved_backend,
            name=resolved_name,
            description=description or getattr(adapter_cls, "__doc__", "") or "",
            tags=tuple(dict.fromkeys(list(tags or ()))),
            aliases=normalized_aliases,
            default_params=dict(default_params or {}),
        )

        self._entries[resolved_key] = entry
        for alias in normalized_aliases:
            self._aliases[alias] = resolved_key
        return entry

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------
    def _resolve_key(self, key: str) -> str:
        if key in self._entries:
            return key
        if key in self._aliases:
            return self._aliases[key]
        raise KeyError(f"Model '{key}' is not registered.")

    def get_entry(self, key: str) -> RegistryEntry:
        return self._entries[self._resolve_key(key)]

    def create(self, key: str, **params: Any) -> BaseModelAdapter:
        entry = self.get_entry(key)
        return entry.instantiate(**params)

    def list_entries(self) -> List[RegistryEntry]:
        return sorted(self._entries.values(), key=lambda entry: entry.key)

    def describe(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": entry.key,
                "backend": entry.backend,
                "name": entry.name,
                "tags": entry.tags,
                "aliases": entry.aliases,
                "description": entry.description.strip(),
            }
            for entry in self.list_entries()
        ]

    def __contains__(self, key: str) -> bool:
        try:
            self._resolve_key(key)
            return True
        except KeyError:
            return False


MODEL_REGISTRY = ModelRegistry()


def registry_summary(registry: ModelRegistry = MODEL_REGISTRY) -> str:
    """Return a human-readable summary of registered adapters."""
    lines = ["Registered Models:"]
    for entry in registry.list_entries():
        tag_str = f" [{' '.join(entry.tags)}]" if entry.tags else ""
        alias_str = f" (aliases: {', '.join(entry.aliases)})" if entry.aliases else ""
        lines.append(f"- {entry.key}: {entry.name} [{entry.backend}]{tag_str}{alias_str}")
    return "\n".join(lines)


