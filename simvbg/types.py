from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias


TraitValue: TypeAlias = str | int | float | bool
TraitInput: TypeAlias = "Trait | Mapping[str, Any] | tuple[str, TraitValue]"


@dataclass(frozen=True, slots=True)
class Trait:
    """A single trait, belief, preference, or biographical fact for an actor."""

    name: str
    value: TraitValue
    dimension: str = "profile"
    weight: float = 1.0
    evidence: str | None = None

    def render(self) -> str:
        text = f"{self.name}: {self.value}" if self.name else str(self.value)
        if self.evidence:
            text = f"{text} ({self.evidence})"
        return text


@dataclass(slots=True)
class TraitVector:
    """Ordered collection of traits that defines an actor."""

    traits: list[Trait] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, TraitValue],
        *,
        dimension: str = "profile",
        metadata: Mapping[str, Any] | None = None,
    ) -> "TraitVector":
        return cls(
            traits=[
                Trait(name=str(name), value=value, dimension=dimension)
                for name, value in values.items()
            ],
            metadata=dict(metadata or {}),
        )

    @classmethod
    def coerce(cls, value: "TraitVector | Mapping[str, TraitValue] | Iterable[TraitInput]") -> "TraitVector":
        if isinstance(value, TraitVector):
            return value
        if isinstance(value, Mapping):
            return cls.from_mapping(value)

        traits: list[Trait] = []
        for item in value:
            if isinstance(item, Trait):
                traits.append(item)
            elif isinstance(item, Mapping):
                traits.append(
                    Trait(
                        name=str(item["name"]),
                        value=item["value"],
                        dimension=str(item.get("dimension", "profile")),
                        weight=float(item.get("weight", 1.0)),
                        evidence=item.get("evidence"),
                    )
                )
            else:
                name, trait_value = item
                traits.append(Trait(name=str(name), value=trait_value))
        return cls(traits=traits)

    def render(self, *, dimensions: Sequence[str] | None = None, max_traits: int | None = None) -> str:
        selected = self.traits
        if dimensions is not None:
            allowed = set(dimensions)
            selected = [trait for trait in selected if trait.dimension in allowed]
        if max_traits is not None:
            selected = selected[:max_traits]
        return "\n".join(trait.render() for trait in selected)

    def by_dimension(self, dimension: str) -> list[Trait]:
        return [trait for trait in self.traits if trait.dimension == dimension]

    def __iter__(self):
        return iter(self.traits)

    def __len__(self) -> int:
        return len(self.traits)


@dataclass(frozen=True, slots=True)
class Scenario:
    """Question/scenario text plus optional answer options."""

    description: str
    choices: Mapping[str, str] | Sequence[str] | None = None

    def render(self) -> str:
        parts: list[str] = [self.description]
        if self.choices:
            parts.append(f"Options:\n{_render_choices(self.choices)}")
        return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class ActorResponse:
    """Structured response returned by Actor.turn."""

    content: str
    prompt: str
    raw: Any = None
    answer: str | int | None = None
    analysis: str = ""
    mode: str = "single"
    perspectives: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.content


def _render_choices(choices: Mapping[str, str] | Sequence[str]) -> str:
    if isinstance(choices, Mapping):
        return "\n".join(f"{key}: {value}" for key, value in choices.items())
    return "\n".join(f"{index}: {value}" for index, value in enumerate(choices, start=1))
