"""Immutable data types shared by the prototype implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

Action: TypeAlias = Literal["coast", "brake"]
DisplayMode: TypeAlias = Literal["exact", "coarse", "hidden"]
Observation: TypeAlias = tuple[int, int, int, int]
TraceStep: TypeAlias = tuple[Observation, Action, str]
Trace: TypeAlias = tuple[TraceStep, ...]


@dataclass(frozen=True, slots=True)
class KernelSpec:
    """Finite braking-task mechanism parameters."""

    name: str
    start_speed: int
    friction: int
    brake_force: int
    goal_start: int
    goal_end: int
    horizon: int
    start_position: int = 0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("kernel name must not be empty")
        if self.start_speed <= 0:
            raise ValueError("start_speed must be positive")
        if self.friction < 0:
            raise ValueError("friction must be non-negative")
        if self.brake_force <= 0:
            raise ValueError("brake_force must be positive")
        if self.goal_start < self.start_position:
            raise ValueError("goal must not start behind the vehicle")
        if self.goal_end < self.goal_start:
            raise ValueError("goal_end must be at least goal_start")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")


@dataclass(frozen=True, slots=True)
class PresentationSpec:
    """Read-only communication layer applied to a mechanism state."""

    name: str
    speed_mode: DisplayMode
    distance_mode: DisplayMode
    delay: int = 0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("presentation name must not be empty")
        if self.speed_mode not in {"exact", "coarse", "hidden"}:
            raise ValueError(f"unsupported speed mode: {self.speed_mode}")
        if self.distance_mode not in {"exact", "coarse", "hidden"}:
            raise ValueError(f"unsupported distance mode: {self.distance_mode}")
        if self.delay not in {0, 1}:
            raise ValueError("prototype supports only zero- or one-step delay")


@dataclass(frozen=True, slots=True)
class WorldState:
    position: int
    speed: int
    step: int
    used_brake: bool


@dataclass(frozen=True, slots=True)
class AgentMemory:
    believed_speed: int
    believed_distance: int
    previous_action: Action = "coast"


@dataclass(frozen=True, slots=True)
class DisplayMemory:
    previous: Observation | None = None


@dataclass(slots=True)
class WorkMetrics:
    graph_builds: int = 0
    graph_states: int = 0
    graph_transitions: int = 0
    observation_calls: int = 0
    policy_calls: int = 0
    transition_calls: int = 0
    prefix_groups: int = 0
    candidates: int = 0

    def add(self, other: "WorkMetrics") -> None:
        self.graph_builds += other.graph_builds
        self.graph_states += other.graph_states
        self.graph_transitions += other.graph_transitions
        self.observation_calls += other.observation_calls
        self.policy_calls += other.policy_calls
        self.transition_calls += other.transition_calls
        self.prefix_groups += other.prefix_groups
        self.candidates += other.candidates

    def as_dict(self) -> dict[str, int]:
        return {
            "graph_builds": self.graph_builds,
            "graph_states": self.graph_states,
            "graph_transitions": self.graph_transitions,
            "observation_calls": self.observation_calls,
            "policy_calls": self.policy_calls,
            "transition_calls": self.transition_calls,
            "prefix_groups": self.prefix_groups,
            "candidates": self.candidates,
        }

