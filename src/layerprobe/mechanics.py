"""Reference braking mechanics, presentations, and behavioral agents."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .model import (
    Action,
    AgentMemory,
    DisplayMemory,
    KernelSpec,
    Observation,
    PresentationSpec,
    Trace,
    WorldState,
)

ACTIONS: tuple[Action, ...] = ("coast", "brake")
AGENT_NAMES: tuple[str, ...] = (
    "reference",
    "instant_stop",
    "speed_only",
    "friction_blind",
)


@dataclass(frozen=True, slots=True)
class KernelVerification:
    faithful: bool
    solvable: bool
    principle_required: bool
    states: int
    transitions: int
    witness: tuple[Action, ...] | None

    @property
    def valid(self) -> bool:
        return self.faithful and self.solvable and self.principle_required


def initial_state(spec: KernelSpec) -> WorldState:
    return WorldState(
        position=spec.start_position,
        speed=spec.start_speed,
        step=0,
        used_brake=False,
    )


def terminal_status(state: WorldState, spec: KernelSpec) -> str:
    if state.speed == 0:
        if spec.goal_start <= state.position <= spec.goal_end:
            return "win"
        return "stopped"
    if state.position > spec.goal_end:
        return "overshoot"
    if state.step >= spec.horizon:
        return "timeout"
    return "running"


def transition(state: WorldState, action: Action, spec: KernelSpec) -> WorldState:
    """One deterministic reference transition.

    The vehicle first loses speed through friction and optional braking, then
    advances by the remaining speed for this logical step.
    """

    if terminal_status(state, spec) != "running":
        return state
    deceleration = spec.friction + (spec.brake_force if action == "brake" else 0)
    new_speed = max(0, state.speed - deceleration)
    return WorldState(
        position=state.position + new_speed,
        speed=new_speed,
        step=state.step + 1,
        used_brake=state.used_brake or action == "brake",
    )


def verify_kernel(spec: KernelSpec) -> KernelVerification:
    """Exhaustively verify bounded solvability and brake-use necessity."""

    try:
        spec.validate()
    except ValueError:
        return KernelVerification(False, False, False, 0, 0, None)

    start = initial_state(spec)
    queue: deque[tuple[WorldState, tuple[Action, ...]]] = deque([(start, ())])
    visited: set[WorldState] = {start}
    transitions = 0
    witness: tuple[Action, ...] | None = None
    winning_without_brake = False

    while queue:
        state, path = queue.popleft()
        status = terminal_status(state, spec)
        if status == "win":
            if witness is None:
                witness = path
            if not state.used_brake:
                winning_without_brake = True
            continue
        if status != "running":
            continue
        for action in ACTIONS:
            next_state = transition(state, action, spec)
            transitions += 1
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + (action,)))

    solvable = witness is not None
    return KernelVerification(
        faithful=True,
        solvable=solvable,
        principle_required=solvable and not winning_without_brake,
        states=len(visited),
        transitions=transitions,
        witness=witness,
    )


def _encode(value: int, mode: str, width: int) -> int:
    if mode == "hidden":
        return -1
    if mode == "coarse":
        return (value // width) * width
    return value


def _raw_observation(state: WorldState, spec: KernelSpec, presentation: PresentationSpec) -> Observation:
    status_codes = {"running": 0, "win": 1, "stopped": 2, "overshoot": 3, "timeout": 4}
    # Report remaining distance to the goal region, not signed displacement
    # from its leading edge. Clamping at zero keeps every visible distance
    # non-negative, so -1 remains an unambiguous hidden/missing sentinel.
    distance = max(0, spec.goal_start - state.position)
    return (
        _encode(state.speed, presentation.speed_mode, 2),
        _encode(distance, presentation.distance_mode, 3),
        int(spec.goal_start <= state.position <= spec.goal_end),
        status_codes[terminal_status(state, spec)],
    )


def observe(
    state: WorldState,
    spec: KernelSpec,
    presentation: PresentationSpec,
    memory: DisplayMemory,
) -> tuple[Observation, DisplayMemory]:
    """Read a state without mutating it and update presentation-only memory."""

    presentation.validate()
    current = _raw_observation(state, spec, presentation)
    if presentation.delay == 0:
        return current, memory
    output = memory.previous
    if output is None:
        output = (-1, -1, current[2], current[3])
    return output, DisplayMemory(previous=current)


def initial_agent_memory(spec: KernelSpec) -> AgentMemory:
    return AgentMemory(
        believed_speed=spec.start_speed,
        believed_distance=spec.goal_start - spec.start_position,
    )


def ingest(memory: AgentMemory, observation: Observation) -> AgentMemory:
    speed, distance, _, _ = observation
    return AgentMemory(
        believed_speed=memory.believed_speed if speed < 0 else speed,
        believed_distance=memory.believed_distance if distance < 0 else distance,
        previous_action=memory.previous_action,
    )


def stopping_distance(speed: int, deceleration: int) -> int:
    if deceleration <= 0:
        return 10**9
    total = 0
    current = speed
    while current > 0:
        current = max(0, current - deceleration)
        total += current
    return total


def choose_action(agent: str, memory: AgentMemory, spec: KernelSpec) -> Action:
    speed = max(0, memory.believed_speed)
    distance = memory.believed_distance
    if agent == "reference":
        deceleration = spec.friction + spec.brake_force
        return "brake" if stopping_distance(speed, deceleration) >= distance else "coast"
    if agent == "instant_stop":
        return "brake" if distance <= max(1, speed) else "coast"
    if agent == "speed_only":
        return "brake" if speed >= 3 else "coast"
    if agent == "friction_blind":
        return "brake" if stopping_distance(speed, spec.brake_force) >= distance else "coast"
    raise ValueError(f"unknown agent: {agent}")


def advance_belief(agent: str, memory: AgentMemory, action: Action, spec: KernelSpec) -> AgentMemory:
    speed = max(0, memory.believed_speed)
    if agent == "instant_stop":
        new_speed = 0
    elif agent == "friction_blind":
        deceleration = spec.brake_force if action == "brake" else 0
        new_speed = max(0, speed - deceleration)
    else:
        deceleration = spec.friction + (spec.brake_force if action == "brake" else 0)
        new_speed = max(0, speed - deceleration)
    return AgentMemory(
        believed_speed=new_speed,
        believed_distance=memory.believed_distance - new_speed,
        previous_action=action,
    )


def simulate_flat(
    spec: KernelSpec,
    presentation: PresentationSpec,
    agent: str,
) -> Trace:
    state = initial_state(spec)
    display_memory = DisplayMemory()
    agent_memory = initial_agent_memory(spec)
    trace: list[tuple[Observation, Action, str]] = []

    while terminal_status(state, spec) == "running":
        observation, display_memory = observe(state, spec, presentation, display_memory)
        perceived = ingest(agent_memory, observation)
        action = choose_action(agent, perceived, spec)
        next_state = transition(state, action, spec)
        agent_memory = advance_belief(agent, perceived, action, spec)
        trace.append((observation, action, terminal_status(next_state, spec)))
        state = next_state
    return tuple(trace)
