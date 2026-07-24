"""Independent naive interpreter for the grid transfer case.

This module imports only immutable data containers from
``grid_transfer_domain``.  It deliberately duplicates validation,
observation, policy, memory, transition, and replay semantics.  In particular,
it does not call the implementation's ``observe``, ``ingest``,
``choose_action``, ``advance_memory``, ``transition``, or either evaluator.
The duplication gives the audit a separately executable trace oracle while
keeping serialization types shared and unambiguous.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from experiments.grid_transfer_domain import (
    GridAction,
    GridAgentMemory,
    GridDisplayMemory,
    GridMechanism,
    GridObservation,
    GridPresentation,
    GridSimulation,
    GridTraceStep,
    GridWorldState,
)


ORACLE_ACTIONS: tuple[GridAction, ...] = (
    "forward",
    "turn_left",
    "turn_right",
    "wait",
)
ORACLE_AGENTS: tuple[str, ...] = (
    "x_first",
    "y_first",
    "history_tiebreak",
    "open_loop",
)
ORACLE_DELTAS: tuple[tuple[int, int], ...] = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)
ORACLE_STATUS_CODES = {"running": 0, "win": 1, "timeout": 2}


@dataclass(frozen=True, slots=True)
class OracleVerification:
    well_formed: bool
    reachable: bool
    states: int
    transitions: int
    shortest_win: int | None

    @property
    def valid(self) -> bool:
        return self.well_formed and self.reachable


def oracle_well_formed(spec: GridMechanism) -> bool:
    if spec.name == "" or spec.width < 3 or spec.height < 3:
        return False
    if spec.start_heading < 0 or spec.start_heading > 3:
        return False
    if spec.horizon <= 0:
        return False
    start_cell = (spec.start_x, spec.start_y)
    goal_cell = (spec.goal_x, spec.goal_y)
    if start_cell == goal_cell:
        return False
    if not (
        0 <= start_cell[0] < spec.width
        and 0 <= start_cell[1] < spec.height
        and 0 <= goal_cell[0] < spec.width
        and 0 <= goal_cell[1] < spec.height
    ):
        return False
    seen_obstacles: set[tuple[int, int]] = set()
    for obstacle_x, obstacle_y in spec.obstacles:
        obstacle = (obstacle_x, obstacle_y)
        if obstacle in seen_obstacles:
            return False
        seen_obstacles.add(obstacle)
        if not (
            0 <= obstacle_x < spec.width
            and 0 <= obstacle_y < spec.height
        ):
            return False
    return start_cell not in seen_obstacles and goal_cell not in seen_obstacles


def oracle_presentation_valid(presentation: GridPresentation) -> bool:
    valid_modes = ("exact", "coarse", "hidden")
    return (
        presentation.name != ""
        and presentation.x_mode in valid_modes
        and presentation.y_mode in valid_modes
        and (presentation.delay == 0 or presentation.delay == 1)
    )


def oracle_start(spec: GridMechanism) -> GridWorldState:
    return GridWorldState(
        x=spec.start_x,
        y=spec.start_y,
        heading=spec.start_heading,
        step=0,
    )


def oracle_status(state: GridWorldState, spec: GridMechanism) -> str:
    if state.x == spec.goal_x and state.y == spec.goal_y:
        return "win"
    if state.step >= spec.horizon:
        return "timeout"
    return "running"


def oracle_transition(
    state: GridWorldState,
    action: GridAction,
    spec: GridMechanism,
) -> GridWorldState:
    if oracle_status(state, spec) != "running":
        return state
    next_x = state.x
    next_y = state.y
    next_heading = state.heading
    if action == "turn_left":
        next_heading = (next_heading + 3) % 4
    elif action == "turn_right":
        next_heading = (next_heading + 1) % 4
    elif action == "forward":
        horizontal, vertical = ORACLE_DELTAS[next_heading]
        proposed_x = next_x + horizontal
        proposed_y = next_y + vertical
        proposed_cell = (proposed_x, proposed_y)
        inside = (
            proposed_x >= 0
            and proposed_x < spec.width
            and proposed_y >= 0
            and proposed_y < spec.height
        )
        if inside and proposed_cell not in spec.obstacles:
            next_x = proposed_x
            next_y = proposed_y
    elif action != "wait":
        raise ValueError(f"oracle received unknown action {action!r}")
    return GridWorldState(
        x=next_x,
        y=next_y,
        heading=next_heading,
        step=state.step + 1,
    )


def oracle_verify(spec: GridMechanism) -> OracleVerification:
    if not oracle_well_formed(spec):
        return OracleVerification(False, False, 0, 0, None)
    first = oracle_start(spec)
    pending: deque[GridWorldState] = deque()
    pending.append(first)
    discovered: set[GridWorldState] = {first}
    edge_count = 0
    earliest: int | None = None
    while pending:
        current = pending.popleft()
        current_status = oracle_status(current, spec)
        if current_status == "win":
            if earliest is None or current.step < earliest:
                earliest = current.step
            continue
        if current_status != "running":
            continue
        for candidate_action in ORACLE_ACTIONS:
            successor = oracle_transition(current, candidate_action, spec)
            edge_count += 1
            if successor not in discovered:
                discovered.add(successor)
                pending.append(successor)
    return OracleVerification(
        well_formed=True,
        reachable=earliest is not None,
        states=len(discovered),
        transitions=edge_count,
        shortest_win=earliest,
    )


def _oracle_coordinate(value: int, display_mode: str) -> int:
    if display_mode == "hidden":
        return -1
    if display_mode == "coarse":
        return value - value % 2
    if display_mode == "exact":
        return value
    raise ValueError(f"oracle received unknown coordinate mode {display_mode!r}")


def oracle_observe(
    state: GridWorldState,
    spec: GridMechanism,
    presentation: GridPresentation,
    display_memory: GridDisplayMemory,
) -> tuple[GridObservation, GridDisplayMemory]:
    if not oracle_presentation_valid(presentation):
        raise ValueError(f"oracle received invalid presentation {presentation!r}")
    current: GridObservation = (
        _oracle_coordinate(state.x, presentation.x_mode),
        _oracle_coordinate(state.y, presentation.y_mode),
        state.heading,
        ORACLE_STATUS_CODES[oracle_status(state, spec)],
    )
    if presentation.delay == 0:
        return current, display_memory
    if display_memory.previous is None:
        displayed: GridObservation = (-1, -1, current[2], current[3])
    else:
        displayed = display_memory.previous
    return displayed, GridDisplayMemory(previous=current)


def oracle_initial_memory(
    spec: GridMechanism,
    agent: str,
) -> GridAgentMemory:
    if agent not in ORACLE_AGENTS:
        raise ValueError(f"oracle received unknown agent {agent!r}")
    return GridAgentMemory(
        believed_x=spec.start_x,
        believed_y=spec.start_y,
        believed_heading=spec.start_heading,
        previous_action="forward",
        decision_count=0,
        history_code=0,
    )


def oracle_ingest(
    agent: str,
    memory: GridAgentMemory,
    observation: GridObservation,
) -> GridAgentMemory:
    if agent == "open_loop":
        return memory
    x_value, y_value, heading_value, status_value = observation
    next_belief_x = memory.believed_x
    next_belief_y = memory.believed_y
    if x_value >= 0:
        next_belief_x = x_value
    if y_value >= 0:
        next_belief_y = y_value
    next_history = (
        131 * memory.history_code
        + 17 * (x_value + 2)
        + 19 * (y_value + 2)
        + 23 * (heading_value + 1)
        + 29 * (status_value + 1)
    ) % 65521
    return GridAgentMemory(
        believed_x=next_belief_x,
        believed_y=next_belief_y,
        believed_heading=heading_value,
        previous_action=memory.previous_action,
        decision_count=memory.decision_count + 1,
        history_code=next_history,
    )


def _oracle_axis_heading(
    memory: GridAgentMemory,
    spec: GridMechanism,
    axis: str,
) -> int | None:
    if axis == "x":
        difference = spec.goal_x - memory.believed_x
        if difference > 0:
            return 1
        if difference < 0:
            return 3
        return None
    if axis == "y":
        difference = spec.goal_y - memory.believed_y
        if difference > 0:
            return 2
        if difference < 0:
            return 0
        return None
    raise ValueError(f"oracle received unknown axis {axis!r}")


def _oracle_steer(
    heading: int,
    target_heading: int | None,
    clockwise_tie: bool,
) -> GridAction:
    if target_heading is None:
        return "wait"
    turn_distance = (target_heading - heading) % 4
    if turn_distance == 0:
        return "forward"
    if turn_distance == 1:
        return "turn_right"
    if turn_distance == 3:
        return "turn_left"
    if clockwise_tie:
        return "turn_right"
    return "turn_left"


def oracle_policy(
    agent: str,
    memory: GridAgentMemory,
    spec: GridMechanism,
) -> GridAction:
    if agent == "open_loop":
        return "forward"
    if agent == "x_first":
        target = _oracle_axis_heading(memory, spec, "x")
        if target is None:
            target = _oracle_axis_heading(memory, spec, "y")
        return _oracle_steer(memory.believed_heading, target, False)
    if agent == "y_first":
        target = _oracle_axis_heading(memory, spec, "y")
        if target is None:
            target = _oracle_axis_heading(memory, spec, "x")
        return _oracle_steer(memory.believed_heading, target, True)
    if agent == "history_tiebreak":
        should_wait = (
            memory.previous_action == "forward"
            and memory.history_code % 11 == 0
        )
        if should_wait:
            return "wait"
        if memory.history_code % 2 == 0:
            first_axis, fallback_axis = "x", "y"
        else:
            first_axis, fallback_axis = "y", "x"
        target = _oracle_axis_heading(memory, spec, first_axis)
        if target is None:
            target = _oracle_axis_heading(memory, spec, fallback_axis)
        return _oracle_steer(
            memory.believed_heading,
            target,
            memory.history_code % 3 == 0,
        )
    raise ValueError(f"oracle received unknown agent {agent!r}")


def oracle_advance_memory(
    agent: str,
    memory: GridAgentMemory,
    action: GridAction,
    spec: GridMechanism,
) -> GridAgentMemory:
    if agent == "open_loop":
        return memory
    belief_x = memory.believed_x
    belief_y = memory.believed_y
    belief_heading = memory.believed_heading
    if action == "turn_left":
        belief_heading = (belief_heading + 3) % 4
    elif action == "turn_right":
        belief_heading = (belief_heading + 1) % 4
    elif action == "forward":
        horizontal, vertical = ORACLE_DELTAS[belief_heading]
        proposed = (belief_x + horizontal, belief_y + vertical)
        if (
            proposed[0] >= 0
            and proposed[0] < spec.width
            and proposed[1] >= 0
            and proposed[1] < spec.height
            and proposed not in spec.obstacles
        ):
            belief_x, belief_y = proposed
    return GridAgentMemory(
        believed_x=belief_x,
        believed_y=belief_y,
        believed_heading=belief_heading,
        previous_action=action,
        decision_count=memory.decision_count,
        history_code=memory.history_code,
    )


def oracle_simulate(
    spec: GridMechanism,
    presentation: GridPresentation,
    agent: str,
) -> GridSimulation:
    state = oracle_start(spec)
    memory = oracle_initial_memory(spec, agent)
    display_memory = GridDisplayMemory()
    records: list[GridTraceStep] = []
    while oracle_status(state, spec) == "running":
        observation, display_memory = oracle_observe(
            state,
            spec,
            presentation,
            display_memory,
        )
        perceived = oracle_ingest(agent, memory, observation)
        action = oracle_policy(agent, perceived, spec)
        next_state = oracle_transition(state, action, spec)
        next_memory = oracle_advance_memory(
            agent,
            perceived,
            action,
            spec,
        )
        next_status = oracle_status(next_state, spec)
        records.append(
            GridTraceStep(
                observation=observation,
                action=action,
                next_state=next_state,
                next_memory=next_memory,
                status=next_status,
            )
        )
        state = next_state
        memory = next_memory
    return GridSimulation(
        trace=tuple(records),
        termination=oracle_status(state, spec),
    )

