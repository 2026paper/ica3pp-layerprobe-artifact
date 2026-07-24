"""Second finite-state transfer case for contract-guided exact reuse.

The domain is deliberately unlike the braking family used by the main
experiment.  A mechanism is a bounded two-dimensional grid with obstacles, a
start pose, a goal cell, and a finite horizon.  Agents can move forward, turn
in place, or wait.  Presentation variants are observation-only: they change
the precision and delay of displayed x/y coordinates but never modify the
world, action set, transition rule, goal, or horizon.

The LayerProbe implementation in this file keeps one cache per
mechanism--agent pair.  Its complete key is exactly

    (world state, pre-ingest agent memory, observation)

and the cached value contains the deterministic policy/transition output.  The
three projected keys are included only for the fault-injection audit.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from itertools import combinations, product
from typing import Any, Iterable, Literal, TypeAlias


GridAction: TypeAlias = Literal[
    "forward",
    "turn_left",
    "turn_right",
    "wait",
]
CoordinateMode: TypeAlias = Literal["exact", "coarse", "hidden"]
GridObservation: TypeAlias = tuple[int, int, int, int]
CacheVariant: TypeAlias = Literal[
    "full",
    "drop_state",
    "drop_memory",
    "drop_observation",
]

ACTIONS: tuple[GridAction, ...] = (
    "forward",
    "turn_left",
    "turn_right",
    "wait",
)
AGENT_NAMES: tuple[str, ...] = (
    "x_first",
    "y_first",
    "history_tiebreak",
    "open_loop",
)
AGENT_PAIRS: tuple[tuple[str, str], ...] = tuple(
    combinations(AGENT_NAMES, 2)
)
CACHE_VARIANTS: tuple[CacheVariant, ...] = (
    "full",
    "drop_state",
    "drop_memory",
    "drop_observation",
)
ORDERS: tuple[str, ...] = ("canonical", "reverse")
HEADING_DELTAS: tuple[tuple[int, int], ...] = (
    (0, -1),   # north
    (1, 0),    # east
    (0, 1),    # south
    (-1, 0),   # west
)
STATUS_CODES = {"running": 0, "win": 1, "timeout": 2}


@dataclass(frozen=True, slots=True)
class GridMechanism:
    name: str
    width: int
    height: int
    start_x: int
    start_y: int
    start_heading: int
    goal_x: int
    goal_y: int
    horizon: int
    obstacles: tuple[tuple[int, int], ...] = ()
    obstacle_pattern: str = "open"


@dataclass(frozen=True, slots=True)
class GridPresentation:
    name: str
    x_mode: CoordinateMode
    y_mode: CoordinateMode
    delay: int


@dataclass(frozen=True, slots=True)
class GridWorldState:
    x: int
    y: int
    heading: int
    step: int


@dataclass(frozen=True, slots=True)
class GridAgentMemory:
    believed_x: int
    believed_y: int
    believed_heading: int
    previous_action: GridAction
    decision_count: int
    history_code: int


@dataclass(frozen=True, slots=True)
class GridDisplayMemory:
    previous: GridObservation | None = None


@dataclass(frozen=True, slots=True)
class GridVerification:
    well_formed: bool
    reachable: bool
    states: int
    transitions: int
    shortest_win: int | None

    @property
    def valid(self) -> bool:
        return self.well_formed and self.reachable


@dataclass(frozen=True, slots=True)
class GridTraceStep:
    """One complete semantic step used by the exactness audit.

    ``next_memory`` is intentionally retained.  Agent memory is part of the
    deterministic output replaced by a cache hit, even though the behavioral
    signature below projects it away.
    """

    observation: GridObservation
    action: GridAction
    next_state: GridWorldState
    next_memory: GridAgentMemory
    status: str


GridTrace: TypeAlias = tuple[GridTraceStep, ...]


@dataclass(frozen=True, slots=True)
class GridSimulation:
    trace: GridTrace
    termination: str


@dataclass(frozen=True, slots=True)
class StepOutput:
    action: GridAction
    next_state: GridWorldState
    next_memory: GridAgentMemory
    status: str


@dataclass(slots=True)
class GridCounters:
    observation_calls: int = 0
    policy_calls: int = 0
    transition_calls: int = 0
    cache_lookups: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    peak_cache_entries: int = 0
    guarded_replays: int = 0

    def add(self, other: "GridCounters") -> None:
        self.observation_calls += other.observation_calls
        self.policy_calls += other.policy_calls
        self.transition_calls += other.transition_calls
        self.cache_lookups += other.cache_lookups
        self.cache_hits += other.cache_hits
        self.cache_misses += other.cache_misses
        self.peak_cache_entries = max(
            self.peak_cache_entries,
            other.peak_cache_entries,
        )
        self.guarded_replays += other.guarded_replays

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ViewBatch:
    simulations: dict[str, GridSimulation]
    counters: GridCounters


def mechanism_well_formed(spec: GridMechanism) -> bool:
    if not spec.name or spec.width < 3 or spec.height < 3:
        return False
    if spec.start_heading not in range(4) or spec.horizon <= 0:
        return False
    start = (spec.start_x, spec.start_y)
    goal = (spec.goal_x, spec.goal_y)
    if start == goal:
        return False
    for x, y in (start, goal):
        if not (0 <= x < spec.width and 0 <= y < spec.height):
            return False
    obstacle_set = set(spec.obstacles)
    if len(obstacle_set) != len(spec.obstacles):
        return False
    if start in obstacle_set or goal in obstacle_set:
        return False
    return all(
        0 <= x < spec.width and 0 <= y < spec.height
        for x, y in obstacle_set
    )


def presentation_well_formed(presentation: GridPresentation) -> bool:
    return (
        bool(presentation.name)
        and presentation.x_mode in {"exact", "coarse", "hidden"}
        and presentation.y_mode in {"exact", "coarse", "hidden"}
        and presentation.delay in {0, 1}
    )


def initial_state(spec: GridMechanism) -> GridWorldState:
    return GridWorldState(
        x=spec.start_x,
        y=spec.start_y,
        heading=spec.start_heading,
        step=0,
    )


def terminal_status(state: GridWorldState, spec: GridMechanism) -> str:
    if (state.x, state.y) == (spec.goal_x, spec.goal_y):
        return "win"
    if state.step >= spec.horizon:
        return "timeout"
    return "running"


def transition(
    state: GridWorldState,
    action: GridAction,
    spec: GridMechanism,
) -> GridWorldState:
    if terminal_status(state, spec) != "running":
        return state
    x = state.x
    y = state.y
    heading = state.heading
    if action == "turn_left":
        heading = (heading - 1) % 4
    elif action == "turn_right":
        heading = (heading + 1) % 4
    elif action == "forward":
        dx, dy = HEADING_DELTAS[heading]
        candidate = (x + dx, y + dy)
        if (
            0 <= candidate[0] < spec.width
            and 0 <= candidate[1] < spec.height
            and candidate not in spec.obstacles
        ):
            x, y = candidate
    elif action != "wait":
        raise ValueError(f"unknown action: {action}")
    return GridWorldState(x=x, y=y, heading=heading, step=state.step + 1)


def verify_mechanism(spec: GridMechanism) -> GridVerification:
    if not mechanism_well_formed(spec):
        return GridVerification(False, False, 0, 0, None)

    start = initial_state(spec)
    queue: deque[GridWorldState] = deque((start,))
    visited: set[GridWorldState] = {start}
    transitions = 0
    shortest_win: int | None = None
    while queue:
        state = queue.popleft()
        status = terminal_status(state, spec)
        if status == "win":
            if shortest_win is None or state.step < shortest_win:
                shortest_win = state.step
            continue
        if status != "running":
            continue
        for action in ACTIONS:
            next_state = transition(state, action, spec)
            transitions += 1
            if next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)
    return GridVerification(
        well_formed=True,
        reachable=shortest_win is not None,
        states=len(visited),
        transitions=transitions,
        shortest_win=shortest_win,
    )


def _encode_coordinate(value: int, mode: CoordinateMode) -> int:
    if mode == "hidden":
        return -1
    if mode == "coarse":
        return (value // 2) * 2
    return value


def _raw_observation(
    state: GridWorldState,
    spec: GridMechanism,
    presentation: GridPresentation,
) -> GridObservation:
    return (
        _encode_coordinate(state.x, presentation.x_mode),
        _encode_coordinate(state.y, presentation.y_mode),
        state.heading,
        STATUS_CODES[terminal_status(state, spec)],
    )


def observe(
    state: GridWorldState,
    spec: GridMechanism,
    presentation: GridPresentation,
    memory: GridDisplayMemory,
) -> tuple[GridObservation, GridDisplayMemory]:
    """Read the world and update presentation-local delay state only."""

    if not presentation_well_formed(presentation):
        raise ValueError(f"invalid presentation: {presentation}")
    current = _raw_observation(state, spec, presentation)
    if presentation.delay == 0:
        return current, memory
    if memory.previous is None:
        output = (-1, -1, current[2], current[3])
    else:
        output = memory.previous
    return output, GridDisplayMemory(previous=current)


def initial_agent_memory(
    spec: GridMechanism,
    agent: str,
) -> GridAgentMemory:
    if agent not in AGENT_NAMES:
        raise ValueError(f"unknown agent: {agent}")
    return GridAgentMemory(
        believed_x=spec.start_x,
        believed_y=spec.start_y,
        believed_heading=spec.start_heading,
        previous_action="forward",
        decision_count=0,
        history_code=0,
    )


def ingest(
    agent: str,
    memory: GridAgentMemory,
    observation: GridObservation,
) -> GridAgentMemory:
    """Update an agent's belief from one displayed observation."""

    if agent == "open_loop":
        return memory
    observed_x, observed_y, observed_heading, status_code = observation
    believed_x = memory.believed_x if observed_x < 0 else observed_x
    believed_y = memory.believed_y if observed_y < 0 else observed_y
    history_code = (
        memory.history_code * 131
        + (observed_x + 2) * 17
        + (observed_y + 2) * 19
        + (observed_heading + 1) * 23
        + (status_code + 1) * 29
    ) % 65521
    return GridAgentMemory(
        believed_x=believed_x,
        believed_y=believed_y,
        believed_heading=observed_heading,
        previous_action=memory.previous_action,
        decision_count=memory.decision_count + 1,
        history_code=history_code,
    )


def _desired_heading(
    memory: GridAgentMemory,
    spec: GridMechanism,
    axis: str,
) -> int | None:
    if axis == "x":
        if memory.believed_x < spec.goal_x:
            return 1
        if memory.believed_x > spec.goal_x:
            return 3
        return None
    if axis == "y":
        if memory.believed_y < spec.goal_y:
            return 2
        if memory.believed_y > spec.goal_y:
            return 0
        return None
    raise ValueError(f"unknown axis: {axis}")


def _turn_or_forward(
    current_heading: int,
    desired_heading: int | None,
    *,
    clockwise_tie: bool,
) -> GridAction:
    if desired_heading is None:
        return "wait"
    delta = (desired_heading - current_heading) % 4
    if delta == 0:
        return "forward"
    if delta == 1:
        return "turn_right"
    if delta == 3:
        return "turn_left"
    return "turn_right" if clockwise_tie else "turn_left"


def choose_action(
    agent: str,
    memory: GridAgentMemory,
    spec: GridMechanism,
) -> GridAction:
    if agent == "open_loop":
        return "forward"
    if agent == "x_first":
        desired = _desired_heading(memory, spec, "x")
        if desired is None:
            desired = _desired_heading(memory, spec, "y")
        return _turn_or_forward(
            memory.believed_heading,
            desired,
            clockwise_tie=False,
        )
    if agent == "y_first":
        desired = _desired_heading(memory, spec, "y")
        if desired is None:
            desired = _desired_heading(memory, spec, "x")
        return _turn_or_forward(
            memory.believed_heading,
            desired,
            clockwise_tie=True,
        )
    if agent == "history_tiebreak":
        if (
            memory.previous_action == "forward"
            and memory.history_code % 11 == 0
        ):
            return "wait"
        first_axis = "x" if memory.history_code % 2 == 0 else "y"
        second_axis = "y" if first_axis == "x" else "x"
        desired = _desired_heading(memory, spec, first_axis)
        if desired is None:
            desired = _desired_heading(memory, spec, second_axis)
        return _turn_or_forward(
            memory.believed_heading,
            desired,
            clockwise_tie=memory.history_code % 3 == 0,
        )
    raise ValueError(f"unknown agent: {agent}")


def advance_memory(
    agent: str,
    memory: GridAgentMemory,
    action: GridAction,
    spec: GridMechanism,
) -> GridAgentMemory:
    if agent == "open_loop":
        return memory
    x = memory.believed_x
    y = memory.believed_y
    heading = memory.believed_heading
    if action == "turn_left":
        heading = (heading - 1) % 4
    elif action == "turn_right":
        heading = (heading + 1) % 4
    elif action == "forward":
        dx, dy = HEADING_DELTAS[heading]
        candidate = (x + dx, y + dy)
        if (
            0 <= candidate[0] < spec.width
            and 0 <= candidate[1] < spec.height
            and candidate not in spec.obstacles
        ):
            x, y = candidate
    return GridAgentMemory(
        believed_x=x,
        believed_y=y,
        believed_heading=heading,
        previous_action=action,
        decision_count=memory.decision_count,
        history_code=memory.history_code,
    )


def _compute_step(
    state: GridWorldState,
    pre_ingest_memory: GridAgentMemory,
    observation: GridObservation,
    agent: str,
    spec: GridMechanism,
) -> StepOutput:
    perceived = ingest(agent, pre_ingest_memory, observation)
    action = choose_action(agent, perceived, spec)
    next_state = transition(state, action, spec)
    next_memory = advance_memory(agent, perceived, action, spec)
    return StepOutput(
        action=action,
        next_state=next_state,
        next_memory=next_memory,
        status=terminal_status(next_state, spec),
    )


def _projected_key(
    variant: CacheVariant,
    state: GridWorldState,
    memory: GridAgentMemory,
    observation: GridObservation,
) -> tuple[Any, ...]:
    if variant == "full":
        return state, memory, observation
    if variant == "drop_state":
        return memory, observation
    if variant == "drop_memory":
        return state, observation
    if variant == "drop_observation":
        return state, memory
    raise ValueError(f"unsupported cache variant: {variant}")


def simulate_flat(
    spec: GridMechanism,
    presentation: GridPresentation,
    agent: str,
) -> tuple[GridSimulation, GridCounters]:
    state = initial_state(spec)
    agent_memory = initial_agent_memory(spec, agent)
    display_memory = GridDisplayMemory()
    trace: list[GridTraceStep] = []
    counters = GridCounters()
    while terminal_status(state, spec) == "running":
        observation, display_memory = observe(
            state,
            spec,
            presentation,
            display_memory,
        )
        counters.observation_calls += 1
        output = _compute_step(
            state,
            agent_memory,
            observation,
            agent,
            spec,
        )
        counters.policy_calls += 1
        counters.transition_calls += 1
        trace.append(
            GridTraceStep(
                observation=observation,
                action=output.action,
                next_state=output.next_state,
                next_memory=output.next_memory,
                status=output.status,
            )
        )
        state = output.next_state
        agent_memory = output.next_memory
    return (
        GridSimulation(trace=tuple(trace), termination=terminal_status(state, spec)),
        counters,
    )


def simulate_layerprobe_views(
    spec: GridMechanism,
    presentations: Iterable[GridPresentation],
    agent: str,
    *,
    variant: CacheVariant = "full",
    order: str = "canonical",
) -> ViewBatch:
    """Replay views independently with one mechanism--agent scoped cache."""

    if variant not in CACHE_VARIANTS:
        raise ValueError(f"unsupported cache variant: {variant}")
    presentation_list = tuple(presentations)
    if order == "canonical":
        ordered_presentations = presentation_list
    elif order == "reverse":
        ordered_presentations = tuple(reversed(presentation_list))
    else:
        raise ValueError(f"unsupported replay order: {order}")

    cache: dict[tuple[Any, ...], StepOutput] = {}
    simulations: dict[str, GridSimulation] = {}
    counters = GridCounters()
    for presentation in ordered_presentations:
        state = initial_state(spec)
        agent_memory = initial_agent_memory(spec, agent)
        display_memory = GridDisplayMemory()
        trace: list[GridTraceStep] = []
        termination = terminal_status(state, spec)
        maximum_iterations = spec.horizon * 4 + 16
        for _ in range(maximum_iterations):
            if terminal_status(state, spec) != "running":
                termination = terminal_status(state, spec)
                break
            observation, display_memory = observe(
                state,
                spec,
                presentation,
                display_memory,
            )
            counters.observation_calls += 1
            counters.cache_lookups += 1
            key = _projected_key(variant, state, agent_memory, observation)
            output = cache.get(key)
            if output is None:
                output = _compute_step(
                    state,
                    agent_memory,
                    observation,
                    agent,
                    spec,
                )
                cache[key] = output
                counters.cache_misses += 1
                counters.policy_calls += 1
                counters.transition_calls += 1
                counters.peak_cache_entries = max(
                    counters.peak_cache_entries,
                    len(cache),
                )
            else:
                counters.cache_hits += 1
            trace.append(
                GridTraceStep(
                    observation=observation,
                    action=output.action,
                    next_state=output.next_state,
                    next_memory=output.next_memory,
                    status=output.status,
                )
            )
            if (
                output.status == "running"
                and output.next_state.step <= state.step
            ):
                termination = "cache_nonprogress"
                counters.guarded_replays += 1
                break
            state = output.next_state
            agent_memory = output.next_memory
            termination = output.status
        else:
            termination = "iteration_guard"
            counters.guarded_replays += 1
        simulations[presentation.name] = GridSimulation(
            trace=tuple(trace),
            termination=termination,
        )
    return ViewBatch(simulations=simulations, counters=counters)


def behavioral_projection(simulation: GridSimulation) -> tuple[Any, ...]:
    """Observable action/state projection used for pairwise signatures."""

    return (
        tuple(
            (
                step.observation,
                step.action,
                step.next_state,
                step.status,
            )
            for step in simulation.trace
        ),
        simulation.termination,
    )


def signature_for(
    simulations: dict[str, GridSimulation],
) -> int:
    mask = 0
    for index, (left, right) in enumerate(AGENT_PAIRS):
        if behavioral_projection(simulations[left]) != behavioral_projection(
            simulations[right]
        ):
            mask |= 1 << index
    return mask


def _obstacles_for(
    width: int,
    height: int,
    pattern_name: str,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    obstacles: set[tuple[int, int]] = set()
    if pattern_name == "open":
        pass
    elif pattern_name == "gate":
        wall_x = width // 2
        gap_y = height // 2
        obstacles.update(
            (wall_x, y)
            for y in range(height)
            if y != gap_y
        )
    elif pattern_name == "staggered":
        for x in range(1, width - 1):
            y = 1 if x % 2 else height - 2
            obstacles.add((x, y))
    else:
        raise ValueError(f"unknown obstacle pattern: {pattern_name}")
    obstacles.discard(start)
    obstacles.discard(goal)
    return tuple(sorted(obstacles))


def make_mechanisms(
    limit: int | None = None,
) -> tuple[GridMechanism, ...]:
    """Return the frozen 1,296-mechanism grid transfer family."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    sizes = (5, 6, 7)
    start_labels = ("north_west", "north_east", "south_west")
    goal_labels = ("south_east", "east_middle", "south_middle")
    patterns = ("open", "gate", "staggered")
    horizons = (12, 14, 16, 18)
    mechanisms: list[GridMechanism] = []
    for (
        size,
        start_label,
        goal_label,
        heading,
        pattern_name,
        horizon,
    ) in product(
        sizes,
        start_labels,
        goal_labels,
        range(4),
        patterns,
        horizons,
    ):
        width = height = size
        starts = {
            "north_west": (0, 0),
            "north_east": (width - 1, 0),
            "south_west": (0, height - 1),
        }
        goals = {
            "south_east": (width - 1, height - 1),
            "east_middle": (width - 1, height // 2),
            "south_middle": (width // 2, height - 1),
        }
        start = starts[start_label]
        goal = goals[goal_label]
        index = len(mechanisms)
        mechanisms.append(
            GridMechanism(
                name=(
                    f"grid_{index:04d}_n{size}_{start_label}_{goal_label}"
                    f"_h{heading}_{pattern_name}_t{horizon}"
                ),
                width=width,
                height=height,
                start_x=start[0],
                start_y=start[1],
                start_heading=heading,
                goal_x=goal[0],
                goal_y=goal[1],
                horizon=horizon,
                obstacles=_obstacles_for(
                    width,
                    height,
                    pattern_name,
                    start,
                    goal,
                ),
                obstacle_pattern=pattern_name,
            )
        )
    if len(mechanisms) != 1296:
        raise AssertionError(
            f"frozen transfer grid produced {len(mechanisms)} mechanisms"
        )
    selected = tuple(mechanisms if limit is None else mechanisms[:limit])
    return selected


def make_presentations() -> tuple[GridPresentation, ...]:
    presentations: list[GridPresentation] = []
    for index, (x_mode, y_mode, delay) in enumerate(
        product(
            ("exact", "coarse", "hidden"),
            ("exact", "coarse", "hidden"),
            (0, 1),
        )
    ):
        presentations.append(
            GridPresentation(
                name=f"grid_view_{index:02d}_{x_mode}_{y_mode}_d{delay}",
                x_mode=x_mode,
                y_mode=y_mode,
                delay=delay,
            )
        )
    if len(presentations) != 18:
        raise AssertionError("the frozen presentation grid must contain 18 views")
    return tuple(presentations)


def witness_mechanism() -> GridMechanism:
    """Small deterministic mechanism that exposes all three weak keys.

    It is used only by unit tests and as a diagnostic if a full audit fails.
    It is also a member of the declared structural family: a 6x6 open grid,
    starting at the north-east corner and targeting the south-middle cell.
    """

    return GridMechanism(
        name="grid_witness_n6_ne_sm_h3_open_t12",
        width=6,
        height=6,
        start_x=5,
        start_y=0,
        start_heading=3,
        goal_x=3,
        goal_y=5,
        horizon=12,
        obstacles=(),
        obstacle_pattern="open",
    )
