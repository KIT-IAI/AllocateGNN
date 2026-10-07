"""Common numbered-chain scheduling without stage imports or numerical policy."""
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class StageOrderError(RuntimeError):
    pass


@dataclass(frozen=True)
class StepReport:
    selected: tuple
    ran: tuple
    skipped: tuple
    prepared: tuple


@dataclass(frozen=True)
class RunHooks:
    status: Callable
    execute: Callable
    skip_reason: Callable
    selected: Callable = lambda unit: True
    partial: Callable = lambda unit: False
    read_only_audit: Callable = lambda unit: False


ARCHIVE_DIRECTORY = '_archive'


def reject_archived_root(path, repo_root=None):
    """Refuse any results root inside the results archive: archived batches are records, never inputs."""

    resolved = Path(path).resolve()
    if repo_root is not None:
        try:
            resolved = resolved.relative_to(Path(repo_root).resolve())
        except ValueError:
            pass
    if ARCHIVE_DIRECTORY in resolved.parts:
        raise ValueError(f'results roots under results/{ARCHIVE_DIRECTORY} are archived records and cannot be read: {path}')
    return Path(path)


def resolve_results_root(repo_root, profile, results_root=None):
    if profile not in {'smoke', 'preflight', 'formal'}:
        raise ValueError(f'unknown profile: {profile}')
    path = Path(results_root) if results_root is not None else Path('results') / {
        'smoke': '_smoke', 'preflight': '_preflight', 'formal': ''}[profile]
    resolved = (Path(repo_root).resolve() / path).resolve()
    reject_archived_root(resolved, repo_root)
    return resolved


def select_units(units, members, *, label='chain', sort=True):
    wanted = set(members)
    selected = [unit.id for unit in units.values() if unit.id in wanted or unit.step in wanted or unit.member in wanted]
    if not selected:
        raise ValueError(f'no {label} units match {sorted(wanted) if sort else members}')
    return sorted(selected) if sort else selected


def topological_order(units, selected=None, *, method='depth', error=ValueError, label='chain'):
    """Preserve each registry's traversal: strict breadth-first or external-aware DFS."""
    if method == 'breadth':
        wanted = set(selected or units)
        stack = list(wanted)
        while stack:
            current = stack.pop()
            missing = set(units[current].depends_on) - set(units)
            if missing:
                raise error(f'{current}: missing dependencies {sorted(missing)}')
            for dependency in units[current].depends_on:
                if dependency not in wanted:
                    wanted.add(dependency)
                    stack.append(dependency)
        indegree = {key: 0 for key in wanted}
        downstream = defaultdict(list)
        for key in wanted:
            for dependency in units[key].depends_on:
                if dependency in wanted:
                    indegree[key] += 1
                    downstream[dependency].append(key)
        ready = deque(sorted(key for key, value in indegree.items() if value == 0))
        ordered = []
        while ready:
            key = ready.popleft()
            ordered.append(units[key])
            for child in sorted(downstream[key]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(ordered) != len(wanted):
            raise error(f'{label} DAG contains a cycle')
        return ordered
    if method != 'depth':
        raise ValueError(f'unknown dependency traversal: {method}')
    ordered, visiting, visited = [], set(), set()

    def visit(key):
        if key in visited or key not in units:
            return
        if key in visiting:
            raise error(f'{label} dependency cycle')
        visiting.add(key)
        for dependency in units[key].depends_on:
            visit(dependency)
        visiting.remove(key)
        visited.add(key)
        ordered.append(units[key])

    for key in selected or units:
        if key not in units:
            raise error(f'unknown {label} unit: {key}')
        visit(key)
    return ordered


def run_units(ctx, selected, hooks, *, ordered=None, expand=False, refresh=False,
              precheck=True, dependency_first=False, sort_selected=True, snapshot_status=False):
    """Run a supplied registry order with six stage-specific operations.

    Scalar options preserve existing scheduling semantics: preflight all local
    prerequisites or check while traversing; inspect dependencies before DONE
    for injected upstream chains; and retain Experiment's initial status view.
    The caller supplies its already resolved registry order, not another hook.
    """
    selected = tuple(selected)
    chosen = set(selected)
    ordered = list(ordered) if ordered is not None else topological_order(ctx.units, selected)
    if expand:
        chosen = {unit.id for unit in ordered}
    initial = {unit.id: hooks.status(unit) for unit in ordered} if snapshot_status else {}

    def state(unit):
        return initial[unit.id] if snapshot_status else hooks.status(unit)

    def require_dependencies(unit):
        missing = {}
        for key in unit.depends_on:
            dependency = ctx.units.get(key, key)
            if key in ctx.units and not hooks.selected(dependency):
                continue
            value = hooks.status(dependency)
            if value != 'DONE':
                missing[key] = value
        if missing:
            raise StageOrderError(f'{unit.id}: predecessors are not DONE: {missing}')

    if precheck:
        missing = [f'{unit.id} ({state(unit)})' for unit in ordered
                   if unit.id not in chosen and state(unit) != 'DONE' and hooks.selected(unit)]
        if missing:
            raise StageOrderError('run earlier numbered steps first: ' + ', '.join(missing))
    if refresh and ctx.read_only:
        raise StageOrderError('a closed results root is read-only; select a fresh results root')
    if hasattr(ctx, 'backend'):
        print(f'backend={ctx.backend} results_root={ctx.results} profile={ctx.profile} read_only={ctx.read_only}', flush=True)
    else:
        print(f'profile={ctx.profile} results_root={ctx.results} read_only={ctx.read_only}', flush=True)
    ran, skipped, prepared = [], [], []
    for unit in ordered:
        current = state(unit) if not precheck or unit.id in chosen and hooks.selected(unit) else None
        if unit.id not in chosen:
            if not precheck and current != 'DONE':
                raise StageOrderError(f'run earlier numbered steps first: {unit.id} ({current})')
            continue
        if not hooks.selected(unit):
            continue
        if current == 'INVALID':
            raise StageOrderError(f'INVALID {unit.id}: receipt/marker/node/registration contract differs')
        if dependency_first:
            require_dependencies(unit)
        if current == 'DONE' and not (refresh and unit.id in selected):
            reason = hooks.skip_reason(unit)
            print(f'SKIP {unit.id}: {reason}', flush=True)
            skipped.append(unit.id)
            continue
        if ctx.read_only:
            if hooks.read_only_audit(unit):
                prepared.append(unit.id)
                continue
            raise StageOrderError(f'closed/formal root has unfinished unit: {unit.id} ({current})')
        if not dependency_first:
            require_dependencies(unit)
        print(f'RUN {unit.id}', flush=True)
        if not hooks.execute(unit):
            print(f'PREPARED {unit.id}: execution receipts are still required', flush=True)
            prepared.append(unit.id)
            continue
        final = hooks.status(unit)
        if final != 'DONE':
            if hooks.partial(unit):
                prepared.append(unit.id)
                continue
            raise StageOrderError(f'{unit.id}: output contract is not DONE ({final})')
        ran.append(unit.id)
        print(f'PASS {unit.id}', flush=True)
    reported = tuple(sorted(chosen)) if sort_selected else selected
    return StepReport(reported, tuple(ran), tuple(skipped), tuple(prepared))
