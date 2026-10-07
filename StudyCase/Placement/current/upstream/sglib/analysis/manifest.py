"""Analysis leaf hash gate: one leaf per registered unit."""
import json
from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.leaf_manifest import build, read_current, verify
from .config import CLAIMS


def enumerate_leaves(root, country):
    files = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != root / 'manifest.json'}
    leaves = {}
    for member in ('synthesis',) if country == 'cross' else (*CLAIMS, 'support'):
        leaves[member] = sorted(name for name in files if name.startswith(member + '/'))
    leaves['audit'] = ['audit.json'] if 'audit.json' in files else []
    if (root / 'setup').is_dir():
        leaves['setup'] = sorted(name for name in files if name.startswith('setup/'))
    return leaves, sorted(files - set().union(*(set(paths) for paths in leaves.values())))


def write_setup(ctx):
    """Record the run setup of a writable root once, before its manifest baseline is built."""
    from sglib.core.infra.setup_snapshot import dirty_allowed, verify_setup, write_setup as snapshot
    from .stage import code_checks, setup_config_files

    existing = verify_setup(ctx.root)
    if existing['status'] != 'ABSENT' or ctx.read_only:
        print(f"setup {existing['status']}: {ctx.root / 'setup'}", flush=True)
        return existing
    snapshot(ctx.root, repo_root=ctx.repo, stage='4_Analysis', country=ctx.country, profile=ctx.profile,
             config_files=setup_config_files(ctx), code_checks=code_checks(ctx),
             allow_dirty=dirty_allowed(ctx.repo, ctx.results, ctx.profile))
    report = verify_setup(ctx.root)
    print(f"setup WRITTEN ({report['status']}): {ctx.root / 'setup'}", flush=True)
    return report


def run_gate(ctx):
    from .stage import figures_root
    baseline = ctx.root / 'manifest.json'
    current, unclaimed = enumerate_leaves(ctx.root, ctx.country)
    view = figures_root(ctx)
    if baseline.is_file():
        report = verify(json.loads(baseline.read_text(encoding='utf-8')), current, ctx.root)
    else:
        if not (ctx.root / 'audit.json').is_file():
            raise RuntimeError('place audit.json before generating the manifest baseline')
        if unclaimed:
            raise RuntimeError(f'unclaimed Analysis files: {unclaimed}')
        generated = build('4_Analysis', ctx.country, read_current(current, ctx.root))
        atomic_json(generated, view / 'manifest.json' if ctx.read_only else baseline)
        report = {'status': 'GENERATED', 'root_fingerprint': generated['root_fingerprint'],
                  'leaves': {key: {'status': 'GENERATED'} for key in generated['leaves']}}
    report['unclaimed'] = unclaimed
    if unclaimed:
        report['status'] = 'FAIL'
    atomic_json(report, view / 'gate.json')
    for key, leaf in report['leaves'].items():
        print(f"{leaf['status']} {key}")
    print(f"{report['status']} gate; unclaimed={len(unclaimed)}")
    return report
