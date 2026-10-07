"""Read-only verification of complete, isolated NL and NZ smoke products."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from sglib.core.infra.hashing import sha256_file, sha256_json
from sglib.core.infra.paths import case_study_root
from sglib.generator.delivery import exact_set
from sglib.generator.generation import input_identity_binding
from sglib.generator.weighter.candidates import load_candidate_registry
from sglib.generator.weighter.learned.common import kfold_splits
from sglib.generator.weighter.learned.training.preparation import load_prepared_task
from sglib.generator.weighter.learned.training.verify import verify_task_artifacts
from sglib.generator.weighter.learned.inference.preparation import load_inference_task
from sglib.generator.weighter.learned.inference.verify import verify_inputs, verify_inference_artifacts


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"smoke products verification failed: {message}; use a new empty output directory")


def _path(root: Path, relative: Any) -> Path:
    _require(isinstance(relative, str) and bool(relative), "artifact path must be nonempty")
    pure = PurePosixPath(relative)
    _require(not pure.is_absolute() and not PureWindowsPath(relative).drive
             and '\\' not in relative and pure.as_posix() == relative
             and '..' not in pure.parts and ':' not in relative,
             f"unsafe artifact path: {relative!r}")
    path = root.joinpath(*pure.parts).resolve()
    _require(path != root and path.is_relative_to(root), f"artifact path escapes root: {relative}")
    return path


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    _require(isinstance(value, dict), f"JSON object required: {path.name}")
    return value


def _same(document: Mapping, expected: Mapping, label: str) -> None:
    _require(isinstance(document, Mapping), f"{label} must be an object")
    for key, value in expected.items():
        observed = document.get(key)
        _require(observed is value if isinstance(value, bool) else observed == value,
                 f"{label}.{key} differs")


def _record(path: Path, record: Mapping, *, size: bool = True) -> None:
    _require(isinstance(record, Mapping) and path.is_file(), f"missing artifact or record: {path}")
    _require(record.get('sha256') == sha256_file(path), f"artifact hash differs: {path}")
    if size:
        _require(type(record.get('bytes')) is int and record['bytes'] == path.stat().st_size,
                 f"artifact size differs: {path}")


def _files(root: Path, receipt_path: Path) -> dict[str, Path]:
    files = {}
    for path in root.rglob('*'):
        _require(not path.is_symlink() and path.resolve().is_relative_to(root),
                 f"linked artifact escapes the isolated file inventory: {path}")
        if path.is_file() and path.resolve() != receipt_path:
            files[path.relative_to(root).as_posix()] = path
    return files


def artifact_manifest(root: Path | str, receipt_path: Path | str) -> dict[str, dict]:
    """Hash every product, excluding only the fixed final NL or NZ receipt."""
    root = Path(root).resolve()
    receipt = Path(receipt_path)
    receipt = receipt.resolve() if receipt.is_absolute() else _path(root, receipt.as_posix())
    _require(receipt in {root / 'pre_hpc_smoke.json', root / '2_Generator/5_NZ/smoke_receipt.json'},
             "only the fixed NL/NZ smoke receipt may be excluded")
    return {name: {'sha256': sha256_file(path), 'bytes': path.stat().st_size}
            for name, path in sorted(_files(root, receipt).items())}


def _inputs(repo: Path, generator: Path, document: dict, country: str) -> tuple[dict, dict]:
    receipt_path = generator / 'inputs/receipt.json'
    receipt, params = _json(receipt_path), _json(generator / 'inputs/worker_params.json')
    regions = document.get('regions' if country == 'nl' else 'selected_regions')
    _require(isinstance(regions, list) and bool(regions) and len(regions) == len(set(regions)),
             "smoke regions must be nonempty and unique")
    _same(receipt, {'schema_version': 'sg_generator_inputs_receipt_v3', 'country': country,
                    'regions': regions}, 'input receipt')
    _same(params, {'country': country, 'regions': regions}, 'worker parameters')
    _require(params.get('config_map', {}).get('baseline', {}).get('epochs') == 2,
             'worker parameters must declare two baseline epochs')
    binding = input_identity_binding(receipt_path)
    _same(binding, {'field': 'inputs_scientific_fingerprint',
                   'schema': 'sg_generator_inputs_scientific_identity_v3'}, 'input binding')
    identity = receipt['scientific_identity']
    _same(identity, {'country': country, 'regions': regions,
                     'worker_params_sha256': sha256_file(generator / 'inputs/worker_params.json'),
                     'config_fingerprint': receipt.get('config_fingerprint'),
                     'config_fingerprint_schema': receipt.get('config_fingerprint_schema')}, 'input scientific identity')
    bundle = receipt.get('bundle', {})
    _require(bundle.get('path') == 'bundle.pkl', 'input bundle must belong to this root')
    _record(generator / 'inputs/bundle.pkl', bundle)
    inventory_ref = receipt.get('dataoverview_inventory', {})
    base = inventory_ref.get('path_base')
    _require(base in {'generator_root', 'repo_root'}, 'fixture inventory locator base is invalid')
    inventory_path = _path(generator if base == 'generator_root' else repo, inventory_ref.get('path'))
    _require(inventory_path == generator / 'inputs/fixture/data_inventory.json',
             'fixture inventory must resolve into this generator root')
    inventory = _json(inventory_path)
    _same(inventory, {'schema_version': 'sg_dataoverview_inventory_v1', 'country': country, 'formal': False},
          'fixture inventory')
    records = inventory.get('artifacts')
    _require(isinstance(records, list) and bool(records), 'fixture artifact inventory is empty')
    names = [row.get('path') for row in records if isinstance(row, dict)]
    _require(len(names) == len(records) and len(names) == len(set(names)), 'fixture artifacts are malformed or duplicated')
    for row in records:
        _record(_path(generator, row['path']), row)
    expected_files = {path.relative_to(generator).as_posix() for path in (generator / 'inputs/fixture').rglob('*')
                      if path.is_file() and path != inventory_path}
    _require(set(names) == expected_files, 'fixture inventory does not cover every actual input')
    required = {'inputs/fixture/sources.parquet', 'inputs/fixture/stations.parquet'}
    for region in regions:
        required.update(f'inputs/fixture/{region}/{name}' for name in
                        ('grid.parquet', 'grid_metadata.json', 'landuse.npz', 'built_surface.npz', 'cuz_support.npz', 'ntl.npz'))
    _require(required <= set(names), 'fixture inventory omits required region inputs')
    fingerprint = sha256_json(records)
    _same(inventory, {'fingerprint': fingerprint}, 'fixture inventory')
    _same(inventory_ref, {'fingerprint': fingerprint, 'artifact_count': len(records)}, 'input inventory binding')
    _same(identity, {'dataoverview_country_fingerprint': fingerprint}, 'scientific fixture binding')
    return params, binding


def _products(repo: Path, generator: Path, params: dict, country: str) -> None:
    """Check required scientific coordinates independently of the outer manifest."""
    regions = set(params['regions'])

    def index(relative, field, expected_header):
        document = _json(generator / relative)
        _same(document, expected_header, relative)
        rows = document.get(field)
        _require(isinstance(rows, list) and bool(rows) and all(isinstance(row, dict) for row in rows),
                 f'{relative} has no valid artifact records')
        for row in rows:
            _record(_path(generator, row.get('path')), row)
        return rows

    for component in ('assignments', 'uniform', 'gpm', 'proximity', 'public_activity'):
        rows = index(f'static/{component}/index.json', 'regions',
                     {'schema_version': 'sg_generator_static_index_v1', 'country': country, 'component': component})
        exact_set([row.get('region') for row in rows], regions, component)
        for row in rows:
            _same(row, {'path': f'static/{component}/{row["region"]}.npz'}, component)

    families = ('Uni', 'GPM', 'Equal')
    registry = load_candidate_registry(case_study_root(repo) / '2_Generator/general/candidate_registry.json')
    definitions = {row['label']: row for row in registry['candidates']
                   if row['family'] in families and row['materialize']}
    candidates = {}
    for family in families:
        rows = index(f'candidates/index_{family}.json', 'entries',
                     {'schema_version': 'sg_candidate_family_index_v1', 'country': country, 'family': family})
        expected = {(label, region, None) for label, spec in definitions.items()
                    if spec['family'] == family for region in regions}
        exact_set([(row.get('label'), row.get('region'), row.get('seed')) for row in rows], expected, family)
        for row in rows:
            label, region = row['label'], row['region']
            _same(row, {'family': family, 'qa_only': definitions[label]['qa_only'],
                        'path': f'candidates/{label}/{region}.npz'}, 'static candidate')
            _require(row.get('fold') is None, 'static smoke candidate cannot declare a learned fold')
            candidates[label, region] = row
    for qa, filename, schema in ((False, 'candidate_index.json', 'sg_candidate_index_v1'),
                                 (True, 'candidate_qa_index.json', 'sg_candidate_qa_index_v1')):
        rows = index(f'candidates/{filename}', 'entries',
                     {'schema_version': schema, 'formal': False, 'profile': 'smoke', 'families': list(families)})
        expected = {key: row for key, row in candidates.items() if row['qa_only'] is qa}
        exact_set([(row.get('label'), row.get('region')) for row in rows], set(expected), filename)
        _require({(row['label'], row['region']): row for row in rows} == expected,
                 f'{filename} differs from verified family indexes')

    formal = {key: row for key, row in candidates.items() if not row['qa_only']}
    for kind, schema in (('civd', 'sg_civd_extension_index_v1'), ('idr_fixed', 'sg_idr_fixed_index_v1'),
                         ('idr_matched', 'sg_idr_matched_index_v1')):
        rows = index(f'{kind}/index.json', 'entries', {'schema_version': schema, 'country': country})
        if kind == 'idr_matched':
            exact_set([(row.get('candidate'), row.get('region'), row.get('seed')) for row in rows],
                      {(label, region, 0) for label, region in formal}, kind)
            for row in rows:
                candidate = formal[row['candidate'], row['region']]
                _same(row, {'candidate_path': candidate['path'], 'candidate_sha256': candidate['sha256'],
                            'path': f'idr_matched/{row["candidate"]}/seed_0/{row["region"]}.npz'}, kind)
        else:
            exact_set([row.get('region') for row in rows], regions, kind)
            for row in rows:
                _same(row, {'path': f'{kind}/{row["region"]}.npz'}, kind)


def _nonformal(identity: Mapping, binding: dict) -> None:
    _same(identity, {'formal': False}, 'execution identity')
    run = identity.get('run_identity', {})
    _same(run, {'schema_version': 'sg_generator_execution_generation_v3',
                'inputs_identity_schema': binding['schema'],
                'inputs_scientific_fingerprint': binding['fingerprint'],
                'landuse_supervision_representation': 'edge_flat_id_scatter_add_v1'}, 'run identity')
    payload = {key: value for key, value in run.items() if key not in {'run_fingerprint', 'formal_reuse_allowed'}}
    _same(run, {'run_fingerprint': sha256_json(payload)}, 'run identity')


def _training(root: Path, generator: Path, row: dict, country: str, family: str,
              params: dict, binding: dict):
    group = f'B-{country.upper()}-{family.upper()}'
    task_id = f'{group}-S42-F1'
    task_path = generator / f'training/tasks/{group}/{task_id}.json'
    task = load_prepared_task(task_path, repo_root=root, results_root=root)
    expected = {'country': country, 'family': family, 'group': group, 'seed': 42, 'fold': 1,
                'config': 'baseline', 'feature_set': 'lu5', 'signal': 'none',
                'parameter': 'fixed', 'value': 'default', 'execution_backend': 'local'}
    _same(vars(task), expected, 'prepared training task')
    output = generator / f'2_Weighter/base/{family}/seed_42/fold1'
    _require(Path(task.output_path) == output and Path(task.frozen_params_path) == generator / 'inputs/worker_params.json'
             and Path(task.graph_cache_path) == generator / 'training/graphs/lu5.pkl',
             'training task does not bind this root parameters, graph, and model')
    _same(task.input_fingerprints, {binding['field']: binding['fingerprint']}, 'training input fingerprints')
    _require(set(task.input_fingerprints) == {binding['field'], 'graph_cache'},
             'training input fingerprint roles differ from the smoke contract')
    _nonformal(task.execution_identity, binding)
    completion = verify_task_artifacts(task)
    _same(completion, {**expected, 'task_id': task_id, 'epochs_observed': 2}, 'training completion')
    _nonformal(completion['execution_identity'], binding)
    _same(completion.get('runtime', {}), {'device_resolved': 'cpu', 'slurm_job_id': None}, 'training runtime')
    _require(completion['region_order'] == kfold_splits(params['regions'], 42, params['n_folds'])[0][0],
             'training completion region order differs from the declared fold')
    key = 'completion' if country == 'nl' else 'completion_path'
    _same(row, {'task_id': task_id, 'epochs': 2, key: (output / 'task_completion.json').relative_to(root).as_posix(),
                'selected_training_loss': completion['selected_training_loss']}, 'top-level training record')
    if country == 'nl':
        _same(row, {'family': family, 'formal': False, 'training_verify': f'training/verify/{group}.json'}, 'NL training record')
        verified = _json(_path(root, row['training_verify']))
        _same(verified, {'status': 'PASS', 'formal': False, 'task_id': task_id,
                         'completion_sha256': sha256_file(output / 'task_completion.json')}, 'NL training verification')
    else:
        _same(row, {'device': 'cpu'}, 'NZ training record')
    return task, task_path


def _inference(root: Path, row: dict, training, training_path: Path, params: dict, binding: dict) -> None:
    task_id = f'infer-{training.task_id}'
    task = load_inference_task(root / f'inference/tasks/{training.group}/{task_id}.json',
                               repo_root=root, results_root=root)
    _same(vars(task), {name: getattr(training, name) for name in
                       ('country', 'family', 'group', 'seed', 'fold', 'config', 'feature_set', 'signal', 'parameter', 'value')},
          'prepared inference task')
    expected_paths = {'training_task_path': training_path,
                      'checkpoint_path': Path(training.output_path) / 'model.pth',
                      'frozen_params_path': Path(training.frozen_params_path),
                      'graph_cache_path': Path(training.graph_cache_path),
                      'output_path': root / f'inference/outputs/{training.group}/{task_id}'}
    _require(all(Path(getattr(task, name)) == path for name, path in expected_paths.items()),
             'inference task does not bind this root training and inputs')
    _require(list(task.regions) == params['regions'], 'inference regions differ from worker parameters')
    _nonformal(task.execution_identity, binding)
    verify_inputs(task)
    completion = verify_inference_artifacts(task)
    _same(completion, {'execution_backend': 'local', 'device': 'cpu', 'supervision_sanitized': True}, 'inference completion')
    _nonformal(completion.get('execution_identity', {}), binding)
    _require(completion['execution_identity']['run_identity'] == task.execution_identity['run_identity'],
             'inference completion generation differs from prepared task')
    _same(row, {'inference_task_id': task_id, 'inference_regions': len(params['regions']),
                'inference_completion': (expected_paths['output_path'] / 'inference_completion.json').relative_to(root).as_posix(),
                'inference_verify': f'inference/verify/{training.group}.json'}, 'top-level inference record')
    _same(_json(_path(root, row['inference_verify'])), {'status': 'PASS', 'formal': False, 'group': training.group},
          'inference verification view')


def verify_smoke_products(repo_root: Path | str, root: Path | str, document: dict, *, country: str) -> None:
    """Verify bytes and scientific receipts without loading models or pickle data.

    Legacy receipts without a complete product manifest cannot authorize reuse.
    The caller must execute a new smoke in a new empty directory instead.
    """
    try:
        root, repo = Path(root).resolve(), Path(repo_root).resolve()
        _require(country in {'nl', 'nz'} and root != repo, 'only isolated NL/NZ smoke roots are supported')
        receipt_relative = 'pre_hpc_smoke.json' if country == 'nl' else '2_Generator/5_NZ/smoke_receipt.json'
        schema = 'sg_nl_pre_hpc_smoke_v2' if country == 'nl' else 'sg_nz_pre_hpc_smoke_receipt_v1'
        _same(document, {'status': 'PASS', 'formal': False, 'country': country, 'schema_version': schema}, 'smoke receipt')
        manifest = document.get('artifacts')
        _require(isinstance(manifest, dict) and bool(manifest), 'legacy or empty receipt lacks a nonempty artifacts manifest')
        actual = _files(root, root / receipt_relative)
        _require(set(actual) == set(manifest), 'artifact inventory differs from actual files (missing or extra products)')
        for name, record in manifest.items():
            _record(_path(root, name), record)
        generator = root if country == 'nl' else root / '2_Generator/5_NZ'
        params, binding = _inputs(repo, generator, document, country)
        _products(repo, generator, params, country)
        if country == 'nl':
            rows = document.get('training')
            _require(isinstance(rows, list) and len(rows) == 2 and all(isinstance(row, dict) for row in rows)
                     and {row.get('family') for row in rows} == {'gnn', 'mlp'}, 'NL must include exactly GNN and MLP training')
        else:
            rows = [document.get('training')]
            _require(isinstance(rows[0], dict), 'NZ training record is missing')
        for row in rows:
            family = row['family'] if country == 'nl' else 'gnn'
            task, task_path = _training(root, generator, row, country, family, params, binding)
            if country == 'nl':
                _inference(root, row, task, task_path, params, binding)
        if country == 'nz':
            _same(document, {'hpc_submission_authorized': False}, 'NZ smoke receipt')
            for reference, relative, expected, declared in (
                (document.get('real_inputs', {}).get('nz_gate', {}), '1_DataOverview/5_NZ/audit/nz_gate.json', {'status': 'PASS', 'country': 'nz'}, {}),
                (document.get('generator_audit', {}), '2_Generator/5_NZ/audit.json', {'status': 'PASS', 'country': 'nz', 'profile': 'smoke', 'failures': []}, {'status': 'PASS'}),
                (document.get('size_gate', {}), '1_DataOverview/5_NZ/audit/nz_size_gate.json', {'status': 'PASS', 'country': 'nz', 'admission_decision': 'ADMITTED', 'authority_status': 'FROZEN', 'hard_gate_failures': []}, {'status': 'PASS', 'admission_decision': 'ADMITTED', 'authority_status': 'FROZEN'}),
            ):
                _same(reference, {'path': relative, **declared}, 'NZ gate locator')
                path = _path(root, relative)
                _record(path, reference, size=False)
                _same(_json(path), expected, 'NZ gate')
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        if isinstance(error, ValueError) and 'use a new empty output directory' in str(error):
            raise
        raise ValueError(f'smoke products verification failed: {error}; use a new empty output directory') from error
