"""Mechanical cross-country table assembly and registered stable-ID outputs."""
import pandas as pd


def requested(member, countries):
    core = {'C1': ('contrasts', 'region_metrics', 'equal_route_regions'),
            'C2': ('contrasts',), 'C3': ('contrasts',), 'C4': ('contrasts',), 'C5': ('contrasts',)}
    support = {
        'C2': ('C2_mechanism_summary', 'C2_sweep_curves', 'C2_protocol_flips'),
        'C3': ('C3_gate_summary', 'C3_T1_summary', 'C3_CIVD_summary', 'C3_structural_duplicates'),
        'C4': ('C4_secondary_effects', 'C4_scenario_summary', 'C4_matching_summary', 'C4_defense_status'),
        'C5': ('C5_scale_curves', 'C5_support_summary', 'C5_sensitivity'),
        'C6': ('C6_tolerance_curves', 'C6_bounds_audit', 'C6_saturation', 'C6_calibration', 'C6_support_status'),
    }
    keys = {}
    for country in countries:
        if member in core:
            keys[f'{country}:{member}'] = core[member]
        if member in support:
            keys[f'{country}:support'] = support[member]
    if member == 'C4':
        keys['cross:synthesis'] = ('connection_map',)
    if member == 'SYN':
        keys['cross:synthesis'] = ('claim_evidence_status', 'engineering_choices', 'inheritance_revision',
            'claim_experiment_figure_coverage', 'T_INV_01', 'claim_coordinate_audit',
            'inference_resolution_audit', 'coverage_corrections', 'target_count_audit', 'limitations_registry')
    return keys


def unpack(tables, countries):
    data = {cc: {'core': {}, 'support': {}} for cc in countries}
    synthesis = {}
    for key, part in tables.items():
        country, member = key.split(':')
        if country == 'cross':
            synthesis = part
        elif member == 'support':
            data[country]['support'] = part
        else:
            data[country]['core'][member] = part
    return data, synthesis


def combine(country_data, section, name):
    rows = []
    for country, data in country_data.items():
        frame = data[section][name].copy()
        frame['country'] = country
        rows.append(frame)
    return pd.concat(rows, ignore_index=True, sort=False)


def mixed(frames):
    result = []
    for section, frame in frames:
        if frame is None or not len(frame):
            continue
        table = frame.copy()
        table.insert(0, 'section', section)
        result.append(table)
    return pd.concat(result, ignore_index=True, sort=False) if result else pd.DataFrame(columns=['section'])


def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.17g', lineterminator='\n')


def assemble(member, data, registration, target):
    countries, spec = registration['shared']['countries'], registration['spec']
    country_data, synthesis = unpack(data, countries)

    def core():
        return combine({cc: {'core': country_data[cc]['core'][member]} for cc in countries}, 'core', 'contrasts')

    def support(name):
        return combine(country_data, 'support', name)

    if member == 'C1':
        frames = [core()]
    elif member == 'C2':
        frames = [core(), support('C2_mechanism_summary'), support('C2_protocol_flips')]
    elif member == 'C3':
        frames = [core(), mixed(zip(spec['sections'], [support(name) for name in
            ('C3_T1_summary', 'C3_CIVD_summary', 'C3_structural_duplicates')], strict=True))]
    elif member == 'C4':
        frames = [core(), support('C4_scenario_summary'), mixed(zip(spec['sections'], [support(name) for name in
            ('C4_secondary_effects', 'C4_matching_summary', 'C4_defense_status')], strict=True))]
    elif member == 'C5':
        frames = [core(), support('C5_sensitivity')]
    elif member == 'C6':
        frames = [support('C6_bounds_audit'), mixed(zip(spec['sections'], [support(name) for name in
            ('C6_calibration', 'C6_support_status')], strict=True))]
    elif member == 'SYN':
        frames = [synthesis['claim_evidence_status'][spec['status_columns']], synthesis['engineering_choices'],
                  synthesis['inheritance_revision'], synthesis['claim_experiment_figure_coverage'], synthesis['T_INV_01']]
        # Preserve the six non-ID audit/limitation tables formerly beside the figures.
        for name in ('claim_coordinate_audit', 'claim_evidence_status', 'inference_resolution_audit',
                     'coverage_corrections', 'target_count_audit', 'limitations_registry'):
            write_csv(synthesis[name], target / f'{name}.csv')
    else:
        raise ValueError(f'unknown Report member: {member}')
    for identifier, frame in zip(registration['tables'], frames, strict=True):
        write_csv(frame, target / 'tables' / f'{identifier}.csv')
