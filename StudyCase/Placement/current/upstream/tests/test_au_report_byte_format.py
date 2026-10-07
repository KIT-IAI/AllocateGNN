"""The AU publication boundary retains its historical compact JSON bytes."""
import json
import os

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from sglib.core.infra.hashing import sha256_file
from sglib.dataoverview.processing.config import CountryPipelineContext
from sglib.dataoverview.processing.derive.au.au_fy2024 import publish_au_fy2024_demand

pytestmark = pytest.mark.produce


def test_au_publication_compact_json_matches_legacy_text_bytes(tmp_path):
    codes = [f'S{index:03d}' for index in range(34)]
    groups = [f'A{index % 12:02d}' for index in range(34)]
    context = CountryPipelineContext(tmp_path, {
        'country': {'code': 'au'}, 'temporal': {'protocol': 'FY2024_same_year'},
        'regions': {'items': [{'source_key_order': codes}]},
    })
    derived = context.derived_root
    (derived / 'ledger').mkdir(parents=True)
    regions = gpd.GeoDataFrame({
        'sa3_code': codes, 'sa4_code': groups, 'sa4_name': ['区域' + key for key in groups],
        'loc_key': groups, 'population_erp_2009': [100] * 34,
    }, geometry=[box(index, 0, index + .8, .8) for index in range(34)], crs='EPSG:4326')
    regions.to_file(derived / 'regions_sa3.gpkg', layer='regions_sa3', driver='GPKG', index=False)
    regions.dissolve(by='sa4_code').reset_index().to_file(
        derived / 'regions_sa4.gpkg', layer='regions_sa4', driver='GPKG', index=False)
    pd.DataFrame({'sa3_code': codes}).to_csv(derived / 'region_attributes.csv', index=False)
    outside_name = '外部站点\n第二行\r\n原样\\n字符'
    pd.DataFrame({
        'station': [f'站点{index}' for index in range(34)] + [outside_name],
        'status': ['usable'] * 35,
        'lon_wgs84': [index + .4 for index in range(34)] + [40.],
        'lat_wgs84': [.4] * 35,
        'peak_mw': [1.] * 35, 'energy_gwh': [2.] * 35,
    }).to_csv(derived / 'ledger/station_table_fy2024.csv', index=False, encoding='utf-8-sig')

    report = publish_au_fy2024_demand(context=context)
    assert report['n_sa3'] == 34 and report['n_sa4'] == 12
    assert report['outside_scope_stations'][0]['station'] == outside_name
    target = derived / 'ledger/fy2024_region_demand_report.json'
    reference = tmp_path / 'legacy_au_report.json'
    reference.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n',
        encoding='utf-8',
    )
    assert target.read_bytes() == reference.read_bytes()
    assert sha256_file(target) == sha256_file(reference)
    assert target.read_bytes().endswith(os.linesep.encode('utf-8'))
    assert json.loads(target.read_text(encoding='utf-8')) == report
