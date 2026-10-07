"""扫描观测的合成合同测试；不读取正式数据，不执行模型或求解器。"""

from types import SimpleNamespace

import numpy as np
import pytest

from sglib.experiment.sweep_observations import observe

pytestmark = pytest.mark.consume


def sample(parameter="lambda"):
    region = SimpleNamespace(country="uk", region="fixture", unit="MVA", assignment=np.array([0, 0, 1, 1]),
                             observed=np.array([3., 7.]), station_ids=np.array(["a", "b"]))
    field = SimpleNamespace(parameter=parameter, signal="N", value=.05, seed=42, fold=1, region="fixture",
                            values=np.array([1., 2., 3., 4.]), lineage={"sha256": "a" * 64})
    return region, field


def test_preserves_only_actual_seed_fold_and_separates_parameter_from_metric():
    region, field = sample()
    tables = observe(region, [field])
    metrics = tables["metrics"]
    assert len(metrics) == 5 and len(tables["predictions"]) == 2
    assert set(metrics.seed) == {42} and set(metrics.fold) == {1}
    assert set(metrics.parameter_value) == {.05}
    assert metrics.loc[metrics.metric.eq("rmse"), "value"].item() == 0.
    assert tables["predictions"].predicted.tolist() == [3., 7.]


def test_duplicate_or_cross_region_sweep_is_rejected():
    region, field = sample()
    with pytest.raises(ValueError, match="重复"):
        observe(region, [field, field])
    field.region = "outside"
    with pytest.raises(ValueError, match="跨区域"):
        observe(region, [field])


def test_nonfinite_or_wrong_length_is_rejected():
    region, field = sample("tau")
    for bad in (np.array([1.]), np.array([1., 2., np.nan, 4.]), np.array([1., 2., -1., 4.])):
        field.values = bad
        with pytest.raises(ValueError, match="不符"):
            observe(region, [field])
