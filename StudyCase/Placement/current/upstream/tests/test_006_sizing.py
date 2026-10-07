"""定容合法零预测必须贡献 RSD/CE，不能被 TUR 的零分母连带删除。"""

from sglib.experiment.planning_tasks import sizing_scores

import pytest
pytestmark = pytest.mark.consume


def test_zero_prediction_is_retained_in_rsd_and_ce_but_not_tur():
    rows = {r["metric"]: r for r in sizing_scores([0., 10.], [0., 15.], [10., 10.], [20., 20.])}
    assert rows["RSD_median"]["value"] == 50.
    assert rows["RSD_median"]["n_valid_matches"] == 2
    assert rows["CE_median"]["value"] == 75.
    assert rows["TUR_mean"]["n_valid_matches"] == 1
    assert rows["FCE_mean"]["n_valid_matches"] == 2
