"""C2 的守恒反例、完整均值恒等式与不可识别状态。"""

import numpy as np
import pytest

from sglib.experiment.correction import coordinates

pytestmark = pytest.mark.consume


def test_conserved_totals_do_not_remove_log_mean_term():
    y = [32.46757171249298, 31.54901162043157, 32.20494205571397, 3.7784746113614744]
    b = [1.4232233398534218, 23.91858568224978, 23.215517534604775, 51.44267344329202]
    p = [3.4697272773258696, 45.795037286628485, 3.0821268603659657, 47.65310857567969]
    assert all(np.isclose(sum(v), 100.) for v in (y, b, p))
    result = coordinates(y, b, p, 1e-4)
    assert result["old_threshold_help"] and not result["log_mse_help"]
    assert result["delta_log_mse"] == pytest.approx(.0719666642320016)
    assert result["mean_term"] > 0 and result["variance_term"] < 0
    assert result["identity_pass"] and abs(result["identity_residual"]) < 1e-12


def test_constant_effect_is_not_a_failed_identity_or_a_valid_threshold():
    result = coordinates([1., 2.], [1., 2.], [1., 2.], 1e-4)
    assert not result["identifiable"]
    assert result["old_threshold_help"] is None
    assert result["identity_pass"] and result["delta_log_mse"] == 0.
    assert coordinates([], [], [], 1e-4)["status"] == "METRIC_NOT_ASSESSABLE"
    assert coordinates([0.], [0.], [0.], 0.)["status"] == "METRIC_NOT_ASSESSABLE"
