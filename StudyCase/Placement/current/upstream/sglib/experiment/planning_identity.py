"""规划计算的统一代码投影；运行目录和调度描述不参与身份。"""

from copy import deepcopy
import inspect

from sglib.core.infra.content_chain import code_projection
from sglib.experiment import planning_tasks, planning_pool
from sglib.experiment.connection import reference_field
from sglib.core.algorithms import pmedian, planning_geometry, planning_metrics, planning_sizing, planning_candidates


def numerical_code():
    modules = (planning_tasks, planning_pool, pmedian, planning_geometry, planning_metrics, planning_sizing, planning_candidates)
    symbols = {f"{m.__name__}.{name}": value for m in modules for name, value in inspect.getmembers(m, inspect.isfunction)
               if value.__module__ == m.__name__}
    symbols["VD_reference"] = reference_field
    return code_projection(symbols)["code_sha256"]


def scientific_projection(parameters):
    result = deepcopy(parameters)
    # v1 的实际执行也固定为一个计算线程；此处把隐含值显式化。
    result.setdefault("compute_threads", 1)
    for value in result.get("pools", {}).values():
        value.get("kmeans", {}).pop("verbose", None)
    return result
