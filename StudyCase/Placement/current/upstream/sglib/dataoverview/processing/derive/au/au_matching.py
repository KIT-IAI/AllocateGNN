"""AU 站名归一化与匹配工具

被 derive/au.py 复用:电压/财年 token 正则、归一化规则、编辑距离相似度、匹配阈值。
全部常量与函数逐行对应源脚本(注释亦保留——它们登记的是踩坑决策)。
"""

from __future__ import annotations

import re

# 电压 token:负荷文件名特有的 "一次_二次kV" 形态(如 " 33_11kV" / " 33_5kV"),
# 可出现在任意位置(如 "Morisset 33_11kV Temp")。刻意要求下划线形态:
# 图层名中的单数字电压(如 "Gosford 66kV STS" 与 "Gosford STS" 是两座不同站)不受影响。
VOLTAGE_TOKEN_RE = re.compile(r"\s+\d+_\d+(?:[._]\d+)*\s*KV\b", re.I)
# 财年后缀:文件名尾部 " FY2009"
FY_SUFFIX_RE = re.compile(r"\s+FY\d{4}$", re.I)
# 剥离的机构后缀 token(token 级精确匹配,不伤及正名)。
# ⚠ 刻意不剥 "STS":图层中 "Argenton"/"Argenton STS"、"Kurri"/"Kurri STS" 等
# 是成对并存的两座真实不同站(zone substation vs sub-transmission),剥离会碰撞合并。
# 同理不剥 "TEMP"/"NEW":负荷侧 "Morisset 132_11kV" 与 "Morisset 33_11kV Temp"、
# "Raymond Terrace" 与 "Raymond Terr NEW" 同年并存,为不同实体,交人工兜底。
SUFFIX_TOKENS = {"ZS", "SUBSTATION"}
# "ZONE SUBSTATION" 的 ZONE 仅在与 SUBSTATION 同现时剥离
ZONE_TOKEN = "ZONE"
# 常见缩写展开表(token 级,缩写 -> 全称)
ABBREV = {
    "MT": "MOUNT",
    "NTH": "NORTH",
    "STH": "SOUTH",
    "PT": "POINT",
    "TERR": "TERRACE",
}
FUZZY_THRESHOLD = 0.85
AMBIGUITY_GAP = 0.03  # 最优与次优相似度差 < 该值视为歧义,不自动采纳


def normalise(name: str) -> str:
    """站名归一化:大写化、去电压/财年后缀、去连字符标点、剥机构后缀、展开缩写。"""
    s = FY_SUFFIX_RE.sub("", name.strip())
    s = VOLTAGE_TOKEN_RE.sub("", s)
    s = s.upper()
    s = s.replace("&", " AND ")
    s = re.sub(r"[-_/]", " ", s)      # 连字符类 → 空格
    s = re.sub(r"[.'’()]", "", s)     # 撇号/句点/括号 → 删除
    tokens = [t for t in s.split() if t]
    # 剥机构后缀:ZS / STS / SUBSTATION;ZONE 仅在与 SUBSTATION 同现时剥
    has_substation = "SUBSTATION" in tokens
    tokens = [t for t in tokens
              if t not in SUFFIX_TOKENS and not (t == ZONE_TOKEN and has_substation)]
    # 缩写展开
    tokens = [ABBREV.get(t, t) for t in tokens]
    return " ".join(tokens)


def levenshtein(a: str, b: str) -> int:
    """标准编辑距离(插入/删除/替换各计 1),纯 Python 实现(站名短,性能无虞)。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """归一化相似度 = 1 - 编辑距离 / max(len)。"""
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))
