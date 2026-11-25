import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hp_search import sample_configs


def test_first_is_cs02_control_and_distinct() -> None:
    cfgs = sample_configs(12, 1)
    assert len(cfgs) == 12 and len(set(cfgs.values())) == 12
    first = cfgs["hp_000"]
    assert (first.colsample, first.subsample_freq, first.num_leaves, first.min_child, first.reg_lambda, first.extra_trees) == (0.2, 1, 63, 80, 1.0, False)


def test_deterministic_and_bounded() -> None:
    assert sample_configs(8, 3) == sample_configs(8, 3)
    for c in sample_configs(30, 2).values():
        assert 0.1 <= c.colsample <= 0.4 and 0.3 <= c.reg_lambda <= 30.0 and c.subsample_freq == 1
