"""Leakage guardrail: splits must never share a subject across train/test.

    pytest tests/test_no_leakage.py

This protects against the single bug that can invalidate an entire paper.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splits import (  # noqa: E402
    assert_no_subject_leakage,
    grouped_kfold,
    loso_splits,
)

# Synthetic: 6 subjects, 5 windows each (mimics per-window features).
SUBJECT_IDS = [s for s in range(6) for _ in range(5)]  # 30 windows


def test_loso_one_fold_per_subject():
    folds = list(loso_splits(SUBJECT_IDS))
    assert len(folds) == 6
    for train_idx, test_idx in folds:
        assert_no_subject_leakage(train_idx, test_idx, SUBJECT_IDS)
        assert set(train_idx).isdisjoint(test_idx)


def test_grouped_kfold_no_leakage():
    folds = list(grouped_kfold(SUBJECT_IDS, k=3, seed=0))
    assert len(folds) == 3
    for train_idx, test_idx in folds:
        assert_no_subject_leakage(train_idx, test_idx, SUBJECT_IDS)


def test_leakage_is_detected():
    # idx 0 -> subject 0 (train); idx 2 -> subject 0 (test) => must raise.
    with pytest.raises(AssertionError):
        assert_no_subject_leakage([0, 1], [2, 7], SUBJECT_IDS)
