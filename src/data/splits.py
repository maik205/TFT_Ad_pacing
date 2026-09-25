"""Subject-independent splitting and a leakage assertion.

The single rule: the same subject (or session/recording) must never appear in
both train and test. Random row-wise splits on per-window data leak identity and
inflate every result, so build splits by group and assert disjointness.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterator, List, Sequence, Tuple


def assert_no_subject_leakage(
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    subject_ids: Sequence,
) -> None:
    """Raise AssertionError if any subject appears in both train and test."""
    train_subj = {subject_ids[i] for i in train_idx}
    test_subj = {subject_ids[i] for i in test_idx}
    overlap = train_subj & test_subj
    assert not overlap, (
        f"Subject leakage: {len(overlap)} subject(s) in both train and test "
        f"-> {sorted(map(str, overlap))[:10]}. Splits MUST group by subject."
    )


def _index_by_subject(subject_ids: Sequence) -> "OrderedDict":
    groups: OrderedDict = OrderedDict()
    for i, s in enumerate(subject_ids):
        groups.setdefault(s, []).append(i)
    return groups


def loso_splits(subject_ids: Sequence) -> Iterator[Tuple[List[int], List[int]]]:
    """Leave-one-subject-out splits.

    Yields (train_idx, test_idx) with one subject held out per fold. Every fold
    is leakage-checked before it is yielded.
    """
    groups = _index_by_subject(subject_ids)
    if len(groups) < 2:
        raise ValueError("LOSO needs at least 2 distinct subjects.")
    for held_out, test_idx in groups.items():
        train_idx = [i for s, idxs in groups.items() if s != held_out for i in idxs]
        assert_no_subject_leakage(train_idx, test_idx, subject_ids)
        yield train_idx, test_idx


def grouped_kfold(
    subject_ids: Sequence, k: int, seed: int = 0
) -> Iterator[Tuple[List[int], List[int]]]:
    """Subject-grouped k-fold: subjects are partitioned into k folds; all windows
    of a subject stay together. Folds are leakage-checked before yielding.
    """
    import random as _random

    subjects = list(OrderedDict.fromkeys(subject_ids))
    if k < 2 or k > len(subjects):
        raise ValueError(f"k must be in [2, n_subjects={len(subjects)}].")
    rng = _random.Random(seed)
    rng.shuffle(subjects)
    folds = [subjects[i::k] for i in range(k)]
    idx_by_subj = _index_by_subject(subject_ids)
    for f in range(k):
        test_subj = set(folds[f])
        test_idx = [i for s in test_subj for i in idx_by_subj[s]]
        train_idx = [i for s in subjects if s not in test_subj for i in idx_by_subj[s]]
        assert_no_subject_leakage(train_idx, test_idx, subject_ids)
        yield train_idx, test_idx
