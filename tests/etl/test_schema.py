"""schema 一致性单测。"""
from retailinsight.io.schema import (
    ANCHOR_COMBOS,
    CANONICAL,
    CATEGORIES,
    CATEGORY_ALIASES,
    PERSONAS,
)


def test_categories_count():
    assert len(CATEGORIES) == 60
    assert len(set(CATEGORIES)) == 60


def test_aliases_point_to_canonical():
    for alias, canonical in CATEGORY_ALIASES.items():
        assert alias not in CATEGORIES
        assert canonical in CATEGORIES


def test_anchor_combos_in_categories():
    for a, b, prob in ANCHOR_COMBOS:
        assert a in CATEGORIES and b in CATEGORIES
        assert 0 < prob < 1


def test_personas_weights_sum():
    total = sum(p["weight"] for p in PERSONAS.values())
    assert abs(total - 1.0) < 1e-9
    assert len(PERSONAS) == 8


def test_canonical_tables():
    assert set(CANONICAL) == {"members", "stores", "products", "transactions", "items"}
    for cols in CANONICAL.values():
        assert len(cols) == len(set(cols))
