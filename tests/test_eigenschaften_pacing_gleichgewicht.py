"""Eigenschaftstests (ORACLE-PLAYBOOK §5): Aussagen, die für jeden Markt gelten müssen, unabhängig von einem Vergleichswert."""

import numpy as np
import pytest

import pac_algorithm as A
import pac_constants as C
import pac_scenario as S

SCALES = (0.05, 0.3, 1.0, 3.0)


def _markets(count=3, sizes=((3, 4), (5, 8))):
    for kind in C.KINDS:
        for scale in SCALES:
            for n, m in sizes:
                for seed in range(count):
                    yield kind, scale, S.make_market(kind, n, m, "ungleich", scale, seed)


def test_the_equilibrium_satisfies_the_definition_everywhere():
    for kind, scale, mk in _markets():
        res = A.solve(mk.v, mk.budgets)
        assert res.ok and A.definition_ok(mk.v, mk.budgets, res.p, res.alpha, res.spend), (kind, scale)
        assert (res.alpha > 0).all() and (res.alpha <= 1 + 1e-12).all() and (res.spend >= 0).all()


def test_the_price_is_the_highest_bid_and_the_budget_is_never_exceeded():
    for kind, scale, mk in _markets():
        res = A.solve(mk.v, mk.budgets)
        assert (res.alpha[:, None] * mk.v).max(axis=0) == pytest.approx(res.p, abs=1e-8), (kind, scale)
        assert (res.spend.sum(axis=1) <= mk.budgets + 1e-8).all()


def test_only_buyers_who_spend_their_whole_budget_slow_down():
    for kind, scale, mk in _markets():
        res = A.solve(mk.v, mk.budgets)
        paced = res.alpha < 1 - 1e-9
        assert np.allclose(res.spend.sum(axis=1)[paced], mk.budgets[paced], atol=1e-8), (kind, scale)


def test_the_equilibrium_is_unique_from_random_start_points():
    rng = np.random.default_rng(1)
    for kind, scale, mk in _markets(2, ((4, 6),)):
        ref = A.solve(mk.v, mk.budgets)
        b0 = rng.random(mk.v.shape) * (mk.v > 0) + 1e-3
        b0 = b0 / b0.sum(axis=1, keepdims=True) * mk.budgets[:, None] * 0.7
        b = A.gradient_phase(mk.v, mk.budgets, 3000, False, b0)
        p, alpha, spend, res = A.polish(mk.v, mk.budgets, b, False)
        assert res < 1e-9 and p == pytest.approx(ref.p, rel=1e-7), (kind, scale)


def test_revenue_does_not_fall_when_a_budget_grows():
    rng = np.random.default_rng(2)
    for kind, scale, mk in _markets(2):
        r0 = float(A.solve(mk.v, mk.budgets).p.sum())
        i = int(rng.integers(mk.n))
        b2 = mk.budgets.copy()
        b2[i] *= 1.25
        assert float(A.solve(mk.v, b2).p.sum()) >= r0 - 1e-8, (kind, scale)


def test_revenue_is_bounded_by_the_budgets_and_by_the_free_revenue():
    for kind, scale, mk in _markets():
        r = float(A.solve(mk.v, mk.budgets).p.sum())
        assert r <= mk.budgets.sum() + 1e-8 and r <= mk.free_revenue + 1e-8, (kind, scale)


def test_scaling_values_and_budgets_scales_the_prices_and_leaves_the_factors():
    for kind, scale, mk in _markets(2):
        if kind == "ganzzahlig":
            continue
        a, b = A.solve(mk.v, mk.budgets), A.solve(4.0 * mk.v, 4.0 * mk.budgets)
        assert b.p == pytest.approx(4.0 * a.p, rel=1e-7) and b.alpha == pytest.approx(a.alpha, abs=1e-7), (kind, scale)


def test_renaming_the_buyers_only_permutes_the_factors():
    rng = np.random.default_rng(3)
    for kind, scale, mk in _markets(2):
        if kind == "ganzzahlig":
            continue
        perm = rng.permutation(mk.n)
        a, b = A.solve(mk.v, mk.budgets), A.solve(mk.v[perm], mk.budgets[perm])
        assert b.p == pytest.approx(a.p, rel=1e-7) and b.alpha == pytest.approx(a.alpha[perm], abs=1e-7), (kind, scale)


def test_when_every_budget_binds_the_allocation_does_not_depend_on_the_scale():
    for seed in range(10):
        small = S.make_market("stetig", 4, 8, "ungleich", 0.05, seed)
        smaller = S.make_market("stetig", 4, 8, "ungleich", 0.2, seed)
        a, b = A.solve(small.v, small.budgets), A.solve(smaller.v, smaller.budgets)
        assert (a.alpha < 1).all() and (b.alpha < 1).all()
        assert a.x == pytest.approx(b.x, abs=1e-7) and b.p == pytest.approx(4.0 * a.p, rel=1e-7), seed


def test_with_huge_budgets_nobody_slows_down_and_the_price_is_the_highest_value():
    for seed in range(8):
        mk = S.make_market("stetig", 4, 8, "ungleich", 50.0, seed)
        res = A.solve(mk.v, mk.budgets)
        assert res.ok and (res.alpha == pytest.approx(1.0)) and res.p == pytest.approx(mk.v.max(axis=0), rel=1e-9)


def test_every_budget_binding_gives_the_fisher_market_in_prices_and_allocation():
    for kind in C.KINDS:
        for seed in range(6):
            mk = S.make_market(kind, 4, 6, "ungleich", 0.05, seed)
            pac, fisher = A.solve(mk.v, mk.budgets), A.solve(mk.v, mk.budgets, equality=True)
            assert (pac.alpha < 1).all() and pac.p == pytest.approx(fisher.p, rel=1e-7), (kind, seed)
            if kind == "stetig":
                assert pac.x == pytest.approx(fisher.x, abs=1e-7)


def test_the_second_price_equilibrium_always_exists_for_two_buyers():
    rng = np.random.default_rng(4)
    for level in C.SPPE_LEVELS:
        for _ in range(60):
            v = rng.random((2, C.SPPE_M)) + 0.05
            b = (rng.random(2) + 0.5) * level * v.max(axis=0).sum() / 2
            assert A.sppe_two(v, b)


def test_without_budget_pressure_second_price_earns_less_than_first_price():
    rng = np.random.default_rng(5)
    for _ in range(40):
        v = rng.random((2, C.SPPE_M)) + 0.05
        b = np.array([100.0, 100.0])
        eqs = A.sppe_two(v, b)
        assert len(eqs) == 1
        assert eqs[0][2] + eqs[0][3] == pytest.approx(float(np.sort(v, axis=0)[0].sum()), rel=1e-9)        # Summe der zweithöchsten Werte
        assert eqs[0][2] + eqs[0][3] <= float(v.max(axis=0).sum()) + 1e-9
