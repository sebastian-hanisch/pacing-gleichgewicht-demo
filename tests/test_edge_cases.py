"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import time

import numpy as np
import pytest

import pac_algorithm as A
import pac_constants as C
import pac_evaluation as ev
import pac_scenario as S


def test_smallest_market_for_both_kinds_and_extreme_scales():
    for kind in C.KINDS:
        for mode in C.BUDGET_MODES:
            for scale in (C.SCALE_MIN, C.SCALE_MAX):
                a = ev.analyse(ev.Settings(kind, C.N_MIN, C.M_MIN, mode, scale, 3))
                assert a.fppe.ok and a.fisher.ok, (kind, mode, scale)
                assert (a.fppe.alpha > 0).all() and (a.fppe.alpha <= 1 + 1e-12).all()


def test_the_largest_market_runs_in_time_for_both_kinds():
    t = time.time()
    for kind in C.KINDS:
        a = ev.analyse(ev.Settings(kind, C.N_MAX, C.M_MAX, "ungleich", 1.0, 5))
        assert a.fppe.ok and a.fisher.ok
    assert time.time() - t < 30


def test_the_scale_curve_and_the_census_run_in_time():
    t = time.time()
    rows = ev.scale_curve(ev.Settings(n=C.N_MAX, m=C.M_MAX))
    assert len(rows) == C.SCALE_POINTS and all(r["ok"] for r in rows)
    ev.sppe_census()
    ev.monotonicity()
    assert time.time() - t < 90


def test_identical_buyers_get_identical_factors():
    v = np.tile(np.array([[1.0, 2.0, 0.5, 3.0]]), (3, 1))
    res = A.solve(v, np.array([0.8, 0.8, 0.8]))
    assert res.ok and res.alpha == pytest.approx(np.full(3, res.alpha[0])) and res.spend.sum(axis=1) == pytest.approx([0.8] * 3)


def test_a_buyer_with_a_tiny_budget_still_gets_a_consistent_equilibrium():
    v = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    res = A.solve(v, np.array([1e-3, 5.0]))
    assert res.ok and res.spend.sum(axis=1)[0] == pytest.approx(1e-3, rel=1e-6) and res.alpha[0] < 1
    assert res.alpha[0] * v[0, 2] == pytest.approx(res.p[2])                                    # der kleine Kunde bietet gerade den Preis der Anfrage 3 und kauft davon nur ein winziges Stück
    assert res.x[0, 2] == pytest.approx(1e-3 / res.p[2], rel=1e-6)


def test_one_buyer_with_a_budget_exactly_at_the_free_price_does_not_slow_down():
    v = np.array([[2.0]])
    res = A.solve(v, np.array([2.0]))                                      # Budget gleich dem Wert: gerade eben ausreichend, Faktor 1
    assert res.ok and res.alpha == pytest.approx([1.0]) and res.p == pytest.approx([2.0])


def test_the_generators_are_deterministic_and_positive():
    for kind in C.KINDS:
        a, b = S.make_market(kind, 5, 7, "ungleich", 1.0, 9), S.make_market(kind, 5, 7, "ungleich", 1.0, 9)
        assert np.array_equal(a.v, b.v) and np.array_equal(a.budgets, b.budgets) and (a.v > 0).all() and (a.budgets > 0).all()
        assert not np.array_equal(a.v, S.make_market(kind, 5, 7, "ungleich", 1.0, 10).v)
    assert S.make_market("stetig", 4, 6, "gleich", 1.0, 1).budgets.std() == pytest.approx(0.0)
    assert S.make_market("stetig", 4, 6, "ungleich", 1.0, 1).budgets.std() > 0
    assert set(S.make_market("ganzzahlig", 6, 8, "gleich", 1.0, 1).v.ravel().tolist()) <= set(float(x) for x in range(1, C.INT_VALUE_MAX + 1))


def test_the_budget_scale_multiplies_every_budget():
    base = S.make_market("stetig", 4, 6, "ungleich", 1.0, 3)
    scaled = S.make_market("stetig", 4, 6, "ungleich", 2.5, 3)
    assert np.array_equal(base.v, scaled.v) and scaled.budgets == pytest.approx(2.5 * base.budgets)


def test_free_revenue_and_second_price_references():
    mk = S.Market(np.array([[1.0, 5.0], [3.0, 2.0], [2.0, 4.0]]), np.array([1.0, 1.0, 1.0]))
    assert mk.free_revenue == pytest.approx(3.0 + 5.0) and mk.free_second == pytest.approx(2.0 + 4.0)


def test_the_second_price_enumeration_with_one_query_and_identical_values_is_a_tie():
    v, b = np.array([[3.0], [3.0]]), np.array([5.0, 5.0])
    eqs = A.sppe_two(v, b)
    assert len(eqs) == 1 and eqs[0][:2] == (1.0, 1.0) and eqs[0][2] + eqs[0][3] == pytest.approx(3.0)


def test_the_hand_market_and_the_tie_market_are_what_the_texts_say():
    h = S.hand_market()
    assert h.v.tolist() == [[3.0, 1.0, 6.0], [6.0, 4.0, 1.0]] and h.budgets.tolist() == [0.5, 3.5]
    t = S.tie_market()
    assert t.v.tolist() == [[4.0], [2.0]] and t.budgets.tolist() == [1.0, 1.0]
    assert S.sppe_market("hand", 1, 1.0).v.tolist() == h.v.tolist()
    z = S.sppe_market("zufall", 4, 0.3)
    assert z.v.shape == (2, C.SPPE_M) and np.array_equal(z.v, S.sppe_market("zufall", 4, 0.3).v) and not np.array_equal(z.v, S.sppe_market("zufall", 5, 0.3).v)
