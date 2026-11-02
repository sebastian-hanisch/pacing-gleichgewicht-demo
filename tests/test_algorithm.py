"""Einheitentests des Kerns an von Hand gerechneten Mini-Märkten (DEMO-PLAYBOOK §1b)."""

import numpy as np
import pytest

import pac_algorithm as A
import pac_constants as C


def test_one_buyer_one_query_by_hand(single):
    low = A.solve(single, np.array([1.0]))
    assert low.ok and low.p == pytest.approx([1.0]) and low.alpha == pytest.approx([0.5]) and low.spend.ravel() == pytest.approx([1.0])
    rich = A.solve(single, np.array([3.0]))
    assert rich.ok and rich.p == pytest.approx([2.0]) and rich.alpha == pytest.approx([1.0]) and rich.spend.ravel() == pytest.approx([2.0])


def test_two_buyers_one_query_with_a_tie_by_hand(tie):
    v, b = tie
    res = A.solve(v, b)
    assert res.ok and res.p == pytest.approx([2.0]) and res.alpha == pytest.approx([0.5, 1.0]) and res.spend.ravel() == pytest.approx([1.0, 1.0])
    assert res.x.ravel() == pytest.approx([0.5, 0.5])                     # der Gleichstand wird geteilt


def test_the_first_price_equilibrium_of_the_hand_example(hand):
    v, b = hand
    res = A.solve(v, b)
    assert res.ok and res.alpha == pytest.approx([1 / 12, 0.35]) and res.p == pytest.approx([2.1, 1.4, 0.5])
    assert res.spend.sum(axis=1) == pytest.approx([0.5, 3.5]) and float(res.p.sum()) == pytest.approx(4.0)
    # Kunde 1 gewinnt Anfrage 3 (Gebot 0,5 gegen 0,35), Kunde 2 gewinnt Anfragen 1 und 2 (Gebote 2,1 und 1,4 gegen 0,25 und 1/12)
    assert res.x.round(9).tolist() == [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]


def test_the_second_price_equilibria_of_the_hand_example(hand):
    v, b = hand
    eqs = A.sppe_two(v, b)
    got = sorted((round(e[0], 9), round(e[1], 9), round(e[2], 9), round(e[3], 9)) for e in eqs)
    want = sorted([(round(1 / 6, 9), 1.0, 0.5, round(7 / 6, 9)), (round(8 / 9, 9), round(4 / 9, 9), 0.5, 3.5), (0.875, 0.5, 0.5, 3.5)])
    assert got == want
    assert sorted(round(e[2] + e[3], 9) for e in eqs) == [round(5 / 3, 9), 4.0, 4.0]
    assert all(A.sppe_feasible(v, b, e[0], e[1]) for e in eqs)


def test_a_budget_equal_to_a_sum_of_bids_gives_a_continuum_and_is_refused():
    v, b = np.array([[3.0, 2.0, 6.0], [4.0, 6.0, 1.0]]), np.array([1.0, 5.0])     # B1 = 1 ist der Wert von Kunde 2 für Anfrage 3: Kunde 1 gibt für alle a in (1/6, 1) genau 1 aus
    with pytest.raises(A.DegenerateMarket):
        A.sppe_two(v, b)


def test_second_price_with_one_query_and_no_budget_pressure():
    v, b = np.array([[5.0], [3.0]]), np.array([10.0, 10.0])
    eqs = A.sppe_two(v, b)
    assert len(eqs) == 1 and eqs[0][:2] == (1.0, 1.0) and eqs[0][2] + eqs[0][3] == pytest.approx(3.0)      # Kunde 1 gewinnt und zahlt das zweite Gebot 3


def test_the_projection_has_the_right_row_sums_and_respects_the_mask():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(5, 4))
    mask = rng.random((5, 4)) < 0.7
    mask[np.arange(5), rng.integers(0, 4, 5)] = True
    budgets = rng.random(5) + 0.5
    eq = A.project(y, budgets, mask, True)
    assert eq.sum(axis=1) == pytest.approx(budgets) and (eq >= 0).all() and (eq[~mask] == 0).all()
    cap = A.project(y, budgets, mask, False)
    assert (cap.sum(axis=1) <= budgets + 1e-12).all() and (cap >= 0).all() and (cap[~mask] == 0).all()
    small = np.where(mask, 0.01, 0.0)
    assert A.project(small, budgets, mask, False) == pytest.approx(small)          # innerhalb der Grenzen unverändert
    assert A.project(small, budgets, mask, True).sum(axis=1) == pytest.approx(budgets)


def test_the_definition_check_detects_violations(single, tie):
    v, b = tie
    res = A.solve(v, b)
    assert A.definition_ok(v, b, res.p, res.alpha, res.spend)
    assert not A.definition_ok(v, b, res.p, np.array([1.0, 1.0]), res.spend)                     # beide bieten voll: Kunde 1 wäre über dem Budget
    assert not A.definition_ok(v, b, res.p, res.alpha, res.spend * 0.5)                          # Spaltensumme stimmt nicht mehr
    assert not A.definition_ok(v, b, res.p * 1.1, res.alpha, res.spend)                          # Preis ist nicht das höchste Gebot
    sv = np.array([[2.0]])
    r = A.solve(sv, np.array([3.0]))
    assert not A.definition_ok(sv, np.array([3.0]), r.p, np.array([0.8]), r.spend)               # bremst, obwohl das Budget nicht ausgeschöpft ist


def test_the_fisher_equilibrium_by_hand_when_every_budget_binds():
    v, b = np.array([[2.0, 1.0], [1.0, 2.0]]), np.array([1.0, 1.0])
    fisher = A.solve(v, b, equality=True)
    assert fisher.ok and fisher.p == pytest.approx([1.0, 1.0]) and fisher.x.round(9).tolist() == [[1.0, 0.0], [0.0, 1.0]]
    paced = A.solve(v, b)                                                                         # Budgets 1 bei Werten 2: beide bremsen
    assert paced.ok and paced.p == pytest.approx([1.0, 1.0]) and paced.alpha == pytest.approx([0.5, 0.5])


def test_welfare_is_the_value_of_the_won_shares(tie):
    v, b = tie
    res = A.solve(v, b)
    assert A.welfare(v, res) == pytest.approx(0.5 * 4.0 + 0.5 * 2.0)


def test_zero_values_are_allowed_as_long_as_every_query_has_a_buyer():
    v = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]])
    res = A.solve(v, np.array([0.5, 0.5]))
    assert res.ok and (res.spend[v == 0] == 0).all() and (res.alpha <= 1).all()
