"""Orakel-Tests des Kerns gegen unabhängige Rechenwege (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §2).

Orakel: (1) das Erstpreis-Programm mit SLSQP (scipy) gegen das Polieren, (2) die Definition als eigene Prüffunktion, (3) das Eisenberg-Gale-Programm des Fisher-Marktes in der Zuteilung (SLSQP)
gegen das Pacing-Gleichgewicht bei lauter bindenden Budgets, (4) die Zweitpreis-Aufzählung gegen eine Rastersuche über alle Faktorenpaare mit rationalen Werten und eine reine Intervallprüfung
der Gleichstandsaufteilung, (5) Handbeispiele von Hand. Alle < 10 s."""

import itertools
from fractions import Fraction

import numpy as np
import pytest

import pac_algorithm as A
import pac_constants as C
import pac_scenario as S

so = pytest.importorskip("scipy.optimize")


def slsqp_first_price(v, budgets):
    """Das Erstpreis-Programm in den Ausgaben b, gelöst mit SLSQP: (Preise p_j = sum_i b_ij, Zielwert). SLSQP ist die ungenauere Seite und konvergiert nicht immer."""
    n, m = v.shape
    mask = v > 0
    logv = A.log_values(v)

    def f(z):
        b = z.reshape(n, m) * mask
        p = np.maximum(b.sum(axis=0), 1e-300)
        return float((p * np.log(p)).sum() - p.sum() - (b * logv).sum())

    cons = [{"type": "ineq", "fun": (lambda z, i=i: budgets[i] - z.reshape(n, m)[i].sum())} for i in range(n)]
    z0 = (mask / mask.sum(axis=1, keepdims=True) * budgets[:, None] * 0.5).ravel()
    r = so.minimize(f, z0, constraints=cons, bounds=[(1e-9, None)] * (n * m), method="SLSQP", options={"ftol": 1e-15, "maxiter": 3000})
    b = r.x.reshape(n, m) * mask
    violation = float(np.max(np.maximum(b.sum(axis=1) - budgets, 0.0) / budgets))        # SLSQP hält die Nebenbedingungen nur näherungsweise ein
    return b.sum(axis=0), float(r.fun), violation


def our_objective(mk, res):
    return A.objective(res.spend, A.log_values(mk.v))


def eisenberg_gale_prices(v, budgets):
    """Fisher-Markt: max sum_i B_i log u_i, sum_i x_ij = 1 (SLSQP in der Zuteilung); Preise p_j = max_i B_i v_ij / u_i."""
    n, m = v.shape

    def neg(z):
        x = z.reshape(n, m)
        return -(budgets * np.log((v * x).sum(axis=1))).sum()

    cons = [{"type": "eq", "fun": (lambda z, j=j: z.reshape(n, m)[:, j].sum() - 1.0)} for j in range(m)]
    r = so.minimize(neg, np.full(n * m, 1.0 / n), constraints=cons, bounds=[(1e-9, 1.0)] * (n * m), method="SLSQP", options={"ftol": 1e-15, "maxiter": 2000})
    u = (v * r.x.reshape(n, m)).sum(axis=1)
    return (budgets[:, None] * v / u[:, None]).max(axis=0)


@pytest.mark.parametrize("scale", [0.1, 0.5, 1.2, 3.0])
@pytest.mark.parametrize("kind", C.KINDS)
def test_first_price_satisfies_the_definition_and_is_not_beaten_by_a_feasible_slsqp_point(kind, scale):
    """Das Programm ist konvex: Ein zulässiger SLSQP-Punkt hat nie einen niedrigeren Zielwert als unsere Lösung (SLSQP verletzt die Budgets zuweilen um Spuren und liegt dann scheinbar darunter)."""
    for seed in range(5):
        mk = S.make_market(kind, 4, 6, "ungleich", scale, seed)
        res = A.solve(mk.v, mk.budgets)
        assert res.ok and A.definition_ok(mk.v, mk.budgets, res.p, res.alpha, res.spend)
        _, f_sl, violation = slsqp_first_price(mk.v, mk.budgets)
        if violation < 1e-9:
            assert our_objective(mk, res) <= f_sl + 1e-9, (kind, scale, seed)


def test_first_price_prices_agree_with_slsqp_where_slsqp_converged():
    """Über 40 Märkte: wo SLSQP zulässig ist und den Zielwert fast erreicht (innerhalb 1e-6), stimmen die Preise überein; das gilt für mindestens die Hälfte der Märkte."""
    agreed = total = 0
    for kind in C.KINDS:
        for scale in (0.1, 0.5, 1.2, 3.0):
            for seed in range(5):
                mk = S.make_market(kind, 4, 6, "ungleich", scale, seed)
                res = A.solve(mk.v, mk.budgets)
                p_sl, f_sl, violation = slsqp_first_price(mk.v, mk.budgets)
                total += 1
                if violation < 1e-9 and abs(f_sl - our_objective(mk, res)) < 1e-6:
                    agreed += 1
                    assert res.p == pytest.approx(p_sl, rel=1e-3), (kind, scale, seed)
    assert total == 40 and agreed >= 20


def test_every_budget_binding_is_the_fisher_market_against_eisenberg_gale():
    for seed in range(8):
        mk = S.make_market("stetig", 4, 5, "ungleich", 0.05, seed)
        res = A.solve(mk.v, mk.budgets)
        assert res.ok and (res.alpha < 1 - 1e-9).all()
        assert res.p == pytest.approx(eisenberg_gale_prices(mk.v, mk.budgets), rel=1e-5), seed
        fisher = A.solve(mk.v, mk.budgets, equality=True)
        assert fisher.ok and res.p == pytest.approx(fisher.p, rel=1e-7)


def test_the_fisher_program_against_eisenberg_gale_even_when_pacing_would_not_bind():
    for seed in range(5):
        mk = S.make_market("stetig", 4, 5, "ungleich", 3.0, seed)
        fisher = A.solve(mk.v, mk.budgets, equality=True)
        assert fisher.ok and fisher.p == pytest.approx(eisenberg_gale_prices(mk.v, mk.budgets), rel=1e-5)


def test_the_hand_example_against_slsqp(hand):
    v, b = hand
    assert A.solve(v, b).p == pytest.approx([2.1, 1.4, 0.5]) and slsqp_first_price(v, b)[0] == pytest.approx([2.1, 1.4, 0.5], rel=1e-3)


def rational_grid(q_max):
    return sorted({Fraction(k, q) for q in range(1, q_max + 1) for k in range(1, q + 1)})


def test_the_second_price_enumeration_against_a_search_over_all_rational_factor_pairs():
    """Mini-Märkte mit kleinen ganzzahligen Werten und Budgets mit halben Zahlen: Jedes Faktorenpaar (k/q, l/r) des Rasters, für das die reine Intervallprüfung eine Aufteilung der Gleichstandsposten findet,
    ist in der Aufzählung, und jedes Gleichgewicht der Aufzählung auf dem Raster besteht die Intervallprüfung."""
    grid = [float(f) for f in rational_grid(24)]
    rng = np.random.default_rng(5)
    checked = found = 0
    for _ in range(12):
        v = rng.integers(1, 5, (2, 3)).astype(float)
        b = rng.integers(1, 7, 2) / 2.0 - 0.0 + np.array([0.5, 0.5])        # halbe Zahlen 1, 1,5, ..., 3,5: kein Budget ist eine Summe ganzer Gebote (sonst Kontinuum)
        enum = A.sppe_two(v, b)
        enum_pairs = {(round(e[0], 8), round(e[1], 8)) for e in enum}
        grid_pairs = {(round(a, 8), round(c, 8)) for a, c in itertools.product(grid, grid) if A.sppe_feasible(v, b, a, c)}
        assert grid_pairs <= enum_pairs, (v.tolist(), b.tolist(), sorted(grid_pairs - enum_pairs))
        on_grid = {p for p in enum_pairs if p[0] in {round(x, 8) for x in grid} and p[1] in {round(x, 8) for x in grid}}
        assert on_grid <= grid_pairs, (v.tolist(), b.tolist(), sorted(on_grid - grid_pairs))
        checked += 1
        found += len(grid_pairs)
    assert checked == 12 and found >= 12                                  # die Mini-Märkte haben Gleichgewichte auf dem Raster, der Vergleich ist nicht leer


def test_every_enumerated_second_price_equilibrium_passes_the_interval_check_in_random_markets():
    rng = np.random.default_rng(6)
    for level in C.SPPE_LEVELS:
        for _ in range(40):
            v = rng.random((2, 6)) + 0.05
            b = (rng.random(2) + 0.5) * level * v.max(axis=0).sum() / 2
            eqs = A.sppe_two(v, b)                                          # Zufallsbudgets: nie entartet
            assert eqs, (level, v.tolist())                                 # Existenz
            assert all(A.sppe_feasible(v, b, e[0], e[1]) for e in eqs)
            for e in eqs:                                                   # Ausgaben liegen im erlaubten Rahmen
                assert e[2] <= b[0] + 1e-9 and e[3] <= b[1] + 1e-9 and (e[0] >= 1 - 1e-12 or abs(e[2] - b[0]) < 1e-9) and (e[1] >= 1 - 1e-12 or abs(e[3] - b[1]) < 1e-9)


def test_the_interval_check_rejects_wrong_factors(hand):
    v, b = hand
    assert not A.sppe_feasible(v, b, 0.5, 1.0)                              # Kunde 1 bremst, ohne dass sein Budget genau ausgeschöpft wird
    assert not A.sppe_feasible(v, b, 1.0, 0.5)
    assert not A.sppe_feasible(v, b, 1.0, 1.0)                              # beide bieten voll: Kunde 2 würde sein Budget 3,5 überschreiten
    assert A.sppe_feasible(v, b, 1 / 6, 1.0) and A.sppe_feasible(v, b, 8 / 9, 4 / 9) and A.sppe_feasible(v, b, 7 / 8, 0.5)
