"""Jede Zahl der README steht hier und wird über dieselben Funktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4, §11a).

Die Tabellen der README werden Zeile für Zeile gegen `tools/readme_rows.py` geprüft (dieselbe Rundung wie die App); die einzelnen Sätze der README
haben je einen Test unter der Bedingung, die in der README daneben steht. Preise, Einnahmen und Wohlfahrt nur auf drei Stellen (Lehre aus Stück 1)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import pac_algorithm as A  # noqa: E402
import pac_constants as C  # noqa: E402
import pac_evaluation as ev  # noqa: E402
import readme_rows as R  # noqa: E402

README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_tables_in_the_readme_are_the_computed_rows():
    for name, rows in (("Tabelle 1", R.hand_rows()), ("Tabelle 2", R.curve_rows()), ("Tabelle 3", R.sppe_rows()), ("Tabelle 4", R.census_rows())):
        for row in rows:
            assert row in README, (name, row)


def test_the_hand_examples_of_table_1():
    one = A.solve(np.array([[2.0]]), np.array([1.0]))
    assert one.alpha == pytest.approx([0.5]) and one.p == pytest.approx([1.0])
    tie = A.solve(np.array([[4.0], [2.0]]), np.array([1.0, 1.0]))
    assert tie.alpha == pytest.approx([0.5, 1.0]) and tie.p == pytest.approx([2.0]) and tie.spend.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert "Im dritten Beispiel bieten beide 2 (Kunde 1 bremst auf 0,5): Gleichstand, jeder gibt sein Budget 1 aus, der Preis ist 2" in README


def test_the_standard_market_numbers():
    a = ev.analyse(ev.Settings())
    assert (a.market.n, a.market.m, a.market.budgets.size, C.DEFAULT_SCALE, C.DEFAULT_SEED) == (4, 8, 4, 1.2, 5)
    assert int(a.paced.sum()) == 3
    assert f"{a.revenue:.3f}" == "6.577" and f"{100 * a.revenue_ratio:.1f}" == "89.3" and f"{100 * a.welfare_ratio:.1f}" == "98.8"
    assert 0.225 < a.fisher_gap < 0.235
    for part in ("**3 von 4** Kunden bremsen, die Einnahme ist **6,577** (**89,3 %** der Einnahme ohne Budgets), die Wohlfahrt **98,8 %**", "um bis zu **23 %**"):
        assert part in README, part


def test_the_budget_scale_sentences():
    rows = R.curve_result()
    binding = [r for r in rows if r["all_bind"]]
    assert len(binding) == 8 and [r["scale"] for r in rows][:8] == [r["scale"] for r in binding]
    assert all(f"{100 * r['welfare_ratio']:.1f}" == "97.2" for r in binding)
    assert all(r["fisher_gap"] < C.FISHER_TOL for r in binding)
    assert not any(r["all_bind"] for r in rows[8:]) and rows[-1]["paced_share"] == 0.0 and rows[-1]["revenue_ratio"] == pytest.approx(1.0)
    assert all(r["ok"] for r in rows)
    for part in ("Solange **jedes** Budget bindet (die ersten acht Zeilen), ist die **Wohlfahrt konstant** (97,2 %)", "Abweichung unter 1e-7", "zwölf Maßstäbe von 0,05 bis 4"):
        assert part in README, part
    assert C.SCALE_POINTS == 12 and (C.SCALE_GRID[0], C.SCALE_GRID[-1]) == (0.05, 4.0)


def test_the_monotonicity_sentence():
    r = R.mono_result()
    assert (r["pairs"], r["drops"]) == (100, 0) and C.MONO_FACTOR == 1.25
    assert "sinkt die Einnahme in **0 von 100** Paaren" in README and "(2 bis 5 Kunden, 3 bis 8 Anfragen, Maßstab 0,1, 0,3 oder 1)" in README


def test_the_second_price_sentences_of_the_hand_example(hand):
    rep = R.sppe_hand()
    revs = sorted(e["revenue"] for e in rep["equilibria"])
    assert [round(x, 3) for x in revs] == [1.667, 4.0, 4.0] and f"{max(revs) / min(revs):.2f}" == "2.40" and f"{rep['fp_revenue']:.3f}" == "4.000"
    assert len(rep["equilibria"]) == 3 and rep["fppe"].ok
    for part in ("Die Einnahmen liegen zwischen **1,667** und **4,000** (das 2,40-Fache)", "mit der Einnahme 4,000", "im dritten bremsen beide (7/8 und 1/2) ganz **ohne** Gleichstand"):
        assert part in README, part


def test_the_census_sentences():
    c = R.census_result()
    assert [(r["markets"], r["none"], r["multi"]) for r in c["rows"]] == [(100, 0, 0), (100, 0, 8), (100, 0, 0), (100, 0, 0)]
    assert (c["int_multi"], c["int_total"]) == (60, 1000) and f"{100 * c['int_multi'] / c['int_total']:.1f}" == "6.0"
    assert (C.CENSUS_MARKETS, C.CENSUS_SEED, C.SPPE_M, C.INT_CENSUS, C.INT_CENSUS_SEED, C.INT_VALUE_MAX) == (100, 21, 6, 1000, 22, 7)
    ratios = {r["level"]: (round(r["ratio_lo"], 3), round(r["ratio_hi"], 3)) for r in c["rows"]}
    assert ratios == {0.1: (1.0, 1.0), 0.3: (1.0, 1.0), 1.0: (0.634, 0.634), 3.0: (0.525, 0.525)}
    for part in ("in Zufallsmärkten **selten** (0 bis 8 von 100)", "(0,634 und 0,525 der Erstpreis-Einnahme)", "In **60 von 1 000** Märkten mit ganzzahligen Werten", "(**6,0 %**)", "(Werte 1 bis 7, 2 Kunden, 3 Anfragen, Zufallsbudgets 0,5 bis 4,5; Seed 22;"):
        assert part in README, part


def test_the_degenerate_example_named_in_the_readme_is_a_continuum():
    v, b = np.array([[3.0, 2.0, 6.0], [4.0, 6.0, 1.0]]), np.array([1.0, 5.0])
    with pytest.raises(A.DegenerateMarket):
        A.sppe_two(v, b)
    assert "Budget 1 gleich dem Wert 1 von Kunde 2 für Anfrage 3" in README


def test_the_readme_hand_example_matches_the_scenario():
    from pac_scenario import hand_market
    mk = hand_market()
    assert mk.v.tolist() == [[3.0, 1.0, 6.0], [6.0, 4.0, 1.0]] and mk.budgets.tolist() == [0.5, 3.5]
    assert "Werte (3, 1, 6) und (6, 4, 1), Budgets 0,5 und 3,5" in README
