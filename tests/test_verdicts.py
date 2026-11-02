"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import pac_evaluation as ev


def _row(scale, rev, welf, share, bind, gap):
    return {"scale": scale, "revenue_ratio": rev, "welfare_ratio": welf, "paced_share": share, "all_bind": bind, "fisher_gap": gap}


def test_equilibrium_verdict_branches():
    kind, text = ev.verdict_equilibrium(True, 3, 4, 0.8934)
    assert kind == "success" and "**3 von 4** Kunden bremsen" in text and "89,3 % der Einnahme ohne Budgets" in text and "von der Definition bestätigt" in text
    kind, text = ev.verdict_equilibrium(False, 3, 4, 0.89)
    assert kind == "error" and "nicht gelungen" in text and "Näherungen" in text


def test_scale_verdict_with_binding_scales():
    rows = [_row(0.05, 0.043, 0.972, 1.0, True, 0.0), _row(0.5, 0.473, 0.972, 1.0, True, 0.0), _row(4.0, 1.0, 1.0, 0.0, False, 0.77)]
    kind, text = ev.verdict_scale(rows)
    assert kind == "info"
    for part in ("Von Maßstab 0,05 bis 4,00", "von 4,3 % auf 100,0 % der Einnahme ohne Budgets", "die Wohlfahrt von 97,2 % auf 100,0 %", "von 100 % auf 0 %",
                 "Auf den 2 Maßstäben, auf denen alle Budgets binden, sind die Preise gleich dem Fisher-Gleichgewicht (Abweichung unter 1e-7)"):
        assert part in text, part


def test_scale_verdict_flags_a_deviation_from_the_fisher_market():
    rows = [_row(0.05, 0.043, 0.972, 1.0, True, 1e-3), _row(4.0, 1.0, 1.0, 0.0, False, 0.77)]
    kind, text = ev.verdict_scale(rows)
    assert kind == "error" and "**Achtung:**" in text and "bis 0,00" in text
    assert ev.verdict_scale([_row(0.05, 0.043, 0.972, 1.0, True, 9.9e-8), _row(4.0, 1.0, 1.0, 0.0, False, 0.5)])[0] == "info"          # knapp unter der Schwelle 1e-7
    assert ev.verdict_scale([_row(0.05, 0.043, 0.972, 1.0, True, 1.1e-7), _row(4.0, 1.0, 1.0, 0.0, False, 0.5)])[0] == "error"        # knapp darüber


def test_scale_verdict_without_binding_scales():
    kind, text = ev.verdict_scale([_row(1.0, 0.9, 0.99, 0.5, False, 0.2), _row(4.0, 1.0, 1.0, 0.0, False, 0.7)])
    assert kind == "info" and "Auf keinem der Maßstäbe binden alle Budgets" in text


def test_second_price_verdict_branches():
    kind, text = ev.verdict_sppe([5 / 3, 4.0, 4.0], 4.0)
    assert kind == "warning" and "**3** Zweitpreis-Gleichgewichte" in text and "von 1,667 bis 4,000 (das 2,40-Fache)" in text and "Erstpreis hat genau eines und nimmt 4,000 ein" in text
    kind, text = ev.verdict_sppe([2.0, 2.0], 2.0)
    assert kind == "warning" and "2,000 in allen" in text
    kind, text = ev.verdict_sppe([3.0], 5.0)
    assert kind == "info" and "Genau **ein** Zweitpreis-Gleichgewicht, Einnahme 3,000" in text and "nimmt 5,000 ein" in text
    kind, text = ev.verdict_sppe([], 5.0)
    assert kind == "error" and "Existenzsatz" in text


def test_census_and_monotone_verdicts():
    rows = [{"level": 0.1, "markets": 100, "multi": 0}, {"level": 0.3, "markets": 100, "multi": 8}]
    kind, text = ev.verdict_census(rows, 60, 1000)
    assert kind == "info" and "Stufe 0,1: 0 von 100" in text and "Stufe 0,3: 8 von 100" in text and "60 von 1000 (6,0 %)" in text
    kind, text = ev.verdict_monotone(0, 100)
    assert kind == "success" and "In allen 100 Paaren" in text and "kein Rückgang" in text
    kind, text = ev.verdict_monotone(3, 100)
    assert kind == "warning" and "In 3 von 100 Paaren sinkt die Einnahme" in text


def test_table_rows_use_the_rounding_of_the_verdicts():
    census = {"rows": [{"level": 0.3, "markets": 100, "none": 0, "multi": 8, "spread_median": 1.0234, "spread_max": 1.0701, "ratio_lo": 1.0, "ratio_hi": 1.0},
                       {"level": 1.0, "markets": 100, "none": 0, "multi": 0, "spread_median": None, "spread_max": None, "ratio_lo": 0.6338, "ratio_hi": 0.6338}], "int_multi": 0, "int_total": 1}
    assert ev.census_rows(census) == ["| 0,3 | 100 | 0 | 8 | 1,02 / 1,07 | 1,000 | 1,000 |", "| 1,0 | 100 | 0 | 0 | – | 0,634 | 0,634 |"]
    rows = ev.hand_rows()
    assert rows[0] == "| 1 Kunde, 1 Anfrage, Wert 2, Budget 1 | 0,500 | 1,000 | 1,000 | 1,000 |"
    assert rows[1] == "| 1 Kunde, 1 Anfrage, Wert 2, Budget 3 | 1,000 | 2,000 | 2,000 | 2,000 |"
    assert rows[2] == "| 2 Kunden, 1 Anfrage, Werte 4 und 2, Budgets 1 und 1 | 0,500; 1,000 | 2,000 | 1,000; 1,000 | 2,000 |"
