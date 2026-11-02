"""Zeilen der README-Tabellen, gerechnet über dieselben Funktionen wie in der App (pac_evaluation, pac_algorithm) und in der Rundung der App.

`tests/test_claims.py` prüft jede dieser Zeilen gegen die README; `python tools/readme_rows.py` druckt sie zum Einfügen.
Alle Zahlen sind deterministisch (feste Seeds); Preise, Einnahmen und Wohlfahrt auf drei Stellen (das Polieren benutzt lstsq, Zahlen am Rundungsboden werden nicht berichtet)."""

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pac_constants as C  # noqa: E402
import pac_evaluation as ev  # noqa: E402
from pac_scenario import hand_market  # noqa: E402


@lru_cache(maxsize=None)
def curve_result() -> list:
    return ev.scale_curve(ev.Settings())


@lru_cache(maxsize=None)
def sppe_hand() -> dict:
    return ev.sppe_report(hand_market())


@lru_cache(maxsize=None)
def census_result() -> dict:
    return ev.sppe_census()


@lru_cache(maxsize=None)
def mono_result() -> dict:
    return ev.monotonicity()


def hand_rows() -> list[str]:
    return ev.hand_rows()


def curve_rows() -> list[str]:
    return ev.scale_rows(curve_result())


def sppe_rows() -> list[str]:
    return ev.sppe_rows(sppe_hand())


def census_rows() -> list[str]:
    return ev.census_rows(census_result())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("Handbeispiele"); print("\n".join(hand_rows()))
    print("\nKurve (Standardmarkt)"); print("\n".join(curve_rows()))
    print("\nZweitpreis, Handbeispiel"); print("\n".join(sppe_rows()))
    print("\nZählung"); print("\n".join(census_rows())); c = census_result(); print("ganzzahlige Werte:", c["int_multi"], "von", c["int_total"])
    print("\nMonotonie", mono_result())
    a = ev.analyse(ev.Settings())
    print("\nStandardmarkt", int(a.paced.sum()), "von", a.market.n, "bremsen; Einnahme", round(a.revenue, 3), f"{100 * a.revenue_ratio:.1f} % ; Wohlfahrt {100 * a.welfare_ratio:.1f} %")
