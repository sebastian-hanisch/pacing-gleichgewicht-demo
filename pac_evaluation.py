"""Kennzahlen, Verdikt-Logik und Sweeps (Komposition der Regeln aus `pac_algorithm.py`; Verdikte sind reine Funktionen)."""

from dataclasses import dataclass

import numpy as np

import pac_algorithm as A
import pac_constants as C
from pac_scenario import Market, make_market, market_from_rng, sppe_market


@dataclass(frozen=True)
class Settings:
    kind: str = C.DEFAULT_KIND
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    budgets: str = C.DEFAULT_BUDGETS
    scale: float = C.DEFAULT_SCALE
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True, eq=False)
class Analysis:
    market: Market
    fppe: A.Result                  # Erstpreis-Pacing-Gleichgewicht
    fisher: A.Result                # Fisher-Gleichgewicht desselben Marktes (alle Budgets bindend, Stück 1)

    @property
    def paced(self) -> np.ndarray:
        return self.fppe.alpha < 1 - 1e-9

    @property
    def revenue(self) -> float:
        return float(self.fppe.p.sum())

    @property
    def revenue_ratio(self) -> float:
        return self.revenue / self.market.free_revenue

    @property
    def welfare_ratio(self) -> float:
        return A.welfare(self.market.v, self.fppe) / self.market.free_revenue

    @property
    def paced_share(self) -> float:
        return float(self.paced.mean())

    @property
    def all_bind(self) -> bool:
        return bool(self.paced.all())

    @property
    def fisher_gap(self) -> float:
        """Größter relativer Preisunterschied zum Fisher-Gleichgewicht (gleich 0 nur, wenn alle Budgets binden)."""
        return float(np.max(np.abs(self.fppe.p - self.fisher.p) / self.fisher.p))


def analyse(settings: Settings) -> Analysis:
    mk = make_market(settings.kind, settings.n, settings.m, settings.budgets, settings.scale, settings.seed)
    return Analysis(mk, A.solve(mk.v, mk.budgets), A.solve(mk.v, mk.budgets, equality=True))


# --- Verdikte als reine Funktionen -------------------------------------------------------------------------------------------------------

def _num(v: float, d: int = 3) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def _pct(v: float, d: int = 1) -> str:
    return _num(100.0 * v, d)


def verdict_equilibrium(ok: bool, n_paced: int, n: int, rev_ratio: float) -> tuple[str, str]:
    """Kernfrage: Ist das ein Pacing-Gleichgewicht? Definition erfüllt, wie viele Kunden bremsen, Einnahme gegen die Einnahme ohne Budgets."""
    if not ok:
        return "error", "Das Polieren auf Rechengenauigkeit ist nicht gelungen oder die Definition ist verletzt: Die Zahlen unten sind nur Näherungen."
    return "success", (f"Gleichgewicht, von der Definition bestätigt (Preis = höchstes Gebot, Ausgabe ≤ Budget, Faktor unter 1 nur bei ausgeschöpftem Budget): **{n_paced} von {n}** Kunden bremsen. "
                       f"Die Einnahme ist {_pct(rev_ratio)} % der Einnahme ohne Budgets.")


def verdict_scale(rows: list[dict]) -> tuple[str, str]:
    """Budgetmaßstab: Verlauf von Einnahme, Wohlfahrt und Anteil gepacter Kunden vom kleinsten zum größten Maßstab; Fisher-Grenze auf den Maßstäben, auf denen alle Budgets binden."""
    first, last = rows[0], rows[-1]
    binding = [r for r in rows if r["all_bind"]]
    if binding:
        worst = max(r["fisher_gap"] for r in binding)
        equal = all(r["fisher_gap"] < C.FISHER_TOL for r in binding)
        tail = (f" Auf den {len(binding)} Maßstäben, auf denen alle Budgets binden, sind die Preise gleich dem Fisher-Gleichgewicht (Abweichung unter 1e-7)." if equal
                else f" **Achtung:** Auf Maßstäben mit lauter bindenden Budgets weicht das Gleichgewicht vom Fisher-Markt ab (bis {_num(worst, 2)}).")
    else:
        equal = True
        tail = " Auf keinem der Maßstäbe binden alle Budgets."
    head = (f"Von Maßstab {_num(first['scale'], 2)} bis {_num(last['scale'], 2)} steigt die Einnahme von {_pct(first['revenue_ratio'])} % auf {_pct(last['revenue_ratio'])} % der Einnahme ohne Budgets, "
            f"die Wohlfahrt von {_pct(first['welfare_ratio'])} % auf {_pct(last['welfare_ratio'])} %; der Anteil bremsender Kunden fällt von {_pct(first['paced_share'], 0)} % auf {_pct(last['paced_share'], 0)} %.")
    return ("info" if equal else "error"), head + tail


def verdict_sppe(revenues: list[float], fp_revenue: float) -> tuple[str, str]:
    """Zweitpreis, zwei Kunden: Zahl der Gleichgewichte, Spanne der Einnahmen, Erstpreis (eindeutig) zum Vergleich."""
    if not revenues:
        return "error", "Keine Aufzählung gefunden: Das darf nach dem Existenzsatz nicht passieren."
    lo, hi = min(revenues), max(revenues)
    if len(revenues) == 1:
        return "info", f"Genau **ein** Zweitpreis-Gleichgewicht, Einnahme {_num(lo)}; der Erstpreis nimmt {_num(fp_revenue)} ein (eindeutig)."
    spread = f"von {_num(lo)} bis {_num(hi)} (das {_num(hi / lo, 2)}-Fache)" if hi > lo * (1 + C.GAP_REL) else f"{_num(lo)} in allen"
    return "warning", (f"**{len(revenues)}** Zweitpreis-Gleichgewichte, Einnahmen {spread}; der Erstpreis hat genau eines und nimmt {_num(fp_revenue)} ein.")


def verdict_census(rows: list[dict], int_multi: int, int_total: int) -> tuple[str, str]:
    """Wie oft hat ein Zweitpreis-Markt mit zwei Kunden mehrere Gleichgewichte mit verschiedener Einnahme, je Budgetstufe und bei ganzzahligen Werten?"""
    parts = [f"Stufe {_num(r['level'], 1)}: {r['multi']} von {r['markets']}" for r in rows]
    return "info", ("Märkte mit mehreren Zweitpreis-Gleichgewichten verschiedener Einnahme – " + "; ".join(parts)
                    + f". Bei ganzzahligen Werten (viele Gleichstände): {int_multi} von {int_total} ({_pct(int_multi / int_total)} %).")


def verdict_monotone(drops: int, pairs: int) -> tuple[str, str]:
    """Einnahme im Budget (Erstpreis): in wie vielen Paaren sinkt sie, wenn ein Kunde mehr Budget bekommt?"""
    if drops == 0:
        return "success", f"In allen {pairs} Paaren steigt die Einnahme nicht, wenn ein Kunde 25 % mehr Budget bekommt (kein Rückgang)."
    return "warning", f"In {drops} von {pairs} Paaren sinkt die Einnahme, wenn ein Kunde 25 % mehr Budget bekommt."


# --- Kurve über den Budgetmaßstab ---------------------------------------------------------------------------------------------------------------

def scale_curve(settings: Settings, grid=C.SCALE_GRID) -> list[dict]:
    """Je Maßstab s dasselbe Markt-Gerüst (Werte und Gewichte aus dem Seed) mit Budgets s mal Basis: Einnahme, Wohlfahrt (relativ zur Einnahme ohne Budgets), Anteil bremsender Kunden,
    Abstand zum Fisher-Gleichgewicht und ob alle Budgets binden."""
    rows = []
    for s in grid:
        a = analyse(Settings(settings.kind, settings.n, settings.m, settings.budgets, float(s), settings.seed))
        rows.append({"scale": float(s), "revenue_ratio": a.revenue_ratio, "welfare_ratio": a.welfare_ratio, "paced_share": a.paced_share, "all_bind": a.all_bind,
                     "fisher_gap": a.fisher_gap, "ok": a.fppe.ok and a.fisher.ok})
    return rows


# --- Zweitpreis, zwei Kunden ------------------------------------------------------------------------------------------------------------------------

def sppe_report(market: Market) -> dict:
    """Alle Zweitpreis-Gleichgewichte des kleinen Marktes (Faktoren, Ausgaben, Einnahme) und das Erstpreis-Gleichgewicht zum Vergleich."""
    eqs = [{"alpha1": e[0], "alpha2": e[1], "spend1": e[2], "spend2": e[3], "revenue": e[2] + e[3]} for e in A.sppe_two(market.v, market.budgets)]
    fp = A.solve(market.v, market.budgets)
    return {"equilibria": eqs, "fppe": fp, "fp_revenue": float(fp.p.sum())}


def has_multiple(eqs: list) -> bool:
    revs = [e[2] + e[3] for e in eqs]
    return len(eqs) > 1 and max(revs) - min(revs) > C.GAP_REL


def sppe_census(levels=C.SPPE_LEVELS, markets: int = C.CENSUS_MARKETS, seed: int = C.CENSUS_SEED, int_markets: int = C.INT_CENSUS, int_seed: int = C.INT_CENSUS_SEED) -> dict:
    """Je Budgetstufe (Zufallsmärkte mit 2 Kunden und SPPE_M Anfragen): Märkte ohne Gleichgewicht (muss 0 sein), mit mehreren Gleichgewichten verschiedener Einnahme, Spanne größte / kleinste Einnahme,
    Einnahme Zweitpreis / Erstpreis (Median der kleinsten und der größten); dazu Märkte mit ganzzahligen Werten (2 Kunden, 3 Anfragen, Werte 1 bis 7, Zufallsbudgets 0,5 bis 4,5)."""
    rng = np.random.default_rng(seed)
    rows = []
    for level in levels:
        none = multi = 0
        spreads, lo_r, hi_r = [], [], []
        for _ in range(markets):
            v = rng.random((2, C.SPPE_M)) + 0.05
            b = (rng.random(2) + 0.5) * level * v.max(axis=0).sum() / 2
            eqs = A.sppe_two(v, b)
            if not eqs:
                none += 1
                continue
            revs = [e[2] + e[3] for e in eqs]
            if has_multiple(eqs):
                multi += 1
                spreads.append(max(revs) / min(revs))
            fp = float(A.solve(v, b).p.sum())
            lo_r.append(min(revs) / fp)
            hi_r.append(max(revs) / fp)
        rows.append({"level": level, "markets": markets, "none": none, "multi": multi, "spread_median": float(np.median(spreads)) if spreads else None,
                     "spread_max": max(spreads) if spreads else None, "ratio_lo": float(np.median(lo_r)), "ratio_hi": float(np.median(hi_r))})
    rng = np.random.default_rng(int_seed)
    int_multi = 0
    for _ in range(int_markets):
        v = rng.integers(1, C.INT_VALUE_MAX + 1, (2, 3)).astype(float)
        b = rng.random(2) * 4.0 + 0.5                                  # Zufallsbudgets: ganzzahlige Werte erzeugen Gleichstände, aber kein Budget ist eine Summe von Geboten
        int_multi += has_multiple(A.sppe_two(v, b))
    return {"rows": rows, "int_multi": int_multi, "int_total": int_markets}


def census_rows(census: dict) -> list[str]:
    """Zeilen der Tabelle (App und README rechnen dieselben)."""
    out = []
    for r in census["rows"]:
        spread = "–" if r["spread_median"] is None else f"{_num(r['spread_median'], 2)} / {_num(r['spread_max'], 2)}"
        out.append(f"| {_num(r['level'], 1)} | {r['markets']} | {r['none']} | {r['multi']} | {spread} | {_num(r['ratio_lo'], 3)} | {_num(r['ratio_hi'], 3)} |")
    return out


def monotonicity(pairs: int = C.MONO_PAIRS, seed: int = C.MONO_SEED) -> dict:
    """Erhöht man das Budget eines Kunden um 25 %, sinkt die Einnahme (Erstpreis) dann? Zufallsmärkte mit 2 bis 5 Kunden und 3 bis 8 Anfragen, Maßstab 0,1 / 0,3 / 1."""
    rng = np.random.default_rng(seed)
    drops = 0
    for _ in range(pairs):
        n, m = int(rng.integers(2, 6)), int(rng.integers(3, 9))
        mk = market_from_rng("stetig", C.DEFAULT_BUDGETS, n, m, float(rng.choice([0.1, 0.3, 1.0])), rng)
        r0 = float(A.solve(mk.v, mk.budgets).p.sum())
        b2 = mk.budgets.copy()
        i = int(rng.integers(n))
        b2[i] *= C.MONO_FACTOR
        drops += float(A.solve(mk.v, b2).p.sum()) < r0 - 1e-8
    return {"pairs": pairs, "drops": int(drops)}


# --- Handbeispiele ------------------------------------------------------------------------------------------------------------------------------

HAND_CASES = (
    ("1 Kunde, 1 Anfrage, Wert 2, Budget 1", [[2.0]], [1.0]),
    ("1 Kunde, 1 Anfrage, Wert 2, Budget 3", [[2.0]], [3.0]),
    ("2 Kunden, 1 Anfrage, Werte 4 und 2, Budgets 1 und 1", [[4.0], [2.0]], [1.0, 1.0]),
)


def hand_rows() -> list[str]:
    """Zeilen der Handbeispiel-Tabelle: Faktoren, Preise, Ausgaben je Kunde, Einnahme (Erstpreis); App und README rechnen dieselben."""
    rows = []
    for label, v, b in HAND_CASES:
        res = A.solve(np.array(v), np.array(b))
        rows.append(f"| {label} | {'; '.join(_num(float(a)) for a in res.alpha)} | {'; '.join(_num(float(p)) for p in res.p)} | "
                    f"{'; '.join(_num(float(s)) for s in res.spend.sum(axis=1))} | {_num(float(res.p.sum()))} |")
    return rows


def sppe_rows(report: dict) -> list[str]:
    """Zeilen der Zweitpreis-Tabelle (alle Gleichgewichte) und des eindeutigen Erstpreis-Gleichgewichts; App und README rechnen dieselben."""
    rows = [f"| Zweitpreis {i + 1} | {_num(e['alpha1'])} | {_num(e['alpha2'])} | {_num(e['spend1'])} | {_num(e['spend2'])} | {_num(e['revenue'])} |" for i, e in enumerate(report["equilibria"])]
    fp = report["fppe"]
    rows.append(f"| **Erstpreis** (eindeutig) | {_num(float(fp.alpha[0]))} | {_num(float(fp.alpha[1]))} | {_num(float(fp.spend[0].sum()))} | {_num(float(fp.spend[1].sum()))} | {_num(report['fp_revenue'])} |")
    return rows


def scale_rows(rows: list[dict]) -> list[str]:
    """Zeilen der Tabelle zur Kurve über den Budgetmaßstab; App und README rechnen dieselben."""
    return [f"| {_num(r['scale'], 3)} | {_pct(r['revenue_ratio'])} % | {_pct(r['welfare_ratio'])} % | {_pct(r['paced_share'], 0)} % | {'ja' if r['all_bind'] else 'nein'} |" for r in rows]
