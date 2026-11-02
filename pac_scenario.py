"""Szenario: synthetische Märkte für Pacing-Gleichgewichte (nur numpy). Zufall nur über den übergebenen Generator, nie über `hash()`.

Vehikel: Werbekunden einer Frachtbörse mit Tagesbudget, viele einzelne Suchanfragen. v[i, j] = Wert, den Kunde i einer Anfrage j beimisst; budgets[i] = Tagesbudget.
Keine Personen-, Firmen- oder Börsendaten."""

from dataclasses import dataclass

import numpy as np

import pac_constants as C


@dataclass(frozen=True, eq=False)
class Market:
    v: np.ndarray           # n x m Werte (>= 0)
    budgets: np.ndarray     # n Budgets > 0

    @property
    def n(self) -> int:
        return self.v.shape[0]

    @property
    def m(self) -> int:
        return self.v.shape[1]

    @property
    def free_revenue(self) -> float:
        """Einnahme ohne Budgets im Erstpreis: Summe der höchsten Werte je Anfrage."""
        return float(self.v.max(axis=0).sum())

    @property
    def free_second(self) -> float:
        """Einnahme ohne Budgets im Zweitpreis: Summe der zweithöchsten Werte je Anfrage."""
        return float(np.sort(self.v, axis=0)[-2].sum())


def random_values(kind: str, n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "stetig":
        return rng.random((n, m)) + 0.05
    if kind == "ganzzahlig":
        return rng.integers(1, C.INT_VALUE_MAX + 1, (n, m)).astype(float)
    raise ValueError(f"unbekannte Erzeugerart {kind!r}")


def base_budgets(v: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Budget bei Maßstab 1: (Zufall 0,5 bis 1,5 oder 1) mal Summe der Höchstwerte durch n; Maßstab s skaliert alle Budgets."""
    n = v.shape[0]
    weights = rng.random(n) + 0.5 if mode == "ungleich" else np.ones(n)
    return weights * v.max(axis=0).sum() / n


def market_from_rng(kind: str, mode: str, n: int, m: int, scale: float, rng: np.random.Generator) -> Market:
    v = random_values(kind, n, m, rng)
    return Market(v, scale * base_budgets(v, mode, rng))


def make_market(kind: str, n: int, m: int, mode: str, scale: float, seed: int) -> Market:
    """Deterministisch aus (kind, n, m, mode, scale, seed): gleiche Eingabe, gleicher Markt."""
    return market_from_rng(kind, mode, n, m, scale, np.random.default_rng(seed))


def hand_market() -> Market:
    """Handbeispiel mit drei Zweitpreis-Gleichgewichten: v = ((3, 2, 6), (4, 6, 1)), B = (1, 5)."""
    v, b = C.HAND_C
    return Market(np.array(v, dtype=float), np.array(b, dtype=float))


def tie_market() -> Market:
    """Zwei Kunden, eine Anfrage: v = (4; 2), B = (1; 1): Gleichstand beim Gebot 2, beide geben 1 aus."""
    return Market(np.array([[4.0], [2.0]]), np.array([1.0, 1.0]))


def sppe_market(mode: str, seed: int, level: float) -> Market:
    """Kleiner Markt für die Zweitpreis-Aufzählung: das Handbeispiel oder ein Zufallsmarkt (2 Kunden, SPPE_M Anfragen, Budgetstufe `level`)."""
    if mode == "hand":
        return hand_market()
    rng = np.random.default_rng(seed)
    v = rng.random((2, C.SPPE_M)) + 0.05
    return Market(v, (rng.random(2) + 0.5) * level * v.max(axis=0).sum() / 2)
