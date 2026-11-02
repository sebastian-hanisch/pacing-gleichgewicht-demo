"""Kern: Pacing-Gleichgewicht (nur numpy).

n Werbekunden mit Budgets B_i, m Anfragen (je einmal), Werte v_ij >= 0. Jeder Kunde bietet alpha_i v_ij mit einem Faktor alpha_i in (0, 1]. Der höchste Bieter gewinnt (Gleichstand: beliebige
Aufteilung). Erstpreis: er zahlt sein Gebot; Zweitpreis: das zweithöchste. Pacing-Gleichgewicht: Ausgabe_i <= B_i, und alpha_i < 1 nur, wenn Ausgabe_i = B_i.

Erstpreis (eindeutig) als konvexes Programm mit den Ausgaben b_ij und p_j = sum_i b_ij:
    min  sum_j p_j log p_j - sum_j p_j - sum_ij b_ij log v_ij      unter  sum_j b_ij <= B_i,  b >= 0.
Optimalität: max_j v_ij / p_j = 1 / alpha_i >= 1 mit Gleichheit auf dem Träger, alpha_i = 1 bei nicht bindendem Budget; Preis p_j = max_i alpha_i v_ij.
Mit Gleichheit sum_j b_ij = B_i (alle Budgets bindend) ist es das Shmyrev-Programm des Fisher-Marktes (Stück 1; der Zusatzterm ist dann eine Konstante)."""

from dataclasses import dataclass

import numpy as np

import pac_constants as C


@dataclass(frozen=True, eq=False)
class Result:
    p: np.ndarray           # Preise je Anfrage
    alpha: np.ndarray       # Pacing-Faktoren (im Fisher-Fall: 1 / bester Nutzen je Preiseinheit, kann über 1 liegen)
    spend: np.ndarray       # Ausgaben b_ij
    ok: bool                # Polieren gelungen und Definition erfüllt
    iters: int              # Gradientenschritte der Vorstufe

    @property
    def x(self) -> np.ndarray:
        """Anteile x_ij = Ausgabe / Preis."""
        return self.spend / self.p[None, :]


# --- Programm (Gradient mit Armijo) --------------------------------------------------------------------------------------------------------

def log_values(v: np.ndarray) -> np.ndarray:
    mask = v > 0
    return np.where(mask, np.log(np.where(mask, v, 1.0)), 0.0)


def _project_simplex(y: np.ndarray, budgets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Euklidische Projektion jeder Zeile auf {b >= 0, Summe = B_i, b = 0 außerhalb der Maske} (Sortierverfahren, vektorisiert)."""
    n, m = y.shape
    z = np.where(mask, y, -1e300)
    u = -np.sort(-z, axis=1)
    css = np.cumsum(u, axis=1) - budgets[:, None]
    k = np.arange(1, m + 1)[None, :]
    cond = u - css / k > 0
    rho = m - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = css[np.arange(n), rho] / (rho + 1)
    return np.where(mask, np.maximum(y - theta[:, None], 0.0), 0.0)


def project(y: np.ndarray, budgets: np.ndarray, mask: np.ndarray, equality: bool) -> np.ndarray:
    """Projektion auf {b >= 0, Zeilensumme <= B_i} (Pacing) bzw. {Zeilensumme = B_i} (Fisher)."""
    if equality:
        return _project_simplex(y, budgets, mask)
    yp = np.where(mask, np.maximum(y, 0.0), 0.0)
    over = yp.sum(axis=1) > budgets
    if over.any():
        yp = np.where(over[:, None], _project_simplex(y, budgets, mask), yp)
    return yp


def objective(b: np.ndarray, logv: np.ndarray) -> float:
    p = b.sum(axis=0)
    if (p <= 0).any():
        return float("inf")
    return float((p * np.log(p)).sum() - p.sum() - (b * logv).sum())


def gradient_phase(v: np.ndarray, budgets: np.ndarray, iters: int, equality: bool, b0: np.ndarray | None = None) -> np.ndarray:
    """Projizierter Gradient mit Armijo-Rückverfolgung (Schrittweite wächst um 1,5 je Schritt, halbiert sich bis zum Abstieg); Start: die Hälfte des Budgets gleichmäßig verteilt (Fisher: das ganze) oder `b0`."""
    mask = v > 0
    logv = log_values(v)
    b = mask.astype(float)
    b = b / b.sum(axis=1, keepdims=True) * budgets[:, None] * (1.0 if equality else 0.5)
    if b0 is not None:
        b = b0.copy()
    t, f = 1.0, objective(b, logv)
    for _ in range(iters):
        grad = np.where(mask, np.log(b.sum(axis=0))[None, :] - logv, 0.0)
        t = min(t * 1.5, 1e6)
        while True:
            cand = project(b - t * grad, budgets, mask, equality)
            fc = objective(cand, logv)
            d = cand - b
            if np.isfinite(fc) and fc <= f + float((grad * d).sum()) + float((d * d).sum()) / (2 * t) + 1e-15:
                break
            t *= 0.5
            if t < 1e-14:
                break
        if float(np.abs(cand - b).max()) == 0.0:
            break
        b, f = cand, fc
    return b


# --- Polieren und Definition --------------------------------------------------------------------------------------------------------------

def polish(v: np.ndarray, budgets: np.ndarray, b: np.ndarray, equality: bool, rel: float = C.POLISH_REL):
    """Exakte Lösung aus dem Träger einer genäherten Lösung: Kanten mit b_ij > rel B_i; p_j = alpha_i v_ij, Spaltensummen = p_j, Zeilensumme = B_i bei bindendem Budget (Fisher: immer), sonst alpha_i = 1.
    Gibt (p, alpha, Ausgaben, Restfehler des linearen Systems) zurück."""
    n, m = v.shape
    si, sj = np.nonzero(b > rel * budgets[:, None])
    k = len(si)
    binding = np.ones(n, dtype=bool) if equality else b.sum(axis=1) >= budgets * (1 - C.BIND_TOL)
    size = k + m + n
    A = np.zeros((size, size))
    y = np.zeros(size)
    e = np.arange(k)
    A[e, k + sj] = 1.0                                    # p_j - alpha_i v_ij = 0
    A[e, k + m + si] = -v[si, sj]
    for i in range(n):
        if binding[i]:
            A[k + i, e[si == i]] = 1.0                    # sum_j b_ij = B_i
            y[k + i] = budgets[i]
        else:
            A[k + i, k + m + i] = 1.0                     # alpha_i = 1
            y[k + i] = 1.0
    A[k + n + sj, e] = 1.0                                # sum_i b_ij - p_j = 0
    A[k + n + np.arange(m), k + np.arange(m)] = -1.0
    sol = np.linalg.lstsq(A, y, rcond=None)[0]
    spend = np.zeros((n, m))
    spend[si, sj] = sol[:k]
    return sol[k:k + m], sol[k + m:], spend, float(np.abs(A @ sol - y).max())


def definition_ok(v: np.ndarray, budgets: np.ndarray, p: np.ndarray, alpha: np.ndarray, spend: np.ndarray, tol: float = C.DEF_TOL) -> bool:
    """Definition des Erstpreis-Gleichgewichts, unabhängig vom Programm: Preis = höchstes Gebot, nur Höchstbietende geben aus, Spaltensummen = Preis,
    Ausgabe <= Budget, alpha < 1 nur bei Ausgabe = Budget, alpha in (0, 1]."""
    bids = alpha[:, None] * v
    top = bids.max(axis=0)
    row = spend.sum(axis=1)
    return bool(
        np.abs(top - p).max() < tol
        and ((spend > tol) <= (bids >= top[None, :] - tol)).all()
        and np.abs(spend.sum(axis=0) - p).max() < tol
        and (row <= budgets + tol).all()
        and ((alpha >= 1 - tol) | (np.abs(row - budgets) < tol)).all()
        and ((alpha > 0) & (alpha <= 1 + tol)).all()
    )


def fisher_definition_ok(v: np.ndarray, budgets: np.ndarray, p: np.ndarray, spend: np.ndarray, tol: float = C.DEF_TOL) -> bool:
    """Definition des Fisher-Gleichgewichts: jeder Käufer gibt sein Budget aus und kauft nur beste Güter (Nutzen je Preiseinheit), Spaltensummen = Preis."""
    bpb = v / p[None, :]
    best = bpb.max(axis=1)
    return bool(
        np.abs(spend.sum(axis=1) - budgets).max() < tol
        and np.abs(spend.sum(axis=0) - p).max() < tol
        and ((spend > tol) <= (bpb >= best[:, None] * (1 - tol))).all()
    )


def solve(v: np.ndarray, budgets: np.ndarray, equality: bool = False, start: int = C.SOLVE_START, cap: int = C.SOLVE_CAP) -> Result:
    """Erstpreis-Pacing-Gleichgewicht (equality = False) oder Fisher-Gleichgewicht (equality = True): Gradient mit verdoppelter Schrittzahl, bis das Polieren die Definition erfüllt."""
    iters = start
    while True:
        b = gradient_phase(v, budgets, iters, equality)
        p, alpha, spend, res = polish(v, budgets, b, equality)
        spend = np.maximum(spend, 0.0)
        valid = res < 1e-9 and bool((p > 0).all()) and bool((alpha > 0).all())
        if valid and not equality:
            alpha = np.minimum(alpha, 1.0)
            valid = definition_ok(v, budgets, p, alpha, spend)
        elif valid:
            valid = fisher_definition_ok(v, budgets, p, spend)
        if valid or iters >= cap:
            return Result(p, alpha, spend, valid, iters)
        iters *= 2


def welfare(v: np.ndarray, res: Result) -> float:
    """Summe der Werte der gewonnenen Anteile: sum_ij v_ij x_ij mit x_ij = Ausgabe / Preis."""
    return float((v * res.x).sum())


# --- Zweitpreis, zwei Kunden: exakte Aufzählung ---------------------------------------------------------------------------------------------

def _spends(v: np.ndarray, a: float, b: float, tol: float = 1e-9):
    """Ausgaben ohne Gleichstände und Summe der Preise der Gleichstandsposten bei Faktoren (a, b): (Ausgabe 1, Ausgabe 2, Preissumme der Gleichstände)."""
    x, y = a * v[0], b * v[1]
    tie = np.abs(x - y) <= tol * np.maximum(np.maximum(x, y), 1e-12)
    win1 = (x > y) & ~tie
    win2 = (y > x) & ~tie
    return float(y[win1].sum()), float(x[win2].sum()), float(x[tie].sum())


class DegenerateMarket(ValueError):
    """Ein Budget ist genau gleich einer Summe von Geboten: dann gibt es ein Kontinuum von Zweitpreis-Gleichgewichten statt endlich vieler (bei Zufallsbudgets mit Wahrscheinlichkeit 0)."""


def sppe_two(v: np.ndarray, budgets: np.ndarray, tol: float = 1e-9) -> list[tuple[float, float, float, float]]:
    """Alle Zweitpreis-Pacing-Gleichgewichte (alpha1, alpha2, Ausgabe 1, Ausgabe 2) für zwei Kunden durch Aufzählung der Fälle: (1,1); (a < 1, 1) bei einem Gleichstandsverhältnis a = v2j/v1j;
    (1, b < 1); beide < 1 mit einem Gleichstandsposten (zwei lineare Gleichungen) oder ohne Gleichstand (jeder Faktor aus dem eigenen Budget: b = B1 / W1, a = B2 / W2 für eine Gewinnaufteilung nach dem Verhältnis v1j / v2j).
    Ausgaben an Gleichstandsposten sind frei aufteilbar; die Gleichgewichte sind nach (alpha1, alpha2) verschieden. Entartete Märkte (ein Budget gleich einer Summe von Geboten, Kontinuum) lösen DegenerateMarket aus."""
    v1, v2 = v[0], v[1]
    B1, B2 = float(budgets[0]), float(budgets[1])
    m = len(v1)
    out = []
    # (1, 1)
    s1, s2, tie = _spends(v, 1.0, 1.0)
    lo, hi = max(0.0, tie - (B2 - s2)), min(tie, B1 - s1)
    if s1 <= B1 + 1e-12 and s2 <= B2 + 1e-12 and lo <= hi + 1e-12:
        out.append((1.0, 1.0, s1 + lo, s2 + tie - lo))
    # (a < 1, 1): Kunde 1 gepaced
    for a in sorted({v2[j] / v1[j] for j in range(m) if v1[j] > 0 and v2[j] / v1[j] < 1.0}):
        s1, s2, tie = _spends(v, a, 1.0)
        x = B1 - s1
        if -1e-12 <= x <= tie + 1e-12 and s2 + tie - x <= B2 + 1e-9:
            out.append((a, 1.0, B1, s2 + tie - x))
    # (1, b < 1): Kunde 2 gepaced
    for b in sorted({v1[j] / v2[j] for j in range(m) if v2[j] > 0 and v1[j] / v2[j] < 1.0}):
        s1, s2, tie = _spends(v, 1.0, b)
        x = B2 - s2
        if -1e-12 <= x <= tie + 1e-12 and s1 + tie - x <= B1 + 1e-9:
            out.append((1.0, b, s1 + tie - x, B2))
    # beide < 1: ein Gleichstandsposten legt r = alpha1 / alpha2 = v2_j / v1_j fest; b (W1 + z) = B1 und b (W2 + T - z) = B2
    for jt in range(m):
        if v1[jt] <= 0 or v2[jt] <= 0:
            continue
        r = v2[jt] / v1[jt]
        W1 = float(sum(v2[j] for j in range(m) if r * v1[j] > v2[j] * (1 + tol)))
        W2 = float(sum(r * v1[j] for j in range(m) if r * v1[j] < v2[j] * (1 - tol)))
        T = float(sum(v2[j] for j in range(m) if abs(r * v1[j] - v2[j]) <= tol * max(v2[j], 1e-12)))
        den = W1 + W2 + T
        if den <= 0:
            continue
        b = (B1 + B2) / den
        a = r * b
        z = B1 / b - W1
        if 1e-12 < b < 1 - 1e-12 and 1e-12 < a < 1 - 1e-12 and -1e-9 <= z <= T + 1e-9:
            out.append((a, b, B1, B2))
    # beide < 1 ohne Gleichstand: Gewinnaufteilung nach dem Verhältnis v1j / v2j (Kunde 1 gewinnt die Anfragen mit dem größten Verhältnis)
    ratio = np.where(v2 > 0, v1 / np.where(v2 > 0, v2, 1.0), np.inf)
    order = np.argsort(-ratio, kind="stable")
    for k in range(m + 1):
        s1_set, s2_set = order[:k], order[k:]
        W1, W2 = float(v2[s1_set].sum()), float(v1[s2_set].sum())
        if W1 <= 0 or W2 <= 0:
            continue
        b, a = B1 / W1, B2 / W2
        if not (1e-12 < a < 1 - 1e-12 and 1e-12 < b < 1 - 1e-12):
            continue
        if all(a * v1[j] > b * v2[j] * (1 + tol) for j in s1_set) and all(a * v1[j] < b * v2[j] * (1 - tol) for j in s2_set):
            out.append((a, b, B1, B2))
    # Entartung: Zwischen zwei Gleichstandsverhältnissen ist die Ausgabe des Kunden mit Faktor 1 gegenüber dem anderen konstant; trifft sie das Budget genau, gibt es ein Kontinuum
    for ratios, solo_is_one in ((sorted({v2[j] / v1[j] for j in range(m) if v1[j] > 0 and v2[j] / v1[j] < 1.0}), True), (sorted({v1[j] / v2[j] for j in range(m) if v2[j] > 0 and v1[j] / v2[j] < 1.0}), False)):
        edges = [0.0] + ratios + [1.0]
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi - lo < 1e-9:
                continue
            mid = 0.5 * (lo + hi)
            s1, s2, _ = _spends(v, mid, 1.0) if solo_is_one else _spends(v, 1.0, mid)
            if solo_is_one and abs(s1 - B1) < tol * max(B1, 1.0) and s2 <= B2 + 1e-9:
                raise DegenerateMarket("Budget 1 ist genau eine Summe von Geboten von Kunde 2")
            if not solo_is_one and abs(s2 - B2) < tol * max(B2, 1.0) and s1 <= B1 + 1e-9:
                raise DegenerateMarket("Budget 2 ist genau eine Summe von Geboten von Kunde 1")
    uniq: list = []
    for e in out:
        if not any(abs(e[0] - u[0]) < 1e-9 and abs(e[1] - u[1]) < 1e-9 for u in uniq):
            uniq.append(e)
    return uniq


def sppe_feasible(v: np.ndarray, budgets: np.ndarray, a: float, b: float, tol: float = 1e-9) -> bool:
    """Gibt es zu den Faktoren (a, b) eine Aufteilung der Gleichstandsposten, die ein Zweitpreis-Gleichgewicht ergibt? Unabhängig von `sppe_two` (reine Intervallprüfung):
    Ausgabe 1 liegt in [Basis 1, Basis 1 + T], die Summe der Ausgaben ist fest; Budgetgrenzen und gepacte Kunden geben genau ihr Budget aus."""
    base1, base2, tie = _spends(v, a, b)
    total = base1 + base2 + tie
    lo, hi = base1, base1 + tie                              # erlaubter Bereich für Ausgabe 1
    lo, hi = max(lo, total - budgets[1]), min(hi, budgets[0])  # Ausgabe 1 <= B1 und Ausgabe 2 = total - Ausgabe 1 <= B2
    if a < 1 - tol:
        lo, hi = max(lo, budgets[0]), min(hi, budgets[0])      # gepacter Kunde 1: Ausgabe = B1
    if b < 1 - tol:
        lo, hi = max(lo, total - budgets[1]), min(hi, total - budgets[1])   # gepacter Kunde 2: Ausgabe 2 = B2
    return bool(lo <= hi + tol)
