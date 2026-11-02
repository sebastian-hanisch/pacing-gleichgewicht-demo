"""Pacing-Gleichgewicht: Auktionen mit Budgets - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 3 der Reihe „Marktmechanismen“: Werbekunden einer Frachtbörse haben ein Tagesbudget und bieten auf viele einzelne Anfragen. Jeder bremst alle seine Gebote mit einem Faktor, bis das Budget
reicht. Im Erstpreis ist das Gleichgewicht eindeutig und folgt aus dem Fisher-Programm von Stück 1; im Zweitpreis kann es mehrere geben.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import pac_constants as C
from pac_evaluation import (
    Settings,
    analyse,
    census_rows,
    hand_rows,
    monotonicity,
    scale_curve,
    scale_rows,
    sppe_census,
    sppe_report,
    sppe_rows,
    verdict_census,
    verdict_equilibrium,
    verdict_monotone,
    verdict_scale,
    verdict_sppe,
)
from pac_layout import dunkles_design, kopf_css
from pac_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from pac_scenario import sppe_market
from pac_visualization import build_alpha, build_scale_curve, build_sppe, build_spend

st.set_page_config(page_title="Pacing-Gleichgewicht – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Suche das Gleichgewicht …")
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Rechne das Gleichgewicht für zwölf Budgetmaßstäbe …")
def _curve(settings):
    return scale_curve(settings)


@st.cache_data(show_spinner=False)
def _hand():
    return hand_rows()


@st.cache_data(show_spinner=False)
def _sppe(mode, seed, level):
    return sppe_report(sppe_market(mode, seed, level))


@st.cache_data(show_spinner="Zähle Zweitpreis-Gleichgewichte über viele Märkte (einige Sekunden) …")
def _census():
    return sppe_census()


@st.cache_data(show_spinner="Prüfe die Einnahme im Budget …")
def _mono():
    return monotonicity()


def _de(v, d=3):
    return f"{v:.{d}f}".replace(".", ",")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in pac_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("💰 Pacing-Gleichgewicht: Auktionen mit Budgets")
st.markdown(
    """
Eine Frachtbörse versteigert **viele einzelne Suchanfragen**; Werbekunden haben je ein **Tagesbudget**. Damit es für den Tag reicht, **bremst** jeder Kunde alle seine Gebote mit **einem Faktor** $\\alpha_i \\le 1$ (**Pacing**):
Er bietet $\\alpha_i$ mal seinen Wert und bremst nur so weit, dass das Budget gerade ausgeschöpft wird. Im **Erstpreis** (der Höchstbietende zahlt sein Gebot) ist das Gleichgewicht **eindeutig**, und es folgt aus demselben Programm wie der Fisher-Markt
aus Stück 1: Binden alle Budgets, *ist* es der Fisher-Markt. Im **Zweitpreis** (der Höchstbietende zahlt das zweithöchste Gebot) gibt es immer ein Gleichgewicht, aber **mehrere sind möglich**, mit sehr verschiedenen Einnahmen.
Drittes Stück der Reihe **Marktmechanismen**; alle Märkte sind synthetisch, es gibt keine Daten echter Kunden.

Weiter unten: zwei Abschnitte **🔬 Experiment**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das Pacing", expanded=True):
    st.markdown(
        """
1. **Der Markt.** $n$ Kunden mit Budgets $B_i$, $m$ Anfragen (je einmal), Wert $v_{ij}$ des Kunden $i$ für Anfrage $j$.
2. **Das Bremsen.** Kunde $i$ bietet $\\alpha_i v_{ij}$ auf jede Anfrage. Wer am höchsten bietet, gewinnt; bei Gleichstand teilen sich die Höchstbietenden die Anfrage.
3. **Das Gleichgewicht.** Jeder Kunde gibt höchstens sein Budget aus, und er bremst ($\\alpha_i < 1$) nur, wenn er sein Budget **ganz** ausgibt: Wer Geld übrig hat, bietet voll.
4. **Erstpreis.** Der Preis einer Anfrage ist das höchste Gebot, $p_j = \\max_i \\alpha_i v_{ij}$. Die Faktoren folgen aus einem konvexen Programm: dem Shmyrev-Programm des Fisher-Marktes mit „Budget höchstens“ statt „genau“.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Markt**")
    kind = st.radio("Werte", C.KINDS, key="kind_radio", format_func=lambda k: C.KIND_LABEL[k], help="Stetige Zufallswerte oder ganzzahlige Werte 1 bis 7 (viele Gleichstände).")
    budgets_mode = st.radio("Budgets", C.BUDGET_MODES, key="budgets_radio", format_func=lambda k: C.BUDGET_LABEL[k], help="Alle Kunden gleich reich oder mit verschiedenen Budgets.")
    n = st.slider("Kunden n", *bounds("n_slider"), key="n_slider", help="Wie viele Werbekunden es gibt.")
    m = st.slider("Anfragen m", *bounds("m_slider"), key="m_slider", help="Wie viele Anfragen versteigert werden.")
    scale = st.slider("Budgetmaßstab s", *bounds("scale_slider"), key="scale_slider", step=C.SCALE_STEP,
                      help="Alle Budgets sind s mal das Basisbudget (Summe der Höchstwerte durch n, mit Zufallsgewichten). Klein: alle Budgets binden. Groß: niemand bremst.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Bestimmt Werte und Budgetgewichte (und den kleinen Markt im Zweitpreis-Abschnitt).")
    st.button("🎲 Neuer Markt", width="stretch", on_click=randomize_seed)

sync_query_params({"kind_radio": kind, "budgets_radio": budgets_mode, "n_slider": int(n), "m_slider": int(m), "scale_slider": round(float(scale), 2), "seed_input": int(seed)})

settings = Settings(kind, int(n), int(m), budgets_mode, round(float(scale), 2), int(seed))
a = _analysis(settings)
mk, res = a.market, a.fppe

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Bremsende Kunden", f"{int(a.paced.sum())} von {mk.n}", delta=f"Faktor α im Mittel {_de(float(res.alpha.mean()), 2)}", delta_color="off")
m2.metric("Einnahme (Erstpreis)", _de(a.revenue, 3), delta=f"{_de(100 * a.revenue_ratio, 1)} % der Einnahme ohne Budgets", delta_color="off")
m3.metric("Wohlfahrt", f"{_de(100 * a.welfare_ratio, 1)} %", delta="Summe der gewonnenen Werte, relativ", delta_color="off")
m4.metric("Abstand zum Fisher-Markt", "0 (alle binden)" if a.all_bind else _de(a.fisher_gap, 2), delta="Preise gleich Stück 1" if a.all_bind else "nicht alle Budgets binden", delta_color="off")
c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(build_spend(res.spend, res.p), width="stretch", key="main_chart")
    st.caption("Ausgaben je Kunde und Anfrage; unter jeder Spalte der Preis. Die Spalten summieren sich zum Preis, die Zeilen zu höchstens dem Budget.")
with c2:
    st.plotly_chart(build_alpha(res.alpha, res.spend, mk.budgets), width="stretch", key="alpha_chart")
    st.caption("Rot: bremst (α < 1), das Budget ist ausgeschöpft. Grün: bietet seinen vollen Wert.")
with st.expander("Drei Handbeispiele zum Nachrechnen"):
    lines = ["| Beispiel | Faktoren α | Preise | Ausgaben je Kunde | Einnahme |", "|---|---|---|---|---|"]
    st.markdown("\n".join(lines + _hand()))
    st.caption("Ein Kunde mit Wert 2 und Budget 1 bietet 1 (α = 0,5) und zahlt 1; mit Budget 3 bietet er 2 und zahlt 2. Im dritten Beispiel bieten beide 2 (Kunde 1 bremst auf 0,5): Gleichstand, jeder gibt 1 aus.")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Erfüllt das die Definition – bremst jeder nur so viel, wie sein Budget verlangt?")
kind_v, text_v = verdict_equilibrium(res.ok and a.fisher.ok, int(a.paced.sum()), mk.n, a.revenue_ratio)
getattr(st, kind_v)(text_v)

st.markdown("---")

# --- Experiment 1: Budgetmaßstab --------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wie wirkt die Größe der Budgets?")
rows = _curve(settings)
st.caption(f"Derselbe Markt (Werte und Budgetgewichte aus dem Seed) mit Budgets s mal Basis, s von {_de(rows[0]['scale'], 2)} bis {_de(rows[-1]['scale'], 2)} (logarithmisch). Einnahme und Wohlfahrt relativ zur Einnahme ohne Budgets.")
st.plotly_chart(build_scale_curve(rows), width="stretch", key="scale_chart")
kind_s, text_s = verdict_scale(rows)
getattr(st, kind_s)(text_s)
st.caption("Solange **jedes** Budget bindet (grau hinterlegt), ist das Gleichgewicht der Fisher-Markt aus Stück 1: Die **Zuteilung** ändert sich mit dem Maßstab nicht (die Wohlfahrt ist waagerecht), nur die Preise wachsen mit den Budgets. "
           "Wo Budgets nicht mehr binden, gewinnen die Kunden mit dem höheren Wert, und Wohlfahrt und Einnahme steigen zum budgetfreien Wert.")
with st.expander("Zahlen zur Kurve"):
    st.markdown("\n".join(["| Maßstab s | Einnahme | Wohlfahrt | bremsende Kunden | alle Budgets binden |", "|---|---|---|---|---|"] + scale_rows(rows)))
    st.caption("Einnahme und Wohlfahrt relativ zur Einnahme ohne Budgets (Summe der Höchstwerte).")
if st.button("🔎 Einnahme im Budget prüfen", key="mono_button", help="100 Zufallsmärkte: ein Kunde bekommt 25 % mehr Budget; sinkt die Einnahme?"):
    st.session_state["mono_done"] = True
if st.session_state.get("mono_done"):
    mono = _mono()
    kind_m, text_m = verdict_monotone(mono["drops"], mono["pairs"])
    getattr(st, kind_m)(text_m)
    st.caption("Zufallsmärkte mit 2 bis 5 Kunden und 3 bis 8 Anfragen, Maßstab 0,1, 0,3 oder 1; ein zufälliger Kunde bekommt 25 % mehr Budget, die Einnahme wird vorher und nachher verglichen.")

st.markdown("---")

# --- Experiment 2: Erstpreis gegen Zweitpreis ------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Erstpreis gegen Zweitpreis – eindeutig oder nicht?")
st.markdown("Im **Erstpreis** gibt es genau ein Gleichgewicht. Im **Zweitpreis** gibt es immer eines, aber unter Umständen mehrere mit sehr verschiedenen Einnahmen; für viele Kunden ist es sogar schwer zu finden (PPAD-vollständig, laut Literatur). "
            "Für **zwei Kunden** lässt es sich exakt aufzählen: Entweder bremst keiner, oder einer (dann an einem Gleichstandsposten), oder beide (mit oder ohne Gleichstandsposten).")
sppe_mode = st.radio("Markt", C.SPPE_MODES, key="sppe_mode", format_func=lambda k: C.SPPE_LABEL[k], horizontal=True)
level = 1.0
if sppe_mode == "zufall":
    level = float(st.radio("Budgetstufe", C.SPPE_LEVELS, index=1, key="sppe_level", horizontal=True, help="Budgets = Stufe mal (Zufall 0,5 bis 1,5) mal Summe der Höchstwerte durch 2."))
rep = _sppe(sppe_mode, int(seed), level)
smk = sppe_market(sppe_mode, int(seed), level)
st.caption("Werte (Zeilen: Kunden, Spalten: Anfragen): " + "; ".join("(" + ", ".join(f"{x:g}" if sppe_mode == "hand" else f"{x:.2f}" for x in row).replace(".", ",") + ")" for row in smk.v)
           + f"; Budgets: {_de(float(smk.budgets[0]), 2)} und {_de(float(smk.budgets[1]), 2)}.")
lines = ["| Gleichgewicht | α₁ | α₂ | Ausgabe 1 | Ausgabe 2 | Einnahme |", "|---|---|---|---|---|---|"]
st.markdown("\n".join(lines + sppe_rows(rep)))
st.plotly_chart(build_sppe(rep["equilibria"], rep["fp_revenue"]), width="stretch", key="sppe_chart")
kind_p, text_p = verdict_sppe([e["revenue"] for e in rep["equilibria"]], rep["fp_revenue"])
getattr(st, kind_p)(text_p)
if sppe_mode == "hand":
    st.caption("Im Handbeispiel (Werte (3, 1, 6) und (6, 4, 1), Budgets 0,5 und 3,5) bremst im ersten Zweitpreis-Gleichgewicht nur Kunde 1 (auf 1/6) und teilt sich Anfrage 3 mit Kunde 2 (Gleichstand beim Gebot 1); im zweiten bremsen beide (8/9 und 4/9) mit einem Gleichstand bei Anfrage 1; "
               "im dritten bremsen beide (7/8 und 1/2) ganz **ohne** Gleichstand: Jeder Faktor folgt allein aus dem eigenen Budget. Der Erstpreis ist eindeutig: Kunde 1 bremst auf 1/12, Kunde 2 auf 0,35.")
if st.button("📊 Mehrere Gleichgewichte über viele Märkte zählen", key="census_button", help=f"{C.CENSUS_MARKETS} Zufallsmärkte je Budgetstufe und {C.INT_CENSUS} Märkte mit ganzzahligen Werten, je 2 Kunden."):
    st.session_state["census_done"] = True
if st.session_state.get("census_done"):
    census = _census()
    kind_c, text_c = verdict_census(census["rows"], census["int_multi"], census["int_total"])
    getattr(st, kind_c)(text_c)
    head = ["| Budgetstufe | Märkte | ohne Gleichgewicht | mehrere (verschiedene Einnahme) | Einnahme größte / kleinste (Median / Maximum) | Zweitpreis / Erstpreis (kleinste) | (größte) |", "|---|---|---|---|---|---|---|"]
    st.markdown("\n".join(head + census_rows(census)))
    st.caption(f"Zufallsmärkte mit 2 Kunden und {C.SPPE_M} Anfragen (Seed {C.CENSUS_SEED}); Zweitpreis / Erstpreis: Median über die Märkte, bei bindenden Budgets 1: beide nehmen die Budgetsumme ein, bei großen Budgets nimmt der Zweitpreis weniger ein. "
               f"Märkte mit ganzzahligen Werten: Werte 1 bis {C.INT_VALUE_MAX}, 3 Anfragen, Zufallsbudgets 0,5 bis 4,5 (Seed {C.INT_CENSUS_SEED}).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Multiplikatives Pacing mit einem Faktor je Kunde** | Echte Systeme steuern Gebote anders (Budgetglättung über die Zeit, Gebotsobergrenzen, Zielrenditen); das Gleichgewicht ist der Endzustand, nicht der Weg dorthin. | – |
| **Budget als harte Grenze, bekannte Werte** | Weiche Budgets (Überziehen gegen Gebühr) sind ein anderes Modell; bei unbekannten Werten entstehen Bayes-Nash-Spiele. | – |
| **Eine Anfrage je Auktion** | Mit mehreren Plätzen je Anfrage kommt GSP mit Budgets ins Spiel (Stück 2 allein rechnet keine Budgets). | Stück 2 (Positionsauktion) |
| **Zweitpreis nur für zwei Kunden** | Mit mehr Kunden ist das Gleichgewicht laut Literatur schwer zu berechnen (PPAD); die Demo rechnet es dort nicht. | – |
| **Keine Anreizanalyse** | Ob ein Kunde durch falsche Angaben seines Werts gewinnt, wird nicht gerechnet. | – |
| **Synthetische Märkte** | Die Häufigkeiten gelten für diese Zufallsmärkte und Budgetstufen, nicht für echte Börsen. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Kunden $i = 1..n$ mit Budget $B_i$, Anfragen $j = 1..m$, Werte $v_{ij} \ge 0$. Faktor $\alpha_i \in (0, 1]$, Gebot $\alpha_i v_{ij}$. Der Höchstbietende gewinnt, bei Gleichstand beliebige Aufteilung. Erstpreis: Preis $p_j = \max_i \alpha_i v_{ij}$;
Zweitpreis: das zweithöchste Gebot.

**Pacing-Gleichgewicht.** Ausgabe$_i \le B_i$, und $\alpha_i < 1 \Rightarrow$ Ausgabe$_i = B_i$.

**Erstpreis als Programm.** Mit den Ausgaben $b_{ij} \ge 0$ und $p_j = \sum_i b_{ij}$:
$$\min \sum_j p_j \log p_j - \sum_j p_j - \sum_{ij} b_{ij} \log v_{ij} \quad \text{unter} \quad \sum_j b_{ij} \le B_i .$$
Die Optimalitätsbedingungen mit den Multiplikatoren $\mu_i \ge 0$ der Budgets sind $\log(p_j / v_{ij}) + \mu_i \ge 0$ mit Gleichheit, wo $b_{ij} > 0$, und $\mu_i = 0$ bei nicht ausgeschöpftem Budget. Mit $\alpha_i = e^{-\mu_i}$ heißt das: $\alpha_i v_{ij} \le p_j$ für alle $j$ mit Gleichheit
auf dem Träger (der Kunde bietet, was die Anfrage kostet) und $\alpha_i = 1$, wenn das Budget nicht bindet. Das ist genau die Erstpreis-Bedingung (eigene Herleitung der Form; in der Demo gegen die Definition und gegen SLSQP geprüft).

**Anschluss an den Fisher-Markt.** Binden alle Budgets ($\sum_j b_{ij} = B_i$), ist $\sum_j p_j = \sum_i B_i$ konstant, und das Programm ist das Shmyrev-Programm aus Stück 1: Preise und Zuteilung sind die des Fisher-Marktes.

**Polieren.** Aus dem Träger einer genäherten Lösung ein lineares System in $(b, p, \alpha)$: $p_j = \alpha_i v_{ij}$ auf den Kanten, Spaltensummen $p_j$, Zeilensumme $B_i$ bei bindendem Budget, sonst $\alpha_i = 1$.

**Zweitpreis, zwei Kunden.** Faktoren $(a, b)$; ohne Gleichstände ist die Ausgabe stückweise konstant, ein Gleichgewicht mit gebremstem Kunden braucht deshalb einen Gleichstandsposten ($a v_{1j} = b v_{2j}$). Die Fälle: $(1,1)$; $(a, 1)$ mit $a = v_{2j}/v_{1j}$; $(1, b)$ mit $b = v_{1j}/v_{2j}$;
beide unter 1 mit $a/b = v_{2j}/v_{1j}$ und zwei linearen Gleichungen für das Niveau $b$ und den Anteil des Gleichstandspostens. Die Aufzählung ist endlich.

**Literatur** (nur Titel, Abstract und Sekundärquellen geprüft, keine Volltexte). Conitzer, Kroer, Panigrahi, Schrijvers, Stier-Moses, Sodomka, Wilkens: *Pacing Equilibrium in First Price Auction Markets.* Management Science 68(12), 2022, S. 8515–8535.
Conitzer, Kroer, Sodomka, Stier-Moses: *Multiplicative Pacing Equilibria in Auction Markets.* Operations Research 70(2), 2022, S. 963–989.
Chen, Kroer, Kumar: *The Complexity of Pacing for Second-Price Auctions.* Mathematics of Operations Research, 2024.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
