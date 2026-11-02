"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import pac_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.28), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def buyer_labels(n: int) -> list[str]:
    return [f"Kunde {i + 1}" for i in range(n)]


def query_labels(m: int) -> list[str]:
    return [f"Anfrage {j + 1}" for j in range(m)]


def build_spend(spend, prices) -> go.Figure:
    """Ausgaben je Kunde und Anfrage (Heatmap, Zahlen in den Zellen); unter jeder Spalte der Preis der Anfrage."""
    n, m = spend.shape
    cols = [f"{j + 1}<br>p = {p:.3f}" for j, p in enumerate(prices)]
    fig = go.Figure(go.Heatmap(
        z=spend, x=cols, y=buyer_labels(n), colorscale=[[0, "#f4f1ea"], [1, C.COLOR_FREE]], zmin=0, showscale=True, colorbar=dict(title="Ausgabe"),
        texttemplate="%{z:.2f}" if n * m <= 60 else None, hovertemplate="%{y}, Anfrage %{x}<br>Ausgabe %{z:.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Anfrage")
    fig.update_layout(height=max(260, 34 * n + 130), margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_alpha(alpha, spend, budgets) -> go.Figure:
    """Pacing-Faktor je Kunde (1 = bietet seinen Wert, darunter = bremst) mit der Budgetausschöpfung als Zahl."""
    n = len(alpha)
    used = spend.sum(axis=1) / budgets
    colors = [C.COLOR_PACED if a < 1 - 1e-9 else C.COLOR_FREE for a in alpha]
    fig = go.Figure(go.Bar(x=buyer_labels(n), y=alpha, marker_color=colors, text=[f"Budget zu {100 * u:.0f} %" for u in used], textposition="outside",
                           hovertemplate="%{x}<br>Faktor %{y:.4f}<extra></extra>"))
    fig.update_yaxes(title="Pacing-Faktor α", range=[0, 1.18])
    return _base(fig, 300)


def build_scale_curve(rows) -> go.Figure:
    """Einnahme und Wohlfahrt (relativ zur Einnahme ohne Budgets) und Anteil bremsender Kunden über den Budgetmaßstab (logarithmisch)."""
    x = [r["scale"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[r["revenue_ratio"] for r in rows], mode="lines+markers", name="Einnahme", line=dict(color=C.COLOR_REV, width=2)))
    fig.add_trace(go.Scatter(x=x, y=[r["welfare_ratio"] for r in rows], mode="lines+markers", name="Wohlfahrt", line=dict(color=C.COLOR_WELF, width=2)))
    fig.add_trace(go.Scatter(x=x, y=[r["paced_share"] for r in rows], mode="lines+markers", name="Anteil bremsender Kunden", line=dict(color=C.COLOR_SHARE, width=2, dash="dot")))
    binding = [r["scale"] for r in rows if r["all_bind"]]
    if binding:
        fig.add_vrect(x0=min(binding) / 1.1, x1=max(binding) * 1.05, fillcolor=C.COLOR_FISHER, opacity=0.12, line_width=0,
                      annotation_text="alle Budgets binden = Fisher-Markt", annotation_position="top left")
    fig.update_xaxes(type="log", title="Budgetmaßstab s")
    fig.update_yaxes(title="relativ zur Einnahme ohne Budgets", range=[0, 1.1])
    return _base(fig, 380)


def build_sppe(equilibria, fp_revenue: float) -> go.Figure:
    """Einnahme je Zweitpreis-Gleichgewicht (Balken) und die des eindeutigen Erstpreis-Gleichgewichts (waagerechte Linie)."""
    labels = [f"Gleichgewicht {i + 1}<br>α = ({e['alpha1']:.2f}; {e['alpha2']:.2f})" for i, e in enumerate(equilibria)]
    fig = go.Figure(go.Bar(x=labels, y=[e["revenue"] for e in equilibria], marker_color=C.COLOR_REV, name="Zweitpreis",
                           text=[f"{e['revenue']:.3f}" for e in equilibria], textposition="outside"))
    fig.add_hline(y=fp_revenue, line=dict(color=C.COLOR_PACED, width=2, dash="dash"), annotation_text="Erstpreis (eindeutig)", annotation_position="top left")
    top = max([e["revenue"] for e in equilibria] + [fp_revenue]) * 1.2
    fig.update_yaxes(title="Einnahme", range=[0, top])
    return _base(fig, 320)
