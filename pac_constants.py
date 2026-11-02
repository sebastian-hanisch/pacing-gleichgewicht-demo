"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import numpy as np

KINDS = ("stetig", "ganzzahlig")
KIND_LABEL = {"stetig": "stetige Werte", "ganzzahlig": "ganzzahlige Werte (Gleichstände)"}
BUDGET_MODES = ("gleich", "ungleich")
BUDGET_LABEL = {"gleich": "alle gleich", "ungleich": "ungleich"}
SPPE_MODES = ("hand", "zufall")
SPPE_LABEL = {"hand": "Handbeispiel (drei Gleichgewichte)", "zufall": "Zufallsmarkt (Seed links)"}
SPPE_LEVELS = (0.1, 0.3, 1.0, 3.0)

# Regler. Komplexität am Maximum: Das Erstpreis-Gleichgewicht ist ein kleines konvexes Programm (n x m Variablen, höchstens GRAD_CAP Gradientenschritte mit Verdopplung); die Kurve über den
# Budgetmaßstab löst SCALE_POINTS Programme; die Zweitpreis-Aufzählung rechnet O(m^2) Fälle (Worst-Case-Test in tests/test_edge_cases.py).
N_MIN, N_MAX, DEFAULT_N = 2, 8, 4
M_MIN, M_MAX, DEFAULT_M = 3, 15, 8
SCALE_MIN, SCALE_MAX, DEFAULT_SCALE, SCALE_STEP = 0.05, 4.0, 1.2, 0.05
SEED_MAX, DEFAULT_SEED = 999999, 5
DEFAULT_KIND, DEFAULT_BUDGETS = "stetig", "ungleich"
INT_VALUE_MAX = 7                                      # ganzzahlige Werte 1 bis 7

SOLVE_START, SOLVE_CAP = 100, 12800                    # Gradientenverfahren: Schrittzahl wird verdoppelt, bis das Polieren gelingt
POLISH_REL = 1e-5                                      # eine Kante gehört zum Träger, wenn die Ausgabe mehr als diesen Anteil des Budgets beträgt
BIND_TOL = 1e-4                                        # ein Budget gilt als bindend, wenn die Ausgabe mindestens (1 - BIND_TOL) B ist
DEF_TOL = 1e-8                                         # Toleranz der Definitionsprüfung
FISHER_TOL = 1e-7                                      # Preise gleich dem Fisher-Gleichgewicht, wenn relativ näher als dies
SCALE_POINTS = 12
SCALE_GRID = tuple(round(float(x), 3) for x in np.geomspace(0.05, 4.0, SCALE_POINTS))

# Zählungen über feste Instanzen (DEMO-PLAYBOOK §4: nie ein einzelner Seed)
SPPE_M = 6                                             # Anfragen im Zweitpreis-Zufallsmarkt und in der Zählung
CENSUS_MARKETS, CENSUS_SEED = 100, 21                  # Märkte je Budgetstufe
INT_CENSUS = 1000                                      # Märkte mit ganzzahligen Werten (2 Kunden, 3 Anfragen, Werte 1 bis 7, Zufallsbudgets 0,5 bis 4,5)
INT_CENSUS_SEED = 22
MONO_PAIRS, MONO_SEED = 100, 23                        # Paare für die Monotonie der Einnahme im Budget
MONO_FACTOR = 1.25
GAP_REL = 1e-9                                         # Einnahmen unterscheiden sich, wenn sie mehr als dies auseinanderliegen

COLOR_PACED = "#c0392b"
COLOR_FREE = "#2F6B65"
COLOR_REV = "#7a6aa8"
COLOR_WELF = "#4c78a8"
COLOR_SHARE = "#e08a2c"
COLOR_FISHER = "#8c6d31"

# Handbeispiel mit drei Zweitpreis-Gleichgewichten (Nutzenmatrix, Budgets); Budgets mit halben Zahlen, damit kein Budget eine Summe von Geboten ist (sonst Kontinuum)
HAND_C = (((3.0, 1.0, 6.0), (6.0, 4.0, 1.0)), (0.5, 3.5))

PRESETS = {
    "Standardfall (Voreinstellung)": {"kind": "stetig", "n": DEFAULT_N, "m": DEFAULT_M, "budgets": DEFAULT_BUDGETS, "scale": DEFAULT_SCALE, "seed": DEFAULT_SEED},
    "Alle Budgets binden: Fisher-Markt": {"kind": "stetig", "n": DEFAULT_N, "m": DEFAULT_M, "budgets": DEFAULT_BUDGETS, "scale": 0.2, "seed": DEFAULT_SEED},
    "Reiche Kunden: kein Pacing": {"kind": "stetig", "n": 3, "m": DEFAULT_M, "budgets": DEFAULT_BUDGETS, "scale": 4.0, "seed": DEFAULT_SEED},
    "Ganzzahlige Werte: Gleichstände": {"kind": "ganzzahlig", "n": DEFAULT_N, "m": DEFAULT_M, "budgets": "gleich", "scale": 0.9, "seed": DEFAULT_SEED},
    "Großer Markt": {"kind": "stetig", "n": N_MAX, "m": M_MAX, "budgets": DEFAULT_BUDGETS, "scale": 1.0, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Vier Werbekunden, acht Anfragen: einige Kunden stoßen an ihr Budget und bieten mit einem Faktor unter 1, andere nicht.",
    "Alle Budgets binden: Fisher-Markt": "Knappe Budgets: Jeder Kunde gibt sein ganzes Budget aus; das Gleichgewicht ist genau der Fisher-Markt aus Stück 1.",
    "Reiche Kunden: kein Pacing": "Große Budgets: Niemand muss bremsen, alle bieten ihren Wert, der Preis ist der höchste Wert.",
    "Ganzzahlige Werte: Gleichstände": "Ganzzahlige Werte erzeugen Gleichstände zwischen Geboten; die Ausgaben werden dort aufgeteilt, die Preise bleiben eindeutig.",
    "Großer Markt": "Acht Kunden und fünfzehn Anfragen bei mittlerem Budget.",
}
