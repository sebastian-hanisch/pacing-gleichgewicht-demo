"""Gemeinsame Fixtures: die Mini-Märkte, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def single():
    """Ein Kunde, eine Anfrage, Wert 2. Budget 1: Faktor 0,5, Preis 1, Ausgabe 1. Budget 3: Faktor 1, Preis 2, Ausgabe 2."""
    return np.array([[2.0]])


@pytest.fixture
def tie():
    """Zwei Kunden, eine Anfrage, Werte (4; 2), Budgets (1; 1): Kunde 1 bremst auf 0,5, beide bieten 2 (Gleichstand), jeder gibt 1 aus, Preis 2."""
    return np.array([[4.0], [2.0]]), np.array([1.0, 1.0])


@pytest.fixture
def hand():
    """Handbeispiel: v = ((3, 1, 6), (6, 4, 1)), B = (0,5; 3,5). Erstpreis (eindeutig): alpha = (1/12; 7/20), Preise (2,1; 1,4; 0,5), Einnahme 4.
    Zweitpreis, drei Gleichgewichte: alpha = (1/6; 1) (Ausgaben 1/2 und 7/6, Einnahme 5/3), (8/9; 4/9) (Ausgaben 1/2 und 7/2, Einnahme 4), (7/8; 1/2) (Ausgaben 1/2 und 7/2, Einnahme 4)."""
    return np.array([[3.0, 1.0, 6.0], [6.0, 4.0, 1.0]]), np.array([0.5, 3.5])
