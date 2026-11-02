"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen."""

import pac_constants as C
import pac_evaluation as ev
import pac_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, preset in C.PRESETS.items():
        assert preset["kind"] in C.KINDS and preset["budgets"] in C.BUDGET_MODES, name
        assert C.N_MIN <= preset["n"] <= C.N_MAX and C.M_MIN <= preset["m"] <= C.M_MAX, name
        assert C.SCALE_MIN <= preset["scale"] <= C.SCALE_MAX and 0 <= preset["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_preset_runs_and_is_an_equilibrium():
    for name, preset in C.PRESETS.items():
        a = ev.analyse(ev.Settings(**preset))
        assert a.fppe.ok and a.fisher.ok, name


def test_the_presets_show_what_their_help_says():
    by = {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}
    assert by["Alle Budgets binden: Fisher-Markt"].all_bind and by["Alle Budgets binden: Fisher-Markt"].fisher_gap < C.FISHER_TOL
    assert not by["Reiche Kunden: kein Pacing"].paced.any() and by["Reiche Kunden: kein Pacing"].revenue_ratio == __import__("pytest").approx(1.0)
    mixed = by["Standardfall (Voreinstellung)"]
    assert 0 < mixed.paced.sum() < mixed.market.n                          # einige bremsen, andere nicht
    assert by["Ganzzahlige Werte: Gleichstände"].market.v.max() <= C.INT_VALUE_MAX
    big = by["Großer Markt"]
    assert (big.market.n, big.market.m) == (C.N_MAX, C.M_MAX)
