# Pacing-Gleichgewicht: Auktionen mit Budgets – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-pacing-gleichgewicht-demo.streamlit.app/)**

Drittes Stück der Methodenreihe **Marktmechanismen** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Ein Mechanismus macht aus Geboten, Budgets oder Überzeugungen Preise und eine Zuteilung; die Reihe prüft ihn an Effizienz, Anreizen, Budgetausgleich und Erlös und rechnet ihn als Programm. Das Stück führt die beiden Vorgänger zusammen: den [Fisher-Markt](https://github.com/sebastian-hanisch/fisher-markt-demo) (Preise aus Budgets) und die [Positionsauktion](https://github.com/sebastian-hanisch/positionsauktion-demo) (Gebote und Zahlungsregeln). Vehikel: **Werbekunden einer Frachtbörse mit Tagesbudget, viele einzelne Suchanfragen.** Damit das Budget für den Tag reicht, **bremst** jeder Kunde alle seine Gebote mit einem Faktor (**Pacing**).

## Warum dieses Problem

Wer ein Tagesbudget über viele Auktionen strecken muss, kann nicht einfach seinen Wert bieten: Er würde das Geld vormittags verbrauchen. Die übliche Lösung ist ein einziger Faktor $\alpha_i \le 1$, mit dem alle Gebote multipliziert werden. Die Kernfrage der Demo: **Welche Faktoren stellen sich ein, ist das Ergebnis eindeutig, und was ändert der Wechsel von Erst- auf Zweitpreis?** Die Antwort verbindet die Reihe: Im Erstpreis ist das Gleichgewicht **eindeutig** und folgt aus demselben Programm wie der Fisher-Markt; im Zweitpreis fand die Demo in jedem gerechneten Markt mindestens eines (die Existenz selbst ist eine Literaturaussage), aber **unter Umständen mehrere** mit verschiedenen Einnahmen (im Handbeispiel das 2,40-Fache, in den Zufallsmärkten von Tabelle 4 im Median das 1,02-Fache).

## Modell

$n$ Kunden mit Budgets $B_i$, $m$ Anfragen (je einmal), Wert $v_{ij} \ge 0$ des Kunden $i$ für Anfrage $j$. Kunde $i$ bietet $\alpha_i v_{ij}$ auf jede Anfrage; der Höchstbietende gewinnt (Gleichstand: beliebige Aufteilung). **Erstpreis:** Er zahlt sein Gebot, der Preis ist $p_j = \max_i \alpha_i v_{ij}$. **Zweitpreis:** Er zahlt das zweithöchste Gebot. **Pacing-Gleichgewicht:** Ausgabe$_i \le B_i$, und $\alpha_i < 1$ nur, wenn Ausgabe$_i = B_i$ (wer Geld übrig hat, bietet voll).

**Erstpreis als Programm** (eigene Herleitung der Form aus den Bedingungen, in der Demo gegen die Definition und gegen SLSQP geprüft): Mit den Ausgaben $b_{ij}$ und $p_j = \sum_i b_{ij}$
$$\min \sum_j p_j \log p_j - \sum_j p_j - \sum_{ij} b_{ij} \log v_{ij} \quad \text{unter} \quad \sum_j b_{ij} \le B_i,\ b \ge 0 .$$
Das ist das Shmyrev-Programm des Fisher-Marktes (Stück 1) mit „Budget höchstens“ statt „genau“. **Binden alle Budgets, ist das Gleichgewicht genau der Fisher-Markt.** Laut Abstract entsprechen die Lösungen des Eisenberg–Gale-Programms für quasi-lineare Fisher-Märkte genau den Erstpreis-Pacing-Gleichgewichten.

**Zweitpreis:** Das Gleichgewicht existiert, ist aber nicht eindeutig, und seine Berechnung ist für viele Kunden schwer (näherungsweise PPAD-vollständig laut Abstract). Die Demo rechnet **zwei Kunden exakt** durch Aufzählung der Fälle: beide ungebremst; einer gebremst (er füllt sein Budget an einem Gleichstandsposten); beide gebremst, mit einem Gleichstandsposten oder **ohne Gleichstand** (jeder Faktor folgt dann allein aus dem eigenen Budget). Trifft das Budget des gebremsten Kunden **genau seine Ausgabe, die in einem Faktorintervall konstant ist** (etwa weil das Budget eine Summe von Geboten ist), gibt es ein ganzes Kontinuum von Gleichgewichten; die Demo versucht, solche entarteten Märkte zu erkennen, und lehnt sie ab (bei Zufallsbudgets kommen sie nicht vor).

Die Märkte sind synthetisch (Zufallswerte aus festen Seeds); es gibt keine Daten echter Kunden oder Börsen.

## Methodik

- **Erstpreis:** projizierter Gradient (Armijo) auf dem Programm mit verdoppelter Schrittzahl, bis das **Polieren** gelingt: Aus dem Träger der genäherten Lösung wird ein lineares System in Ausgaben, Preisen und Faktoren gelöst und gegen die **Definition** geprüft (Preis = höchstes Gebot, nur Höchstbietende geben aus, Spaltensummen = Preis, Ausgabe ≤ Budget, $\alpha < 1$ nur bei ausgeschöpftem Budget). Dasselbe Verfahren mit „Zeilensumme = Budget“ liefert das Fisher-Gleichgewicht zum Vergleich.
- **Budgetmaßstab:** Dasselbe Markt-Gerüst (Werte und Budgetgewichte aus dem Seed) mit Budgets $s$ mal Basis (Summe der Höchstwerte durch $n$, mit Zufallsgewichten), zwölf Maßstäbe von 0,05 bis 4. Einnahme und Wohlfahrt (Summe der Werte der gewonnenen Anteile) relativ zur Einnahme ohne Budgets (Summe der Höchstwerte).
- **Zweitpreis, zwei Kunden:** Aufzählung der Fälle (siehe Modell), jedes gefundene Gleichgewicht wird durch eine **eigene Intervallprüfung** bestätigt (gibt es eine Aufteilung der Gleichstandsposten, die Budgets und Pacing-Bedingung erfüllt?).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen sind deterministisch (feste Seeds) und stehen in `tests/test_claims.py`; Preise, Einnahmen und Wohlfahrt auf drei Stellen (das Polieren benutzt `lstsq`, Zahlen am Rundungsboden werden nicht berichtet).

**Tabelle 1: Handbeispiele** (Erstpreis, von Hand nachrechenbar).

| Beispiel | Faktoren α | Preise | Ausgaben je Kunde | Einnahme |
|---|---|---|---|---|
| 1 Kunde, 1 Anfrage, Wert 2, Budget 1 | 0,500 | 1,000 | 1,000 | 1,000 |
| 1 Kunde, 1 Anfrage, Wert 2, Budget 3 | 1,000 | 2,000 | 2,000 | 2,000 |
| 2 Kunden, 1 Anfrage, Werte 4 und 2, Budgets 1 und 1 | 0,500; 1,000 | 2,000 | 1,000; 1,000 | 2,000 |

Im dritten Beispiel bieten beide 2 (Kunde 1 bremst auf 0,5): Gleichstand, jeder gibt sein Budget 1 aus, der Preis ist 2.

**Standardmarkt** (4 Kunden, 8 Anfragen, Maßstab 1,2, Seed 5): **3 von 4** Kunden bremsen, die Einnahme ist **6,577** (**89,3 %** der Einnahme ohne Budgets), die Wohlfahrt **98,8 %**; die Preise weichen um bis zu **23 %** (relativ, größte Abweichung einer Anfrage) vom Fisher-Gleichgewicht ab, weil nicht alle Budgets binden.

**Tabelle 2: Budgetmaßstab** (derselbe Standardmarkt mit Maßstab $s$; Einnahme und Wohlfahrt relativ zur Einnahme ohne Budgets).

| Maßstab s | Einnahme | Wohlfahrt | bremsende Kunden | alle Budgets binden |
|---|---|---|---|---|
| 0,050 | 4,3 % | 97,2 % | 100 % | ja |
| 0,074 | 6,4 % | 97,2 % | 100 % | ja |
| 0,111 | 9,6 % | 97,2 % | 100 % | ja |
| 0,165 | 14,3 % | 97,2 % | 100 % | ja |
| 0,246 | 21,3 % | 97,2 % | 100 % | ja |
| 0,366 | 31,7 % | 97,2 % | 100 % | ja |
| 0,546 | 47,3 % | 97,2 % | 100 % | ja |
| 0,813 | 70,5 % | 97,2 % | 100 % | ja |
| 1,211 | 89,5 % | 98,8 % | 75 % | nein |
| 1,803 | 99,0 % | 99,8 % | 25 % | nein |
| 2,686 | 100,0 % | 100,0 % | 0 % | nein |
| 4,000 | 100,0 % | 100,0 % | 0 % | nein |

Solange **jedes** Budget bindet (die ersten acht Zeilen), ist die **Wohlfahrt konstant** (97,2 %): Die Zuteilung ist die des Fisher-Marktes und hängt nicht vom Maßstab ab, nur die Preise wachsen mit den Budgets; die Preise sind dort gleich dem Fisher-Gleichgewicht (Test: Abweichung unter 1e-7, zusätzlich gegen das Eisenberg–Gale-Programm in der Zuteilung). Wo Budgets nicht mehr binden, steigen in diesem Markt Wohlfahrt und Einnahme bis zum budgetfreien Wert 100 % (für die Einnahme ist Monotonie im Budget laut Literatur eine Eigenschaft des Erstpreises, für die Wohlfahrt nicht allgemein gezeigt).

**Einnahme im Budget.** Erhöht man in 100 Zufallsmärkten (2 bis 5 Kunden, 3 bis 8 Anfragen, Maßstab 0,1, 0,3 oder 1) das Budget eines Kunden um 25 %, sinkt die Einnahme in **0 von 100** Paaren (laut Abstract ist Monotonie eine Eigenschaft des Erstpreises).

**Tabelle 3: Zweitpreis gegen Erstpreis im Handbeispiel** (2 Kunden, 3 Anfragen: Werte (3, 1, 6) und (6, 4, 1), Budgets 0,5 und 3,5).

| Gleichgewicht | α₁ | α₂ | Ausgabe 1 | Ausgabe 2 | Einnahme |
|---|---|---|---|---|---|
| Zweitpreis 1 | 0,167 | 1,000 | 0,500 | 1,167 | 1,667 |
| Zweitpreis 2 | 0,889 | 0,444 | 0,500 | 3,500 | 4,000 |
| Zweitpreis 3 | 0,875 | 0,500 | 0,500 | 3,500 | 4,000 |
| **Erstpreis** (eindeutig) | 0,083 | 0,350 | 0,500 | 3,500 | 4,000 |

Im ersten Zweitpreis-Gleichgewicht bremst nur Kunde 1 (auf 1/6) und teilt sich Anfrage 3 mit Kunde 2 (Gleichstand beim Gebot 1); im zweiten bremsen beide (8/9 und 4/9) mit einem Gleichstand bei Anfrage 1; im dritten bremsen beide (7/8 und 1/2) ganz **ohne** Gleichstand. Die Einnahmen liegen zwischen **1,667** und **4,000** (das 2,40-Fache); der Erstpreis hat genau ein Gleichgewicht mit der Einnahme 4,000. Alle Werte lassen sich von Hand nachrechnen (siehe `tests/conftest.py`).

**Tabelle 4: Wie oft gibt es mehrere Zweitpreis-Gleichgewichte?** 100 Zufallsmärkte je Budgetstufe (2 Kunden, 6 Anfragen, Seed 21; Budgets = Stufe mal Zufall 0,5 bis 1,5 mal Summe der Höchstwerte durch 2). „Mehrere“: mehr als ein Gleichgewicht mit verschiedener Einnahme; Einnahme größte / kleinste: Median / Maximum über diese Märkte; die letzten Spalten: Zweitpreis- durch Erstpreis-Einnahme (Median, kleinste und größte Einnahme der Gleichgewichte).

| Budgetstufe | Märkte | ohne Gleichgewicht | mehrere | Einnahme größte / kleinste | Zweitpreis / Erstpreis (kleinste) | (größte) |
|---|---|---|---|---|---|---|
| 0,1 | 100 | 0 | 0 | – | 1,000 | 1,000 |
| 0,3 | 100 | 0 | 8 | 1,02 / 1,07 | 1,000 | 1,000 |
| 1,0 | 100 | 0 | 0 | – | 0,634 | 0,634 |
| 3,0 | 100 | 0 | 0 | – | 0,525 | 0,525 |

In jedem gerechneten Markt gab es mindestens ein Gleichgewicht, mehrere mit verschiedener Einnahme sind in Zufallsmärkten **selten** (0 bis 8 von 100) und kommen dort vor, wo die Budgets teils binden. Bei bindenden Budgets nehmen Zweit- und Erstpreis dasselbe ein (die Budgetsumme, Verhältnis 1,000), bei großen Budgets nimmt der Zweitpreis weniger ein (0,634 und 0,525 der Erstpreis-Einnahme). In **60 von 1 000** Märkten mit ganzzahligen Werten (Werte 1 bis 7, 2 Kunden, 3 Anfragen, Zufallsbudgets 0,5 bis 4,5; Seed 22; viele Gleichstände) gibt es mehrere Gleichgewichte mit verschiedener Einnahme (**6,0 %**).

## Ehrliche Grenzen

- **Multiplikatives Pacing mit einem Faktor je Kunde:** Echte Systeme steuern Gebote anders (Budgetglättung über die Zeit, Gebotsobergrenzen, Zielrenditen). Das Gleichgewicht ist der Endzustand, nicht der Weg dorthin; Dynamiken sind nicht gerechnet.
- **Budget als harte Grenze, bekannte Werte, eine Anfrage je Auktion** (keine Plätze, kein GSP mit Budgets).
- **Zweitpreis nur für zwei Kunden**, per Aufzählung; für mehr Kunden ist das Gleichgewicht laut Literatur schwer zu berechnen (PPAD).
- **Entartete Märkte** (das Budget des gebremsten Kunden trifft genau seine im Faktorintervall konstante Ausgabe) haben ein Kontinuum von Zweitpreis-Gleichgewichten; die Demo versucht, sie zu erkennen, und lehnt sie ab; ob der andere Kunde seine Ausgabe bezahlen kann, wird über das ganze Faktorintervall geprüft (ein früherer Fehler prüfte nur den Mittelpunkt und übersah ein Kontinuum, das nur am unteren Rand des Intervalls leistbar ist; Handbeispiel in `tests/test_edge_cases.py`). Ein erstes Handbeispiel dieser Art (Budget 1 gleich dem Wert 1 von Kunde 2 für Anfrage 3) wurde beim Bau durch die Orakelprüfung als Kontinuum erkannt und ersetzt.
- **Die Häufigkeit mehrerer Gleichgewichte hängt von Erzeuger und Budgetstufe ab** (0 bis 8 % in Zufallsmärkten, 6 % bei ganzzahligen Werten); die Spanne der Einnahmen im Handbeispiel (2,40) ist ein ausgesuchtes Beispiel, kein Durchschnitt.
- **Das Programm ist eine eigene Herleitung** der Form aus den Bedingungen, gedeckt durch Definition und SLSQP, nicht Zeile für Zeile mit der Quelle verglichen.
- **Keine Anreizanalyse:** Ob ein Kunde durch falsche Angaben seines Werts gewinnt, wird nicht gerechnet.
- **Literatur:** Alle Angaben beruhen auf Titel, Abstract und Sekundärquellen, nicht auf Volltexten.

## Verifikation

- **Kern:** Handbeispiele von Hand (ein Kunde, Gleichstand, drei Gleichgewichte); Projektion auf die Budgetmenge; Definitionsprüfer fängt Verstöße ab (alle Gebote voll trotz Budget, falsche Preise, Bremsen ohne ausgeschöpftes Budget).
- **Orakel (anderer Rechenweg):** das Erstpreis-Programm mit **SLSQP**; die **Definition** als eigene Prüffunktion; das **Eisenberg–Gale-Programm** des Fisher-Marktes in der Zuteilung (SLSQP) bei lauter bindenden Budgets; die Zweitpreis-Aufzählung gegen eine **Rastersuche über alle rationalen Faktorenpaare** mit reiner Intervallprüfung der Gleichstandsaufteilung (kleine Märkte mit Budgets mit halben Zahlen; die Aufzählung ist darin vollständig und korrekt, für nicht entartete Märkte). Diese Rastersuche hat beim Bau eine Lücke der ersten Aufzählung gefunden (beide Kunden gebremst ohne Gleichstand), die behoben wurde.
- **Eigenschaftstests:** Definition in jedem Markt, Preis = höchstes Gebot, Eindeutigkeit von zufälligen Startpunkten, Einnahme monoton im Budget und begrenzt durch Budgetsumme und Höchstwerte, Skalierung, Umbenennen, maßstabsunabhängige Zuteilung bei bindenden Budgets, Fisher-Gleichheit, Existenz im Zweitpreis.
- **Randfälle:** kleinste und größte Märkte, Maßstabsgrenzen, gleiche Kunden, winziges Budget, Budget gleich dem freien Preis, entartete Märkte, Laufzeit.
- **Verdikte** als reine Funktionen an handgesetzten Werten; **Presets, Layout, AppTest-Rauchtest** (jeder Button, jeder Regler an Minimum und Maximum), **Footer**, **Laufzeit-Importe**.
- **Behauptungen:** `tests/test_claims.py` rechnet jede Zahl dieser Datei nach.
- **Umfang:** siehe `pytest tests/`; `tools/readme_rows.py` ist kein Test, sondern erzeugt die Tabellen, die `test_claims.py` prüft.
- **Nicht gelaufen:** Mutationstests (seit 2026-10-28 nicht mehr Teil des Ablaufs); die Aussagenprüfung durch unabhängige Prüfer steht noch aus.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Ergebnis, Kernfrage, zwei Experimente, Grenzen, Formeln |
| `pac_algorithm.py` | Kern: Programm, Polieren, Definitionsprüfer, Fisher-Fall, Zweitpreis-Aufzählung |
| `pac_constants.py` | Regler-Grenzen, Seeds, Presets, Handbeispiel |
| `pac_evaluation.py` | Analyse, Kurve über den Budgetmaßstab, Zählungen, Tabellenzeilen, Verdikte (reine Funktionen) |
| `pac_layout.py` | Fester Kopf (Stylesheet) |
| `pac_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button, Auswahlregler |
| `pac_scenario.py` | Märkte (stetig, ganzzahlig), Handbeispiele |
| `pac_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/readme_rows.py` | erzeugt die README-Tabellen über dieselben Funktionen wie die App |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur für SLSQP-Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Mini-Märkte, Orakel, Eigenschaften, Randfälle, Claims, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Dynamiken der Budgetglättung**, Lernen der Faktoren und **ROI-Vorgaben.**
- **Zweitpreis mit mehr als zwei Kunden** (PPAD-schwer laut Abstract) und die Maximierung von Einnahme oder Wohlfahrt über die Gleichgewichte (NP-schwer laut Abstract).
- **Mehrere Plätze je Anfrage** (GSP mit Budgets) und **Anreizanalyse.**
- **Echte Auktionsdaten.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die CI läuft mit Python 3.12; für die Streamlit Community Cloud ist die Python-Version im Dashboard einzustellen (3.12 empfohlen).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Marktmechanismen: von Preisen bis Marktmachern](https://sebastianhanisch.net/konzepte-marktmechanismen.html).
