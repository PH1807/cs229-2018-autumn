# CS229 -- Verstaendnisfragen und Antworten

Strukturiert nach Thematik. Wird laufend erweitert.

---

## 1. Logistic Regression -- Loss-Funktion und Gradient

### F: Warum gilt die Umrechnung zwischen Gradient und Loss-Funktion?

Der Gradient der logistischen Regression ist:
```
nabla J(theta) = -(1/m) * sum  y^(i) * x^(i) / (1 + exp(y^(i) * theta^T x^(i)))
```

Behauptung: Dieser Gradient gehoert zur Loss-Funktion:
```
ell(theta) = -(1/m) * sum  log(1 / (1 + exp(-y^(i) * theta^T x^(i))))
           =  (1/m) * sum  log(1 + exp(-y^(i) * theta^T x^(i)))
```

Nachrechnen durch Ableiten von ell:

```
d/d(theta) log(1 + exp(-y * theta^T x))
= 1/(1 + exp(-y*theta^T*x)) * exp(-y*theta^T*x) * (-y*x)
```

Vereinfachung des Vorfaktors:
```
exp(-a) / (1 + exp(-a))  =  1 / (exp(a) + 1)  =  1 / (1 + exp(a))
```

Einsetzen:
```
= -y*x / (1 + exp(y * theta^T x))
```

Gesamt:
```
nabla ell = -(1/m) * sum  y^(i) * x^(i) / (1 + exp(y^(i) * theta^T x^(i)))
```

Das stimmt mit dem angegebenen Gradienten ueberein.

---

### F: Was bedeutet "ell"?

`ell` ist der Name fuer die Loss-Funktion, abgeleitet vom Buchstaben **l** (kleines L, kursiv), der in der Mathematik haeufig fuer "Loss" oder "Likelihood" steht. In diesem Kontext ist es die negative Log-Likelihood der logistischen Regression, die minimiert wird.

---

## 2. Lineare Trennbarkeit und Konvergenz

### F: Warum gilt, dass ein Dataset linear trennbar ist, wenn y^(i) * theta^T x^(i) > 0 fuer alle i?

Die Entscheidungsgrenze ist die Menge aller x mit `theta^T x = 0`. Ein Punkt liegt auf einer der zwei Seiten je nach Vorzeichen von `theta^T x`.

```
y = +1 und theta^T x > 0  -->  y * theta^T x = (+1)*(positiv) = positiv > 0  (korrekt)
y = -1 und theta^T x < 0  -->  y * theta^T x = (-1)*(negativ) = positiv > 0  (korrekt)
y = +1 und theta^T x < 0  -->  y * theta^T x = (+1)*(negativ) = negativ < 0  (falsch)
y = -1 und theta^T x > 0  -->  y * theta^T x = (-1)*(positiv) = negativ < 0  (falsch)
```

`y^(i) * theta^T x^(i) > 0` bedeutet also genau: Punkt i liegt auf der richtigen Seite der Trenngeraden. Gilt das fuer alle i, trennt theta alle Punkte korrekt -- das ist die Definition von linearer Trennbarkeit.

**Im Skript:** cs229-notes3.pdf, Zeilen 94-103: "if y^(i) (w^T x + b) > 0, then our prediction on this example is correct." Das ist der sogenannte **funktionale Margin**.

---

### F: Warum konvergiert Gradient Descent nicht fuer linear trennbare Datasets?

**Schritt 1: Loss-Funktion hat kein Minimum**

Sei das Dataset trennbar, d.h. es gibt theta* mit `y^(i) * theta*^T x^(i) > 0` fuer alle i. Skaliere: `c * theta*` fuer wachsendes c > 0. Dann gilt:
```
ell(c * theta*) = (1/m) * sum  log(1 + exp(-c * y^(i) * theta*^T x^(i)))
```

Da der Exponent `-c * (positiv)` immer negativer wird:
```
c = 1:    log(1 + exp(-1))  ≈ 0.31
c = 10:   log(1 + exp(-10)) ≈ 0.000045
c → inf:  → 0, aber nie = 0
```

Der Loss sinkt immer weiter, erreicht aber nie sein Infimum 0. Es gibt kein echtes Minimum -- der Gradient wird nie exakt 0.

**Schritt 2: Was Gradient Descent daraus macht**

Da kein endliches Minimum existiert, laeuft `||theta|| → inf`. Das Abbruchkriterium `||theta_{t+1} - theta_t|| < epsilon` wird nie erfullt.

**Schritt 3: Nicht-trennbarer Fall zum Vergleich**

Wenn es mindestens einen falsch klassifizierten Punkt gibt, hilft Skalieren nicht: der Loss dieses Punktes wird durch groesseres c **groesser**, nicht kleiner. Der Loss hat daher ein echtes Minimum bei endlichem theta.

---

### F: Ist das Divergenzproblem generell bei Logistic Regression?

Ja, aber nur wenn das Dataset **linear trennbar** ist. Das ist kein Implementierungsfehler, sondern ein mathematisches Problem: die Log-Likelihood hat kein endliches Maximum.

**Loesungen in der Praxis:**

1. **L2-Regularisierung** (haeufigstes Vorgehen):
   ```
   ell_reg(theta) = ell(theta) + lambda * ||theta||^2
   ```
   Der Regularisierungsterm bestraft grosses `||theta||` und erzwingt ein endliches Minimum.

2. **Early Stopping** -- Training abbrechen bevor theta zu gross wird.

3. **Abnehmende Lernrate**

In scikit-learn ist L2-Regularisierung standardmaessig aktiv (`C=1.0`), weshalb das Problem dort normalerweise nicht auftritt.

---

## 3. SVM und Hinge Loss

### F: Was ist Hinge Loss?

Hinge Loss ist die Loss-Funktion der SVMs. Definition fuer einen Punkt:

```
L(theta, x^(i), y^(i)) = max(0, 1 - y^(i) * theta^T x^(i))
```

Verhalten:
```
y * theta^T x >= 1  -->  Loss = 0       (korrekt, mit ausreichend Abstand)
y * theta^T x = 0   -->  Loss = 1       (auf der Grenze)
y * theta^T x = -1  -->  Loss = 2       (falsch klassifiziert)
```

ASCII-Darstellung:
```
Loss
 |
2|  \
 |   \
1|    \
 |     \____________________
 |
 +--+--+--+--+--> y * theta^T x
   -1  0  1  2
```

**Vergleich mit logistischem Loss:**
```
Logistic Loss:  log(1 + exp(-z))   -- geht nie auf 0, immer positiv
Hinge Loss:     max(0, 1-z)        -- wird exakt 0 fuer z >= 1
```

Deshalb hat das SVM-Konvergenzproblem bei trennbaren Datasets nicht: sobald alle Punkte `y^(i) * theta^T x^(i) >= 1` erfuellen, ist der Loss exakt 0 -- kein weiterer Gradient, kein Wachstum von theta.

---

## 4. Mathematische Notation

### F: Was bedeutet |{i ∈ I_{a,b}}|?

Das ist die **Kardinalitaet** (Anzahl der Elemente) einer Menge:
```
|{i ∈ I_{a,b}}|  =  Anzahl der Indizes i, die in der Menge I_{a,b} enthalten sind
```

Beispiel: wenn `I_{a,b} = {2, 5, 7, 9}`, dann `|I_{a,b}| = 4`.

---

### F: Was bedeutet I{y^(i) = 1}?

Das ist die **Indikatorfunktion**:
```
I{y^(i) = 1}  =  1  falls y^(i) = 1
                  0  sonst
```

Gibt 1 zurueck wenn die Bedingung wahr ist, sonst 0. In Python entspricht das `(y == 1).astype(int)`.

---

## 5. Model Calibration (PS2-2)

### F: Erklaere den Beweis von Aufgabe (b): Impliziert perfekte Kalibrierung perfekte Genauigkeit (und umgekehrt)?

**Antwort: Nein, in beide Richtungen.**

**Richtung 1: Kalibrierung impliziert NICHT Genauigkeit**

Gegenbeispiel: alle y^(i) = 1 (nur positive Klasse).

Fuer den Bucket (a=0.5, b=1) muss bei perfekter Kalibrierung gelten:
```
(Anteil y=1 im Bucket) = (durchschnittliche Vorhersage im Bucket)
linke Seite = 1
```

Da h(x) = sigma(theta^T x) < 1 immer gilt, muss `avg h(x) < 1` sein. Dann ist der rechte Wert < 1, und das Modell sagt fuer manche Punkte h(x) < 0.5 vorher -- also falsch klassifiziert.

**Richtung 2: Genauigkeit impliziert NICHT Kalibrierung**

Perfekte Genauigkeit heisst alle Punkte korrekt klassifiziert:
```
Anteil y=1 im Bucket = 1
```

Aber das Modell koennte trotzdem h(x) = 0.6 vorhersagen:
```
avg P(y=1|x) = 0.6  ≠  1
```

Kalibrierung verletzt, obwohl alle Entscheidungen korrekt sind.

**Kernaussage:** Kalibrierung prueft ob Wahrscheinlichkeiten realistisch sind. Genauigkeit prueft nur ob die Entscheidung (> 0.5 oder nicht) stimmt. Beides ist voneinander unabhaengig.

---

## 6. Kursstruktur

### F: Greifen die PS2-Aufgaben auf Wissen aus spaetere Kapitel vor?

Ja. Uebersicht:

| Problem | Erwartet | Benoetigt zusaetzlich | Hinweis |
|---------|----------|-----------------------|---------|
| PS2-1 | Notes1 (LR) | Notes3 (Hinge Loss/SVM) fuer Teil (d); Notes5 (Regularisierung) fuer Teil (c) | Teil (d) setzt SVM-Konzepte voraus |
| PS2-2 | Notes1 (LR, MLE) | -- | In Notes1 enthalten |
| PS2-3 | Notes1/5 | **Notes6** (Bayesian Statistics) als Hauptquelle | Notes6 kommt deutlich spaeter im Kurs |
| PS2-4 | -- | Notes3 (Kernels, Mercer) vollstaendig | Komplett aus Notes3 |
| PS2-5 | -- | Notes3 (Kernel Trick, Representer Theorem) | Komplett aus Notes3 |
| PS2-6 | Notes2 (NB) | Notes3 (SVM) fuer Teil (d) | Hybrid-Problem |
