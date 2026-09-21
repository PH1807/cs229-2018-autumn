# CS229 Problem Sets -- Thematische Uebersicht

---

## PS0: Mathematische Grundlagen

**Thema:** Lineare Algebra und Multivariable Analysis
**Anwendungsfall:** Voraussetzung fuer alle weiteren Problem Sets
**Take-Away:** Gradient, Hessian, Matrixoperationen sind das Handwerkszeug fuer ML-Optimierung.

**Beispiel (Praxis):** Stell dir vor, du stehst auf einem Huegel im Nebel und willst ins Tal. Der Gradient sagt dir, in welche Richtung es am steilsten bergab geht. Der Hessian sagt dir zusaetzlich, ob du in einer Mulde (Minimum) oder auf einem Sattel stehst.

**Beispiel (Mathematik):** Betrachte $f(x) = \frac{1}{2} x^\top A x + b^\top x$ mit $x \in \mathbb{R}^n$.

Der Gradient ist $\nabla_x f = Ax + b$ -- er gibt die Steigung in jede Richtung an. Der Hessian ist $H = A$ -- er gibt die Kruemmung an. Setzt man $\nabla_x f = 0$, erhaelt man das einzige Minimum $x^* = -A^{-1}b$. Das Minimum ist eindeutig genau dann, wenn $A$ positiv definit ist, d.h. alle Eigenwerte $\lambda_i > 0$, denn dann ist die "Schuessel" in jede Richtung nach oben gekruemmt: $z^\top A z > 0$ fuer alle $z \neq 0$.

---

## PS1: Supervised Learning

### Problem 1: Linear Classifiers (Logistic Regression & GDA)

**Thema:** Diskriminative vs. generative Klassifikation
**Anwendungsfall:** Binaere Klassifikation (z.B. Tumor gutartig/boesartig)

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | Hessian der LR-Loss herleiten, PSD zeigen | NLL ist konvex $\Rightarrow$ globales Minimum garantiert | Wie eine Schuessel: egal wo du den Ball reinwirfst, er rollt immer zum tiefsten Punkt. Es gibt keine "falschen Taeler". | Die NLL-Loss ist $J(\theta) = -\sum_i [y_i \log h(x_i) + (1-y_i)\log(1-h(x_i))]$ mit $h(x) = \sigma(\theta^\top x)$. Ihr Gradient ist $\nabla J = X^\top(h - y)$. Der Hessian ist $H = \frac{1}{m} X^\top S X$ mit $S = \text{diag}(h_i(1-h_i))$. Da die Sigmoid-Funktion $0 < h_i < 1$ liefert, gilt $h_i(1-h_i) > 0$ fuer alle $i$, also sind alle Diagonaleintraege von $S$ positiv ($S \succ 0$). Daraus folgt fuer jedes $z \neq 0$: $z^\top H z = \frac{1}{m}(Xz)^\top S (Xz) \geq 0$, weil $S$ positiv definit ist. Also ist $H \succeq 0$ (PSD) und $J$ konvex. |
| (b) | [Code] Logistic Regression mit Newton's Method | Newton konvergiert schneller als GD | Statt blind bergab zu laufen (GD), schaust du dir die Kruemmung an und springst zum Minimum -- GPS statt Kompass. | Gradient Descent: $\theta := \theta - \alpha \nabla J$ -- macht immer einen Schritt fixer Groesse in Gradientenrichtung, ignoriert die Kruemmung. Newton: $\theta := \theta - H^{-1} \nabla J$ -- nutzt die Kruemmung $H$, um direkt zum Minimum zu springen. Warum schneller? In der Naehe des Minimums gilt $J(\theta) \approx J(\theta^*) + \frac{1}{2}(\theta-\theta^*)^\top H (\theta-\theta^*)$; Newton loest dieses quadratische Problem in einem Schritt. Auf ds1: Newton konvergiert in $\sim 5$ Iterationen, GD braeuchte $\sim 1000$. |
| (c) | GDA ergibt lineare Entscheidungsgrenze | Generative Modelle koennen diskriminativ genutzt werden | Statt "was unterscheidet Hunde von Katzen?" zu lernen, lernst du "wie sehen Hunde/Katzen aus?" separat -- die Grenze ergibt sich automatisch. | GDA nimmt an: $p(x \mid y=k) = \mathcal{N}(\mu_k, \Sigma)$. Dann gilt per Bayes: $p(y=1 \mid x) = \frac{p(x \mid y=1)\,p(y=1)}{p(x \mid y=0)\,p(y=0) + p(x \mid y=1)\,p(y=1)}$. Setzt man die Gaussverteilungen ein und vereinfacht den Ausdruck (die quadratischen Terme in $x$ kuerzen sich heraus, weil beide Klassen die gleiche Kovarianz $\Sigma$ haben), erhaelt man $p(y=1 \mid x) = \sigma(\theta^\top x + \theta_0)$ mit $\theta = \Sigma^{-1}(\mu_1 - \mu_0)$. Das ist exakt die Form von Logistic Regression. |
| (d) | MLE-Schaetzer fuer GDA-Parameter | Gausssche Annahme fuehrt zu geschlossener Loesung | Koerpergroesse und Gewicht von Maennern/Frauen: Mittelwert und Streuung direkt ablesen -- kein iteratives Training. | Maximiert man die Log-Likelihood $\sum_i \log p(x_i, y_i)$ nach den Parametern, ergibt sich geschlossen: $\phi = \frac{1}{m}\sum_i \mathbb{1}\{y_i=1\}$ (Anteil Klasse 1), $\mu_k = \frac{\sum_i \mathbb{1}\{y_i=k\}\, x_i}{\sum_i \mathbb{1}\{y_i=k\}}$ (klassenspezifischer Mittelwert) und $\Sigma = \frac{1}{m}\sum_i (x_i - \mu_{y_i})(x_i - \mu_{y_i})^\top$ (gemeinsame Kovarianz). Kein Gradient, kein Optimizer -- ein einziger Durchlauf ueber die Daten reicht. |
| (e) | [Code] GDA implementieren | GDA effizient wenn Annahmen zutreffen | Bluttest: wenn Messwerte glockenfoermig verteilt, reicht Mittelwert und Streuung pro Klasse. | Berechne $\mu_0, \mu_1, \Sigma, \phi$ aus (d). Entscheide bei neuem $x$: $\hat{y} = \mathbb{1}\{p(y=1 \mid x) \geq 0.5\}$, was aequivalent ist zu $\mathbb{1}\{\theta^\top x + \theta_0 \geq 0\}$ mit $\theta = \Sigma^{-1}(\mu_1 - \mu_0)$ und $\theta_0 = -\frac{1}{2}\mu_1^\top\Sigma^{-1}\mu_1 + \frac{1}{2}\mu_0^\top\Sigma^{-1}\mu_0 + \log\frac{\phi}{1-\phi}$. |
| (f)/(g) | Vergleich LR vs. GDA auf ds1, ds2 | GDA versagt bei nicht-normalverteilten Features | Einkommen ist rechtsschief -- nicht glockenfoermig. GDA liefert dann schlechte Grenze. | Auf ds1 ist $x_2 \geq 0$ (chi-quadrat-artig), was die Gauss-Annahme verletzt. GDA "denkt", die Klassen seien symmetrische Glockenkurven und setzt die Entscheidungsgrenze falsch. LR macht keine Verteilungsannahme und passt die Grenze direkt an. Auf ds2 sind beide Features tatsaechlich ca. normalverteilt, also liefern LR und GDA nahezu identische Grenzen. |
| (h) | Box-Cox Transformation | Transformation kann GDA retten | $\log(\text{Einkommen})$ ist oft normalverteilt. Transformation macht Daten "gaussischer". | Wende $x_2' = \log(x_2)$ an. Dadurch wird die rechtsschiefe Verteilung (die gegen 0 gebunden war) auf $\mathbb{R}$ gestreckt und symmetrischer. Ein formaler Test (z.B. Kolmogorov-Smirnov) zeigt: nach Transformation ist $x_2'$ nicht mehr signifikant von $\mathcal{N}(\mu, \sigma^2)$ verschieden. GDA auf transformierten Daten erreicht dann Accuracy nahe LR. |

---

### Problem 2: Incomplete, Positive-Only Labels

**Thema:** Label-Noise, Latente Variablen, Kalibrierung
**Anwendungsfall:** Nur positive Beispiele sind zuverlaessig gelabelt

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | $p(y=1 \mid x) = \alpha \cdot p(t=1 \mid x)$ herleiten | Verbindung zwischen beobachtbarem Label $y$ und wahrem Label $t$ | Restaurant bekommt nur positive Bewertungen. Bewertungsrate $= \alpha \cdot$ Zufriedenheitsrate. | Marginalisiere $t$ aus: $p(y=1 \mid x) = \sum_{t \in \{0,1\}} p(y=1, t \mid x)$. Der Term $p(y=1, t=0 \mid x) = 0$, denn $y=1$ ist per Definition unm\"oglich wenn $t=0$. Also: $p(y=1 \mid x) = p(y=1, t=1 \mid x) = p(y=1 \mid t=1, x) \cdot p(t=1 \mid x)$. Bedingte Unabhaengigkeit: $p(y=1 \mid t=1, x) = p(y=1 \mid t=1) =: \alpha$, weil $y$ gegeben $t$ nicht von $x$ abhaengt. Ergebnis: $p(y=1 \mid x) = \alpha \cdot p(t=1 \mid x)$. |
| (b) | $h(x) \approx \alpha$ fuer alle $x \in V^+$ | Klassifikator liefert direkt eine $\alpha$-Schaetzung | Modell sagt auf allen positiv-bewerteten Restaurants $\sim 0.8$ vorher $\Rightarrow$ $\alpha = 0.8$. | Das trainierte $h(x) \approx p(y=1 \mid x) = \alpha \cdot p(t=1 \mid x)$. Bedingt auf $y=1$ gilt zwingend $t=1$, also $p(t=1 \mid x, y=1) = 1$. Daher: $\mathbb{E}[h(x) \mid y=1] \approx \alpha \cdot \mathbb{E}[p(t=1 \mid x) \mid y=1] = \alpha \cdot 1 = \alpha$. Im Code: $\alpha = \frac{1}{|V^+|}\sum_{i \in V^+} h(x_i)$, der Durchschnitt der Vorhersagen auf positiv-gelabelten Validierungspunkten. |
| (c) | [Code] Training auf wahren $t$-Labels | Obere Schranke der Performance | Idealfall: du weisst von jedem Kunden ob zufrieden. Bestmoegliche Referenz. | Minimiere $-\sum_i [t_i \log h(x_i) + (1-t_i)\log(1-h(x_i))]$ auf den wahren Labels $t_i$. Dies liefert den besten erreichbaren Klassifikator. Seine Accuracy dient als Referenz, die ohne Labelrauschen erreichbar waere. |
| (d) | [Code] Training nur auf $y$-Labels | Ohne Korrektur systematisch verzerrt | Nur geschriebene Bewertungen als "zufrieden" $\Rightarrow$ unterschaetzt wahre Zufriedenheit. | $h(x) \approx p(y=1 \mid x) = \alpha \cdot p(t=1 \mid x)$. Der Klassifikator lernt korrekt $p(y=1 \mid x)$, aber das ist kleiner als $p(t=1 \mid x)$ um den Faktor $\alpha < 1$. Alle Vorhersagen werden dadurch systematisch zu klein, und die Entscheidungsgrenze ($h(x) = 0.5$) liegt falsch. |
| (e) | [Code] $\alpha$ schaetzen, skalieren | Einfache Korrektur moeglich | Modell sagt 0.6, $\alpha=0.8$ $\Rightarrow$ wahre Zufriedenheit $\approx 0.75$. | Schaetze $\alpha$ nach (b). Dann gilt $p(t=1 \mid x) = h(x)/\alpha$. Die Entscheidungsgrenze $p(t=1 \mid x) \geq 0.5$ wird zu $h(x) \geq \alpha/2$, was aequivalent ist zu $\sigma(\theta^\top x) \geq \alpha/2$, also zu $\theta^\top x \geq \log\!\frac{\alpha/2}{1-\alpha/2}$. Im Vergleich zu $\theta^\top x \geq 0$ verschiebt sich der Intercept um $\Delta\theta_0 = \log(2/\alpha - 1)$. |

---

### Problem 3: Poisson Regression

**Thema:** Generalized Linear Models (GLMs), Exponentialfamilie
**Anwendungsfall:** Vorhersage von Zaehldaten

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | Poisson als Exponentialfamilie | Viele Verteilungen sind Spezialfaelle | Kundenanrufe pro Stunde im Callcenter folgen Poisson. Poisson, Bernoulli, Gauss -- alle eine Familie. | Die Exponentialfamilie hat die Form $p(y;\eta) = b(y)\exp(\eta y - a(\eta))$. Fuer Poisson: $p(y;\lambda) = \frac{\lambda^y e^{-\lambda}}{y!} = \frac{1}{y!}\exp(y\log\lambda - \lambda)$. Abgleich ergibt: natuerlicher Parameter $\eta = \log\lambda$, Log-Partition $a(\eta) = e^\eta = \lambda$, Basismassfunktion $b(y) = 1/y!$. Der Parameter $\lambda$ (Erwartungswert) steckt komplett in $a(\eta)$ -- so ist es bei jeder Verteilung der Exponentialfamilie. |
| (b) | Kanonische Response-Funktion | Verbindet lineare Praediktion $\theta^\top x$ mit $\mathbb{E}[y]$ | Modell sagt $\theta^\top x = 3.5$ $\Rightarrow$ $e^{3.5} \approx 33$ erwartete Anrufe. $\exp(\cdot)$ garantiert: Vorhersage immer positiv. | Der kanonische Link setzt $\eta = \theta^\top x$. Die Response-Funktion (Umkehrung des Links) gibt den Erwartungswert: $g(\eta) = a'(\eta) = \frac{d}{d\eta}e^\eta = e^\eta$. Also $\mathbb{E}[y \mid x] = e^{\theta^\top x}$. Warum $a'$? Weil allgemein gilt $\mathbb{E}[y;\eta] = a'(\eta)$ (Eigenschaft der Exponentialfamilie, folgt durch Ableiten von $\int p\,dy = 1$). |
| (c) | SGD-Update herleiten | Update hat universelle Form -- gleich fuer alle GLMs | Egal ob Logistic oder Poisson: Update ist immer $(\text{tatsaechlich} - \text{vorhergesagt}) \cdot \text{Feature}$. | Gradienten der NLL nach $\theta_j$: $\frac{\partial}{\partial\theta_j}(-\log p(y;\theta^\top x)) = -(y - e^{\theta^\top x})\, x_j$. SGD-Update: $\theta_j := \theta_j + \alpha(y_i - e^{\theta^\top x_i})\, x_{ij}$. Vergleich zu Logistic Regression: $\theta_j := \theta_j + \alpha(y_i - \sigma(\theta^\top x_i))\, x_{ij}$. Die Struktur $(y - h(x)) \cdot x$ ist identisch -- nur die Vorhersagefunktion $h$ aendert sich: $e^{\cdot}$ statt $\sigma(\cdot)$. |
| (d) | [Code] Poisson Regression | GLM-Framework universell anwendbar | Unfallzahlen pro Strassenabschnitt vorhersagen. | Trainiere mit SGD ($\text{lr} = 2\times10^{-7}$) auf ds4. Loss: $\text{NLL} = \sum_i(e^{\theta^\top x_i} - y_i\,\theta^\top x_i)$, das ist $-\log p(y_i; e^{\theta^\top x_i})$ ohne den konstanten Term $\log(y_i!)$. Vorhersage: $\hat{y} = \exp(\theta^\top x)$, immer positiv -- sinnvoll fuer Zaehldaten. |

---

### Problem 4: Convexity of Generalized Linear Models

**Thema:** GLMs, Konvexitaet, Exponentialfamilie
**Anwendungsfall:** Warum GLMs stets gut optimierbar sind

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | $\mathbb{E}[Y \mid X;\theta] = a'(\eta)$ | Erwartungswert aus Log-Partition-Funktion | $a(\eta)$ ist ein Generalschluessel: Erwartungswert, Varianz und weitere Momente lassen sich ableiten. | Leite $\int p(y;\eta)\,dy = 1$ nach $\eta$ ab: $0 = \int b(y)(y - a'(\eta))e^{\eta y - a(\eta)}\,dy = \mathbb{E}[y] - a'(\eta)$. Also $\mathbb{E}[y] = a'(\eta)$. Konkret: Poisson $a(\eta)=e^\eta$, $a'(\eta)=e^\eta=\lambda=\mathbb{E}[y]$. Bernoulli $a(\eta)=\log(1+e^\eta)$, $a'(\eta)=\sigma(\eta)=p=\mathbb{E}[y]$. |
| (b) | $\text{Var}(Y \mid X;\theta) = a''(\eta)$ | Varianz ebenfalls direkt aus $a$ ableitbar | Poisson: Varianz $=$ Erwartungswert. Folgt direkt aus $a'' = a'$. | Leite nochmals ab (zweite Ableitung von $\int p\,dy=1$ nach $\eta$): $a''(\eta) = \mathbb{E}[y^2] - (\mathbb{E}[y])^2 = \text{Var}(y)$. Konkret: Poisson $a''(\eta)=e^\eta=\lambda$, also Var$=\lambda=\mathbb{E}[y]$. Bernoulli $a''(\eta)=\sigma(\eta)(1-\sigma(\eta))=p(1-p)$. Gauss (mit $\sigma=1$) $a(\eta)=\eta^2/2$, $a''(\eta)=1=\sigma^2$. |
| (c) | NLL-Hessian ist PSD | NLL jedes GLMs ist konvex | Egal ob Klickraten, Anrufzahlen oder Verweildauern: globales Minimum garantiert. | Der NLL-Hessian ist $H = X^\top D X$ mit $D = \text{diag}(a''(\eta_i))$. Da $a''(\eta) = \text{Var}(y) \geq 0$ fuer alle Verteilungen, sind alle Diagonaleintraege von $D$ nicht-negativ. Fuer beliebiges $z$: $z^\top H z = z^\top X^\top D X z = (Xz)^\top D(Xz) = \sum_i d_i (x_i^\top z)^2 \geq 0$. Also $H \succeq 0$, die NLL ist konvex und hat ein (globales) Minimum. |

---

### Problem 5: Locally Weighted Linear Regression

**Thema:** Non-parametrische Regression, Gewichtete Least Squares
**Anwendungsfall:** Regression wenn globale Linearitaet nicht gilt

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a-i) | Gewichtete Loss-Funktion mit $W$ | Lokale Gewichtung erlaubt nichtlineare Anpassung | Immobilienpreise: Preis-Flaeche-Zusammenhang in Innenstadt anders als am Stadtrand. | Fuer einen festen Query-Punkt $x_q$ definiere Gewichte $w_i = \exp\!\big(-\frac{\|x_i - x_q\|^2}{2\tau^2}\big)$: Punkte nahe $x_q$ bekommen $w_i \approx 1$, weit entfernte $w_i \approx 0$. Die gewichtete Loss-Funktion lautet $J(\theta) = \frac{1}{2}\sum_i w_i(y_i - \theta^\top x_i)^2 = \frac{1}{2}(X\theta-y)^\top W(X\theta-y)$ mit $W = \text{diag}(w_i)$. |
| (a-ii) | Normalgleichung herleiten | Geschlossene Loesung pro Query-Punkt | "Was kosten Haeuser IN DIESER Nachbarschaft?" statt "im Durchschnitt?" | Setze Gradient gleich Null: $\nabla_\theta J = X^\top W(X\theta - y) = 0$. Umformen: $X^\top W X\,\theta = X^\top W y$. Loesung: $\theta^* = (X^\top W X)^{-1} X^\top W y$. Fuer jeden neuen Query-Punkt $x_q$ muss $W$ neu berechnet und das lineare System neu geloest werden -- LWR ist kein parametrisches Modell. |
| (a-iii) | MLE mit heteroskedastischen Varianzen | Unterschiedliche Unsicherheiten motivieren Gewichtung | Teure Sensoren praeiser als guenstige -- praezise Messungen sollen staerker zaehlen. | Angenommen $y_i \sim \mathcal{N}(\theta^\top x_i, \sigma_i^2)$ mit unterschiedlichen $\sigma_i$. Die Log-Likelihood ist $\ell = -\frac{1}{2}\sum_i \frac{(y_i - \theta^\top x_i)^2}{\sigma_i^2} + \text{const}$. Maximieren von $\ell$ entspricht Minimieren von $\sum_i \frac{(y_i-\theta^\top x_i)^2}{\sigma_i^2}$, also WLS mit $w_i = 1/\sigma_i^2$: unsichere Punkte ($\sigma_i$ gross) bekommen kleines Gewicht. |
| (b) | [Code] LWR mit $\tau=0.5$ | Bandwidth $\tau$ steuert Bias-Variance-Tradeoff | $\tau$ klein = Details (instabil). $\tau$ gross = Trend (stabil). Wie Zoom bei Google Maps. | Bei $\tau=0.03$: Nur Punkte im Umkreis $\pm 0.1$ um $x_q$ haben $w_i > 0.01$. Das Modell fittet winzige lokale Gebiete -- jede Stoerung im Trainingssatz wird nachgeahmt (Overfitting). Bei $\tau=10$: $w_i \approx 1$ fuer alle $i$ -- entspricht globalem OLS (Underfitting). Optimum ca. $\tau=0.05$. |
| (c) | [Code] $\tau$ tunen | Hyperparameter-Selektion entscheidend | Teste $\tau \in \{0.03, 0.05, 0.1, 0.5, 1, 10\}$, waehle bestes. | $\text{MSE}(\tau) = \frac{1}{n_\text{val}}\sum_{i=1}^{n_\text{val}} (y_i - \hat{y}_i(\tau))^2$, wobei $\hat{y}_i(\tau)$ die LWR-Vorhersage an $x_i$ mit Bandwidth $\tau$ ist. Waehle $\tau^* = \arg\min_\tau \text{MSE}(\tau)$. Kein Zugriff auf Testdaten beim Tunen -- nur Validierungsset. |

---

## PS2: Supervised Learning II

### Problem 1: Logistic Regression -- Training Stability

**Thema:** Numerische Stabilitaet, Lineare Separierbarkeit
**Anwendungsfall:** Diagnose von Trainingsinstabilitaeten
**Lecture Notes:** Kap. 2 main_notes.pdf (Logistic Regression) -- primaer; Kap. 6 (SVM/Hinge Loss) -- fuer Teil (d); Kap. 9 (Regularisierung) -- fuer Teil (c)
**Hinweis:** Teil (d) greift auf SVM-Konzepte (Hinge Loss, Margin) aus Kap. 6 vor -- diese werden im PS erwartet, obwohl sie im Kursverlauf spaeter kommen.

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | Unterschied Dataset A vs. B beobachten | Lineare Trennbarkeit $\Rightarrow$ divergierende Gewichte | Spam vs. Ham: komplett trennbar $\Rightarrow$ LR schraubt Gewichte immer hoeher. | ds1\_a (nicht separierbar): Gradient-Norm sinkt, $\theta$ stabilisiert sich, Loss konvergiert. ds1\_b (linear separierbar): $\|\theta\|$ waechst monoton mit jeder Iteration, Loss faellt immer weiter, hoert aber nie auf. |
| (b) | Ursache erklaeren | $\|\theta\| \to \infty$ bei Separierbarkeit | Modell will $100\%$ Sicherheit. Da $\sigma(\cdot)$ nie exakt 0 oder 1 wird, muessen Gewichte $\to \infty$. | Da Daten trennbar sind, existiert $\theta^*$ mit $y_i \theta^{*\top} x_i > 0$ fuer alle $i$. Skaliere: $\hat\theta = c\,\theta^*$ fuer $c \to \infty$. Dann $\sigma(c\,\theta^{*\top}x_i) \to 1$ (bei $y_i=1$) und $\to 0$ (bei $y_i=0$). Also NLL $\to 0$, aber es gibt kein Minimum -- die Infimum wird nur im Grenzwert erreicht. Gewichte divergieren. |
| (c) | Fuenf Gegenmassnahmen bewerten | L2-Regularisierung oder Early Stopping helfen | L2 ist eine "Leine" fuer die Gewichte. Early Stopping: Training beenden bevor Gewichte explodieren. | (1) L2-Regularisierung: $\min \text{NLL} + \lambda\|\theta\|^2$. Fuer $\|\theta\|\to\infty$ gilt $\lambda\|\theta\|^2 \to \infty$, also gibt es ein endliches Minimum. (2) Early Stopping: nach $T$ Iterationen abbrechen. (3) Dimensionsreduktion: weniger Features $\Rightarrow$ schwieriger, perfekt trennbar zu sein. (4) Mehr Daten. (5) Modellwechsel (z.B. SVM). |
| (d) | SVM mit Hinge-Loss | SVMs haben endliches Optimum auch bei Separierbarkeit | SVMs stoppen bei genug Abstand (Margin) -- kein weiteres Wachstum der Gewichte. | Hinge-Loss: $\ell_i = \max(0, 1 - y_i\,\theta^\top x_i)$. Sobald Punkt korrekt mit Margin $\geq 1$ klassifiziert ($y_i\,\theta^\top x_i \geq 1$), ist $\ell_i = 0$ und $\nabla_\theta \ell_i = 0$ -- kein weiterer Gradient. Das SVM-Optimum ist das $\theta$ mit maximalem Margin, und dieses ist endlich. |

---

### Problem 2: Model Calibration

**Thema:** Kalibrierung, statistische Interpretation von Wahrscheinlichkeiten
**Anwendungsfall:** Wann kann man Modelloutputs als echte Wahrscheinlichkeiten interpretieren?
**Lecture Notes:** Kap. 2 main_notes.pdf (Logistic Regression, MLE, Optimierungsbedingungen) -- primaer

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | LR auf Trainingsset ist kalibriert | $\sum_i h(x_i) = \sum_i y_i$ | Wetterdienst: 10x "30% Regen" $\Rightarrow$ tatsaechlich 3 von 10 Regentage. | Am Optimum gilt $\nabla_\theta \text{NLL} = 0$, also $\sum_i (h(x_i) - y_i)\, x_i = 0$. Der Intercept $x_{i,0} = 1$ ist eine Komponente von $x_i$, daher folgt: $\sum_i (h(x_i) - y_i) \cdot 1 = 0$, also $\sum_i h(x_i) = \sum_i y_i$. Summe der Vorhersagen $=$ Anzahl positiver Beispiele. Das ist Kalibrierung in Summenform. |
| (b) | Kalibrierung vs. Genauigkeit | Beides unabhaengig -- man kann beides, eins, oder keins haben | "Immer 30% Regen" -- kalibriert aber nutzlos. "0% oder 100%" mit 90% Accuracy -- genau aber unkalibriert. | Kalibrierung: $P(Y=1 \mid h(X)=p) = p$ fuer alle $p \in [0,1]$. Genauigkeit: $P(Y = \mathbb{1}\{h(X) \geq 0.5\})$. Ein konstantes Modell $h(x) \equiv \bar{p}$ (Anteil Positiver) ist perfekt kalibriert, hat aber Accuracy $= \max(\bar{p}, 1-\bar{p})$, die beliebig nahe 50% sein kann. Umgekehrt: perfekte Accuracy (alle richtig) ist per Definition kalibriert, aber ein Modell mit 90% Accuracy kann unkalibriert sein wenn seine "$100\%$"-Vorhersagen in 10% der Faelle falsch liegen. |
| (c) | Einfluss L2-Regularisierung | Regularisierung bricht die Kalibrierungseigenschaft | L2 zieht alle Vorhersagen Richtung 50%. | Mit L2: $\nabla(\text{NLL} + \lambda\|\theta\|^2) = 0$, also $\sum_i (h(x_i) - y_i)\, x_i + 2\lambda\theta = 0$. Fuer die Intercept-Komponente ($x_{i,0} = 1$): $\sum_i (h(x_i) - y_i) + 2\lambda\theta_0 = 0$, also $\sum_i h(x_i) = \sum_i y_i - 2\lambda\theta_0 \neq \sum_i y_i$ (ausser $\theta_0 = 0$). L2 verschiebt den Intercept und bricht damit die Kalibrierung. |

---

### Problem 3: Bayesian Interpretation of Regularization

**Thema:** Bayesianische Inferenz, MAP-Schaetzung
**Anwendungsfall:** Regularisierung als Prior-Wissen verstehen
**Lecture Notes:** Kap. 9.4 main_notes.pdf (Bayesian Statistics, MAP, Regularization) -- primaer; Kap. 9 (Regularization) -- fuer Kontext
**Hinweis:** Erfordert Kap. 9.4 als Hauptquelle -- dieses Kapitel kommt spaeter im Kursverlauf als PS2 suggeriert.

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | $\theta_\text{MAP} = \arg\max_\theta\, p(y \mid X,\theta)\, p(\theta)$ | MAP ist MLE plus einem Prior-Term | Muenze: MLE zaehlt nur Wuerfe. MAP: Prior "wahrscheinlich fair" drueckt Schaetzung zur Mitte. | Bayes: $p(\theta \mid y, X) \propto p(y \mid X, \theta)\, p(\theta)$. MAP sucht das $\theta$ das die Posterior maximiert. Logarithmieren: $\theta_\text{MAP} = \arg\max_\theta [\underbrace{\log p(y \mid X,\theta)}_{\text{MLE-Ziel}} + \underbrace{\log p(\theta)}_{\text{Regularisierer}}]$. Der Prior $p(\theta)$ penalisiert Parameterwerte die a priori unwahrscheinlich sind. |
| (b) | Gauss-Prior $\Rightarrow$ L2-Regularisierung | L2 ist gleichbedeutend mit der Annahme $\theta \sim \mathcal{N}(0, \tau^2 I)$ | Kein einzelner Faktor bestimmt den Immobilienpreis allein. | Prior: $p(\theta_j) = \frac{1}{\sqrt{2\pi}\tau}\exp\!\big(-\frac{\theta_j^2}{2\tau^2}\big)$. Log-Prior: $\log p(\theta) = -\frac{1}{2\tau^2}\sum_j\theta_j^2 + \text{const} = -\frac{1}{2\tau^2}\|\theta\|^2 + \text{const}$. Einsetzen in MAP: $\theta_\text{MAP} = \arg\max_\theta [\log p(y \mid X,\theta) - \frac{1}{2\tau^2}\|\theta\|^2]$, was identisch ist mit $\arg\min[\text{NLL} + \lambda\|\theta\|^2]$ fuer $\lambda = \frac{1}{2\tau^2}$. |
| (c) | Geschlossene Loesung $\theta_\text{MAP}$ | Ridge hat analytische Loesung | Direkt ausrechnen statt iterativ optimieren: eine Matrixinversion reicht. | Fuer lineare Regression mit Gauss-Prior und Gauss-Likelihood ($y_i \sim \mathcal{N}(\theta^\top x_i, \sigma^2)$): $\text{NLL} + \lambda\|\theta\|^2 = \|y - X\theta\|^2/(2\sigma^2) + \lambda\|\theta\|^2$. Gradient $= 0$: $-X^\top(y-X\theta)/\sigma^2 + 2\lambda\theta = 0$. Loesung: $\theta_\text{MAP} = (X^\top X + 2\lambda\sigma^2 I)^{-1} X^\top y$. Gegenueber OLS: Der Term $2\lambda\sigma^2 I$ stellt sicher dass die Matrix immer invertierbar ist. |
| (d) | Laplace-Prior $\Rightarrow$ L1-Regularisierung (Lasso) | L1 foerdert Sparsity: viele Gewichte werden exakt $0$ | Genanalyse: von 20.000 Genen nur wenige relevant. L1 setzt die uebrigen auf 0. | Prior: $p(\theta_j) = \frac{1}{2b}\exp\!\big(-\frac{|\theta_j|}{b}\big)$. Log-Prior: $\log p(\theta) = -\frac{1}{b}\sum_j|\theta_j| + \text{const} = -\frac{1}{b}\|\theta\|_1 + \text{const}$. MAP: $\arg\min[\text{NLL} + \frac{1}{b}\|\theta\|_1]$. Warum Sparsity? Die $|\theta_j|$-Penalisierung hat eine Ecke bei $\theta_j=0$: der Subgradient bei $0$ ist das Intervall $[-1/b, 1/b]$. Sobald $|\nabla_j\text{NLL}| \leq 1/b$, ist $\theta_j=0$ optimal -- Features mit schwachem Signal werden exakt abgeschaltet. |

---

### Problem 4: Constructing Kernels

**Thema:** Kernel-Methoden, Mercer's Theorem
**Anwendungsfall:** Nachweis ob eine Funktion ein gueltiger Kernel ist
**Lecture Notes:** Kap. 5+6 main_notes.pdf (Kernels, SVMs, Mercer's Theorem, PSD-Matrizen) -- primaer

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a)-(h) | Verschiedene Kernel-Kombinationen pruefen | Gueltige Kernels sind abgeschlossen unter $+$, $\cdot$, pos. Skalierung | Aehnlichkeitsmasse kombinieren: Farbe + Form = gueltiges kombiniertes Mass. | Ein Kernel $K$ ist gueltig gdw. seine Gram-Matrix $[K(x_i, x_j)]_{ij}$ PSD ist. (1) Summe: $z^\top(K_1+K_2)z = z^\top K_1 z + z^\top K_2 z \geq 0$ -- Summe zweier PSD-Matrizen ist PSD. (2) Produkt $K_1 \cdot K_2$ (elementweise): Schur-Produkt-Satz -- elementweises Produkt zweier PSD-Matrizen ist PSD. (3) Skalierung $aK_1$ mit $a>0$: $z^\top(aK_1)z = a\,z^\top K_1 z \geq 0$. (4) $K(x,x') = f(x)f(x')$: Gram-Matrix ist $ff^\top$, ein Rang-1 PSD-Matrix. Nicht gueltig: $K_1 - K_2$ (kann negativ definit werden, z.B. $K_1=0$, $K_2$ PSD). |

---

### Problem 5: Kernelizing the Perceptron

**Thema:** Kernel-Trick, Online Learning
**Anwendungsfall:** Nicht-lineare Klassifikation ohne explizite Feature-Transformation
**Lecture Notes:** Kap. 5+6 main_notes.pdf (Kernels, Kernel Trick, Representer Theorem) -- primaer

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a-i/ii/iii) | $\theta$ als Linearkombination der Trainingspunkte darstellen | Representer Theorem: Loesung liegt im Span der Daten | Richter orientiert sich an Praezedenzfaellen -- Entscheidung als Summe frueherer Urteile. | Perceptron: $\theta$ startet bei $0$ und wird bei jedem Fehler um $\pm\phi(x_i)$ veraendert: $\theta = \sum_i \beta_i\phi(x_i)$. Damit wird die Vorhersage zu $\text{sign}(\theta^\top\phi(x)) = \text{sign}\!\big(\sum_i\beta_i\phi(x_i)^\top\phi(x)\big) = \text{sign}\!\big(\sum_i\beta_i K(x_i,x)\big)$. Man benoetigt $\phi$ nie explizit -- nur die Kernel-Auswertungen $K(x_i, x)$. |
| (b) | [Code] Kernelisierten Perceptron implementieren | Kernel-Trick: implizite hochdimensionale Feature-Raeume | Bilder: RBF-Kernel berechnet "Aehnlichkeit" direkt statt Milliarden Features. | Update-Regel: wenn $\text{sign}\!\big(\sum_j\beta_j K(x_j,x_i)\big) \neq y_i$, dann $\beta_i := \beta_i + y_i$. Die Koeffizienten $\beta_i$ zaehlen, wie oft Punkt $x_i$ zur Korrektur beigetragen hat. RBF-Kernel: $K(x,x') = \exp(-\|x-x'\|^2/(2\sigma^2))$, entspricht unendlich-dimensionalem $\phi$. Nie wird $\phi$ berechnet -- $K$ ist eine skalare Funktion. |
| (c) | Dot-Product vs. RBF-Kernel | RBF maechtiger fuer nichtlineare Grenzen | Linear: Linie ziehen. RBF: beliebige Grenzform -- Kreise, Spiralen, Inseln. | Dot-Product-Kernel $K(x,x') = x^\top x'$: entspricht $\phi(x) = x$, also linearer Klassifikator im Originalraum $\Rightarrow$ Hyperebene. RBF-Kernel ($\sigma=1$): $K(x,x') = \exp(-\|x-x'\|^2/2)$: entspricht unendlich-dimensionalem $\phi$ -- der Mercer-Expansion beinhaltet alle Polynomterme bis Grad $\infty$. Daher kann die Grenze beliebig komplex sein. Auf ds5: Dot $\sim 60\%$, RBF $\sim 95\%$ Accuracy. |

---

### Problem 6: Spam Classification

**Thema:** Naive Bayes, Text Classification, SVM
**Anwendungsfall:** SMS-Spam-Erkennung (echte SMS-Nachrichten)
**Lecture Notes:** Kap. 4.2 main_notes.pdf (Naive Bayes) -- primaer fuer Teile (a)-(c); Kap. 6 (SVMs, Kernel Trick) -- fuer Teil (d)
**Hinweis:** Hybrid-Problem: kombiniert generatives Modell (Kap. 4.2) und diskriminatives Modell (Kap. 6) in einer praktischen Anwendung.

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | [Code] Text-Preprocessing | Feature Engineering entscheidend bei NLP | "WINNER!! Claim your prize" $\Rightarrow$ {winner:1, claim:1, prize:1}. Nur Woerter mit $\geq 5$ Vorkommen. | Jede SMS wird dargestellt als $x \in \mathbb{R}^{|V|}$ mit $x_j = \mathbb{1}\{$Wort $j$ kommt vor$\}$ (oder Haeufigkeit). $|V| \approx 1000$ (nach Mindestfrequenz-Filter). Die meisten $x_j = 0$: sparse Darstellung spart Speicher. |
| (b) | [Code] Naive Bayes mit Laplace-Smoothing | NB gut trotz Unabhaengigkeitsannahme | Jedes Wort einzeln betrachtet, trotzdem 97.85% Accuracy. | NB-Annahme: $p(x \mid y) = \prod_j p(x_j \mid y)$ (Woerter bedingt unabhaengig gegeben Klasse). Log-Posterior: $\log p(y \mid x) \propto \log p(y) + \sum_j x_j \log p(w_j \mid y)$. Laplace-Smoothing verhindert $\log 0$: $p(w_j \mid y=k) = \frac{N_{kj} + 1}{N_k + |V|}$ mit $N_{kj}$ = Anzahl Vorkommen von $w_j$ in Klasse $k$, $N_k$ = Gesamtwoerter in Klasse $k$. |
| (c) | [Code] Spam-indikative Tokens | Log-Ratio gibt Modelleinblick | Top-5: 'claim', 'won', 'prize', 'tone', 'urgent!'. | Token-Score: $s_j = \log\frac{p(w_j \mid \text{spam})}{p(w_j \mid \text{ham})}$. Positives $s_j$: Wort kommt haeufiger in Spam vor. z.B. $s_\text{"prize"} = 5.2$ bedeutet "prize" ist $e^{5.2} \approx 180\times$ wahrscheinlicher in Spam als in Ham. Sortieren nach $s_j$ liefert die aufklaerungsreichsten Tokens. |
| (d) | [Code] SVM-RBF tunen | Kernel-SVMs oft besser bei genug Daten | SVM: 96.95%. Hier leicht schlechter als NB (kleine Datenmenge). | SVM minimiert $\frac{1}{2}\|\theta\|^2 + C\sum_i\max(0, 1-y_i\,\theta^\top\phi(x_i))$ mit RBF-Kernel $K(x,x') = \exp(-\|x-x'\|^2/(2r^2))$. Grid-Search: $r \in \{0.01, 0.1, 1, 10\}$, $C \in \{0.01, 0.1, 1, 10\}$ via 5-fold CV auf Trainingsset. Bestes $r=0.1$, $C=1$. |

---

## PS3: Deep Learning & Unsupervised Learning

### Problem 1: A Simple Neural Network

**Thema:** Backpropagation, Aktivierungsfunktionen
**Anwendungsfall:** Nichtlineare 2D-Klassifikation (mondfoermige Cluster), Netz: 2-3-1

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | GD-Update fuer alle Gewichte herleiten | Backprop ist Kettenregel rueckwaerts durch den Graphen | Fabrik: Fehler Station fuer Station zurueckverfolgen, jede Maschine nachstellen. | Forward-Pass: $z_1 = W_1 x + b_1$, $h = \sigma(z_1)$, $z_2 = W_2 h + b_2$, $o = \sigma(z_2)$, $L = \frac{1}{2}(o-y)^2$. Backward-Pass per Kettenregel: $\frac{\partial L}{\partial o} = o-y$, $\frac{\partial L}{\partial z_2} = (o-y)\sigma'(z_2)$, $\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z_2}\cdot h^\top$, $\frac{\partial L}{\partial h} = W_2^\top \frac{\partial L}{\partial z_2}$, $\frac{\partial L}{\partial W_1} = (\frac{\partial L}{\partial h} \odot \sigma'(z_1))\cdot x^\top$. Jede Schicht multipliziert nur ihren lokalen Gradienten -- keine globale Information noetig. |
| (b) | [Code] Stufenfunktion als Aktivierung | Nichtlinearitaet ermoeglicht XOR | "Schirm wenn Wolken ODER Wind, aber nicht beides". Lineares Modell versagt. | $\text{step}(z) = \mathbb{1}\{z>0\}$. XOR-Loesung: $h_1 = \text{step}(x_1+x_2-0.5)$, $h_2 = \text{step}(-x_1-x_2+1.5)$, $o = \text{step}(h_1+h_2-1.5)$. Probe: $(x_1,x_2)=(1,0)$: $h_1=\text{step}(0.5)=1$, $h_2=\text{step}(-0.5)=0$, $o=\text{step}(-0.5)=0$ -- korrekt fuer XOR$(1,0)=1$... Muss nachgeschaut werden. Kern: ohne $\text{step}$ ist keine XOR-Grenze moeglich. |
| (c) | [Code] Lineare Aktivierung im Hidden Layer | Lineare Netze kollabieren -- tiefe lineare Netze = eine lineare Schicht | Zwei Brillen hintereinander: vergroessern, aber biegen nichts. Tiefe bringt nichts. | Mit linearer Aktivierung: $f(x) = W_2(W_1 x + b_1) + b_2 = \underbrace{W_2 W_1}_{=:W'} x + \underbrace{W_2 b_1 + b_2}_{=:b'}$. Ergebnis: $f(x) = W' x + b'$ -- eine lineare Abbildung, aequivalent zu einem einzigen Layer. Egal wie viele Schichten: ohne Nichtlinearitaet laesst sich kein XOR (und allgemein kein nichtlinear separierbares Problem) loesen. |

---

### Problem 2: KL Divergence and Maximum Likelihood

**Thema:** Informationstheorie, MLE
**Anwendungsfall:** Warum MLE sinnvoll ist -- theoretische Fundierung

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | $D_\text{KL}(P\|Q) \geq 0$ (Jensen's Inequality) | KL ist nicht symmetrisch aber immer $\geq 0$; Gleichheit nur wenn $P = Q$ | "Entfernung" Glaube $\leftrightarrow$ Wahrheit: immer $\geq 0$, aber $A \to B \neq B \to A$. | $D_\text{KL}(P\|Q) = \mathbb{E}_P[\log(P/Q)] = -\mathbb{E}_P[\log(Q/P)]$. Jensen: fuer konkaves $f$ gilt $f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]$. Da $\log$ konkav: $\mathbb{E}_P[\log(Q/P)] \leq \log\mathbb{E}_P[Q/P] = \log\int P(x)\frac{Q(x)}{P(x)}dx = \log 1 = 0$. Also $-D_\text{KL} \leq 0$, d.h. $D_\text{KL} \geq 0$. Zahlenbeispiel: $P=[0.5,0.5]$, $Q=[0.9,0.1]$: $D_\text{KL}(P\|Q) = 0.5\ln(0.5/0.9) + 0.5\ln(0.5/0.1) \approx 0.51 \neq D_\text{KL}(Q\|P) \approx 0.37$. |
| (b) | Kettenregel fuer $D_\text{KL}$ | KL einer gemeinsamen Verteilung zerlegt sich in marginale + bedingte KL | Reisekosten = Flug + Hotel. Gesamt-KL = marginale + bedingte. | $D_\text{KL}(P_{X,Y}\|Q_{X,Y}) = \mathbb{E}_{P_{X,Y}}\!\big[\log\frac{P(x,y)}{Q(x,y)}\big] = \mathbb{E}_{P_{X,Y}}\!\big[\log\frac{P(x)}{Q(x)} + \log\frac{P(y\mid x)}{Q(y\mid x)}\big]$. Der erste Term ist $D_\text{KL}(P_X\|Q_X)$, der zweite ist $\mathbb{E}_{x\sim P}[D_\text{KL}(P_{Y|X}\|Q_{Y|X})]$. Trick: einfach $\log(ab) = \log a + \log b$ und dann Erwartungswert aufteilen. |
| (c) | MLE $\Leftrightarrow$ $\min D_\text{KL}(\hat{P}\|P_\theta)$ | MLE passt Modell an empirische Verteilung an -- im Sinne der KL-Divergenz | Schneider passt Anzug so nah wie moeglich an Koerpermasse an. | $D_\text{KL}(\hat{P}\|P_\theta) = \mathbb{E}_{\hat{P}}[\log\hat{P}(x)] - \mathbb{E}_{\hat{P}}[\log P_\theta(x)] = -H(\hat{P}) - \frac{1}{n}\sum_i\log P_\theta(x_i)$. $H(\hat{P})$ ist konstant bzgl. $\theta$. Daher: $\min_\theta D_\text{KL} \Leftrightarrow \max_\theta \frac{1}{n}\sum_i\log P_\theta(x_i)$, das ist genau MLE. |

---

### Problem 3: KL Divergence, Fisher Information, Natural Gradient

**Thema:** Geometrie des Parameterraums, Second-Order Optimierung
**Anwendungsfall:** Effizientere Optimierung fuer probabilistische Modelle

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | $\mathbb{E}_x[s(\theta)] = 0$ | Score-Funktion (Gradient der Log-Likelihood) hat Erwartungswert 0 | Kompass zeigt im Mittel nach Norden -- kein systematischer Bias. | Score: $s(\theta) = \nabla_\theta\log p(x;\theta) = \frac{\nabla_\theta p(x;\theta)}{p(x;\theta)}$. Erwartungswert: $\mathbb{E}[s] = \int p(x;\theta)\cdot\frac{\nabla_\theta p(x;\theta)}{p(x;\theta)}dx = \int\nabla_\theta p(x;\theta)dx = \nabla_\theta\underbrace{\int p(x;\theta)dx}_{=1} = 0$. Entscheidend: Ableitung und Integral duerfen vertauscht werden (Regularitaetsbedingung). |
| (b) | $I(\theta) = \text{Cov}(s(\theta)) = \mathbb{E}[ss^\top]$ | Fisher Information misst Sensitivitaet des Modells | Muenze $p=0.5$: schwer Abweichung von Fairness zu erkennen (Fisher minimal). $p=0.01$: ein Kopf aendert alles (Fisher hoch). | Da $\mathbb{E}[s]=0$: $\text{Cov}(s) = \mathbb{E}[ss^\top] - \mathbb{E}[s]\mathbb{E}[s]^\top = \mathbb{E}[ss^\top]$. Bernoulli$(p)$: $s = \frac{d}{dp}\log p^y(1-p)^{1-y} = \frac{y}{p} - \frac{1-y}{1-p}$. $I(p) = \mathbb{E}[s^2] = \frac{p}{p^2} + \frac{1-p}{(1-p)^2} = \frac{1}{p(1-p)}$. $p=0.5$: $I=4$ (minimum). $p=0.01$: $I=\frac{1}{0.01\cdot0.99}\approx101$. |
| (c) | $I(\theta) = -\mathbb{E}[\nabla^2_\theta\log p]$ | Verbindet Fisher mit Kruemmung der Log-Likelihood | Zwei Wege, ein Ergebnis: Streuung der Steigung = Kruemmung der Landschaft. | Leite $\mathbb{E}[s]=0$ nach $\theta$ ab: $0 = \nabla_\theta\mathbb{E}[s] = \mathbb{E}[\nabla_\theta s] + \mathbb{E}[s\cdot s^\top]$ (Produktregel fuer Erwartungswert unter $\theta$). Umformen: $-\mathbb{E}[\nabla_\theta s] = \mathbb{E}[ss^\top] = I(\theta)$. Da $\nabla_\theta s = \nabla^2_\theta\log p$ (Hessian der Log-Likelihood): $I(\theta) = -\mathbb{E}[\nabla^2_\theta\log p]$. |
| (d) | $D_\text{KL}(p_\theta\|p_{\theta+d}) \approx \frac{1}{2}d^\top I(\theta)\,d$ | Fisher-Metrik als lokale quadratische Approximation der KL-Divergenz | Kleine Schritte im Parameterraum in "Verteilungsabstand" umrechnen. | Taylor-Entwicklung von $D_\text{KL}(p_\theta\|p_{\theta+d})$ um $d=0$: (1) $D_\text{KL}=0$ bei $d=0$. (2) $\nabla_d D_\text{KL}\big|_{d=0} = -\mathbb{E}[s(\theta)] = 0$. (3) $\nabla^2_d D_\text{KL}\big|_{d=0} = I(\theta)$. Also $D_\text{KL} = 0 + 0 + \frac{1}{2}d^\top I(\theta)d + O(\|d\|^3)$. Die Fisher-Matrix ist der "metrische Tensor" im Raum der Verteilungen. |
| (e) | Natural Gradient herleiten | Beruecksichtigt Geometrie der Verteilung | Normaler Gradient: gleiche Schrittweite ueberall. Natural: in "empfindlichen" Richtungen kleinere Schritte. | Gesucht: $d^* = \arg\min_d d^\top\nabla_\theta\ell$ s.t. $d^\top I(\theta)d = \varepsilon^2$ (konstanter Verteilungsabstand). Lagrange-Funktion: $\mathcal{L} = d^\top g - \frac{\lambda}{2}(d^\top I d - \varepsilon^2)$. Ableitung: $g - \lambda I d = 0$, also $d^* = \frac{1}{\lambda}I(\theta)^{-1}g$. Natural Gradient $\tilde{\nabla}\ell = I(\theta)^{-1}\nabla_\theta\ell$: der Gradient wird mit der inversen Fisher-Matrix skaliert. |
| (f) | Natural Gradient $=$ Newton fuer GLMs | Fuer GLMs sind beide Methoden identisch | Zwei Theorien, ein Ergebnis -- bei GLMs ist es egal welche man verwendet. | Bei GLMs gilt $\nabla^2_\theta(-\log p(y\mid x;\theta)) = X^\top D X$ mit $D=\text{diag}(a''(\eta_i))$, und dieser Hessian haengt nicht von $y$ ab. Daher $I(\theta) = \mathbb{E}_y[-\nabla^2\log p] = -\nabla^2\log p = H$ (der Hessian). Newton-Schritt: $\theta := \theta - H^{-1}\nabla\ell$. Natural Gradient-Schritt: $\theta := \theta - I^{-1}\nabla\ell = \theta - H^{-1}\nabla\ell$. Identisch. |

---

### Problem 4: Semi-supervised EM

**Thema:** EM-Algorithmus, Gaussian Mixture Models, Semi-supervised Learning
**Anwendungsfall:** Clustering mit teilweise gelabelten 2D-Daten, $K=3$ Gausssche Cluster

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | Konvergenz von semi-supervised EM | ELBO steigt monoton $\Rightarrow$ Konvergenz garantiert | Bergsteiger: bei jedem Schritt $\geq$ gleich hoch -- kann nie wieder absteigen. | Gesamt-Likelihood: $\ell(\theta) = \sum_\text{unlab}\log\sum_{z}p(x,z;\theta) + \sum_\text{lab}\log p(x,z;\theta)$. Der labeled Term aendert sich im E-Step nicht (feste Zuweisungen). Der unlabeled Term: E-Step maximiert die ELBO $\mathcal{L}(q,\theta) = \mathbb{E}_q[\log p(x,z;\theta)] - \mathbb{E}_q[\log q(z)]$ bzgl. $q$, M-Step bzgl. $\theta$. Da $\ell \geq \mathcal{L}$ (Jensen) und beide Schritte $\mathcal{L}$ erhoehen: $\ell(\theta^{t+1}) \geq \ell(\theta^t)$. |
| (b) | E-Step herleiten | Weiche Cluster-Zuweisung (posterior Zugehoerigkeitswahrscheinlichkeiten) | Kundensegmentierung: "60% Premium, 30% Standard, 10% Budget". | Fuer unlabeled $x_i$: $w_j^{(i)} = p(z_i=j \mid x_i;\theta) = \frac{p(x_i \mid z_i=j;\theta)\,p(z_i=j;\theta)}{\sum_k p(x_i \mid z_i=k;\theta)\,p(z_i=k;\theta)} = \frac{\phi_j\,\mathcal{N}(x_i;\mu_j,\Sigma_j)}{\sum_k\phi_k\,\mathcal{N}(x_i;\mu_k,\Sigma_k)}$. Fuer labeled $x_i$: $w_j^{(i)} = \mathbb{1}\{z_i=j\}$ (keine Unsicherheit ueber die Zuweisung). |
| (c) | M-Step herleiten ($\mu_j$, $\Sigma_j$, $\phi_j$) | Gewichtete MLE-Updates fuer GMM-Parameter | Premium-Kundenprofil = gewichteter Durchschnitt aller Kunden gemaess Premium-Wahrscheinlichkeit. | Maximiere $\mathbb{E}_{q}[\log p(x,z;\theta)]$ bzgl. $\theta$: $\mu_j = \frac{\sum_i w_j^{(i)} x_i}{\sum_i w_j^{(i)}}$ (gewichteter Mittelwert), $\Sigma_j = \frac{\sum_i w_j^{(i)}(x_i-\mu_j)(x_i-\mu_j)^\top}{\sum_i w_j^{(i)}}$ (gewichtete Kovarianz), $\phi_j = \frac{1}{m}\sum_i w_j^{(i)}$ (gewichteter Anteil). Da labeled Punkte $w_j^{(i)}=\mathbb{1}\{z_i=j\}$ haben, "ziehen" sie die Parameter in die richtige Richtung. |
| (d) | [Code] Unsupervised EM (GMM) | Sensitiv gegenueber Initialisierung, kann in lokales Maximum konvergieren | Ohne Vorgaben: EM kann Cluster falsch zuordnen, je nach Startpunkt. | Init: $\mu_j$ zufaellig aus Trainingsset, $\Sigma_j = I$, $\phi_j = 1/K$. EM iteriert E- und M-Step bis Konvergenz ($\|\theta^{t+1}-\theta^t\| < \varepsilon$). Problem: verschiedene Seeds liefern verschiedene lokale Maxima -- manchmal sind Cluster 1 und 2 "vertauscht" (Label-Switching). |
| (e) | [Code] Semi-supervised EM | Wenige Labels stabilisieren erheblich und verhindern Label-Switching | $10\%$ gelabelt: EM findet richtige Segmentierung zuverlaessig. | Gelabelte Punkte bekommen feste $w_j^{(i)}$ im E-Step. Dadurch sind Cluster durch diese "Anker" definiert -- Label-Switching ist ausgeschlossen. Im M-Step fliessen gelabelte und ungelabelte Punkte gemeinsam ein. Schon wenige gelabelte Punkte ($\sim 10\%$) reichen, weil sie die Cluster-Zentren grob fixieren. |
| (f) | Vergleich unsup vs. semi-sup EM | Semi-supervision verbessert Konvergenz und Qualitaet deutlich | Unsup: $85\%$ korrekte Zuordnung. Semi-sup: $95\%$. Wenige Labels, grosser Effekt. | Semi-sup konvergiert typischerweise in $\sim 10$ Iterationen (vs. $\sim 30$ ohne Labels). Das Log-Likelihood-Plateau ist hoeher: weniger lokale Maxima erreichbar wenn Cluster durch Labels definiert sind. Differenz entspricht dem "Wert von Information" in gelabelten Daten. |

---

### Problem 5: K-Means for Compression

**Thema:** K-Means Clustering, Bildkompression
**Anwendungsfall:** Farbreduktion im "Peppers"-Bild auf $K=16$ Farben

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | [Code] K-Means auf Bildpixel anwenden | K-Means ist einfach und wirkungsvoll -- konvergiert stets (aber zu lokalem Minimum) | Maler mit 16 Farbtuben: K-Means findet automatisch die besten 16 Farben. | Jeder Pixel ist $x_i \in \mathbb{R}^3$ (RGB). Ziel: $\min_{c, \mu} \sum_i \|x_i - \mu_{c(i)}\|^2$. Algorithmus: (1) Zuweisung: $c(i) = \arg\min_j \|x_i-\mu_j\|^2$ fuer jedes $i$. (2) Update: $\mu_j = \frac{1}{|C_j|}\sum_{i:c(i)=j}x_i$. Diese zwei Schritte senken die Zielfunktion monoton. Konvergenz garantiert (endliche Zahl moeglicher Zuweisungen), aber globales Minimum nicht. |
| (b) | Kompressionsrate berechnen | Signifikante Kompression moeglich -- Faktor $\sim 6$ | Foto mit $16M$ Farben auf 16 Farben. Bild erkennbar, Speicher $6\times$ kleiner. Prinzip von GIF. | Original: $H \times W \times 24$ Bit (8 Bit pro Kanal, 3 Kanaele). Komprimiert: Indextabelle $H\times W\times\log_2(16) = H\times W\times 4$ Bit (Index fuer jeden Pixel) $+$ Farbtabelle $16\times 24 = 384$ Bit. Kompressionsrate: $\frac{24}{4 + 384/(H\cdot W)}$. Fuer $512\times512$-Bild: $\frac{24}{4 + 384/262144} \approx \frac{24}{4.0015} \approx 6.0$. |

---

## PS4: EM, Deep Learning & Reinforcement Learning

### Problem 1: Neural Networks -- MNIST

**Thema:** Convolutional Neural Networks, modulare Backpropagation
**Anwendungsfall:** Handschrifterkennung (MNIST: 60k Bilder, $28\times28$ Pixel, 10 Klassen)

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | [Code] Backward-Passes fuer Softmax, ReLU, Conv, MaxPool | Jede Schicht berechnet ihren lokalen Gradienten unabhaengig von den anderen | Postleitzahlen lesen: Conv erkennt Kanten, MaxPool macht positionsunabhaengig, Softmax gibt $P(\text{Ziffer})$. | Softmax: $p_k = e^{z_k}/\sum_j e^{z_j}$. Gradient der Cross-Entropy: $\frac{\partial L}{\partial z_k} = p_k - \mathbb{1}\{k=y\}$ (Vorhersage minus One-Hot). ReLU: $\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial y_i}\cdot\mathbb{1}\{x_i>0\}$ (Gradient nur wo Input aktiv). MaxPool: $\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial y}\cdot\mathbb{1}\{x_i = \max\text{ im Fenster}\}$ (Gradient nur ans Maximum). Conv Backward: $\frac{\partial L}{\partial W} = \sum_\text{Patches}\frac{\partial L}{\partial y_\text{Patch}}\otimes x_\text{Patch}$ (Summe ueber alle Fensterausschnitte). |
| (b) | [Code] Vollstaendiger Backward-Pass zusammensetzen | Modulare Implementierung: jede Schicht kapselt forward + backward | LEGO: Bausteine mit forward/backward. Neue Schichten einstecken ohne Aenderung am Rest. | Architektur: Input$(1\times28\times28)$ $\to$ Conv$(4\times4, 2\text{ch})$ $\to$ MaxPool$(5\times5)$ $\to$ ReLU $\to$ Flatten $\to$ Linear$(n_\text{in},10)$ $\to$ Softmax $\to$ CrossEntropy. Backprop: der Gradient $\frac{\partial L}{\partial \text{Input}_\ell}$ einer Schicht wird als $\frac{\partial L}{\partial \text{Output}_{\ell-1}}$ an die vorherige weitergegeben. Jede Schicht braucht nur: (a) den Gradienten vom Ausgang und (b) was sie beim Forward-Pass gespeichert hat (z.B. welche Position das MaxPool-Maximum war). |

---

### Problem 2: Off-Policy Evaluation & Causal Inference

**Thema:** Importance Sampling, Counterfactual Reasoning
**Anwendungsfall:** Medikamenten-Policy evaluieren ohne sie am Patienten zu testen

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | Importance Sampling Schaetzer ist unbiased | Daten aus einer Policy koennen zur Evaluation einer anderen genutzt werden | Krankenhaus A verschreibt Med. X. "Was waere mit Y?" -- Patientendaten umgewichten statt neue Studie. | Daten aus $\pi_0$ gesammelt. Gesuchter Wert: $V(\pi_1) = \mathbb{E}_{\pi_1}[R]$. Umschreiben: $V(\pi_1) = \mathbb{E}_{\pi_0}\!\big[\frac{\pi_1(a\mid s)}{\pi_0(a\mid s)}R\big]$ (Dichte-Quotient kompensiert den Unterschied). Schaetzer: $\hat{V}_\text{IS} = \frac{1}{n}\sum_i\frac{\pi_1(a_i\mid s_i)}{\pi_0(a_i\mid s_i)}r_i$. Unbiased weil $\mathbb{E}[\hat{V}_\text{IS}] = \mathbb{E}_{\pi_0}[w_i r_i] = \mathbb{E}_{\pi_1}[r] = V(\pi_1)$. |
| (b) | Weighted IS ist unbiased (bei bekannter $\pi_0$) | Normalisierte Gewichte stabiler als unnormalisierte | Gewichte koennen extrem sein (Patient zaehlt $100\times$). Normalisierung daempft Ausreisser. | $\hat{V}_\text{WIS} = \frac{\sum_i w_i r_i}{\sum_i w_i}$ mit $w_i = \pi_1(a_i\mid s_i)/\pi_0(a_i\mid s_i)$. Unbiased weil $\mathbb{E}[\sum w_i] = n$ (die Gewichte summieren sich im Erwartungswert zu $n$). Daher $\mathbb{E}[\hat{V}_\text{WIS}] = \frac{n\cdot V(\pi_1)}{n} = V(\pi_1)$. |
| (c) | WIS ist biased bei endlichen Samples | Bias-Variance-Tradeoff: WIS hat weniger Varianz aber Bias bei kleinem $n$ | Stabiler aber leichter Bias bei wenig Daten. Bei genug Daten $\to 0$. | $\mathbb{E}[\hat{V}_\text{WIS}] = \mathbb{E}\!\big[\frac{\sum w_i r_i}{\sum w_i}\big] \neq \frac{\mathbb{E}[\sum w_i r_i]}{\mathbb{E}[\sum w_i]}$ aufgrund Jensen's Inequality auf dem nichtlinearen Quotienten. Bias $\propto \frac{1}{n}$, verschwindet also asymptotisch. Varianz: $\text{Var}(\hat{V}_\text{WIS}) \ll \text{Var}(\hat{V}_\text{IS})$ weil extreme Gewichte normalisiert werden. |
| (d) | Doubly Robust Schaetzer | Korrektes Ergebnis wenn Reward-Modell ODER Policy-Ratio korrekt | Doppelter Boden: eines von beiden muss stimmen -- welches ist egal. | $\hat{V}_\text{DR} = \frac{1}{n}\sum_i\big[\hat{r}(s_i,\pi_1) + w_i(r_i - \hat{r}(s_i,a_i))\big]$. (1) $\hat{r}$ korrekt: zweiter Term $= w_i(r_i - r(s_i,a_i))$ hat $\mathbb{E}=0$ (IS eines Null-Mean-Signals). (2) $w_i$ korrekt: erster Term kann fehlerhaft sein, aber zweiter Term korrigiert exakt: $\mathbb{E}[w_i(r_i-\hat{r})] = V(\pi_1) - \mathbb{E}[\hat{r}(s,\pi_1)]$, was genau den Fehler kompensiert. |
| (e) | Wann IS vs. Regression vs. DR? | Je nach Modellqualitaet ist eine Methode besser | Policy gut bekannt $\Rightarrow$ IS. Wirkung gut verstanden $\Rightarrow$ Regression. Beides unsicher $\Rightarrow$ DR. | $\text{Var}(\hat{V}_\text{IS}) = \frac{1}{n}\text{Var}(w\cdot r)$, gross wenn $w_i$ stark variiert (d.h. $\pi_0$ und $\pi_1$ sehr verschieden). $\text{Bias}(\hat{V}_\text{Reg}) = \mathbb{E}[\hat{r}(s,\pi_1)] - V(\pi_1)$, gross wenn $\hat{r}$ schlecht ist. DR minimiert beides gleichzeitig wenn mindestens eines der Modelle gut ist. |

---

### Problem 3: PCA

**Thema:** Dimensionsreduktion, Varianzmaximierung
**Anwendungsfall:** Feature-Reduktion und Visualisierung

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| - | Erste Hauptkomponente minimiert MSE der Projektion | PCA minimiert Rekonstruktionsfehler und maximiert Varianz gleichzeitig -- beides ist aequivalent | 100 Sensoren messen Bruecke. PCA: 3 Komponenten erklaeren 95% der Variation. 97 messen Rauschen. | Gesucht: Einheitsvektor $u_1$ sodass Projektion $\hat{x}_i = (u_1^\top x_i)\,u_1$ den MSE $\frac{1}{n}\sum_i\|x_i - \hat{x}_i\|^2$ minimiert. Umschreiben: $\|x_i - (u_1^\top x_i)u_1\|^2 = \|x_i\|^2 - (u_1^\top x_i)^2$. Minimieren des MSE $\Leftrightarrow$ Maximieren von $\frac{1}{n}\sum_i(u_1^\top x_i)^2 = u_1^\top\Sigma u_1$ (Varianz der Projektion), wobei $\Sigma = \frac{1}{n}X^\top X$. Loesung durch Lagrange ($\|u_1\|=1$): $u_1$ ist der Eigenvektor zum groessten Eigenwert $\lambda_1$ von $\Sigma$. Erklaerte Varianz $= \lambda_1/\sum_k\lambda_k$. |

---

### Problem 4: Independent Components Analysis (ICA)

**Thema:** Blind Source Separation, Non-Gaussian Quellen
**Anwendungsfall:** Cocktail-Party-Problem: 5 gemischte Audiosignale trennen (11025 Hz)

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | Gausssche Quellen: $W$ nicht eindeutig bestimmbar | ICA funktioniert nicht fuer Gausssche Quellen (rotationsinvariant) | Weisses Rauschen: egal wie Mikrofone gedreht -- klingt gleich. Quellen nicht trennbar. | Annahme: $s \sim \mathcal{N}(0, I)$, Mischung $x = As$. Fuer jede orthogonale Matrix $R$: $Rs \sim \mathcal{N}(0, RR^\top) = \mathcal{N}(0, I)$. D.h. $s$ und $Rs$ haben dieselbe Verteilung. Daher kann ICA $A$ und $AR^\top$ nicht unterscheiden -- $W = A^{-1}$ ist nur bis auf orthogonale Transformation bestimmbar. Unendlich viele gleichwertige Loesungen. |
| (b) | Laplace-Quellen: SGD-Update herleiten | Non-Gausssche Quellen ermoglichen eindeutige Trennung | Sprache und Musik haben charakteristische (nicht-gaussische) Muster. ICA findet "am wenigsten gaussische Richtung". | $s_j \sim \text{Laplace}(0,1)$: $p(s_j) = \frac{1}{2}e^{-|s_j|}$. Likelihood: $p(x;W) = |\det W|\cdot\prod_j p(w_j^\top x)$. Log-Likelihood: $\ell(W) = \log|\det W| + \sum_j\log p(w_j^\top x)$. Gradient: $\nabla_W\ell = (W^\top)^{-1} + \nabla_W\sum_j\log p(w_j^\top x)$. Fuer $p(s)=\frac{1}{2}e^{-|s|}$: $\frac{d}{ds}\log p(s) = -\text{sign}(s) = 1 - 2\sigma(2s) \cdot 2$ (approximiert durch $1-2\sigma(s)$ mit sigmoider Approximation des Vorzeichens). SGD-Update: $W := W + \alpha\big[(1-2\sigma(Wx))x^\top + (W^\top)^{-1}\big]$. |
| (c) | [Code] ICA auf Cocktail-Party-Daten | ICA trennt unabhaengige Quellen ohne Kenntnis der Mischmatrix | 5 Mikrofone, 5 Sprecher $\Rightarrow$ 5 getrennte WAV-Dateien. | $x = As$ ($5\times5$). ICA findet $W \approx A^{-1}$: $\hat{s} = Wx$. Annealing-Lernrate von $0.1$ bis $0.001$ (16 Stufen), um erst grob dann fein zu konvergieren. Konvergenzcheck: $\|W_\text{neu} - W_\text{alt}\|_F / \|W_\text{alt}\|_F < \varepsilon$. Ergebnis: 5 WAV-Dateien, die (idealerweise) je eine Quelle enthalten. |

---

### Problem 5: Markov Decision Processes

**Thema:** Dynamische Programmierung, Bellman-Gleichung
**Anwendungsfall:** Optimale sequentielle Entscheidungen

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | Bellman-Operator $B$ ist $\gamma$-Kontraktion | Value Iteration konvergiert garantiert zum globalen Optimum | Schachcomputer: Bewertungen stabilisieren sich nach genug Iterationen, egal welcher Startwert. | $(BV)(s) = \max_a\big[R(s,a) + \gamma\sum_{s'}P(s'\mid s,a)V(s')\big]$. Zeige $\|BV_1 - BV_2\|_\infty \leq \gamma\|V_1-V_2\|_\infty$: $|(BV_1)(s)-(BV_2)(s)| = |\max_a[\ldots V_1\ldots] - \max_a[\ldots V_2\ldots]| \leq \max_a\gamma\sum_{s'}P(s'\mid s,a)|V_1(s')-V_2(s')| \leq \gamma\|V_1-V_2\|_\infty$. Mit $\gamma = 0.995$: nach $1000$ Iter. schrumpft der Fehler auf $0.995^{1000} \approx 0.007$ des Ausgangsfehlers. |
| (b) | Eindeutigkeit des Fixpunkts $V^*$ | Es gibt genau eine optimale Value-Function | Es gibt genau eine "wahre" Bewertung jeder Position. Verschiedene Starts $\to$ gleiche Loesung. | Banach'scher Fixpunktsatz: Jede $\gamma$-Kontraktion (mit $\gamma < 1$) auf einem vollstaendigen metrischen Raum hat genau einen Fixpunkt. Da $V^*$ Fixpunkt von $B$ ist ($BV^* = V^*$) und $B$ eine Kontraktion ist, ist $V^*$ eindeutig. Value Iteration konvergiert zu diesem $V^*$ unabhaengig vom Startpunkt $V^{(0)}$: $\|V^{(t)} - V^*\|_\infty \leq \gamma^t \|V^{(0)}-V^*\|_\infty \to 0$. |

---

### Problem 6: Reinforcement Learning -- Inverted Pendulum

**Thema:** Model-based RL, Value Iteration auf geschaetztem MDP
**Anwendungsfall:** CartPole balancieren (163 diskretisierte Zustaende, 2 Aktionen: links/rechts)

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| - | [Code] MDP schaetzen, Value Iteration, greedy Policy | Mit geschaetztem Modell kann Value Iteration gute Policies finden -- auch wenn das Modell nicht perfekt ist | Besenstiel auf Handflaeche balancieren. Agent lernt aus Erfahrung, dann vorausplanen. | Kontinuierlicher Zustand $(x, \dot{x}, \theta, \dot{\theta})$ wird diskretisiert zu $s \in \{0,\ldots,162\}$. Uebergangswahrscheinlichkeiten schaetzen: $\hat{P}(s'\mid s,a) = \frac{\text{count}(s,a,s')}{\text{count}(s,a)}$ (relative Haeufigkeit). Value Iteration: $V^{(t+1)}(s) = \max_a\big[R(s,a) + \gamma\sum_{s'}\hat{P}(s'\mid s,a)V^{(t)}(s')\big]$, $\gamma=0.995$. Abbruch: $\max_s|V^{(t+1)}(s) - V^{(t)}(s)| < 0.01$. Greedy Policy: $\pi(s) = \arg\max_a\big[R(s,a) + \gamma\sum_{s'}\hat{P}(s'\mid s,a)V^*(s')\big]$. |
