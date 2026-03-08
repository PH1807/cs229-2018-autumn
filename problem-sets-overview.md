# CS229 Problem Sets -- Thematische Uebersicht

---

## PS0: Mathematische Grundlagen

**Thema:** Lineare Algebra und Multivariable Analysis
**Anwendungsfall:** Voraussetzung fuer alle weiteren Problem Sets
**Take-Away:** Gradient, Hessian, Matrixoperationen sind das Handwerkszeug fuer ML-Optimierung.
**Beispiel (Praxis):** Stell dir vor, du stehst auf einem Huegel im Nebel und willst ins Tal. Der Gradient sagt dir, in welche Richtung es am steilsten bergab geht. Der Hessian sagt dir zusaetzlich, ob du in einer Mulde (Minimum) oder auf einem Sattel stehst.
**Beispiel (Mathematik):**

$$f(x) = \frac{1}{2} x^\top A x + b^\top x$$

$$\nabla f = Ax + b, \quad H = A$$

Wenn $A$ positiv definit (alle Eigenwerte $> 0$), hat $f$ genau ein Minimum bei $x^* = -A^{-1}b$.

---

## PS1: Supervised Learning

### Problem 1: Linear Classifiers (Logistic Regression & GDA)

**Thema:** Diskriminative vs. generative Klassifikation
**Anwendungsfall:** Binaere Klassifikation (z.B. Tumor gutartig/boesartig)

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | Hessian der LR-Loss herleiten, PSD zeigen | NLL ist konvex $\Rightarrow$ globales Minimum garantiert | Wie eine Schuessel: egal wo du den Ball reinwirfst, er rollt immer zum tiefsten Punkt. Es gibt keine "falschen Taeler". | $H = \frac{1}{m} X^\top S X$, mit $S = \text{diag}(h_i(1 - h_i))$. Da $0 < h_i < 1$ gilt $s_{ii} > 0$, also $S \succ 0$. Fuer beliebiges $z$: $z^\top H z = \frac{1}{m}(Xz)^\top S(Xz) \geq 0$, also $H \succeq 0$. |
| (b) | [Code] Logistic Regression mit Newton's Method | Newton konvergiert schneller als GD | Statt blind bergab zu laufen (GD), schaust du dir die Kruemmung an und springst zum Minimum -- GPS statt Kompass. | Newton-Update: $\theta := \theta - H^{-1} \nabla_\theta \ell$. Auf ds1 konvergiert Newton in $\sim 5$ Iterationen. GD mit $\alpha=0.01$ braeuchte $\sim 1000$ Iterationen. |
| (c) | GDA ergibt lineare Entscheidungsgrenze | Generative Modelle koennen diskriminativ genutzt werden | Statt "was unterscheidet Hunde von Katzen?" zu lernen, lernst du "wie sehen Hunde/Katzen aus?" separat -- die Grenze ergibt sich automatisch. | $p(y=1 \mid x) = \sigma(\theta^\top x + \theta_0)$ mit $\theta = \Sigma^{-1}(\mu_1 - \mu_0)$. Das ist exakt Logistic Regression -- die Entscheidungsgrenze ist $\theta^\top x + \theta_0 = 0$. |
| (d) | MLE-Schaetzer fuer GDA-Parameter | Gausssche Annahme fuehrt zu geschlossener Loesung | Koerpergroesse und Gewicht von Maennern/Frauen: Mittelwert und Streuung direkt ablesen -- kein iteratives Training. | $\phi = \frac{1}{m}\sum_i \mathbb{1}\{y_i=1\}$, $\mu_k = \frac{\sum_i \mathbb{1}\{y_i=k\}\, x_i}{\sum_i \mathbb{1}\{y_i=k\}}$, $\Sigma = \frac{1}{m}\sum_i (x_i - \mu_{y_i})(x_i - \mu_{y_i})^\top$ |
| (e) | [Code] GDA implementieren | GDA effizient wenn Annahmen zutreffen | Bluttest: wenn Messwerte glockenfoermig verteilt, reicht Mittelwert und Streuung pro Klasse. | Berechne $\mu_0, \mu_1, \Sigma, \phi$ direkt. Entscheidungsgrenze: $\theta^\top x + \theta_0 = 0$. |
| (f)/(g) | Vergleich LR vs. GDA auf ds1, ds2 | GDA versagt bei nicht-normalverteilten Features | Einkommen ist rechtsschief -- nicht glockenfoermig. GDA liefert dann schlechte Grenze. | ds1: $x_2$ nicht-negativ ($\chi^2$-artig), nicht gaussisch $\Rightarrow$ GDA schlecht. ds2: $x_1, x_2 \sim \mathcal{N}(\mu, \sigma^2)$ $\Rightarrow$ LR $\approx$ GDA. |
| (h) | Box-Cox Transformation | Transformation kann GDA retten | $\log(\text{Einkommen})$ ist oft normalverteilt. Transformation macht Daten "gaussischer". | $x_2' = \log(x_2)$. Nach Transformation $x_2' \approx \mathcal{N}(\mu, \sigma^2)$. GDA auf transformierten Daten $\approx$ LR Accuracy. |

---

### Problem 2: Incomplete, Positive-Only Labels

**Thema:** Label-Noise, Latente Variablen, Kalibrierung
**Anwendungsfall:** Nur positive Beispiele sind zuverlaessig gelabelt

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | $p(y=1 \mid x) = \alpha \cdot p(t=1 \mid x)$ herleiten | Verbindung zwischen beobachtbarem und wahrem Label | Restaurant bekommt nur positive Bewertungen (unzufriedene schreiben nichts). Bewertungsrate $= \alpha \cdot$ Zufriedenheitsrate. | $p(y\!=\!1 \mid x) = p(y\!=\!1, t\!=\!1 \mid x) + \underbrace{p(y\!=\!1, t\!=\!0 \mid x)}_{=0} = p(y\!=\!1 \mid t\!=\!1) \cdot p(t\!=\!1 \mid x) = \alpha \cdot p(t\!=\!1 \mid x)$ |
| (b) | $h(x) \approx \alpha$ fuer alle $x \in V^+$ | Klassifikator liefert $\alpha$-Schaetzung | Modell sagt auf positiv-bewerteten Restaurants $\sim 0.8$ vorher $\Rightarrow$ $\alpha = 0.8$. | $\mathbb{E}[h(x) \mid y=1] = \alpha$, da $h(x) \approx p(y=1 \mid x) = \alpha \cdot p(t=1 \mid x)$ und $p(t=1 \mid x, y=1) = 1$. |
| (c) | [Code] Training auf wahren $t$-Labels | Obere Schranke der Performance | Idealfall: du weisst von jedem Kunden ob zufrieden. Bestmoegliche Referenz. | $\min_\theta -\sum_i \big[t_i \log h(x_i) + (1-t_i)\log(1-h(x_i))\big]$. Accuracy $\sim 95\%$. |
| (d) | [Code] Training nur auf $y$-Labels | Ohne Korrektur systematisch verzerrt | Nur geschriebene Bewertungen als "zufrieden" $\Rightarrow$ unterschaetzt wahre Zufriedenheit. | $h(x) \approx p(y=1 \mid x) = \alpha \cdot p(t=1 \mid x)$. Vorhersagen um Faktor $\alpha$ zu klein. |
| (e) | [Code] $\alpha$ schaetzen, skalieren | Einfache Korrektur moeglich | Modell sagt 0.6, $\alpha=0.8$ $\Rightarrow$ wahre Zufriedenheit $\sim 0.75$. | $\alpha = \frac{1}{|V^+|}\sum_{i \in V^+} h(x_i)$. Korrektur: $p(t\!=\!1 \mid x) = h(x)/\alpha$. Neue Grenze: $\theta_0' = \theta_0 + \log(2/\alpha - 1)$. |

---

### Problem 3: Poisson Regression

**Thema:** Generalized Linear Models (GLMs), Exponentialfamilie
**Anwendungsfall:** Vorhersage von Zaehldaten

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | Poisson als Exponentialfamilie | Viele Verteilungen sind Spezialfaelle | Kundenanrufe pro Stunde im Callcenter folgen Poisson. Poisson, Bernoulli, Gauss -- alle eine Familie. | $p(y;\lambda) = \frac{\lambda^y}{y!}e^{-\lambda} = \exp\!\big(y\log\lambda - \lambda - \log y!\big)$. Exponentialfamilie: $\eta = \log\lambda$, $a(\eta) = e^\eta$, $b(y) = 1/y!$. |
| (b) | Kanonische Response-Funktion | Verbindet lineare Praediktion mit $\mathbb{E}[y]$ | Modell sagt $\theta^\top x = 3.5$ $\Rightarrow$ $e^{3.5} = 33$ erwartete Anrufe. $\exp(\cdot)$ stellt sicher: nie negativ. | $g(\eta) = \mathbb{E}[y \mid \eta] = a'(\eta) = e^\eta$. Also: $\mathbb{E}[y \mid x] = \exp(\theta^\top x)$. Inverse: $\eta = \log\mu$ (kanonischer Link). |
| (c) | SGD-Update herleiten | Update hat universelle Form | Egal ob Logistic oder Poisson: Update ist immer $(\text{tatsaechlich} - \text{vorhergesagt}) \cdot \text{Feature}$. | $\theta_j := \theta_j + \alpha\,(y_i - e^{\theta^\top x_i})\, x_{ij}$. Vgl. LR: $\theta_j := \theta_j + \alpha\,(y_i - \sigma(\theta^\top x_i))\, x_{ij}$. Gleiche Struktur $(y - h(x)) \cdot x$. |
| (d) | [Code] Poisson Regression | GLM-Framework universell anwendbar | Unfallzahlen pro Strassenabschnitt vorhersagen. | SGD mit $\text{lr} = 2 \times 10^{-7}$ auf ds4. Loss: $\text{NLL} = \sum_i \big(e^{\theta^\top x_i} - y_i\,\theta^\top x_i\big)$. |

---

### Problem 4: Convexity of Generalized Linear Models

**Thema:** GLMs, Konvexitaet, Exponentialfamilie
**Anwendungsfall:** Warum GLMs gut optimierbar sind

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | $\mathbb{E}[Y \mid X;\theta] = a'(\eta)$ | Erwartungswert aus Log-Partition-Funktion | $a(\eta)$ ist ein Generalschluessel: Erwartungswert, Varianz und mehr lassen sich ableiten. | Poisson: $a'(\eta) = e^\eta = \lambda$. Bernoulli: $a'(\eta) = \sigma(\eta) = p$. Gauss: $a'(\eta) = \eta = \mu$. |
| (b) | $\text{Var}(Y \mid X;\theta) = a''(\eta)$ | Varianz ebenfalls aus $a$ ableitbar | Poisson: $\text{Var} = \mathbb{E}$. Folgt direkt aus $a'' = a'$. | Poisson: $a''(\eta) = e^\eta = \lambda$. Bernoulli: $a''(\eta) = \sigma(\eta)(1-\sigma(\eta)) = p(1-p)$. Gauss ($\sigma=1$): $a''(\eta) = 1$. |
| (c) | NLL-Hessian ist PSD | NLL jedes GLMs ist konvex | Egal ob Klickraten, Anrufzahlen oder Verweildauern: globales Minimum garantiert. | $H = X^\top D X$, $D = \text{diag}(a''(\eta_i))$. Da $a'' = \text{Var}[Y] \geq 0$: $z^\top H z = (Xz)^\top D(Xz) = \sum_i d_i(x_i^\top z)^2 \geq 0$. |

---

### Problem 5: Locally Weighted Linear Regression

**Thema:** Non-parametrische Regression, Gewichtete Least Squares
**Anwendungsfall:** Regression wenn globale Linearitaet nicht gilt

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a-i) | Gewichtete Loss-Funktion mit $W$ | Lokale Gewichtung fuer nichtlineare Anpassung | Immobilienpreise: Preis-Flaeche-Zusammenhang in der Innenstadt anders als am Stadtrand. | $J(\theta) = \frac{1}{2}(X\theta - y)^\top W(X\theta - y)$, mit $W = \text{diag}(w_i)$, $w_i = \exp\!\Big(-\frac{(x_i - x_q)^2}{2\tau^2}\Big)$. |
| (a-ii) | Normalgleichung herleiten | Geschlossene Loesung pro Query-Punkt | "Was kosten Haeuser IN DIESER Nachbarschaft?" statt "im Durchschnitt?" | $\nabla_\theta J = X^\top W(X\theta - y) = 0 \;\Rightarrow\; \theta = (X^\top W X)^{-1} X^\top W y$. |
| (a-iii) | MLE mit heteroskedastischen Varianzen | Unterschiedliche Unsicherheiten motivieren Gewichtung | Teure Sensoren praeiser als guenstige $\Rightarrow$ praezise Messungen zaehlen staerker. | $y_i \sim \mathcal{N}(\theta^\top x_i,\, \sigma_i^2)$. $\max_\theta \ell = \min_\theta \sum_i \frac{(y_i - \theta^\top x_i)^2}{\sigma_i^2}$. Also $w_i = 1/\sigma_i^2$. |
| (b) | [Code] LWR mit $\tau=0.5$ | Bandwidth steuert Bias-Variance-Tradeoff | $\tau$ klein = Details (instabil). $\tau$ gross = Trend (stabil). Zoom bei Google Maps. | $\tau=0.03$: $w_i \approx 0$ ausserhalb $\pm 0.05$ $\Rightarrow$ Overfitting. $\tau=10$: $w_i \approx 1$ fuer alle $\Rightarrow$ globale lineare Regression. |
| (c) | [Code] $\tau$ tunen | Hyperparameter-Selektion entscheidend | Teste $\tau \in \{0.03, 0.05, 0.1, 0.5, 1, 10\}$, waehle bestes. | $\text{MSE}(\tau) = \frac{1}{n_\text{val}}\sum_i (y_i - \hat{y}_i(\tau))^2$. Bestes $\tau = 0.05$. |

---

## PS2: Supervised Learning II

### Problem 1: Logistic Regression -- Training Stability

**Thema:** Numerische Stabilitaet, Lineare Separierbarkeit
**Anwendungsfall:** Diagnose von Trainingsinstabilitaeten

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | Unterschied Dataset A vs. B | Lineare Trennbarkeit $\Rightarrow$ divergierende Gewichte | Spam vs. Ham: komplett trennbar $\Rightarrow$ LR schraubt Gewichte immer hoeher. | ds1\_a (nicht separierbar): Newton konvergiert in $\sim 30$k Iter. ds1\_b (separierbar): $\|\theta\|$ waechst monoton, keine Konvergenz nach $>100$k Iter. |
| (b) | Ursache erklaeren | $\theta \to \infty$ bei Separierbarkeit | Modell will $100\%$ Sicherheit. Da $\sigma(\cdot)$ nie exakt 0 oder 1 wird, muessen Gewichte $\to \infty$. | $\sigma(\theta^\top x) \to 1$ erfordert $\theta^\top x \to +\infty$. NLL $\to 0$ aber $\|\theta\| \to \infty$. Kein endliches Minimum. |
| (c) | Fuenf Gegenmassnahmen | L2-Regularisierung oder early stopping | L2 ist eine "Leine" fuer die Gewichte. Early Stopping: beenden bevor Gewichte explodieren. | $\min \text{NLL} + \lambda\|\theta\|^2$. Fuer $\|\theta\| \to \infty$ dominiert $\lambda\|\theta\|^2 \to \infty$ $\Rightarrow$ endliches Minimum existiert. |
| (d) | SVM mit Hinge-Loss | SVMs robust bei Separierbarkeit | SVMs stoppen bei genug Abstand (Margin). | Hinge: $\max(0, 1 - y\,\theta^\top x)$. Fuer $y\,\theta^\top x \geq 1$: Loss $= 0$, $\nabla = 0$. Gewichte wachsen nicht weiter. |

---

### Problem 2: Model Calibration

**Thema:** Kalibrierung, statistische Interpretation von Wahrscheinlichkeiten
**Anwendungsfall:** Wann kann man Modelloutputs als echte Wahrscheinlichkeiten interpretieren?

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | LR auf Trainingsset ist kalibriert | $\sum h(x_i) = \sum y_i$ | Wetterdienst: 10x "30% Regen" $\Rightarrow$ tatsaechlich 3 von 10 Regentage. | KKT bei $\theta^*$: $\nabla \text{NLL} = 0 \Rightarrow \sum_i (h(x_i) - y_i)\, x_i = 0$. Fuer $x_0 = 1$ (Intercept): $\sum_i h(x_i) = \sum_i y_i$. |
| (b) | Kalibrierung vs. Genauigkeit | Beides unabhaengig | "Immer 30% Regen" -- kalibriert aber nutzlos. "0% oder 100%" mit 90% Accuracy -- genau aber unkalibriert. | Kalibrierung: $P(Y\!=\!1 \mid h(X)\!=\!p) = p$. Genauigkeit: $P(Y = \lfloor h(X) \rceil)$. Konstantes $h(x) = \bar{p}$: perfekt kalibriert, Accuracy $= \max(\bar{p}, 1-\bar{p})$. |
| (c) | Einfluss L2-Regularisierung | Regularisierung bricht Kalibrierung | L2 zieht Vorhersagen Richtung 50%. | Mit L2: $\sum_i (h(x_i) - y_i)\, x_i + 2\lambda\theta = 0$. Fuer $x_0=1$: $\sum h(x_i) = \sum y_i - 2\lambda\theta_0 \neq \sum y_i$. |

---

### Problem 3: Bayesian Interpretation of Regularization

**Thema:** Bayesianische Inferenz, MAP-Schaetzung
**Anwendungsfall:** Regularisierung als Prior-Wissen verstehen

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | $\theta_\text{MAP} = \arg\max\, p(y \mid X,\theta)\, p(\theta)$ | MAP ist MLE mit Prior | Muenze: MLE zaehlt nur Wuerfe. MAP: Prior "wahrscheinlich fair". | $\theta_\text{MAP} = \arg\max_\theta \big[\log p(y \mid X,\theta) + \log p(\theta)\big]$. Zusaetzlicher Term $\log p(\theta)$ wirkt als Regularisierer. |
| (b) | Gauss-Prior $\Rightarrow$ L2 | L2 = Annahme "Parameter sind klein" | Kein einzelner Faktor bestimmt den Immobilienpreis allein. | $\theta_j \sim \mathcal{N}(0, \tau^2)$. $\log p(\theta) = -\frac{1}{2\tau^2}\|\theta\|^2 + \text{const}$. MAP $=$ MLE $+ \frac{1}{2\tau^2}\|\theta\|^2 =$ MLE $+ \lambda\|\theta\|^2$ mit $\lambda = \frac{1}{2\tau^2}$. |
| (c) | Geschlossene Loesung $\theta_\text{MAP}$ | Ridge hat analytische Loesung | Direkt ausrechnen statt iterativ optimieren. | $\theta_\text{MAP} = (X^\top X + \lambda I)^{-1} X^\top y$, mit $\lambda = \sigma_\varepsilon^2 / \tau^2$. Vgl. OLS: $(X^\top X)^{-1} X^\top y$. $\lambda I$ macht Matrix immer invertierbar. |
| (d) | Laplace-Prior $\Rightarrow$ L1 (Lasso) | L1 foerdert Sparsity | Genanalyse: von 20.000 Genen nur wenige relevant. L1 setzt den Rest auf 0. | $\theta_j \sim \text{Laplace}(0, b)$: $p(\theta_j) = \frac{1}{2b}e^{-|\theta_j|/b}$. $\log p(\theta) = -\frac{1}{b}\|\theta\|_1 + \text{const}$. L1-Norm hat Ecken bei $\theta_j=0$ $\Rightarrow$ viele $\theta_j$ exakt $0$. |

---

### Problem 4: Constructing Kernels

**Thema:** Kernel-Methoden, Mercer's Theorem
**Anwendungsfall:** Nachweis ob eine Funktion ein gueltiger Kernel ist

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a)-(h) | Kernel-Kombinationen pruefen | Gueltige Kernels unter $+$, $\cdot$, pos. Skalierung abgeschlossen | Aehnlichkeitsmasse kombinieren: Farbe + Form = gueltiges Mass. | $K_1, K_2$ gueltig (Gram-Matrizen PSD). (1) $z^\top(K_1+K_2)z = z^\top K_1 z + z^\top K_2 z \geq 0$. (2) $K_1 \circ K_2$ PSD (Schur-Produkt). (3) $aK_1$ mit $a>0$ PSD. Nicht gueltig: $K_1 - K_2$. |

---

### Problem 5: Kernelizing the Perceptron

**Thema:** Kernel-Trick, Online Learning
**Anwendungsfall:** Nicht-lineare Klassifikation ohne explizite Feature-Transformation

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | $\theta$ als Linearkombination der Trainingspunkte | Representer Theorem: Loesung im Span der Daten | Richter orientiert sich an Praezedenzfaellen. | $\theta = \sum_i \beta_i \phi(x_i)$. Vorhersage: $\text{sign}\!\Big(\sum_i \beta_i\, K(x_i, x)\Big)$. Nie $\phi(x)$ explizit berechnen. |
| (b) | [Code] Kernelisierten Perceptron implementieren | Kernel-Trick: implizite hochdim. Raeume | Bilder: RBF-Kernel berechnet "Aehnlichkeit" direkt statt Milliarden Features. | Update: wenn $\text{sign}\!\big(\sum_j \beta_j K(x_j, x_i)\big) \neq y_i$, dann $\beta_i := \beta_i + y_i$. RBF: $K(x,x') = \exp\!\big(-\|x-x'\|^2/(2\sigma^2)\big)$. |
| (c) | Dot-Product vs. RBF | RBF fuer nichtlineare Grenzen | Linear: Linie ziehen. RBF: beliebige Form (Kreise, Spiralen). | Dot-Product: $K = x^\top x'$ $\Rightarrow$ Hyperebene. RBF ($\sigma\!=\!1$): $K = e^{-\|x-x'\|^2/2}$ $\Rightarrow$ beliebig komplex. ds5: Dot $\sim 60\%$, RBF $\sim 95\%$ Accuracy. |

---

### Problem 6: Spam Classification

**Thema:** Naive Bayes, Text Classification, SVM
**Anwendungsfall:** SMS-Spam-Erkennung (echte SMS-Nachrichten)

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | [Code] Text-Preprocessing | Feature Engineering entscheidend bei NLP | "WINNER!! Claim your prize" $\Rightarrow$ {winner:1, claim:1, prize:1}. Nur Woerter mit $\geq 5$ Vorkommen. | $x \in \mathbb{R}^{|V|}$, $x_j = $ Haeufigkeit von Wort $j$. Sparse: meiste $x_j = 0$. |
| (b) | [Code] Naive Bayes mit Laplace-Smoothing | NB gut trotz Unabhaengigkeitsannahme | Jedes Wort einzeln betrachtet, trotzdem 97.85% Accuracy. | $p(y\!=\!1 \mid x) \propto p(y\!=\!1) \prod_j p(x_j \mid y\!=\!1)$. Laplace: $p(w_j \mid y\!=\!k) = \frac{\text{count}(w_j, y\!=\!k) + 1}{\sum_j \text{count}(w_j, y\!=\!k) + |V|}$. |
| (c) | [Code] Spam-indikative Tokens | Log-Ratio gibt Modelleinblick | Top-5: 'claim', 'won', 'prize', 'tone', 'urgent!'. | Token-Score: $\log\frac{p(w_j \mid \text{spam})}{p(w_j \mid \text{ham})}$. z.B. "prize": Score $= 5.2$ $\Rightarrow$ $180\times$ wahrscheinlicher in Spam. |
| (d) | [Code] SVM-RBF tunen | Kernel-SVMs oft besser bei genug Daten | SVM: 96.95%. Hier leicht schlechter als NB (kleine Datenmenge). | RBF: $K(x,x') = \exp(-\|x-x'\|^2 / (2r^2))$. Grid-Search: $r \in \{0.01, 0.1, 1, 10\}$. Bestes $r=0.1$. |

---

## PS3: Deep Learning & Unsupervised Learning

### Problem 1: A Simple Neural Network

**Thema:** Backpropagation, Aktivierungsfunktionen
**Anwendungsfall:** Nichtlineare 2D-Klassifikation (mondfoermige Cluster), Netz: 2-3-1

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | GD-Update herleiten | Backprop ist Kettenregel durch den Graphen | Fabrik: Fehler Station fuer Station zurueckverfolgen, jede Maschine nachstellen. | $o = \sigma(W_2\, \sigma(W_1 x + b_1) + b_2)$. $\frac{\partial L}{\partial W_2} = (o-y)\,\sigma'(z_2)\, h^\top$. $\frac{\partial L}{\partial W_1} = \big((o-y)\,\sigma'(z_2)\, W_2^\top\big) \odot \sigma'(z_1) \cdot x^\top$. |
| (b) | [Code] Stufenfunktion | Nichtlinearitaet ermoeglicht XOR | "Schirm wenn Wolken ODER Wind, aber nicht beides". Lineares Modell versagt. | $\text{step}(z) = \mathbb{1}\{z>0\}$. $h_1 = \text{step}(x_1+x_2-0.5)$, $h_2 = \text{step}(-x_1-x_2+1.5)$. $o = \text{step}(h_1+h_2-1.5)$. Realisiert XOR. |
| (c) | [Code] Lineare Aktivierung | Lineare Netze kollabieren | Zwei Brillen hintereinander: vergroessern, aber biegen nichts. Tiefe bringt nichts. | $f(x) = W_2(W_1 x + b_1) + b_2 = \underbrace{(W_2 W_1)}_{W'} x + \underbrace{(W_2 b_1 + b_2)}_{b'}$. Aequivalent zu einer Schicht. Kann XOR nicht loesen. |

---

### Problem 2: KL Divergence and Maximum Likelihood

**Thema:** Informationstheorie, MLE
**Anwendungsfall:** Warum MLE sinnvoll ist

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | $D_\text{KL} \geq 0$ (Jensen) | KL nicht symmetrisch aber $\geq 0$ | "Entfernung" Glaube $\leftrightarrow$ Wahrheit: immer $\geq 0$, aber $A \to B \neq B \to A$. | $P = [0.5, 0.5]$, $Q = [0.9, 0.1]$. $D_\text{KL}(P\|Q) = 0.5\ln\frac{0.5}{0.9} + 0.5\ln\frac{0.5}{0.1} = 0.51$. $D_\text{KL}(Q\|P) = 0.37 \neq 0.51$. Beweis $\geq 0$: Jensen auf $\mathbb{E}_P[\log(Q/P)] \leq \log\mathbb{E}_P[Q/P] = 0$. |
| (b) | Kettenregel fuer KL | KL dekomponiert in Teile | Reisekosten = Flug + Hotel. Gesamt-KL = marginale + bedingte. | $D_\text{KL}(P_{X,Y} \| Q_{X,Y}) = D_\text{KL}(P_X \| Q_X) + \mathbb{E}_{x \sim P}\!\big[D_\text{KL}(P_{Y|X} \| Q_{Y|X})\big]$ |
| (c) | MLE $=$ min $D_\text{KL}(\hat{P} \| P_\theta)$ | MLE passt Modell an empirische Verteilung an | Schneider passt Anzug an Koerpermasse an. | $D_\text{KL}(\hat{P} \| P_\theta) = -H(\hat{P}) - \mathbb{E}_{\hat{P}}[\log P_\theta]$. $H(\hat{P})$ konstant $\Rightarrow$ $\min D_\text{KL} \Leftrightarrow \max \frac{1}{n}\sum_i \log P_\theta(x_i) = \text{MLE}$. |

---

### Problem 3: KL Divergence, Fisher Information, Natural Gradient

**Thema:** Geometrie des Parameterraums, Second-Order Optimierung
**Anwendungsfall:** Effizientere Optimierung fuer probabilistische Modelle

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | $\mathbb{E}[\text{Score}] = 0$ | Score hat Erwartungswert 0 | Kompass zeigt im Mittel nach Norden -- kein Bias. | $s(\theta) = \nabla_\theta \log p(x;\theta)$. $\mathbb{E}[s] = \int p \cdot \frac{\nabla p}{p}\, dx = \nabla \int p\, dx = \nabla 1 = 0$. |
| (b) | Fisher $= \text{Cov}(\text{Score})$ | Fisher misst Sensitivitaet | Muenze $p=0.5$: Fisher minimal (schwer Bias zu erkennen). $p=0.01$: Fisher hoch. | Bernoulli$(p)$: Score $= y/p - (1-y)/(1-p)$. $I(p) = \text{Var}(\text{Score}) = \frac{1}{p(1-p)}$. $p=0.5$: $I=4$. $p=0.01$: $I=101$. |
| (c) | Fisher $= -\mathbb{E}[H_{\log p}]$ | Verbindet Fisher mit Kruemmung | Zwei Wege, ein Ergebnis: Streuung der Steigung = Kruemmung der Landschaft. | $-\mathbb{E}\!\big[\nabla^2_\theta \log p\big] = \mathbb{E}[s\, s^\top] - \underbrace{(\mathbb{E}[s])(\mathbb{E}[s])^\top}_{=0} = \text{Var}(s) = I(\theta)$. |
| (d) | $D_\text{KL} \approx \frac{1}{2} d^\top I(\theta)\, d$ | Fisher als lokale KL-Approximation | Parameterschritte in der "Welt der Verteilungen" messen. | Taylor: $D_\text{KL}(p_\theta \| p_{\theta+d}) = \underbrace{0 + 0}_{\text{1./2. Ord.}} + \frac{1}{2} d^\top I(\theta)\, d + O(\|d\|^3)$. |
| (e) | Natural Gradient | Beruecksichtigt Geometrie | Normaler Gradient: ueberall gleiche Schrittweite. Natural: in Kurven langsamer, auf Geraden schneller. | Constraint: $d^\top I(\theta)\, d = \varepsilon^2$. Lagrange $\Rightarrow$ $d^* \propto I(\theta)^{-1} \nabla_\theta \ell$. Natural Gradient $= I^{-1} \nabla \ell$. |
| (f) | Natural Gradient $=$ Newton fuer GLMs | Fuer GLMs aequivalent | Zwei Theorien, ein Ergebnis. | GLMs: $H = -X^\top DX$ unabhaengig von $y$. $\Rightarrow$ $I = \mathbb{E}[-H] = -H$. Also $I^{-1}\nabla\ell = (-H)^{-1}\nabla\ell = H^{-1}(-\nabla\ell) =$ Newton. |

---

### Problem 4: Semi-supervised EM

**Thema:** EM-Algorithmus, Gaussian Mixture Models, Semi-supervised Learning
**Anwendungsfall:** Clustering mit teilweise gelabelten 2D-Daten, $K=3$ Cluster

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | Konvergenz von semi-sup EM | ELBO steigt monoton $\Rightarrow$ Konvergenz | Bergsteiger: bei jedem Schritt $\geq$ gleich hoch -- nie absteigen. | $\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$. E-Step maximiert ELBO bzgl. $q$, M-Step bzgl. $\theta$. Beide erhoehen ELBO. |
| (b) | E-Step herleiten | Weiche Cluster-Zuweisung | Kundensegmentierung: "60% Premium, 30% Standard, 10% Budget". | $w_j^{(i)} = \frac{\phi_j\, \mathcal{N}(x_i;\, \mu_j, \Sigma_j)}{\sum_k \phi_k\, \mathcal{N}(x_i;\, \mu_k, \Sigma_k)}$. Gelabelte Punkte: $w_j^{(i)} = \mathbb{1}\{z_i = j\}$ (fest). |
| (c) | M-Step herleiten | Gewichtete MLE-Updates | Premium-Profil = gewichteter Durchschnitt aller Kunden. | $\mu_j = \frac{\sum_i w_j^{(i)} x_i}{\sum_i w_j^{(i)}}$, $\Sigma_j = \frac{\sum_i w_j^{(i)}(x_i - \mu_j)(x_i - \mu_j)^\top}{\sum_i w_j^{(i)}}$, $\phi_j = \frac{1}{m}\sum_i w_j^{(i)}$. |
| (d) | [Code] Unsupervised EM | Sensitiv gegenueber Initialisierung | Ohne Vorgaben: EM gruppiert ggf. falsch, je nach Startpunkt. | Random-Init: $\mu_j$ zufaellig, $\Sigma_j = I$, $\phi_j = 1/K$. Verschiedene Seeds $\Rightarrow$ verschiedene lokale Maxima. |
| (e) | [Code] Semi-supervised EM | Wenige Labels stabilisieren erheblich | $10\%$ gelabelt: EM findet richtige Segmentierung zuverlaessig. | Gelabelte Punkte: $w_j^{(i)} = \mathbb{1}\{z_i=j\}$ fest im E-Step. M-Step mischt feste + weiche Zuweisungen. |
| (f) | Vergleich | Semi-supervision verbessert Qualitaet | Unsup: $85\%$. Semi-sup: $95\%$. Wenige Labels $=$ grosser Unterschied. | Semi-sup konvergiert in $\sim 10$ Iter. (vs. $\sim 30$). Hoeheres Log-Likelihood. Kein Label-Switching. |

---

### Problem 5: K-Means for Compression

**Thema:** K-Means Clustering, Bildkompression
**Anwendungsfall:** Farbreduktion im "Peppers"-Bild auf 16 Farben

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | [Code] K-Means auf Bildpixel | K-Means einfach und wirkungsvoll | Maler mit 16 Farbtuben: K-Means findet die besten 16 Farben. | $x_i \in \mathbb{R}^3$ (RGB). $\min \sum_i \|x_i - \mu_{c(i)}\|^2$, $c(i) = \arg\min_j \|x_i - \mu_j\|^2$. Update: $\mu_j = \frac{1}{|C_j|}\sum_{i \in C_j} x_i$. |
| (b) | Kompressionsrate | Signifikante Kompression | $24$ Bit/Pixel $\to$ $4$ Bit/Pixel $+$ kleine Tabelle. Faktor $6$. Prinzip von GIF. | Original: $H \!\cdot\! W \!\cdot\! 24$ Bit. Komprimiert: $H \!\cdot\! W \!\cdot\! \log_2(16) + 16 \!\cdot\! 24 = H \!\cdot\! W \!\cdot\! 4 + 384$ Bit. Ratio $\approx 24/4 = 6$. |

---

## PS4: EM, Deep Learning & Reinforcement Learning

### Problem 1: Neural Networks -- MNIST

**Thema:** Convolutional Neural Networks, Backpropagation
**Anwendungsfall:** Handschrifterkennung (MNIST: 60k Bilder, $28 \times 28$ Pixel, 10 Klassen)

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | [Code] Backward-Passes | Jede Schicht berechnet lokalen Gradienten | Postleitzahlen lesen: Conv erkennt Kanten, MaxPool macht positionsunabhaengig, Softmax gibt $P(\text{Ziffer})$. | Softmax: $\frac{\partial L}{\partial z_i} = p_i - \mathbb{1}\{i=y\}$. ReLU: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \mathbb{1}\{x>0\}$. Conv: $\frac{\partial L}{\partial W} = \sum_\text{patches} \frac{\partial L}{\partial y} \cdot x_\text{patch}$. |
| (b) | [Code] Vollstaendiger Backward-Pass | Modulare Implementierung erweiterbar | LEGO: Bausteine mit forward/backward. Neue Schichten einstecken. | Input$(28^2)$ $\to$ Conv$(4{\times}4, 2\text{ch})$ $\to$ MaxPool$(5{\times}5)$ $\to$ ReLU $\to$ Flatten $\to$ Linear$(50,10)$ $\to$ Softmax $\to$ CE. Batch$=16$, lr$=0.01$. |

---

### Problem 2: Off-Policy Evaluation & Causal Inference

**Thema:** Importance Sampling, Counterfactual Reasoning
**Anwendungsfall:** Medikamenten-Policy evaluieren ohne sie am Patienten zu testen

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | IS-Schaetzer ist unbiased | Daten aus anderer Policy verwertbar | Krankenhaus verschreibt Med. X. "Was waere mit Y?" $\Rightarrow$ Daten umgewichten. | $\hat{V}_\text{IS} = \frac{1}{n}\sum_i \frac{\pi_1(a_i|s_i)}{\pi_0(a_i|s_i)} r_i$. $\mathbb{E}[\hat{V}_\text{IS}] = \mathbb{E}_{\pi_0}\!\big[\frac{\pi_1}{\pi_0} R\big] = \mathbb{E}_{\pi_1}[R]$. |
| (b) | Weighted IS | Normalisierte Gewichte stabiler | Gewichte koennen extrem sein. Normalisierung daempft Ausreisser. | $\hat{V}_\text{WIS} = \frac{\sum_i w_i\, r_i}{\sum_i w_i}$, $w_i = \frac{\pi_1(a_i|s_i)}{\pi_0(a_i|s_i)}$. Selbst-normalisierend. |
| (c) | Weighted IS biased bei endlichen $n$ | Bias-Variance-Tradeoff | Stabiler aber leichter Bias. Bei genug Daten $\to 0$. | $\mathbb{E}[\hat{V}_\text{WIS}] \neq \frac{\mathbb{E}[\sum w_i r_i]}{\mathbb{E}[\sum w_i]}$ (Jensen). Bias $= O(1/n) \to 0$. $\text{Var}(\hat{V}_\text{WIS}) \ll \text{Var}(\hat{V}_\text{IS})$. |
| (d) | Doubly Robust | Robust wenn EIN Modell korrekt | Doppelter Boden: gutes Wirkungsmodell ODER gute Policy-Schaetzung reicht. | $\hat{V}_\text{DR} = \frac{1}{n}\sum_i \big[\hat{r}(s_i, \pi_1) + w_i(r_i - \hat{r}(s_i, a_i))\big]$. $\hat{r}$ korrekt $\Rightarrow$ 2. Term $\mathbb{E}=0$. $w_i$ korrekt $\Rightarrow$ 2. Term korrigiert 1. |
| (e) | Wann IS vs. Regression? | Abhaengig von Modellqualitaet | Policy gut bekannt $\Rightarrow$ IS. Wirkung verstanden $\Rightarrow$ Regression. Unsicher $\Rightarrow$ DR. | $\text{Var}(\text{IS}) \sim \sum w_i^2$. $\text{Bias}(\text{Reg}) \sim \|\hat{r} - r\|$. DR balanciert beides. |

---

### Problem 3: PCA

**Thema:** Dimensionsreduktion, Varianzmaximierung
**Anwendungsfall:** Feature-Reduktion und Visualisierung

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| - | 1. Hauptkomponente minimiert MSE | PCA $=$ max Varianz $=$ min Rekonstruktionsfehler | 100 Sensoren messen Bruecke. PCA: 3 Komponenten erklaeren 95% der Variation. 97 messen Rauschen. | $u_1 = \arg\max_{\|u\|=1} u^\top \Sigma\, u$, $\Sigma = \frac{1}{n}X^\top X$. Loesung: $u_1 =$ Eigenvektor zum groessten $\lambda_1$. Erklaerte Varianz: $\lambda_1 / \sum_i \lambda_i$. Fehler: $\sum_{i=k+1}^d \lambda_i$. |

---

### Problem 4: Independent Components Analysis (ICA)

**Thema:** Blind Source Separation, Non-Gaussian Quellen
**Anwendungsfall:** Cocktail-Party-Problem: 5 gemischte Audiosignale trennen

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | Gauss-Quellen: $W$ nicht eindeutig | ICA versagt fuer Gausssche Quellen | Weisses Rauschen: egal wie Mikrofone gedreht -- klingt gleich (rotationsinvariant). | $s \sim \mathcal{N}(0, I)$. Fuer orthogonales $R$: $Rs \sim \mathcal{N}(0, RIR^\top) = \mathcal{N}(0, I)$. $s$ und $Rs$ ununterscheidbar $\Rightarrow$ $W$ nur bis auf Rotation bestimmt. |
| (b) | Laplace-Quellen: SGD-Update | Non-Gaussian ermoeglicht Trennung | Sprache und Musik haben Muster (nicht-gaussisch). ICA findet "am wenigsten gaussische Richtung". | $s_j \sim \text{Laplace}(0,1)$. $\ell(W) = \sum_t \big[\log|\det W| + \sum_j \log p(w_j^\top x_t)\big]$. SGD: $W := W + \alpha\big[(1-2\sigma(Wx))x^\top + (W^\top)^{-1}\big]$. |
| (c) | [Code] ICA auf Cocktail-Party | ICA trennt ohne Kenntnis der Mischmatrix | 5 Mikrofone, 5 Sprecher $\Rightarrow$ 5 getrennte WAV-Dateien. | $x = As$ ($5{\times}5$ Mischmatrix). ICA findet $W \approx A^{-1}$. $\hat{s} = Wx$. Annealing: lr von $0.1$ bis $0.001$ in 16 Stufen. |

---

### Problem 5: Markov Decision Processes

**Thema:** Dynamische Programmierung, Bellman-Gleichung
**Anwendungsfall:** Optimale sequentielle Entscheidungen

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| (a) | Bellman-Operator ist $\gamma$-Kontraktion | Value Iteration konvergiert garantiert | Schachcomputer: Bewertungen stabilisieren sich, egal welcher Startwert. Fehler schrumpft um $\gamma$ pro Iteration. | $(BV)(s) = \max_a \big[R(s,a) + \gamma \sum_{s'} P(s'|s,a)\, V(s')\big]$. $\|BV_1 - BV_2\|_\infty \leq \gamma\, \|V_1 - V_2\|_\infty$. $\gamma=0.995$: nach 1000 Iter. Fehler $\times 0.995^{1000} \approx 0.007$. |
| (b) | Eindeutigkeit des Fixpunkts | Genau eine optimale $V^*$ | Eine wahre Bewertung pro Position. Verschiedene Starts $\to$ gleiche Loesung. | Banach: $\gamma$-Kontraktion auf vollst. metr. Raum hat genau einen Fixpunkt $V^*$ mit $BV^* = V^*$. |

---

### Problem 6: Reinforcement Learning -- Inverted Pendulum

**Thema:** Model-based RL, Value Iteration
**Anwendungsfall:** CartPole balancieren (163 Zustaende, 2 Aktionen: links/rechts)

| Teilaufgabe | Inhalt | Take-Away | Beispiel (Praxis) | Beispiel (Mathematik) |
|-------------|--------|-----------|--------------------|-----------------------|
| - | [Code] MDP schaetzen, Value Iteration, Policy | Geschaetztes Modell ermoeglicht Planung | Besenstiel auf Handflaeche balancieren. Agent lernt aus Erfahrung, dann vorausplanen. | Zustand $(x, \dot{x}, \theta, \dot{\theta}) \to 163$ Zustaende. $\hat{P}(s'|s,a) = \frac{\text{count}(s,a,s')}{\text{count}(s,a)}$. $V(s) = \max_a \big[R(s,a) + \gamma \sum_{s'} \hat{P}\, V(s')\big]$, $\gamma=0.995$. $\pi(s) = \arg\max_a [\cdots]$. Konvergenz: $\max_s |V_{t+1}(s) - V_t(s)| < 0.01$. |
