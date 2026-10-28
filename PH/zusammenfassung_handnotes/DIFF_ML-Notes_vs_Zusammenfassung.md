# Diff: `hand-notes/ML Notes.pdf` vs. Handnotes-Zusammenfassung

Verglichen wurden alle 19 Seiten der *ML Notes* (DataXOdyssey, KI-generierte Einseiter) mit dem Stand der Zusammenfassung **vor** dieser Überarbeitung (33 Seiten). Dazu: was jetzt umgesetzt ist, wo die ML Notes besser erklären, wo sie ungenau sind.

## 1. Themen-Diff

| ML Notes (Seite) | Thema | Vorher in der Zusammenfassung | Jetzt |
|---|---|---|---|
| 2 | ML Fundamentals (Lernarten, Workflow, Train/Val/Test, Begriffe) | nur als Landkarte im Inhaltsverzeichnis | Kap. 28 *ML-Workflow & Best Practices* |
| 3 | Linear Regression | ja, aber **ohne Annahmen** (Linearität, Homoskedastizität, …) | Annahmen-Panel in Kap. 1 |
| 4 | Decision Trees | ja; **IG-Formel, ID3/C4.5/CART, sklearn-Hyperparameter fehlten** | Ergänzung in Kap. 9 |
| 5 | Random Forest | ja; **Feature Importance, RF-vs-DT fehlten** | Ergänzung in Kap. 10 |
| 6 | **k-NN** | **fehlte komplett** | Kap. 8 (+ Code) |
| 7 | SVM | ja (tiefer); **SVR, Sigmoid-Kernel, Hyperparameter-Tabelle fehlten** | Ergänzung in Kap. 6 |
| 8 | Naive Bayes | ja; **Varianten (Gauß/Multinomial/Bernoulli), allg. Glättung α·V, Tabellen-Beispiel fehlten** | Ergänzung in Kap. 4 |
| 9 | K-Means | ja; **Silhouette, Hyperparameter, WCSS-Begriff fehlten** | Ergänzung in Kap. 29 |
| 10 | **Hierarchisches Clustering** | **fehlte komplett** | Kap. 30 (+ Code) |
| 11 | PCA | ja (mit Herleitung); **EVR/CEV, Scree fehlten** | Ergänzung in Kap. 34 |
| 12 | Gradient Descent (Lernraten-Bilder, Batch/SGD/Mini-Batch, NumPy-Code) | verteilt in Kap. 1/12 | eigene Seite **G3 Optimierung** + NumPy-Code |
| 13 | Klassifikationsmetriken | ja; **FPR/FNR/NPV, Macro/Micro/Weighted, "When to use what" fehlten** | Ergänzung in Kap. 23 |
| 14 | **Regressionsmetriken** (MAE, MSE, RMSE, R², Adj. R²) | **fehlte komplett** | Kap. 24 (mit Korrektur, s. u.) |
| 15 | Over-/Underfitting, Bias–Varianz | ja (tiefer: Zerlegung, Parameterraum, Double Descent) | unverändert; Herleitung in H2 |
| 16 | **Feature Engineering & Preprocessing** | **fehlte komplett** | Kap. 27 |
| 17 | Cross-Validation & Hyperparameter-Tuning | nur K-Fold in Kap. Regularisierung | Kap. 20 (Stratified, LOOCV, Zeitreihen, Grid/Random, **Nested CV**) |
| 18 | Ensemble Learning | Bagging/Boosting ja; **Stacking/Voting fehlten** | Ergänzung in Kap. 10 |
| 19 | **End-to-End-Workflow, Data Leakage, MLOps** | **fehlte komplett** | Kap. 28 |

**Netto:** 6 komplette Themen fehlten (k-NN, Hierarchisches Clustering, Regressionsmetriken, Feature Engineering, Tuning/Nested CV, Workflow/Leakage) und 10 Seiten hatten inhaltliche Lücken. Alles ist jetzt abgedeckt.

Umgekehrt enthält die Zusammenfassung Themen, die die ML Notes nicht haben (GLM, Exponentialfamilie, Kernel/Mercer/Representer, Perceptron-Schranke, Lerntheorie/VC, EM/ELBO, Faktorenanalyse, ICA, HMM, Gaussian Processes, RL, LQR/DDP/LQG, Policy Gradient, Foundation Models, kritische Perspektiven).

## 2. Wo die ML Notes besser erklären (und was ich übernommen habe)

| Aspekt | Warum besser | Umsetzung |
|---|---|---|
| **Einheitliches Seiten-Template** (Definition → Ablauf → Mathe → **Annahmen** → **Hyperparameter** → Code → **Wann / Wann nicht** → Anwendungen) | Nachschlagen ist trivial, weil jede Methode dieselben Fragen beantwortet. Bei mir waren Annahmen, Hyperparameter und Einsatzgrenzen ungleichmäßig vorhanden. | Neue Kapitel folgen dem Template; bei älteren Kapiteln ein Ergänzungs-Panel „aus den ML Notes". |
| **Entscheidungshilfen als Tabelle** („When to use what": Metriken, CV-Verfahren, Ensemble-Typ) | Direkte Handlungsanweisung statt Fließtext. | Übernommen in Kap. 10, 20, 23, 24. |
| **Bildhafte Prozessdiagramme** (K-Fold-Kästchen, Nested CV, Leakage gut/schlecht, Dendrogramm mit Schnitt, Bagging/Boosting/Stacking-Flüsse, Elbow) | Ein Blick reicht, um den Ablauf zu verstehen. | K-Fold-Grafik, Dendrogramm, Train/Val/Test-Balken, Workflow-Flow, Lernraten-Skizze ergänzt. |
| **Kleine numerische Beispiele in Tabellen** (Regressionsmetriken mit 5 Punkten, Multinomial-NB-Tabelle, Konfusionsmatrix 80/20/10/90) | Man rechnet jeden Schritt mit. | Übernommen und **maschinell nachgerechnet**. |
| **Code mit erwarteter Ausgabe** (scikit-learn) | Verbindet Theorie und Anwendung sofort. | Code auf den Vertiefungsseiten: 21 Snippets, **real ausgeführt**, Ausgabe daneben (scikit-learn und PyTorch). |
| **Hyperparameter-Tabellen** (sklearn-Namen mit Wirkung) | Übersetzt Theorie in konkrete Regler. | In den Ergänzungs-Panels (SVM, Bäume, K-Means, k-NN). |
| **Praxisthemen** (Preprocessing, Leakage, Monitoring) | Fehlten bei mir fast ganz, sind aber in echten Projekten entscheidend. | Kap. 27, 30, 20. |

**Wo meine Zusammenfassung besser ist:** Tiefe (Vertiefungsseiten mit Herleitungen, Beweise, Lerntheorie), Konsistenz der Notation, Verbindungen zwischen Kapiteln (MLE ⇄ Verlust ⇄ Regularisierung, EM ⇄ K-Means ⇄ FA ⇄ VAE), Pseudo-Code, Selbsttests, Quellenangaben, Prüfung aller Zahlen.

## 3. Unschärfen und Fehler in den ML Notes (geprüft)

1. **Rechenfehler (verifiziert):** Regression-Metrics-Beispiel: SST wird mit 13,30 angegeben, korrekt ist **12,30** (Σ(yᵢ−ȳ)² = 4,84+1,44+0,09+0,64+5,29). Damit ist **R² = 0,874**, nicht 0,884. In Kap. 24 korrigiert (`code/verify_examples.py`).
2. **K-Means „works well with high-dimensional data"** (Vorteile-Liste) widerspricht der k-NN-Seite (Fluch der Dimension) – auch K-Means nutzt euklidische Abstände, die in hohen Dimensionen konzentrieren. Ich habe das nicht übernommen.
3. **Lineare Regression „Normality of errors" als Annahme:** Der Least-Squares-Schätzer braucht sie nicht; sie wird für die MLE-Interpretation und Konfidenzintervalle gebraucht. In meinem Panel so eingeordnet.
4. **Hierarchisches Clustering „O(n²) Zeit und Speicher":** Speicher ist O(n²), naiv ist die Zeit O(n³) (optimiert ca. O(n² log n)). In Kap. 30 präzisiert.
5. **Konfusionsmatrix-Orientierung:** ML Notes und Kap. 23: Zeilen = Vorhersage, Spalten = Wahrheit. **scikit-learn** liefert Zeilen = Wahrheit (`[[TN, FP], [FN, TP]]`). Im Code-Anhang ist das ausdrücklich markiert.
6. **Decision Trees „handles missing values"**: nur „to some extent" – abhängig von der Bibliothek/Version (scikit-learn unterstützt es erst in neueren Versionen); nicht als allgemeine Eigenschaft übernommen.
7. **Werbe-/Füllanteile** (Real-World-Application-Symbolzeilen, „Key Takeaway"-Sprüche) tragen wenig Lernwert und wurden nicht übernommen.

## 4. Neu in der Zusammenfassung (Überblick)

- **Nachtrag 1:** *Feature Importance & SHAP* (Kap. 25) und *Fairness-Analyse* (Kap. 26), je eine Theorie- und eine Vertiefungsseite mit Herleitung und getestetem Code; Einordnung gegenüber Leistungsmetriken und Feature Importance auf beiden Theorieseiten.

- **3 Grundlagen-Seiten (G1–G3):** Wahrscheinlichkeit & Statistik (Bayes, MLE/MAP, Entropie/KL), Lineare Algebra & Matrixableitungen, Optimierung (Konvexität, GD, Newton, Lagrange/KKT).
- **9 neue Kapitel:** k-NN, Hierarchisches Clustering, Regressionsmetriken, Feature Engineering, Validierung & Tuning, ML-Workflow, DL-Praxis (Adam, BatchNorm, Dropout, Init), CNN & RNN/LSTM, Generative Modelle (VAE, GAN, Diffusion).
- **Vertiefungsseiten (+) direkt hinter dem jeweiligen Thema (Herleitung + Code):** Normalgleichung, Logistic-Gradient/Hesse/Softmax/GLM · Bias–Varianz, Hoeffding, Union Bound, Perceptron · SVM (Primal → Dual, Soft-Margin, SMO) · ELBO/EM-Monotonie/GMM/PCA/MLE · Backprop (Kettenregel, δ-Rekursion, Softmax+CE) · Bellman, Kontraktion, Policy Gradient, Baseline, Riccati · Newton, Ridge, MAP, Lasso, ICA-Dichte.
- **Code auf den Vertiefungsseiten:** 21 lauffähige Snippets in `code/*.py` mit realen Ausgaben in `code/out/`.
- **Alle Rechenbeispiele** der Seiten werden von `code/verify_examples.py` nachgerechnet (96 Prüfungen, alle bestanden).

## 5. Struktur (Umbau)

- Herleitungen und Code stehen nicht mehr gebündelt am Ende, sondern als **Vertiefungsseite (+) direkt hinter dem Thema** (18 Stück).
- Es gibt zwei PDFs aus derselben Quelle: die **Langfassung** (Übersichten + Vertiefungen) und die **Kompaktfassung** (nur die 40 Übersichtsseiten, Grundlagen und Spickzettel).
- Neue Herleitungen: GDA ⇒ logistische Form, Information Gain ≥ 0, K-Means-Monotonie, AUROC = P(S⁺>S⁻), F₁-Identität, R²-Zerlegung, Adam-Bias-Korrektur, KL-Term und Reparametrisierung.
- Neue Code-Snippets: Regularisierung (Ridge/Lasso), Data Leakage, EM-Monotonie.
