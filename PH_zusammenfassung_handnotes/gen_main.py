"""Erzeugt Reihenfolge, Inhaltsverzeichnisse und die beiden Hauptdateien:
   main.tex          (Langfassung: Übersichten + Vertiefungen)
   main_kompakt.tex  (Kompaktfassung: nur Übersichten + Spickzettel)"""
import re, os, json
B = os.path.dirname(os.path.abspath(__file__)) + "/"
V = "V"   # Kennzeichen Vertiefungsseite

# (Datei, Titel); Übersichtsseiten werden nummeriert, Vertiefungen (V) folgen direkt dahinter
parts = [
 ("Grundlagen", "hnteal", [("g01_wahrscheinlichkeit","Wahrscheinlichkeit \\& Statistik","G1"),("g02_linalg","Lineare Algebra","G2"),("g03_optimierung","Optimierung","G3")]),
 ("Überwachtes Lernen", "hnorange", [
   ("p01_linreg","Lineare Regression"),("v01_linreg",V),
   ("p02_logreg","Logistische Regression"),("p03_glm","Generalisierte Lineare Modelle"),("v02_logreg_glm",V),
   ("p04_generativ","Generative Modelle (GDA, NB)"),("v03_generativ",V),
   ("p05_kernel","Kernel-Methoden"),
   ("p06_svm","Support Vector Machines"),("v04_svm",V),("v04b_svm_smo",V),
   ("p07_perceptron","Perceptron \\& Online Learning"),("p30_knn","k-Nearest Neighbors"),
   ("p08_trees","Decision Trees"),("p09_ensemble","Ensemble-Methoden"),("v05_knn_trees_ens",V),
   ("p10_loss","Verlustfunktionen")]),
 ("Deep Learning", "hnpurple", [
   ("p11_nn","Neuronale Netze"),("p12_backprop","Backpropagation"),("v06_backprop",V),("v06b_backprop_code",V),
   ("p36_dlpraxis","DL-Praxis: Training"),("p37_cnn_rnn","CNN \\& RNN / LSTM"),("v07_dl_code",V),("v07b_dl_lstm",V),
   ("p25_foundation","Foundation Models")]),
 ("Generalisierung \\& Praxis", "hngreen", [
   ("p13_biasvar","Bias \\& Varianz"),("p14_lerntheorie","Lerntheorie"),("v08_theorie",V),
   ("p15_regularisierung","Regularisierung"),("v09_regularisierung",V),
   ("p34_tuning","Validierung \\& Tuning"),("p16_gp","Gaussian Processes"),("p17_fehleranalyse","Fehleranalyse"),
   ("p18_metriken","Klassifikationsmetriken"),("p32_regmetriken","Regressionsmetriken"),("v10_metriken",V),
   ("p39_shap","Feature Importance \\& SHAP"),("p39b_shap_vertiefung",V),("p39c_shap_code",V),
   ("p40_fairness","Fairness-Analyse"),("p40b_fairness_vertiefung",V),("p40c_fairness_code",V),
   ("p33_features","Feature Engineering"),("p35_workflow","ML-Workflow"),("v11_workflow_code",V)]),
 ("Unüberwachtes Lernen", "hnrose", [
   ("p19_kmeans","K-Means"),("p31_hierarchisch","Hierarchisches Clustering"),("v12_clustering",V),
   ("p20_em","EM-Algorithmus"),("v13_em",V),("p21_fa","Faktorenanalyse"),("p22_hmm","Hidden Markov Models"),
   ("p23_pca","PCA"),("p24_ica","ICA"),("v14_pca_ica",V),
   ("p38_vae_gan","Generative Modelle: VAE, GAN"),("v15_vae_gan",V),("p26_kritik","Kritische Perspektiven")]),
 ("Bestärkendes Lernen", "hnnavy", [
   ("p27_mdp","MDPs \\& Value Iteration"),("p28_lqr","LQR, DDP \\& LQG"),("p29_policygradient","Policy Gradient"),("v16_rl",V),("v16b_rl_lqr",V)]),
]
anhang = [("s01_algotabelle","S1","Algorithmen-Tabelle"),("s02_formeln","S2","Formel-Spickzettel")]

# Titel der Vertiefungen aus der Kopfzeile ("X: Vertiefung")
def v_title(f):
    t = open(B + "pages/%s.tex" % f).read()
    m = re.search(r"\\hnheader\[.*?\]\[.*?\]\{(.*?)\}\{(.*?)\}\{(.*?)\}\n", t, re.S)
    return m.group(1).rstrip(":").strip() if m else f

# Skalen aus vorhandener main.tex übernehmen
scales = {}
for fn in ("main.tex", "main_kompakt.tex"):
    if os.path.exists(B + fn):
        for m in re.finditer(r"\\hnscale\{([\d.]+)\}\\input\{pages/(\w+)\}", open(B + fn).read()):
            scales.setdefault(m.group(2), m.group(1))

# Manuelle Skalen (haben Vorrang; Mindestziel ca. 1.1, damit Text und Formeln lesbar bleiben)
scales.update({"p01_linreg": "1.12", "p06_svm": "1.10", "p04_generativ": "1.08", "p36_dlpraxis": "1.15", "p02_logreg": "1.15", "p09_ensemble": "1.10",
               "p18_metriken": "1.10", "p40_fairness": "1.10", "p39_shap": "1.10", "p39b_shap_vertiefung": "1.19",
               "p39c_shap_code": "1.19", "p40b_fairness_vertiefung": "1.19", "p40c_fairness_code": "1.19", "v02_logreg_glm": "1.10", "v04b_svm_smo": "1.19", "v06b_backprop_code": "1.19", "v07b_dl_lstm": "1.19", "v16b_rl_lqr": "1.19", "v08_theorie": "1.10", "v10_metriken": "1.10", "c01_toc": "1.55", "c02_toc": "1.45"})

# ------------------------------------------------------------------ Nummerierung
entries = []   # dict(file,label,title,part,acc,v)
n = 0
for pt, acc, its in parts:
    for it in its:
        if pt == "Grundlagen":
            f, ti, lab = it; entries.append(dict(file=f, label=lab, title=ti, part=pt, acc=acc, v=False)); continue
        f, ti = it
        if ti == V:
            entries.append(dict(file=f, label="+", title="Vertiefung: " + v_title(f), part=pt, acc=acc, v=True))
        else:
            n += 1; entries.append(dict(file=f, label=str(n), title=ti, part=pt, acc=acc, v=False))
n_over = n
n_vert = sum(1 for e in entries if e["v"])
# "p39b"/"p40b" tragen schon einen Vertiefungs-Titel
for e in entries:
    if e["v"] and e["file"] in ("p39b_shap_vertiefung", "p39c_shap_code", "p40b_fairness_vertiefung", "v04b_svm_smo", "v06b_backprop_code", "v07b_dl_lstm", "v16b_rl_lqr", "p40c_fairness_code"):
        e["title"] = "Vertiefung: " + {"p39b_shap_vertiefung": "SHAP-Herleitung", "p39c_shap_code": "SHAP in der Praxis", "p40b_fairness_vertiefung": "Fairness", "v04b_svm_smo": "SVM in der Praxis (SMO, Code)", "v06b_backprop_code": "Backpropagation in der Praxis", "v07b_dl_lstm": "LSTM \\& Parameterzahlen", "v16b_rl_lqr": "LQR-Riccati \\& Tricks", "p40c_fairness_code": "Fairness in der Praxis"}[e["file"]]

def sel(kompakt): return [e for e in entries if not (kompakt and e["v"])]

tocl_def = r"""% Inhaltsverzeichnis-Zeilen (in preamble.tex definiert): \tocl{Label}{Datei}{Titel}, \tocv{Datei}{Titel}
"""

def tocpanel(pt, acc, items, roman, span=6):
    half = (len(items) + 1) // 2
    L, R = items[:half], items[half:]
    def ent(e):
        return ("\\tocv{%s}{%s}" if e["v"] else "\\tocl{%s}{%s}{%s}") % ((e["file"], e["title"]) if e["v"] else (e["label"], e["file"], e["title"]))
    lines = []
    for i in range(half):
        s = ent(L[i])
        if i < len(R): s += "\\hfill" + ent(R[i])
        lines.append(s)
    return "\\begin{hnp}[raster multicolumn=%d,acc=%s]{%s}{%s}\n%s\n\\end{hnp}\n%%\n" % (span, acc, roman, pt, "\\\\[3pt]\n".join(lines))

def tocsingle(pt, acc, items, roman, span=3):
    def ent(e):
        return ("\\tocvw{%s}{%s}" if e["v"] else "\\tocw{%s}{%s}{%s}") % ((e["file"], e["title"]) if e["v"] else (e["label"], e["file"], e["title"]))
    return "\\begin{hnp}[raster multicolumn=%d,acc=%s]{%s}{%s}\n%s\n\\end{hnp}\n%%\n" % (span, acc, roman, pt, "\n\\vspace{2pt}\n".join(ent(e) for e in items))

def items_of(pt, kompakt):
    return [e for e in sel(kompakt) if e["part"] == pt]

def accof(pt): return [a for p, a, _ in parts if p == pt][0]

def build_tocs(kompakt):
    hint = ("Kompaktfassung: nur die Übersichtsseiten. Alle Herleitungen und Code-Beispiele stehen in der Langfassung."
            if kompakt else "Alles auf einen Blick: Übersichtsseiten (Nummern) und dahinter Vertiefungen (+) -- alle Einträge sind anklickbar.")
    c1 = "\\hnheader[%s][Erst Grundlagen, dann Modelle, dann Praxis, dann Herleitungen]{Inhalt}{\\& Übersicht}{Grundlagen, Überwachtes Lernen, Deep Learning, Praxis}\n\\begin{hngrid}\n%%\n" % hint
    c1 += tocpanel("Grundlagen", accof("Grundlagen"), items_of("Grundlagen", kompakt), "0")
    c1 += tocpanel("Überwachtes Lernen", accof("Überwachtes Lernen"), items_of("Überwachtes Lernen", kompakt), "I")
    c1 += tocpanel("Deep Learning", accof("Deep Learning"), items_of("Deep Learning", kompakt), "II")
    c1 += tocpanel("Generalisierung \\& Praxis", accof("Generalisierung \\& Praxis"), items_of("Generalisierung \\& Praxis", kompakt), "III")
    c1 += "\\end{hngrid}\n"
    c2 = "\\hnheader[Weiter: unüberwachtes und bestärkendes Lernen, Spickzettel und die Landkarte des Kurses.][Modell wählen $\\to$ Verlust $\\to$ optimieren $\\to$ prüfen!]{Inhalt \\&}{Landkarte}{Teil IV--V, Anhang, roter Faden}\n\\begin{hngrid}\n%\n"
    c2 += tocpanel("Unüberwachtes Lernen", accof("Unüberwachtes Lernen"), items_of("Unüberwachtes Lernen", kompakt), "IV")
    c2 += tocsingle("Bestärkendes Lernen", accof("Bestärkendes Lernen"), items_of("Bestärkendes Lernen", kompakt), "V")
    c2 += "\\begin{hnp}[raster multicolumn=3,acc=hnteal]{A}{Anhang: Spickzettel}\n" + "\n\\vspace{2pt}\n".join("\\tocw{%s}{%s}{%s}" % (l, f, t) for f, l, t in anhang) + "\n\\end{hnp}\n%\n"
    workflow = r"""\begin{hnp}[raster multicolumn=6,acc=hnnavy]{$\star$}{Der ML-Workflow (roter Faden)}
\centering
\begin{tikzpicture}[>=Stealth,every node/.style={draw=hnnavy,rounded corners=2pt,font=\scriptsize,align=center,inner sep=3.5pt,minimum height=9mm,minimum width=2.1cm}]
  \node[fill=hnorange!20] (a) at (0,0) {1 Problem\\definieren};
  \node[fill=hnmint] (b) at (2.5,0) {2 Daten\\sammeln};
  \node[fill=hnyellow] (c) at (5,0) {3 Modell \&\\Verlust wählen};
  \node[fill=hnblue] (d) at (7.5,0) {4 Optimieren\\(GD, EM, \dots)};
  \node[fill=hnpink] (e) at (10,0) {5 Bewerten\\(Metriken, CV)};
  \node[fill=hnlilac] (f) at (12.5,0) {6 Verbessern\\(Bias/Varianz)};
  \foreach \p/\q in {a/b,b/c,c/d,d/e,e/f}{\draw[->,thick,hnorange] (\p)--(\q);}
  \draw[->,thick,hnorange,dashed] (f.south) -- ++(0,-.6) -| (c.south);
\end{tikzpicture}
\end{hnp}
%
\begin{hnp}[raster multicolumn=3,acc=hnpurple]{$\star$}{Drei Lernparadigmen}
\begin{itemize}
\item \textbf{Überwacht} $(x,y)$: Vorhersage lernen (Regression, Klassifikation).
\item \textbf{Unüberwacht} $x$: Struktur finden (Cluster, Dimensionsreduktion, Sequenzen, Generative Modelle).
\item \textbf{Bestärkend}: Aktionen wählen, um Belohnung zu maximieren.
\end{itemize}
\end{hnp}
%
"""
    lese = ("\\item \\colorbox{hnmint}{Beispiel} mit Rechnung \\& \\hnsol{} (alle Zahlen maschinell geprüft)\n"
            "\\item \\colorbox{hnblue}{Pseudo-Code}, \\colorbox{hnyellow}{Merke-Zettel}, Key Takeaways, \\colorbox{hnlilac}{Selbsttest}\n"
            + ("\\item \\emph{Kompaktfassung}: Herleitungen und Code stehen in der Langfassung.\n" if kompakt else
               "\\item \\textbf{+ Vertiefung}: Herleitung Schritt für Schritt und getesteter Code direkt hinter dem Thema\n")
            + "\\item \\colorbox{hnorange!20}{Zusatzinfo (Internet)} = nicht aus den Notes; \\emph{Quelle:} unter jedem Titel\n")
    workflow += "\\begin{hnp}[raster multicolumn=3,acc=hngreen]{$\\star$}{So liest du diese Notizen}\n\\begin{itemize}\n" + lese + "\\end{itemize}\n\\end{hnp}\n%\n"
    notation = r"""\begin{hnp}[raster multicolumn=6,acc=hnnavy]{$\Sigma$}{Notation: was die Symbole überall bedeuten}
\hnsmall\raggedright
$n$ Anzahl Beispiele, $d$ Anzahl Merkmale, $x^{(i)}\in\R^d$ Merkmalsvektor, $y^{(i)}$ Label des $i$-ten Beispiels, $X\in\R^{n\times d}$ Design-Matrix (eine Zeile pro Beispiel), $\theta$ Modellparameter, $h_\theta(x)$ bzw.\ $\hat y$ Vorhersage, $J(\theta)$ Kosten (zu minimieren), $\ell(\theta)$ Log-Likelihood (zu maximieren), $\alpha$ Lernrate, $\lambda$ Regularisierungsstärke, $\nabla$ Gradient (Vektor der Ableitungen), $\E$ Erwartungswert, $\N(\mu,\Sigma)$ Normalverteilung, $\I\{\cdot\}$ Indikator (1, wenn wahr), $\sigma$ Sigmoid bzw.\ Standardabweichung (je nach Kontext), $A\T$ transponiert, $\|\cdot\|$ Länge (Norm), $\odot$ elementweises Produkt, $\mathcal H$ Hypothesenraum, $K$ Anzahl Klassen bzw.\ Cluster. Wird ein Symbol lokal anders benutzt, steht eine \emph{Legende} direkt im Panel.
\end{hnp}
%
"""
    workflow += notation
    workflow += "\\begin{hnbanner}[raster multicolumn=6]\n„Modell, Verlust, Optimierung -- die drei Fragen hinter jedem Algorithmus dieser Vorlesung.“\n\\end{hnbanner}\n\\end{hngrid}\n"
    c2 += workflow
    return c1, c2

def cover(kompakt):
    t = open(B + "pages/c00_cover_base.tex").read()
    if kompakt:
        feat = "2.7/%d Kapitelseiten/Übersicht je Thema, 7.3/Spickzettel/Algorithmen \\& Formeln, 11.9/Beispiele/mit Rechnung \\& Lösung, 16.5/Selbsttests/drei Fragen pro Seite" % n_over
        t = t.replace("Die komplette Vorlesung als Lernzusammenfassung", "Kompaktfassung: die Übersichtsseiten zum Wiederholen")
    else:
        feat = "2.7/%d Kapitelseiten/+ %d Vertiefungen, 7.3/Herleitungen/Schritt für Schritt, 11.9/Code/getestet \\& ausgeführt, 16.5/Selbsttests/drei Fragen pro Seite" % (n_over, n_vert)
    return t.replace("@@FEATURES@@", feat)

def write_main(kompakt):
    tag = "_kompakt" if kompakt else ""
    c1, c2 = build_tocs(kompakt)
    open(B + "pages/c01_toc%s.tex" % tag, "w").write(c1)
    open(B + "pages/c02_toc%s.tex" % tag, "w").write(c2)
    open(B + "pages/c00_cover%s.tex" % (tag if kompakt else "_full"), "w").write(cover(kompakt))
    lines = ["\\input{preamble}", "\\begin{document}", "\\shorthandoff{\"}", "\\input{pages/c00_cover%s}\\clearpage" % ("_kompakt" if kompakt else "_full")]
    def add(f, title, level, name=None):
        sc = scales.get(f, "1.0")
        lines.append("\\phantomsection\\pdfbookmark[%d]{%s}{bm:%s}\\label{pg:%s}\\hnscale{%s}\\input{pages/%s}\\clearpage"
                     % (level, title, f, f, sc, name or f))
    add("c01_toc", "Inhalt", 0, "c01_toc%s" % tag); add("c02_toc", "Inhalt \\& Landkarte", 0, "c02_toc%s" % tag)
    for e in sel(kompakt):
        add(e["file"], ("%s %s" % (e["label"], e["title"])), 2 if e["v"] else 1)
    for f, l, t in anhang: add(f, "%s %s" % (l, t), 1)
    lines[-1] = lines[-1].replace("\\clearpage", "")
    lines.append("\\end{document}")
    open(B + "main%s.tex" % tag, "w").write("\n".join(lines) + "\n")

write_main(False); write_main(True)
json.dump([e["file"] for e in entries] + [f for f, _, _ in anhang] + ["c01_toc", "c02_toc"], open(B + "pages_order.json", "w"))
print("Uebersichtsseiten:", n_over, "Vertiefungen:", n_vert, "Gesamt Langfassung:", 3 + len(entries) + len(anhang) - 1 + 1)
