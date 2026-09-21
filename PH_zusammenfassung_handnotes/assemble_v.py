"""Einmal-Werkzeug: baut die Vertiefungsseiten v*.tex aus alten d*-Panels, Code-Snippets und neuen Panels."""
import re, os
B = os.path.dirname(os.path.abspath(__file__)) + "/"
P = B + "pages/"

def read(f): return open(P + f).read()

def blocks_of(text):
    body = text[text.index("\\begin{hngrid}") + len("\\begin{hngrid}"):text.index("\\end{hngrid}")]
    out, pos = [], 0
    pat = re.compile(r"\\begin\{(hnp|hnex)\}.*?\\end\{\1\}", re.S)
    for m in pat.finditer(body):
        out.append(m.group(0))
    return out

def title_of(b):
    m = re.match(r"\\begin\{hnp\}\[[^\]]*\]\{[^}]*\}\{(.*?)\}\n", b, re.S)
    if m: return m.group(1)
    m = re.match(r"\\begin\{hnex\}\[[^\]]*\]\{(.*?)\}\n", b, re.S)
    return m.group(1) if m else ""

def take(dfile, *subs):
    bl = blocks_of(read(dfile)); res = []
    for s in subs:
        hit = [b for b in bl if s in title_of(b)]
        assert hit, (dfile, s, [title_of(b) for b in bl])
        res.append(hit[0])
    return res

def quiz(q):  # q = 6 Strings
    return ("\\begin{hnp}[raster multicolumn=6,acc=hnpurple,colback=hnlilac!30]{?}{Selbsttest: kannst du das herleiten?}\n"
            "\\hnquiz" + "".join("{%s}" % s for s in q) + "\n\\end{hnp}")

def snip(name, title, acc="hnorange"):
    return "\\hnsnip[%s]{%s}{%s}" % (acc, name, title)

def span(b):
    if b.startswith("\\hnsnip"): return 6
    m = re.search(r"raster multicolumn=(\d)", b)
    return int(m.group(1)) if m else 6

def setspan(b, n):
    return re.sub(r"raster multicolumn=\d", "raster multicolumn=%d" % n, b, count=1)

def pack(bl):
    rows, cur, s = [], [], 0
    def close():
        nonlocal cur, s
        if cur and s < 6 and not cur[-1].startswith("\\hnsnip"):
            cur[-1] = setspan(cur[-1], span(cur[-1]) + 6 - s)
        rows.append(cur); cur, s = [], 0
    for b in bl:
        w = span(b)
        if s + w > 6: close()
        cur.append(b); s += w
        if s == 6: close()
    close()
    return [b for r in rows for b in r]

def renumber(bl):
    n = 0; out = []
    for b in bl:
        def f(m):
            nonlocal n
            n += 1
            return "%s{%d}" % (m.group(1), n)
        out.append(re.sub(r"(\\begin\{hnp\}\[[^\]]*\])\{\d+\}", f, b, count=1))
    return out

def page(fname, head, src, blocks, banner=None):
    bl = renumber(pack(blocks))
    txt = head + "\n\\hnsrc{" + src + "}\n\n\\begin{hngrid}\n%\n" + "\n%\n".join(bl) + "\n%\n"
    if banner:
        txt += "\\begin{hnbanner}[raster multicolumn=6]\n" + banner + "\n\\end{hnbanner}\n"
    txt += "\\end{hngrid}\n"
    open(P + fname, "w").write(txt)

def hdr(bubble, note, t1, t2, sub):
    return "\\hnheader[%s][%s]{%s}{%s}{%s}" % (bubble, note, t1, t2, sub)

# ------------------------------------------------------------------ neue Panels
gda_logistic = r"""\begin{hnp}[raster multicolumn=3,acc=hnnavy]{0}{GDA $\Rightarrow$ logistische Form}
Gleiches $\Sigma$, Prior $\phi=P(y{=}1)$. Log-Odds per Bayes und Gauß-Dichten:
\begin{align*}
\log\tfrac{p(y{=}1|x)}{p(y{=}0|x)}&=\log\tfrac{p(x|1)}{p(x|0)}+\log\tfrac{\phi}{1-\phi}\\
\log\tfrac{p(x|1)}{p(x|0)}&=-\tfrac12(x{-}\mu_1)\T\Sigma^{-1}(x{-}\mu_1)+\tfrac12(x{-}\mu_0)\T\Sigma^{-1}(x{-}\mu_0)\\
&=(\mu_1{-}\mu_0)\T\Sigma^{-1}x-\tfrac12\bigl(\mu_1\T\Sigma^{-1}\mu_1-\mu_0\T\Sigma^{-1}\mu_0\bigr)
\end{align*}
Der quadratische Term $x\T\Sigma^{-1}x$ \emph{kürzt sich} (gleiches $\Sigma$). Also $p(y{=}1|x)=\sigma(\theta\T x+\theta_0)$ mit $\theta=\Sigma^{-1}(\mu_1-\mu_0)$.
\end{hnp}"""
gda_ex = r"""\begin{hnex}[raster multicolumn=3]{Beispiel: GDA-Posterior nachgerechnet}
$\mu_0=(0,0)$, $\mu_1=(2,1)$, $\Sigma=\begin{psmallmatrix}1&0.3\\0.3&1\end{psmallmatrix}$, $\phi=0.4$, $x=(1.5,1)$.\\
\hnsol\ $\theta=\Sigma^{-1}(\mu_1-\mu_0)=(1.868,\,0.440)$, $\theta_0=-2.493$.\\
$z=\theta\T x+\theta_0=\ora{0.748}\Rightarrow p(y{=}1|x)=\sigma(0.748)=\ora{0.679}$ \hlm{(maschinell geprüft: direkte Bayes-Rechnung liefert dasselbe)}.
\end{hnex}"""
ig_panel = r"""\begin{hnp}[raster multicolumn=3,acc=hnorange]{0}{Information Gain $\ge0$ (Jensen)}
Elternverteilung $p=\sum_kw_kp_k$ ist der \emph{gewichtete Mittelwert} der Kinderverteilungen ($w_k=|R_k|/|R|$). Entropie $H$ ist konkav, also
\[ H(p)=H\Bigl(\sum_kw_kp_k\Bigr)\ge\sum_kw_kH(p_k) \]
$\Rightarrow\mathrm{IG}=H(p)-\sum_kw_kH(p_k)\ge0$; strikt $>0$, wenn die Kinder verschieden sind ($H$ \emph{strikt} konkav). Die Fehlerrate $1-\max p$ ist konkav, aber nicht strikt: Gleichheit möglich, $\mathrm{IG}=0$ (Beispiel Kapitel Decision Trees).
\end{hnp}"""
adam_panel = r"""\begin{hnp}[raster multicolumn=3,acc=hnpurple]{0}{Adam: warum Bias-Korrektur?}
$m_0=0$, $m_t=\beta_1m_{t-1}+(1{-}\beta_1)g_t=(1{-}\beta_1)\sum_{i=1}^t\beta_1^{t-i}g_i$. Bei stationären Gradienten $\E g_i=g$:
\begin{align*}
\E[m_t]&=(1-\beta_1)\,g\sum_{i=1}^t\beta_1^{t-i}&&\why{Erwartungswert}\\
&=(1-\beta_1^{t})\,g&&\why{geometrische Summe}
\end{align*}
Der Schätzer ist am Anfang zu klein $\Rightarrow$ Korrektur $\hat m_t=m_t/(1-\beta_1^t)$ (analog $\hat v_t$). Beispiel $t=1$: $m_1=0.1\,g$, $\hat m_1=g$.
\end{hnp}"""
kmeans_panel = r"""\begin{hnp}[raster multicolumn=3,acc=hnorange]{0}{K-Means: $J$ sinkt in jedem Schritt}
\textbf{Zuweisung}: für festes $\mu$ minimiert $c^{(i)}=\arg\min_k\|x^{(i)}-\mu_k\|^2$ jeden Summanden $\Rightarrow J\downarrow$.\\
\textbf{Update}: für festes $c$ gilt
\begin{align*}
\nabla_{\mu_k}\!\!\sum_{i:c^{(i)}=k}\!\|x^{(i)}-\mu_k\|^2&=-2\!\!\sum_{i:c^{(i)}=k}\!(x^{(i)}-\mu_k)=0\\
\Rightarrow\;\mu_k&=\tfrac1{n_k}\textstyle\sum_{i:c^{(i)}=k}x^{(i)}&&\why{Hesse $2n_kI\succ0$}
\end{align*}
$J\ge0$ nicht steigend, endlich viele Zuweisungen $\Rightarrow$ Konvergenz (\emph{lokales} Minimum, nichtkonvex).
\end{hnp}"""
kmeans_ex = r"""\begin{hnex}[raster multicolumn=3]{Beispiel: Der Mittelwert minimiert}
Punkte $\{1,2,9\}$, ein Cluster, Zentroid $\mu$.\\
\hnsol\ $J(\mu)=\sum(x-\mu)^2$: $J(2)=1+0+49=50$, $J(\mu{=}4)=9+4+25=\ora{38}$ (Mittel), $J(6)=25+16+9=50$.\\
Das Minimum liegt beim arithmetischen Mittel $4$ (numerisch geprüft).
\end{hnex}"""
auc_panel = r"""\begin{hnp}[raster multicolumn=3,acc=hnorange]{0}{AUROC $=P(S^+>S^-)$}
Schwelle $t$: $\mathrm{TPR}(t)=P(S^+>t)$, $\mathrm{FPR}(t)=P(S^->t)$ mit Scores positiver/negativer Beispiele.
\begin{align*}
\mathrm{AUC}&=\int\mathrm{TPR}\;d\mathrm{FPR}=\int P(S^+>t)\,f_{S^-}(t)\,dt&&\why{$d\mathrm{FPR}=-f_{S^-}dt$}\\
&=\E_{S^-}\bigl[P(S^+>S^-)\bigr]=P(S^+>S^-)
\end{align*}
Empirisch: Anteil der (pos., neg.)-Paare mit $s^+>s^-$ (Gleichstand zählt $\frac12$) -- die \emph{Mann--Whitney-$U$-Statistik}. Nur das \emph{Ranking} zählt, deshalb unabhängig von der Prävalenz.
\end{hnp}"""
f1_panel = r"""\begin{hnp}[raster multicolumn=3,acc=hnpurple]{0}{$F_1$ als harmonisches Mittel}
\begin{align*}
F_1&=\frac{2PR}{P+R}=\frac{2\cdot\frac{TP}{TP+FP}\cdot\frac{TP}{TP+FN}}{\frac{TP}{TP+FP}+\frac{TP}{TP+FN}}&&\why{einsetzen}\\
&=\frac{2\,TP^2}{TP\,(TP+FN)+TP\,(TP+FP)}=\frac{2\,TP}{2\,TP+FP+FN}&&\why{$TP$ kürzen}
\end{align*}
$TN$ kommt nicht vor. Harmonisch, weil es Ungleichgewicht bestraft: $P=1,\,R=0.1$ ergibt $F_1=0.18$, das arithmetische Mittel $0.55$.
\end{hnp}"""
r2_panel = r"""\begin{hnp}[raster multicolumn=3,acc=hngreen]{0}{$R^2\in[0,1]$ (OLS mit Achsenabschnitt)}
\begin{align*}
\mathrm{SST}&=\textstyle\sum(y_i-\bar y)^2=\sum\bigl[(\hat y_i-\bar y)+(y_i-\hat y_i)\bigr]^2\\
&=\mathrm{SSR}+\mathrm{SSE}+2\textstyle\sum(\hat y_i-\bar y)(y_i-\hat y_i)&&\why{ausmultiplizieren}\\
&=\mathrm{SSR}+\mathrm{SSE}&&\why{Residuen $\perp$ Spalten von $X$}
\end{align*}
Das Kreuzterm-Null folgt aus den Normalgleichungen ($X\T e=0$; $\mathbf 1$ ist Spalte $\Rightarrow\sum e_i=0$). Also $R^2=\frac{\mathrm{SSR}}{\mathrm{SST}}=1-\frac{\mathrm{SSE}}{\mathrm{SST}}$.
\end{hnp}"""
auc_ex = r"""\begin{hnex}[raster multicolumn=3]{Beispiel: AUROC durch Paare zählen}
10 positive, 10 negative Beispiele $\Rightarrow$ $100$ Paare.\\
\hnsol\ Mit den Scores aus dem Code ergibt das Auszählen \ora{93.5} von 100 Paaren (Gleichstände halb) $\Rightarrow$ $\mathrm{AUROC}=0.935$ \hlm{(identisch zu \texttt{roc\_auc\_score})}.\\
$F_1$: $TP=9$, $FP=2$, $FN=1$: $\frac{2\cdot9}{18+2+1}=\ora{0.857}$.
\end{hnex}"""
kl_panel = r"""\begin{hnp}[raster multicolumn=3,acc=hnorange]{0}{KL-Term in geschlossener Form}
$Q=\N(\mu,\sigma^2)$, $P=\N(0,1)$, $z\sim Q$:
\begin{align*}
\mathrm{KL}(Q\|P)&=\E_Q\bigl[\log Q(z)-\log P(z)\bigr]\\
&=\E\Bigl[-\tfrac12\log\sigma^2-\tfrac{(z-\mu)^2}{2\sigma^2}+\tfrac{z^2}{2}\Bigr]&&\why{$\log2\pi$ kürzt sich}\\
&=-\tfrac12\log\sigma^2-\tfrac12+\tfrac12(\mu^2+\sigma^2)&&\why{$\E(z{-}\mu)^2=\sigma^2$, $\E z^2=\mu^2+\sigma^2$}
\end{align*}
$=\tfrac12\bigl(\mu^2+\sigma^2-1-\log\sigma^2\bigr)$, im Vektorfall summiert über die Dimensionen.
\end{hnp}"""
reparam_panel = r"""\begin{hnp}[raster multicolumn=3,acc=hnpurple]{0}{Warum die Reparametrisierung funktioniert}
$\E_{z\sim\N(\mu,\sigma^2)}[f(z)]=\E_{\xi\sim\N(0,1)}[f(\mu+\sigma\xi)]$, denn $z=\mu+\sigma\xi$. Jetzt hängt die \emph{Verteilung} nicht mehr von $\mu,\sigma$ ab, nur die Funktion:
\[ \nabla_\mu=\E_\xi\bigl[f'(\mu+\sigma\xi)\bigr],\qquad\nabla_\sigma=\E_\xi\bigl[f'(\mu+\sigma\xi)\,\xi\bigr] \]
\emph{Check} $f(z)=z^2$: $\E f=\mu^2+\sigma^2$, also $\nabla_\mu=2\mu$ und $\nabla_\sigma=2\sigma$; rechts: $\E[2(\mu+\sigma\xi)]=2\mu$, $\E[2(\mu+\sigma\xi)\xi]=2\sigma$ \hlm{stimmt}. Der Gradient darf so in den Erwartungswert (Backprop durch das Sampling).
\end{hnp}"""
vae_lesart = r"""\begin{hnp}[raster multicolumn=6,acc=hnteal]{0}{Lesart der Ausgaben}
\textbf{VAE}: die Rekonstruktion fällt von $9.8$ auf $2.3$, der KL-Term steigt von $0.6$ auf $1.6$ -- der latente Code trägt zunehmend Information (Kompromiss der ELBO). \textbf{GAN}: der Generator lernt die Zielverteilung $\N(3,0.5^2)$: Mittel $3.02$ passt, die Standardabweichung $0.68$ ist noch zu breit (nach $3000$ Schritten, kurzes Training). Ein GAN hat \emph{keinen} monoton fallenden Verlust -- die Qualität prüft man an den Samples.
\end{hnp}"""
leak_lesart = r"""\begin{hnp}[raster multicolumn=6,acc=hnteal]{0}{Lesart der Ausgabe: Data Leakage im Zahlenbeispiel}
Die Daten sind \emph{reines Rauschen} (Labels zufällig): ehrlich erreichbar sind $\approx0.5$. Wählt man die $20$ ``besten'' von $1000$ Features auf dem \emph{gesamten} Datensatz, findet man Features, die zufällig mit $y$ korrelieren -- die Kreuzvalidierung meldet trügerische $0.88$. Steht die Auswahl in der Pipeline, wird sie in jedem Fold nur auf den Trainingsdaten gelernt: $0.57\approx$ Zufall. Je mehr Features und je weniger Beispiele, desto größer der Leakage-Effekt. Zweites Snippet: Nested CV über Pipeline und Hyperparameter (Kapitel Validierung \& Tuning).
\end{hnp}"""
em_lesart = r"""\begin{hnp}[raster multicolumn=6,acc=hnteal]{0}{Lesart der Ausgabe}
Die Log-Likelihood steigt in jeder Iteration (\texttt{steigt: True}) -- das beweist die Herleitung oben: E-Schritt macht die ELBO scharf, M-Schritt hebt sie, ELBO $\le\log p$. Der Anstieg ist nicht gleichmäßig (langsam bei Iteration 1--5, dann schneller); EM konvergiert zu einem \emph{lokalen} Maximum, die Startwerte (\texttt{init\_params}) entscheiden.
\end{hnp}"""
metr_lesart = r"""\begin{hnp}[raster multicolumn=6,acc=hnteal]{0}{Lesart der Ausgabe}
\texttt{confusion\_matrix} liefert \emph{Zeilen $=$ Wahrheit} ($[[TN,FP],[FN,TP]]$), anders als die Tabellen in diesem Skript (Zeilen $=$ Vorhersage). Precision $0.82$, Recall $0.90$, $F_1=0.857$ entsprechen dem Beispiel oben; AUROC $0.935$ und AUPRC $0.939$ ändern sich nicht mit der Schwelle. Regression: MAE $0.54$, RMSE $0.557$, $R^2=0.874$ (Kapitel Regressionsmetriken).
\end{hnp}"""
dl_lesart = r"""\begin{hnp}[raster multicolumn=6,acc=hnteal]{0}{Lesart der Ausgaben}
\textbf{MLP}: Verlust $0.45\to0.01$, Accuracy $0.86\to0.97$ auf den Moons-Daten (Trainingsdaten, mit Dropout im Training). \textbf{CNN}: $28{\times}28\to14{\times}14\to7{\times}7$, Conv1 hat $80$ Parameter, insgesamt $9098$ (der FC-Teil $16\cdot49\cdot10+10=7850$ dominiert). \textbf{LSTM}: $1523$ Parameter für Eingabe $5$, versteckt $16$: $4\cdot16\cdot(5+16+1)=1408$ plus Kopf $16\cdot3+3=51$ plus zweiter Bias-Vektor $64$.
\end{hnp}"""
reg_lesart = r"""\begin{hnp}[raster multicolumn=6,acc=hnteal]{0}{Lesart der Ausgabe}
Nur $4$ von $20$ Features sind informativ. \textbf{Ridge} schrumpft die Norm ($77.9\to52.7$), setzt aber \emph{keinen} Koeffizienten auf null. \textbf{Lasso} setzt $16$ von $20$ Koeffizienten \emph{exakt} auf null; übrig bleiben $4$, so viele wie informative Features (Sparsität, Herleitung: Soft-Thresholding). $\alpha$ steuert die Stärke; sklearn nennt den Regularisierungsparameter \texttt{alpha} (bei \texttt{LogisticRegression}: $C=1/\lambda$).
\end{hnp}"""

# ------------------------------------------------------------------ Seiten
d01 = "d01_regression.tex"; d02 = "d02_theorie.tex"; d03 = "d03_svm.tex"; d04 = "d04_em_pca.tex"
d05 = "d05_backprop.tex"; d06 = "d06_rl.tex"; d07 = "d07_reg_opt.tex"

# V01 Lineare Regression
page("v01_linreg.tex",
 hdr("Warum ist die Normalgleichung optimal -- und warum ist Least Squares der MLE?","Kein Gradient ohne Herleitung: Produktregel, Konvexität, Log-Likelihood","Lineare Regression:","Vertiefung","Herleitung, Rechenbeispiel und Code"),
 "main\\_notes.pdf Kap.\\ 1 (LMS, Normalgleichung, probabilistische Interpretation); Herleitung ausformuliert; Code: code/np\\_gd.py, code/sk\\_linreg.py (real ausgeführt)",
 take(d01,"Normalgleichung","Warum Least Squares") + [snip("np_gd","Gradient Descent in NumPy vs.\\ Normalgleichung","hnorange"), snip("sk_linreg","Lineare Regression und Ridge in scikit-learn","hnpurple"),
 quiz(["Warum ist der stationäre Punkt der globale Minimierer?","$J$ ist konvex: $\\nabla^2J=X\\T X\\succeq0$.","Warum ist Least Squares der MLE?","Bei Gauß-Rauschen ist die Log-Likelihood $\\text{const}-\\frac1{2\\sigma^2}\\sum(y-\\theta\\T x)^2$.","Wann GD statt Normalgleichung?","Sehr großes $d$ (Inversion $O(d^3)$) oder singuläres $X\\T X$."])],
 "„Erst ableiten, dann verstehen, warum die Formel stimmt.“")

# V02 Logistische Regression + GLM
lg = take(d01,"Gradient der logistischen","Hesse","Softmax","Warum alle GLMs","Beispiel: Gradient")
lg[1] = lg[1].replace("Newton: $\\theta\\leftarrow\\theta-H^{-1}\\nabla\\ell=\\theta+(X\\T SX)^{-1}X\\T(y-h)$.","Newton folgt aus der Taylor-Näherung $\\ell(\\theta)\\approx\\ell_t+\\nabla\\ell\\T\\delta+\\frac12\\delta\\T H\\delta$: Ableiten nach $\\delta$ und Null setzen gibt $\\delta=-H^{-1}\\nabla\\ell$, also $\\theta\\leftarrow\\theta+(X\\T SX)^{-1}X\\T(y-h)$.")
page("v02_logreg_glm.tex",
 hdr("Warum sieht der Gradient so einfach aus? Herleitung für logistische Regression, Softmax und alle GLMs.","Ein Gradient, viele Modelle: (Vorhersage $-$ Wahrheit) $\\times$ Eingabe","Logistische Regression \\& GLM:","Vertiefung","Herleitung, Newton, Softmax und Code"),
 "main\\_notes.pdf Kap.\\ 2--3; Schritte ausformuliert und ergänzt; Code: code/sk\\_logreg.py (real ausgeführt)",
 lg + [snip("sk_logreg","Logistische Regression mit Skalierung in einer Pipeline","hnpurple"),
 quiz(["Wo taucht in $\\nabla\\ell=(y-h)x$ die Ableitung $g'=g(1-g)$ auf?","Sie kürzt sich mit den Nennern $h$ und $1-h$ der $\\log$-Ableitung heraus.","Warum ist die Log-Likelihood der LogReg konkav?","$H=-X\\T SX$ ist negativ semidefinit ($S\\succeq0$).","Wie sieht der Softmax-Gradient aus?","$(p_k-\\I\\{k{=}y\\})\\,x$ für jede Klasse $k$."])],
 "„Ein Gradient, viele Modelle: (Vorhersage $-$ Wahrheit) $\\times$ Eingabe.“")

# V03 Generative Modelle
page("v03_generativ.tex",
 hdr("MLE-Schätzer von Naive Bayes und GDA -- und warum GDA auf eine logistische Form führt.","Der quadratische Term kürzt sich, weil beide Klassen dasselbe $\\Sigma$ haben","Generative Modelle:","Vertiefung","Herleitung und Code"),
 "main\\_notes.pdf Kap.\\ 4 (GDA, Naive Bayes); Herleitungen ausformuliert; Code: code/sk\\_nb.py (real ausgeführt)",
 take(d04,"MLE für Naive Bayes") + [gda_logistic, gda_ex, snip("sk_nb","Multinomial Naive Bayes für Text mit Laplace-Glättung","hnteal"),
 quiz(["Wie lautet der MLE-Schätzer bei Bernoulli-Naive-Bayes?","$\\hat\\phi_{j|k}=\\#\\{x_j{=}1,y{=}k\\}/\\#\\{y{=}k\\}$ (Bernoulli-MLE $k/n$ je Klasse).","Warum ist der GDA-Posterior logistisch?","Bei gleichem $\\Sigma$ kürzt sich $x\\T\\Sigma^{-1}x$; die Log-Odds sind linear in $x$.","Wozu die Laplace-Glättung?","Verhindert $P=0$ bei ungesehenen Kombinationen ($+1$ im Zähler, $+2$ im Nenner)."])],
 "„Generativ modelliert, diskriminativ entschieden: bei Gauß-Klassen führen beide zur selben Grenze.“")

# V04 SVM
d03b = blocks_of(read(d03))
svm = [b for b in d03b if "Selbsttest" not in title_of(b) and "hnbanner" not in b[:20]]
qz = [b for b in d03b if "Selbsttest" in title_of(b)][0]
page("v04_svm.tex",
 hdr("Vom Margin zum Dual: Lagrange-Funktion, Stationarität und der Kernel-Trick.","Nur Skalarprodukte tauchen im Dual auf -- deshalb Kernels!","Support Vector Machines:","Vertiefung","Herleitung, Soft-Margin, SMO und Code"),
 "main\\_notes.pdf Kap.\\ 6 (Margins, Lagrange duality, dual form, SMO), materials/smo.pdf, section/cs229-cvxopt2.pdf; Schritte ergänzt; Code: code/sk\\_svm.py (real ausgeführt)",
 svm + [snip("sk_svm","SVM mit RBF-Kernel und Gitter-Suche für $C,\\gamma$","hnteal"), qz],
 "„Primal in $w$, Dual in $\\alpha$ -- und im Dual stecken nur noch Skalarprodukte.“")

# V05 kNN / Trees / Ensembles
page("v05_knn_trees_ens.tex",
 hdr("Warum Entropie-Splits nie schaden und warum Mittelung die Varianz senkt -- plus Code für k-NN, Baum und Ensembles.","Konkavität + Jensen erklärt Information Gain","k-NN, Bäume \\& Ensembles:","Vertiefung","Herleitung und Code"),
 "notes/cs229-notes-dt.pdf, notes/cs229-notes-ensemble.pdf (R.\\ Townshend); hand-notes/ML Notes.pdf (k-NN, Random Forest, Ensemble); Herleitungen ausformuliert; Code: code/sk\\_knn.py, sk\\_trees.py, sk\\_ensemble.py (real ausgeführt)",
 [ig_panel] + take(d02,"Varianz eines Ensembles") + [snip("sk_knn","k-Nearest Neighbors: Einfluss von $k$","hnpurple"), snip("sk_trees","Entscheidungsbaum vs.\\ Random Forest, Feature-Importance","hngreen"), snip("sk_ensemble","AdaBoost, Voting und Stacking","hnorange"),
 quiz(["Warum ist der Information Gain nie negativ?","Entropie ist konkav: $H(\\sum w_kp_k)\\ge\\sum w_kH(p_k)$ (Jensen).","Warum senkt Bagging die Varianz nicht auf null?","Es bleibt $\\rho\\sigma^2$; nur weniger Korrelation $\\rho$ hilft weiter (Random Forest).","Warum wird bei k-NN skaliert?","Ein Feature mit großer Skala dominiert den Abstand."])],
 "„Mehrere Bäume, weniger Varianz -- solange sie nicht alle dasselbe lernen.“")

# V06 Backpropagation
d05b = blocks_of(read(d05))
bp = [b for b in d05b if "Selbsttest" not in title_of(b)]
qz5 = [b for b in d05b if "Selbsttest" in title_of(b)][0]
page("v06_backprop.tex",
 hdr("Kettenregel rückwärts: wie Gradienten durch ein Netz fließen -- mit Autograd-Check.","$\\delta_{\\ell-1}=(W_\\ell\\T\\delta_\\ell)\\odot\\sigma'(z_{\\ell-1})$","Backpropagation:","Vertiefung","Herleitung, Beispiel und Code"),
 "main\\_notes.pdf Kap.\\ 7.4--7.5, notes/cs229-notes-backprop.pdf; Schritte und Zahlenbeispiel ergänzt; Code: code/torch\\_autograd.py (real ausgeführt)",
 bp + [snip("torch_autograd","Autograd: das Backprop-Beispiel von Hand nachgerechnet","hnorange"), qz5],
 "„Backprop ist keine Magie -- nur die Kettenregel mit gutem Buchhalter.“")

# V07 DL-Praxis / CNN / RNN Code
page("v07_dl_code.tex",
 hdr("Trainingsschleife, CNN und LSTM in PyTorch -- plus die Herleitung der Adam-Bias-Korrektur.","Backprop: \\texttt{loss.backward()} -- Update: \\texttt{opt.step()}","DL-Praxis, CNN \\& RNN:","Vertiefung","Herleitung und Code"),
 "main\\_notes.pdf (Deep Learning); Adam, Dropout, CNN, LSTM als Web-Ergänzung (markiert); Herleitung ausformuliert; Code: code/torch\\_mlp.py, torch\\_cnn.py, torch\\_lstm.py (real ausgeführt)",
 [adam_panel, dl_lesart, snip("torch_mlp","MLP mit Mini-Batches, Adam, Dropout und Weight Decay","hnpurple"), snip("torch_cnn","CNN: Formen und Parameterzahlen","hnteal"), snip("torch_lstm","LSTM-Sequenzklassifikator","hngreen"),
 quiz(["Warum teilt Adam durch $1-\\beta_1^t$?","$m_t$ startet bei $0$: $\\E m_t=(1-\\beta_1^t)g$ ist zu klein, die Korrektur skaliert zurück.","Wie viele Parameter hat Conv $3{\\times}3$ ($1\\to8$)?","$3\\cdot3\\cdot1\\cdot8+8=80$.","Was macht \\texttt{opt.zero\\_grad()}?","Löscht die akkumulierten Gradienten vor dem nächsten Backward-Pass."])],
 "„Frameworks rechnen die Gradienten -- du musst wissen, was sie tun.“")

# V08 Lerntheorie
d02b = blocks_of(read(d02))
lt = [b for b in d02b if "Selbsttest" not in title_of(b) and "Varianz eines Ensembles" not in title_of(b)]
qz2 = [b for b in d02b if "Selbsttest" in title_of(b)][0]
page("v08_theorie.tex",
 hdr("Bias--Varianz, Konzentrationsungleichungen und Generalisierungsschranken -- mit allen Zwischenschritten.","Trainingsfehler $\\to$ wahrer Fehler: die Brücke der Lerntheorie","Generalisierung \\& Lerntheorie:","Vertiefung","Herleitung, Hoeffding, Perceptron"),
 "main\\_notes.pdf Kap.\\ 8 (Bias-Varianz, Sample Complexity), extra-notes/hoeffding.pdf, notes/cs229-notes6.pdf; Schritte ergänzt",
 lt + [qz2],
 "„Erst zerlegen, dann abschätzen: Fehler = Rauschen + Bias + Varianz.“")

# V09 Regularisierung
page("v09_regularisierung.tex",
 hdr("Ridge, MAP und Lasso ausgerechnet -- Regularisierung ist ein Prior in Verkleidung.","$\\lambda=\\sigma^2/\\tau^2$: starker Prior $=$ starke Regularisierung","Regularisierung:","Vertiefung","Herleitung und Code"),
 "main\\_notes.pdf Kap.\\ 9 (Regularisierung, Bayes); Schritte ergänzt; Code: code/sk\\_regularization.py (real ausgeführt)",
 take(d07,"Ridge-Lösung","MAP","Lasso","Beispiel: Ridge") + [snip("sk_regularization","Ridge schrumpft, Lasso macht spärlich","hnorange"), reg_lesart,
 quiz(["Wie hängen $\\lambda$ und der Prior zusammen?","$\\lambda=\\sigma^2/\\tau^2$ (Rauschvarianz durch Prior-Varianz).","Warum macht L1 Koeffizienten exakt null?","Der Subgradient von $|\\theta|$ ist bei $0$ das Intervall $[-1,1]$; für $|z|\\le\\lambda$ liegt das Optimum bei $0$.","Warum ist $X\\T X+\\lambda I$ immer invertierbar?","Alle Eigenwerte sind $\\ge\\lambda>0$."])],
 "„Was wie ein Trick aussieht, ist meist ein Prior in Verkleidung.“")

# V10 Metriken
page("v10_metriken.tex",
 hdr("AUROC als Wahrscheinlichkeit, $F_1$ als harmonisches Mittel und $R^2$ als Varianzzerlegung.","Kennzahlen sind Rechnungen -- man kann sie herleiten","Klassifikations- \\& Regressionsmetriken:","Vertiefung","Herleitung und Code"),
 "section/evaluation\\_metrics\\_spring2020.pdf, hand-notes/ML Notes.pdf; Herleitungen ausformuliert; Code: code/sk\\_metrics.py (real ausgeführt)",
 [auc_panel, f1_panel, r2_panel, auc_ex, snip("sk_metrics","Klassifikations- und Regressionsmetriken in scikit-learn","hnpurple"), metr_lesart,
 quiz(["Was ist AUROC als Wahrscheinlichkeit?","$P(S^+>S^-)$ für zufällig gezogene positive/negative Beispiele (Mann--Whitney).","Warum kommt $TN$ in $F_1$ nicht vor?","$F_1=\\frac{2TP}{2TP+FP+FN}$ -- $TN$ kürzt sich.","Warum ist $R^2\\ge0$ bei OLS mit Achsenabschnitt?","$\\mathrm{SST}=\\mathrm{SSR}+\\mathrm{SSE}$ mit verschwindendem Kreuzterm."])],
 "„Wer eine Kennzahl herleiten kann, versteht, was sie nicht misst.“")

# V11 Workflow Code
page("v11_workflow_code.tex",
 hdr("Data Leakage in Zahlen und Nested CV mit Pipeline -- der Code hinter Feature Engineering, Tuning und Workflow.","Vorverarbeitung gehört \\emph{in} die Pipeline!","Feature Engineering, Tuning \\& Workflow:","Vertiefung","Code und Lesart"),
 "hand-notes/ML Notes.pdf (Feature Engineering, Cross-Validation, End-to-End-Workflow); Beispiele selbst erstellt; Code: code/sk\\_leakage.py, sk\\_pipeline.py (real ausgeführt)",
 [snip("sk_leakage","Data Leakage: Feature-Auswahl außerhalb vs.\\ innerhalb der Kreuzvalidierung","hnred"), leak_lesart, snip("sk_pipeline","ColumnTransformer + Pipeline + Nested Cross-Validation","hnteal"),
 quiz(["Warum sind $0.88$ im Leakage-Beispiel trügerisch?","Die Feature-Auswahl sah alle Labels; bei Rauschen ist ehrlich nur $\\approx0.5$ erreichbar.","Was tut die Pipeline?","Sie lernt Imputer, Skalierer und Selektor in jedem Fold nur auf dem Trainingsteil.","Wozu Nested CV?","Tuning und Bewertung trennen, damit der Score nicht optimistisch verzerrt ist."])],
 "„Was in der Pipeline steht, kann nicht ins Testset lecken.“")

# V12 Clustering
page("v12_clustering.tex",
 hdr("Warum K-Means konvergiert -- und Code für K-Means, hierarchisches Clustering und Gauß-Gemische.","Jeder Schritt senkt $J$: zuweisen minimiert, mitteln minimiert","K-Means \\& Clustering:","Vertiefung","Herleitung und Code"),
 "main\\_notes.pdf Kap.\\ 10 (K-Means), notes/cs229-notes7a.pdf; hand-notes/ML Notes.pdf (K-Means, Hierarchical Clustering); Herleitung ausformuliert; Code: code/sk\\_cluster.py (real ausgeführt)",
 [kmeans_panel, kmeans_ex, snip("sk_cluster","K-Means (Inertia, Silhouette), hierarchisches Clustering, Gauß-Gemisch","hnpurple"),
 quiz(["Warum sinkt $J$ im Zuweisungsschritt?","Jedes $x^{(i)}$ wählt den nächsten Zentroid, das minimiert seinen Summanden.","Warum ist der Zentroid der Mittelwert?","Ableiten und Null setzen: $\\sum(x-\\mu)=0$, Hesse $2n_kI\\succ0$.","Warum nur ein lokales Minimum?","$J$ ist nichtkonvex; das Ergebnis hängt von der Initialisierung ab."])],
 "„Zuweisen, mitteln, wiederholen -- und jedes Mal wird es besser.“")

# V13 EM
d04b = blocks_of(read(d04))
em = take(d04,"Jensen","Wann ist die Schranke","Warum EM nie","M-Schritt für Gauß","Beispiel: die Likelihood")
page("v13_em.tex",
 hdr("Jensen, ELBO, Monotonie und der M-Schritt des Gauß-Gemischs -- mit Code, der die Log-Likelihood steigen lässt.","Alle Schätzer sind Maximum Likelihood mit Nebenbedingungen","EM-Algorithmus:","Vertiefung","Herleitung und Code"),
 "main\\_notes.pdf Kap.\\ 11 (EM, Jensen, ELBO), notes/cs229-notes8.pdf; Schritte ergänzt; Code: code/sk\\_gmm.py (real ausgeführt)",
 em + [snip("sk_gmm","EM für ein Gauß-Gemisch: die Log-Likelihood steigt monoton","hnorange"), em_lesart,
 quiz(["Welche Ungleichung liefert die ELBO?","Jensen für den konkaven $\\log$.","Warum ist $Q=p(z|x)$ die beste Wahl im E-Schritt?","Dann verschwindet der KL-Term: ELBO $=\\log p(x;\\theta)$.","Wie erhält man $\\phi_k$ im M-Schritt?","Lagrange mit $\\sum\\phi_k=1$: $\\phi_k=\\frac1n\\sum_i\\gamma_{ik}$."])],
 "„Rate klug, schätze neu -- und die Likelihood kann nicht sinken.“")

# V14 PCA / ICA
page("v14_pca_ica.tex",
 hdr("PCA als Eigenwertproblem und ICA-Dichte per Variablenwechsel -- mit PCA-Code.","Lagrange mit $u\\T u=1$ führt auf $\\hat\\Sigma u=\\lambda u$","PCA \\& ICA:","Vertiefung","Herleitung und Code"),
 "main\\_notes.pdf Kap.\\ 12--13 (PCA, ICA), notes/cs229-notes10.pdf, cs229-notes11.pdf; hand-notes/ML Notes.pdf (PCA); Schritte ergänzt; Code: code/sk\\_pca.py (real ausgeführt)",
 take(d04,"PCA: Varianz") + take(d07,"ICA: Dichte") + [snip("sk_pca","PCA mit Standardisierung: erklärte Varianz","hnorange"),
 quiz(["Warum führt PCA auf ein Eigenwertproblem?","Lagrange mit $u\\T u=1$: $\\hat\\Sigma u=\\lambda u$; die Varianz ist $\\lambda$.","Woher kommt $|\\det W|$ in der ICA-Dichte?","Variablenwechsel $s=Wx$: die Jacobi-Determinante korrigiert das Volumen.","Warum vor PCA skalieren?","Sonst dominiert das Feature mit der größten Skala die Varianz."])],
 "„Behalte die Richtungen, in denen die Daten etwas zu erzählen haben.“")

# V15 VAE / GAN
page("v15_vae_gan.tex",
 hdr("KL-Term und Reparametrisierung hergeleitet -- plus VAE- und GAN-Training in PyTorch.","Zufall steckt nur in $\\xi$, nicht in $\\mu,\\sigma$","VAE \\& GAN:","Vertiefung","Herleitung und Code"),
 "main\\_notes.pdf Kap.\\ 11.5 (Variational Auto-Encoder); GAN als Web-Zusatz (markiert); Herleitungen ausformuliert; Code: code/torch\\_vae.py, torch\\_gan.py (real ausgeführt)",
 [kl_panel, reparam_panel, snip("torch_vae","Variational Autoencoder: Reparametrisierung, Rekonstruktion + KL","hnrose"), snip("torch_gan","GAN: Generator gegen Diskriminator auf 1D-Daten","hnnavy"), vae_lesart,
 quiz(["Wie lautet der KL-Term für zwei Gauß-Verteilungen?","$\\frac12(\\mu^2+\\sigma^2-1-\\log\\sigma^2)$ gegen $\\N(0,1)$.","Warum braucht man die Reparametrisierung?","Damit der Gradient durch das Sampling fließt: $z=\\mu+\\sigma\\xi$.","Was ist Mode Collapse?","Der GAN-Generator erzeugt nur wenige Varianten der Daten."])],
 "„Wer die Daten erzeugen kann, hat sie verstanden.“")

# V16 RL
d06b = blocks_of(read(d06))
rl = [b for b in d06b if "Selbsttest" not in title_of(b)]
qz6 = [b for b in d06b if "Selbsttest" in title_of(b)][0]
page("v16_rl.tex",
 hdr("Bellman-Gleichung, Konvergenz der Value Iteration, Policy-Gradient-Theorem und LQR-Riccati.","Der Log-Trick macht aus dem Gradienten einen Erwartungswert","Bestärkendes Lernen:","Vertiefung","MDPs, LQR und Policy Gradient"),
 "main\\_notes.pdf Kap.\\ 15--17 (MDPs, Value/Policy Iteration, LQR, Policy Gradient), notes/cs229-notes12.pdf, cs229-notes13.pdf; Schritte ergänzt",
 rl + [qz6],
 "„Rekursion für Werte, Log-Trick für Policies -- die zwei Werkzeuge des RL.“")
print("ok")
