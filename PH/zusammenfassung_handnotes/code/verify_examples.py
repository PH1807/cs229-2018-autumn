"""Prueft die Zahlen aller Rechenbeispiele der Handnotes-Seiten (Kapitel in Klammern)."""
import numpy as np
from math import log, exp, sqrt, e, pi
def chk(name, got, exp_, tol=0.006):
    ok = np.allclose(got, exp_, atol=tol)
    print(("OK   " if ok else "FEHL ") + name, np.round(got, 4), "erwartet", exp_)
sig = lambda z: 1 / (1 + np.exp(-z))
# 1 Lineare Regression
X = np.array([[1, 1], [1, 2], [1, 3.]]); y = np.array([1, 2, 2.])
chk("LinReg theta", np.linalg.solve(X.T @ X, X.T @ y), [2/3, .5]); chk("LinReg h(4)", 2/3 + .5*4, 2.67)
# 2 LogReg
chk("LogReg h", sig(2), .88); chk("LogReg dtheta", .5*(1-sig(2))*np.array([1,2,3]), [.06,.12,.18], .01)
# 4 Naive Bayes
ps, ph = (3+1)/(4+2)*.4, (1+1)/(6+2)*.6; chk("NB P(Spam|x)", ps/(ps+ph), .64)
# 5 Kernel
x, z = np.array([1, 2.]), np.array([3, 1.]); phi = lambda v: np.array([v[0]**2, sqrt(2)*v[0]*v[1], v[1]**2])
chk("Kernel (x^T z)^2", (x@z)**2, 25); chk("Kernel phi.phi", phi(x)@phi(z), 25)
# 6 SVM 1D
from sklearn.svm import SVC
m = SVC(kernel="linear", C=1e6).fit([[-1],[0],[2],[3]], [-1,-1,1,1])
chk("SVM w,b", [m.coef_[0][0], m.intercept_[0]], [1, -1]); chk("SVM alpha", sorted(np.abs(m.dual_coef_[0])), [.5,.5]); chk("SVM f(1.4)", m.decision_function([[1.4]])[0], .4)
# 7 Perceptron
th = np.zeros(2)
for xx, yy in [((1,2),-1),((2,1),1),((1,2),-1),((2,1),1)]:
    xx = np.array(xx); yhat = 1 if th@xx >= 0 else -1
    if yhat != yy: th = th + yy*xx
chk("Perceptron theta", th, [1,-1]); chk("Perceptron Schranke", (sqrt(5)/(1/sqrt(2)))**2, 10)
# 8 Decision Tree
H = lambda p: -sum(q*log(q, 2) for q in (p, 1-p) if q > 0)
chk("DT Entropie Eltern", H(.8), .722); chk("DT Gain A", H(.8) - .5*H(.6), .237); chk("DT Gain B", H(.8) - .8*H(.75), .073)
# 9 AdaBoost
err = .2; a = log((1-err)/err); w = np.array([.8, .2, .2, .2, .2]); chk("AdaBoost alpha", a, 1.39); chk("AdaBoost w_norm", (w/w.sum())[:2], [.5, .125])
# 10 Verluste
z = np.array([3, .5, -1.]); chk("Verluste logistisch", np.log(1+np.exp(-z)), [.05,.47,1.31]); chk("Verluste exp", np.exp(-z), [.05,.61,2.72])
# 11 NN
W1 = np.array([[1,-1],[.5,1]]); a1 = np.maximum(W1@np.array([1,2.]), 0); yh = np.array([1,2.])@a1 - 1; chk("NN y_hat", yh, 4); chk("NN Params", 100*50+50+50*10+10, 5560)
# 12 Backprop
chk("Backprop nach SGD", .5*(.3*max(0,.9*2))**2, .146, .001)
# 14 Lerntheorie
chk("Hoeffding n", np.log(2e6/.05)/(2*.05**2), 3500, 5)
# 15 Ridge
chk("Ridge theta", [11/14, 11/16, 11/28], [.786,.688,.393])
# 16 GP
K = 1.1; ks = exp(-.5); chk("GP mu*", ks/K, .55); chk("GP Sigma*", 1-ks**2/K, .67)
# 18 Metriken
tp, fp, fn, tn = 9, 2, 1, 8; P, R = tp/(tp+fp), tp/(tp+fn); chk("Metriken F1@0.5", 2*P*R/(P+R), .857); tp, fp, fn = 7, 2, 3; P, R = tp/(tp+fp), tp/(tp+fn); chk("Metriken F1@0.6", 2*P*R/(P+R), .737)
# 19 K-Means
def J(cl): return sum(((np.array(c) - np.mean(c))**2).sum() for c in cl)
chk("K-Means J lokal", J([[1,2],[9,10,20]]), 74.5); chk("K-Means J besser", J([[1,2,9,10],[20]]), 65)
# 20 EM
xs = np.array([0,1,4.]); n = lambda x, m: exp(-(x-m)**2/2)/sqrt(2*pi); g1 = np.array([n(v,0)/(n(v,0)+n(v,4)) for v in xs])
chk("EM gamma1", g1, [.9997,.982,.0003], .001); chk("EM mu1", (g1*xs).sum()/g1.sum(), .50); chk("EM mu2", ((1-g1)*xs).sum()/(1-g1).sum(), 3.95)
# 21 FA
L = np.array([[1],[.5]]); S = L@L.T + np.diag([.1,.2]); chk("FA rho", S[0,1]/sqrt(S[0,0]*S[1,1]), .71)
# 22 HMM
A = np.array([[.7,.3],[.4,.6]]); B = np.array([[.2,.4,.4],[.5,.4,.1]]); pi_ = np.array([.5,.5]); obs = [2,0]
al = pi_*B[:,obs[0]]; al2 = (al@A)*B[:,obs[1]]; chk("HMM P(x)", al2.sum(), .077); d2 = np.max(al[:,None]*A, axis=0)*B[:,obs[1]]; chk("HMM Viterbi delta", d2, [.028,.03])
# 23 PCA
P4 = np.array([[2,0],[-2,0],[0,1],[0,-1.]]); ev = np.linalg.eigvalsh(P4.T@P4/4); chk("PCA EV", sorted(ev), [.5, 2]); chk("PCA erklaert", 2/2.5, .8)
# 24 ICA
A2 = np.array([[1,.5],[.5,1]]); chk("ICA Wx", np.linalg.inv(A2)@(A2@np.array([1,-1.])), [1,-1])
# 25 Attention
S = np.eye(2)/sqrt(2); Pm = np.exp(S)/np.exp(S).sum(1, keepdims=True); chk("Attention Zeile1", (Pm@np.array([[1,2],[3,4.]]))[0], [1.66, 2.66])
# 26 Adversarial
chk("Adversarial w^T eta", .1*1000*.01, 1.0)
# 27 MDP
V = np.zeros(2)
for _ in range(400): V = np.array([0 + .9*max(V[0], V[1]), 1 + .9*V[1]])
chk("MDP V*", V, [9, 10], .01)
# 28 LQR
P_ = 1.
for _ in range(60): P_ = 1 + P_ - P_**2/(1+P_)
chk("LQR P*", P_, 1.618); chk("LQR K*", P_/(1+P_), .618)
# 29 Policy Gradient
th = 0.; th += .5*(1-sig(th))*1; chk("PG theta", th, .25); chk("PG pi", sig(.25), .56)
# ML Notes: Regressionsmetriken
yt = np.array([5, 6, 7.5, 8, 9.5]); yp = np.array([4.6, 5.5, 8, 7.2, 10]); sst = ((yt-yt.mean())**2).sum(); sse = ((yt-yp)**2).sum()
print("ML-Notes SST:", round(sst, 2), "(Notes: 13.30) SSE:", round(sse, 2), "R2:", round(1-sse/sst, 3), "(Notes: 0.884)")

print("---- neue Seiten ----")
# Grundlagen
chk("Bayes Medizintest", .95*.01/(.95*.01+.10*.99), .088, .001)
chk("GD alpha=.1", 4*.8**np.arange(1,4), [3.2,2.56,2.048], .01); chk("GD alpha=1.1", [(1-2.2)*4, (1-2.2)**2*4], [-4.8, 5.76])
# kNN
d = np.array([1,2,np.sqrt(5),3,4]); lab = np.array(["A","B","B","A","A"])
for k,expct in [(1,"A"),(3,"B"),(5,"A")]:
    v,c = np.unique(lab[np.argsort(d)[:k]], return_counts=True); print(("OK   " if v[c.argmax()]==expct else "FEHL ")+f"kNN k={k}", v[c.argmax()])
# Hierarchisch (single/complete)
from scipy.cluster.hierarchy import linkage
pts = np.array([[1],[2],[6],[8],[15.]])
chk("Hier single", linkage(pts,"single")[:,2], [1,2,4,7]); chk("Hier complete", linkage(pts,"complete")[:,2], [1,2,7,14])
# Regressionsmetriken
chk("RegMetr", [np.abs(yt-yp).mean(), ((yt-yp)**2).mean(), np.sqrt(((yt-yp)**2).mean()), 1-sse/sst], [.54,.31,.557,.874])
# Feature Scaling
xs = np.array([10,20,30,100.]); chk("Skalierung z", (xs-xs.mean())/xs.std(), [-.85,-.57,-.28,1.70]); chk("Skalierung minmax", (xs-10)/90, [0,.11,.22,1])
# Tuning
sc = np.array([.82,.85,.80,.84,.83]); chk("CV mean/std", [sc.mean(), sc.std(ddof=1)], [.828,.019]); chk("Nested Fits", 5*(3*6+1), 95)
# Adam
g=2.; m=.1*g; v=.001*g*g; mh=m/(1-.9); vh=v/(1-.999); chk("Adam Schritt", .001*mh/(np.sqrt(vh)+1e-8), .001, 1e-5)
# Conv
chk("Conv1D", [np.dot([1,2,3],[1,0,-1]), np.dot([2,3,4],[1,0,-1]), np.dot([3,4,5],[1,0,-1])], [-2,-2,-2]); chk("CNN Params", [3*3*1*8+8, 3*3*8*16+16, 16*7*7*10+10], [80,1168,7850])
# VAE KL
mu, s = .5, .8; chk("VAE KL", .5*(mu**2+s**2-1-np.log(s**2)), .168)
# EM Log-Likelihood vor/nach
def ll(mu, phi, xs=np.array([0,1,4.])):
    return sum(np.log(sum(p*np.exp(-(x-m)**2/2)/np.sqrt(2*np.pi) for m,p in zip(mu,phi))) for x in xs)
chk("EM LL vor", ll([0,4],[.5,.5]), -5.32, .01); chk("EM LL nach", ll([.4962,3.946],[.661,.339]), -4.91, .01)
# SVM Dual 1D
al = np.linspace(0,1,10001); W = 2*al-2*al**2; chk("SVM Dual alpha", al[W.argmax()], .5, .001); chk("SVM Dual W", W.max(), .5)
# Backprop Beispiel 2-2-1 (d05) mit torch-freier Rechnung
W1 = np.array([[1,-1],[.5,1]]); xx = np.array([1,2.]); z1 = W1@xx; a1 = np.maximum(z1,0); yhat = np.array([1,2.])@a1-1; d2 = yhat-3
dW2 = d2*a1; d1 = (np.array([1,2.])*d2)*(z1>0); dW1 = np.outer(d1,xx)
chk("Backprop dW2", dW2, [0,2.5]); chk("Backprop delta1", d1, [0,2]); chk("Backprop dW1", dW1.ravel(), [0,0,2,4])
# Logistic Gradient-Check
def l(th): return np.log(sig(th@np.array([1,2.])))
e=1e-5; chk("LogReg Gradient-Check", (l(np.array([0,e]))-l(np.array([0,-e])))/(2*e), 1.0, 1e-3)
# Hoeffding
chk("Hoeffding 2e^-5", 2*np.exp(-5), .0135, .001)
# Bandit Softmax-Gradient (d06)
pi_ = np.exp([0,0])/2; chk("PG grad log pi", np.array([1,0])-pi_, [.5,-.5])

print("---- SHAP / Fairness ----")
from itertools import permutations
from math import factorial, comb
f = lambda x: 3*x[0] + 2*x[1] + 4*x[0]*x[1]
def val(S, x=np.array([1.,1.]), base=np.array([0.,0.])):
    z = base.copy()
    for j in S: z[j] = x[j]
    return f(z)
phi = np.zeros(2)
for j in range(2):
    for S in ([], [1-j]):
        w = factorial(len(S))*factorial(1-len(S))/factorial(2)
        phi[j] += w*(val(S+[j]) - val(S))
chk("Shapley Formel (Gewichte)", phi, [5, 4]); chk("Shapley Effizienz", phi.sum(), f([1,1]) - f([0,0]))
# lineares Modell: phi_j = w_j (x_j - E x_j)
w = np.array([2,-1.]); Ex = np.array([1,2.]); x = np.array([3,1.]); ph = w*(x-Ex); b = w@Ex + 3
chk("Linear-SHAP phi", ph, [4, 1]); chk("Linear-SHAP Summe", b + ph.sum(), w@x + 3)
# Fairness
tA = dict(tp=6, fn=2, fp=2, tn=10); tB = dict(tp=1, fn=3, fp=1, tn=15)
def m(t):
    n = sum(t.values()); p = (t["tp"]+t["fn"])/n
    return dict(sel=(t["tp"]+t["fp"])/n, tpr=t["tp"]/(t["tp"]+t["fn"]), fpr=t["fp"]/(t["fp"]+t["tn"]), ppv=t["tp"]/(t["tp"]+t["fp"]), acc=(t["tp"]+t["tn"])/n, p=p)
a, bb = m(tA), m(tB)
chk("Fair A sel,TPR,FPR,PPV", [a["sel"],a["tpr"],a["fpr"],a["ppv"]], [.4,.75,.1667,.75])
chk("Fair B sel,TPR,FPR,PPV", [bb["sel"],bb["tpr"],bb["fpr"],bb["ppv"]], [.1,.25,.0625,.5])
chk("Fair Accuracy gleich", [a["acc"], bb["acc"]], [.8,.8]); chk("Fair Disparate Impact", bb["sel"]/a["sel"], .25)
for nm, d in (("A", a), ("B", bb)):
    chk(f"Unmoeglichkeit FPR-Formel {nm}", d["p"]/(1-d["p"])*(1-d["ppv"])/d["ppv"]*d["tpr"], d["fpr"])
    chk(f"Selektionsrate = p TPR+(1-p) FPR {nm}", d["p"]*d["tpr"]+(1-d["p"])*d["fpr"], d["sel"])
# Reweighing
cnt = {("A",1):8, ("A",0):12, ("B",1):4, ("B",0):16}; N = 40
wgt = {k: (sum(v for (g,_),v in cnt.items() if g==k[0])/N)*(sum(v for (_,y),v in cnt.items() if y==k[1])/N)/(c/N) for k,c in cnt.items()}
chk("Reweighing Gewichte", [wgt[("A",1)], wgt[("A",0)], wgt[("B",1)], wgt[("B",0)]], [.75, 1.1667, 1.5, .875])
chk("Reweighing pos. Rate A,B", [wgt[("A",1)]*8/(wgt[("A",1)]*8+wgt[("A",0)]*12), wgt[("B",1)]*4/(wgt[("B",1)]*4+wgt[("B",0)]*16)], [.3,.3])

print("---- Vertiefungsseiten ----")
# GDA -> logistische Form: theta = Sigma^-1 (mu1-mu0), theta0 = -1/2(mu1'S^-1 mu1 - mu0'S^-1 mu0) + log(phi/(1-phi))
mu0, mu1 = np.array([0,0.]), np.array([2,1.]); Sg = np.array([[1,.3],[.3,1.]]); ph = .4; xq = np.array([1.5,1.])
Si = np.linalg.inv(Sg)
lp = lambda x, m: -.5*(x-m)@Si@(x-m)
post = np.exp(lp(xq,mu1))*ph/(np.exp(lp(xq,mu1))*ph + np.exp(lp(xq,mu0))*(1-ph))
th = Si@(mu1-mu0); th0 = -.5*(mu1@Si@mu1 - mu0@Si@mu0) + np.log(ph/(1-ph))
chk("GDA Posterior = Sigmoid(theta'x+theta0)", sig(th@xq+th0), post, 1e-9)
# K-Means: Mittelwert minimiert Summe quadr. Abstaende
pts = np.array([1,2,9.]); mm = np.linspace(0,10,100001)
chk("K-Means Mittel minimiert", mm[np.argmin(((pts[:,None]-mm)**2).sum(0))], pts.mean(), 1e-3)
# AUROC = Mann-Whitney, F1 = 2TP/(2TP+FP+FN)
from sklearn.metrics import roc_auc_score, f1_score
yy = np.array([1]*10+[0]*10); ss = np.array([.9,.8,.85,.7,.95,.6,.75,.9,.4,.55,.3,.2,.6,.1,.45,.35,.65,.25,.15,.05])
pos, neg = ss[yy==1], ss[yy==0]; U = np.mean([(p>n)+.5*(p==n) for p in pos for n in neg])
chk("AUROC = P(s+>s-)", U, roc_auc_score(yy, ss), 1e-9)
pr = (ss>=.5).astype(int); tp=((pr==1)&(yy==1)).sum(); fp=((pr==1)&(yy==0)).sum(); fn=((pr==0)&(yy==1)).sum()
chk("F1 = 2TP/(2TP+FP+FN)", 2*tp/(2*tp+fp+fn), f1_score(yy, pr), 1e-9)
# Adam Bias-Korrektur: mit m_0=0 und konstantem g ist E[m_t]=(1-b1^t) g
b1=.9; m_=0.; gg=2.
for t in range(1,6): m_=b1*m_+(1-b1)*gg
chk("Adam m_5 = (1-b1^5) g", m_, (1-b1**5)*gg, 1e-12); chk("Adam m_hat = g", m_/(1-b1**5), gg, 1e-12)
# KL(N(mu,s^2)||N(0,1)) numerisch
from scipy.integrate import quad
mu_, s_ = .5, .8
pdf = lambda x, m, s: np.exp(-(x-m)**2/(2*s*s))/np.sqrt(2*np.pi*s*s)
kl_num = quad(lambda x: pdf(x,mu_,s_)*np.log(pdf(x,mu_,s_)/pdf(x,0,1)), -10, 10)[0]
chk("KL numerisch = Formel", kl_num, .5*(mu_**2+s_**2-1-np.log(s_**2)), 1e-6)
# Newton aus Taylor (1D) und Ridge/Lasso 1D
chk("Lasso Soft-Threshold z=.786, lam=.3", np.sign(.786)*max(abs(.786)-.3,0), .486); chk("Lasso lam=1", np.sign(.786)*max(abs(.786)-1,0), 0)
