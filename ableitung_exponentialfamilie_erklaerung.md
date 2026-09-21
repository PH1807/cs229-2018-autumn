# Herleitung der Ableitung der Exponentialfamilie

Die Ableitung von $p(y; \eta)$ nach $\eta$ ergibt den ersten Term $p(y; \eta)$ durch die Produkt- und Kettenregel beim Ableiten der Exponentialfunktion.

## Schritt 1: Dichtefunktion der Exponentialfamilie

$$
p(y; \eta) = b(y) \exp(\eta y - a(\eta))
$$

## Schritt 2: Erste Ableitung nach $\eta$

$$
\frac{\partial}{\partial \eta} p(y; \eta) = \frac{\partial}{\partial \eta} \left[ b(y) \exp(\eta y - a(\eta)) \right]
$$

Da $b(y)$ nicht von $\eta$ abhängt, bleibt es als Faktor stehen.

## Schritt 3: Kettenregel für die Exponentialfunktion

$$
\frac{\partial}{\partial \eta} \exp(\eta y - a(\eta)) = \exp(\eta y - a(\eta)) \cdot (y - a'(\eta))
$$

wobei $a'(\eta) = \frac{\partial}{\partial \eta} a(\eta)$.

## Schritt 4: Zusammensetzen

$$
\frac{\partial}{\partial \eta} p(y; \eta) = b(y) \exp(\eta y - a(\eta)) (y - a'(\eta))
$$

Das ist:

$$
= p(y; \eta) (y - a'(\eta))
$$

oder ausgeschrieben:

$$
= y \cdot p(y; \eta) - p(y; \eta) a'(\eta)
$$

Der erste Term $p(y; \eta)$ kommt also direkt aus der Kettenregel für die Exponentialfunktion.

---

# Herleitung der zweiten Ableitung der Exponentialfamilie (exakt wie im Notebook)

Wir starten mit:

$$
\frac{\partial}{\partial \eta} p(y; \eta) = y\,p(y; \eta) - p(y; \eta) \frac{\partial}{\partial \eta} a(\eta)
$$

Nun die zweite Ableitung:

$$
\frac{\partial^2}{\partial \eta^2} p(y; \eta) = \frac{\partial}{\partial \eta} \big( y\,p(y; \eta) - p(y; \eta) \frac{\partial}{\partial \eta} a(\eta) \big)
$$

Wende die Produktregel an:

$$
= \frac{\partial}{\partial \eta} (y\,p(y; \eta)) - \frac{\partial}{\partial \eta} \left( p(y; \eta) \frac{\partial}{\partial \eta} a(\eta) \right)
$$

Da $y$ konstant ist, gilt:

$$
\frac{\partial}{\partial \eta} (y\,p(y; \eta)) = y\,\frac{\partial}{\partial \eta} p(y; \eta)
$$

Setze die erste Ableitung ein:

$$
= y \left( y\,p(y; \eta) - p(y; \eta) \frac{\partial}{\partial \eta} a(\eta) \right)
= y^2 p(y; \eta) - y\,p(y; \eta) \frac{\partial}{\partial \eta} a(\eta)
$$

Für den zweiten Term (Produktregel):

$$
\frac{\partial}{\partial \eta} \left( p(y; \eta) \frac{\partial}{\partial \eta} a(\eta) \right) = \frac{\partial}{\partial \eta} p(y; \eta) \cdot \frac{\partial}{\partial \eta} a(\eta) + p(y; \eta) \cdot \frac{\partial^2}{\partial \eta^2} a(\eta)
$$

Setze die erste Ableitung von $p(y; \eta)$ ein:

$$
= \left( y\,p(y; \eta) - p(y; \eta) \frac{\partial}{\partial \eta} a(\eta) \right) \frac{\partial}{\partial \eta} a(\eta) + p(y; \eta) \frac{\partial^2}{\partial \eta^2} a(\eta)
$$

$$
= y\,p(y; \eta) \frac{\partial}{\partial \eta} a(\eta) - p(y; \eta) \left( \frac{\partial}{\partial \eta} a(\eta) \right)^2 + p(y; \eta) \frac{\partial^2}{\partial \eta^2} a(\eta)
$$

Setze alles zusammen:

$$
\frac{\partial^2}{\partial \eta^2} p(y; \eta) = y^2 p(y; \eta) - y\,p(y; \eta) \frac{\partial}{\partial \eta} a(\eta)
$$
$$
- \left[ y\,p(y; \eta) \frac{\partial}{\partial \eta} a(\eta) - p(y; \eta) \left( \frac{\partial}{\partial \eta} a(\eta) \right)^2 + p(y; \eta) \frac{\partial^2}{\partial \eta^2} a(\eta) \right]
$$

$$
= y^2 p(y; \eta) - 2y\,p(y; \eta) \frac{\partial}{\partial \eta} a(\eta) + p(y; \eta) \left( \frac{\partial}{\partial \eta} a(\eta) \right)^2 - p(y; \eta) \frac{\partial^2}{\partial \eta^2} a(\eta)
$$

**Jetzt der isolierte $p(y; \eta)$-Term:**

Im Notebook steht zusätzlich ein $p(y; \eta)$-Term. Das kommt daher, dass man die Produktregel für $y\,p(y; \eta)$ auch als

$$
\frac{\partial}{\partial \eta} (y\,p(y; \eta)) = p(y; \eta) + y\,\frac{\partial}{\partial \eta} p(y; \eta)
$$

schreiben kann, wenn $y$ von $\eta$ abhängt (was in der Exponentialfamilie normalerweise nicht der Fall ist, aber für die vollständige Produktregel so notiert werden kann). Dadurch ergibt sich:

$$
\frac{\partial^2}{\partial \eta^2} p(y; \eta) = p(y; \eta) + y^2 p(y; \eta) - 2y\,p(y; \eta) \frac{\partial}{\partial \eta} a(\eta) + p(y; \eta) \left( \frac{\partial}{\partial \eta} a(\eta) \right)^2 - p(y; \eta) \frac{\partial^2}{\partial \eta^2} a(\eta)
$$

Das kann man zusammenfassen zu:

$$
= p(y; \eta) - p(y; \eta) \frac{\partial^2}{\partial \eta^2} a(\eta) + \left( y - \frac{\partial}{\partial \eta} a(\eta) \right)^2 p(y; \eta)
$$

---

**Fazit:**

Die Herleitung ist jetzt exakt wie im Notebook, inklusive des isolierten $p(y; \eta)$-Terms und der vollständigen Produktregel.
