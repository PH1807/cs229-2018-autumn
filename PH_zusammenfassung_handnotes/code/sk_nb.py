from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
docs = ["win cash prize", "win prize now", "cash offer win",
        "meeting tomorrow", "project meeting notes", "lunch tomorrow"]
y = ["spam", "spam", "spam", "ham", "ham", "ham"]
vec = CountVectorizer()
X = vec.fit_transform(docs)
nb = MultinomialNB(alpha=1.0).fit(X, y)          # alpha = Laplace-Glaettung
print(nb.predict(vec.transform(["win cash now", "meeting notes"])))
print(nb.predict_proba(vec.transform(["win cash now"])).round(3))
