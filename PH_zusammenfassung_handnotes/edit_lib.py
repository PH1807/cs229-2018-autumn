import sys
def rep(f, old, new, count=1):
    p = "pages/%s.tex" % f
    t = open(p).read()
    assert old in t, (f, old[:60])
    t = t.replace(old, new, count)
    open(p, "w").write(t)
