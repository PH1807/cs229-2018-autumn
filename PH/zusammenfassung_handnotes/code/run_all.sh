#!/bin/zsh
# fuehrt alle Snippets aus und schreibt die (gekuerzten) Ausgaben nach out/
cd "$(dirname "$0")"
PY=${PY:-/private/tmp/claude-501/venv/bin/python}
for f in *.py; do
  n=${f%.py}
  $PY $f > out/$n.txt 2> out/$n.err || { echo "FEHLER in $f"; cat out/$n.err; }
  [ -s out/$n.err ] || rm -f out/$n.err
done
echo "fertig"
