#!/bin/zsh
# Baut Langfassung (main.pdf) und Kompaktfassung (main_kompakt.pdf); je 3 Läufe für stabile Verweise.
cd "$(dirname "$0")"
python3 gen_main.py > /dev/null
# Druckvariante (weiß, Graustufen) aus derselben Quelle
sed '1s/^/\\def\\PRINTMODE{1}\
/' main.tex > main_druck.tex
for f in main main_kompakt main_druck; do
  for i in 1 2 3; do xelatex -interaction=nonstopmode $f.tex > $f.log 2>&1; done
  echo "$f: $(pdfinfo $f.pdf | grep Pages) Fehler: $(grep -c '^!' $f.log)"
done
