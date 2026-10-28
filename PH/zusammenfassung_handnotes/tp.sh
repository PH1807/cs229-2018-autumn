#!/bin/zsh
# Testet eine Seite: ./tp.sh <datei-ohne-pages/> <scale>  -> Seitenzahl (1 = passt)
cd "$(dirname "$0")"
n=tp_$1
printf '\\input{preamble}\n\\begin{document}\n\\shorthandoff{"}\n\\hnscale{%s}\\input{pages/%s}\\clearpage\n\\end{document}\n' $2 $1 > $n.tex
xelatex -interaction=nonstopmode $n.tex > $n.log 2>&1
echo "$1 scale=$2 pages=$(pdfinfo $n.pdf 2>/dev/null | awk '/Pages/{print $2}') err=$(grep -c '^!' $n.log)"
rm -f $n.aux $n.out $n.toc
