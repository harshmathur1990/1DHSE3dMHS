#!/bin/bash
# Build every mode document: docs/{physics,technical}/<mode>.pdf (pdflatex, twice for references).
set -e
cd "$(dirname "$0")"
for d in physics technical; do
    for t in $d/*.tex; do
        (cd $d && pdflatex -interaction=nonstopmode -halt-on-error "$(basename $t)" > /dev/null && \
                  pdflatex -interaction=nonstopmode -halt-on-error "$(basename $t)" > /dev/null) || { echo "FAILED: $t"; tail -30 "${t%.tex}.log"; exit 1; }
        echo "built ${t%.tex}.pdf"
    done
done
rm -f */*.aux */*.log */*.out
