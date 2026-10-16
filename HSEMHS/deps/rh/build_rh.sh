#!/bin/bash
# Build a private copy of RH (rhf1d) for the Gate 1/2 cross-checks. The user's RH checkout
# is only read. usage: deps/rh/build_rh.sh [RH_SRC] [DEST]
set -euo pipefail
SRC=${1:-/Users/harshm/Documents/WorkRepo/RH}
DEST=${2:-$(cd "$(dirname "$0")/../.." && pwd)/cache/rh}
mkdir -p "$DEST"
rsync -a --delete --exclude .git "$SRC"/ "$DEST"/
( cd "$SRC" && git rev-parse HEAD 2>/dev/null || echo unknown ) > "$DEST/SOURCE_COMMIT"
cd "$DEST"
make -j8 CC=gcc 2>&1 | tail -5
cd rhf1d
# backgrcontr needs backgrOpac (backgropac_xdr.c), which is not part of librh.a
sed -i '' 's/^BACKGROPAC = anglequad.o  backgrcontr.o  multiatmos.o  project.o$/BACKGROPAC = anglequad.o  backgrcontr.o  multiatmos.o  project.o  backgropac_xdr.o/' Makefile
printf '\nbackgropac_xdr.o:  ../backgropac_xdr.c\n\t$(CC) $(CFLAGS) -c ../backgropac_xdr.c -o backgropac_xdr.o\n' >> Makefile
make CC=gcc 2>&1 | tail -5
ls -la rhf1d solveray backgrcontr 2>/dev/null || true
