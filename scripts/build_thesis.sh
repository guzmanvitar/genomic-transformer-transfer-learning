#!/usr/bin/env bash
#
# build_thesis.sh — reproducible LOCAL build of the PUCRS PPGCC thesis in thesis/.
#
# Why not uv?  LaTeX / TeX Live is a C/Perl toolchain, not a Python package, so it
# cannot live inside a uv environment. The closest reproducible, no-sudo option is
# TinyTeX (a self-contained TeX Live install under ~/Library/TinyTeX). This script
# provisions it on demand and compiles the thesis EXACTLY as Overleaf does, without
# modifying any source file in thesis/ (so it still recompiles unchanged on Overleaf).
#
# Usage:
#   scripts/build_thesis.sh          # provision (if needed) + build thesis/main.pdf
#   scripts/build_thesis.sh clean    # remove LaTeX build artifacts (keeps main.pdf)
#
# Notes on the two/three pre-existing DATA quirks in thesis/bibliography.bib (these
# are NOT environment problems — they behave identically on Overleaf, which is why
# the passes below tolerate non-zero exit codes and run in nonstopmode):
#   1. `jedrzejewski2018estimating` and `dallatorre2025nucleotide` use comma-separated
#      author lists -> BibTeX prints "Too many commas in name" and exits 2, but still
#      writes a valid .bbl.
#   2. one journal field contains a raw "&" ("Trends in Ecology & Evolution") -> a
#      recoverable "Misplaced alignment tab" LaTeX error; the "&" is dropped in that
#      one reference. nonstopmode recovers and the full PDF is still produced.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THESIS_DIR="$REPO_ROOT/thesis"
TINYTEX_ROOT="${TINYTEX_ROOT:-$HOME/Library/TinyTeX}"

# TeX packages required by main.tex + pucrs-ppgcc.cls (see grep of \usepackage/
# \RequirePackage). Names are TeX Live container names; tlmgr pulls dependencies.
REQUIRED_PKGS="latexmk \
  bera enumitem units multirow float algorithms algorithm2e sfmath psnfss \
  setspace textcase indentfirst babel babel-english babel-portuges geometry \
  xcolor colortbl subfigure hyperref lmodern amsfonts amsmath tools graphics \
  collection-fontsrecommended hyphen-portuguese"

find_texbin() { ls -d "$TINYTEX_ROOT"/bin/*/ 2>/dev/null | head -1; }

# --- clean mode ---------------------------------------------------------------
if [ "${1:-}" = "clean" ]; then
  TEXBIN="$(find_texbin)"; [ -n "$TEXBIN" ] && export PATH="$TEXBIN:$PATH"
  cd "$THESIS_DIR"
  if command -v latexmk >/dev/null 2>&1; then latexmk -C >/dev/null 2>&1; fi
  rm -f main.aux main.log main.bbl main.blg main.out main.toc main.lof main.lot \
        main.fls main.fdb_latexmk main.loa main.lob main.los main.lov chapter-*.aux
  echo "Cleaned LaTeX build artifacts in thesis/ (main.pdf kept)."
  exit 0
fi

# --- 1. provision TinyTeX (idempotent) ---------------------------------------
if [ -z "$(find_texbin)" ]; then
  echo ">> TinyTeX not found. Installing to $TINYTEX_ROOT (no sudo required)..."
  curl -fsSL "https://raw.githubusercontent.com/rstudio/tinytex/master/tools/install-bin-unix.sh" | sh
fi
TEXBIN="$(find_texbin)"
if [ -z "$TEXBIN" ]; then echo "ERROR: TinyTeX install failed." >&2; exit 1; fi
export PATH="$TEXBIN:$PATH"

# --- 2. ensure TeX packages (idempotent; no-ops if already installed) ---------
echo ">> Ensuring required TeX packages via tlmgr..."
# shellcheck disable=SC2086
tlmgr install $REQUIRED_PKGS >/dev/null 2>&1 || true

# --- 3. build: pdflatex -> bibtex -> pdflatex -> pdflatex ---------------------
# A fixed pass sequence is used instead of plain `latexmk` because BibTeX's exit
# code 2 (data quirk #1) makes vanilla latexmk stop before resolving citations.
cd "$THESIS_DIR"
echo ">> pdflatex (1/3)..."; pdflatex -shell-escape -interaction=nonstopmode main >/dev/null 2>&1 || true
echo ">> bibtex...";         bibtex main >/dev/null 2>&1 || true
echo ">> pdflatex (2/3)..."; pdflatex -shell-escape -interaction=nonstopmode main >/dev/null 2>&1 || true
echo ">> pdflatex (3/3)..."; pdflatex -shell-escape -interaction=nonstopmode main >/dev/null 2>&1 || true

# --- 4. report ----------------------------------------------------------------
if [ -f main.pdf ] && grep -aq "Output written on main.pdf" main.log; then
  echo ">> BUILD OK: $(grep -a 'Output written' main.log | tail -1 | sed 's/^ *//')"
  undef=$(grep -ac 'undefined' main.log 2>/dev/null); undef=${undef:-0}
  echo ">> undefined references/citations: $undef (expected 0)"
  echo ">> Output: $THESIS_DIR/main.pdf"
else
  echo ">> BUILD FAILED — inspect $THESIS_DIR/main.log" >&2
  exit 1
fi
