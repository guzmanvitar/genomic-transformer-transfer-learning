#!/usr/bin/env bash
#
# build_thesis.sh — reproducible LOCAL build of the PUCRS PPGCC thesis in thesis/.
#
# Why not uv?  LaTeX / TeX Live is a C/Perl toolchain, not a Python package, so it
# cannot live inside a uv environment. The closest reproducible, no-sudo option is
# TinyTeX (a self-contained TeX Live install). This script provisions it on demand
# and compiles the thesis.
#
# Usage:
#   scripts/build_thesis.sh          # provision (if needed) + build thesis/main.pdf
#   scripts/build_thesis.sh clean    # remove LaTeX build artifacts (keeps main.pdf)
#
# Notes:
#   - Success is judged from the LaTeX log, not from pdflatex's exit code: a benign
#     pdfTeX "duplicate destination" warning at \contracapa makes pdflatex return
#     non-zero even on an otherwise-clean build. The script fails on real LaTeX
#     errors (log lines starting with "!") and on any undefined reference/citation.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THESIS_DIR="$REPO_ROOT/thesis"

# TinyTeX lives under ~/Library/TinyTeX (macOS) or ~/.TinyTeX (Linux). Honor an
# explicit $TINYTEX_ROOT; otherwise prefer an existing install, else the OS default.
if [ -z "${TINYTEX_ROOT:-}" ]; then
  if   [ -d "$HOME/Library/TinyTeX" ]; then TINYTEX_ROOT="$HOME/Library/TinyTeX"
  elif [ -d "$HOME/.TinyTeX" ];        then TINYTEX_ROOT="$HOME/.TinyTeX"
  elif [ "$(uname)" = "Darwin" ];      then TINYTEX_ROOT="$HOME/Library/TinyTeX"
  else                                      TINYTEX_ROOT="$HOME/.TinyTeX"
  fi
fi

# TeX packages required by main.tex + pucrs-ppgcc.cls (see grep of \usepackage/
# \RequirePackage). Names are TeX Live container names; tlmgr pulls dependencies.
REQUIRED_PKGS="latexmk \
  bera enumitem units multirow float algorithms algorithm2e sfmath psnfss \
  setspace textcase indentfirst babel babel-english babel-portuges geometry \
  xcolor colortbl subfigure hyperref lmodern amsfonts amsmath tools graphics \
  collection-fontsrecommended hyphen-portuguese"

# A few style files whose presence confirms provisioning actually succeeded.
VERIFY_STY="bera.sty hyperref.sty colortbl.sty algorithm2e.sty sfmath.sty babel.sty subfigure.sty"

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
if [ -z "$TEXBIN" ]; then echo "ERROR: TinyTeX install failed (no bin dir under $TINYTEX_ROOT)." >&2; exit 1; fi
export PATH="$TEXBIN:$PATH"

# --- 2. ensure TeX packages, then verify provisioning succeeded ---------------
echo ">> Ensuring required TeX packages via tlmgr..."
# tlmgr can exit non-zero even on success (e.g. hyphenation format regeneration),
# so its exit code is not trustworthy here; the kpsewhich check below is the gate.
# shellcheck disable=SC2086
tlmgr install $REQUIRED_PKGS >/dev/null 2>&1 || true
missing=""
for sty in $VERIFY_STY; do
  kpsewhich "$sty" >/dev/null 2>&1 || missing="$missing ${sty%.sty}"
done
if [ -n "$missing" ]; then
  echo "ERROR: required LaTeX package(s) missing after tlmgr:$missing" >&2
  echo "       Check network / CTAN mirror access and re-run." >&2
  exit 1
fi

# --- 3. build: pdflatex -> bibtex -> pdflatex -> pdflatex ---------------------
cd "$THESIS_DIR"
echo ">> pdflatex (1/3)..."; pdflatex -shell-escape -interaction=nonstopmode main >/dev/null 2>&1 || true
echo ">> bibtex...";         bibtex main >/dev/null 2>&1 || true
echo ">> pdflatex (2/3)..."; pdflatex -shell-escape -interaction=nonstopmode main >/dev/null 2>&1 || true
echo ">> pdflatex (3/3)..."; pdflatex -shell-escape -interaction=nonstopmode main >/dev/null 2>&1 || true

# --- 4. verify: judge success from the log, not from pdflatex's exit code -----
if [ ! -f main.pdf ] || ! grep -aq "Output written on main.pdf" main.log 2>/dev/null; then
  echo ">> BUILD FAILED — no PDF produced; inspect $THESIS_DIR/main.log" >&2
  exit 1
fi
real_errors=$(grep -a '^!' main.log 2>/dev/null || true)
if [ -n "$real_errors" ]; then
  echo ">> BUILD FAILED — LaTeX reported errors:" >&2
  echo "$real_errors" >&2
  echo ">> inspect $THESIS_DIR/main.log" >&2
  exit 1
fi
undef=$(grep -acE 'Citation .*undefined|Reference .*undefined|undefined (references|citations)' main.log 2>/dev/null)
undef=${undef:-0}
if [ "$undef" -ne 0 ]; then
  echo ">> BUILD FAILED — $undef undefined reference(s)/citation(s); inspect $THESIS_DIR/main.log" >&2
  exit 1
fi

echo ">> BUILD OK: $(grep -a 'Output written' main.log | tail -1 | sed 's/^ *//')"
echo ">> undefined references/citations: 0"
echo ">> Output: $THESIS_DIR/main.pdf"
