# Enter the name(s) of the .tex file(s) that you want to compile.
TEX_FILES = thresher nonnegative

# You shouldn't need to edit below here.
LATEX     = pdflatex
RM        = rm -rf

TMP_SUFFS = pdf aux bbl blg log dvi ps eps out
SUFF      = pdf

DOCS = $(foreach d, ${TEX_FILES}, ${d}.${SUFF})
CHECK_RERUN = grep Rerun $*.log

default: ${DOCS}

%.pdf: %.tex
	${LATEX} $<
	( ${CHECK_RERUN} && ${LATEX} $< ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} $< ) || echo "Done."

clean:
	${RM} $(foreach d, ${TEX_FILES}, $(foreach suff, ${TMP_SUFFS}, ${d}.${suff}))

