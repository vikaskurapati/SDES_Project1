FILES := 130010042_Paper.pdf
AUXFILES := $(FILES:.pdf=.aux)
LOGFILES := $(FILES:.pdf=.log)
BBLFILES := $(FILES:.pdf=.bbl)
BLGFILES := $(FILES:.pdf=.blg)
#OTHERS := mass_spring_damper.nav mass_spring_damper.out mass_spring_damper.snm mass_spring_damper.toc
PICS := $(wildcard *.png)
#PICSDEL := $(filter-out spring_mass_damper.png,$(PICS))
PYC := $(wildcard *.pyc)

main :
	python 130010042_plots.py
	make pdf

pdf : $(FILES:.pdf=.tex)
	make 130010042_Paper.pdf

%.pdf: %.tex
	pdflatex $<
	bibtex `echo $< |cut -d "." -f1`.aux
	pdflatex $<
	pdflatex $<

.PHONY: clean clean-all
clean-all:
	$(RM) $(AUXFILES) $(FILES) $(LOGFILES) $(BBLFILES) $(BLGFILES) $(OTHERS) $(PICSDEL) $(PYC)

clean:
	$(RM) $(AUXFILES) $(LOGFILES) $(BBLFILES) $(BLGFILES) $(OTHERS) $(PICSDEL) $(PYC)
