# This Makefile uses pyreverse (part of pylint) to generate class diagrams
# for the classes in the packages of PyPhySim.
#
# Each package in PyPhySim should have two corresponding targets here:
# "PACKAGE_diagrams" and a "PACKAGE_diagrams_simple" targets.
#
# Each of these targets generate two PDF files, "classes_something.pdf" and
# "packages_something.pdf". The first one has the class diagram while the
# second has the package diagram. At last, in the "simple" target the class
# diagrams only have the class names.
#
# 
# Extending this Makefile
# -----------------------
#
# If new packages are added to PyPhySim, simple add the corresponding
# dependence in the regular_diagrams and simple_diagrams targets.
#
# For instance suppose a new package called "cool" was added to
# PyPhySim. You should edit this Makefile and add "classes_cool.pdf" as a
# dependence in the regular_diagrams target and "classes_simple_cool.pdf"
# as a dependence in the simple_diagrams target.
#
#
# Notes
# -----
#
# Module functions are not shown in any of these diagrams.


COMMON_COMPILE_PART:= pyreverse -o pdf -f ALL
COMMON_SIMPLE_COMPILE_PART:= pyreverse -o pdf -k

all: regular_diagrams simple_diagrams

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Regular version diagrams for all targets
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
regular_diagrams: classes_cell.pdf classes_cell.pdf classes_comp.pdf classes_ia.pdf classes_MATLAB.pdf classes_plot.pdf classes_subspace.pdf classes_util.pdf
	@echo All regular diagrams generated
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Simple version diagrams for all targets
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
simple_diagrams: classes_simple_cell.pdf classes_simple_comm.pdf classes_simple_comp.pdf classes_simple_ia.pdf classes_simple_MATLAB.pdf classes_simple_plot.pdf classes_simple_subspace.pdf classes_simple_util.pdf
	@echo All simple diagrams generated
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# General Rule to generate the regular diagrams
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
classes_%.pdf: ../%/*.py
	@echo Generating regular diagrams for the $(subst .pdf,,$(subst classes_,,$@)) package
	@$(COMMON_COMPILE_PART) ../$(subst .pdf,,$(subst classes_,,$@)) -p $(subst .pdf,,$(subst classes_,,$@)) > /dev/null
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# General Rule to generate the simple diagrams
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
classes_simple_%.pdf: ../%/*.py
	@echo Generating simple diagrams for the $(subst .pdf,,$(subst classes_simple_,,$@)) package
	@$(COMMON_SIMPLE_COMPILE_PART) ../$(subst .pdf,,$(subst classes_simple_,,$@)) -p simple_$(subst .pdf,,$(subst classes_simple_,,$@)) > /dev/null
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Remove all diagrams from the folder
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
clean:
	@rm -f *.pdf
