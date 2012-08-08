
all: classes_py-physim.pdf packages_py-physim.pdf

# PHONY target only because I'm not including the prerequisite for them
# right now
.PHONY: classes_py-physim.pdf packages_py-physim.pdf
classes_py-physim.pdf packages_py-physim.pdf:
	@pyreverse ../../ -p py-physim -o pdf

clean:
	@rm -f classes_py-physim.pdf packages_py-physim.pdf *~
