TGEC := $(shell (echo ${TGEC}))

SOURCES = noc.py \
	setup.py \
	nocpkg/__init__.py \
	nocpkg/ComputeOptimal.py \
	nocpkg/LevMarAlgo.py \
	nocpkg/NOCMain.py \
	nocpkg/Parameter.py \
	nocpkg/Seismic.py \
	nocpkg/SeismicConstraints.py \
	nocpkg/Setup.py \
	nocpkg/Target.py \
	nocpkg/utils.py \
	tgec/__init__.py \
	tgec/constants.py \
	tgec/Model.py \
	tgec/Parameters.py \
	tgec/RunModel.py \
	tgec/SeismicModel.py

all: build install

default debug: install

debug: install

build: $(SOURCES)
	@python3 setup.py -q build

install: $(SOURCES)
	@python3 -m pip install .

clean clean_all:
	@python3 -m pip uninstall -y noc
	@rm -rf build
	@rm -rf nocpkg/__pycache__
	@rm -rf tgec/__pycache__
	@rm -rf *.pyc
