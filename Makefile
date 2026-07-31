.PHONY: check demo test

check:
	python3 -m compileall -q src tests
	PYTHONPATH=src python3 -m unittest discover -s tests -v

demo:
	PYTHONPATH=src python3 -m edge_vision.cli --scenario nominal

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v
