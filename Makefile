.PHONY: check demo detect-person test

VIDEO ?=
OUTPUT_DIR ?= outputs/person_baseline

check:
	python3 -m compileall -q src tests
	PYTHONPATH=src python3 -m unittest discover -s tests -v

demo:
	PYTHONPATH=src python3 -m edge_vision.cli --scenario nominal

detect-person:
	@test -n "$(VIDEO)" || (echo "usage: make detect-person VIDEO=/path/to/video.mp4" >&2; exit 2)
	PYTHONPATH=src .venv/bin/python -m edge_vision.video_detection "$(VIDEO)" --output-dir "$(OUTPUT_DIR)"

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v
