.PHONY: check demo detect-person detect-person-yolo detect-white-person test

VIDEO ?=
WORLD_OUTPUT_DIR ?= outputs/person_yolo_world
YOLO_OUTPUT_DIR ?= outputs/person_yolo
WHITE_OUTPUT_DIR ?= outputs/white_clothes_yolo_world_verified
WORLD_CONFIDENCE ?= 0.05

check:
	python3 -m compileall -q src tests
	PYTHONPATH=src python3 -m unittest discover -s tests -v

demo:
	PYTHONPATH=src python3 -m edge_vision.cli --scenario nominal

detect-person:
	@test -n "$(VIDEO)" || (echo "usage: make detect-person VIDEO=/path/to/video.mp4" >&2; exit 2)
	PYTHONPATH=src .venv/bin/python -m edge_vision.video_detection "$(VIDEO)" --backend yolo-world --prompts person --confidence "$(WORLD_CONFIDENCE)" --output-dir "$(WORLD_OUTPUT_DIR)"

detect-person-yolo:
	@test -n "$(VIDEO)" || (echo "usage: make detect-person-yolo VIDEO=/path/to/video.mp4" >&2; exit 2)
	PYTHONPATH=src .venv/bin/python -m edge_vision.video_detection "$(VIDEO)" --backend yolo --class-ids 0 --output-dir "$(YOLO_OUTPUT_DIR)"

detect-white-person:
	@test -n "$(VIDEO)" || (echo "usage: make detect-white-person VIDEO=/path/to/video.mp4" >&2; exit 2)
	PYTHONPATH=src .venv/bin/python -m edge_vision.video_detection "$(VIDEO)" --backend yolo-world --prompts person --confidence "$(WORLD_CONFIDENCE)" --white-clothing --output-dir "$(WHITE_OUTPUT_DIR)"

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v
