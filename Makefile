.PHONY: check demo detect-person detect-person-yolo detect-white-person track-white-person track-white-person-local-vlm run-white-person vlm-install vlm-serve vlm-pull test

VIDEO ?=
WORLD_OUTPUT_DIR ?= outputs/person_yolo_world
YOLO_OUTPUT_DIR ?= outputs/person_yolo
WHITE_OUTPUT_DIR ?= outputs/white_clothes_yolo_world_verified
TRACK_OUTPUT_DIR ?= outputs/white_clothes_yolo_world_bytetrack
RUNTIME_OUTPUT_DIR ?= outputs/white_person_mission_runtime
WORLD_CONFIDENCE ?= 0.05
OLLAMA_BIN ?= artifacts/ollama-runtime/bin/ollama
VLM_MODEL ?= qwen3-vl:2b

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

track-white-person:
	@test -n "$(VIDEO)" || (echo "usage: make track-white-person VIDEO=/path/to/video.mp4" >&2; exit 2)
	PYTHONPATH=src .venv/bin/python -m edge_vision.video_detection "$(VIDEO)" --backend yolo-world --vlm-plan configs/vlm_white_clothing_example.json --confidence "$(WORLD_CONFIDENCE)" --white-clothing --tracker bytetrack --tracker-config configs/bytetrack.yaml --output-dir "$(TRACK_OUTPUT_DIR)"

track-white-person-local-vlm:
	@test -n "$(VIDEO)" || (echo "usage: make track-white-person-local-vlm VIDEO=/path/to/video.mp4" >&2; exit 2)
	PYTHONPATH=src .venv/bin/python -m edge_vision.video_detection "$(VIDEO)" --backend yolo-world --vlm-provider ollama --vlm-model "$(VLM_MODEL)" --vlm-frame-index 0 --confidence "$(WORLD_CONFIDENCE)" --white-clothing --tracker bytetrack --tracker-config configs/bytetrack.yaml --output-dir "$(TRACK_OUTPUT_DIR)"

run-white-person:
	@test -n "$(VIDEO)" || (echo "usage: make run-white-person VIDEO=/path/to/video.mp4" >&2; exit 2)
	PYTHONPATH=src .venv/bin/python -m edge_vision.realtime "$(VIDEO)" --vlm-provider ollama --vlm-model "$(VLM_MODEL)" --confidence "$(WORLD_CONFIDENCE)" --output-dir "$(RUNTIME_OUTPUT_DIR)"

vlm-install:
	./scripts/install_local_vlm.sh

vlm-serve:
	./scripts/start_local_vlm.sh

vlm-pull:
	OLLAMA_HOST=127.0.0.1:11434 "$(OLLAMA_BIN)" pull "$(VLM_MODEL)"

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v
