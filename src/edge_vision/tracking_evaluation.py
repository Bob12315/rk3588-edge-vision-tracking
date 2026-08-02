"""Compare tracker runs and evaluate MOTChallenge-format identity annotations."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MotDetection:
    frame_id: int
    track_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.frame_id < 1:
            raise ValueError("MOT frame IDs must start at 1")
        if self.track_id < 0:
            raise ValueError("track IDs must be non-negative")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("box width and height must be positive")


def box_iou(left: MotDetection, right: MotDetection) -> float:
    intersection_x1 = max(left.x, right.x)
    intersection_y1 = max(left.y, right.y)
    intersection_x2 = min(left.x + left.width, right.x + right.width)
    intersection_y2 = min(left.y + left.height, right.y + right.height)
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0.0 else 0.0


def maximum_weight_pairs(weights: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Return a deterministic maximum-weight one-to-one assignment in O(n^3)."""

    if not weights or not weights[0]:
        return []
    column_count = len(weights[0])
    if any(len(row) != column_count for row in weights):
        raise ValueError("assignment matrix must be rectangular")

    transposed = len(weights) > column_count
    matrix = (
        [[weights[row][column] for row in range(len(weights))] for column in range(column_count)]
        if transposed
        else [list(row) for row in weights]
    )
    row_count = len(matrix)
    column_count = len(matrix[0])
    potentials_rows = [0.0] * (row_count + 1)
    potentials_columns = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        matched_row[0] = row
        current_column = 0
        minimum_values = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[current_column] = True
            current_row = matched_row[current_column]
            delta = float("inf")
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                cost = (
                    -matrix[current_row - 1][column - 1]
                    - potentials_rows[current_row]
                    - potentials_columns[column]
                )
                if cost < minimum_values[column]:
                    minimum_values[column] = cost
                    previous_column[column] = current_column
                if minimum_values[column] < delta:
                    delta = minimum_values[column]
                    next_column = column
            for column in range(column_count + 1):
                if used[column]:
                    potentials_rows[matched_row[column]] += delta
                    potentials_columns[column] -= delta
                else:
                    minimum_values[column] -= delta
            current_column = next_column
            if matched_row[current_column] == 0:
                break
        while True:
            next_column = previous_column[current_column]
            matched_row[current_column] = matched_row[next_column]
            current_column = next_column
            if current_column == 0:
                break

    pairs = [
        (row - 1, column - 1)
        for column, row in enumerate(matched_row[1:], start=1)
        if row
    ]
    if transposed:
        pairs = [(column, row) for row, column in pairs]
    return sorted(pairs)


def match_detections(
    ground_truth: Sequence[MotDetection],
    predictions: Sequence[MotDetection],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be in (0, 1]")
    weights = [
        [
            overlap if (overlap := box_iou(gt, prediction)) >= iou_threshold else 0.0
            for prediction in predictions
        ]
        for gt in ground_truth
    ]
    return [
        (gt_index, prediction_index, weights[gt_index][prediction_index])
        for gt_index, prediction_index in maximum_weight_pairs(weights)
        if weights[gt_index][prediction_index] >= iou_threshold
    ]


def _group_detections(
    detections: Sequence[MotDetection],
) -> dict[int, list[MotDetection]]:
    grouped: dict[int, list[MotDetection]] = defaultdict(list)
    identities: set[tuple[int, int]] = set()
    for detection in detections:
        identity = (detection.frame_id, detection.track_id)
        if identity in identities:
            raise ValueError(
                f"duplicate track ID {detection.track_id} in frame {detection.frame_id}"
            )
        identities.add(identity)
        grouped[detection.frame_id].append(detection)
    for frame in grouped.values():
        frame.sort(key=lambda item: item.track_id)
    return dict(grouped)


def load_mot(
    path: Path,
    *,
    ground_truth: bool,
    ground_truth_class_ids: Sequence[int] = (1,),
    minimum_visibility: float = 0.0,
) -> list[MotDetection]:
    """Load MOTChallenge rows, filtering invalid/distractor ground-truth boxes."""

    if not 0.0 <= minimum_visibility <= 1.0:
        raise ValueError("minimum visibility must be in [0, 1]")
    class_ids = set(ground_truth_class_ids)
    detections: list[MotDetection] = []
    for line_number, raw_line in enumerate(
        path.expanduser().read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 6:
            raise ValueError(f"{path}:{line_number}: expected at least 6 MOT fields")
        try:
            frame_id = int(float(fields[0]))
            track_id = int(float(fields[1]))
            x, y, width, height = (float(value) for value in fields[2:6])
            confidence = float(fields[6]) if len(fields) > 6 else 1.0
            class_id = int(float(fields[7])) if len(fields) > 7 else 1
            visibility = float(fields[8]) if len(fields) > 8 else 1.0
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric MOT field") from exc
        if ground_truth and (
            confidence <= 0.0
            or (class_ids and class_id not in class_ids)
            or visibility < minimum_visibility
        ):
            continue
        detections.append(
            MotDetection(frame_id, track_id, x, y, width, height, confidence)
        )
    _group_detections(detections)
    return detections


def load_reviewed_frame_ids(path: Path) -> set[int]:
    """Load the explicit evaluation scope from an identity annotation project."""

    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
        frame_count = int(payload["video"]["frame_count"])
        reviewed = {int(frame_id) for frame_id in payload["reviewed_frames"]}
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid annotation project: {path}") from exc
    if not reviewed:
        raise ValueError("annotation project contains no reviewed frames")
    if min(reviewed) < 1 or max(reviewed) > frame_count:
        raise ValueError("annotation project contains an out-of-range reviewed frame")
    return reviewed


def load_detection_jsonl(
    path: Path,
) -> tuple[list[MotDetection], int, int]:
    """Load this project's JSONL records as tracked MOT detections."""

    detections: list[MotDetection] = []
    processed_frames = 0
    untracked_observations = 0
    with path.expanduser().open(encoding="utf-8") as records:
        for line_number, raw_line in enumerate(records, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
                frame_id = int(record["frame_id"]) + 1
                frame_detections = record.get("detections", [])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid detection record") from exc
            processed_frames += 1
            for item in frame_detections:
                track_id = item.get("track_id")
                if track_id is None:
                    untracked_observations += 1
                    continue
                x1, y1, x2, y2 = (float(value) for value in item["box_xyxy_pixels"])
                detections.append(
                    MotDetection(
                        frame_id=frame_id,
                        track_id=int(track_id),
                        x=x1,
                        y=y1,
                        width=x2 - x1,
                        height=y2 - y1,
                        confidence=float(item.get("confidence", 1.0)),
                    )
                )
    _group_detections(detections)
    return detections, processed_frames, untracked_observations


def export_mot_jsonl(source: Path, destination: Path) -> Mapping[str, Any]:
    detections, processed_frames, untracked = load_detection_jsonl(source)
    destination = destination.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        (
            f"{item.frame_id},{item.track_id},{item.x:.3f},{item.y:.3f},"
            f"{item.width:.3f},{item.height:.3f},{item.confidence:.6f},-1,-1,-1"
        )
        for item in detections
    ]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {
        "source": str(source.expanduser().resolve()),
        "output": str(destination.resolve()),
        "processed_frames": processed_frames,
        "exported_detections": len(detections),
        "skipped_untracked_observations": untracked,
    }


def tracking_diagnostics(
    detections: Sequence[MotDetection],
    *,
    processed_frames: int,
    untracked_observations: int = 0,
    iou_threshold: float = 0.5,
) -> Mapping[str, Any]:
    grouped = _group_detections(detections)
    frames_by_track: dict[int, list[int]] = defaultdict(list)
    confidence_by_track: dict[int, list[float]] = defaultdict(list)
    for item in detections:
        frames_by_track[item.track_id].append(item.frame_id)
        confidence_by_track[item.track_id].append(item.confidence)

    lengths = [len(frames) for frames in frames_by_track.values()]
    segments_by_track: dict[int, int] = {}
    for track_id, frames in frames_by_track.items():
        ordered = sorted(frames)
        segments_by_track[track_id] = 1 + sum(
            current > previous + 1
            for previous, current in zip(ordered, ordered[1:])
        )

    proxy_matches = 0
    proxy_switches = 0
    for frame_id in range(2, processed_frames + 1):
        previous = grouped.get(frame_id - 1, ())
        current = grouped.get(frame_id, ())
        for previous_index, current_index, _ in match_detections(
            previous, current, iou_threshold
        ):
            proxy_matches += 1
            if previous[previous_index].track_id != current[current_index].track_id:
                proxy_switches += 1

    fragmented = {
        str(track_id): segments
        for track_id, segments in sorted(segments_by_track.items())
        if segments > 1
    }
    track_details = [
        {
            "track_id": track_id,
            "observed_frames": len(frames),
            "first_frame": min(frames),
            "last_frame": max(frames),
            "segments": segments_by_track[track_id],
            "mean_confidence": statistics.fmean(confidence_by_track[track_id]),
        }
        for track_id, frames in sorted(
            frames_by_track.items(), key=lambda item: (-len(item[1]), item[0])
        )
    ]
    established_minimum_frames = 5
    return {
        "processed_frames": processed_frames,
        "tracked_observations": len(detections),
        "untracked_observations": untracked_observations,
        "frames_with_tracks": len(grouped),
        "unique_track_ids": len(frames_by_track),
        "established_minimum_frames": established_minimum_frames,
        "established_track_ids": sum(
            length >= established_minimum_frames for length in lengths
        ),
        "short_track_ids": [
            detail["track_id"]
            for detail in track_details
            if detail["observed_frames"] < established_minimum_frames
        ],
        "tracks": track_details,
        "track_length_frames": {
            "min": min(lengths, default=0),
            "median": statistics.median(lengths) if lengths else 0.0,
            "mean": statistics.fmean(lengths) if lengths else 0.0,
            "max": max(lengths, default=0),
        },
        "single_frame_track_ids": sum(length == 1 for length in lengths),
        "fragmented_track_ids": fragmented,
        "extra_track_segments": sum(segments - 1 for segments in segments_by_track.values()),
        "spatial_continuity_proxy": {
            "iou_threshold": iou_threshold,
            "consecutive_frame_matches": proxy_matches,
            "approximate_id_switches": proxy_switches,
            "approximate_switch_rate": (
                proxy_switches / proxy_matches if proxy_matches else 0.0
            ),
            "warning": "Proxy only; do not report as ground-truth ID switches.",
        },
    }


def diagnose_jsonl(path: Path, iou_threshold: float = 0.5) -> Mapping[str, Any]:
    detections, processed_frames, untracked = load_detection_jsonl(path)
    report = dict(
        tracking_diagnostics(
            detections,
            processed_frames=processed_frames,
            untracked_observations=untracked,
            iou_threshold=iou_threshold,
        )
    )
    summary_path = path.expanduser().with_name("summary.json")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        report["source"] = summary.get("source")
        report["tracker"] = summary.get("inference", {}).get("tracker")
        report["tracker_config"] = summary.get("inference", {}).get("tracker_config")
        report["throughput_fps"] = summary.get("performance", {}).get("throughput_fps")
        report["mean_inference_ms"] = (
            summary.get("performance", {}).get("inference_ms", {}).get("mean")
        )
    report["detections_jsonl"] = str(path.expanduser().resolve())
    return report


def compare_jsonl_runs(
    bytetrack_path: Path,
    reid_path: Path,
    *,
    iou_threshold: float = 0.5,
) -> Mapping[str, Any]:
    bytetrack = diagnose_jsonl(bytetrack_path, iou_threshold)
    reid = diagnose_jsonl(reid_path, iou_threshold)
    warnings: list[str] = []
    if bytetrack.get("source") and reid.get("source"):
        if bytetrack["source"] != reid["source"]:
            warnings.append("runs use different source videos")
    if bytetrack["processed_frames"] != reid["processed_frames"]:
        warnings.append("runs contain different processed frame counts")
    byte_fps = bytetrack.get("throughput_fps")
    reid_fps = reid.get("throughput_fps")
    return {
        "schema_version": 1,
        "runs": {"bytetrack": bytetrack, "reid": reid},
        "comparison": {
            "reid_to_bytetrack_fps_ratio": (
                reid_fps / byte_fps if byte_fps and reid_fps else None
            ),
            "unique_track_id_delta": (
                reid["unique_track_ids"] - bytetrack["unique_track_ids"]
            ),
            "established_track_id_delta": (
                reid["established_track_ids"] - bytetrack["established_track_ids"]
            ),
            "approximate_id_switch_delta": (
                reid["spatial_continuity_proxy"]["approximate_id_switches"]
                - bytetrack["spatial_continuity_proxy"]["approximate_id_switches"]
            ),
            "warning": (
                "Unique IDs and lower proxy counts do not prove identity quality without GT."
            ),
        },
        "warnings": warnings,
    }


def evaluate_mot(
    ground_truth: Sequence[MotDetection],
    predictions: Sequence[MotDetection],
    *,
    iou_threshold: float = 0.5,
) -> Mapping[str, Any]:
    """Compute deterministic local CLEAR/identity metrics for rapid iteration."""

    gt_frames = _group_detections(ground_truth)
    prediction_frames = _group_detections(predictions)
    frame_ids = sorted(set(gt_frames) | set(prediction_frames))
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    id_switches = 0
    overlap_total = 0.0
    previous_frame_matches: dict[int, int] = {}
    last_identity_matches: dict[int, int] = {}
    identity_counts: dict[tuple[int, int], int] = defaultdict(int)

    for frame_id in frame_ids:
        gt = gt_frames.get(frame_id, [])
        predicted = prediction_frames.get(frame_id, [])
        gt_by_id = {item.track_id: (index, item) for index, item in enumerate(gt)}
        pred_by_id = {
            item.track_id: (index, item) for index, item in enumerate(predicted)
        }
        matched_gt: set[int] = set()
        matched_predicted: set[int] = set()
        frame_matches: list[tuple[int, int, float]] = []

        for gt_id, prediction_id in previous_frame_matches.items():
            if gt_id not in gt_by_id or prediction_id not in pred_by_id:
                continue
            gt_index, gt_item = gt_by_id[gt_id]
            prediction_index, prediction_item = pred_by_id[prediction_id]
            overlap = box_iou(gt_item, prediction_item)
            if overlap < iou_threshold:
                continue
            matched_gt.add(gt_index)
            matched_predicted.add(prediction_index)
            frame_matches.append((gt_index, prediction_index, overlap))

        remaining_gt = [
            (index, item) for index, item in enumerate(gt) if index not in matched_gt
        ]
        remaining_predictions = [
            (index, item)
            for index, item in enumerate(predicted)
            if index not in matched_predicted
        ]
        for gt_index, prediction_index, overlap in match_detections(
            [item for _, item in remaining_gt],
            [item for _, item in remaining_predictions],
            iou_threshold,
        ):
            frame_matches.append(
                (
                    remaining_gt[gt_index][0],
                    remaining_predictions[prediction_index][0],
                    overlap,
                )
            )

        current_frame_matches: dict[int, int] = {}
        for gt_index, prediction_index, overlap in frame_matches:
            gt_id = gt[gt_index].track_id
            prediction_id = predicted[prediction_index].track_id
            if (
                gt_id in last_identity_matches
                and last_identity_matches[gt_id] != prediction_id
            ):
                id_switches += 1
            last_identity_matches[gt_id] = prediction_id
            current_frame_matches[gt_id] = prediction_id
            identity_counts[(gt_id, prediction_id)] += 1
            overlap_total += overlap

        previous_frame_matches = current_frame_matches
        true_positives += len(frame_matches)
        false_negatives += len(gt) - len(frame_matches)
        false_positives += len(predicted) - len(frame_matches)

    gt_ids = sorted({item.track_id for item in ground_truth})
    prediction_ids = sorted({item.track_id for item in predictions})
    identity_weights = [
        [identity_counts[(gt_id, prediction_id)] for prediction_id in prediction_ids]
        for gt_id in gt_ids
    ]
    identity_true_positives = sum(
        identity_weights[gt_index][prediction_index]
        for gt_index, prediction_index in maximum_weight_pairs(identity_weights)
    )
    total_gt = len(ground_truth)
    total_predictions = len(predictions)
    identity_false_negatives = total_gt - identity_true_positives
    identity_false_positives = total_predictions - identity_true_positives
    idf1_denominator = (
        2 * identity_true_positives
        + identity_false_positives
        + identity_false_negatives
    )
    return {
        "schema_version": 1,
        "iou_threshold": iou_threshold,
        "counts": {
            "ground_truth_detections": total_gt,
            "predicted_detections": total_predictions,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "id_switches": id_switches,
        },
        "detection": {
            "precision": (
                true_positives / (true_positives + false_positives)
                if true_positives + false_positives
                else 0.0
            ),
            "recall": true_positives / total_gt if total_gt else 0.0,
            "mota": (
                1.0 - (false_negatives + false_positives + id_switches) / total_gt
                if total_gt
                else 0.0
            ),
            "motp_iou": overlap_total / true_positives if true_positives else 0.0,
        },
        "identity": {
            "id_true_positives": identity_true_positives,
            "id_false_positives": identity_false_positives,
            "id_false_negatives": identity_false_negatives,
            "idf1": (
                2 * identity_true_positives / idf1_denominator
                if idf1_denominator
                else 0.0
            ),
        },
        "warnings": [
            "Use official TrackEval for publication-grade HOTA and benchmark submission."
        ],
    }


def _write_or_print(report: Mapping[str, Any], output: Path | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        print(rendered, end="")
        return
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output.resolve())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    diagnose = commands.add_parser("diagnose", help="summarize one tracking JSONL")
    diagnose.add_argument("detections", type=Path)
    diagnose.add_argument("--iou-threshold", type=float, default=0.5)
    diagnose.add_argument("--output", type=Path)

    compare = commands.add_parser("compare", help="compare ByteTrack and ReID JSONL runs")
    compare.add_argument("--bytetrack", type=Path, required=True)
    compare.add_argument("--reid", type=Path, required=True)
    compare.add_argument("--iou-threshold", type=float, default=0.5)
    compare.add_argument("--output", type=Path)

    export = commands.add_parser("export-mot", help="convert project JSONL to MOT text")
    export.add_argument("detections", type=Path)
    export.add_argument("output", type=Path)

    evaluate = commands.add_parser("evaluate", help="evaluate MOT predictions against GT")
    evaluate.add_argument("ground_truth", type=Path)
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("--iou-threshold", type=float, default=0.5)
    evaluate.add_argument("--gt-class-ids", nargs="+", type=int, default=[1])
    evaluate.add_argument("--minimum-visibility", type=float, default=0.0)
    evaluate.add_argument(
        "--annotation-project",
        type=Path,
        help="annotations.json used to restrict scoring to manually reviewed frames",
    )
    evaluate.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "diagnose":
        report = diagnose_jsonl(args.detections, args.iou_threshold)
        _write_or_print(report, args.output)
        return
    if args.command == "compare":
        report = compare_jsonl_runs(
            args.bytetrack,
            args.reid,
            iou_threshold=args.iou_threshold,
        )
        _write_or_print(report, args.output)
        return
    if args.command == "export-mot":
        _write_or_print(export_mot_jsonl(args.detections, args.output), None)
        return

    ground_truth = load_mot(
        args.ground_truth,
        ground_truth=True,
        ground_truth_class_ids=args.gt_class_ids,
        minimum_visibility=args.minimum_visibility,
    )
    predictions = load_mot(args.predictions, ground_truth=False)
    reviewed_frames = None
    if args.annotation_project is not None:
        reviewed_frames = load_reviewed_frame_ids(args.annotation_project)
        ground_truth = [
            item for item in ground_truth if item.frame_id in reviewed_frames
        ]
        predictions = [
            item for item in predictions if item.frame_id in reviewed_frames
        ]
    report = evaluate_mot(
        ground_truth,
        predictions,
        iou_threshold=args.iou_threshold,
    )
    report = {
        **report,
        "ground_truth": str(args.ground_truth.expanduser().resolve()),
        "predictions": str(args.predictions.expanduser().resolve()),
        "evaluation_scope": {
            "mode": "reviewed_frames" if reviewed_frames is not None else "all_rows",
            "annotation_project": (
                str(args.annotation_project.expanduser().resolve())
                if args.annotation_project is not None
                else None
            ),
            "reviewed_frame_count": (
                len(reviewed_frames) if reviewed_frames is not None else None
            ),
        },
    }
    _write_or_print(report, args.output)


if __name__ == "__main__":
    main()
