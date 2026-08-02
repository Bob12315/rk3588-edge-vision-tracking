"""Local browser UI for reviewed person-identity MOT annotations."""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

from .identity_annotation import IdentityAnnotationStore, load_prediction_suggestions


class AnnotationSession:
    def __init__(
        self,
        source: Path,
        output_dir: Path,
        *,
        predictions: Path | None = None,
        jpeg_quality: int = 90,
    ) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("annotation UI requires OpenCV") from exc

        self._cv2 = cv2
        self.source = source.expanduser().resolve()
        if not self.source.is_file():
            raise FileNotFoundError(self.source)
        self._capture = cv2.VideoCapture(str(self.source))
        if not self._capture.isOpened():
            raise RuntimeError(f"cannot open video: {self.source}")
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if (
            self.frame_count < 1
            or self.width < 1
            or self.height < 1
            or self.fps <= 0.0
        ):
            self._capture.release()
            raise RuntimeError("video metadata is invalid")

        self.jpeg_quality = jpeg_quality
        self.store = IdentityAnnotationStore(
            output_dir,
            source=self.source,
            frame_count=self.frame_count,
            width=self.width,
            height=self.height,
            fps=self.fps,
        )
        self.predictions_path = (
            predictions.expanduser().resolve() if predictions is not None else None
        )
        if self.predictions_path is not None and not self.predictions_path.is_file():
            self._capture.release()
            raise FileNotFoundError(self.predictions_path)
        self._suggestions = (
            load_prediction_suggestions(
                self.predictions_path,
                frame_count=self.frame_count,
                width=self.width,
                height=self.height,
            )
            if self.predictions_path is not None
            else {}
        )
        self._lock = threading.RLock()
        self._frame_cache: OrderedDict[int, bytes] = OrderedDict()

    def project(self) -> Mapping[str, Any]:
        project = dict(self.store.project())
        project["predictions"] = {
            "loaded": self.predictions_path is not None,
            "source": str(self.predictions_path) if self.predictions_path else None,
            "frames_with_suggestions": len(self._suggestions),
            "warning": "Model suggestions are not ground truth until manually reviewed.",
        }
        return project

    def frame(self, frame_id: int) -> Mapping[str, Any]:
        payload = dict(self.store.frame(frame_id))
        payload["suggestions"] = [
            dict(item) for item in self._suggestions.get(frame_id, ())
        ]
        return payload

    def update_frame(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        frame_id = int(payload.get("frame_id", 0))
        annotations = payload.get("annotations", [])
        if not isinstance(annotations, list):
            raise ValueError("annotations must be a list")
        frame = dict(self.store.update_frame(frame_id, annotations))
        frame["suggestions"] = [
            dict(item) for item in self._suggestions.get(frame_id, ())
        ]
        return {"frame": frame, "project": self.project()}

    def frame_jpeg(self, frame_id: int) -> bytes:
        self.store.frame(frame_id)
        with self._lock:
            if frame_id in self._frame_cache:
                encoded = self._frame_cache.pop(frame_id)
                self._frame_cache[frame_id] = encoded
                return encoded
            self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
            ok, frame = self._capture.read()
            if not ok:
                raise RuntimeError(f"cannot decode frame {frame_id}")
            ok, encoded = self._cv2.imencode(
                ".jpg",
                frame,
                [self._cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if not ok:
                raise RuntimeError(f"cannot encode frame {frame_id}")
            result = encoded.tobytes()
            self._frame_cache[frame_id] = result
            while len(self._frame_cache) > 8:
                self._frame_cache.popitem(last=False)
            return result

    def close(self) -> None:
        with self._lock:
            self._capture.release()
            self._frame_cache.clear()


class AnnotationHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def build_handler(
    session: AnnotationSession, *, static_dir: Path
) -> type[BaseHTTPRequestHandler]:
    static_files = {
        "/": (static_dir / "annotation.html", "text/html; charset=utf-8"),
        "/annotation.js": (
            static_dir / "annotation.js",
            "text/javascript; charset=utf-8",
        ),
        "/annotation.css": (static_dir / "annotation.css", "text/css; charset=utf-8"),
        "/favicon.svg": (static_dir / "favicon.svg", "image/svg+xml"),
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "IdentityAnnotationUI/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in static_files:
                file_path, content_type = static_files[parsed.path]
                self._send_bytes(file_path.read_bytes(), content_type)
                return
            try:
                if parsed.path == "/api/project":
                    self._send_json(session.project())
                    return
                if parsed.path == "/api/frame":
                    frame_id = self._frame_id(parsed.query)
                    self._send_json(session.frame(frame_id))
                    return
                if parsed.path == "/api/frame-image":
                    frame_id = self._frame_id(parsed.query)
                    self._send_bytes(session.frame_jpeg(frame_id), "image/jpeg")
                    return
                self._send_json({"error": "not found"}, status=404)
            except (ValueError, FileNotFoundError) as exc:
                self._send_json({"error": str(exc)}, status=400)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)

        def do_POST(self) -> None:
            try:
                if urlparse(self.path).path != "/api/frame":
                    self._send_json({"error": "not found"}, status=404)
                    return
                self._send_json(session.update_frame(self._read_json()))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)

        @staticmethod
        def _frame_id(query: str) -> int:
            values = parse_qs(query).get("frame", [])
            if len(values) != 1:
                raise ValueError("one frame query parameter is required")
            return int(values[0])

        def _read_json(self) -> Mapping[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0 or length > 256 * 1024:
                raise ValueError("invalid JSON request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("JSON request must be an object")
            return payload

        def _send_json(self, payload: Mapping[str, Any], status: int = 200) -> None:
            self._send_bytes(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def _send_bytes(
            self, data: bytes, content_type: str, status: int = 200
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'",
            )
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args: object) -> None:
            if urlparse(self.path).path in {"/api/project", "/api/frame-image"}:
                return
            print(f"annotation-ui {self.address_string()} {fmt % args}")

    return Handler


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--open-browser", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    source = args.source.expanduser().resolve()
    output_dir = args.output_dir or Path("datasets/identity_eval") / source.stem
    session = AnnotationSession(
        source,
        output_dir,
        predictions=args.predictions,
        jpeg_quality=args.jpeg_quality,
    )
    static_dir = Path(__file__).with_name("web")
    server = AnnotationHttpServer(
        (args.host, args.port), build_handler(session, static_dir=static_dir)
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"Identity Annotation UI: {url}")
    print(f"Project JSON: {session.store.project_path}")
    print(f"MOT ground truth: {session.store.ground_truth_path}")
    print("Press Ctrl+C to stop.")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("Warning: the annotation UI has no authentication.")
    if args.open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        session.close()


if __name__ == "__main__":
    main()
