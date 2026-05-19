"""EO/IR 융합 드론 탐지 추론 클라이언트 (DeepX M1 NPU).

두 가지 실행 모드:
  - WebSocket 모드 (--ws_url): 640×640 JPEG + 검출 결과를 DPX1 바이너리 패킷으로 실시간 전송
  - 파일 저장 모드 (기본):     프레임별 JPEG + JSON을 --output_dir에 저장
"""

import argparse
import asyncio
import inspect
import json
import os
import queue
import ssl
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.preprocessing import extract_features
from src.core.wbf import fuse_results

try:
    import dx_engine
    DEEPX_AVAILABLE = True
except ImportError:
    DEEPX_AVAILABLE = False
    print("[WARN] dx_engine not found — NPU inference unavailable.")

try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False


MAGIC = b"DPX1"
VERSION = 1
MODEL_VERSION = "fusion-npu-1.0"
OUTPUT_SIZE = 640
JPEG_QUALITY = 85

_SCALER_MEAN = np.array([0.13293, 0.59114, 0.40444, 0.09676], dtype=np.float32)
_SCALER_STD  = np.array([0.06240, 0.19376, 0.19958, 0.12217], dtype=np.float32)


def normalize_features(feat: np.ndarray) -> np.ndarray:
    return (feat - _SCALER_MEAN) / (_SCALER_STD + 1e-8)


# ── .env loader ──────────────────────────────────────────────────────────────

def _load_dotenv_manual(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def _load_env() -> None:
    candidates = [
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    for p in candidates:
        if p.exists():
            if _DOTENV_AVAILABLE:
                load_dotenv(p, override=False)
            else:
                _load_dotenv_manual(p)
            print(f"[i] loaded env: {p}")
            return


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="DeepX NPU EO/IR Fusion Inference — WebSocket streaming or file output"
    )
    p.add_argument("--eo_video",      default="data/video/visible.mp4")
    p.add_argument("--ir_video",      default="data/video/infrared.mp4")

    p.add_argument("--eo_model",      default="weights/dxcom/yolo26n_eo.dxnn")
    p.add_argument("--ir_model",      default="weights/dxcom/yolo26n_ir.dxnn")
    p.add_argument("--mlp_model",     default="weights/dxcom/fusion.dxnn")

    p.add_argument("--conf",          type=float, default=0.25)
    p.add_argument("--iou_thr",       type=float, default=0.55,
                   help="같은 모달리티 내 WBF 병합 IoU 임계값")

    p.add_argument("--fps",           type=float,
                   default=float(os.environ.get("FPS", "0")),
                   help="Max streaming FPS (0 = unlimited)")
    p.add_argument("--loop",          action="store_true", help="Loop video when it ends")

    p.add_argument("--class_names",
                   default=os.environ.get("CLASS_NAMES", "UAV"),
                   help="콤마 구분 클래스명 목록")

    p.add_argument("--ws_url",        default=os.environ.get("WS_URL"))
    p.add_argument("--ws_client_id",  default=os.environ.get("WS_CLIENT_ID", "fusion-01"))
    p.add_argument("--ws_token",      default=os.environ.get("WS_TOKEN", "change-me"))
    p.add_argument("--cert",          default=os.environ.get("CERT"))

    p.add_argument("--output_modal",  default="eo", choices=["eo", "ir"])
    p.add_argument("--max_dets",      type=int,
                   default=int(os.environ.get("MAX_DETS", "0")))

    p.add_argument("--send_queue",    type=int, default=16,
                   help="WebSocket 송신 큐 크기")
    p.add_argument("--prefetch",      type=int, default=64,
                   help="비디오 프레임 prefetch 큐 크기 (RAM 여유 있으면 늘려도 OK)")
    p.add_argument("--cpu_workers",   type=int, default=4,
                   help="추론+인코딩용 CPU 워커 수 (코어 수에 맞춤)")
    p.add_argument("--lookahead",     type=int, default=1,
                   help="동시 in-flight 프레임 수 (0=직렬, 1=한 프레임 앞서, 2+는 prev_eo 정확성 손실)")
    p.add_argument("--no_parallel",   action="store_true",
                   help="EO/IR 병렬 추론 비활성화 (디버그용)")

    p.add_argument("--output_dir",    default="output/demo_video")
    return p.parse_args()


# ── DeepX runtime — 비동기 API 자동 감지 ──────────────────────────────────────

class DXModel:
    """DeepX M1 NPU 추론 엔진 래퍼.

    dx_engine의 비동기 API를 런타임에 자동 감지합니다:
      - run() / Run()      : 동기 (필수)
      - run_async() / RunAsync() : 비동기 (있으면 사용)
    """

    def __init__(self, path: str):
        if not DEEPX_AVAILABLE:
            raise RuntimeError("dx_engine not installed.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"DeepX model not found: {path}")
        self._model = dx_engine.InferenceEngine(path)

        # 동기 메서드 (필수)
        self._run = getattr(self._model, "run", None) or getattr(self._model, "Run", None)
        if self._run is None:
            raise RuntimeError("dx_engine.InferenceEngine has no run() / Run() method")

        # 비동기 메서드 (선택)
        self._run_async = (
            getattr(self._model, "run_async", None)
            or getattr(self._model, "RunAsync", None)
        )
        # 비동기 결과 수거 메서드
        self._wait = (
            getattr(self._model, "wait", None)
            or getattr(self._model, "Wait", None)
            or getattr(self._model, "get_result", None)
            or getattr(self._model, "GetResult", None)
        )

    @classmethod
    def report_capabilities(cls, model):
        """디버그용: dx_engine InferenceEngine의 메서드 목록 출력."""
        methods = [m for m in dir(model._model) if not m.startswith("_")]
        print(f"  dx_engine methods: {methods}")

    def run(self, inputs: list) -> list:
        return self._run(inputs)

    def has_async(self) -> bool:
        return self._run_async is not None and self._wait is not None

    def submit_async(self, inputs: list):
        """비동기 제출. 핸들/요청ID를 반환 (Wait에 사용)."""
        return self._run_async(inputs)

    def wait_async(self, handle) -> list:
        return self._wait(handle)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_eo(img: np.ndarray) -> np.ndarray:
    r = cv2.resize(img, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(r[np.newaxis].astype(np.uint8))


def preprocess_ir(img: np.ndarray) -> np.ndarray:
    r = cv2.resize(img, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(r[np.newaxis].astype(np.uint8))


# ── Decoding ──────────────────────────────────────────────────────────────────

def decode_yolo(raw: list, conf_thres: float, img_w: int, img_h: int) -> dict:
    out  = raw[0].squeeze(0)
    mask = out[:, 4] >= conf_thres
    out  = out[mask]
    if len(out) == 0:
        return {"boxes":  np.zeros((0, 4), dtype=np.float32),
                "scores": np.array([], dtype=np.float32),
                "labels": np.array([], dtype=np.float32)}
    sx, sy = img_w / 640.0, img_h / 640.0
    boxes  = out[:, :4] * np.array([sx, sy, sx, sy], dtype=np.float32)
    boxes  = np.clip(boxes, 0, [img_w, img_h, img_w, img_h]).astype(np.float32)
    return {"boxes":  boxes,
            "scores": out[:, 4].astype(np.float32),
            "labels": out[:, 5].astype(np.float32)}


def decode_mlp(raw: list) -> float:
    return float(np.clip(raw[0].flatten()[0], 0.0, 1.0))


# ── Parallel inference core ──────────────────────────────────────────────────

class ParallelInfer:
    """EO ‖ IR ‖ MLP 동시 추론 컨트롤러.

    DeepX SDK가 RunAsync를 제공하면 그쪽을 우선 사용 (진짜 비동기).
    없으면 ThreadPoolExecutor로 두 모델을 별도 스레드에서 동시 호출.
    GIL 영향이 의심되면 --no_parallel 로 직렬 폴백 가능.
    """

    def __init__(self, eo_model: DXModel, ir_model: DXModel, mlp_model: DXModel,
                 parallel: bool = True):
        self.eo  = eo_model
        self.ir  = ir_model
        self.mlp = mlp_model
        self.parallel = parallel
        self.use_async = (
            parallel
            and eo_model.has_async()
            and ir_model.has_async()
            and mlp_model.has_async()
        )
        self._pool = ThreadPoolExecutor(max_workers=3) if parallel else None

        if self.use_async:
            print("[i] inference path: dx_engine native async (RunAsync)")
        elif parallel:
            print("[i] inference path: ThreadPoolExecutor parallel (3 workers)")
        else:
            print("[i] inference path: sequential (debug mode)")

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)

    def infer(self, eo_640: np.ndarray, ir_640: np.ndarray,
              prev_eo: np.ndarray, conf: float) -> tuple:
        """한 프레임에 대해 EO + IR + MLP 추론을 동시 수행.

        Returns: (eo_out, ir_out, fusion_weight)
        모든 박스는 640x640 픽셀 공간 (EO/IR 입력이 이미 640x640으로 리사이즈됨).
        """
        eo_inp = preprocess_eo(eo_640)
        ir_inp = preprocess_ir(ir_640)
        feat   = extract_features(eo_640, prev_eo)
        feat_n = np.ascontiguousarray(
                    normalize_features(feat).astype(np.float32)[np.newaxis])

        if self.use_async:
            h_eo = self.eo.submit_async([eo_inp])
            h_ir = self.ir.submit_async([ir_inp])
            h_ml = self.mlp.submit_async([feat_n])
            eo_raw = self.eo.wait_async(h_eo)
            ir_raw = self.ir.wait_async(h_ir)
            ml_raw = self.mlp.wait_async(h_ml)
        elif self.parallel:
            f_eo = self._pool.submit(self.eo.run,  [eo_inp])
            f_ir = self._pool.submit(self.ir.run,  [ir_inp])
            f_ml = self._pool.submit(self.mlp.run, [feat_n])
            eo_raw = f_eo.result()
            ir_raw = f_ir.result()
            ml_raw = f_ml.result()
        else:
            eo_raw = self.eo.run([eo_inp])
            ir_raw = self.ir.run([ir_inp])
            ml_raw = self.mlp.run([feat_n])

        eo_out = decode_yolo(eo_raw, conf, OUTPUT_SIZE, OUTPUT_SIZE)
        ir_out = decode_yolo(ir_raw, conf, OUTPUT_SIZE, OUTPUT_SIZE)
        fw     = decode_mlp(ml_raw)
        return eo_out, ir_out, fw


# ── Fusion (run.py 흐름) ──────────────────────────────────────────────────────

def fuse_one_frame(eo_out: dict, ir_out: dict, fusion_weight: float,
                   iou_thr: float, max_dets: int) -> tuple:
    """EO/IR 모두 640x640 입력이므로 IR 박스 재스케일링 불필요."""
    f_boxes, scores, labels = fuse_results(
        eo_out, ir_out, fusion_weight, iou_thr=iou_thr
    )
    boxes_out = f_boxes if len(f_boxes) else np.zeros((0, 4), dtype=np.float32)

    if max_dets > 0 and len(boxes_out) > max_dets:
        top_idx   = np.argsort(-scores)[:max_dets]
        boxes_out = boxes_out[top_idx]
        scores    = scores[top_idx]
        labels    = labels[top_idx]

    return boxes_out, scores, labels


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_detections(boxes_640, scores, labels, class_names: dict) -> list:
    return [
        {
            "class_id":   int(lbl),
            "class_name": class_names.get(int(lbl), str(int(lbl))),
            "score":      round(float(sc), 4),
            "bbox":       [round(float(v), 2) for v in bx],
        }
        for bx, sc, lbl in zip(boxes_640, scores, labels)
    ]


def build_packet(jpeg: bytes, frame_seq: int, capture_ns: int,
                 infer_done_ns: int, detections: list,
                 fusion_weight: float) -> bytes:
    header = {
        "type":          "frame",
        "frame_seq":     frame_seq,
        "capture_ns":    capture_ns,
        "infer_done_ns": infer_done_ns,
        "send_ns":       time.monotonic_ns(),
        "fusion_weight": round(fusion_weight, 4),
        "image": {
            "format": "jpeg",
            "w":      OUTPUT_SIZE,
            "h":      OUTPUT_SIZE,
            "size":   len(jpeg),
        },
        "detections":    detections,
        "model_version": MODEL_VERSION,
    }
    hb = json.dumps(header, ensure_ascii=False).encode("utf-8")
    return MAGIC + bytes([VERSION]) + struct.pack(">H", len(hb)) + hb + jpeg


# ── Frame prefetcher (background video reader) ───────────────────────────────

class VideoPrefetcher:
    """cv2.VideoCapture.read()를 백그라운드 스레드로 분리.

    NPU 추론과 디스크/디코딩 I/O를 겹쳐 NPU 대기 시간을 제거합니다.
    EO/IR 쌍을 같은 튜플로 묶어 큐에 넣으므로 짝이 깨지지 않습니다.
    """

    _SENTINEL = None
    _LOOP_MARK = "loop"   # 루프 경계 마커 (prev_eo 초기화용)

    def __init__(self, eo_path: str, ir_path: str, loop: bool, queue_size: int):
        self._eo_path = eo_path
        self._ir_path = ir_path
        self._loop    = loop
        self._q       = queue.Queue(maxsize=queue_size)
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            while not self._q.empty():
                self._q.get_nowait()
        except queue.Empty:
            pass

    def _worker(self):
        cap_eo = cv2.VideoCapture(self._eo_path)
        cap_ir = cv2.VideoCapture(self._ir_path)
        if not cap_eo.isOpened() or not cap_ir.isOpened():
            self._q.put(("error", f"cannot open: eo={self._eo_path} ir={self._ir_path}"))
            self._q.put(self._SENTINEL)
            return

        try:
            while not self._stop.is_set():
                ret_eo, eo_img = cap_eo.read()
                ret_ir, ir_img = cap_ir.read()

                if not ret_eo or not ret_ir:
                    if self._loop:
                        cap_eo.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        cap_ir.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._q.put(("loop_mark",))   # 메인 측에서 prev_eo 초기화
                        continue
                    break

                # 640x640으로 미리 리사이즈 — 추론 스레드 부담 경감
                eo_640 = cv2.resize(eo_img, (OUTPUT_SIZE, OUTPUT_SIZE))
                ir_640 = cv2.resize(ir_img, (OUTPUT_SIZE, OUTPUT_SIZE))
                self._q.put(("frame", eo_640, ir_640, time.monotonic_ns()))
        finally:
            cap_eo.release()
            cap_ir.release()
            self._q.put(self._SENTINEL)

    async def aiter(self):
        """asyncio용 async iterator. 큐 get은 executor로 위임."""
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, self._q.get)
            if item is self._SENTINEL:
                return
            yield item


# ── Background sender ────────────────────────────────────────────────────────

_SENTINEL = b""


async def _sender_task(ws, q: asyncio.Queue):
    while True:
        pkt = await q.get()
        if pkt is _SENTINEL or pkt == _SENTINEL:
            return
        try:
            await ws.send(pkt)
        finally:
            q.task_done()


# ── Encoding helper ──────────────────────────────────────────────────────────

def _encode_jpeg(img: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


# ── Handshake ────────────────────────────────────────────────────────────────

async def _handshake(ws, client_id: str, token: str, fps: float):
    await ws.send(json.dumps({
        "type":           "hello",
        "client_id":      client_id,
        "auth":           token,
        "client_version": "1.0.0",
        "capabilities": {
            "image_format": "jpeg",
            "image_size":   [OUTPUT_SIZE, OUTPUT_SIZE],
            "target_fps":   int(fps),
            "runtime":      "deepx-npu",
        },
    }))
    ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
    if ack.get("type") != "hello_ack":
        raise RuntimeError(f"auth fail: {ack}")
    return ack


# ── Streaming session ────────────────────────────────────────────────────────

async def _stream_session(ws, args, parallel: ParallelInfer,
                          class_names: dict) -> int:
    """한 WebSocket 세션 동안 prefetch → infer → encode → send 파이프라인 실행.

    Lookahead 파이프라이닝:
      args.lookahead=0 → 직렬 (한 프레임 끝나야 다음 시작)
      args.lookahead=1 → 한 프레임 미리 추론 시작 (안전, prev_eo 거의 정확)
      args.lookahead≥2 → 더 깊게 (prev_eo는 지연된 값 사용, temporal feature 살짝 부정확)

    송신 순서는 frame_seq로 정렬되어 항상 정확합니다.
    """
    prefetcher = VideoPrefetcher(args.eo_video, args.ir_video,
                                 args.loop, args.prefetch)
    prefetcher.start()

    send_q = asyncio.Queue(maxsize=args.send_queue)
    sender = asyncio.create_task(_sender_task(ws, send_q))

    frame_idx  = 0
    prev_eo    = None
    last_t     = time.monotonic()
    sent_count = 0
    fps_period = (1.0 / args.fps) if args.fps > 0 else 0.0
    next_t     = time.monotonic()

    loop = asyncio.get_running_loop()

    cpu_pool = ThreadPoolExecutor(max_workers=args.cpu_workers,
                                  thread_name_prefix="frame-worker")

    def _infer_and_encode(eo_640, ir_640, prev_eo_local, frame_seq, capture_ns):
        """추론 → 융합 → 인코딩까지 한 번에 (스레드에서 실행)."""
        eo_out, ir_out, fw = parallel.infer(
            eo_640, ir_640, prev_eo_local, args.conf)
        boxes, scores, labels = fuse_one_frame(
            eo_out, ir_out, fw, args.iou_thr, args.max_dets)
        infer_done_ns = time.monotonic_ns()

        out_img = eo_640 if args.output_modal == "eo" else ir_640
        jpeg    = _encode_jpeg(out_img)
        dets    = _build_detections(boxes, scores, labels, class_names)
        pkt     = build_packet(jpeg, frame_seq, capture_ns,
                               infer_done_ns, dets, fw)
        return frame_seq, pkt, fw, len(boxes)

    # in-flight 추론 결과 큐 (frame_seq, future)
    # lookahead 깊이만큼 동시에 던지고, 순서대로 결과를 받음
    inflight: "list[tuple[int, asyncio.Future]]" = []
    max_inflight = max(1, args.lookahead + 1)  # 최소 1 (직렬)

    try:
        prefetch_iter = prefetcher.aiter().__aiter__()
        stream_done   = False

        while True:
            # ① in-flight가 가득 차지 않았고 아직 prefetch에 프레임이 있으면 새로 submit
            while len(inflight) < max_inflight and not stream_done:
                try:
                    item = await prefetch_iter.__anext__()
                except StopAsyncIteration:
                    stream_done = True
                    break

                kind = item[0]
                if kind == "error":
                    print(f"[FATAL] {item[1]}")
                    stream_done = True
                    break
                if kind == "loop_mark":
                    # 루프 경계: in-flight 다 끝나고 prev_eo 초기화
                    if inflight:
                        # 일단 큐에 있는 것 다 비워야 안전
                        break
                    prev_eo = None
                    print("[i] Looping video...")
                    continue

                _, eo_640, ir_640, capture_ns = item

                fut = loop.run_in_executor(
                    cpu_pool, _infer_and_encode,
                    eo_640, ir_640, prev_eo, frame_idx, capture_ns,
                )
                inflight.append((frame_idx, fut))
                frame_idx += 1
                # 다음 프레임의 prev_eo는 *이* 프레임 (한 프레임 지연됨)
                # lookahead=1이면 거의 무손실, ≥2면 살짝 부정확
                prev_eo = eo_640

            # ② in-flight가 비었고 stream 끝났으면 종료
            if not inflight:
                if stream_done:
                    break
                continue

            # ③ 가장 오래된 in-flight 결과 회수 → 순서대로 송신
            seq, fut = inflight.pop(0)
            try:
                _, pkt, fw, det_n = await fut
            except Exception as e:
                print(f"[!] infer error on frame {seq}: {type(e).__name__}: {e}")
                continue

            if sender.done():
                sender.result()  # 예외 위로
                break

            await send_q.put(pkt)
            sent_count += 1

            now = time.monotonic()
            if now - last_t >= 1.0:
                print(f"[>] frame={seq+1}  {sent_count}/s  fw={fw:.3f}  "
                      f"dets={det_n}  inflight={len(inflight)}  sendq={send_q.qsize()}")
                last_t     = now
                sent_count = 0

            if fps_period > 0:
                next_t += fps_period
                sleep_s = next_t - time.monotonic()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                else:
                    next_t = time.monotonic()

        print("[i] End of stream — finishing up.")
    finally:
        prefetcher.stop()
        # 남은 in-flight 비우기
        for seq, fut in inflight:
            try:
                _, pkt, _, _ = await asyncio.wait_for(fut, timeout=5.0)
                await send_q.put(pkt)
            except Exception:
                pass
        await send_q.put(_SENTINEL)
        try:
            await asyncio.wait_for(sender, timeout=10.0)
        except asyncio.TimeoutError:
            print("[!] sender flush timeout — cancelling")
            sender.cancel()
        except Exception as e:
            print(f"[!] sender error: {type(e).__name__}: {e}")
        cpu_pool.shutdown(wait=True)

    return frame_idx


# ── WebSocket mode (with auto-reconnect) ─────────────────────────────────────

async def run_ws(args, parallel: ParallelInfer, class_names: dict):
    ws_url = args.ws_url.strip()

    ssl_ctx = None
    if ws_url.startswith("wss://"):
        cafile = args.cert if args.cert else None
        if cafile and not os.path.exists(cafile):
            print(f"[FATAL] cert file not found: {cafile}")
            sys.exit(1)
        ssl_ctx = ssl.create_default_context(cafile=cafile)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    backoff = 0.2

    while True:
        try:
            async with websockets.connect(
                ws_url,
                ssl=ssl_ctx,
                max_size=32 * 1024 * 1024,
                open_timeout=10,
            ) as ws:
                ack = await _handshake(ws, args.ws_client_id, args.ws_token, args.fps)
                print(f"[ok] streaming @ {args.fps:.0f} fps  session={ack.get('session_id')}")
                backoff = 0.2

                total = await _stream_session(ws, args, parallel, class_names)

                if not args.loop:
                    print(f"[i] Done. Sent {total} frames.")
                    return

        except Exception as e:
            print(f"[!] connection error: {type(e).__name__}: {e}  retry in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 5.0)


# ── File-save mode ───────────────────────────────────────────────────────────

def run_save(args, parallel: ParallelInfer, class_names: dict):
    img_dir = os.path.join(args.output_dir, "images")
    lbl_dir = os.path.join(args.output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    prefetcher = VideoPrefetcher(args.eo_video, args.ir_video,
                                 loop=False, queue_size=args.prefetch)
    prefetcher.start()

    frame_idx = 0
    prev_eo   = None
    print(f"Saving output to {args.output_dir}")

    try:
        while True:
            item = prefetcher._q.get()
            if item is None:
                break
            kind = item[0]
            if kind == "error":
                print(f"[FATAL] {item[1]}")
                break
            if kind == "loop_mark":
                prev_eo = None
                continue
            _, eo_640, ir_640, _capture_ns = item

            eo_out, ir_out, fw = parallel.infer(
                eo_640, ir_640, prev_eo, args.conf)
            boxes, scores, labels = fuse_one_frame(
                eo_out, ir_out, fw, args.iou_thr, args.max_dets)
            dets = _build_detections(boxes, scores, labels, class_names)
            base = f"frame_{frame_idx:04d}"

            out_img = eo_640 if args.output_modal == "eo" else ir_640
            jpeg = _encode_jpeg(out_img)
            with open(os.path.join(img_dir, f"{base}.jpg"), "wb") as f:
                f.write(jpeg)

            meta = {
                "frame":         frame_idx,
                "fusion_weight": round(fw, 4),
                "image":         {"format": "jpeg", "w": OUTPUT_SIZE, "h": OUTPUT_SIZE},
                "detections":    dets,
                "model_version": MODEL_VERSION,
            }
            with open(os.path.join(lbl_dir, f"{base}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)

            if frame_idx % 50 == 0:
                print(f"frame {frame_idx}  fw={fw:.3f}  dets={len(boxes)}")

            prev_eo    = eo_640
            frame_idx += 1
    finally:
        prefetcher.stop()

    print(f"\nDone. {frame_idx} frames saved to {args.output_dir}")


# ── Entry ────────────────────────────────────────────────────────────────────

def _parse_class_names(arg):
    if not arg:
        return {}
    return {i: n.strip() for i, n in enumerate(arg.split(",")) if n.strip()}


def main():
    _load_env()
    args = parse_args()

    if not DEEPX_AVAILABLE:
        print("[FATAL] dx_engine 모듈을 찾을 수 없습니다.")
        sys.exit(1)

    print("Runtime: DeepX M1 NPU")
    print("Loading models...")
    eo_model  = DXModel(args.eo_model)
    ir_model  = DXModel(args.ir_model)
    mlp_model = DXModel(args.mlp_model)
    print(f"  EO  : {args.eo_model}")
    print(f"  IR  : {args.ir_model}")
    print(f"  MLP : {args.mlp_model}")

    DXModel.report_capabilities(eo_model)

    parallel = ParallelInfer(eo_model, ir_model, mlp_model,
                             parallel=not args.no_parallel)

    class_names = _parse_class_names(args.class_names)
    if class_names:
        print(f"Classes: {class_names}")
    else:
        print("Classes: (이름 미지정 — 숫자 ID 사용)")

    try:
        if args.ws_url:
            print(f"WS    : {args.ws_url}")
            print(f"Client: {args.ws_client_id}  FPS={args.fps}  "
                  f"max_dets={args.max_dets}  loop={args.loop}")
            asyncio.run(run_ws(args, parallel, class_names))
        else:
            run_save(args, parallel, class_names)
    finally:
        parallel.shutdown()


if __name__ == "__main__":
    main()