# -*- coding: utf-8 -*-
"""
Smoke Guard AI - Web phát hiện hút thuốc lá/vape bằng YOLO
===========================================================
2 chế độ:
  1. CAMERA TRỰC TIẾP: dùng camera laptop, hoặc mở web trên ĐIỆN THOẠI
     (cùng WiFi) để dùng camera điện thoại. Trình duyệt gửi từng khung hình
     về server, YOLO phát hiện, vẽ box realtime.
  2. UPLOAD FILE: chọn ảnh/video có sẵn trong máy để phân tích.

Cách chạy:
    pip install -r requirements.txt
    python app.py                ->  http://localhost:5000  (camera laptop)

Dùng camera ĐIỆN THOẠI:
    - Đặt USE_HTTPS=1 trong file .env rồi chạy python app.py
    - Điện thoại cùng WiFi, mở:  https://<IP-laptop>:5000
      (trình duyệt cảnh báo chứng chỉ tự ký -> chọn Advanced -> Proceed)
    - IP laptop xem bằng lệnh:  ipconfig  (dòng IPv4 Address)
"""

import os
import subprocess
import time
import uuid

import cv2
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

# QUAN TRỌNG: import custom blocks TRƯỚC khi load model,
# vì file .pt tham chiếu tới các class trong module này.
import custom_yolo_blocks  # noqa: F401

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # tối đa 200MB

# ----------------------------------------------------------------------------
# Cấu hình
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", os.path.join("model", "EMA_LSKA_SCRATCH_P3P4P5_640.pt"))
if not os.path.isabs(YOLO_MODEL_PATH):
    YOLO_MODEL_PATH = os.path.join(BASE_DIR, YOLO_MODEL_PATH)

USE_HTTPS = os.getenv("USE_HTTPS", "0") == "1"
DEFAULT_CONF = 0.5
TARGET_CLASSES = ["cigarette", "vape"]

RESULT_DIR = os.path.join(BASE_DIR, "static", "results")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

BOX_COLOR = (30, 84, 217)  # BGR - cam "đốm lửa"

# ----------------------------------------------------------------------------
# YOLO model (load 1 lần, dùng lại cho mọi request)
# ----------------------------------------------------------------------------
_yolo = None


def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(
                f"Không tìm thấy file model: {YOLO_MODEL_PATH}. "
                "Hãy đặt file .pt vào thư mục model/."
            )
        _yolo = YOLO(YOLO_MODEL_PATH)
    return _yolo


def predict_boxes(model, frame, conf):
    """Chạy YOLO, trả về list box [{x1,y1,x2,y2,class,conf}] (không vẽ)."""
    results = model.predict(frame, conf=conf, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])]
            if cls_name not in TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "class": cls_name, "conf": round(float(box.conf[0]), 2),
            })
    return boxes


def draw_box(frame, x1, y1, x2, y2, label):
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ty = max(y1 - 8, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 6, ty + 4), BOX_COLOR, -1)
    cv2.putText(frame, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def reencode_h264(src, dst):
    """Chuyển video sang H.264 để phát được trên trình duyệt."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [ffmpeg, "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-an", dst],
            check=True, capture_output=True,
        )
        return True
    except Exception:
        app.logger.exception("Re-encode H.264 thất bại, giữ file gốc")
        return False


def summarize(class_counts, max_conf, extra=""):
    if not class_counts:
        return "Không phát hiện thuốc lá hay vape nào." + extra
    parts = [f"{v} {k}" for k, v in class_counts.items()]
    return f"Phát hiện {', '.join(parts)} (độ tin cậy cao nhất {max_conf:.2f})." + extra


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect_frame", methods=["POST"])
def detect_frame():
    """
    Camera trực tiếp: nhận 1 khung hình JPEG (form-data: frame, conf)
    -> trả JSON {boxes:[...], elapsed_ms} để client tự vẽ lên canvas.
    """
    f = request.files.get("frame")
    if f is None:
        return jsonify({"error": "Thiếu khung hình."}), 400
    try:
        conf = min(max(float(request.form.get("conf", DEFAULT_CONF)), 0.05), 0.95)
    except ValueError:
        conf = DEFAULT_CONF

    try:
        model = get_yolo()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    import numpy as np
    data = np.frombuffer(f.read(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Không đọc được khung hình."}), 400

    t0 = time.time()
    boxes = predict_boxes(model, frame, conf)
    return jsonify({
        "boxes": boxes,
        "elapsed_ms": round((time.time() - t0) * 1000),
    })


@app.route("/api/detect", methods=["POST"])
def detect():
    """Upload file ảnh/video (form-data: file, conf) -> JSON kết quả."""
    f = request.files.get("file")
    if f is None or f.filename == "":
        return jsonify({"error": "Chưa chọn file."}), 400

    try:
        conf = min(max(float(request.form.get("conf", DEFAULT_CONF)), 0.05), 0.95)
    except ValueError:
        conf = DEFAULT_CONF

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in IMAGE_EXTS | VIDEO_EXTS:
        return jsonify({"error": f"Định dạng {ext} chưa hỗ trợ. Dùng ảnh (jpg/png) hoặc video (mp4/avi/mov)."}), 400

    try:
        model = get_yolo()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    uid = uuid.uuid4().hex[:10]
    in_path = os.path.join(UPLOAD_DIR, uid + ext)
    f.save(in_path)
    t0 = time.time()

    try:
        # ---------------- ẢNH ----------------
        if ext in IMAGE_EXTS:
            frame = cv2.imread(in_path)
            if frame is None:
                return jsonify({"error": "Không đọc được ảnh."}), 400
            boxes = predict_boxes(model, frame, conf)
            for b in boxes:
                draw_box(frame, b["x1"], b["y1"], b["x2"], b["y2"], f'{b["class"]} {b["conf"]:.2f}')
            out_name = f"{uid}.jpg"
            cv2.imwrite(os.path.join(RESULT_DIR, out_name), frame)

            counts = {}
            for b in boxes:
                counts[b["class"]] = counts.get(b["class"], 0) + 1
            max_conf = max((b["conf"] for b in boxes), default=0.0)

            return jsonify({
                "type": "image",
                "url": f"/static/results/{out_name}",
                "counts": counts,
                "summary": summarize(counts, max_conf),
                "elapsed": round(time.time() - t0, 2),
            })

        # ---------------- VIDEO ----------------
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            return jsonify({"error": "Không đọc được video."}), 400
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        raw_path = os.path.join(RESULT_DIR, f"{uid}_raw.mp4")
        writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        counts, max_conf, frames_hit, n_frames = {}, 0.0, 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n_frames += 1
            boxes = predict_boxes(model, frame, conf)
            if boxes:
                frames_hit += 1
                for b in boxes:
                    counts[b["class"]] = counts.get(b["class"], 0) + 1
                    max_conf = max(max_conf, b["conf"])
                    draw_box(frame, b["x1"], b["y1"], b["x2"], b["y2"], f'{b["class"]} {b["conf"]:.2f}')
            writer.write(frame)
        cap.release()
        writer.release()

        out_name = f"{uid}.mp4"
        out_path = os.path.join(RESULT_DIR, out_name)
        if not reencode_h264(raw_path, out_path):
            os.replace(raw_path, out_path)
        else:
            os.remove(raw_path)

        extra = f" Có phát hiện trong {frames_hit}/{n_frames} khung hình."
        verdict = "CÓ HÚT THUỐC" if frames_hit > 0 else "KHÔNG PHÁT HIỆN"
        return jsonify({
            "type": "video",
            "url": f"/static/results/{out_name}",
            "counts": counts,
            "frames_hit": frames_hit,
            "total_frames": n_frames,
            "verdict": verdict,
            "summary": summarize(counts, max_conf, extra),
            "elapsed": round(time.time() - t0, 2),
        })

    except Exception as e:  # noqa: BLE001
        app.logger.exception("Lỗi khi xử lý file")
        return jsonify({"error": f"Lỗi xử lý: {e}"}), 500
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)


if __name__ == "__main__":
    ssl = "adhoc" if USE_HTTPS else None
    if USE_HTTPS:
        print("* HTTPS bật (chứng chỉ tự ký). Điện thoại truy cập: https://<IP-laptop>:5000")
        print("* Trình duyệt cảnh báo chứng chỉ -> Advanced -> Proceed/Tiếp tục.")
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=ssl)
