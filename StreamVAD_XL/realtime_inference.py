import torch
import cv2
import numpy as np
import threading
import time
import sys
from collections import deque
from queue import Queue
from model import StreamVADWithXL
from realtime_input import RealtimeVideoStream
from clip_extractor import CLIPFeatureExtractor


# ─────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────

def draw_overlay_rect(frame, x1, y1, x2, y2, color=(0, 0, 0), alpha=0.5):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_rounded_rect(frame, x1, y1, x2, y2, color, alpha=0.62, radius=6):
    """Semi-transparent panel with rounded corners and a thin border."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # thin white border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)


def draw_score_bar(frame, score, threshold, x, y, width=130, height=10):
    """Slim score bar with threshold marker."""
    # background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    # fill
    fill_w = int(np.clip(score, 0.0, 1.0) * width)
    if score < threshold * 0.75:
        bar_color = (50, 200, 80)    # green
    elif score < threshold:
        bar_color = (30, 190, 240)   # cyan / warning
    else:
        bar_color = (50, 50, 220)    # red / alert
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + height), bar_color, -1)
    # threshold marker
    thresh_x = x + int(threshold * width)
    cv2.line(frame, (thresh_x, y - 3), (thresh_x, y + height + 3),
             (220, 220, 220), 1)
    # outer border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 1)


def put_text_right(frame, text, right_x, y, font, scale, color, thickness=1):
    """Draw text so its right edge aligns with right_x."""
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(frame, text, (right_x - tw, y),
                font, scale, color, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  Inference worker
# ─────────────────────────────────────────────

def inference_worker(model, feature_queue, result_queue, device):
    model.eval()
    memories = None
    with torch.no_grad():
        while True:
            item = feature_queue.get()
            if item is None:
                break
            try:
                clip_feature, enqueue_time = item

                if isinstance(clip_feature, torch.Tensor):
                    input_feature = clip_feature.clone().to(device)
                else:
                    input_feature = torch.from_numpy(clip_feature.copy()).to(device)

                input_feature = input_feature.float()

                if input_feature.dim() == 2:
                    input_feature = input_feature.unsqueeze(0)
                elif input_feature.dim() == 1:
                    input_feature = input_feature.unsqueeze(0).unsqueeze(0)

                scores, memories, _ = model(input_feature, memories)
                scores = torch.sigmoid(scores)
                scores = scores.squeeze().cpu().numpy()

                latency_ms = (time.time() - enqueue_time) * 1000
                result_queue.put((scores, latency_ms))
            except Exception as e:
                print(f"\n[ERROR] Inference thread exception: {e}")
                import traceback
                traceback.print_exc()
                break


# ─────────────────────────────────────────────
#  Session summary
# ─────────────────────────────────────────────

def print_session_summary(score_history, threshold, total_frames,
                          elapsed_sec, all_frame_scores=None):
    print()
    print("=" * 60)
    print("  StreamVAD-XL  Session Summary")
    print("=" * 60)
    if not score_history:
        print("[INFO] No inference results received.")
        print("=" * 60)
        return

    arr = np.array(score_history)
    avg_fps = total_frames / elapsed_sec if elapsed_sec > 0 else 0.0

    print(f"  Duration        : {elapsed_sec:.1f} s")
    print(f"  Frames          : {total_frames}")
    print(f"  Avg FPS         : {avg_fps:.1f}")
    print(f"  Clip max score  : {arr.max():.4f}")
    print(f"  Clip min score  : {arr.min():.4f}")

    if all_frame_scores is not None and len(all_frame_scores) > 0:
        frame_arr = np.array(all_frame_scores)
        k = min(10, len(frame_arr))
        topk_scores = np.sort(frame_arr)[-k:]
        topk_mean = topk_scores.mean()
        print(f"  ──────────────────────────────────────")
        print(f"  Frame total     : {len(frame_arr)}")
        print(f"  Frame Top-{k} mean : {topk_mean:.4f}")
        judge_str = ">" if topk_mean > threshold else "<="
        print(f"  Decision        : Top-{k} mean {judge_str} threshold ({threshold:.2f})")

    print(f"  Threshold       : {threshold:.2f}")
    print()

    if all_frame_scores is not None and len(all_frame_scores) > 0:
        is_anomaly = topk_mean > threshold
    else:
        is_anomaly = arr.max() > threshold

    verdict = "[ANOMALY]" if is_anomaly else "[NORMAL]"
    print(f"  Result          : {verdict}")
    print("=" * 60)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    feature_dim     = 512
    hidden_dim      = 256
    mem_len         = 32
    num_layers      = 2
    nhead           = 4
    chunk_size      = 16
    anomaly_threshold = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")

    WEIGHT_PATH  = "./weights/xl_best.pth"
    VIDEO_SOURCE = r"C:\Users\freddie\OneDrive\桌面\Normal_048.mp4"
    FRAME_INTERVAL = 1

    model = StreamVADWithXL(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        mem_len=mem_len,
        num_layers=num_layers,
        nhead=nhead,
    ).to(device)

    try:
        checkpoint = torch.load(WEIGHT_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        print("[SUCCESS] Model loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    feature_queue = Queue(maxsize=5)
    result_queue  = Queue(maxsize=5)

    inference_thread = threading.Thread(
        target=inference_worker,
        args=(model, feature_queue, result_queue, device),
    )
    inference_thread.start()
    print("[INFO] Inference thread started")

    try:
        video_stream = RealtimeVideoStream(
            source=VIDEO_SOURCE,
            clip_length=chunk_size,
            frame_interval=FRAME_INTERVAL,
        ).start()
        clip_extractor = CLIPFeatureExtractor(device=device)
        print("[SUCCESS] Video stream and feature extractor initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize video stream: {e}")
        sys.exit(1)

    current_score     = 0.0
    current_max_score = 0.0
    current_latency_ms = 0.0
    score_history     = []
    all_frame_scores  = []
    frame_times       = deque(maxlen=30)
    latency_window    = deque(maxlen=10)
    total_frames      = 0
    session_start     = time.time()

    print("=" * 50)
    print("StreamVAD-XL  Real-time Detection")
    print("Press  q  to quit")
    print("=" * 50)

    # ── HUD layout constants ──────────────────
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_DUPLEX = cv2.FONT_HERSHEY_DUPLEX
    FSCALE     = 0.42
    LINE_H     = 17
    PAD        = 10
    PANEL_W    = 268
    PANEL_H    = 126
    PX, PY     = 12, 12          # panel top-left
    LABEL_C    = (150, 180, 200)
    VALUE_C    = (220, 235, 255)
    TITLE_C    = (80, 150, 255)
    DIV_C      = (80, 80, 80)

    while video_stream.running:
        # ── read one clip ─────────────────────
        result = video_stream.read_clip()
        if result is not None:
            pil_images, bgr_frames = result
            clip_feature  = clip_extractor.extract_clip_feature(pil_images)
            enqueue_time  = time.time()
            if not feature_queue.full():
                feature_queue.put((clip_feature, enqueue_time))
            display_frame = bgr_frames[len(bgr_frames) // 2]
        else:
            break

        # ── receive inference result ──────────
        if not result_queue.empty():
            scores, lat = result_queue.get()

            current_score = float(np.max(scores))
            if current_score > current_max_score:
                current_max_score = current_score
            score_history.append(current_score)
            all_frame_scores.extend(scores.tolist())

            latency_window.append(lat)
            current_latency_ms = float(np.mean(latency_window))

            if current_score > anomaly_threshold:
                print(f"  [ANOMALY] Score: {current_score:.4f}  "
                      f"Latency: {current_latency_ms:.1f} ms")

        # ── render frame ──────────────────────
        frame = display_frame.copy()
        total_frames += 1
        now = time.time()
        frame_times.append(now)
        h_frame, w_frame = frame.shape[:2]
        elapsed = now - session_start

        display_fps = (
            (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
            if len(frame_times) >= 2 else 0.0
        )

        # ── accent colour based on current max score ──
        if current_max_score >= anomaly_threshold:
            accent      = (50,  50, 220)
            txt_accent  = (80,  80, 255)
            status_lbl  = "ANOMALY"
        elif current_max_score >= anomaly_threshold * 0.75:
            accent      = (30, 190, 240)
            txt_accent  = (60, 210, 255)
            status_lbl  = "WARNING"
        else:
            accent      = (50, 200,  80)
            txt_accent  = (80, 230, 100)
            status_lbl  = "NORMAL"

        # ── HUD panel ────────────────────────
        draw_rounded_rect(frame, PX, PY, PX + PANEL_W, PY + PANEL_H,
                          color=(10, 10, 10), alpha=0.62, radius=6)

        # title
        title_y = PY + PAD + 8
        cv2.putText(frame, "StreamVAD-XL",
                    (PX + PAD, title_y),
                    FONT, 0.44, TITLE_C, 1, cv2.LINE_AA)
        cv2.line(frame,
                 (PX + PAD, title_y + 5),
                 (PX + PANEL_W - PAD, title_y + 5),
                 (255, 255, 255), 1)

        # stats rows
        stats = [
            ("Elapsed",  f"{elapsed:.1f} s"),
            ("Frames",   f"{total_frames}"),
            ("FPS",      f"{display_fps:.1f}"),
            ("Latency",  f"{current_latency_ms:.1f} ms"),
        ]
        row_y = title_y + LINE_H + 6
        right_x = PX + PANEL_W - PAD
        for label, value in stats:
            cv2.putText(frame, label,
                        (PX + PAD, row_y),
                        FONT, FSCALE, LABEL_C, 1, cv2.LINE_AA)
            put_text_right(frame, value, right_x, row_y,
                           FONT, FSCALE, VALUE_C)
            row_y += LINE_H

        # divider above score row
        cv2.line(frame,
                 (PX + PAD, row_y - 4),
                 (PX + PANEL_W - PAD, row_y - 4),
                 DIV_C, 1)

        # score label
        cv2.putText(frame, "Score",
                    (PX + PAD, row_y + 9),
                    FONT, FSCALE, LABEL_C, 1, cv2.LINE_AA)

        # score bar
        bar_x = PX + PAD + 46
        bar_y = row_y - 1
        draw_score_bar(frame, current_max_score, anomaly_threshold,
                       x=bar_x, y=bar_y, width=128, height=10)

        # score value (right-aligned, coloured)
        put_text_right(frame, f"{current_max_score:.4f}",
                       right_x, row_y + 9,
                       FONT, FSCALE, txt_accent)


        # ── anomaly alert overlay ─────────────
        if current_max_score > anomaly_threshold:
            # red border
            cv2.rectangle(frame, (0, 0),
                          (w_frame - 1, h_frame - 1),
                          (0, 0, 200), 4)
            # bottom banner
            banner_h = 40
            draw_overlay_rect(frame, 0, h_frame - banner_h,
                              w_frame, h_frame,
                              color=(0, 0, 150), alpha=0.75)
            alert_text = "ANOMALY DETECTED"
            (aw, _), _ = cv2.getTextSize(
                alert_text, FONT_DUPLEX, 0.80, 2)
            cv2.putText(frame, alert_text,
                        ((w_frame - aw) // 2, h_frame - 10),
                        FONT_DUPLEX, 0.80, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("StreamVAD-XL  Real-time Detection", frame)

        wait_time = (
            max(1, int(1000 / video_stream.fps))
            if hasattr(video_stream, "fps") and video_stream.fps > 0
            else 1
        )
        if cv2.waitKey(wait_time) & 0xFF == ord("q"):
            break

    # ── cleanup ───────────────────────────────
    feature_queue.put(None)
    inference_thread.join()
    video_stream.stop()

    elapsed_total = time.time() - session_start
    print_session_summary(score_history, anomaly_threshold,
                          total_frames, elapsed_total, all_frame_scores)
    print("[INFO] Program exited normally")


if __name__ == "__main__":
    main()