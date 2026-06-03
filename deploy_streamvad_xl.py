"""
deploy_streamvad_xl.py
======================
StreamVAD-XL 一键部署脚本
用法：python deploy_streamvad_xl.py
"""

import sys
import subprocess
import shutil
import platform
import textwrap
from pathlib import Path

# ─────────────────────────────────────────────────────────────
#  终端颜色
# ─────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"

def ok(msg):   print(f"  {C.GREEN}✔  {msg}{C.RESET}")
def warn(msg): print(f"  {C.YELLOW}⚠  {msg}{C.RESET}")
def err(msg):  print(f"  {C.RED}✘  {msg}{C.RESET}")
def step(n, total, title):
    print(f"\n{C.BOLD}{C.BLUE}[{n}/{total}] {title}{C.RESET}")
    print(f"  {'─' * 54}")

def banner():
    print(f"""
{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════╗
║          StreamVAD-XL  一键部署脚本                  ║
║    Video Anomaly Detection · Real-time Inference     ║
╚══════════════════════════════════════════════════════╝{C.RESET}
""")

# ─────────────────────────────────────────────────────────────
#  辅助函数
# ─────────────────────────────────────────────────────────────
def run(cmd, cwd=None, capture=False):
    kwargs = dict(cwd=cwd, text=True)
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        stderr = getattr(result, "stderr", "")
        raise RuntimeError(f"命令失败 (code {result.returncode}): {' '.join(cmd)}\n{stderr}")
    return result

def pip_install(packages: list, extra_args: list = None):
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    if extra_args:
        cmd += extra_args
    run(cmd)

# ─────────────────────────────────────────────────────────────
#  各步骤实现
# ─────────────────────────────────────────────────────────────
TOTAL_STEPS = 7

# ── [1/7] Python 版本检查 ──────────────────────────────────
def check_python():
    step(1, TOTAL_STEPS, "检查 Python 版本")
    major, minor, micro = sys.version_info[:3]
    ver_str = f"{major}.{minor}.{micro}"
    MIN, MAX = (3, 8, 0), (3, 10, 12)
    if (major, minor, micro) < MIN:
        err(f"版本过低，最低要求 Python {'.'.join(map(str, MIN))}")
        sys.exit(1)
    if (major, minor, micro) > MAX:
        warn(f"Python {ver_str} 超出测试范围（推荐 3.8–3.10），继续部署…")
    else:
        ok(f"Python {ver_str}")

# ── [2/7] Git 检查 ────────────────────────────────────────
def check_git():
    step(2, TOTAL_STEPS, "检查 Git 安装")
    if shutil.which("git") is None:
        err("未检测到 git，请先安装：https://git-scm.com/downloads")
        sys.exit(1)
    result = run(["git", "--version"], capture=True)
    ok(result.stdout.strip())

# ── [3/7] 创建项目目录 ────────────────────────────────────
PROJECT_DIR = Path.cwd() / "StreamVAD_XL"

def create_project_dir():
    step(3, TOTAL_STEPS, "创建 StreamVAD_XL 项目目录")
    if PROJECT_DIR.exists():
        warn(f"目录已存在，跳过：{PROJECT_DIR}")
    else:
        PROJECT_DIR.mkdir(parents=True)
        ok(f"已创建：{PROJECT_DIR}")
    for d in ["weights", "features", "data"]:
        (PROJECT_DIR / d).mkdir(exist_ok=True)
    ok("子目录已就绪：weights / features / data")

# ── [4/7] 安装依赖 ────────────────────────────────────────
def install_dependencies():
    step(4, TOTAL_STEPS, "安装全部依赖")

    pip_install(["pip", "setuptools", "wheel"])

    try:
        import torch
        ok(f"PyTorch 已安装（CUDA={torch.cuda.is_available()}），跳过")
    except ImportError:
        system = platform.system().lower()
        if system == "darwin":
            pip_install(["torch", "torchvision", "torchaudio"])
        else:
            pip_install(
                ["torch", "torchvision", "torchaudio"],
                extra_args=["--index-url", "https://download.pytorch.org/whl/cu121"],
            )
        ok("PyTorch 安装完成")

    pip_install(["git+https://github.com/openai/CLIP.git"])
    ok("CLIP 安装完成")

    pip_install(["numpy", "pandas", "scikit-learn",
                 "opencv-python", "Pillow", "tqdm", "ftfy", "regex"])
    ok("其余依赖安装完成")

# ── [5/7] 生成实时推理文件 ────────────────────────────────
REALTIME_INFERENCE_CODE = textwrap.dedent('''\
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


    def draw_overlay_rect(frame, x1, y1, x2, y2, color=(0, 0, 0), alpha=0.5):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


    def draw_rounded_rect(frame, x1, y1, x2, y2, color, alpha=0.62, radius=6):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [
            (x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
            (x1 + radius, y2 - radius), (x2 - radius, y2 - radius),
        ]:
            cv2.circle(overlay, (cx, cy), radius, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)


    def draw_score_bar(frame, score, threshold, x, y, width=130, height=10):
        cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
        fill_w = int(np.clip(score, 0.0, 1.0) * width)
        if score < threshold * 0.75:
            bar_color = (50, 200, 80)
        elif score < threshold:
            bar_color = (30, 190, 240)
        else:
            bar_color = (50, 50, 220)
        if fill_w > 0:
            cv2.rectangle(frame, (x, y), (x + fill_w, y + height), bar_color, -1)
        thresh_x = x + int(threshold * width)
        cv2.line(frame, (thresh_x, y - 3), (thresh_x, y + height + 3), (220, 220, 220), 1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 1)


    def put_text_right(frame, text, right_x, y, font, scale, color, thickness=1):
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.putText(frame, text, (right_x - tw, y), font, scale, color, thickness, cv2.LINE_AA)


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
                    import traceback
                    traceback.print_exc()
                    break


    def print_session_summary(score_history, threshold, total_frames,
                              elapsed_sec, all_frame_scores=None):
        if not score_history:
            return
        arr = np.array(score_history)
        avg_fps = total_frames / elapsed_sec if elapsed_sec > 0 else 0.0
        print()
        print("=" * 60)
        print("  StreamVAD-XL  Session Summary")
        print("=" * 60)
        print(f"  Duration   : {elapsed_sec:.1f} s   |   Avg FPS : {avg_fps:.1f}")
        print(f"  Score max  : {arr.max():.4f}   |   Score min : {arr.min():.4f}")
        if all_frame_scores is not None and len(all_frame_scores) > 0:
            frame_arr = np.array(all_frame_scores)
            k = min(10, len(frame_arr))
            topk_mean = np.sort(frame_arr)[-k:].mean()
            verdict = "ANOMALY" if topk_mean > threshold else "NORMAL"
            print(f"  Top-{k} mean : {topk_mean:.4f}   |   Result : {verdict}")
        print("=" * 60)


    def main():
        feature_dim       = 512
        hidden_dim        = 256
        mem_len           = 32
        num_layers        = 2
        nhead             = 4
        chunk_size        = 16
        anomaly_threshold = 0.5
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        WEIGHT_PATH  = "./weights/xl_best.pth"
        VIDEO_SOURCE = 0          # 0 = 默认摄像头，也可改为视频文件路径
        FRAME_INTERVAL = 1

        model = StreamVADWithXL(
            feature_dim=feature_dim, hidden_dim=hidden_dim,
            mem_len=mem_len, num_layers=num_layers, nhead=nhead,
        ).to(device)

        try:
            model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
        except Exception as e:
            print(f"[ERROR] 加载模型失败：{e}")
            sys.exit(1)

        feature_queue = Queue(maxsize=5)
        result_queue  = Queue(maxsize=5)
        inference_thread = threading.Thread(
            target=inference_worker,
            args=(model, feature_queue, result_queue, device),
        )
        inference_thread.start()

        try:
            video_stream   = RealtimeVideoStream(
                source=VIDEO_SOURCE, clip_length=chunk_size,
                frame_interval=FRAME_INTERVAL,
            ).start()
            clip_extractor = CLIPFeatureExtractor(device=device)
        except Exception as e:
            print(f"[ERROR] 初始化失败：{e}")
            sys.exit(1)

        print(f"StreamVAD-XL  |  device={device}  |  press q to quit")

        current_score      = 0.0
        current_max_score  = 0.0
        current_latency_ms = 0.0
        score_history      = []
        all_frame_scores   = []
        frame_times        = deque(maxlen=30)
        latency_window     = deque(maxlen=10)
        total_frames       = 0
        session_start      = time.time()

        FONT        = cv2.FONT_HERSHEY_SIMPLEX
        FONT_DUPLEX = cv2.FONT_HERSHEY_DUPLEX
        FSCALE      = 0.42
        LINE_H      = 17
        PAD         = 10
        PANEL_W     = 268
        PANEL_H     = 126
        PX, PY      = 12, 12
        LABEL_C     = (150, 180, 200)
        VALUE_C     = (220, 235, 255)
        TITLE_C     = (80, 150, 255)
        DIV_C       = (80, 80, 80)

        while video_stream.running:
            result = video_stream.read_clip()
            if result is not None:
                pil_images, bgr_frames = result
                clip_feature = clip_extractor.extract_clip_feature(pil_images)
                enqueue_time = time.time()
                if not feature_queue.full():
                    feature_queue.put((clip_feature, enqueue_time))
                display_frame = bgr_frames[len(bgr_frames) // 2]
            else:
                break

            if not result_queue.empty():
                scores, lat = result_queue.get()
                current_score = float(np.max(scores))
                if current_score > current_max_score:
                    current_max_score = current_score
                score_history.append(current_score)
                all_frame_scores.extend(scores.tolist())
                latency_window.append(lat)
                current_latency_ms = float(np.mean(latency_window))

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

            if current_max_score >= anomaly_threshold:
                txt_accent = (80, 80, 255)
            elif current_max_score >= anomaly_threshold * 0.75:
                txt_accent = (60, 210, 255)
            else:
                txt_accent = (80, 230, 100)

            draw_rounded_rect(frame, PX, PY, PX + PANEL_W, PY + PANEL_H,
                              color=(10, 10, 10), alpha=0.62, radius=6)
            title_y = PY + PAD + 8
            cv2.putText(frame, "StreamVAD-XL", (PX + PAD, title_y),
                        FONT, 0.44, TITLE_C, 1, cv2.LINE_AA)
            cv2.line(frame, (PX + PAD, title_y + 5),
                     (PX + PANEL_W - PAD, title_y + 5), (255, 255, 255), 1)

            stats = [
                ("Elapsed", f"{elapsed:.1f} s"),
                ("Frames",  f"{total_frames}"),
                ("FPS",     f"{display_fps:.1f}"),
                ("Latency", f"{current_latency_ms:.1f} ms"),
            ]
            row_y   = title_y + LINE_H + 6
            right_x = PX + PANEL_W - PAD
            for label, value in stats:
                cv2.putText(frame, label, (PX + PAD, row_y),
                            FONT, FSCALE, LABEL_C, 1, cv2.LINE_AA)
                put_text_right(frame, value, right_x, row_y, FONT, FSCALE, VALUE_C)
                row_y += LINE_H

            cv2.line(frame, (PX + PAD, row_y - 4),
                     (PX + PANEL_W - PAD, row_y - 4), DIV_C, 1)
            cv2.putText(frame, "Score", (PX + PAD, row_y + 9),
                        FONT, FSCALE, LABEL_C, 1, cv2.LINE_AA)
            draw_score_bar(frame, current_max_score, anomaly_threshold,
                           x=PX + PAD + 46, y=row_y - 1, width=128, height=10)
            put_text_right(frame, f"{current_max_score:.4f}", right_x, row_y + 9,
                           FONT, FSCALE, txt_accent)

            if current_max_score > anomaly_threshold:
                cv2.rectangle(frame, (0, 0), (w_frame - 1, h_frame - 1), (0, 0, 200), 4)
                banner_h = 40
                draw_overlay_rect(frame, 0, h_frame - banner_h, w_frame, h_frame,
                                  color=(0, 0, 150), alpha=0.75)
                alert_text = "ANOMALY DETECTED"
                (aw, _), _ = cv2.getTextSize(alert_text, FONT_DUPLEX, 0.80, 2)
                cv2.putText(frame, alert_text,
                            ((w_frame - aw) // 2, h_frame - 10),
                            FONT_DUPLEX, 0.80, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("StreamVAD-XL  Real-time Detection", frame)
            wait_time = (
                max(1, int(1000 / video_stream.fps))
                if hasattr(video_stream, "fps") and video_stream.fps > 0 else 1
            )
            if cv2.waitKey(wait_time) & 0xFF == ord("q"):
                break

        feature_queue.put(None)
        inference_thread.join()
        video_stream.stop()
        print_session_summary(score_history, anomaly_threshold,
                              total_frames, time.time() - session_start, all_frame_scores)


    if __name__ == "__main__":
        main()
''')

REALTIME_INPUT_CODE = textwrap.dedent('''\
    import cv2
    from PIL import Image


    class RealtimeVideoStream:
        def __init__(self, source=0, clip_length=16, frame_interval=1):
            self.source = source
            self.clip_length = clip_length
            self.frame_interval = frame_interval
            self.cap = cv2.VideoCapture(source)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_buffer   = []
            self.display_buffer = []
            self.running = False

        def start(self):
            self.running = True
            return self

        def read_clip(self):
            self.frame_buffer.clear()
            self.display_buffer.clear()
            while len(self.frame_buffer) < self.clip_length and self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False
                    break
                self.display_buffer.append(frame.copy())
                self.frame_buffer.append(
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                )
            if len(self.frame_buffer) == self.clip_length:
                return self.frame_buffer, self.display_buffer
            return None

        def stop(self):
            self.running = False
            self.cap.release()
            cv2.destroyAllWindows()
''')

CLIP_EXTRACTOR_CODE = textwrap.dedent('''\
    import clip
    import torch
    import torch.nn.functional as F


    class CLIPFeatureExtractor:
        def __init__(self, model_name="ViT-B/16", device="cuda"):
            self.device = device
            self.model, self.preprocess = clip.load(model_name, device=device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        @torch.no_grad()
        def extract_clip_feature(self, pil_images):
            features = []
            for pil_img in pil_images:
                image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                feat = self.model.encode_image(image_input).float()
                feat = F.normalize(feat, p=2, dim=-1)
                features.append(feat.squeeze(0))
            return torch.stack(features, dim=0).unsqueeze(0)  # (1, T, 512)
''')


def generate_inference_files():
    step(5, TOTAL_STEPS, "生成实时推理文件")
    files = {
        "realtime_inference.py": REALTIME_INFERENCE_CODE,
        "realtime_input.py":     REALTIME_INPUT_CODE,
        "clip_extractor.py":     CLIP_EXTRACTOR_CODE,
    }
    for filename, code in files.items():
        dest = PROJECT_DIR / filename
        if dest.exists():
            warn(f"{filename} 已存在，跳过")
        else:
            dest.write_text(code, encoding="utf-8")
            ok(f"已生成 {filename}")

# ── [6/7] 预下载 CLIP ViT-B/16 ───────────────────────────
def predownload_clip():
    step(6, TOTAL_STEPS, "预下载 CLIP ViT-B/16（约 335 MB）")
    try:
        import clip as _clip
        import torch as _torch
    except ImportError:
        err("clip / torch 未能正确安装，请检查上一步输出")
        sys.exit(1)
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    try:
        _clip.load("ViT-B/16", device=device)
        ok("CLIP ViT-B/16 已就绪")
    except Exception as e:
        err(f"下载失败：{e}")
        warn("可手动运行：python -c \"import clip; clip.load('ViT-B/16')\"")

# ── [7/7] 完成提示 ────────────────────────────────────────
def print_final_instructions():
    step(7, TOTAL_STEPS, "部署完成")
    proj = PROJECT_DIR
    print(f"""
{C.GREEN}{'═' * 60}
  StreamVAD-XL 部署成功！
{'═' * 60}{C.RESET}

  {C.YELLOW}下一步：放置文件{C.RESET}

  1. 将 model.py、dataset.py 复制到：
     {C.CYAN}{proj}{C.RESET}

  2. 将权重文件放入：
     {C.CYAN}{proj / 'weights' / 'xl_best.pth'}{C.RESET}

  {C.YELLOW}运行检测{C.RESET}

     cd {proj}
     python realtime_inference.py

  修改视频源：编辑 realtime_inference.py 中的 VIDEO_SOURCE
  按 {C.BOLD}q{C.RESET} 退出检测窗口。
{C.GREEN}{'═' * 60}{C.RESET}
""")

# ─────────────────────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────────────────────
def main():
    banner()
    try:
        check_python()
        check_git()
        create_project_dir()
        install_dependencies()
        generate_inference_files()
        predownload_clip()
        print_final_instructions()
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}[中断] 用户取消部署。{C.RESET}")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as exc:
        import traceback
        err(f"意外错误：{exc}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
