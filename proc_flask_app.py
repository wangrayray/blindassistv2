"""Flask routes + state"""
import time
import queue

from flask import Flask, render_template, request, jsonify

from shared_utils import zh
from shared_config import SIG_RETURN_IDLE, CONFIG


class State:
    speech_q = queue.Queue(maxsize=60)   # 加大:長描述會切多句,避免佇列塞爆丟句
    ai_cmd_q = None
    vlm_in_q = None
    cam = None
    gen3_status = None       # 第三代管線最近一次 step() 的結果 (分區/信心等級)
    cam_height_mm = None     # UI 目前顯示的配戴高度 (手動基準)

    current_mode = "idle"
    target_object = None
    vlm_busy = False
    vlm_busy_since = 0.0     # busy 開始時間,供 timeout 自動解鎖

    _last_signal = {"text": None, "time": 0.0}


def signal_speech(text):
    try:
        now = time.time()
        if (State._last_signal["text"] == text
                and now - State._last_signal["time"] < 1.0):
            return
        State._last_signal = {"text": text, "time": now}
        if State.speech_q.full():
            try: State.speech_q.get_nowait()
            except queue.Empty: pass
        State.speech_q.put_nowait(text)
        print(f"📡 語音: {text}")
    except Exception:
        pass


def return_to_idle(reason=""):
    State.current_mode = "idle"
    State.target_object = None
    if State.ai_cmd_q is not None:
        State.ai_cmd_q.put({"mode": "idle", "target": None,
                            "reset_timers": True})
    if reason:
        print(f"↩️ 返回 idle: {reason}")


def handle_speech_signal(txt):
    if txt == SIG_RETURN_IDLE:
        return_to_idle("AI worker 要求")
        return True
    return False


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("ui_v9.html")


@app.route("/api/status")
def api_status():
    # VLM busy 卡死防護:超過 timeout 沒收到 final → 自動解鎖
    if State.vlm_busy and State.vlm_busy_since > 0:
        if time.time() - State.vlm_busy_since > CONFIG.VLM_BUSY_TIMEOUT:
            State.vlm_busy = False
            State.vlm_busy_since = 0.0
            print("⚠️ VLM busy 逾時,自動解鎖")
    try:
        msg = State.speech_q.get_nowait()
    except queue.Empty:
        msg = ""
    return jsonify({"speech": msg})


@app.route("/api/mode", methods=["POST"])
def api_mode():
    d = request.json or {}
    new_mode = d.get("mode", "idle")
    if new_mode == "cancel":
        new_mode = "idle"

    State.current_mode = new_mode
    State.target_object = d.get("target")

    if State.ai_cmd_q is not None:
        State.ai_cmd_q.put({"mode": State.current_mode,
                            "target": State.target_object,
                            "reset_timers": True})

    if State.current_mode == "search":
        signal_speech(f"開始尋找{zh(State.target_object)}")
    elif State.current_mode == "idle":
        signal_speech("系統待機中")
    elif State.current_mode == "daily":
        signal_speech("日常警報模式")
    elif State.current_mode == "traffic":
        signal_speech("紅綠燈模式,請尋找號誌")

    return jsonify({"status": "ok"})


@app.route("/api/height", methods=["GET", "POST"])
def api_height():
    """
    配戴高度(= 相機離地高度)。
    POST {"height_cm": 175} → 設為手動基準,自動估計之後只做比對警告。
    ★ 這裡刻意用「公分」對 UI、「公釐」對內部:UI 給人看,內部算幾何。
    """
    if request.method == "GET":
        return jsonify({"height_mm": State.cam_height_mm,
                        "source": "manual" if State.cam_height_mm else "auto"})
    d = request.json or {}
    try:
        cm = float(d.get("height_cm", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid height"}), 400
    mm = cm * 10.0
    if not (120.0 <= cm <= 210.0):
        return jsonify({"error": "height out of range (120-210cm)"}), 400
    State.cam_height_mm = mm
    if State.ai_cmd_q is not None:
        State.ai_cmd_q.put({"cam_height_mm": mm})
    signal_speech(f"配戴高度設定為{int(cm)}公分")
    return jsonify({"status": "ok", "height_mm": mm})


@app.route("/api/height/reset", methods=["POST"])
def api_height_reset():
    """換人配戴:清掉手動基準,交還給自動估計。"""
    State.cam_height_mm = None
    if State.ai_cmd_q is not None:
        State.ai_cmd_q.put({"reset_height": True})
    signal_speech("配戴高度已重設,系統會自動估計")
    return jsonify({"status": "ok"})


@app.route("/api/zone")
def api_zone():
    """
    第三代分區狀態(給照顧者後台顯示)。
    ★ 隱私:只回傳分區名稱與信心等級,不回傳物件清單、不回傳座標,
      也不回傳偽VLM猜的房型——猜錯會造成信任錯位(§5.4 明確排除)。
    """
    g = State.gen3_status
    if not g:
        return jsonify({"enabled": False})
    zone = g.get("zone") or {}
    return jsonify({
        "enabled": True,
        "level": g.get("level"),
        "zone_name": zone.get("display_name"),
        "assumed": zone.get("assumed", False),
    })


@app.route("/api/vlm", methods=["POST"])
def api_vlm():
    d = request.json or {}
    sub_mode = d.get("sub_mode", "describe")

    if State.cam is None:
        return jsonify({"error": "camera not ready"}), 503

    # busy 先擋掉,避免 OCR still 擷取 (可能阻塞快 1 秒) 做白工
    if State.vlm_busy:
        signal_speech("正在分析,請稍候")
        return jsonify({"status": "busy"})

    if sub_mode == "ocr":
        # OCR 讀小字用「兩顆鏡頭」的照片一起送:
        #   - OAK 1920x1080 still:向下俯角,近距離桌面/手持文件清楚
        #   - 廣角 (上面那顆):水平視角,牆上站牌/招牌/菜單清楚
        # 兩張互補,雲端 Gemini 吃多圖能挑到讀得清楚的那張。
        # (本地 VLM 只用第一張,見 vlm_worker;多圖對本地模型太慢/太弱)
        oak_still = State.cam.capture_still()
        if oak_still is None:
            # webcam fallback 或 still 逾時 → 退回 OAK 預覽幀
            ret, oak_still, _ = State.cam.read()
            if not ret or oak_still is None:
                oak_still = None
        wide_frame = None
        wc = getattr(State, "wide_cam", None)
        if wc is not None and getattr(wc, "available", False):
            latest = wc.get_latest()
            if latest is not None:
                _, wide_frame = latest
        frames = [f for f in (oak_still, wide_frame) if f is not None]
        if not frames:
            return jsonify({"error": "no frame"}), 503
        frame = frames[0]           # 本地 VLM / 相容欄位:用第一張 (OAK 優先)
        extra_frames = frames[1:]   # 雲端多圖用:第二張以後
    else:
        # 環境語意描述:兩顆鏡頭都送,雲端多圖綜合判讀。
        #   主圖 = 廣角 (上面那顆,水平視角,較接近人眼看到的整體場景)
        #   輔圖 = OAK 預覽幀 (俯角,貼近地面/近處,幫忙判斷「有沒有障礙」)
        # OCR 用 OAK still (慢、清楚);describe 要即時反應,兩張都用預覽幀,不擷取 still。
        # (本地 VLM 只用第一張,見 vlm_worker —— 一樣是廣角優先)
        frame = None
        wc = getattr(State, "wide_cam", None)
        if wc is not None and getattr(wc, "available", False):
            latest = wc.get_latest()
            if latest is not None:
                _, frame = latest

        ret, oak_frame, _ = State.cam.read()
        oak_frame = oak_frame if ret else None

        if frame is None:
            # 廣角不可用 → OAK 預覽幀頂上當主圖,沒有輔圖
            frame = oak_frame
            oak_frame = None
        if frame is None:
            return jsonify({"error": "no frame"}), 503
        extra_frames = [oak_frame] if oak_frame is not None else []

    signal_speech("正在分析畫面")
    State.vlm_busy = True
    State.vlm_busy_since = time.time()

    if State.vlm_in_q is not None:
        while not State.vlm_in_q.empty():
            try: State.vlm_in_q.get_nowait()
            except queue.Empty: break
        try:
            State.vlm_in_q.put_nowait({"frame": frame,
                                       "extra_frames": extra_frames,
                                       "sub_mode": sub_mode})
        except queue.Full:
            State.vlm_busy = False
            return jsonify({"error": "vlm queue full"}), 503

    return jsonify({"status": "processing", "sub_mode": sub_mode})