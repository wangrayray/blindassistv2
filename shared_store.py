"""
shared_store.py — 共用層:參數存取 / 持久化 / 分歧記錄
=======================================================
公開介面(對外承諾,改實作不改簽名):
    cfg(name, default)                  # 參數存取(config 沒設就用預設值)
    data_dir() / ensure_dir(path)
    read_json(path, default) / write_json_atomic(path, obj)
    iso(ts) / now_or(ts)
    log_divergence(record) -> str|None  # System1/2 分歧記錄
    iter_divergence(days) -> List[dict]  # 讀回配對資料做實驗統計
不對外開放:log 檔名格式、輪替邏輯、寫入鎖。

--------------------------------------------------------------------
為什麼不是叫 `shared_utils.py`
  既有 `utils.py` 改名後就是 `shared_utils.py`(檔名對照表既定),
  第三代新增的東西如果也叫這個名字會直接撞名。而且兩者職責本來就不同:
    shared_utils.py : 純函式,方位換算/深度取值,無狀態、不碰檔案系統
    shared_store.py : 碰檔案系統,持久化與記錄
  分開是有理由的分工,不是為了避開撞名而硬拆。將來想合併也只是複製貼上。

★ 匯入相容:本檔同時支援改名前後兩種狀態(`shared_utils` 找不到就退回
  `utils`,`shared_config` 找不到就退回 `config`),所以檔名重構可以分批做,
  做到一半系統也不會壞。
"""
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta

# ---- 過渡期轉發:既有 utils.py 全部公開函式 ----------------------------------
# 將來 utils.py → shared_store.py 改名時,把下面 try 區塊刪掉,把 utils.py
# 的內容貼到本檔上方即可,呼叫端(新模組)完全不用改。
_UTIL_NAMES = ("zh", "horizontal_to_motor_7way", "ratio_to_clock", "memory_fresh",
               "hysteretic_motor_7way", "horizontal_to_zone_3way", "zone_to_motor",
               "vibrate_payload", "get_point_distance_mm", "get_box_distance_mm",
               "get_object_distance_mm", "get_finger_distance_mm", "ZONE_MOTOR")


def _load_utils():
    """轉發既有工具函式。改名前後都吃(shared_utils 優先,退回 utils)。"""
    for mod in ("shared_utils", "utils"):
        try:
            m = __import__(mod)
            for n in _UTIL_NAMES:
                globals()[n] = getattr(m, n)
            return True
        except Exception:
            continue
    return False


_UTILS_OK = _load_utils()


def _load_config():
    for mod in ("shared_config", "config"):
        try:
            return getattr(__import__(mod), "CONFIG")
        except Exception:
            continue
    return None


_CONFIG = _load_config()


def cfg(name, default):
    """
    讀 CONFIG 參數,沒設就用預設值。
    沿用 stairs_fusion.py 既有的 `_cfg` 慣例——新參數在正式併入 config 之前,
    新模組先靠預設值跑,不會因為 config 沒更新就 crash。
    """
    return getattr(_CONFIG, name, default) if _CONFIG is not None else default


# ============================================================
# 路徑 / 檔案
# ============================================================
def data_dir():
    """
    第三代持久化資料根目錄(分區/物件記憶/log 都放這)。
    優先序:環境變數 GEN3_DATA_DIR > config 設定 > 本檔同層的 gen3_data/。
    環境變數擺第一,是為了讓測試與清除工具能在不改 config 的情況下換路徑。
    """
    d = os.environ.get("GEN3_DATA_DIR") or cfg(
        "GEN3_DATA_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen3_data"))
    return ensure_dir(d)


def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    return path


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


_WRITE_LOCK = threading.Lock()


def write_json_atomic(path, obj):
    """
    原子寫入:先寫暫存檔再 os.replace。
    理由:zone/物件記憶是「掉電也不能壞」的檔案,直接覆寫遇到斷電會留下半截
    JSON,下次開機整份記憶讀不回來。
    """
    with _WRITE_LOCK:
        try:
            ensure_dir(os.path.dirname(os.path.abspath(path)))
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                                       prefix=".tmp_", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return True
        except Exception as e:
            print(f"⚠️ [shared_store] 寫入失敗 {path}: {e}")
            return False


# ============================================================
# 時間
# ============================================================
def iso(ts=None):
    return datetime.fromtimestamp(ts if ts is not None else time.time()).isoformat(timespec="seconds")


def now_or(ts):
    """統一的「沒給時間就用現在」寫法,全部模組共用同一個慣例。"""
    return time.time() if ts is None else ts


# ============================================================
# System1 / System2 分歧記錄(§5.3 主要場景第三點)
# ============================================================
_DIVERGENCE_SCHEMA = ("ts", "kind", "mode", "system1", "system2", "detail")


def log_divergence(record):
    """
    記錄一筆「偽VLM(System1) 與 真VLM(System2) 判斷不一致」事件。

    record 建議欄位(缺的自動補 None,多的照收):
        kind     : "scene_mismatch" / "memory_gap" / "vlm_timeout" ...
        mode     : daily / search / traffic / vlm
        system1  : 偽VLM 的結論(字串或 dict)
        system2  : 真VLM 的結論(字串或 dict);逾時填 None
        detail   : 任意補充 dict(分數、margin、證據清單)

    ★ 隱私:本函式只寫「類別 + 分數 + 時間戳」,呼叫端不得把原始影像或
      可識別個資塞進 record(隱私政策第九節)。
    回傳寫入的檔案路徑;失敗回 None。
    """
    if not isinstance(record, dict):
        return None
    rec = {k: record.get(k) for k in _DIVERGENCE_SCHEMA}
    rec.update({k: v for k, v in record.items() if k not in _DIVERGENCE_SCHEMA})
    if rec.get("ts") is None:
        rec["ts"] = time.time()
    rec["iso"] = iso(rec["ts"])

    log_dir = ensure_dir(os.path.join(data_dir(), "logs"))
    path = os.path.join(log_dir, f"divergence_{datetime.fromtimestamp(rec['ts']).strftime('%Y%m%d')}.jsonl")
    try:
        with _WRITE_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return path
    except Exception as e:
        print(f"⚠️ [shared_store] 分歧記錄寫入失敗: {e}")
        return None


def iter_divergence(days=7, now=None):
    """讀回最近 N 天的分歧記錄(給實驗統計用,非即時路徑)。"""
    now = time.time() if now is None else now
    out = []
    log_dir = os.path.join(data_dir(), "logs")
    if not os.path.isdir(log_dir):
        return out
    for name in sorted(os.listdir(log_dir)):
        if not name.startswith("divergence_"):
            continue
        try:
            d = datetime.strptime(name[len("divergence_"):-len(".jsonl")], "%Y%m%d")
        except Exception:
            continue
        if datetime.fromtimestamp(now) - d > timedelta(days=days + 1):
            continue
        with open(os.path.join(log_dir, name), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out
