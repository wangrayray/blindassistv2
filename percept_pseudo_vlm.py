"""
percept_pseudo_vlm.py — 偽VLM:規則式場景投票層(感知層)
=========================================================
公開介面(對外承諾):
    infer_scene(detections, mode="daily", now=None) -> SceneVote | None
    suggest_zone_labels(detections, top_k=3) -> List[dict]      # 建置模式
    reset(mode=None)
    audit_table() -> dict                                        # 權重表自我稽核
不對外開放:Tier 權重表、TF-IDF 計算、滑動視窗結構。

--------------------------------------------------------------------
一、兩層信任判斷(與 DEPTH_CONFIDENCE_THRESH 同角色的「原始證據門檻」)
--------------------------------------------------------------------
第一層(環境無關,先過):
    ① 單筆偵測 conf ≥ PSEUDO_VLM_MIN_TRUST      → 濾掉低信心雜訊
    ② 該類別在最近 PSEUDO_VLM_WINDOW_FRAMES 幀中,至少出現
       PSEUDO_VLM_MIN_HITS 幀                    → 濾掉單幀閃爍
    這一層完全不看場景、不看模式,是「這個偵測本身可不可信」。

第二層(環境相依,過關才算):
    admitted 類別 × 場景權重表(1–10 分) × TF-IDF 排他性 × 數量遞減
    → 各場景加權總分 → 雙重門檻決策。

二、雙重門檻(§5.1,沉默優先)
    總分 ≥ PSEUDO_VLM_SCORE_THRESH(15)  且  領先第二名 ≥ PSEUDO_VLM_MARGIN(5)
    兩個都過才 `passed=True`;沒過照樣回傳 SceneVote,但 passed=False,
    呼叫端必須據此講「不確定」而不是硬猜——這是本設計的核心主張之一。

三、權重怎麼來的(報告要寫,不能拍腦袋)
    權重不是憑感覺填的,而是由「排他性」推導,再用 TF-IDF 做同一直覺的
    資料化版本(Heikel & Espinosa-Leal, 2022, J. Imaging 8(8), 209):
        df(c) = 該類別出現在幾個場景表中
        df == 1        → Tier1,9–10 分(近乎專屬,如 toilet→浴室)
        df in {2,3}    → Tier2,5–8 分(數個場景共用,如 sink→廚房/浴室)
        df >= 4        → Tier3,1–3 分(到處都有,如 person/chair)
    `audit_table()` 會自動檢查表中每一格是否落在 df 對應的分數帶,
    不一致就報出來——權重表因此是可稽核的,不是黑箱。

    TF-IDF 排他性因子:idf(c) = log(1 + S/df(c)) / log(1 + S),S=場景數。
    df=1 → 1.00,df=8 → 0.32。與人工 Tier 同向但連續,兩者相乘。

    數量遞減(§5.2):第 N 個同類物件權重 × 0.5^(N-1)。
    理由:三張椅子不代表「三倍像客廳」,邊際證據力遞減。

四、模式差異(daily / search / traffic / vlm / setup)
    投票邏輯共用同一套,差別只在:視窗長度、是否允許對使用者發話。
    - vlm    : 視窗 1 幀(語意敘述模式要「秒答」,不能等 5 幀),可發話
    - daily  : 視窗 5 幀,可發話(僅供記憶落差偵測與 L4/L5 備用線索)
    - search : 可跑,不發話。§5.4 明確排除偽VLM介入 Search 排序
    - traffic: 可跑,不發話。§5.5 明確排除偽VLM介入即時避障
    - setup  : 視窗 10 幀,只產生建議標籤給人工確認(§5.6 投票是建議者,QR 才是權威)
"""
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from shared_store import cfg, now_or


# ============================================================
# 場景中文名
# ============================================================
SCENE_ZH = {
    "living_room": "客廳", "bedroom": "臥室", "kitchen": "廚房",
    "bathroom": "浴室", "entrance": "玄關", "dining_room": "餐廳",
    "study": "書房", "balcony": "陽台",
}

# ============================================================
# 場景 × 物件 權重表(Tier 1–10)
# 分數帶由 df(出現在幾個場景)決定,見檔頭第三節;audit_table() 可稽核。
# 標 ※ 者為自訓練 daily 權重的類別,官方 COCO 沒有。
# ============================================================
SCENE_OBJECT_TIER: Dict[str, Dict[str, int]] = {
    "living_room": {
        "couch": 10, "remote": 9,
        "tv": 8, "potted plant": 5, "teddy bear": 5, "book": 5,
        "clock": 5, "vase": 5, "handbag": 5, "dining table": 5,
        "chair": 2, "cell phone": 2, "person": 1,
    },
    "bedroom": {
        "bed": 10,
        "teddy bear": 6, "book": 5, "clock": 5, "laptop": 5, "backpack": 5,
        "tv": 5,
        "chair": 2, "cell phone": 2, "person": 1,
    },
    "kitchen": {
        "refrigerator": 10, "microwave": 10, "oven": 9, "toaster": 9,
        "kettle": 10,                                    # ※ daily 自訓練類別
        "sink": 7, "bowl": 6, "spoon": 5, "fork": 5, "knife": 5,
        "cup": 5, "bottle": 5, "banana": 5, "apple": 5, "pizza": 5,
        "person": 1,
    },
    "bathroom": {
        "toilet": 10, "toothbrush": 9, "hair drier": 9,
        "sink": 6,
        "person": 1,
    },
    "entrance": {
        "suitcase": 9, "door": 10,                       # ※ door 為 daily 自訓練類別
        "umbrella": 8, "backpack": 6, "handbag": 5, "bicycle": 5,
        "person": 1,
    },
    "dining_room": {
        "wine glass": 9,
        "dining table": 8, "fork": 6, "knife": 6, "spoon": 6, "bowl": 5,
        "cup": 5, "bottle": 5, "vase": 5, "pizza": 5, "banana": 5, "apple": 5,
        "chair": 2, "cell phone": 2, "person": 1,
    },
    "study": {
        "keyboard": 10, "mouse": 9,
        "laptop": 8, "book": 7, "backpack": 5,
        "chair": 2, "cell phone": 2, "person": 1,
    },
    "balcony": {
        "potted plant": 8, "umbrella": 5, "bicycle": 5,
        "chair": 2, "person": 1,
    },
}

# 模式設定檔:只調視窗與是否發話,投票邏輯共用
MODE_PROFILES = {
    "vlm":     {"window": 1,  "min_hits": 1, "announce": True},
    "daily":   {"window": 5,  "min_hits": 3, "announce": True},
    "search":  {"window": 5,  "min_hits": 3, "announce": False},
    "traffic": {"window": 8,  "min_hits": 5, "announce": False},
    "setup":   {"window": 10, "min_hits": 4, "announce": False},
}


# ============================================================
# 結果資料結構
# ============================================================
@dataclass
class SceneVote:
    scene: Optional[str]
    scene_zh: Optional[str]
    score: float
    runner_up: Optional[str]
    runner_up_score: float
    margin: float
    passed: bool                     # 雙重門檻是否通過
    mode: str
    announce_allowed: bool
    evidence: List[tuple] = field(default_factory=list)   # [(cls, contribution), ...]
    all_scores: Dict[str, float] = field(default_factory=dict)
    admitted: List[str] = field(default_factory=list)
    n_frames: int = 0
    ts: float = 0.0

    def to_dict(self):
        return {
            "scene": self.scene, "scene_zh": self.scene_zh,
            "score": round(self.score, 2), "runner_up": self.runner_up,
            "runner_up_score": round(self.runner_up_score, 2),
            "margin": round(self.margin, 2), "passed": self.passed,
            "mode": self.mode, "n_frames": self.n_frames,
            "evidence": [(c, round(v, 2)) for c, v in self.evidence[:8]],
            "ts": self.ts,
        }

    def describe(self):
        """
        語音措辭。沒過雙重門檻 → 誠實說不確定,不硬猜。
        ★ 這裡的措辭刻意與信心階梯 L1–L3 的分區播報不同語氣(§5.3 延伸場景四),
          避免使用者把「猜的房間」誤當成「定位到的分區」。
        """
        if self.scene is None:
            return "目前看不出這是什麼空間"
        if not self.passed:
            return "不確定目前在什麼空間"
        return f"看起來像是{self.scene_zh}"


# ============================================================
# TF-IDF 排他性因子
# ============================================================
def _build_idf():
    S = len(SCENE_OBJECT_TIER)
    df = Counter()
    for table in SCENE_OBJECT_TIER.values():
        for c in table:
            df[c] += 1
    return {c: math.log(1.0 + S / d) / math.log(1.0 + S) for c, d in df.items()}, df, S


_IDF, _DF, _N_SCENES = _build_idf()


def audit_table():
    """
    權重表自我稽核:檢查每一格分數是否落在 df 對應的 Tier 分數帶。
    回傳 {"ok": bool, "violations": [...], "df": {...}}。
    報告可直接引用這個結果證明權重表有一致的推導規則。
    """
    bands = {1: (9, 10), 2: (5, 8), 3: (5, 8)}
    violations = []
    for scene, table in SCENE_OBJECT_TIER.items():
        for c, w in table.items():
            lo, hi = bands.get(_DF[c], (1, 3))
            if not (lo <= w <= hi):
                violations.append({"scene": scene, "cls": c, "weight": w,
                                   "df": _DF[c], "expected": [lo, hi]})
    return {"ok": not violations, "violations": violations,
            "df": dict(_DF), "n_scenes": _N_SCENES}


# ============================================================
# 偽VLM 本體
# ============================================================
def _normalize(detections):
    """
    接受兩種格式,與既有程式碼相容:
      ① ai_worker.infer_detect 的 tuple: (x1, y1, x2, y2, cls, conf)
      ② dict: {"cls"/"label"/"name":..., "conf"/"confidence":...}
    回傳 [(cls_lower, conf), ...]
    """
    out = []
    for d in detections or []:
        try:
            if isinstance(d, dict):
                cls = d.get("cls") or d.get("label") or d.get("name")
                conf = float(d.get("conf", d.get("confidence", 1.0)))
            elif isinstance(d, (list, tuple)) and len(d) >= 6:
                cls, conf = d[4], float(d[5])
            elif isinstance(d, (list, tuple)) and len(d) == 2:
                cls, conf = d[0], float(d[1])
            else:
                continue
            if cls:
                out.append((str(cls).strip().lower(), conf))
        except Exception:
            continue
    return out


class PseudoVLM:
    def __init__(self, mode="daily"):
        self.default_mode = mode
        self._windows = {}                 # mode → deque of frame dicts {cls: (count, maxconf)}

    # ---------- 內部 ----------
    def _profile(self, mode):
        """
        設定來源:MODE_PROFILES 為預設,daily 這一組額外開放 config 覆寫,
        因為它是《引導總覽》§5.1 點名的基準組,實測校準時最常被調。
        其他模式(vlm 秒答、search/traffic 不發話)是設計決策不是可調參數,
        刻意不開放覆寫,避免有人把 vlm 的視窗調成 5 幀而失去「秒答」這個前提。
        """
        p = dict(MODE_PROFILES.get(mode, MODE_PROFILES["daily"]))
        if mode == "daily":
            p["window"] = int(cfg("PSEUDO_VLM_WINDOW_FRAMES", p["window"]))
            p["min_hits"] = int(cfg("PSEUDO_VLM_MIN_HITS", p["min_hits"]))
        return p

    def _window(self, mode, size):
        dq = self._windows.get(mode)
        if dq is None or dq.maxlen != size:
            dq = deque(maxlen=size)
            self._windows[mode] = dq
        return dq

    def _admit(self, dq, min_hits):
        """
        第一層:環境無關的原始證據門檻 + 滑動視窗多幀累積。
        回傳 {cls: count},count 取「有出現的那些幀」的中位數(抗閃爍)。
        """
        hits = Counter()
        counts = {}
        for frame in dq:
            for c, (n, _cf) in frame.items():
                hits[c] += 1
                counts.setdefault(c, []).append(n)
        admitted = {}
        for c, h in hits.items():
            if h >= min_hits:
                arr = sorted(counts[c])
                admitted[c] = arr[len(arr) // 2]
        return admitted

    def _score(self, admitted):
        """第二層:環境相依權重 × TF-IDF × 數量遞減。"""
        scores = {s: 0.0 for s in SCENE_OBJECT_TIER}
        evid = {s: [] for s in SCENE_OBJECT_TIER}
        decay = float(cfg("PSEUDO_VLM_COUNT_DECAY", 0.5))
        cap = int(cfg("PSEUDO_VLM_MAX_COUNT", 4))
        for c, n in admitted.items():
            idf = _IDF.get(c)
            if idf is None:
                continue                                   # 不在任何場景表 → 無資訊,略過
            n = max(1, min(int(n), cap))
            mult = sum(decay ** (i) for i in range(n))      # 1 + 0.5 + 0.25 ...
            for s, table in SCENE_OBJECT_TIER.items():
                w = table.get(c)
                if not w:
                    continue
                v = w * idf * mult
                scores[s] += v
                evid[s].append((c, v))
        return scores, evid

    # ---------- 公開 ----------
    def infer_scene(self, detections, mode=None, now=None) -> Optional[SceneVote]:
        mode = mode or self.default_mode
        now = now_or(now)
        prof = self._profile(mode)
        dq = self._window(mode, prof["window"])

        min_trust = float(cfg("PSEUDO_VLM_MIN_TRUST", 0.45))
        frame = {}
        for cls, conf in _normalize(detections):
            if conf < min_trust:
                continue
            n, mx = frame.get(cls, (0, 0.0))
            frame[cls] = (n + 1, max(mx, conf))
        dq.append(frame)

        admitted = self._admit(dq, prof["min_hits"])
        if not admitted:
            return None                                    # 完全沒證據 → 不猜
        return self._vote_from(admitted, mode, prof, len(dq), now)

    def _vote_from(self, admitted, mode, prof, n_frames, now) -> SceneVote:
        scores, evid = self._score(admitted)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top, top_s = ranked[0]
        second, second_s = (ranked[1] if len(ranked) > 1 else (None, 0.0))
        margin = top_s - second_s

        thr = float(cfg("PSEUDO_VLM_SCORE_THRESH", 15.0))
        mgn = float(cfg("PSEUDO_VLM_MARGIN", 5.0))
        passed = bool(top_s >= thr and margin >= mgn)

        return SceneVote(
            scene=top if top_s > 0 else None,
            scene_zh=SCENE_ZH.get(top) if top_s > 0 else None,
            score=top_s, runner_up=second, runner_up_score=second_s, margin=margin,
            passed=passed, mode=mode, announce_allowed=bool(prof["announce"]),
            evidence=sorted(evid[top], key=lambda kv: -kv[1]),
            all_scores={k: round(v, 2) for k, v in ranked},
            admitted=sorted(admitted), n_frames=n_frames, ts=now,
        )

    def peek(self, mode=None, now=None) -> Optional[SceneVote]:
        """
        用「目前視窗內已累積的證據」算一次投票,但**不塞入新的一幀**。
        建置模式與 debug 面板要看結果時用這個,避免查詢動作本身汙染視窗
        (連續呼叫 infer_scene([]) 會把空幀灌進視窗,把 min_hits 洗掉)。
        """
        mode = mode or self.default_mode
        now = now_or(now)
        prof = self._profile(mode)
        dq = self._windows.get(mode)
        if not dq:
            return None
        admitted = self._admit(dq, prof["min_hits"])
        if not admitted:
            return None
        return self._vote_from(admitted, mode, prof, len(dq), now)

    def suggest_zone_labels(self, detections=None, top_k=3, mode="setup", now=None):
        """
        建置模式用:回傳前 top_k 個候選房型給人工確認(§5.6 投票只是建議者)。
        detections=None → 只讀現有視窗(peek),不新增觀測。
        [{"scene":..., "scene_zh":..., "score":..., "evidence":[...]}, ...]
        """
        vote = (self.peek(mode=mode, now=now) if detections is None
                else self.infer_scene(detections, mode=mode, now=now))
        if vote is None:
            return []
        _, evid = self._score({c: 1 for c in vote.admitted})
        out = []
        for s, sc in list(vote.all_scores.items())[:top_k]:
            out.append({"scene": s, "scene_zh": SCENE_ZH.get(s, s), "score": sc,
                        "evidence": [c for c, _ in
                                     sorted(evid.get(s, []), key=lambda kv: -kv[1])[:5]]})
        return out

    def reset(self, mode=None):
        if mode is None:
            self._windows.clear()
        else:
            self._windows.pop(mode, None)


# ============================================================
# 模組層單例(符合規格表的 infer_scene(detections) 簽名)
# ============================================================
_singleton = PseudoVLM()


def infer_scene(detections, mode="daily", now=None):
    return _singleton.infer_scene(detections, mode=mode, now=now)


def suggest_zone_labels(detections, top_k=3, now=None):
    return _singleton.suggest_zone_labels(detections, top_k=top_k, now=now)


def reset(mode=None):
    _singleton.reset(mode)


def instance():
    return _singleton


# ============================================================
# 偽VLM vs 真VLM 配對記錄(核心驗證實驗用)
# ============================================================
def make_pair_record(vote: Optional[SceneVote], real_vlm_text=None,
                     real_vlm_latency=None, matched=None):
    """
    產生一筆「System1(偽VLM) vs System2(真VLM)」配對資料,交給
    shared_store.log_divergence 落檔。部署邏輯採方案B:每次 VLM 模式觸發
    都先由偽VLM秒答、真VLM跑完再補充,因此每次觸發都會產生一筆配對。

    matched=None 表示尚未人工標註;實驗時再離線標。
    """
    return {
        "kind": "pseudo_vs_real",
        "mode": vote.mode if vote else None,
        "system1": vote.to_dict() if vote else None,
        "system2": {"text": real_vlm_text, "latency_s": real_vlm_latency},
        "detail": {"matched": matched},
    }


# ============================================================
# 自我檢查
# ============================================================
if __name__ == "__main__":
    a = audit_table()
    print("權重表稽核:", "✅ 一致" if a["ok"] else f"❌ {a['violations']}")
    kitchen = [(0, 0, 1, 1, "refrigerator", 0.9), (0, 0, 1, 1, "microwave", 0.8),
               (0, 0, 1, 1, "sink", 0.7), (0, 0, 1, 1, "person", 0.9)]
    p = PseudoVLM()
    for _ in range(5):
        v = p.infer_scene(kitchen, mode="daily")
    print(v.to_dict())
    print("措辭:", v.describe())
