"""
memory_object.py — 物件記憶(3D 座標版 + 記憶衰減)(狀態層)
============================================================
公開介面(對外承諾):
    register_detection(cls, local_xyz, zone_id, conf, confidence_level, ...) -> dict|None
    get_nearest(cls, current_zone=None, current_xyz=None, now=None, limit=3) -> List[dict]
    decay_tick(now=None) -> List[dict]
    purge_expired(retention_days=30, now=None) -> int
不對外開放:實例 list、去重邏輯、排序鍵、檔案格式。

--------------------------------------------------------------------
與既有 `ai_worker.object_memory` 的關係(§4.4)
  既有版本:`object_memory[cls] = {ts, clock, zone, conf}`,一類一筆、只有
  鐘向、TTL 60 秒,是「即時世界模型」。
  本檔:一類多筆(上限 5)、帶 zone_id + 3D 局部座標 + 信心等級 + 衰減,
  是「持久空間記憶」。兩者不衝突也不互相取代:
    ai_worker.object_memory → Search 模式的即時線索(維持不動,向後相容)
    memory_object           → 跨時間的分區記憶,供「這東西上次在哪」查詢
  未來三合一投票融合(既有討論已定案方向、尚未實作)以本檔為承接點。

★ 只在 L1/L2 寫入(§4.3):L3 只知道分區、不知道精確位置,寫進去的座標是假的;
  L4/L5 連分區都是假設。寧可少記,不可記錯——記錯會讓使用者伸手撲空,
  信任一次就沒了。

★ 記憶衰減(§4.7):可信度 = 0.5^(天數 / 半衰期),低於 40% 措辭改為
  「可能已被移動」。半衰期預設 7 天 → 約第 9.7 天跌破 40%。
"""
import math
import os
import uuid
from typing import List

import numpy as np

from shared_store import cfg, data_dir, now_or, read_json, write_json_atomic

# 中文標籤:轉發自既有 utils/shared_utils;兩者都沒有就原樣輸出英文類別名
try:
    from shared_store import zh as _zh
except ImportError:
    def _zh(x):
        return x


class ObjectMemory:
    def __init__(self, path=None, autoload=True):
        self.path = path or os.path.join(data_dir(), "object_memory.json")
        self.items = {}                 # cls → [instance dict, ...]
        self._below_threshold = set()   # 已經播報過「可能已被移動」的 instance_id
        if autoload:
            self.load()

    # ============================================================
    # 持久化
    # ============================================================
    def load(self):
        raw = read_json(self.path, default={}) or {}
        self.items = {k: list(v) for k, v in raw.items() if isinstance(v, list)}
        return self.items

    def save(self):
        return write_json_atomic(self.path, self.items)

    # ============================================================
    # 寫入
    # ============================================================
    def register_detection(self, cls, local_xyz, zone_id, conf,
                           confidence_level, coord_quality="tag_pose", now=None):
        """
        登記一次物件觀測。回傳寫入/更新後的 instance dict;被拒絕回 None。
        拒絕條件:信心等級 >= L3、沒有分區、座標無效。
        """
        now = now_or(now)
        try:
            level = int(confidence_level)
        except Exception:
            return None
        if level > int(cfg("OBJ_WRITE_MAX_LEVEL", 2)):     # 只有 L1/L2 能寫
            return None
        if not cls or not zone_id or local_xyz is None:
            return None
        try:
            xyz = [float(v) for v in local_xyz]
            if len(xyz) != 3 or any(math.isnan(v) or math.isinf(v) for v in xyz):
                return None
        except Exception:
            return None

        cls = str(cls).strip().lower()
        conf = float(conf if conf is not None else 0.0)
        lst = self.items.setdefault(cls, [])

        # ---- 去重:同分區、距離 < 門檻視為同一個實體 ----
        dedup_mm = float(cfg("OBJ_DEDUP_MM", 300.0))
        for inst in lst:
            if inst.get("zone_id") != zone_id:
                continue
            d = _dist(inst.get("local_xyz"), xyz)
            if d is not None and d < dedup_mm:
                # 位置取加權平均(新觀測權重 = 信心),避免單次抖動整個搬家
                w = max(0.05, min(0.9, conf))
                inst["local_xyz"] = [(1 - w) * a + w * b
                                     for a, b in zip(inst["local_xyz"], xyz)]
                inst["ts"] = now
                inst["conf"] = max(float(inst.get("conf", 0.0)), conf)
                inst["seen"] = int(inst.get("seen", 1)) + 1
                inst["confidence_level"] = min(int(inst.get("confidence_level", 5)), level)
                inst["coord_quality"] = coord_quality
                self._below_threshold.discard(inst["instance_id"])
                self.save()
                return dict(inst)

        inst = {
            "instance_id": uuid.uuid4().hex[:12],
            "cls": cls, "zone_id": zone_id, "local_xyz": xyz,
            "coord_quality": coord_quality, "ts": now, "conf": conf,
            "confidence_level": level, "seen": 1, "created": now,
        }
        lst.append(inst)

        # ---- 每類上限(§4.4):超量丟可信度最低的 ----
        cap = int(cfg("OBJ_MAX_PER_CLASS", 5))
        if len(lst) > cap:
            lst.sort(key=lambda i: (self.reliability(i, now), i.get("ts", 0)), reverse=True)
            del lst[cap:]
        self.items[cls] = lst
        self.save()
        return dict(inst)

    # ============================================================
    # 衰減
    # ============================================================
    def reliability(self, inst, now=None):
        """0–1。半衰期式指數衰減,`seen` 多的實體衰減較慢(反覆確認過)。"""
        now = now_or(now)
        half = float(cfg("OBJ_HALFLIFE_DAYS", 7.0))
        seen_bonus = 1.0 + 0.2 * math.log(max(1, int(inst.get("seen", 1))), 2)
        age_d = max(0.0, (now - float(inst.get("ts", now)))) / 86400.0
        r = 0.5 ** (age_d / max(0.1, half * seen_bonus))
        return float(max(0.0, min(1.0, r)))

    def decay_tick(self, now=None) -> List[dict]:
        """
        記憶衰減計時器(§2.2:永遠運作,包含待機)。
        回傳這次「剛跌破 40% 門檻」的實體 list,供上層決定要不要提醒。
        本函式不刪資料——刪除是 standalone_privacy_purge 的職責。
        """
        now = now_or(now)
        thr = float(cfg("OBJ_DECAY_WARN_THRESH", 0.40))
        newly = []
        for cls, lst in self.items.items():
            for inst in lst:
                r = self.reliability(inst, now)
                iid = inst["instance_id"]
                if r < thr and iid not in self._below_threshold:
                    self._below_threshold.add(iid)
                    newly.append({**inst, "reliability": r,
                                  "text": f"{_zh(cls)}的位置記錄較舊,可能已被移動"})
                elif r >= thr:
                    self._below_threshold.discard(iid)
        return newly

    # ============================================================
    # 查詢(§4.5 排序)
    # ============================================================
    def get_nearest(self, cls, current_zone=None, current_xyz=None,
                    now=None, limit=3) -> List[dict]:
        """
        排序:同分區優先 → 跨分區比距離 → 同距離比新鮮度。
        每筆附 `phrase`(依信心等級與衰減決定措辭)。
        """
        now = now_or(now)
        cls = str(cls or "").strip().lower()
        lst = list(self.items.get(cls, []))
        if not lst:                                  # 寬鬆比對(沿用既有查表精神)
            for k, v in self.items.items():
                if cls and (cls in k or k in cls):
                    lst = list(v)
                    break
        if not lst:
            return []

        out = []
        for inst in lst:
            d = _dist(inst.get("local_xyz"), current_xyz) if current_xyz is not None else None
            same_zone = (current_zone is not None and inst.get("zone_id") == current_zone)
            r = self.reliability(inst, now)
            out.append({**inst, "distance_mm": d, "same_zone": same_zone,
                        "reliability": r, "phrase": self._phrase(inst, r, same_zone)})
        out.sort(key=lambda i: (
            0 if i["same_zone"] else 1,
            i["distance_mm"] if i["distance_mm"] is not None else 1e12,
            -i["ts"],
        ))
        return out[:limit]

    def _phrase(self, inst, reliability, same_zone):
        name = _zh(inst["cls"])
        thr = float(cfg("OBJ_DECAY_WARN_THRESH", 0.40))
        where = "這個空間裡" if same_zone else "別的空間"
        if reliability < thr:
            return f"{name}上次記錄在{where},但時間較久,可能已被移動"
        if inst.get("coord_quality") != "tag_pose":
            return f"{name}大約記錄在{where},位置僅供參考"
        return f"{name}記錄在{where}"

    # ============================================================
    # 隱私保留期限(由 standalone_privacy_purge 呼叫)
    # ============================================================
    def purge_expired(self, retention_days=30, now=None) -> int:
        now = now_or(now)
        cutoff = now - float(retention_days) * 86400.0
        removed = 0
        for cls in list(self.items.keys()):
            keep = [i for i in self.items[cls] if float(i.get("ts", 0)) >= cutoff]
            removed += len(self.items[cls]) - len(keep)
            if keep:
                self.items[cls] = keep
            else:
                del self.items[cls]
        if removed:
            self.save()
        return removed

    def stats(self, now=None):
        now = now_or(now)
        n = sum(len(v) for v in self.items.values())
        return {"classes": len(self.items), "instances": n,
                "zones": sorted({i.get("zone_id") for v in self.items.values() for i in v}),
                "low_reliability": sum(1 for v in self.items.values() for i in v
                                       if self.reliability(i, now) < 0.40)}


def _dist(a, b):
    if a is None or b is None:
        return None
    try:
        return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
    except Exception:
        return None


# ============================================================
# 模組層單例
# ============================================================
_singleton = None


def memory() -> ObjectMemory:
    global _singleton
    if _singleton is None:
        _singleton = ObjectMemory()
    return _singleton


def register_detection(cls, local_xyz, zone_id, conf, confidence_level,
                       coord_quality="tag_pose", now=None):
    return memory().register_detection(cls, local_xyz, zone_id, conf,
                                       confidence_level, coord_quality, now)


def get_nearest(cls, current_zone=None, current_xyz=None, now=None, limit=3):
    return memory().get_nearest(cls, current_zone, current_xyz, now, limit)


def decay_tick(now=None):
    return memory().decay_tick(now)


def purge_expired(retention_days=30, now=None):
    return memory().purge_expired(retention_days, now)
