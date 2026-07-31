"""
standalone_gen3_selftest.py — 第三代合成情境自我測試(不需硬體)
================================================================
執行:  python standalone_gen3_selftest.py
用途:每次改動後先跑這支,確認信心階梯、雙重門檻、去重、衰減、隱私清除
      這些「講不出口就是沒驗證」的邏輯行為沒有跑掉。
所有時間都用注入的假 now,不依賴真實時鐘,結果可重現。
"""
import os
import shutil
import sys
import tempfile

import numpy as np

# ---- 測試資料目錄隔離(不碰到實機資料) ----
TMP = tempfile.mkdtemp(prefix="gen3_test_")
os.environ["GEN3_DATA_DIR"] = TMP          # 沒有 config 也能隔離測試資料
CONFIG = None
for _m in ("shared_config", "config"):          # 改名前後都吃
    try:
        CONFIG = getattr(__import__(_m), "CONFIG")
        CONFIG.GEN3_DATA_DIR = TMP
        break
    except Exception:
        continue

import cv2

import percept_apriltag as apriltag
import percept_pseudo_vlm as pv
import percept_slam as slam
import standalone_privacy_purge as purge
from hw_confidence import ConfidenceHaptic
from memory_zone_state import ConfLevel, ZoneState
from memory_object import ObjectMemory

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f"  ({detail})" if detail and not cond else ""))


def D(cls, conf=0.9, n=1):
    return [(0, 0, 10, 10, cls, conf)] * n


# ============================================================
print("\n── 1. 偽VLM:權重表稽核與雙重門檻 ──")
a = pv.audit_table()
check("權重表與 df 分數帶一致", a["ok"], str(a["violations"]))

p = pv.PseudoVLM()
kitchen = D("refrigerator") + D("microwave") + D("sink") + D("person")
for _ in range(5):
    v = p.infer_scene(kitchen, mode="daily")
check("廚房強證據 → 通過雙重門檻", v.passed and v.scene == "kitchen",
      f"{v.scene}/{v.score:.1f}/{v.margin:.1f}")

p2 = pv.PseudoVLM()
for _ in range(5):
    v2 = p2.infer_scene(D("couch") + D("chair") + D("person"), mode="daily")
check("單一強物件不足以通過(沉默優先)", not v2.passed, f"score={v2.score:.1f}")

p3 = pv.PseudoVLM()
for _ in range(5):
    v3 = p3.infer_scene(D("chair") + D("cup") + D("bottle"), mode="daily")
check("模糊場景不通過", not v3.passed, f"{v3.scene}/{v3.margin:.1f}")

p4 = pv.PseudoVLM()
v4 = p4.infer_scene(D("toilet") + D("toothbrush") + D("sink"), mode="vlm")
check("vlm 模式單幀即可秒答", v4 is not None and v4.passed and v4.n_frames == 1)

p5 = pv.PseudoVLM()
for _ in range(5):
    v5 = p5.infer_scene(D("refrigerator", conf=0.20) + D("microwave", conf=0.20), mode="daily")
check("低信心偵測被第一層門檻擋掉", v5 is None)

p6 = pv.PseudoVLM()
for i in range(5):
    v6 = p6.infer_scene(kitchen if i == 0 else D("person"), mode="daily")
check("單幀閃爍不足 min_hits → 不採用", "refrigerator" not in (v6.admitted if v6 else []))

p7 = pv.PseudoVLM()
one = p7._score({"chair": 1})[0]["living_room"]
three = p7._score({"chair": 3})[0]["living_room"]
check("數量遞減 0.5^(N-1) 生效", abs(three - one * 1.75) < 1e-6, f"{one:.2f}→{three:.2f}")
check("traffic 模式不允許發話", pv.MODE_PROFILES["traffic"]["announce"] is False)
check("search 模式不允許發話", pv.MODE_PROFILES["search"]["announce"] is False)

# ============================================================
print("\n── 2. 信心階梯 L1–L5 ──")
z = ZoneState(path=os.path.join(TMP, "z1.json"), autoload=False)
z.register_zone("kitchen_01", "kitchen", "廚房")
tag = {"tag_id": 3, "zone_id": "kitchen_01", "room_type": "kitchen",
       "decision_margin": 60.0}
t0 = 1000.0
z.on_tag_seen(tag, t0)
check("看到高 margin tag → L1", z.get_confidence_level(t0) == ConfLevel.L1)

z.on_slam_update({"tracking": True, "n_features": 200}, t0 + 1.0)
check("tag 消失 1 秒 + SLAM 健康 → L2", z.get_confidence_level(t0 + 1.0) == ConfLevel.L2)
z.on_slam_update({"tracking": False, "n_features": 10}, t0 + 1.2)
check("SLAM 失效 → 不給 L2(退 L3)", z.get_confidence_level(t0 + 1.2) == ConfLevel.L3)
check("tag 消失 5 秒 → L3", z.get_confidence_level(t0 + 5) == ConfLevel.L3)
check("tag 消失 20 秒 → L4", z.get_confidence_level(t0 + 20) == ConfLevel.L4)
check("tag 消失 40 秒 → L5", z.get_confidence_level(t0 + 40) == ConfLevel.L5)
check("L4 仍回報最後已知分區且標 assumed",
      (z.get_current_zone(t0 + 20) or {}).get("assumed") is True)
check("L5 不回報任何分區", z.get_current_zone(t0 + 40) is None)
check("L3 不給精確座標", (z.get_current_zone(t0 + 5) or {}).get("coords_valid") is False)

# margin=None 的連續確認
z2 = ZoneState(path=os.path.join(TMP, "z2.json"), autoload=False)
z2.register_zone("bath_01", "bathroom", "浴室")
nomargin = {"tag_id": 9, "zone_id": "bath_01", "room_type": "bathroom",
            "decision_margin": None}
z2.on_tag_seen(nomargin, 500.0)
check("無 decision_margin:第一次不給 L1", z2.get_confidence_level(500.0) == ConfLevel.L5)
z2.on_tag_seen(nomargin, 501.0)
check("無 decision_margin:連續兩次才給 L1", z2.get_confidence_level(501.0) == ConfLevel.L1)

# 位移降級
z3 = ZoneState(path=os.path.join(TMP, "z3.json"), autoload=False)
z3.register_zone("living_01", "living_room", "客廳")
z3.on_tag_seen({"tag_id": 1, "zone_id": "living_01", "room_type": "living_room",
                "decision_margin": 50}, 800.0)
z3.on_displacement(5000.0, 800.5)
check("位移超過 4.5m → 分區假設降級至 L4", z3.get_confidence_level(800.5) == ConfLevel.L4)

# 造訪計數持久化(修過的舊 bug)
zp = os.path.join(TMP, "z_persist.json")
za = ZoneState(path=zp, autoload=False)
za.register_zone("k1", "kitchen")
za.on_tag_seen({"tag_id": 5, "zone_id": "k1", "room_type": "kitchen",
                "decision_margin": 50}, 1.0)
zb = ZoneState(path=zp)
check("造訪次數重開機後保留", zb.zones.get("k1", {}).get("visits", 0) >= 1,
      str(zb.zones))

# ============================================================
print("\n── 3. 記憶落差偵測(§4.8)與 L4/L5 備用猜測(§5.3) ──")
zg = ZoneState(path=os.path.join(TMP, "z4.json"), autoload=False)
zg.register_zone("bed_01", "bedroom", "臥室")
zg.on_tag_seen({"tag_id": 2, "zone_id": "bed_01", "room_type": "bedroom",
                "decision_margin": 50}, 2000.0)
kv = pv.PseudoVLM()
for _ in range(5):
    kvote = kv.infer_scene(kitchen, mode="daily")
fired = []
for i in range(3):
    fired += zg.report_scene_vote(kvote, 2000.0 + i * 0.1)
check("落差需連續 3 次確認才觸發", len(fired) == 1, f"觸發 {len(fired)} 次")
check("落差事件內容正確",
      fired and fired[0]["registered"] == "bedroom" and fired[0]["observed"] == "kitchen")

zh_ = ZoneState(path=os.path.join(TMP, "z5.json"), autoload=False)
zh_.register_zone("bath_02", "bathroom", "浴室")
zh_.on_tag_seen({"tag_id": 4, "zone_id": "bath_02", "room_type": "bathroom",
                 "decision_margin": 50}, 3000.0)
bath = pv.PseudoVLM()
for _ in range(5):
    bvote = bath.infer_scene(D("toilet") + D("toothbrush") + D("sink"), mode="daily")
check("L1 期間場景投票不產生備用猜測",
      zh_.report_scene_vote(bvote, 3000.0) == [] and zh_.get_scene_hint(3000.0) is None)
zh_.report_scene_vote(bvote, 3040.0)                       # 此時已 L5
hint = zh_.get_scene_hint(3040.0)
check("L5 期間才啟用備用猜測", hint is not None and hint["scene"] == "bathroom")
check("備用猜測措辭保守且標記低於 L4",
      hint and "無法確定精確位置" in hint["text"] and hint["below_l4"])
zh_.on_tag_seen({"tag_id": 4, "zone_id": "bath_02", "room_type": "bathroom",
                 "decision_margin": 50}, 3041.0)
check("tag 回來立即清除備用猜測", zh_.get_scene_hint(3041.0) is None)

# ============================================================
print("\n── 4. 物件記憶(§4.3–4.7) ──")
om = ObjectMemory(path=os.path.join(TMP, "obj.json"), autoload=False)
T = 10000.0
check("L3 觀測被拒絕寫入",
      om.register_detection("cup", (100, 0, 1000), "k1", 0.9, 3, now=T) is None)
r1 = om.register_detection("cup", (100, 0, 1000), "k1", 0.9, 1, now=T)
check("L1 觀測寫入成功", r1 is not None)
om.register_detection("cup", (120, 0, 1010), "k1", 0.9, 1, now=T + 1)
check("30cm 內視為同一實體(去重)", len(om.items["cup"]) == 1)
om.register_detection("cup", (2000, 0, 1000), "k1", 0.9, 1, now=T + 2)
check("超過去重門檻 → 新實體", len(om.items["cup"]) == 2)
for i in range(6):
    om.register_detection("cup", (5000 + i * 1000, 0, 1000), "k1", 0.5, 1, now=T + 3 + i)
check("每類上限 5 筆", len(om.items["cup"]) == 5)

om2 = ObjectMemory(path=os.path.join(TMP, "obj2.json"), autoload=False)
om2.register_detection("cup", (5000, 0, 0), "zoneA", 0.9, 1, now=T)
om2.register_detection("cup", (100, 0, 0), "zoneB", 0.9, 1, now=T)
res = om2.get_nearest("cup", current_zone="zoneA", current_xyz=(0, 0, 0), now=T)
check("同分區優先於距離較近的跨分區", res[0]["zone_id"] == "zoneA")

old = ObjectMemory(path=os.path.join(TMP, "obj3.json"), autoload=False)
old.register_detection("kettle", (0, 0, 500), "k1", 0.9, 1, now=T)
late = T + 20 * 86400
newly = old.decay_tick(now=late)
check("超過半衰期後跌破 40% 門檻", len(newly) == 1 and "可能已被移動" in newly[0]["text"])
check("衰減事件只報一次", old.decay_tick(now=late + 10) == [])
ph = old.get_nearest("kettle", current_zone="k1", current_xyz=(0, 0, 0), now=late)[0]["phrase"]
check("查詢措辭反映衰減", "可能已被移動" in ph, ph)

# ============================================================
print("\n── 5. 觸覺信心編碼(§7) ──")
sent = []
hp = ConfidenceHaptic(publish_fn=sent.append, motor_id=3)
hp.set_confidence_vibration(1, now=0.0)
check("L1 穩定 100%", sent and sent[-1].startswith("3:100:"), str(sent[-1:]))
hp.set_confidence_vibration(4, now=10.0)
check("L4 弱短脈衝 40%", sent[-1].startswith("3:40:"))
check("L4 附帶語音修飾詞「可能」", hp.pending_voice() == "可能")
n_before = len(sent)
hp.set_confidence_vibration(5, now=20.0)
hp.tick(now=30.0)
check("L5 不發方位震動", len(sent) == n_before)
hp.set_confidence_vibration(1, now=40.0)
n_before = len(sent)
hp.hold_for_safety(2.0, now=40.0)
hp.tick(now=41.0)
check("安全警告期間信心震動讓路", len(sent) == n_before)
hp.tick(now=43.0)
check("讓路結束後恢復", len(sent) == n_before + 1)

# ============================================================
print("\n── 6. AprilTag 感知層 ──")
check("後端可用", apriltag.available(), apriltag.backend_name())
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
img = np.full((480, 640), 255, np.uint8)
img[140:340, 220:420] = cv2.aruco.generateImageMarker(d, 11, 200)
apriltag.bind_tag(11, "study_01", "study", persist=False)
dets = apriltag.detect_tags(img)
check("合成 tag 偵測成功", len(dets) == 1 and dets[0].tag_id == 11)
check("tag → 分區身分解析", dets[0].zone_id == "study_01" and dets[0].resolved)
check("無內參時仍給距離估計", dets[0].distance_mm is not None)
check("後端無 margin 時誠實回 None", dets[0].decision_margin is None
      and dets[0].margin_source == "unavailable")
qr_det = apriltag._resolve_identity(
    apriltag.TagDetection(tag_id=-1, family="qr", center=(0, 0),
                          corners=np.zeros((4, 2), np.float32),
                          payload="ZONE:kitchen:kitchen_02"))
check("QR payload 格式解析", qr_det.room_type == "kitchen" and qr_det.zone_id == "kitchen_02")
check("空影像不炸", apriltag.detect_tags(None) == [])

# ============================================================
print("\n── 7. SLAM 前端 ──")
f = slam.SlamFrontend()
rich = (np.random.rand(480, 640) * 255).astype(np.uint8)
check("紋理豐富 → tracking", f.update(rich, now=1.0).tracking)
check("白牆 → 追蹤失敗", not f.update(np.full((480, 640), 128, np.uint8), now=2.0).tracking)
check("第一版誠實回報無位姿", f.get_state().pose is None)

# ============================================================
print("\n── 8. 整合管線 ──")
import proc_gen3_pipeline as pipe
zone_single = ZoneState(path=os.path.join(TMP, "pipe_zone.json"), autoload=False)
zone_single.register_zone("study_01", "study", "書房")
apriltag.bind_tag(11, "study_01", "study")      # 建置模式會落檔,管線啟動時讀回
gp = pipe.Gen3Pipeline(publish_fn=lambda p: None, zone=zone_single,
                       objects=ObjectMemory(path=os.path.join(TMP, "pipe_obj.json"),
                                            autoload=False))
pre = gp.step(img, dets=D("keyboard") + D("mouse") + D("laptop"), mode="daily", now=5000.0)
check("aruco 後端無 margin:單幀不給 L1(連續確認機制生效)", pre["level"] == 5)
out = gp.step(img, dets=D("keyboard") + D("mouse") + D("laptop"), mode="daily", now=5000.3)
check("連續兩幀確認後 → L1 並鎖定分區",
      out["level"] == 1 and out["zone"]["zone_id"] == "study_01")
check("管線回報進入分區語音", any("書房" in s for s in out["speeches"]), str(out["speeches"]))
apriltag.set_camera_intrinsics(500, 500, 320, 240)
xyz = gp.pixel_to_camera_xyz(320, 240, 1500)
inst = gp.note_object("book", xyz, 0.9, now=5000.4)
check("管線可登記物件座標", inst is not None and inst["zone_id"] == "study_01")
q = gp.query_object("book", now=5000.5)
check("管線可查詢物件記憶", len(q) == 1 and q[0]["same_zone"])
out_idle = gp.step(img, dets=D("keyboard") * 3, mode="idle", now=5001.0)
check("待機模式不做場景投票", out_idle["vote"] is None)

# ============================================================
print("\n── 9. 隱私清除(§九) ──")
pobj = os.path.join(TMP, "object_memory.json")
om4 = ObjectMemory(path=pobj, autoload=False)
now9 = 20000000.0
om4.register_detection("cup", (0, 0, 500), "k1", 0.9, 1, now=now9 - 40 * 86400)
om4.register_detection("bowl", (0, 0, 500), "k1", 0.9, 1, now=now9 - 1 * 86400)
r = purge.purge_expired(retention_days=30, dry_run=True, now=now9)
check("dry-run 不刪資料", r["objects_removed"] == 1
      and len(ObjectMemory(path=pobj).items) == 2)
r = purge.purge_expired(retention_days=30, now=now9)
check("超過保留期限的物件被清除", len(ObjectMemory(path=pobj).items) == 1)

# ============================================================
print("\n── 10. 向後相容 ──")
import shared_store
_here = os.path.dirname(os.path.abspath(__file__))
if any(os.path.exists(os.path.join(_here, m + ".py")) for m in ("utils", "shared_utils")):
    check("shared_store 轉發既有工具函式", shared_store._UTILS_OK
          and shared_store.zone_to_motor("左") == 0)
else:
    print("⏭️  略過 shared_store 轉發檢查(utils/shared_utils 不在同一目錄)")
# 改名前後都要能過:哪個名字存在就檢查哪個
for pair in (("config", "shared_config"), ("utils", "shared_utils"),
             ("trackers", "percept_trackers"), ("crosswalk", "percept_crosswalk"),
             ("motor_zones", "hw_motor_zones")):
    here = os.path.dirname(os.path.abspath(__file__))
    if not any(os.path.exists(os.path.join(here, m + ".py")) for m in pair):
        print(f"⏭️  略過 {pair[0]}/{pair[1]}(檔案不在同一目錄,非錯誤)")
        continue
    ok, err, hit = False, "", None
    for m in pair:
        try:
            __import__(m)
            ok, hit = True, m
            break
        except Exception as e:
            err = str(e)
    check(f"既有模組 {pair[0]}/{pair[1]} 可匯入" + (f" → {hit}" if ok else ""), ok, err)
# ============================================================
print("\n" + "=" * 52)
print(f"通過 {len(PASS)} 項,失敗 {len(FAIL)} 項")
if FAIL:
    for f_ in FAIL:
        print("  ❌", f_)
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if FAIL else 0)
