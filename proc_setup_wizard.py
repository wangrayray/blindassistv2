"""
proc_setup_wizard.py — 建置模式:分區登錄精靈(流程層)
=======================================================
公開介面(對外承諾):
    main()                                   # 獨立 process 入口
    run_wizard(frame_source, ask=None, say=None, max_frames=None) -> List[dict]
不對外開放:互動流程細節。

用途(§2.1 建置模式 / §5.3 延伸場景二):
  安裝時把 AprilTag / QR 貼到各分區門口,操作者站在該分區內按一次 Enter,
  系統連拍數幀 → 偽VLM 投票 → 列出前三個候選房型 → **人工確認**。
  投票只是建議者,QR/人工確認才是權威(§5.6)——所以這裡一定要人按下確認,
  沒有「自動採用」這條路。

★ 這個模式不進日常操作選單(§2.1),獨立執行:
    python proc_setup_wizard.py
★ 與日常路徑共用同一套 percept_pseudo_vlm,零額外開發(§5.3 延伸場景二)。
"""
import sys
import time

import percept_apriltag as apriltag
import percept_pseudo_vlm as pseudo_vlm
import memory_zone_state as zone_state
from percept_pseudo_vlm import SCENE_ZH


def _import_any(module_names, attr):
    """依序嘗試模組名,取出指定屬性。用來同時相容檔名重構前後的狀態。"""
    for m in module_names:
        try:
            return getattr(__import__(m), attr)
        except Exception:
            continue
    return None


def _default_say(msg):
    print(msg)


def _default_ask(prompt):
    return input(prompt)


def run_wizard(frame_source, ask=None, say=None, max_frames=None, detector=None):
    """
    frame_source: callable() -> (frame, detections) 或 (frame, detections, done)
                  detections 用 ai_worker.infer_detect 的 tuple 格式即可。
    ask / say   : 注入互動函式(單元測試可用假函式,不需要真的鍵盤)。
    detector    : 注入 tag 偵測函式(預設 percept_apriltag.detect_tags)。
    回傳已登錄的分區 list。
    """
    ask = ask or _default_ask
    say = say or _default_say
    detect = detector or apriltag.detect_tags
    pv = pseudo_vlm.PseudoVLM()
    zs = zone_state.state()
    apriltag.load_tag_zone_map()

    frames = int(max_frames or 10)
    registered = []

    say("=== 手護視界 建置模式 ===")
    say("請站在要登錄的分區內,讓 AprilTag/QR 進入畫面,然後按 Enter。輸入 q 結束。")

    while True:
        cmd = ask("\n[Enter]=掃描這個分區 / q=結束 > ").strip().lower()
        if cmd == "q":
            break

        tag_found = None
        pv.reset("setup")
        for _ in range(frames):
            got = frame_source()
            if got is None:
                break
            frame, dets = got[0], got[1]
            if frame is not None and tag_found is None:
                t = apriltag.best_tag(detect(frame))
                if t is not None:
                    tag_found = t
            pv.infer_scene(dets, mode="setup")

        suggestions = pv.suggest_zone_labels([], top_k=3, mode="setup")
        if not suggestions:
            say("⚠️ 這次掃描沒有取得足夠的物件證據,偽VLM 無法提供建議,請直接手動輸入房型。")
        else:
            say("偽VLM 建議(僅供參考,以你的確認為準):")
            for i, s in enumerate(suggestions, 1):
                ev = "、".join(s["evidence"][:4])
                say(f"   {i}. {s['scene_zh']}({s['scene']})  分數 {s['score']}  依據:{ev or '—'}")

        if tag_found is None:
            say("⚠️ 這次沒有偵測到 tag/QR。分區必須綁在一個實體錨點上,請調整角度重試。")
            continue
        say(f"偵測到錨點:{tag_found.family} id={tag_found.tag_id}"
            + (f" payload={tag_found.payload}" if tag_found.payload else ""))

        default_room = (tag_found.room_type
                        or (suggestions[0]["scene"] if suggestions else "unknown"))
        room = ask(f"這個分區的房型 [{default_room}] > ").strip() or default_room
        if room not in SCENE_ZH:
            say(f"ℹ️ 「{room}」不在八個標準房型內,仍會登錄,但偽VLM 的落差偵測對它無效。")
        default_zone = tag_found.zone_id or f"{room}_{abs(tag_found.tag_id) % 100:02d}"
        zone_id = ask(f"分區代號 [{default_zone}] > ").strip() or default_zone
        name = ask(f"顯示名稱 [{SCENE_ZH.get(room, room)}] > ").strip() or SCENE_ZH.get(room, room)

        apriltag.bind_tag(tag_found.tag_id, zone_id, room)
        zs.register_zone(zone_id, room, name)
        registered.append({"zone_id": zone_id, "room_type": room, "name": name,
                           "tag_id": tag_found.tag_id})
        say(f"✅ 已登錄:{name}({zone_id})← tag {tag_found.tag_id}")

    say(f"\n完成,本次共登錄 {len(registered)} 個分區。")
    say(f"分區檔:{zs.path}")
    say(f"錨點對照表:{apriltag.zone_map_path()}")
    return registered


def main():
    """實機執行:從 OAK 主鏡頭抓幀 + 背景 YOLO 給偵測。"""
    Cam = _import_any(("hw_camera", "camera"), "CameraManager")   # 改名前後都吃
    if Cam is None:
        print("⚠️ 無法載入相機模組,建置模式需要在實機上執行。")
        return 1
    try:
        from ultralytics import YOLO
        CONFIG = _import_any(("shared_config", "config"), "CONFIG")
        model = YOLO(CONFIG.MODEL_SCAN)
    except Exception as e:
        print(f"⚠️ 無法載入 YOLO({e}),將只做 tag 綁定,不提供房型建議。")
        model = None

    cam = Cam()
    cam.start()

    def source():
        ret, frame, _depth = cam.read()      # CameraManager.read() → (ret, rgb, depth)
        if not ret or frame is None:
            return None
        dets = []
        if model is not None:
            try:
                res = model(frame, verbose=False, conf=0.4)[0]
                dets = [(0, 0, 1, 1, model.names[int(b.cls)].lower(),
                         float(b.conf[0])) for b in res.boxes]
            except Exception:
                pass
        time.sleep(0.05)
        return frame, dets

    try:
        run_wizard(source)
    finally:
        if hasattr(cam, "stop"):
            cam.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
