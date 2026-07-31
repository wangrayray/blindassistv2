#!/usr/bin/env python3
"""
Blind Assist System - Main Entry (v9 模組化最終版)
=====================================================
"""
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;2000000"

import time
import queue
import threading
import multiprocessing as mp

import cv2

from shared_config import CONFIG
from hw_camera import CameraManager
from hw_ir_mode import IrModeController
from hw_wide_cam import WideCamManager, PrescanTagCache
from hw_haptic import HardwareManager
from proc_ai_worker import ai_worker_process
from proc_vlm_worker import vlm_worker_process
from proc_flask_app import app, State, signal_speech, handle_speech_signal
from proc_overlay import (draw_overlay_items, draw_motor_indicator,
                     draw_hud, blank_waiting_frame, make_depth_visualization)
from hw_motor_zones import draw_motor_zones


def main():
    try: mp.set_start_method("spawn", force=True)
    except RuntimeError: pass

    # AI worker
    ai_in_q  = mp.Queue(maxsize=1)
    ai_out_q = mp.Queue(maxsize=1)
    ai_cmd_q = mp.Queue()
    ai_proc = mp.Process(target=ai_worker_process,
                         args=(ai_in_q, ai_out_q, ai_cmd_q), daemon=True)
    ai_proc.start()

    # VLM worker
    vlm_in_q  = mp.Queue(maxsize=2)
    vlm_out_q = mp.Queue(maxsize=20)
    vlm_cmd_q = mp.Queue()
    vlm_proc = mp.Process(target=vlm_worker_process,
                          args=(vlm_in_q, vlm_out_q, vlm_cmd_q), daemon=True)
    vlm_proc.start()

    # 硬體與相機
    hw = HardwareManager()
    cam = CameraManager()
    cam.start()

    # IR 主動/被動深度動態切換 (webcam fallback 時自動停用)
    ir_ctrl = IrModeController(getattr(cam, "device", None))

    # ---- 第三代背景基礎設施層 (AprilTag / SLAM / 分區記憶 / 偽VLM) ----
    # 整包失敗也不能影響主系統:導盲的底線是避障,定位是加值。
    gen3 = None
    if getattr(CONFIG, "GEN3_ENABLED", False):
        try:
            from proc_gen3_pipeline import Gen3Pipeline
            import percept_apriltag as _apriltag
            intr = cam.get_intrinsics() if hasattr(cam, "get_intrinsics") else None
            if intr:
                _apriltag.set_camera_intrinsics(*intr)   # 有內參才給得出 tag 位姿
            gen3 = Gen3Pipeline(hardware=hw)
            print("🗺️ 第三代分區記憶已啟用"
                  + ("(有相機內參,物件座標掛 tag 錨點)" if intr
                     else "(無內參,物件座標僅相機座標系)"))
        except Exception as e:
            print(f"⚠️ 第三代模組載入失敗,系統以純避障模式運行: {e}")
            gen3 = None

    # 廣角輔助鏡頭 + VLM 預掃標籤快取
    wide_cam = WideCamManager()
    wide_cam.start()
    tag_cache = PrescanTagCache()
    last_prescan_push = 0.0   # 上次丟預掃任務給 VLM 的時間
    last_scan_push = 0.0      # 上次丟廣角幀給 AI worker 做背景 YOLO 掃描的時間

    # Flask 注入
    State.ai_cmd_q = ai_cmd_q
    State.vlm_in_q = vlm_in_q
    State.cam = cam
    State.wide_cam = wide_cam      # VLM 改用廣角影像

    # ---- Web UI:HTTP 保底 + HTTPS 可選(雙埠並存)----
    # 為什麼要 HTTPS:瀏覽器的 SpeechRecognition(語音輸入)只在
    # 「安全來源」下可用,http://<區網IP> 不算安全來源。
    # 為什麼不直接只開 HTTPS:自簽憑證要每支手機安裝信任根 CA,
    # 沒裝好就整個連不上——保留 :5000 當保底,語音輸入以外的功能照常用。
    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=5000,
                               debug=False, use_reloader=False),
        daemon=True,
    ).start()
    print("🌐 HTTP  介面: http://<本機IP>:5000")

    if getattr(CONFIG, "HTTPS_ENABLED", False):
        cert, key = CONFIG.HTTPS_CERT_FILE, CONFIG.HTTPS_KEY_FILE
        if os.path.exists(cert) and os.path.exists(key):
            threading.Thread(
                target=lambda: app.run(host="0.0.0.0", port=CONFIG.HTTPS_PORT,
                                       debug=False, use_reloader=False,
                                       ssl_context=(cert, key)),
                daemon=True,
            ).start()
            print(f"🔒 HTTPS 介面: https://<本機IP>:{CONFIG.HTTPS_PORT}(語音輸入需走這個)")
        else:
            print("ℹ️ 未偵測到憑證,HTTPS 未啟動(語音輸入不可用)。"
                  "執行 ./mkcert_setup.sh 產生憑證。")

    print("🚀 主系統啟動")

    frame_counter = 0
    frame_cache = {}
    latest_draws = []
    latest_draw_frame_id = -1
    latest_dets = []                 # ai_worker 回傳的最近偵測,餵給第三代場景投票
    motor_display_until = {}
    last_mqtt_warn = 0.0

    target_dt = 1.0 / CONFIG.MAIN_LOOP_FPS
    main_fps_ts = time.time(); main_fps_cnt = 0; fps_main = 0.0
    ai_fps_ts = time.time();   ai_fps_cnt = 0;   fps_ai = 0.0
    _prev_vlm_busy = False   # 追蹤 VLM busy 變化,用來暫停/恢復 YOLO
    _prev_traffic = False    # 追蹤是否在 traffic,用來切換廣角快/慢擷取

    try:
        while True:
            loop_start = time.time()

            # 安全保險:VLM busy 超時仍未完成 → 強制解除,避免 YOLO 永久暫停
            # (導盤系統不能因 VLM 卡死而失去避障)
            if (State.vlm_busy and State.vlm_busy_since
                    and loop_start - State.vlm_busy_since
                    > CONFIG.VLM_BUSY_TIMEOUT):
                print(f"⚠️ VLM 超過 {CONFIG.VLM_BUSY_TIMEOUT}s 未完成,"
                      f"強制解除並恢復 YOLO")
                State.vlm_busy = False
                State.vlm_busy_since = 0.0

            # VLM 開始忙 → 暫停 YOLO 讓出 GPU;VLM 結束 → 恢復 YOLO
            if State.vlm_busy != _prev_vlm_busy:
                if ai_cmd_q is not None:
                    try:
                        ai_cmd_q.put_nowait({"yolo_pause": State.vlm_busy})
                    except queue.Full:
                        pass
                _prev_vlm_busy = State.vlm_busy

            # traffic 模式切換 → 廣角擷取快/慢切換 (紅綠燈雙鏡頭需即時幀)
            _now_traffic = (State.current_mode == "traffic")
            if _now_traffic != _prev_traffic:
                if wide_cam.available:
                    wide_cam.set_fast_mode(_now_traffic)
                    print(f"📷 廣角擷取切換: {'快頻(traffic)' if _now_traffic else '省電'}")
                _prev_traffic = _now_traffic

            ret, frame, depth_frame = cam.read()

            valid = ret and frame is not None and frame.shape[0] > 0
            if not valid:
                draw_frame = blank_waiting_frame() if CONFIG.SHOW_DEBUG_WINDOW else None
            else:
                h, w = frame.shape[:2]

                frame_counter += 1
                frame_cache[frame_counter] = (frame, loop_start)
                expired = [k for k, (_, t) in frame_cache.items() if loop_start - t > 1.0]
                for k in expired:
                    frame_cache.pop(k, None)

                if ai_in_q.full():
                    try: ai_in_q.get_nowait()
                    except queue.Empty: pass

                # traffic 模式:夾帶最新廣角幀,供 ai_worker 雙鏡頭辨識紅綠燈
                # (廣角水平擺放,補 OAK 俯角造成的上方號誌盲區;斑馬線仍只用 OAK)
                wide_frame_for_ai = None
                if State.current_mode == "traffic" and wide_cam.available:
                    latest_w = wide_cam.get_latest()
                    if latest_w is not None:
                        _, wide_frame_for_ai = latest_w

                # IMU 即時俯角:跨 process 無法共享物件,只能塞進 frame dict 走 queue。
                # None = 沒有 IMU / 資料過期 → ai_worker 端自動退回 CONFIG 常數。
                pitch_deg = cam.get_pitch() if hasattr(cam, "get_pitch") else None

                try:
                    ai_in_q.put_nowait({
                        "frame": frame,
                        "depth_frame": depth_frame,
                        "wide_frame": wide_frame_for_ai,
                        "frame_id": frame_counter,
                        "w": w, "h": h,
                        "pitch_deg": pitch_deg,
                    })
                except Exception:
                    pass

                # IR 主動/被動切換 (依深度無效像素比例,內含遲滯與冷卻)
                ir_ctrl.update(depth_frame, loop_start)

                # ---- 第三代背景基礎設施層 ----
                # 放在主 process 而非 ai_worker:AprilTag 偵測要吃原始彩色幀,
                # 而分區狀態要同時被 Flask(後台顯示)與觸覺讀取,留在主 process
                # 少一次跨 process 序列化。整段包在 try 裡,壞掉不影響避障。
                if gen3 is not None:
                    try:
                        g = gen3.step(frame, dets=latest_dets,
                                      mode=State.current_mode, now=loop_start)
                        for txt in g["speeches"]:
                            signal_speech(txt)
                        State.gen3_status = g
                    except Exception as e:
                        print(f"⚠️ 第三代管線異常(已跳過本幀): {e}")

                # ---- MQTT 斷線提醒 ----
                # 震動是唯一的即時安全回饋管道,斷線時 publish 不會報錯、
                # 訊息直接進虛空,使用者會誤以為「前方沒有障礙物」。
                # 這是靜默失效,必須主動出聲。
                if hw.offline_duration() > CONFIG.MQTT_OFFLINE_WARN_SEC:
                    if loop_start - last_mqtt_warn > CONFIG.MQTT_OFFLINE_WARN_COOLDOWN:
                        last_mqtt_warn = loop_start
                        signal_speech("震動回饋連線中斷,請注意目前只有語音提示")

                # AI 結果
                got_ai = False
                try:
                    result = ai_out_q.get_nowait()
                    latest_draws = result["draws"]
                    latest_draw_frame_id = result["frame_id"]
                    latest_dets = result.get("dets") or []
                    got_ai = True
                    for txt in result["speech"]:
                        if not handle_speech_signal(txt):
                            signal_speech(txt)
                    for vib in result["vib"]:
                        motor_id, mode_name = vib
                        hw.vibrate(motor_id, mode_name)
                        motor_display_until[motor_id] = (
                            loop_start + CONFIG.MOTOR_DISPLAY_DURATION)
                        # 安全震動優先:信心回饋讓路,避免兩種震動疊在一起
                        if gen3 is not None:
                            gen3.on_safety_alert(loop_start)
                except queue.Empty:
                    pass

                # ---- AI worker 要求 VLM 語意推理 (搜尋查表沒命中) ----
                if (CONFIG.SEARCH_REASON_VLM and got_ai
                        and result.get("reason_request")
                        and wide_cam.available and not State.vlm_busy):
                    latest = wide_cam.get_latest()
                    if latest is not None:
                        _, wide_frame = latest
                        tgt = result["reason_request"]
                        try:
                            vlm_in_q.put_nowait({
                                "task": "reason", "frame": wide_frame,
                                "target": tgt,
                                "target_zh": CONFIG.LABELS_ZH.get(tgt, tgt)})
                            State.vlm_busy = True
                            State.vlm_busy_since = loop_start
                        except queue.Full:
                            pass

                if got_ai:
                    ai_fps_cnt += 1

                # VLM 結果
                try:
                    while True:
                        vlm_result = vlm_out_q.get_nowait()

                        # ---- 預掃結果: 不播語音,解析成標籤推給 ai_worker ----
                        if vlm_result.get("is_prescan"):
                            scene = vlm_result.get("scene_text", "")
                            if scene:
                                tag_cache.update_from_scene(scene)
                                # 把標籤快取推給 ai_worker (供 Search 冷啟動查詢)
                                if ai_cmd_q is not None:
                                    snapshot = {
                                        k: dict(v)
                                        for k, v in tag_cache.tags.items()
                                    }
                                    ai_cmd_q.put({"prescan_tags": snapshot})
                            continue

                        # ---- 語意推理結果: 念出推理語句 + 把方位推回 ai_worker ----
                        if vlm_result.get("is_reason"):
                            rtext = vlm_result.get("text", "")
                            if rtext:
                                signal_speech(rtext)
                            if ai_cmd_q is not None:
                                ai_cmd_q.put({"reason_hint": {
                                    "target": vlm_result.get("target"),
                                    "zone": vlm_result.get("zone")}})
                            if vlm_result.get("is_final"):
                                State.vlm_busy = False
                                State.vlm_busy_since = 0.0
                            continue

                        # ---- 一般 VLM (使用者主動請求) ----
                        text = vlm_result.get("text_chunk", "")
                        is_final = vlm_result.get("is_final", False)
                        if text:
                            signal_speech(text)
                        if is_final:
                            State.vlm_busy = False
                            State.vlm_busy_since = 0.0
                            if "total_time" in vlm_result:
                                print(f"📊 VLM 完成: {vlm_result['total_time']:.2f}s")
                except queue.Empty:
                    pass

                # ---- 背景全物件掃描: 定時把廣角幀推給 ai_worker 跑 YOLO11 ----
                if (CONFIG.SCAN_BG_ENABLED and wide_cam.available
                        and ai_cmd_q is not None
                        and loop_start - last_scan_push > CONFIG.SCAN_BG_INTERVAL):
                    latest = wide_cam.get_latest()
                    if latest is not None:
                        _, wf = latest
                        try:
                            ai_cmd_q.put_nowait({"wide_scan_frame": wf})
                            last_scan_push = loop_start
                        except Exception:
                            pass

                # ---- 背景預掃排程: 低頻,且只在 VLM 不忙時丟任務 ----
                if (CONFIG.PRESCAN_ENABLED
                        and wide_cam.available
                        and not State.vlm_busy
                        and loop_start - last_prescan_push
                            > CONFIG.PRESCAN_INTERVAL):
                    latest = wide_cam.get_latest()
                    if latest is not None:
                        _, wide_frame = latest
                        try:
                            vlm_in_q.put_nowait({"task": "prescan",
                                                 "frame": wide_frame})
                            last_prescan_push = loop_start
                        except queue.Full:
                            pass

                # 馬達指示器過期清理
                active_motors = {m for m, t in motor_display_until.items()
                                 if t > loop_start}
                motor_display_until = {m: t for m, t in motor_display_until.items()
                                       if t > loop_start}

                if CONFIG.SHOW_DEBUG_WINDOW:
                    base_frame = frame
                    if latest_draw_frame_id in frame_cache:
                        base_frame = frame_cache[latest_draw_frame_id][0]
                    draw_frame = base_frame.copy()
                    # 馬達分區背景 (最底層):7 顆馬達各自對應的畫面水平區間,震動中轉紅
                    if (getattr(CONFIG, "SHOW_MOTOR_ZONES", True)
                            and State.current_mode in ("daily", "search")):
                        draw_motor_zones(draw_frame, active_motors,
                                         alpha=getattr(CONFIG, "MOTOR_ZONE_ALPHA", 0.16))
                    draw_overlay_items(draw_frame, latest_draws)
                    draw_motor_indicator(draw_frame, active_motors)
                    draw_hud(draw_frame, w, h, fps_main, fps_ai,
                             State.current_mode, State.target_object,
                             State.vlm_busy, hw, cam.use_oak)
                else:
                    draw_frame = None

            if CONFIG.SHOW_DEBUG_WINDOW and draw_frame is not None:
                try:
                    cv2.imshow("Blind Assist Debug View", draw_frame)
                    if CONFIG.SHOW_DEPTH_WINDOW and depth_frame is not None:
                        depth_vis = make_depth_visualization(depth_frame)
                        if depth_vis is not None:
                            cv2.imshow("OAK Depth", depth_vis)
                    # 廣角鏡頭原始畫面 (紅綠燈主鏡頭,最頭上那顆)
                    if getattr(CONFIG, "SHOW_WIDE_WINDOW", False) and wide_cam.available:
                        latest_wide = wide_cam.get_latest()
                        if latest_wide is not None:
                            _, wide_vis = latest_wide
                            cv2.imshow("Wide Cam (traffic)", wide_vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except Exception:
                    pass

            now = time.time()

            main_fps_cnt += 1
            if now - main_fps_ts > 1.0:
                fps_main = main_fps_cnt / (now - main_fps_ts)
                main_fps_cnt = 0; main_fps_ts = now
            if now - ai_fps_ts > 1.0:
                fps_ai = ai_fps_cnt / (now - ai_fps_ts)
                ai_fps_cnt = 0; ai_fps_ts = now

            elapsed = time.time() - loop_start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("🛑 使用者中斷")
    finally:
        try: cv2.destroyAllWindows()
        except Exception: pass
        if cam: cam.stop()
        if wide_cam: wide_cam.stop()
        if hw:  hw.shutdown()
        if ai_proc.is_alive(): ai_proc.terminate()
        if vlm_proc.is_alive():
            try:
                vlm_cmd_q.put({"action": "quit"})
                vlm_proc.join(timeout=3)
            except Exception:
                pass
            if vlm_proc.is_alive():
                vlm_proc.terminate()
        print("👋 程式已關閉")


if __name__ == "__main__":
    main()