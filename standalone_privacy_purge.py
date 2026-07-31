"""
standalone_privacy_purge.py — 記憶保留期限清除(獨立工具)
==========================================================
公開介面(對外承諾):
    purge_expired(retention_days=30, dry_run=False, now=None) -> dict
不對外開放:清除實作細節、檔案格式。

隱私政策(§九)對應:
  - zone memory 只存「類別 + 座標 + 時間戳」,不存原始影像 → 本工具只需
    處理 JSON,不需處理影像檔;若未來新增影像快取,必須一併納入本檔。
  - 保留期限 30 天滾動式 → 超過 retention_days 的物件實體與分歧記錄刪除。
  - 分區登錄(zones.json / tag_zone_map.json)屬於「安裝設定」而非「行為
    紀錄」,不在清除範圍;但分區的 `last_seen` 造訪時間戳會被抹除,
    避免長期留下「這個人幾號在哪個房間」的行為軌跡(第三人隱私考量)。

執行方式(建議設成每日 cron / systemd timer):
    python standalone_privacy_purge.py --days 30
    python standalone_privacy_purge.py --days 30 --dry-run
"""
import argparse
import os
import time
from datetime import datetime

from shared_store import data_dir, read_json, write_json_atomic


def _purge_objects(retention_days, dry_run, now):
    path = os.path.join(data_dir(), "object_memory.json")
    data = read_json(path, default=None)
    if not data:
        return 0, 0
    cutoff = now - retention_days * 86400.0
    total = sum(len(v) for v in data.values() if isinstance(v, list))
    kept = {}
    for cls, lst in data.items():
        if not isinstance(lst, list):
            continue
        keep = [i for i in lst if float(i.get("ts", 0)) >= cutoff]
        if keep:
            kept[cls] = keep
    removed = total - sum(len(v) for v in kept.values())
    if removed and not dry_run:
        write_json_atomic(path, kept)
    return removed, total


def _purge_logs(retention_days, dry_run, now):
    log_dir = os.path.join(data_dir(), "logs")
    if not os.path.isdir(log_dir):
        return 0
    cutoff = now - retention_days * 86400.0
    removed = 0
    for name in os.listdir(log_dir):
        p = os.path.join(log_dir, name)
        try:
            if os.path.getmtime(p) < cutoff:
                removed += 1
                if not dry_run:
                    os.remove(p)
        except Exception:
            pass
    return removed


def _scrub_zone_visit_times(retention_days, dry_run, now):
    """
    分區保留(是地圖),但把過期的造訪時間戳抹掉(是行為軌跡)。
    造訪「次數」保留,因為它不含時間資訊,對第三人隱私風險低,
    且是分區記憶排序會用到的訊號。
    """
    path = os.path.join(data_dir(), "zones.json")
    data = read_json(path, default=None)
    if not data:
        return 0
    cutoff = now - retention_days * 86400.0
    n = 0
    for zid, z in data.items():
        if float(z.get("last_seen", 0)) and float(z["last_seen"]) < cutoff:
            z.pop("last_seen", None)
            n += 1
    if n and not dry_run:
        write_json_atomic(path, data)
    return n


def purge_expired(retention_days=30, dry_run=False, now=None):
    now = time.time() if now is None else now
    obj_removed, obj_total = _purge_objects(retention_days, dry_run, now)
    logs_removed = _purge_logs(retention_days, dry_run, now)
    zones_scrubbed = _scrub_zone_visit_times(retention_days, dry_run, now)
    result = {
        "ts": now, "iso": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
        "retention_days": retention_days, "dry_run": dry_run,
        "objects_removed": obj_removed, "objects_total_before": obj_total,
        "log_files_removed": logs_removed, "zones_timestamp_scrubbed": zones_scrubbed,
        "data_dir": data_dir(),
    }
    return result


def main():
    ap = argparse.ArgumentParser(description="手護視界 記憶保留期限清除")
    ap.add_argument("--days", type=int, default=30, help="保留天數(預設 30)")
    ap.add_argument("--dry-run", action="store_true", help="只統計不刪除")
    a = ap.parse_args()
    r = purge_expired(a.days, a.dry_run)
    print("🧹 隱私清除結果")
    for k, v in r.items():
        print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
