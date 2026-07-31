"""
standalone_rename_migration.py — 既有檔案改名遷移工具(獨立工具)
==================================================================
公開介面:
    plan(root) -> dict            # 只掃描,不動任何檔案
    migrate(root, apply=False)    # apply=False 為預演;True 才真的改
不對外開放:import 改寫的正規表示式細節。

--------------------------------------------------------------------
為什麼需要一支工具而不是手動改
  《技術規劃 v3》2.10 節已經點出風險:19 個檔案互相引用,要逐一改完 import
  陳述式後 `py_compile` + `grep` 交叉驗證,**字串類引用不會在語法檢查時報錯**。
  手動改 19 個檔案 × 平均 4 個 import,漏一個就是執行期才炸。
  這支工具做三件事:
    ① 改名 + 改寫所有 `import X` / `from X import ...` / `import X as Y`
    ② 掃出所有「字串裡提到舊模組名」的地方,列出來給人工判斷(不自動改,
       因為字串可能是註解、log 訊息、或真的是動態匯入,語意差很多)
    ③ 改完自動 `py_compile` 全部檔案,任何一個編不過就報出來

★ 這支工具要在**你自己最新的工作副本**上跑,不要跑在任何舊備份上。
★ 執行前請先 commit 或整個資料夾複製一份;`--apply` 會自動先備份到
  `_backup_rename_<時間戳>/`,但自己再留一份比較安心。

用法:
    python standalone_rename_migration.py            # 預演,只印出會發生什麼
    python standalone_rename_migration.py --apply    # 真的執行
    python standalone_rename_migration.py --root /path/to/project --apply
"""
import argparse
import os
import re
import shutil
import time

# 《技術規劃 v3》2.10 節既定的對照表
RENAME_MAP = {
    # 硬體層:碰硬體的,按硬體歸類
    "camera.py": "hw_camera.py",
    "ir_mode.py": "hw_ir_mode.py",
    "wide_cam.py": "hw_wide_cam.py",
    "hardware.py": "hw_haptic.py",
    "motor_zones.py": "hw_motor_zones.py",
    # 流程層:不碰硬體的,按所屬 process 歸類
    "main.py": "proc_main.py",
    "flask_app.py": "proc_flask_app.py",
    "overlay.py": "proc_overlay.py",
    "ai_worker.py": "proc_ai_worker.py",
    "vlm_worker.py": "proc_vlm_worker.py",
    # 感知層
    "trackers.py": "percept_trackers.py",
    "hands.py": "percept_hands.py",
    "crosswalk.py": "percept_crosswalk.py",
    "stairs_fusion.py": "percept_stairs_fusion.py",
    "light_color.py": "percept_light_color.py",
    # 共用
    "config.py": "shared_config.py",
    "utils.py": "shared_utils.py",
    # 獨立工具 / 已停用
    "scan_debug.py": "standalone_scan_debug.py",
    "stairs.py": "legacy_stairs.py",
}

SCAN_EXT = (".py", ".html", ".sh", ".md", ".txt", ".json", ".service")
SKIP_DIRS = {"__pycache__", ".git", "weights", "node_modules", "gen3_data"}


def _stems():
    return {os.path.splitext(a)[0]: os.path.splitext(b)[0] for a, b in RENAME_MAP.items()}


def _iter_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.startswith("_backup_rename_")]
        for fn in filenames:
            if fn.endswith(SCAN_EXT):
                yield os.path.join(dirpath, fn)


def rewrite_imports(text):
    """
    只改真正的 import 陳述式,不動註解與字串。
    涵蓋三種寫法:
        import X            /  import X as Y   /  import X, Z
        from X import ...
    """
    stems = _stems()
    changed = 0

    def sub_from(m):
        nonlocal changed
        new = stems.get(m.group(2))
        if not new:
            return m.group(0)
        changed += 1
        return f"{m.group(1)}from {new} import"

    text = re.sub(r"(^[ \t]*)from\s+([A-Za-z_][\w]*)\s+import", sub_from, text, flags=re.M)

    def sub_import(m):
        nonlocal changed
        indent, body = m.group(1), m.group(2)
        parts = []
        for piece in body.split(","):
            piece = piece.strip()
            mm = re.match(r"^([A-Za-z_][\w]*)(\s+as\s+[A-Za-z_][\w]*)?$", piece)
            if mm and mm.group(1) in stems:
                changed += 1
                # 沒有 as 別名的話補一個舊名別名,呼叫端 `camera.xxx` 不用全改
                alias = mm.group(2) or f" as {mm.group(1)}"
                parts.append(f"{stems[mm.group(1)]}{alias}")
            else:
                parts.append(piece)
        return f"{indent}import " + ", ".join(parts)

    text = re.sub(r"(^[ \t]*)import\s+([A-Za-z_][\w][^\n#]*)", sub_import, text, flags=re.M)
    return text, changed


def find_string_refs(root):
    """
    找出字串/文字裡提到舊模組名的地方。**不自動改**——這些可能是註解、
    log 訊息、文件、或真的是動態匯入,語意差很多,一律交人工判斷。
    """
    stems = _stems()
    # 只認「<模組名>.py」或「整個字串就是模組名」兩種形態。
    # 不能只用 \b:`search-overlay` 會因為連字號是非字元邊界而誤報 overlay,
    # 這種噪音多到會讓人直接略過整份清單,反而漏掉真的動態匯入。
    names = "|".join(map(re.escape, stems))
    pat = re.compile(r"""["'`](?:[^"'`\n]*?(?<![-_\w])(?:%s)\.py(?![-_\w])[^"'`\n]*?|(?:%s))["'`]"""
                     % (names, names))
    hits = []
    for path in _iter_files(root):
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if pat.search(line):
                hits.append({"file": os.path.relpath(path, root), "line": i,
                             "text": line.strip()[:120]})
    return hits


def plan(root="."):
    present = [f for f in RENAME_MAP if os.path.exists(os.path.join(root, f))]
    missing = [f for f in RENAME_MAP if f not in present]
    return {"root": os.path.abspath(root), "will_rename": present,
            "not_found": missing, "string_refs": find_string_refs(root)}


def migrate(root=".", apply=False):
    root = os.path.abspath(root)
    info = plan(root)
    backup = None

    if apply:
        backup = os.path.join(root, f"_backup_rename_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(backup, exist_ok=True)
        for path in _iter_files(root):
            rel = os.path.relpath(path, root)
            dst = os.path.join(backup, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(path, dst)

    # ---- ① 改寫 import ----
    rewritten = []
    for path in _iter_files(root):
        if not path.endswith(".py"):
            continue
        text = open(path, encoding="utf-8").read()
        new, n = rewrite_imports(text)
        if n:
            rewritten.append({"file": os.path.relpath(path, root), "changes": n})
            if apply:
                open(path, "w", encoding="utf-8").write(new)

    # ---- ② 改名 ----
    renamed = []
    for old, new in RENAME_MAP.items():
        src, dst = os.path.join(root, old), os.path.join(root, new)
        if not os.path.exists(src):
            continue
        renamed.append(f"{old} → {new}")
        if apply:
            if os.path.exists(dst):
                print(f"⚠️ 目標已存在,跳過:{new}")
                continue
            os.rename(src, dst)

    # ---- ③ 驗證 ----
    failed = []
    if apply:
        import py_compile
        for path in _iter_files(root):
            if not path.endswith(".py"):
                continue
            try:
                py_compile.compile(path, doraise=True)
            except Exception as e:
                failed.append({"file": os.path.relpath(path, root), "error": str(e)[:200]})

    info.update({"applied": apply, "backup": backup, "renamed": renamed,
                 "import_rewrites": rewritten, "compile_failures": failed})
    return info


def main():
    ap = argparse.ArgumentParser(description="手護視界 既有檔案改名遷移")
    ap.add_argument("--root", default=".", help="專案根目錄")
    ap.add_argument("--apply", action="store_true", help="真的執行(預設只預演)")
    a = ap.parse_args()

    r = migrate(a.root, a.apply)
    print(f"\n=== 改名遷移({'執行' if a.apply else '預演'})===")
    print(f"根目錄:{r['root']}")
    print(f"\n▍將改名 {len(r['renamed'])} 個檔案:")
    for line in r["renamed"]:
        print(f"   {line}")
    if r["not_found"]:
        print(f"\n▍對照表中不存在的檔案(略過):{', '.join(r['not_found'])}")
    print(f"\n▍import 改寫 {len(r['import_rewrites'])} 個檔案:")
    for x in r["import_rewrites"]:
        print(f"   {x['file']}  ({x['changes']} 處)")
    if r["string_refs"]:
        print(f"\n▍⚠️ 字串裡提到舊模組名的地方共 {len(r['string_refs'])} 處,"
              f"**需人工確認**(工具不會自動改):")
        for h in r["string_refs"][:40]:
            print(f"   {h['file']}:{h['line']}  {h['text']}")
        if len(r["string_refs"]) > 40:
            print(f"   ...(還有 {len(r['string_refs']) - 40} 處)")
    if r["applied"]:
        print(f"\n▍備份:{r['backup']}")
        if r["compile_failures"]:
            print(f"\n❌ 有 {len(r['compile_failures'])} 個檔案編譯失敗:")
            for f in r["compile_failures"]:
                print(f"   {f['file']}: {f['error']}")
        else:
            print("\n✅ 全部檔案 py_compile 通過")
        print("\n下一步:實機啟動改用  python proc_main.py")
    else:
        print("\n(這是預演,沒有任何檔案被更動。確認無誤後加 --apply)")


if __name__ == "__main__":
    main()
