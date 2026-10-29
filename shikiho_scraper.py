# -*- coding: utf-8 -*-
"""
会社四季報オンライン スクレイパー（Playwright）

ログイン（2段階→Piano IDフォーム）後、銘柄ページから以下を抽出する:
  - 四季報スコア（総合＋レーダー6項目: 成長性/収益性/安全性/割安度/規模/値上がり）
  - 四季報コメント（【見出し】本文 の複数ブロック）
  - 実績PER（高値平均 / 安値平均）
  - 予想PER・PBR・決算発表予定日 等（取得できるもの）

認証情報は .env の SHIKIHO_EMAIL / SHIKIHO_PASSWORD から読む。
結果は JSON ディスクキャッシュ（既定7日）に保存し、再取得を避ける。

注意: 四季報オンラインは有料会員サービス。利用規約・個人利用の範囲で使用すること。
"""
import os
import re
import json
import time

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".shikiho_cache")
_LOGIN_URL = "https://shikiho.toyokeizai.net/"
_STOCK_URL = "https://shikiho.toyokeizai.net/stocks/{code}"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120 Safari/537.36")


def _normalize_code(ticker):
    return str(ticker).upper().replace(".T", "").strip()[:4]


def _find_piano_frame(page):
    for fr in page.frames:
        if "pa.toyokeizai.net/id" in fr.url:
            return fr
    return None


def _do_login(page, email, password, log=lambda *a: None):
    """ユーザー手順どおり: ヘッダーのログイン→ポップアップ内ログイン→Piano IDフォーム入力。"""
    page.goto(_LOGIN_URL, wait_until="networkidle", timeout=45000)
    page.wait_for_timeout(2500)
    try:
        page.click("button:has-text('ログイン')", timeout=8000)
    except Exception as e:
        log("header login click:", str(e)[:80])
    page.wait_for_timeout(1500)
    try:
        page.click("button.-userPopup, .header-user-status__button", timeout=6000)
    except Exception as e:
        log("popup login click:", str(e)[:80])
    page.wait_for_timeout(5000)

    fr = _find_piano_frame(page)
    if not fr:
        return False
    try:
        fr.fill("input[name='email']", email)
        fr.fill("input[type='password']", password)
        fr.click("button:has-text('ログイン')")
        page.wait_for_timeout(9000)
        return True
    except Exception as e:
        log("piano form:", str(e)[:120])
        return False


def _extract_stock(page, code, log=lambda *a: None):
    """銘柄ページから各データを抽出して dict を返す。"""
    page.goto(_STOCK_URL.format(code=code), wait_until="networkidle", timeout=45000)
    page.wait_for_timeout(6000)

    result = {
        "code": code,
        "url": page.url,
        "title": page.title(),
        "shikiho_score": None,          # 総合スコア（整数 or None）
        "score_items": {},              # レーダー6項目 {項目名: 値}
        "comments": [],                 # [{"headline": "快　走", "body": "..."}]
        "per_high_avg": None,           # 実績PER 高値平均（倍）
        "per_low_avg": None,            # 実績PER 安値平均（倍）
        "forecast_per": [],             # 会社予想PER [{"period":"連27.3","per":4.49}, ...]
        "earnings_date": None,          # 決算発表予定日
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # --- レーダー6項目（dt/dd 構造）---
    try:
        items = {}
        dts = page.query_selector_all("dt")
        for dt in dts:
            name = (dt.inner_text() or "").strip()
            if name in ("成長性", "収益性", "安全性", "割安度", "規模", "値上がり"):
                dd = dt.evaluate_handle("el => el.nextElementSibling")
                val_txt = ""
                try:
                    val_txt = dd.as_element().inner_text().strip()
                except Exception:
                    pass
                m = re.search(r"\d+", val_txt)
                if m:
                    items[name] = int(m.group(0))
        result["score_items"] = items
    except Exception as e:
        log("score_items:", str(e)[:100])

    # --- 総合スコア（「四季報スコア」ラベル近傍の数字）---
    try:
        # ラベルを含む要素から、その親ブロック内の末尾の数字を拾う
        el = page.query_selector("text=四季報スコア")
        if el:
            container = el.evaluate_handle("el => el.closest('div') || el.parentElement")
            txt = ""
            try:
                txt = container.as_element().inner_text()
            except Exception:
                pass
            # 「四季報スコア ... 4」の末尾数字（1〜5）
            nums = re.findall(r"\b([1-5])\b", txt)
            if nums:
                result["shikiho_score"] = int(nums[-1])
    except Exception as e:
        log("shikiho_score:", str(e)[:100])

    body = page.inner_text("body")

    # --- 四季報コメント 【見出し】本文 ---
    # 本文は「誌面アーカイブ … 【快　走】<本文> 【30周年】<本文> … 2026年4集秋号（…発売」
    # の区間に現れる。四季報コメントは「文末が。で終わる十分な長さの説明文」なので、
    # それを満たすブロックのみ採用し、連結事業欄などのノイズ（例: 43 <26.3>）を除外する。
    try:
        comments = []
        for m in re.finditer(r"【\s*([^】]+?)\s*】[\s\t　]*([^【]+)", body):
            head = re.sub(r"\s+", "", m.group(1))
            text = m.group(2).strip()
            # 「2026年4集秋号」など号数表記以降は切る
            text = re.split(r"\d{4}年\d集", text)[0].strip()
            text = text.strip(" \t　/").strip()
            # 四季報コメントらしさ: 20文字以上かつ句点を含む
            if len(text) >= 20 and ("。" in text):
                comments.append({"headline": head, "body": text})
        # 重複除去
        seen = set()
        uniq = []
        for c in comments:
            if c["headline"] not in seen:
                seen.add(c["headline"])
                uniq.append(c)
        result["comments"] = uniq[:4]
    except Exception as e:
        log("comments:", str(e)[:100])

    # --- 実績PER 高値平均 / 安値平均 ---
    try:
        # 「実績PER 高値平均 40.9倍 安値平均 18.8倍」
        m_hi = re.search(r"高値平均\s*([\d,]+\.?\d*)\s*倍", body)
        m_lo = re.search(r"安値平均\s*([\d,]+\.?\d*)\s*倍", body)
        if m_hi:
            result["per_high_avg"] = float(m_hi.group(1).replace(",", ""))
        if m_lo:
            result["per_low_avg"] = float(m_lo.group(1).replace(",", ""))
    except Exception as e:
        log("per:", str(e)[:100])

    # --- 会社予想PER（「予想PER … 連27.3 4.49倍 連28.3 3.50倍 … 実績PER」の区間）---
    try:
        # 「予想PER」から「実績PER」までの区間を切り出す（無ければ本文全体の前半）
        seg = body
        i0 = body.find("予想PER")
        i1 = body.find("実績PER")
        if i0 >= 0 and i1 > i0:
            seg = body[i0:i1]
        # 「連27.3」「◇27.3」「単27.3」等の決算期 + 直後の「4.49倍」ペアを拾う
        fpers = []
        for m in re.finditer(r"([連単◇]?\s*\d{2}\.\d{1,2})\s*([\d,]+\.?\d*)\s*倍", seg):
            period = re.sub(r"\s+", "", m.group(1))
            try:
                per = float(m.group(2).replace(",", ""))
            except ValueError:
                continue
            # 異常値を除外（PERは通常 0〜1000 倍程度）
            if 0 < per < 1000:
                fpers.append({"period": period, "per": per})
        # 重複除去（期で一意化、先に出た方＝今期を優先）
        seen = set()
        uniq = []
        for fp in fpers:
            if fp["period"] not in seen:
                seen.add(fp["period"])
                uniq.append(fp)
        result["forecast_per"] = uniq[:4]
    except Exception as e:
        log("forecast_per:", str(e)[:100])

    # --- 決算発表予定日 ---
    try:
        m = re.search(r"決算発表予定日[:：]?\s*([0-9]{4}/[0-9]{1,2}/[0-9]{1,2})", body)
        if m:
            result["earnings_date"] = m.group(1)
    except Exception as e:
        log("earnings_date:", str(e)[:100])

    return result


def load_cached_shikiho(ticker, cache_dir=_CACHE_DIR, cache_days=None):
    """キャッシュ済みの四季報データを読むだけ（Playwrightを起動しない）。

    cache_days=None なら期限に関係なく読む。数値指定ならその日数以内のみ返す。
    Returns: dict or None。
    """
    code = _normalize_code(ticker)
    cache_path = os.path.join(cache_dir, f"{code}.json")
    if not os.path.exists(cache_path):
        return None
    if cache_days is not None:
        age_days = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age_days > cache_days:
            return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["from_cache"] = True
        # キャッシュの鮮度（日数）を付与
        data["cache_age_days"] = round((time.time() - os.path.getmtime(cache_path)) / 86400, 1)
        return data
    except Exception:
        return None


def fetch_shikiho(ticker, cache_dir=_CACHE_DIR, cache_days=7,
                  headless=True, force=False):
    """四季報オンラインから銘柄データを取得（キャッシュ付き）。

    Returns: dict（_extract_stock の戻り値）or {"error": "..."}。
    """
    code = _normalize_code(ticker)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{code}.json")

    # キャッシュ
    if not force and os.path.exists(cache_path):
        age_days = (time.time() - os.path.getmtime(cache_path)) / 86400
        if age_days <= cache_days:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["from_cache"] = True
                return data
            except Exception:
                pass

    email = os.environ.get("SHIKIHO_EMAIL", "")
    password = os.environ.get("SHIKIHO_PASSWORD", "")
    if not email or not password:
        return {"error": "SHIKIHO_EMAIL / SHIKIHO_PASSWORD が未設定です（.env）。"}

    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return {"error": "playwright が未インストールです。pip install playwright && playwright install chromium"}

    logs = []
    def log(*a):
        logs.append(" ".join(str(x) for x in a))

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            ctx = browser.new_context(user_agent=_UA, locale="ja-JP")
            page = ctx.new_page()
            try:
                ok = _do_login(page, email, password, log)
                if not ok:
                    return {"error": "ログインに失敗しました（フォーム検出不可）。", "logs": logs}
                data = _extract_stock(page, code, log)
            finally:
                browser.close()
    except Exception as e:
        import traceback
        return {"error": f"取得エラー: {e}", "trace": traceback.format_exc()}

    data["from_cache"] = False
    # キャッシュ保存
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return data


def fetch_shikiho_batch(tickers, cache_dir=_CACHE_DIR, cache_days=90,
                        headless=True, force=False, progress_file=None):
    """複数銘柄を1回のブラウザセッション（1ログイン）で巡回取得する。

    四季報は四半期（約90日）更新なので cache_days=90 を既定とする。
    force=False の場合、キャッシュが新しい銘柄はスキップする。
    progress_file を指定すると "done/total code" を逐次書き込む（UI進捗表示用）。

    Returns: dict {code: {"ok": bool, "error": str or None}}
    """
    os.makedirs(cache_dir, exist_ok=True)
    codes = []
    for t in tickers:
        c = _normalize_code(t)
        if c and c not in codes:
            codes.append(c)

    # 取得対象を絞る（キャッシュが新しいものはスキップ）
    todo = []
    for c in codes:
        cache_path = os.path.join(cache_dir, f"{c}.json")
        if (not force) and os.path.exists(cache_path):
            age_days = (time.time() - os.path.getmtime(cache_path)) / 86400
            if age_days <= cache_days:
                continue
        todo.append(c)

    results = {c: {"ok": True, "error": None, "skipped": True} for c in codes}
    for c in todo:
        results[c] = {"ok": False, "error": None, "skipped": False}

    if not todo:
        return results

    email = os.environ.get("SHIKIHO_EMAIL", "")
    password = os.environ.get("SHIKIHO_PASSWORD", "")
    if not email or not password:
        for c in todo:
            results[c] = {"ok": False, "error": "認証情報未設定", "skipped": False}
        return results

    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        for c in todo:
            results[c] = {"ok": False, "error": "playwright未インストール", "skipped": False}
        return results

    logs = []
    def log(*a):
        logs.append(" ".join(str(x) for x in a))

    def _write_progress(done, total, code):
        if progress_file:
            try:
                with open(progress_file, "w", encoding="utf-8") as f:
                    f.write(f"{done}/{total} {code}")
            except Exception:
                pass

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            ctx = browser.new_context(user_agent=_UA, locale="ja-JP")
            page = ctx.new_page()
            try:
                ok = _do_login(page, email, password, log)
                if not ok:
                    for c in todo:
                        results[c] = {"ok": False, "error": "ログイン失敗", "skipped": False}
                    return results
                total = len(todo)
                for i, c in enumerate(todo, 1):
                    _write_progress(i, total, c)
                    try:
                        data = _extract_stock(page, c, log)
                        data["from_cache"] = False
                        with open(os.path.join(cache_dir, f"{c}.json"), "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        results[c] = {"ok": True, "error": None, "skipped": False}
                    except Exception as e:
                        results[c] = {"ok": False, "error": str(e)[:120], "skipped": False}
                    page.wait_for_timeout(800)  # サイト負荷軽減
            finally:
                browser.close()
    except Exception as e:
        for c in todo:
            if not results[c]["ok"]:
                results[c] = {"ok": False, "error": f"セッションエラー: {str(e)[:100]}", "skipped": False}
    return results


def fetch_shikiho_subprocess(ticker, cache_dir=_CACHE_DIR, cache_days=7,
                             force=False, timeout=120):
    """Playwrightを別プロセスで実行して四季報データを取得する。

    Streamlit等の asyncio イベントループ内から sync_playwright を直接呼ぶと
    "Sync API inside the asyncio loop" エラーになるため、本関数で回避する。

    1) まずキャッシュを確認（force=False時）。
    2) 無ければ `python shikiho_scraper.py <code> --json` を subprocess 実行し、
       生成されたキャッシュJSONを読み込んで返す。
    Returns: dict。
    """
    import subprocess
    import sys as _sys

    code = _normalize_code(ticker)
    cache_path = os.path.join(cache_dir, f"{code}.json")

    # キャッシュ確認
    if not force:
        cached = load_cached_shikiho(ticker, cache_dir=cache_dir, cache_days=cache_days)
        if cached is not None:
            return cached

    # 別プロセスで取得（この .py を CLI として実行）
    script = os.path.abspath(__file__)
    try:
        proc = subprocess.run(
            [_sys.executable, script, code, "--fetch"],
            capture_output=True, text=True, timeout=timeout,
            cwd=os.path.dirname(script),
        )
    except subprocess.TimeoutExpired:
        return {"error": f"取得がタイムアウトしました（{timeout}秒）。"}
    except Exception as e:
        return {"error": f"サブプロセス起動エラー: {e}"}

    # 生成されたキャッシュを読む
    data = load_cached_shikiho(ticker, cache_dir=cache_dir, cache_days=None)
    if data is not None:
        return data

    # キャッシュが無い＝失敗。サブプロセスのstderr末尾を返す
    err_tail = (proc.stderr or proc.stdout or "").strip()[-400:]
    return {"error": f"取得に失敗しました。{err_tail}" if err_tail else "取得に失敗しました（詳細不明）。"}


def fetch_shikiho_batch_subprocess(tickers, cache_dir=_CACHE_DIR, cache_days=90,
                                   force=False, timeout=900, progress_file=None):
    """複数銘柄を1プロセス・1ログインで一括取得する（subprocess経由）。

    Streamlitのasyncioループ回避のため別プロセスで実行。
    Returns: dict {code: {"ok": bool, "error": str}}（CLIのJSON出力）。
    """
    import subprocess
    import sys as _sys

    codes = []
    for t in tickers:
        c = _normalize_code(t)
        if c and c not in codes:
            codes.append(c)
    if not codes:
        return {}

    script = os.path.abspath(__file__)
    args = [_sys.executable, script, ",".join(codes), "--batch", "--days", str(cache_days)]
    if force:
        args.append("--force")
    if progress_file:
        args += ["--progress", progress_file]
    try:
        proc = subprocess.run(args, capture_output=True, text=True,
                              timeout=timeout, cwd=os.path.dirname(script))
    except subprocess.TimeoutExpired:
        return {"error": f"一括取得がタイムアウトしました（{timeout}秒）。"}
    except Exception as e:
        return {"error": f"サブプロセス起動エラー: {e}"}

    # CLIのstdout（JSON）を解析。失敗時もキャッシュは個別に保存済みなので結果は後で読める。
    try:
        out = (proc.stdout or "").strip()
        # 最後のJSON行を拾う
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
    except Exception:
        pass
    err_tail = (proc.stderr or proc.stdout or "").strip()[-400:]
    return {"error": f"一括取得の結果を解析できませんでした。{err_tail}"}


if __name__ == "__main__":
    import sys
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    arg1 = sys.argv[1] if len(sys.argv) > 1 else "8136"
    _force = ("--fetch" in sys.argv) or ("--force" in sys.argv)

    if "--batch" in sys.argv:
        # 一括取得: arg1 はカンマ区切りの銘柄リスト。
        #   --days N で期限（既定90日）。--progress FILE で進捗ファイル。
        _days = 90
        _prog = None
        if "--days" in sys.argv:
            try:
                _days = int(sys.argv[sys.argv.index("--days") + 1])
            except Exception:
                pass
        if "--progress" in sys.argv:
            try:
                _prog = sys.argv[sys.argv.index("--progress") + 1]
            except Exception:
                pass
        _tickers = [t for t in arg1.split(",") if t.strip()]
        res = fetch_shikiho_batch(_tickers, cache_days=_days, force=_force,
                                  headless=True, progress_file=_prog)
        try:
            print(json.dumps(res, ensure_ascii=False))
        except Exception:
            print(json.dumps({"error": "出力エンコードエラー"}))
    else:
        # 単体取得
        r = fetch_shikiho(arg1, force=_force, headless=True)
        try:
            print(json.dumps(r, ensure_ascii=False))
        except Exception:
            print(json.dumps({"error": "出力エンコードエラー"}))
