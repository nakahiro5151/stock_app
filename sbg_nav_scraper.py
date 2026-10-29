# -*- coding: utf-8 -*-
"""
ソフトバンクグループ（9984）公式SOTPページから最新NAV情報を取得する。
ソース: https://group.softbank/ir/stock/sotp（静的HTML。requests + 正規表現で取得可）

NAV = 保有株式 − 純負債（SBG定義, 単位: 兆円）
ディスカウント率 = (NAV − 時価総額) / NAV × 100
1日キャッシュ。
"""
import os
import re
import json
import time

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sbg_nav_cache")
_URL = "https://group.softbank/ir/stock/sotp"
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120 Safari/537.36"}


def _clean_text(html):
    t = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    t = re.sub(r"<style[\s\S]*?</style>", " ", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.replace("&nbsp;", " ").replace("&yen;", "円")
    t = re.sub(r"[ \t\r\n]+", " ", t)
    return t


def fetch_sbg_nav(cache_dir=_CACHE_DIR, cache_hours=24, force=False):
    """SBG公式SOTPページからNAV情報を取得（1日キャッシュ）。

    Returns dict: nav, holdings, net_debt（兆円）, nav_per_share（円）,
      price（円）, shares_mn, usdjpy, ltv（%）, as_of, source, fetched_at,
      from_cache, error。
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "sbg_nav.json")

    if not force and os.path.exists(cache_path):
        if (time.time() - os.path.getmtime(cache_path)) / 3600 <= cache_hours:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                d["from_cache"] = True
                return d
            except Exception:
                pass

    try:
        import requests
        r = requests.get(_URL, headers=_UA, timeout=20)
        if r.status_code != 200:
            return {"error": f"取得失敗 status={r.status_code}"}
        r.encoding = r.apparent_encoding or "utf-8"
        text = _clean_text(r.text)
    except Exception as e:
        return {"error": f"取得エラー: {e}"}

    def _num(pat, src=text, grp=1):
        m = re.search(pat, src)
        return float(m.group(grp).replace(",", "")) if m else None

    # 1株当たりNAV（例: 12,686 円/株）
    nav_per_share = _num(r"1\s*株当たり\s*NAV[^0-9]{0,40}?([\d,]+)\s*円\s*/?\s*株")
    if nav_per_share is None:
        nav_per_share = _num(r"([\d,]+)\s*円\s*/\s*株")

    # 「NAVの算出」セクション以降の『XX.XX 兆円』を順に: NAV, 保有株式, 純負債
    algo = text.find("算出")
    seg = text[algo:] if algo >= 0 else text
    cho = re.findall(r"([\d,]+(?:\.\d+)?)\s*兆円", seg)
    nav = float(cho[0].replace(",", "")) if len(cho) >= 1 else None
    holdings = float(cho[1].replace(",", "")) if len(cho) >= 2 else None
    net_debt = float(cho[2].replace(",", "")) if len(cho) >= 3 else None

    price = _num(r"当社株価[^0-9]{0,30}?([\d,]+)\s*円")
    ltv = _num(r"LTV[^0-9]{0,80}?([\d,]+(?:\.\d+)?)\s*%")
    usdjpy = _num(r"米ドル\s*[／/]\s*円[^0-9]{0,20}?([\d,]+(?:\.\d+)?)")
    shares_mn = _num(r"発行済株式総数（自己株式を除く）（百万株）[^0-9]{0,20}?([\d,]+(?:\.\d+)?)")

    m_as_of = re.search(r"(20\d{2}年\d{1,2}月末)", text)
    as_of = m_as_of.group(1) if m_as_of else None

    # NAV未取得なら 保有株式−純負債 で補完
    if nav is None and holdings is not None and net_debt is not None:
        nav = round(holdings - net_debt, 2)

    if nav is None and nav_per_share is None:
        return {"error": "ページからNAV数値を抽出できませんでした（サイト構造変更の可能性）。"}

    data = {
        "nav": nav, "holdings": holdings, "net_debt": net_debt,
        "nav_per_share": nav_per_share,
        "price": price, "shares_mn": shares_mn, "usdjpy": usdjpy, "ltv": ltv,
        "as_of": as_of, "source": _URL,
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"), "from_cache": False,
        "error": None,
    }
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return data


if __name__ == "__main__":
    print(json.dumps(fetch_sbg_nav(force=True), ensure_ascii=False, indent=2))
