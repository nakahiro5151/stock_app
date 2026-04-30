import requests
from bs4 import BeautifulSoup
import re

url = "https://www.kabudragon.com/ranking/dekizou.html"
try:
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.encoding = resp.apparent_encoding
    print(f"Status: {resp.status_code}")
    ct = resp.headers.get("Content-Type", "")
    print(f"Content-Type: {ct}")
    print(f"Content length: {len(resp.text)}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # Check all tables
    tables = soup.find_all("table")
    print(f"\nTotal tables found: {len(tables)}")

    for i, table in enumerate(tables):
        trs = table.find_all("tr")
        print(f"\nTable {i}: {len(trs)} rows")
        if len(trs) >= 2:
            for j, tr in enumerate(trs[:5]):
                tds = tr.find_all("td")
                ths = tr.find_all("th")
                if ths:
                    print(f"  Row {j} (th): {[th.get_text(strip=True) for th in ths[:15]]}")
                if tds:
                    texts = [td.get_text(strip=True) for td in tds]
                    print(f"  Row {j} ({len(tds)} tds): {texts[:15]}")
except Exception as e:
    print(f"Error: {e}")
