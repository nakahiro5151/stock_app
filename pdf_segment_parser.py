"""
決算PDF セグメント構造化パーサー
PDFからセグメント別業績を抽出し、構造化データに変換する。
"""
import re
import pandas as pd

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


def extract_pdf_data(pdf_file):
    """PDFからテーブルとテキストを抽出（ページ番号付き）"""
    if not HAS_PDFPLUMBER:
        return None, None, None, "pdfplumberがインストールされていません"

    all_tables = []
    all_text = ""
    page_texts = []  # [(page_num, text), ...]
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
                    page_texts.append((i + 1, text))
        if not all_text.strip() and not all_tables:
            return None, None, None, "テキスト抽出不可（画像型PDF）"
        return all_tables, all_text, page_texts, None
    except Exception as e:
        return None, None, None, str(e)


# セグメント合計・調整用の列（セグメントとして扱わない）
_NON_SEGMENT_COLS = ["計", "合計", "小計", "調整額", "消去", "全社", "四半期連結", "損益計算書",
                     "計上額", "連結", "報告セグメント", "セグメント"]


def _parse_num_cell(cell):
    """テーブルセルから代表数値を1つ取り出す（△▲-はマイナス、複数行セルは先頭=外部顧客売上を優先）。"""
    if cell is None:
        return None
    s = str(cell)
    # 複数行セル（外部/内部が改行区切り）は先頭行＝外部顧客への売上高を採用
    first = s.split("\n")[0].strip()
    # 括弧内の内訳（うちロイヤリティ等）は除去
    first = re.sub(r"[（(][^）)]*[)）]", "", first)
    first = first.replace(",", "").replace("，", "").strip()
    neg = first.startswith("△") or first.startswith("▲") or first.startswith("-")
    first = first.lstrip("△▲-").strip()
    if first in ("", "－", "-", "―"):
        return None
    m = re.match(r"^\d+(?:\.\d+)?$", first)
    if not m:
        return None
    v = float(first)
    return -v if neg else v


def _extract_note_table(table):
    """1つの転置型セグメント注記テーブルから {seg_name: {"revenue":x, "profit":y}} を抽出。

    構造: ヘッダー行にセグメント名（列）、データ行に「外部顧客への売上高/計」と
    「セグメント利益」がある決算短信標準形式。取れなければ {} を返す。
    """
    if not table or len(table) < 3:
        return {}

    # セグメント名の列インデックスを特定（ヘッダー行を上から探索）
    name_row_idx = None
    seg_cols = []  # [(col_idx, name), ...]
    for ridx in range(min(4, len(table))):
        row = table[ridx]
        if not row:
            continue
        cand = []
        for cidx, cell in enumerate(row):
            if not cell:
                continue
            name = str(cell).replace("\n", "").replace(" ", "").strip()
            if not name or len(name) > 20:
                continue
            if any(ng == name or ng in name for ng in _NON_SEGMENT_COLS):
                continue
            # 数値だけの列は名前ではない
            if re.match(r"^[△▲\-\d,，\.]+$", name):
                continue
            cand.append((cidx, name))
        # セグメント名候補が2つ以上ある行をヘッダーとみなす
        if len(cand) >= 2:
            name_row_idx = ridx
            seg_cols = cand
            break

    if not seg_cols:
        return {}

    # データ行から売上高・利益を取得
    revenue_row = None
    profit_row = None
    for row in table:
        if not row:
            continue
        head = str(row[0]).replace("\n", "").replace(" ", "") if row[0] else ""
        # 2列目が見出しの場合もある（サンリオ「日本｜計」形式）ため結合も見る
        head2 = ""
        if len(row) > 1 and row[1]:
            head2 = str(row[1]).replace("\n", "").replace(" ", "")

        joined = head + head2
        # 売上高: 「外部顧客への売上高」を最優先、なければ「計」
        if revenue_row is None and ("外部顧客" in joined or "顧客との契約" in joined):
            revenue_row = row
        # 利益: 「セグメント利益」「セグメント利益又は損失」
        if profit_row is None and ("セグメント利益" in joined or "セグメント損益" in joined):
            profit_row = row

    # 「外部顧客」行が無ければ「計」行を売上として使う
    if revenue_row is None:
        for row in table:
            if not row:
                continue
            head = str(row[0]).replace("\n", "").replace(" ", "") if row[0] else ""
            head2 = str(row[1]).replace("\n", "").replace(" ", "") if len(row) > 1 and row[1] else ""
            if head == "計" or head2 == "計" or head == "合計" or head2 == "合計":
                revenue_row = row
                break

    result = {}
    for cidx, name in seg_cols:
        rev = _parse_num_cell(revenue_row[cidx]) if (revenue_row and cidx < len(revenue_row)) else None
        prof = _parse_num_cell(profit_row[cidx]) if (profit_row and cidx < len(profit_row)) else None
        if rev is None and prof is None:
            continue
        result[name] = {"revenue": rev, "profit": prof}
    return result


def parse_segment_note_tables(all_tables):
    """決算短信の「セグメント情報等の注記」から、報告セグメントの数値を確定的に抽出する。

    転置型テーブル（列=セグメント名）が前期・当期の2つ並ぶ標準形式に対応。
    2テーブルを列名でペアリングし、前年比を計算する。

    Returns: list of segment dict（structure_segments 出力互換、qual_comment は未設定）
    """
    note_maps = []  # 各注記テーブルの {name: {revenue, profit}}
    for table in (all_tables or []):
        flat = " ".join(str(c) for row in table for c in (row or []) if c)
        if "セグメント" not in flat and "報告セグメント" not in flat:
            continue
        m = _extract_note_table(table)
        # セグメントが2つ以上、かつ売上か利益が取れているものを採用
        if len(m) >= 2:
            note_maps.append(m)

    if not note_maps:
        return []

    # 前期・当期のペアリング: 同じセグメント名集合を持つ2テーブルを探す
    prev_map = curr_map = None
    if len(note_maps) >= 2:
        # 通常、先に前期、後に当期が並ぶ
        prev_map, curr_map = note_maps[0], note_maps[1]
        # 名前集合が食い違う場合は、共通名が最大の組を選ぶ
        best = 0
        for a in range(len(note_maps)):
            for b in range(len(note_maps)):
                if a == b:
                    continue
                common = set(note_maps[a]) & set(note_maps[b])
                if len(common) > best:
                    best = len(common)
                    prev_map, curr_map = note_maps[a], note_maps[b]
    else:
        # 1テーブルのみ（当期のみ）
        curr_map = note_maps[0]
        prev_map = {}

    segs = []
    for name, cur in curr_map.items():
        prev = prev_map.get(name, {})
        rev_c = cur.get("revenue")
        rev_p = prev.get("revenue")
        oi_c = cur.get("profit")
        oi_p = prev.get("profit")

        rev_yoy = None
        if rev_c is not None and rev_p not in (None, 0):
            rev_yoy = round((rev_c - rev_p) / abs(rev_p) * 100, 1)
        oi_change = (oi_c - oi_p) if (oi_c is not None and oi_p is not None) else None

        # カテゴリ
        if rev_yoy is not None and oi_change is not None:
            rev_up = rev_yoy >= 0
            oi_up = oi_change > 0
            category = ("増収増益" if rev_up and oi_up else
                        "増収減益" if rev_up and not oi_up else
                        "減収増益" if not rev_up and oi_up else "減収減益")
        else:
            category = "不明"

        segs.append({
            "segment_name": name,
            "revenue_previous": rev_p,
            "revenue_current": rev_c,
            "revenue_growth_rate": rev_yoy,
            "operating_profit_previous": oi_p,
            "operating_profit_current": oi_c,
            "operating_profit_change": oi_change,
            "category": category,
            "qual_comment": None,
        })
    return segs


def parse_segments(all_tables, all_text):
    """テーブル＋テキストからセグメント情報を構造化抽出
    
    決算短信のセグメント表は2パターン:
    A) 行=セグメント名（一般的な縦型テーブル）
    B) 列=セグメント名（決算短信の標準形式 — 転置型）
       ヘッダー行2に「精密部品事業」「機能材料事業」等があり、
       データ行に「外部顧客への売上高」「セグメント利益」等がある
    """
    segments = []

    # --- パターンB: 決算短信の転置型テーブル（列=セグメント）---
    # 前期テーブルと当期テーブルを両方見つけてペアリング
    seg_tables = []  # (seg_names, revenue_dict, profit_dict)
    
    for table in (all_tables or []):
        if not table or len(table) < 3:
            continue
        
        # ヘッダー行でセグメント名を探す
        seg_names = []
        seg_col_indices = []
        for row_idx in range(min(3, len(table))):
            row = table[row_idx]
            if not row:
                continue
            for col_idx, cell in enumerate(row):
                if cell and ("事業" in str(cell) or "セグメント" in str(cell)):
                    name = str(cell).strip().replace("\n", "")
                    if name and len(name) < 30 and name not in seg_names:
                        seg_names.append(name)
                        seg_col_indices.append(col_idx)
        
        if len(seg_names) < 1:
            continue
        
        # データ行から売上高とセグメント利益を探す
        revenue_vals = {}
        profit_vals = {}
        for row in table:
            if not row:
                continue
            first_cell = str(row[0]).strip().replace("\n", "") if row[0] else ""
            
            # 外部顧客への売上高
            if "外部顧客" in first_cell or ("売上高" in first_cell and "セグメント間" not in first_cell):
                for i, seg_name in enumerate(seg_names):
                    col_i = seg_col_indices[i] if i < len(seg_col_indices) else i + 1
                    if col_i < len(row) and row[col_i]:
                        val_str = str(row[col_i]).split("\n")[0].replace(",", "").replace("－", "0").replace("△", "-").strip()
                        try:
                            revenue_vals[seg_name] = float(val_str)
                        except ValueError:
                            pass
            
            # セグメント利益
            if "セグメント利益" in first_cell or "セグメント損益" in first_cell:
                for i, seg_name in enumerate(seg_names):
                    col_i = seg_col_indices[i] if i < len(seg_col_indices) else i + 1
                    if col_i < len(row) and row[col_i]:
                        val_str = str(row[col_i]).replace(",", "").replace("－", "0").replace("△", "-").strip()
                        try:
                            profit_vals[seg_name] = float(val_str)
                        except ValueError:
                            pass
            
            # 「顧客との契約から生じる収益」行も売上として使える
            if "顧客との契約" in first_cell:
                for i, seg_name in enumerate(seg_names):
                    col_i = seg_col_indices[i] if i < len(seg_col_indices) else i + 1
                    if col_i < len(row) and row[col_i] and seg_name not in revenue_vals:
                        val_str = str(row[col_i]).replace(",", "").replace("－", "0").replace("△", "-").strip()
                        try:
                            revenue_vals[seg_name] = float(val_str)
                        except ValueError:
                            pass
        
        if revenue_vals or profit_vals:
            seg_tables.append({"names": seg_names, "revenue": revenue_vals, "profit": profit_vals})
    
    # 前期・当期のペアリング（同じセグメント名を持つ2つのテーブル）
    if len(seg_tables) >= 2:
        prev_t = seg_tables[0]  # 前期（先に出現）
        curr_t = seg_tables[1]  # 当期
        
        for seg_name in curr_t["names"]:
            rev_curr = curr_t["revenue"].get(seg_name)
            rev_prev = prev_t["revenue"].get(seg_name)
            oi_curr = curr_t["profit"].get(seg_name)
            oi_prev = prev_t["profit"].get(seg_name)
            
            if rev_curr is not None or oi_curr is not None:
                segments.append({
                    "name": seg_name,
                    "revenue_previous": rev_prev,
                    "revenue_current": rev_curr,
                    "operating_profit_previous": oi_prev,
                    "operating_profit_current": oi_curr,
                })
    elif len(seg_tables) == 1:
        # 1テーブルのみ（当期のみ）
        t = seg_tables[0]
        for seg_name in t["names"]:
            segments.append({
                "name": seg_name,
                "revenue_previous": None,
                "revenue_current": t["revenue"].get(seg_name),
                "operating_profit_previous": None,
                "operating_profit_current": t["profit"].get(seg_name),
            })

    # --- パターンA: 通常の縦型テーブル（フォールバック） ---
    # 損益計算書・貸借対照表・CF計算書の勘定科目行を「セグメント」と誤検出しないよう、
    # ヘッダーに明確に「セグメント」を含む表のみ対象とし、勘定科目名は除外する。
    if not segments:
        # セグメント名として不適切な勘定科目・財務諸表項目（部分一致で除外）
        _account_ng = [
            "売上原価", "販売費", "一般管理費", "研究開発", "営業利益", "営業損失",
            "経常利益", "経常損失", "純利益", "純損失", "当期純", "税引前", "包括利益",
            "受取利息", "支払利息", "受取配当", "営業外", "特別利益", "特別損失",
            "法人税", "減価償却", "のれん", "資産", "負債", "純資産", "資本",
            "現金", "預金", "売掛金", "買掛金", "棚卸", "たな卸", "前払", "未払",
            "未収", "預り金", "引当金", "剰余金", "株式", "新株予約権", "為替差",
            "キャッシュ", "配当金", "補助金", "還付金", "雑収入", "除却損", "調整額",
            "増減額", "受取額", "支払額", "期首残高", "期末残高", "の受取", "の支払",
            "事業費用", "費用合計", "収益合計", "株式報酬",
        ]

        for table in (all_tables or []):
            if not table or len(table) < 2:
                continue
            header = [str(h).strip().replace("\n", " ") if h else "" for h in table[0]]
            header_str = " ".join(header)
            # 「セグメント」を明示的に含む表のみ（損益計算書等の誤検出を防止）
            if "セグメント" not in header_str:
                continue
            for row in table[1:]:
                if not row:
                    continue
                cells = [str(c).strip().replace("\n", "").replace(",", "").replace("△", "-") if c else "" for c in row]
                seg_name = ""
                numbers = []
                for cell in cells:
                    clean = cell.replace("+", "").replace("%", "").replace("百万円", "").strip()
                    if re.match(r'^[\-]?[\d]+\.?\d*$', clean):
                        try:
                            numbers.append(float(clean))
                        except ValueError:
                            pass
                    elif cell and not seg_name and len(cell) > 1 and cell not in ["合計", "計", "小計", "消去", "調整額"]:
                        seg_name = cell
                # 勘定科目名はセグメントとして採用しない
                if seg_name and any(ng in seg_name for ng in _account_ng):
                    continue
                if seg_name and len(numbers) >= 2:
                    segments.append({"name": seg_name, "numbers": numbers})

    return segments


def _clean_sentence(text):
    """抽出した文を表示用に整形（改行・余分な空白除去、長すぎる場合は要約的にトリム）"""
    if not text:
        return ""
    t = re.sub(r"\s+", "", text)  # 決算短信は分かち書きしないので空白除去
    t = t.replace("(", "（").replace(")", "）")
    return t.strip()


def _split_sentences(text):
    """日本語テキストを句点で文に分割"""
    parts = re.split(r"(?<=。)", text)
    return [p for p in (s.strip() for s in parts) if p]


# セグメント名から末尾の事業種別語を外して照合するための語
_SEG_SUFFIXES = ["セグメント", "事業", "ビジネス", "部門", "関連事業", "関連"]


def _segment_aliases(seg_name):
    """セグメント名の照合用エイリアスを生成。
    例: 'Domestic Business' -> ['Domestic Business', 'Domestic', '国内', '日本']
        '精密部品事業' -> ['精密部品事業', '精密部品']
    """
    if not seg_name:
        return []
    name = seg_name.strip()
    aliases = {name}
    # 末尾の事業種別語を外した語幹も候補に
    for suf in _SEG_SUFFIXES:
        if name.endswith(suf) and len(name) > len(suf):
            aliases.add(name[: -len(suf)])
    # 英語→日本語のよくある対応（決算短信は日本語本文が多いため）
    _en_ja = {
        "domestic": ["国内", "日本"],
        "overseas": ["海外", "国外"],
        "japan": ["日本", "国内"],
        "europe": ["欧州", "ヨーロッパ"],
        "asia": ["アジア"],
        "america": ["米州", "北米", "アメリカ"],
        "north america": ["北米"],
        "china": ["中国"],
    }
    low = name.lower()
    for en, jas in _en_ja.items():
        if en in low:
            aliases.update(jas)
    # 末尾 business/segment を外した英語語幹
    for suf in [" business", " segment", " businesses"]:
        if low.endswith(suf):
            aliases.add(name[: -len(suf)].strip())
    return [a for a in aliases if a and len(a) >= 2]


def _is_noise_sentence(text):
    """財務諸表の数値羅列やページ見出しなど、定性コメントに不適切な文を判定"""
    if not text:
        return True
    # ページ見出し（例: -7-株式会社… 決算短信 貸借対照表 …）
    if re.search(r"-\s*\d+\s*-", text) or "貸借対照表" in text or "損益計算書" in text \
            or "キャッシュ・フロー計算書" in text or "財務諸表" in text or "主な注記" in text:
        return True
    # 数字（および桁区切り）が占める割合が高い＝表の数値列
    digits = len(re.findall(r"[\d,，.．]", text))
    if len(text) > 0 and digits / len(text) > 0.35:
        return True
    return False


def extract_segment_commentary(all_text, segment_names):
    """決算短信本文から、各セグメントの定性的な増減要因コメントを抽出する。

    決算短信の「経営成績に関する説明」等には、
    「日本セグメントにおいては、物販事業及びライセンス事業が好調に推移し…増収増益となりました。」
    のようなセグメント別の説明文が含まれる。セグメント名（エイリアス含む）を含む文を
    本文から探し、その文（＋続きの1文）を要因コメントとして返す。

    Returns: dict {segment_name: comment_text}
    """
    result = {}
    if not all_text or not segment_names:
        return result

    # 本文を文単位に分割（全ページ結合済みテキスト）
    sentences = _split_sentences(all_text)

    # 増減要因を示すキーワード（これを含む文を優先的に採用）
    _reason_kw = ["増収", "減収", "増益", "減益", "好調", "伸長", "拡大", "堅調", "苦戦",
                  "低調", "減少", "増加", "寄与", "牽引", "推移", "改善", "悪化",
                  "需要", "販売", "受注", "円安", "円高", "為替", "コスト", "価格",
                  "新製品", "新規", "採用", "カテゴリ", "ライセンス", "物販"]

    # 全セグメントのエイリアス（次文が別セグメントの話かを判定するため）
    all_aliases = {}
    for sn in segment_names:
        all_aliases[sn] = _segment_aliases(sn)

    _conclusion_kw = ["増収", "減収", "増益", "減益", "となりました", "となった", "となっております"]

    for seg_name in segment_names:
        aliases = all_aliases[seg_name]
        # 他セグメントのエイリアス（次文結合の抑止用）
        other_aliases = []
        for sn, al in all_aliases.items():
            if sn != seg_name:
                other_aliases.extend(al)

        best = None
        for idx, sent in enumerate(sentences):
            # セグメント名（エイリアス）を含み、かつ要因キーワードを含む文
            if any(a in sent for a in aliases):
                has_reason = any(k in sent for k in _reason_kw)
                if not has_reason:
                    continue
                # 財務諸表の数値羅列・ページ見出し等のノイズ文は除外
                if _is_noise_sentence(sent):
                    continue
                combo = sent
                # この文自体に結論（増収増益等）が含まれていなければ、次の文を結合
                sent_has_conclusion = any(k in sent for k in ["増収増益", "増収減益", "減収増益", "減収減益"]) or \
                    (("増益" in sent or "減益" in sent) and ("増収" in sent or "減収" in sent))
                if not sent_has_conclusion and idx + 1 < len(sentences) and len(sent) < 120:
                    nxt = sentences[idx + 1]
                    # 次文が別セグメントの説明を始めていないこと & 結論語を含むこと
                    starts_other_seg = any(nxt.startswith(a) or a in nxt[:12] for a in other_aliases)
                    if (not starts_other_seg) and any(k in nxt for k in _conclusion_kw):
                        combo = sent + nxt
                best = combo
                break
        if best:
            c = _clean_sentence(best)
            # 長すぎる場合はトリム（表示用に最大180文字程度）
            if len(c) > 180:
                c = c[:177] + "…"
            result[seg_name] = c
    return result


# 丸数字（①②…）と全角数字を含む見出しの検出用
_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"


def parse_segments_from_narrative(all_text):
    """決算短信本文の「① 日本：売上高…億円（前年同期比…％増）、営業利益…」形式から
    セグメント（多くは地域別）を抽出し、定性コメント付きで返す。

    対応する見出し形式（日本企業の短信で広く使われる定型）:
      A) 丸数字＋コロン: 「① 日本：売上高294億円（前年同期比20.2％増）、営業利益…」（例: サンリオ）
      B) 角括弧見出し:   「〔光ソリューション〕 …により、当セグメントの売上高は600億円（前年同期比45.6％増）、
                          営業利益は92億円（前年同期比99億円改善）となりました。」（例: 古河電工）
    テーブルパースが不十分な多層/事業別セグメント表を補完する。

    Returns: list of segment dict（structure_segments 出力互換）or []
    """
    if not all_text:
        return []

    def _to_yen(num_str, unit):
        v = float(num_str.replace(",", "").replace("，", ""))
        mult = {"億円": 1e8, "百万円": 1e6, "万円": 1e4, "円": 1.0}.get(unit, 1.0)
        return v * mult

    # --- 見出しの検出（位置とセグメント名） ---
    # A) 丸数字＋コロン: ① 日本：
    heading_pat_circle = re.compile(r"[" + _CIRCLED + r"]\s*([^\s：:、。０-９\d]{1,14})\s*[：:]")
    # B) 角括弧見出し: 〔光ソリューション〕 / 【…】 （行頭想定・独立見出し）
    heading_pat_bracket = re.compile(r"[〔【]\s*([^〔〕【】\n]{1,20})\s*[〕】]")

    # セグメント名として不適切な見出し語（全社総括・財務諸表・注記など）
    _bad_names = ["修正再表示", "調整額", "合計", "全社", "消去", "計", "小計", "単位", "注記",
                  "百万円", "千円", "連結", "貸借", "損益", "計算書", "資産", "負債", "純資産",
                  "キャッシュ", "税", "自", "至", "報告セグメント", "日本基準", "基準", "IFRS",
                  "米国基準", "注", "自己資本", "配当", "見通し", "業績予想"]

    headings = []  # (start_pos, end_pos, name)
    for m in heading_pat_circle.finditer(all_text):
        name = m.group(1).strip()
        if any(ng in name for ng in _bad_names):
            continue
        headings.append((m.start(), m.end(), name))
    for m in heading_pat_bracket.finditer(all_text):
        name = m.group(1).strip()
        if any(ng in name for ng in _bad_names):
            continue
        headings.append((m.start(), m.end(), name))

    if not headings:
        return []

    headings.sort(key=lambda x: x[0])

    # PDF抽出テキストは語中に空白/改行が入ることがあるため、各ブロックは空白除去してから数値照合する。
    # 売上高: 「売上高は600億円（前年同期比45.6％増）」/「売上高294億円（前年同期比20.2％増）」
    rev_pat = re.compile(
        r"売上高は?([\d,，\.]+)(億円|百万円|万円|円)"
        r"[（(]前年同期比([\d\.]+)[％%](増|減)[)）]"
    )
    # 営業利益/営業損失: 「営業利益は92億円（前年同期比99億円改善）」「営業利益は74億円（前年同期比105.2％増）」
    #                    「営業損失は11億円（前年同期比…）」
    oi_pat = re.compile(
        r"営業(利益|損失)は?([△\-]?[\d,，\.]+)(億円|百万円|万円|円)"
        r"(?:[（(]前年同期比(?:([\d\.]+)[％%](増|減)|([△\-]?[\d,，\.]+)(億円|百万円|万円|円)(改善|悪化|増|減))[)）])?"
    )

    segs = []
    seen_names = set()
    for i, (hs, he, name) in enumerate(headings):
        if name in seen_names:
            continue
        block_end = headings[i + 1][0] if i + 1 < len(headings) else min(len(all_text), he + 500)
        block_raw = all_text[he:block_end]
        block = re.sub(r"\s+", "", block_raw)  # 語中の空白/改行を除去して数値照合を安定化

        rev_m = rev_pat.search(block)
        if not rev_m:
            continue  # 売上高が読めない見出しはセグメントとして扱わない
        rev_val = _to_yen(rev_m.group(1), rev_m.group(2))
        rev_rate = float(rev_m.group(3)) * (1 if rev_m.group(4) == "増" else -1)
        rev_prev = rev_val / (1 + rev_rate / 100) if rev_rate != -100 else None

        oi_val = oi_prev = oi_change = None
        oi_m = oi_pat.search(block)
        if oi_m:
            is_loss = oi_m.group(1) == "損失"
            oi_raw = oi_m.group(2).replace("△", "-")
            oi_abs = _to_yen(oi_raw.lstrip("-"), oi_m.group(3))
            oi_val = -oi_abs if (is_loss or oi_raw.startswith("-")) else oi_abs
            if oi_m.group(4):  # ％増減
                oi_rate = float(oi_m.group(4)) * (1 if oi_m.group(5) == "増" else -1)
                oi_prev = oi_val / (1 + oi_rate / 100) if oi_rate != -100 else None
                oi_change = (oi_val - oi_prev) if oi_prev is not None else None
            elif oi_m.group(6):  # 金額での増減（「99億円改善」等）
                amt = _to_yen(oi_m.group(6).replace("△", "-").lstrip("-"), oi_m.group(7))
                direction = oi_m.group(8)
                oi_change = amt if direction in ("改善", "増") else -amt
                oi_prev = oi_val - oi_change

        # --- 定性コメント抽出（空白除去済みblockから）---
        # (a) 要因説明: 見出し直後〜「当セグメントの売上高」の手前（角括弧形式で有効）
        # 要因テキストに含まれてはいけない財務諸表・注記系の語
        _cause_ng = ["資産の部", "負債の部", "純資産の部", "剰余金", "評価差額金", "前連結会計年度末",
                     "貸借対照表", "損益計算書", "キャッシュ・フロー", "自己資本比率", "総資産",
                     "配当", "業績予想", "四半期連結", "財政状態", "現金及び預金"]

        def _valid_cause(txt):
            if not (6 <= len(txt) <= 140):
                return False
            if any(ng in txt for ng in _cause_ng):
                return False
            return any(k in txt for k in ["により", "伸長", "売上増", "需要", "好調", "堅調", "牽引",
                                          "寄与", "拡大", "為替", "銅価", "価格", "採用", "受注",
                                          "連結除外", "継続", "影響", "カテゴリ", "ライセンス",
                                          "物販", "客数", "ブランド", "製品"])

        # ブロックから数値表現（「売上高…円（前年同期比…％増）」「営業利益…円（前年同期比…）」）を
        # 除去し、事業説明の地の文だけを残す。角括弧形式（要因が先頭）／丸数字形式（数値が先頭）双方に対応。
        block_no_num = re.sub(
            r"売上高は?[\d,，\.]+(?:億円|百万円|万円|円)[（(]前年同期比[^）)]*[)）]", "", block)
        block_no_num = re.sub(
            r"営業(?:利益|損失)は?[△\-]?[\d,，\.]+(?:億円|百万円|万円|円)(?:[（(]前年同期比[^）)]*[)）])?", "", block_no_num)
        block_no_num = re.sub(r"(?:と|は)?なりました", "", block_no_num)
        block_no_num = block_no_num.replace("、、", "、").strip("、。 ")

        cause = ""
        cut = re.split(r"当セグメントの?|当社グループの?各事業", block_no_num, maxsplit=1)[0].strip("、。 ")
        if _valid_cause(cut):
            cause = cut

        # (b) 要因が取れない/短い場合は、数値除去済みブロックの文を句点区切りで採用
        if not cause:
            for bs in _split_sentences(block_no_num):
                if _is_noise_sentence(bs):
                    continue
                head = bs.strip("、。 ")
                if _valid_cause(head):
                    cause = head
                    break

        # (c) 数値サマリーを末尾に付与（売上前年比・営業利益）
        summary = f"売上高{'+' if rev_rate >= 0 else ''}{rev_rate:.1f}%"
        if oi_change is not None:
            summary += "・営業増益" if oi_change > 0 else "・営業減益"

        if cause:
            comment = f"{cause}（{summary}）"
        else:
            comment = None  # 数値ベースコメントに委ねる
        if comment and len(comment) > 200:
            comment = comment[:197] + "…"

        rev_up = rev_rate >= 0
        oi_up = (oi_change or 0) > 0 if oi_change is not None else None
        if oi_up is None:
            category = "増収" if rev_up else "減収"
        else:
            category = ("増収増益" if rev_up and oi_up else
                        "増収減益" if rev_up and not oi_up else
                        "減収増益" if not rev_up and oi_up else "減収減益")

        segs.append({
            "segment_name": name,
            "revenue_previous": rev_prev,
            "revenue_current": rev_val,
            "revenue_growth_rate": round(rev_rate, 1),
            "operating_profit_previous": oi_prev,
            "operating_profit_current": oi_val,
            "operating_profit_change": oi_change,
            "category": category,
            "qual_comment": comment or None,
        })
        seen_names.add(name)

    return segs


def _extract_cause_only(comment):
    """本文由来コメントから、末尾の『（売上高…）』サマリーと先頭の小見出し番号を除去し、
    事業要因の地の文だけを返す。"""
    if not comment:
        return None
    c = comment
    # 末尾の「（売上高…）」サマリーを除去
    c = re.sub(r"（売上高[^）]*）\s*$", "", c).strip()
    # 先頭の「１．」「①」などの小見出し番号・記号を除去
    c = re.sub(r"^[0-9０-９" + _CIRCLED + r"]+[．\.、）\)]\s*", "", c)
    c = c.strip("、。 ")
    return c or None


def _note_summary(s):
    """注記表の確定値から数値サマリー文字列を作る（例: 売上高+47.8%・営業増益）。"""
    rev_g = s.get("revenue_growth_rate")
    oi_change = s.get("operating_profit_change")
    parts = []
    if rev_g is not None:
        parts.append(f"売上高{'+' if rev_g >= 0 else ''}{rev_g:.1f}%")
    if oi_change is not None:
        parts.append("営業増益" if oi_change > 0 else ("営業減益" if oi_change < 0 else "営業横ばい"))
    return "・".join(parts) if parts else "―"


def build_numeric_comment(s):
    """本文コメントが取れない場合の数値ベース要因コメントを生成する。
    構造化済みセグメント辞書 s（revenue_growth_rate / operating_profit_change /
    revenue_previous / revenue_current / operating_profit_previous / operating_profit_current）を使う。
    """
    rev_g = s.get("revenue_growth_rate")          # %
    oi_diff = s.get("operating_profit_change")    # 金額
    rev_prev = s.get("revenue_previous")
    rev_curr = s.get("revenue_current")
    oi_prev = s.get("operating_profit_previous")
    oi_curr = s.get("operating_profit_current")

    if rev_g is None and oi_diff is None:
        return None

    # 営業利益率の前年差（pt）
    margin_chg_pt = None
    m_prev = (oi_prev / rev_prev) if (rev_prev and oi_prev is not None and rev_prev != 0) else None
    m_curr = (oi_curr / rev_curr) if (rev_curr and oi_curr is not None and rev_curr != 0) else None
    if m_prev is not None and m_curr is not None:
        margin_chg_pt = (m_curr - m_prev) * 100

    rev_up = rev_g is not None and rev_g > 0
    rev_down = rev_g is not None and rev_g < 0
    oi_up = oi_diff is not None and oi_diff > 0
    oi_down = oi_diff is not None and oi_diff < 0

    if margin_chg_pt is not None and margin_chg_pt >= 0.5:
        margin_word = f"採算改善（利益率+{margin_chg_pt:.1f}pt）"
    elif margin_chg_pt is not None and margin_chg_pt <= -0.5:
        margin_word = f"採算悪化（利益率{margin_chg_pt:.1f}pt）"
    else:
        margin_word = "採算横ばい" if margin_chg_pt is not None else None

    if oi_up:
        if rev_up and margin_chg_pt is not None and margin_chg_pt >= 0.5:
            return f"増収に加え{margin_word}が利益を押し上げ"
        if rev_up and margin_chg_pt is not None and margin_chg_pt <= -0.5:
            return f"増収効果が{margin_word}を吸収して増益"
        if rev_up:
            return "売上増（数量・単価）が利益を牽引"
        if rev_down:
            return f"減収も{margin_word or 'コスト削減'}で増益を確保"
        return "営業利益が増加"

    if oi_down:
        if rev_up and margin_chg_pt is not None and margin_chg_pt <= -0.5:
            return f"増収も{margin_word}（コスト増）が利益を圧迫"
        if rev_up:
            return "増収も費用増が利益を上回り減益"
        if rev_down and margin_chg_pt is not None and margin_chg_pt <= -0.5:
            return f"減収と{margin_word}が重なり減益"
        if rev_down:
            return "減収（数量減）が利益を押し下げ"
        return "営業利益が減少"

    if rev_up:
        return "増収も利益はほぼ横ばい"
    if rev_down:
        return "減収も利益はほぼ横ばい"
    return None


def structure_segments(raw_segments, all_text=None):
    """
    生のセグメントデータを構造化。
    parse_segmentsが返す形式:
    - {"name", "revenue_previous", "revenue_current", "operating_profit_previous", "operating_profit_current"}
    - {"name", "numbers"} (フォールバック用)

    all_text を渡すと、決算短信本文からセグメント別の定性コメントを抽出して付与する。
    """
    seg_names = [seg.get("name", "") for seg in raw_segments]
    commentary = extract_segment_commentary(all_text, seg_names) if all_text else {}

    structured = []
    for seg in raw_segments:
        s = {
            "segment_name": seg["name"],
            "revenue_previous": seg.get("revenue_previous"),
            "revenue_current": seg.get("revenue_current"),
            "revenue_growth_rate": None,
            "operating_profit_previous": seg.get("operating_profit_previous"),
            "operating_profit_current": seg.get("operating_profit_current"),
            "operating_profit_change": None,
            "category": "不明",
            "qual_comment": commentary.get(seg["name"]),
        }

        # numbersベースのフォールバック（パターンA）
        if "numbers" in seg and s["revenue_current"] is None:
            nums = seg["numbers"]
            if len(nums) >= 6:
                s["revenue_previous"] = nums[0]
                s["revenue_current"] = nums[1]
                s["operating_profit_previous"] = nums[3]
                s["operating_profit_current"] = nums[4]
            elif len(nums) >= 4:
                s["revenue_previous"] = nums[0]
                s["revenue_current"] = nums[1]
                s["operating_profit_previous"] = nums[2]
                s["operating_profit_current"] = nums[3]
            elif len(nums) >= 2:
                s["revenue_previous"] = nums[0]
                s["revenue_current"] = nums[1]

        # 成長率を自前で計算
        if s["revenue_previous"] and s["revenue_current"] and s["revenue_previous"] != 0:
            s["revenue_growth_rate"] = round(
                (s["revenue_current"] - s["revenue_previous"]) / abs(s["revenue_previous"]) * 100, 1
            )

        if s["operating_profit_previous"] is not None and s["operating_profit_current"] is not None:
            s["operating_profit_change"] = s["operating_profit_current"] - s["operating_profit_previous"]

        # カテゴリ分類
        if s["revenue_growth_rate"] is not None and s["operating_profit_change"] is not None:
            rev_up = s["revenue_growth_rate"] >= 0
            oi_up = s["operating_profit_change"] > 0
            if rev_up and oi_up:
                s["category"] = "増収増益"
            elif rev_up and not oi_up:
                s["category"] = "増収減益"
            elif not rev_up and oi_up:
                s["category"] = "減収増益"
            else:
                s["category"] = "減収減益"

        # 本文からの定性コメントが取れない場合は数値ベースのコメントをフォールバック
        if not s["qual_comment"]:
            s["qual_comment"] = build_numeric_comment(s)

        structured.append(s)
    return structured


def process_pdf(pdf_file):
    """
    PDFファイルを処理してセグメント構造化データを返す。
    Returns: (structured_segments, error_message)
    """
    tables, text, page_texts, error = extract_pdf_data(pdf_file)
    if error:
        return None, error

    raw = parse_segments(tables, text)
    if not raw:
        return None, "セグメントデータを検出できませんでした"

    structured = structure_segments(raw, all_text=text)
    return structured, None


# =====================================================================
# 受注残・KPI 構造化抽出
# =====================================================================

# 受注関連キーワード
_ORDER_KEYWORDS = ["受注高", "受注残高", "受注残", "受注額", "受注状況", "契約残高", "受注"]
# KPI関連キーワード（企業共通 + 業種別）
_KPI_KEYWORDS = [
    "顧客数", "会員数", "ユーザー数", "就業者数", "稼働人数",
    "店舗数", "拠点数", "契約施設数", "契約社数",
    "ARR", "MRR", "解約率", "客単価", "稼働率",
    "利用件数", "新規契約数", "累計契約数", "導入施設数",
    "支援人数", "登録者数", "アクティブ", "DAU", "MAU",
    "ARPU", "LTV", "CAC", "NRR", "チャーンレート",
    "受注件数", "契約件数", "出荷台数", "販売台数",
    "延べ就業者数", "特定技能", "派遣人数",
]


def extract_orders(all_text, all_tables, filename="", page_texts=None):
    """受注高・受注残高を構造化抽出（ページ番号・ファイル名付き）

    Args:
        all_text: 全ページ結合テキスト
        all_tables: 全テーブルリスト
        filename: PDFファイル名
        page_texts: list of (page_num, text) — ページ番号とテキストのペア

    Returns:
        list of {
            "item": str,
            "values": list of {"period": str, "value": float},
            "context": str,
            "unit": str,
            "source_file": str,
            "source_page": int or None,
        }
    """
    results = []

    # テーブルから受注関連行を探す（年度推移テーブル対応）
    for table in (all_tables or []):
        if not table or len(table) < 2:
            continue
        # ヘッダー行から期間情報を取得
        header = table[0] if table[0] else []
        header_strs = [str(h).strip().replace("\n", " ") if h else "" for h in header]

        for row in table[1:] if len(table) > 1 else []:
            if not row:
                continue
            row_str = " ".join([str(c) for c in row if c])
            for kw in _ORDER_KEYWORDS:
                if kw in row_str:
                    # 期間と数値のペアを構築
                    values_with_period = []
                    for col_idx, cell in enumerate(row):
                        if cell:
                            cell_clean = str(cell).replace(",", "").replace("△", "-").replace("－", "").strip()
                            try:
                                val = float(cell_clean)
                                # 対応するヘッダーを期間名として使用
                                period = header_strs[col_idx] if col_idx < len(header_strs) and header_strs[col_idx] else f"col{col_idx}"
                                values_with_period.append({"period": period, "value": val})
                            except (ValueError, IndexError):
                                pass
                    if values_with_period:
                        unit = "百万円" if "百万" in row_str else ("億円" if "億" in row_str else "千円" if "千円" in row_str else "")
                        results.append({
                            "item": kw,
                            "values": values_with_period,
                            "context": row_str[:120],
                            "unit": unit,
                            "source_file": filename,
                            "source_page": None,
                        })
                    break

    # テキストから受注関連情報を探す（ページ番号付き）
    if page_texts:
        for page_num, page_text in page_texts:
            lines = page_text.split("\n")
            for i, line in enumerate(lines):
                line_s = line.strip()
                for kw in _ORDER_KEYWORDS:
                    if kw in line_s:
                        context = line_s
                        nums_raw = re.findall(r'[△\-]?[\d,]+\.?\d*', line_s)
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            nums_raw += re.findall(r'[△\-]?[\d,]+\.?\d*', next_line)
                            context += " " + next_line

                        values_with_period = []
                        for n in nums_raw:
                            try:
                                val = float(n.replace(",", "").replace("△", "-"))
                                values_with_period.append({"period": "", "value": val})
                            except ValueError:
                                pass

                        if values_with_period:
                            unit = "百万円" if "百万" in context else ("億円" if "億" in context else "")
                            if not any(r["item"] == kw and r["source_page"] == page_num for r in results):
                                results.append({
                                    "item": kw,
                                    "values": values_with_period,
                                    "context": context[:150],
                                    "unit": unit,
                                    "source_file": filename,
                                    "source_page": page_num,
                                })
                        break
    elif all_text:
        # page_textsがない場合のフォールバック
        lines = all_text.split("\n")
        for i, line in enumerate(lines):
            line_s = line.strip()
            for kw in _ORDER_KEYWORDS:
                if kw in line_s:
                    context = line_s
                    nums_raw = re.findall(r'[△\-]?[\d,]+\.?\d*', line_s)
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        nums_raw += re.findall(r'[△\-]?[\d,]+\.?\d*', next_line)
                        context += " " + next_line
                    values_with_period = []
                    for n in nums_raw:
                        try:
                            values_with_period.append({"period": "", "value": float(n.replace(",", "").replace("△", "-"))})
                        except ValueError:
                            pass
                    if values_with_period:
                        unit = "百万円" if "百万" in context else ("億円" if "億" in context else "")
                        if not any(r["item"] == kw and r["values"] == values_with_period for r in results):
                            results.append({
                                "item": kw,
                                "values": values_with_period,
                                "context": context[:150],
                                "unit": unit,
                                "source_file": filename,
                                "source_page": None,
                            })
                    break

    return results


def extract_kpis(all_text, all_tables):
    """KPI指標を構造化抽出

    Returns:
        list of {
            "kpi_name": str,
            "values": list,     # 抽出された数値群
            "context": str,     # 周辺テキスト
            "unit": str,        # "人" / "社" / "件" etc.
        }
    """
    results = []

    # テーブルからKPI行を探す
    for table in (all_tables or []):
        if not table or len(table) < 2:
            continue
        for row in table:
            if not row:
                continue
            row_str = " ".join([str(c) for c in row if c])
            for kw in _KPI_KEYWORDS:
                if kw in row_str:
                    numbers = []
                    for cell in row:
                        if cell:
                            cell_clean = str(cell).replace(",", "").replace("△", "-").strip()
                            try:
                                numbers.append(float(cell_clean))
                            except ValueError:
                                pass
                    if numbers:
                        # 単位推定
                        unit = ""
                        for u in ["人", "社", "件", "店", "施設", "円", "％", "%"]:
                            if u in row_str:
                                unit = u
                                break
                        results.append({
                            "kpi_name": kw,
                            "values": numbers,
                            "context": row_str[:100],
                            "unit": unit,
                        })
                    break

    # テキストからKPIを探す
    if all_text:
        lines = all_text.split("\n")
        for i, line in enumerate(lines):
            line_s = line.strip()
            for kw in _KPI_KEYWORDS:
                if kw in line_s:
                    context = line_s
                    nums_raw = re.findall(r'[△\-]?[\d,]+\.?\d*', line_s)
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        nums_raw += re.findall(r'[△\-]?[\d,]+\.?\d*', next_line)
                        context += " " + next_line

                    numbers = []
                    for n in nums_raw:
                        try:
                            numbers.append(float(n.replace(",", "").replace("△", "-")))
                        except ValueError:
                            pass

                    if numbers:
                        unit = ""
                        for u in ["人", "社", "件", "店", "施設", "円", "％", "%"]:
                            if u in context:
                                unit = u
                                break
                        if not any(r["kpi_name"] == kw and r["values"] == numbers for r in results):
                            results.append({
                                "kpi_name": kw,
                                "values": numbers,
                                "context": context[:150],
                                "unit": unit,
                            })
                    break

    return results


def calc_growth_indicators(orders, kpis, revenue_current=None, revenue_previous=None):
    """成長判定指標を計算"""
    indicators = {
        "order_backlog_ratio": None,
        "order_yoy": None,
        "kpi_growth": [],
    }

    # 受注残高 / 売上高 倍率
    backlog = None
    for o in orders:
        if "残" in o["item"] and o["values"]:
            # 新形式（dict）と旧形式（float）両対応
            if isinstance(o["values"][0], dict):
                backlog = max(v["value"] for v in o["values"])
            else:
                backlog = max(o["values"])
            break
    if backlog and revenue_current and revenue_current > 0:
        indicators["order_backlog_ratio"] = round(backlog / revenue_current, 2)

    # 受注高の前年比
    for o in orders:
        if "受注高" in o["item"] and len(o["values"]) >= 2:
            if isinstance(o["values"][0], dict):
                vals = [v["value"] for v in o["values"]]
            else:
                vals = o["values"]
            if len(vals) >= 2:
                prev = vals[0]
                curr = vals[-1]
                if prev and prev > 0:
                    indicators["order_yoy"] = round((curr - prev) / prev * 100, 1)
            break

    # KPI成長率
    for k in kpis:
        if len(k["values"]) >= 2:
            prev = k["values"][0]
            curr = k["values"][-1]
            if prev and prev > 0:
                yoy = round((curr - prev) / prev * 100, 1)
                indicators["kpi_growth"].append({
                    "name": k["kpi_name"],
                    "previous": prev,
                    "current": curr,
                    "yoy": yoy,
                    "unit": k["unit"],
                })

    return indicators


def process_pdf_full(pdf_file, filename=""):
    """PDFファイルを完全処理（セグメント + 受注残 + KPI）

    Returns:
        {
            "segments": list,
            "orders": list,
            "kpis": list,
            "growth_indicators": dict,
            "error": str or None,
        }
    """
    tables, text, page_texts, error = extract_pdf_data(pdf_file)
    if error:
        return {"segments": None, "orders": [], "kpis": [], "growth_indicators": {}, "error": error}

    # ===== セグメント抽出（段階分離）=====
    # STEP1-2: 数値ソースの優先順位
    #   (1) セグメント情報注記の転置型テーブル（最も確定的）
    #   (2) 本文の「① 地域：/〔事業〕」形式（数値＋要因）
    #   (3) 旧テーブルパーサー（フォールバック）
    note_segs = parse_segment_note_tables(tables)          # (1) 数値確定
    narrative_segs = parse_segments_from_narrative(text)   # (2) 数値＋要因（本文）
    raw_segs = parse_segments(tables, text)
    legacy_segs = structure_segments(raw_segs, all_text=text) if raw_segs else []  # (3)

    if note_segs:
        # 注記表を主データに採用。要因コメントは本文抽出から名前照合で補完。
        narr_by_key = {}
        for ns in narrative_segs:
            for a in _segment_aliases(ns.get("segment_name", "")):
                narr_by_key.setdefault(a, ns)

        for s in note_segs:
            name = s.get("segment_name", "")
            # 本文側の同名（エイリアス一致）セグメントのコメントを流用
            matched = None
            for a in _segment_aliases(name):
                if a in narr_by_key:
                    matched = narr_by_key[a]
                    break
            cause_text = None
            if matched and matched.get("qual_comment"):
                cause_text = _extract_cause_only(matched["qual_comment"])
            if cause_text:
                # 要因テキスト + 注記表の確定値サマリー
                s["qual_comment"] = f"{cause_text}（{_note_summary(s)}）"
            else:
                # 本文が無ければ数値ベースコメント
                s["qual_comment"] = build_numeric_comment(s)
        structured_segs = note_segs
    else:
        # 注記表が取れない場合: 本文抽出（2件以上）→ 旧テーブルの順で採用
        _legacy_names = {s.get("segment_name") for s in legacy_segs}
        _legacy_weak = (len(legacy_segs) <= 1) or _legacy_names.issubset(
            {"報告セグメント", "セグメント", "全社", "合計", "計"}
        )
        if len(narrative_segs) >= 2 and _legacy_weak:
            structured_segs = narrative_segs
        else:
            structured_segs = legacy_segs

    # 受注残（ページ番号・ファイル名付き）
    orders = extract_orders(text, tables, filename=filename, page_texts=page_texts)

    # KPI
    kpis = extract_kpis(text, tables)

    # 成長判定
    revenue_curr = None
    if structured_segs:
        revenue_curr = sum(s["revenue_current"] for s in structured_segs if s.get("revenue_current"))
    growth = calc_growth_indicators(orders, kpis, revenue_curr)

    return {
        "segments": structured_segs,
        "orders": orders,
        "kpis": kpis,
        "growth_indicators": growth,
        "error": None,
    }


# =====================================================================
# TDnet 決算短信PDF 自動取得
# =====================================================================
import os as _os
import io as _io
import time as _time
import json as _json
import datetime as _datetime

_TDNET_BASE = "https://www.release.tdnet.info/inbs/"
_TDNET_LIST_URL = "https://www.release.tdnet.info/inbs/I_list_001_{ymd}.html"
_TDNET_UA = {"User-Agent": "Mozilla/5.0 (fundamentals-dashboard; contact: local use)"}


def _normalize_code(ticker_code):
    """'7014.T' や '70140' を 4桁コード '7014' に正規化"""
    code = str(ticker_code).upper().replace(".T", "").strip()
    # TDnetは末尾0付きの5桁表記（例 45990 = 4599）。5桁で末尾0なら4桁化。
    if len(code) == 5 and code.endswith("0"):
        code4 = code[:4]
    else:
        code4 = code[:4]
    return code4, code


def find_tdnet_kessan_pdf(ticker_code, days_back=45, sleep_sec=0.3, max_pages=50):
    """TDnet適時開示一覧を直近days_back日分さかのぼり、指定銘柄の決算短信PDFのURLを探す。

    Returns: dict or None
        {"code": str, "title": str, "pdf_url": str, "date": "YYYY-MM-DD"}
        最も新しい（直近日付の）決算短信を返す。見つからなければ None。
    """
    import requests
    code4, _ = _normalize_code(ticker_code)

    for back in range(0, days_back + 1):
        d = _datetime.date.today() - _datetime.timedelta(days=back)
        # 土日はTDnet開示なし → スキップ（無駄なリクエスト削減）
        if d.weekday() >= 5:
            continue
        ymd = d.strftime("%Y%m%d")
        url = _TDNET_LIST_URL.format(ymd=ymd)
        try:
            r = requests.get(url, headers=_TDNET_UA, timeout=15)
        except Exception:
            continue
        if r.status_code != 200 or len(r.text) < 500:
            continue
        r.encoding = "utf-8"
        html = r.text
        if "決算短信" not in html:
            continue

        # <tr> 単位で、当該銘柄コード & 決算短信 & PDFリンク を探す
        rows = re.findall(r"<tr.*?</tr>", html, re.S)
        candidates = []
        for tr in rows:
            if "決算短信" not in tr:
                continue
            code_m = re.search(r"<td[^>]*>\s*(\d{4,5})\s*<", tr)
            row_code = code_m.group(1) if code_m else ""
            row_code4 = row_code[:4] if row_code else ""
            if row_code4 != code4:
                continue
            pdf_m = re.search(r'href="([^"]+\.pdf)"', tr)
            title_m = re.search(r'\.pdf"[^>]*>(.*?)</a>', tr, re.S)
            if not pdf_m:
                continue
            pdf = pdf_m.group(1)
            pdf_url = pdf if pdf.startswith("http") else (_TDNET_BASE + pdf)
            title = re.sub(r"<.*?>", "", title_m.group(1)).strip() if title_m else "決算短信"
            candidates.append({
                "code": code4,
                "title": title[:80],
                "pdf_url": pdf_url,
                "date": d.strftime("%Y-%m-%d"),
            })
        if candidates:
            # 訂正版より本体を優先（タイトルに「訂正」を含まないものを優先）
            non_correction = [c for c in candidates if "訂正" not in c["title"]]
            return (non_correction or candidates)[0]
        _time.sleep(sleep_sec)  # サイト負荷軽減

    return None


def fetch_tdnet_kessan_pdf(ticker_code, cache_dir=None, cache_days=7, days_back=45):
    """TDnetから決算短信PDFを取得（ディスクキャッシュ付き）。

    Returns: dict
        {"ok": bool, "pdf_bytes": bytes or None, "meta": dict or None,
         "from_cache": bool, "error": str or None}
    """
    import requests
    code4, _ = _normalize_code(ticker_code)

    meta = None
    pdf_bytes = None
    from_cache = False

    # キャッシュ確認
    if cache_dir:
        _os.makedirs(cache_dir, exist_ok=True)
        meta_path = _os.path.join(cache_dir, f"{code4}_kessan.json")
        pdf_path = _os.path.join(cache_dir, f"{code4}_kessan.pdf")
        if _os.path.exists(meta_path) and _os.path.exists(pdf_path):
            age_days = (_time.time() - _os.path.getmtime(meta_path)) / 86400
            if age_days <= cache_days:
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = _json.load(f)
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    from_cache = True
                    return {"ok": True, "pdf_bytes": pdf_bytes, "meta": meta,
                            "from_cache": True, "error": None}
                except Exception:
                    meta = None
                    pdf_bytes = None

    # TDnet検索
    found = find_tdnet_kessan_pdf(ticker_code, days_back=days_back)
    if not found:
        return {"ok": False, "pdf_bytes": None, "meta": None,
                "from_cache": False,
                "error": "NOT_FOUND"}

    # PDF本体をDL
    try:
        pr = requests.get(found["pdf_url"], headers=_TDNET_UA, timeout=30)
        if pr.status_code != 200 or "pdf" not in (pr.headers.get("Content-Type", "").lower()):
            return {"ok": False, "pdf_bytes": None, "meta": found,
                    "from_cache": False,
                    "error": f"PDFのダウンロードに失敗しました（status={pr.status_code}）。"}
        pdf_bytes = pr.content
        meta = found
    except Exception as e:
        return {"ok": False, "pdf_bytes": None, "meta": found,
                "from_cache": False, "error": f"ダウンロードエラー: {e}"}

    # キャッシュ保存
    if cache_dir and pdf_bytes:
        try:
            with open(_os.path.join(cache_dir, f"{code4}_kessan.json"), "w", encoding="utf-8") as f:
                _json.dump(meta, f, ensure_ascii=False)
            with open(_os.path.join(cache_dir, f"{code4}_kessan.pdf"), "wb") as f:
                f.write(pdf_bytes)
        except Exception:
            pass

    return {"ok": True, "pdf_bytes": pdf_bytes, "meta": meta,
            "from_cache": from_cache, "error": None}


def process_tdnet_kessan(ticker_code, cache_dir=None, cache_days=7, days_back=45):
    """TDnetから決算短信を自動取得し、process_pdf_full と同じ構造で返す。

    Returns: process_pdf_full の戻り値 dict に "tdnet_meta" を付与したもの。
             取得失敗時は {"segments": None, ..., "error": str, "tdnet_meta": None}
    """
    res = fetch_tdnet_kessan_pdf(ticker_code, cache_dir=cache_dir,
                                 cache_days=cache_days, days_back=days_back)
    if not res["ok"]:
        return {"segments": None, "orders": [], "kpis": [],
                "growth_indicators": {}, "error": res["error"], "tdnet_meta": res.get("meta")}

    parsed = process_pdf_full(_io.BytesIO(res["pdf_bytes"]),
                              filename=(res["meta"] or {}).get("title", "決算短信.pdf"))
    parsed["tdnet_meta"] = res["meta"]
    parsed["from_cache"] = res["from_cache"]
    return parsed


# =====================================================================
# kabutan 経由の決算短信PDF 取得（過去分もカバー）
# =====================================================================
_KABUTAN_NEWS_URL = "https://kabutan.jp/stock/news?code={code}&nmode=3"
_KABUTAN_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (fundamentals-dashboard)"}


def find_kabutan_kessan_pdf(ticker_code):
    """kabutanの決算ニュースページ（nmode=3）から、最新の決算短信PDFのURLを特定する。

    kabutanのラッパーページ（/disclosures/pdf/YYYYMMDD/ID/）内に実PDF
    （https://tdnet-pdf.kabutan.jp/YYYYMMDD/ID.pdf）へのリンクがある。

    Returns: dict or None
        {"code": str, "title": str, "pdf_url": str, "date": "YYYY-MM-DD", "source": "kabutan"}
    """
    import requests
    code4, _ = _normalize_code(ticker_code)
    url = _KABUTAN_NEWS_URL.format(code=code4)
    try:
        r = requests.get(url, headers=_KABUTAN_UA, timeout=15)
    except Exception:
        return None
    if r.status_code != 200 or len(r.text) < 500:
        return None
    r.encoding = r.apparent_encoding or "utf-8"
    html = r.text

    # aタグから「決算短信」を含み、説明資料でないリンクを探す（新しい順に並んでいる想定）
    links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.S)
    wrapper = None
    wrapper_title = None
    for href, text in links:
        t = re.sub(r"<.*?>", "", text).strip()
        if "決算短信" in t and "説明資料" not in t and "disclosures/pdf" in href:
            wrapper = href
            wrapper_title = t
            break
    if not wrapper:
        return None

    if wrapper.startswith("/"):
        wrapper = "https://kabutan.jp" + wrapper

    # ラッパーページのURLから日付とIDを抽出（/disclosures/pdf/YYYYMMDD/ID/）
    m = re.search(r"/disclosures/pdf/(\d{8})/([0-9]+)/?", wrapper)
    date_str = None
    real_pdf = None
    if m:
        ymd, doc_id = m.group(1), m.group(2)
        date_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
        # 実PDF URLを規則から直接生成
        real_pdf = f"https://tdnet-pdf.kabutan.jp/{ymd}/{doc_id}.pdf"

    # 念のためラッパーページを取得して実PDF URLを確認（規則が変わっても拾えるように）
    if not real_pdf:
        try:
            wr = requests.get(wrapper, headers=_KABUTAN_UA, timeout=15)
            wr.encoding = wr.apparent_encoding or "utf-8"
            cand = re.search(r'(https?://[^"\']*tdnet-pdf\.kabutan\.jp/[^"\']+\.pdf)', wr.text)
            if not cand:
                cand = re.search(r'(https?://[^"\']+\.pdf)', wr.text)
            if cand:
                real_pdf = cand.group(1)
        except Exception:
            pass

    if not real_pdf:
        return None

    return {
        "code": code4,
        "title": (wrapper_title or "決算短信")[:80],
        "pdf_url": real_pdf,
        "date": date_str or "?",
        "source": "kabutan",
    }


def fetch_kabutan_kessan_pdf(ticker_code, cache_dir=None, cache_days=7):
    """kabutan経由で決算短信PDFを取得（ディスクキャッシュ付き）。

    Returns: dict
        {"ok": bool, "pdf_bytes": bytes or None, "meta": dict or None,
         "from_cache": bool, "error": str or None}
    """
    import requests
    code4, _ = _normalize_code(ticker_code)

    # キャッシュ確認
    if cache_dir:
        _os.makedirs(cache_dir, exist_ok=True)
        meta_path = _os.path.join(cache_dir, f"{code4}_kessan.json")
        pdf_path = _os.path.join(cache_dir, f"{code4}_kessan.pdf")
        if _os.path.exists(meta_path) and _os.path.exists(pdf_path):
            age_days = (_time.time() - _os.path.getmtime(meta_path)) / 86400
            if age_days <= cache_days:
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = _json.load(f)
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    return {"ok": True, "pdf_bytes": pdf_bytes, "meta": meta,
                            "from_cache": True, "error": None}
                except Exception:
                    pass

    found = find_kabutan_kessan_pdf(ticker_code)
    if not found:
        return {"ok": False, "pdf_bytes": None, "meta": None,
                "from_cache": False, "error": "NOT_FOUND"}

    try:
        pr = requests.get(found["pdf_url"], headers=_KABUTAN_UA, timeout=30)
        if pr.status_code != 200 or pr.content[:4] != b"%PDF":
            return {"ok": False, "pdf_bytes": None, "meta": found,
                    "from_cache": False,
                    "error": f"PDFのダウンロードに失敗しました（status={pr.status_code}）。"}
        pdf_bytes = pr.content
    except Exception as e:
        return {"ok": False, "pdf_bytes": None, "meta": found,
                "from_cache": False, "error": f"ダウンロードエラー: {e}"}

    if cache_dir and pdf_bytes:
        try:
            with open(_os.path.join(cache_dir, f"{code4}_kessan.json"), "w", encoding="utf-8") as f:
                _json.dump(found, f, ensure_ascii=False)
            with open(_os.path.join(cache_dir, f"{code4}_kessan.pdf"), "wb") as f:
                f.write(pdf_bytes)
        except Exception:
            pass

    return {"ok": True, "pdf_bytes": pdf_bytes, "meta": found,
            "from_cache": False, "error": None}


def fetch_kessan_pdf_auto(ticker_code, cache_dir=None, cache_days=7, tdnet_days_back=45):
    """決算短信PDFを自動取得。kabutan（過去分も可）を主軸に、失敗時はTDnetをフォールバック。

    Returns: fetch_*_kessan_pdf と同じ dict 形式。
    """
    # 1) kabutan（過去分もカバーできるため優先）
    res = fetch_kabutan_kessan_pdf(ticker_code, cache_dir=cache_dir, cache_days=cache_days)
    if res.get("ok"):
        return res

    # 2) TDnet フォールバック（直近開示のみ）
    res_tdnet = fetch_tdnet_kessan_pdf(ticker_code, cache_dir=cache_dir,
                                       cache_days=cache_days, days_back=tdnet_days_back)
    if res_tdnet.get("ok"):
        return res_tdnet

    # 両方失敗
    return {"ok": False, "pdf_bytes": None, "meta": None,
            "from_cache": False, "error": "NOT_FOUND"}


def process_kessan_auto(ticker_code, cache_dir=None, cache_days=7, tdnet_days_back=45):
    """決算短信を自動取得（kabutan優先→TDnet）し、process_pdf_full と同じ構造で返す。

    Returns: process_pdf_full の戻り値 dict に "tdnet_meta"（取得元メタ）と "from_cache" を付与。
             取得失敗時は {"segments": None, ..., "error": "NOT_FOUND", "tdnet_meta": None}
    """
    res = fetch_kessan_pdf_auto(ticker_code, cache_dir=cache_dir,
                                cache_days=cache_days, tdnet_days_back=tdnet_days_back)
    if not res["ok"]:
        return {"segments": None, "orders": [], "kpis": [],
                "growth_indicators": {}, "error": res["error"], "tdnet_meta": res.get("meta")}

    parsed = process_pdf_full(_io.BytesIO(res["pdf_bytes"]),
                              filename=(res["meta"] or {}).get("title", "決算短信.pdf"))
    parsed["tdnet_meta"] = res["meta"]
    parsed["from_cache"] = res.get("from_cache", False)
    return parsed
