import io

import fitz
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

ZOOM = 2.0
MAX_CANVAS_WIDTH = 900


def _reflow_items(items):
    """bbox位置情報を持つテキスト断片をリーディング順に並べ直す共通ロジック。"""
    if not items:
        return ""
    items.sort(key=lambda it: (it["yc"], it["x0"]))
    heights = sorted(it["h"] for it in items if it["h"] > 0)
    median_h = heights[len(heights) // 2] if heights else 10.0
    y_tol = max(2.0, median_h * 0.5)
    groups = []
    for it in items:
        if groups and abs(it["yc"] - groups[-1]["yc"]) <= y_tol:
            g = groups[-1]
            g["members"].append(it)
            g["yc"] = sum(m["yc"] for m in g["members"]) / len(g["members"])
        else:
            groups.append({"yc": it["yc"], "members": [it]})
    out_lines = []
    gap_threshold = median_h * 0.8
    for g in groups:
        members = sorted(g["members"], key=lambda m: m["x0"])
        line_text = ""
        prev_x1 = None
        for m in members:
            if prev_x1 is not None:
                gap = m["x0"] - prev_x1
                if gap > gap_threshold and line_text and not line_text.endswith(" "):
                    line_text += "　"
            line_text += m["text"]
            prev_x1 = m["x1"]
        out_lines.append(line_text.rstrip())
    return "\n".join(out_lines)


def smart_extract(page, clip=None):
    """PDFのテキスト情報をbbox単位で取得し、視覚的に同じ行にあるものを結合する。"""
    kwargs = {"clip": clip} if clip else {}
    d = page.get_text("dict", **kwargs)
    items = []
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(s.get("text", "") for s in line.get("spans", []))
            if not text.strip():
                continue
            x0, y0, x1, y1 = line["bbox"]
            items.append(
                {"text": text, "x0": x0, "x1": x1, "yc": (y0 + y1) / 2, "h": y1 - y0}
            )
    return _reflow_items(items)


def ocr_extract(image, reader, reflow=True):
    """PIL画像をRapidOCR(日本語モデル)に通し、検出結果を行ごとに整形する。"""
    arr = np.array(image.convert("RGB"))
    result = reader(arr)
    if not result or not getattr(result, "txts", None):
        return ""
    items = []
    for box, text, conf in zip(result.boxes, result.txts, result.scores):
        if conf is None or conf < 0.3:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        items.append(
            {"text": text, "x0": x0, "x1": x1, "yc": (y0 + y1) / 2, "h": y1 - y0}
        )
    if reflow:
        return _reflow_items(items)
    items.sort(key=lambda it: (it["yc"], it["x0"]))
    return "\n".join(it["text"] for it in items)


@st.cache_resource(show_spinner="OCRモデル(日本語)を読み込み中... (初回はモデルをダウンロード)")
def get_ocr_reader():
    from rapidocr import RapidOCR, LangRec

    return RapidOCR(params={"Rec.lang_type": LangRec.JAPAN})


def parse_pages(spec: str, num_pages: int) -> list[int]:
    """'1-3, 5, 7-9' のような指定文字列を 0始まりのページindexリストに変換する。
    'all' または '*' は全ページ。
    """
    spec = (spec or "").strip().lower()
    if not spec or spec in ("all", "*", "全て", "すべて"):
        return list(range(num_pages))
    result = set()
    for part in spec.replace("、", ",").replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part or "~" in part or "〜" in part:
            tokens = part.replace("~", "-").replace("〜", "-").split("-", 1)
            try:
                start = int(tokens[0])
                end = int(tokens[1])
            except (ValueError, IndexError):
                raise ValueError(f"範囲指定が不正です: '{part}'")
            if start < 1 or end > num_pages or start > end:
                raise ValueError(
                    f"範囲 '{part}' は 1〜{num_pages} の間で start ≦ end になるよう指定してください"
                )
            for p in range(start, end + 1):
                result.add(p - 1)
        else:
            try:
                p = int(part)
            except ValueError:
                raise ValueError(f"数値が不正です: '{part}'")
            if p < 1 or p > num_pages:
                raise ValueError(f"ページ {p} は 1〜{num_pages} の範囲外です")
            result.add(p - 1)
    return sorted(result)


def render_page_image(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
    return Image.open(io.BytesIO(pix.tobytes("png")))


st.set_page_config(page_title="PDF テキスト抽出ツール", layout="wide")
st.title("PDF テキスト抽出ツール")
st.caption("PDFをアップロードして、ページ全体または範囲選択でテキストを抽出します。スキャン画像PDFはOCRで読み取り可能。")

uploaded_file = st.file_uploader("PDFファイルを選択", type=["pdf"])

if not uploaded_file:
    st.info("上のボタンからPDFファイルをアップロードしてください。")
    st.stop()

pdf_bytes = uploaded_file.read()
doc = fitz.open(stream=pdf_bytes, filetype="pdf")
num_pages = len(doc)

col_settings, col_result = st.columns([1.3, 1])

with col_settings:
    st.subheader("設定")
    page_spec = st.text_input(
        f"ページ範囲 (全{num_pages}ページ)",
        value="1",
        help="例: 1 / 1-3 / 1,3,5 / 1-3,5,7-9 / all (全ページ)",
        placeholder="1-3, 5, 7-9",
    )
    try:
        page_indices = parse_pages(page_spec, num_pages)
    except ValueError as e:
        st.error(f"ページ指定エラー: {e}")
        st.stop()
    if not page_indices:
        st.warning("ページを1つ以上指定してください。")
        st.stop()
    if len(page_indices) > 1:
        st.caption(f"対象: {len(page_indices)} ページ (ページ {', '.join(str(i + 1) for i in page_indices)})")

    mode = st.radio(
        "抽出モード",
        ["ページ全体", "範囲選択 (マウスでドラッグ)"],
        horizontal=True,
    )
    extract_method = st.radio(
        "抽出方法",
        ["自動 (推奨)", "テキスト抽出のみ", "OCR (スキャン画像PDF用)"],
        help="自動: PDFのテキスト情報を試して、無ければOCR / OCR: 画像認識で読み取り",
        horizontal=True,
    )
    use_smart = st.checkbox(
        "横並びの文字を1行にまとめる (推奨)",
        value=True,
        help="字間が広く取られた『春 学 期』のような表記を『春学期』として1行に結合します。",
    )


def extract_for(page, image, clip=None):
    """指定ページについてテキスト抽出を行い、(text, used_method) を返す。"""
    used = None
    text = ""
    if extract_method in ("自動 (推奨)", "テキスト抽出のみ"):
        if use_smart:
            text = smart_extract(page, clip=clip)
        else:
            text = page.get_text("text", clip=clip) if clip else page.get_text("text")
        used = "text"

    if extract_method == "OCR (スキャン画像PDF用)" or (
        extract_method == "自動 (推奨)" and not text.strip()
    ):
        reader = get_ocr_reader()
        text = ocr_extract(image, reader, reflow=use_smart)
        used = "ocr"
    return text, used


def page_header(idx):
    return f"===== ページ {idx + 1} ====="


# 先頭の対象ページを「プレビュー」用に使う
preview_idx = page_indices[0]
preview_page = doc[preview_idx]
preview_img = render_page_image(preview_page)

extracted_text = ""
used_methods = set()

if mode == "ページ全体":
    parts = []
    progress = None
    if len(page_indices) > 3:
        progress = st.progress(0, text="抽出中...")
    for n, idx in enumerate(page_indices):
        pg = doc[idx]
        img = preview_img if idx == preview_idx else render_page_image(pg)
        t, m = extract_for(pg, img)
        if m:
            used_methods.add(m)
        if t.strip():
            block = t.rstrip()
            if len(page_indices) > 1:
                block = f"{page_header(idx)}\n{block}"
            parts.append(block)
        if progress:
            progress.progress((n + 1) / len(page_indices), text=f"抽出中... {n + 1}/{len(page_indices)}")
    if progress:
        progress.empty()
    extracted_text = "\n\n".join(parts)

    with col_settings:
        st.subheader(f"プレビュー (ページ {preview_idx + 1})")
        if len(page_indices) > 1:
            st.caption("※ プレビューは最初の対象ページのみ表示。抽出は指定した全ページを対象に実行されます。")
        st.image(preview_img, use_column_width=True)

else:
    scale = min(1.0, MAX_CANVAS_WIDTH / preview_img.width)
    canvas_w = int(preview_img.width * scale)
    canvas_h = int(preview_img.height * scale)
    bg_img = preview_img.resize((canvas_w, canvas_h))

    with col_settings:
        st.markdown(f"**プレビュー (ページ {preview_idx + 1}) 上でドラッグして範囲を選択**")
        if len(page_indices) > 1:
            st.caption(f"※ 描いた矩形と同じ座標を、指定した {len(page_indices)} ページ全てに適用します。")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)",
            stroke_width=2,
            stroke_color="#FF6600",
            background_image=bg_img,
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="rect",
            key=f"canvas_p{preview_idx}",
        )

    rects = []
    if canvas_result.json_data and canvas_result.json_data.get("objects"):
        for obj in canvas_result.json_data["objects"]:
            if obj.get("type") != "rect":
                continue
            cx0 = obj["left"]
            cy0 = obj["top"]
            cx1 = cx0 + obj["width"] * obj.get("scaleX", 1)
            cy1 = cy0 + obj["height"] * obj.get("scaleY", 1)
            pdf_rect = fitz.Rect(
                cx0 / scale / ZOOM,
                cy0 / scale / ZOOM,
                cx1 / scale / ZOOM,
                cy1 / scale / ZOOM,
            )
            img_rect = (
                int(cx0 / scale),
                int(cy0 / scale),
                int(cx1 / scale),
                int(cy1 / scale),
            )
            rects.append((pdf_rect, img_rect))

    if rects:
        parts = []
        progress = None
        if len(page_indices) > 3:
            progress = st.progress(0, text="抽出中...")
        for n, idx in enumerate(page_indices):
            pg = doc[idx]
            img = preview_img if idx == preview_idx else render_page_image(pg)
            page_parts = []
            for r_no, (pdf_rect, img_rect) in enumerate(rects, 1):
                cropped = img.crop(img_rect)
                t, m = extract_for(pg, cropped, clip=pdf_rect)
                if m:
                    used_methods.add(m)
                if t.strip():
                    label = f"[矩形{r_no}] " if len(rects) > 1 else ""
                    page_parts.append(f"{label}{t.rstrip()}")
            if page_parts:
                page_block = "\n--- 次の範囲 ---\n".join(page_parts) if len(rects) > 1 else page_parts[0]
                if len(page_indices) > 1:
                    page_block = f"{page_header(idx)}\n{page_block}"
                parts.append(page_block)
            if progress:
                progress.progress((n + 1) / len(page_indices), text=f"抽出中... {n + 1}/{len(page_indices)}")
        if progress:
            progress.empty()
        extracted_text = "\n\n".join(parts)


# 出力方法のキャプション用
if len(used_methods) == 1:
    used_label = used_methods.pop()
elif len(used_methods) > 1:
    used_label = "mixed"
else:
    used_label = None


def build_download_name():
    base = uploaded_file.name.rsplit(".", 1)[0]
    if len(page_indices) == 1:
        return f"{base}_p{page_indices[0] + 1}.txt"
    if page_indices == list(range(page_indices[0], page_indices[-1] + 1)):
        return f"{base}_p{page_indices[0] + 1}-{page_indices[-1] + 1}.txt"
    return f"{base}_pages.txt"


with col_result:
    st.subheader("抽出結果")
    if extracted_text:
        if used_label == "ocr":
            st.caption("使用方法: OCR (画像認識)")
        elif used_label == "text":
            st.caption("使用方法: PDFのテキスト抽出")
        elif used_label == "mixed":
            st.caption("使用方法: テキスト抽出とOCRの併用")
        st.text_area(
            "テキスト",
            value=extracted_text,
            height=600,
            label_visibility="collapsed",
        )
        st.download_button(
            "テキストファイルとしてダウンロード",
            data=extracted_text.encode("utf-8"),
            file_name=build_download_name(),
            mime="text/plain",
        )
        c1, c2 = st.columns(2)
        with c1:
            st.metric("文字数", len(extracted_text))
        with c2:
            st.metric("対象ページ数", len(page_indices))
    else:
        if mode == "範囲選択 (マウスでドラッグ)":
            st.info("左側のPDFプレビュー上でマウスをドラッグして範囲を指定してください。複数の矩形を選択することもできます。")
        else:
            st.warning("このページからはテキストを抽出できませんでした。抽出方法を『OCR』に切り替えてみてください。")

doc.close()
