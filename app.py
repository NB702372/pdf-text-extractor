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


def page_has_text(page):
    return bool(page.get_text("text").strip())


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
    page_num = st.number_input(
        "ページ番号", min_value=1, max_value=num_pages, value=1, step=1
    )
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

page = doc[page_num - 1]

mat = fitz.Matrix(ZOOM, ZOOM)
pix = page.get_pixmap(matrix=mat)
page_img = Image.open(io.BytesIO(pix.tobytes("png")))


def extract_text(clip=None, image=None):
    """clip: fitz.Rect (PDF座標), image: PIL Image (clip適用済み)。"""
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
        if image is None:
            image = page_img
        reader = get_ocr_reader()
        text = ocr_extract(image, reader, reflow=use_smart)
        used = "ocr"
    return text, used


if mode == "ページ全体":
    extracted_text, used_method = extract_text(image=page_img)

    with col_settings:
        st.subheader("プレビュー")
        st.image(page_img, use_column_width=True)

else:
    scale = min(1.0, MAX_CANVAS_WIDTH / page_img.width)
    canvas_w = int(page_img.width * scale)
    canvas_h = int(page_img.height * scale)
    bg_img = page_img.resize((canvas_w, canvas_h))

    with col_settings:
        st.markdown("**PDFページ上でドラッグして範囲を選択**")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)",
            stroke_width=2,
            stroke_color="#FF6600",
            background_image=bg_img,
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="rect",
            key=f"canvas_p{page_num}",
        )

    extracted_text = ""
    used_method = None
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
        methods = set()
        for pdf_rect, img_rect in rects:
            cropped_img = page_img.crop(img_rect)
            t, m = extract_text(clip=pdf_rect, image=cropped_img)
            if t.strip():
                parts.append(t.rstrip())
                if m:
                    methods.add(m)
        extracted_text = "\n--- 次の範囲 ---\n".join(parts)
        if len(methods) == 1:
            used_method = methods.pop()
        elif methods:
            used_method = "mixed"

with col_result:
    st.subheader("抽出結果")
    if extracted_text:
        if used_method == "ocr":
            st.caption("使用方法: OCR (画像認識)")
        elif used_method == "text":
            st.caption("使用方法: PDFのテキスト抽出")
        elif used_method == "mixed":
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
            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_p{page_num}.txt",
            mime="text/plain",
        )
        st.metric("文字数", len(extracted_text))
    else:
        if mode == "範囲選択 (マウスでドラッグ)":
            st.info("左側のPDFプレビュー上でマウスをドラッグして範囲を指定してください。複数の矩形を選択することもできます。")
        else:
            st.warning("このページからはテキストを抽出できませんでした。抽出方法を『OCR』に切り替えてみてください。")

doc.close()
