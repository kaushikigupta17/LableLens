import io
import os
import re
import time
import html
import difflib
import calendar
import datetime
import functools
import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image, ImageOps
from deep_translator import GoogleTranslator, MyMemoryTranslator
from gtts import gTTS

try:
    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 4))
except Exception:
    pass

# Words that really appear on labels but are missing from the normal dictionary.
# Adding a word here only stops it being flagged yellow. Misreads stay yellow.
LABEL_WORDS = [
    "cabergoline", "caberlin", "gynaecologist", "gynaecologists", "pharma",
    "mrp", "incl", "uncoated", "excipients", "ltd", "mfg", "tablets",
    "tablet", "schedule", "prescription", "paracetamol", "ibuprofen",
    "palashbari", "kokjhar", "kamrup", "assam", "mirza", "vill", "dist",
    "pharmacist", "dermatologist", "cetaphil", "cetirizine", "azithromycin",
    "amoxicillin", "pantoprazole", "metformin", "atorvastatin", "ayurvedic",
    "sun", "cipla", "zydus", "lupin", "mankind", "dabur", "himalaya",
    "regd", "lic", "taxes", "strip", "strips", "capsules", "syrup",
]
try:
    from spellchecker import SpellChecker
    spell = SpellChecker()
    spell.word_frequency.load_words(LABEL_WORDS)
except Exception:
    spell = None  # word check is skipped if pyspellchecker is missing

st.set_page_config(page_title="LabelLens", page_icon="🔍")
st.title("LabelLens")
st.caption("Scan a label to get readable text.")

DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
BLUR_LIMIT = 60
HIDE_LIMIT = 0.30    # below this, a line is hidden
CHECK_LIMIT = 0.50   # below this, the whole line is yellow (dial this to taste)

MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july",
               "august", "september", "october", "november", "december"]
MONTHS = {}
for i, n in enumerate(MONTH_NAMES, 1):
    MONTHS[n] = i
    MONTHS[n[:3]] = i

MONTH_WORD = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
              r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?")
DATE_RE = re.compile(
    r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})\b"
    r"|\b(\d{1,2})[/\-](\d{4}|\d{2})\b"
    r"|\b(" + MONTH_WORD + r")\b[\s.,\-/]*(\d{4}|\d{2})\b",
    re.I,
)
KEY_RE = re.compile(
    r"(expiry|expiration|exp(?![a-z])\.?|expd(?:t|ate)?|use\s*before|best\s*before|use\s*by)",
    re.I,
)
BATCH_RE = re.compile(r"\bB\.?\s*N[o0]\.?\s*[:.\-]*\s*([A-Z0-9]{4,12})", re.I)
MRP_RE = re.compile(
    r"\bM\.?\s*R\.?\s*P\.?\s*(?:Rs\.?|₹|INR)?\s*[:.\-]*\s*(?:Rs\.?\s*)?(\d[\d,]*(?:\.\d{1,2})?)",
    re.I,
)
MFG_RE = re.compile(
    r"\b(?:Mfg|Mfd)\.?\s*(?:Dt\.?|Date)?\s*[:.\-]*\s*(\d{1,2}\s*[/\-]\s*\d{2,4})", re.I
)


# ---------- OCR ----------
@st.cache_resource(show_spinner="Loading OCR model (first time only)...")
def load_reader(with_hindi: bool):
    return easyocr.Reader(["en", "hi"] if with_hindi else ["en"])


def clean_text(t):
    t = t.translate(DIGITS)
    # the degree sign is often misread: "251C", "25*C", "25'C", "25oC" -> "25°C"
    t = re.sub(r"(?<!\d)(\d{1,2})\s*[1lI|*'\"^ºo˚°]\s*C\b", r"\1°C", t)
    return t


def preprocess(image, use_clahe=True):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if gray.shape[1] < 600:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return gray


@st.cache_data(show_spinner="Reading label...")
def run_ocr(data: bytes, use_clahe: bool, max_side: int, with_hindi: bool):
    reader = load_reader(with_hindi)          # model loading is NOT timed
    t0 = time.time()
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    image.thumbnail((max_side, max_side))
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    processed = preprocess(image, use_clahe)
    raw = reader.readtext(processed, detail=1, batch_size=8)
    results = [(clean_text(t), float(c)) for _, t, c in raw]
    results = [(t, c) for t, c in results if len(t.strip()) >= 3]
    return results, processed, blur, time.time() - t0


@st.cache_data(show_spinner="Looking for rotated text...")
def scan_rotated(data: bytes):
    reader = load_reader(False)
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    image.thumbnail((900, 900))
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    out = []
    for k in (1, 3):  # rotate by 90 and 270 degrees
        rot = np.ascontiguousarray(np.rot90(gray, k=k))
        for _, t, c in reader.readtext(rot, detail=1, batch_size=8):
            out.append((clean_text(t), float(c)))
    return out


# ---------- red warning boxes (light text on a red background) ----------
def find_red_boxes(rgb):
    """Return the (x, y, w, h) of solid red rectangles, top to bottom."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    # OPEN removes thin red/magenta print, CLOSE fills the lettering inside a box
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 45), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask.shape
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        fill = cv2.contourArea(cnt) / float(w * h)
        if (w * h >= 0.004 * H * W and w >= 2 * h and w >= 0.15 * W
                and h <= 0.2 * H and fill >= 0.45):
            # trim rows that are mostly not red, so the crop is just the box
            rows = (mask[y:y + h, x:x + w] > 0).mean(axis=1)
            keep = np.where(rows > 0.5)[0]
            if len(keep):
                y, h = y + int(keep[0]), int(keep[-1] - keep[0] + 1)
            boxes.append((x, y, w, h))
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:3]
    return sorted(boxes, key=lambda b: b[1])


def red_box_variants(crop):
    """Yield cleaned-up versions of the crop. Light text on red is faint in a
    normal grey image, so we use one colour channel plus local contrast."""
    for ch in (2, 1):                              # blue channel first, then green
        a = cv2.resize(crop[:, :, ch], None, fx=2, fy=2,
                       interpolation=cv2.INTER_CUBIC).astype(np.float32)
        a = cv2.GaussianBlur(a, (0, 0), 1.0)
        a = a - cv2.GaussianBlur(a, (0, 0), 12)    # text vs the foil right next to it
        lo, hi = np.percentile(a, 2), np.percentile(a, 98)
        a = np.clip((a - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
        yield a
        yield 255 - a


def read_red_box(reader, crop):
    best, best_score = [], 0.0
    for img in red_box_variants(crop):
        lines = []
        for _, t, c in reader.readtext(img, detail=1, batch_size=8):
            t = clean_text(t)
            if len(t.strip()) >= 3 and c >= HIDE_LIMIT:
                lines.append((t, float(c)))
        score = sum(len(t) * c for t, c in lines)
        if score > best_score:
            best, best_score = lines, score
        if best_score >= 40:      # good enough, skip the other versions (faster)
            break
    return best


@st.cache_data(show_spinner="Checking red warning boxes...")
def scan_red_boxes(data: bytes):
    reader = load_reader(False)
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    image.thumbnail((1100, 1100))
    rgb = np.array(image.convert("RGB"))
    boxes = find_red_boxes(rgb)
    out = []
    for x, y, w, h in boxes:
        out += read_red_box(reader, rgb[y:y + h, x:x + w])
    return out, len(boxes)


def drop_similar(new, known, cutoff=0.8):
    """Remove lines that nearly repeat a line we already have."""
    kept, seen = [], [t.lower() for t in known]
    for t, c in new:
        tl = t.lower()
        if any(difflib.SequenceMatcher(None, tl, s).ratio() >= cutoff for s in seen):
            continue
        kept.append((t, c))
        seen.append(tl)
    return kept


# ---------- how trustworthy is each line / word? ----------
@functools.lru_cache(maxsize=4096)
def unknown_words(t):
    """Words in t that the spell checker does not recognise."""
    if spell is None:
        return ()
    words = [w for w in re.findall(r"[A-Za-z]+", t) if len(w) >= 3]
    bad = spell.unknown([w.lower() for w in words])
    return tuple(w for w in words if w.lower() in bad)


def word_flags(t):
    """[(word, doubtful?), ...] for one line."""
    bad = {w.lower() for w in unknown_words(t)}
    pairs = []
    for tok in t.split():
        letters = re.findall(r"[A-Za-z]+", tok)
        pairs.append((tok, any(w.lower() in bad for w in letters)))
    return pairs


def level(t, c):
    if c < HIDE_LIMIT:
        return "hidden"
    if c < CHECK_LIMIT:
        return "check"                 # whole line is doubtful
    return "partial" if unknown_words(t) else "clear"   # partial = only some words doubtful


def word_stats(results):
    green = yellow = 0
    for t, c in results:
        lv = level(t, c)
        if lv == "hidden":
            continue
        if lv == "clear":
            green += len(t.split())
        elif lv == "check":
            yellow += len(t.split())
        else:
            for _, bad in word_flags(t):
                yellow += bad
                green += not bad
    return green, yellow


# ---------- expiry detection ----------
def parse_match(m):
    g = m.groups()
    try:
        if g[0]:
            d, mo, y = int(g[0]), int(g[1]), int(g[2])
            y = y + 2000 if y < 100 else y
            return datetime.date(y, mo, d)
        if g[3]:
            mo, y = int(g[3]), int(g[4])
            y = y + 2000 if y < 100 else y
            return datetime.date(y, mo, calendar.monthrange(y, mo)[1])
        mo = MONTHS.get(g[5].lower()[:3])
        if mo:
            y = int(g[6])
            y = y + 2000 if y < 100 else y
            return datetime.date(y, mo, calendar.monthrange(y, mo)[1])
    except ValueError:
        return None
    return None


def find_expiry(text):
    for k in KEY_RE.finditer(text):
        window = text[k.end():k.end() + 30]
        for m in DATE_RE.finditer(window):
            d = parse_match(m)
            if d and 2000 <= d.year <= 2100:
                return d, "labelled", m.group(0)
    found = []
    for m in DATE_RE.finditer(text):
        d = parse_match(m)
        if d and 2000 <= d.year <= 2100:
            found.append((d, m.group(0)))
    if found:
        d, raw = max(found)
        return d, "probable", raw
    return None, None, None


def show_expiry(text):
    d, how, raw = find_expiry(text)
    if not d:
        st.info("📅 No expiry date detected. Please check the packaging.")
        return
    note = "" if how == "labelled" else " (probable, please verify on the pack)"
    if d < datetime.date.today():
        st.error(f"🔴 EXPIRED: {d:%d %b %Y}{note}")
    else:
        st.success(f"🟢 Not expired: valid until {d:%d %b %Y}{note}")
    st.caption(f'Date text found on label: "{raw}". Month/year dates count as the last day of the month.')


def key_facts(text):
    facts = []
    m = BATCH_RE.search(text)
    if m:
        facts.append(("Batch no.", m.group(1).upper()))
    m = MRP_RE.search(text)
    if m:
        facts.append(("MRP", f"Rs. {m.group(1)}"))
    m = MFG_RE.search(text)
    if m:
        facts.append(("Manufactured", re.sub(r"\s+", "", m.group(1))))
    return facts


def show_key_facts(text):
    facts = key_facts(text)
    if not facts:
        return
    with st.container(border=True):
        st.markdown("**🧾 Key facts printed on the label**")
        for k, v in facts:
            st.write(f"{k}: **{v}**")
        st.caption("Copied from the label text. Check them on the pack.")


# ---------- colour-coded text ----------
GREEN = "rgba(46,160,67,0.30)"
YELLOW = "rgba(255,193,7,0.40)"


def span(text, colour):
    return (f'<span style="background:{colour};padding:2px 5px;border-radius:4px;'
            f'line-height:2.2;margin-right:2px">{html.escape(text)}</span>')


def line_html(t):
    """Colour a confident line word by word: only the doubtful words are yellow."""
    groups = []
    for tok, bad in word_flags(t):
        if groups and groups[-1][0] == bad:
            groups[-1][1].append(tok)
        else:
            groups.append((bad, [tok]))
    return " ".join(span(" ".join(g), YELLOW if bad else GREEN) for bad, g in groups)


def colourise(results):
    parts = []
    for t, c in results:
        lv = level(t, c)
        if lv == "hidden":
            continue
        if lv == "clear":
            parts.append(span(t, GREEN))
        elif lv == "check":
            parts.append(span(t, YELLOW))
        else:
            parts.append(line_html(t))
    return " ".join(parts)


# ---------- translation + audio ----------
def chunk_text(text, size=450):
    chunks, cur = [], ""
    for w in text.split():
        if len(cur) + len(w) + 1 > size:
            chunks.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        chunks.append(cur)
    return chunks


@st.cache_data(show_spinner="Translating...")
def to_hindi(text):
    chunks = chunk_text(text)
    for engine in ("google", "mymemory"):
        try:
            out = []
            for c in chunks:
                if engine == "google":
                    out.append(GoogleTranslator(source="en", target="hi").translate(c))
                else:
                    out.append(MyMemoryTranslator(source="english", target="hindi").translate(c))
                time.sleep(0.4)
            return " ".join(out)
        except Exception:
            continue
    raise RuntimeError("All translators failed")


@st.cache_data(show_spinner="Making audio...")
def make_audio(text, lang_code):
    buf = io.BytesIO()
    gTTS(text[:1500], lang=lang_code).write_to_fp(buf)
    return buf.getvalue()


# ---------- section cards ----------
SECTION_HEADERS = [
    ("Directions", r"^\s*(directions?|how to use|dosage|usage)"),
    ("Warnings", r"^\s*(warnings?|cautions?|precautions?|side effects?)|external use|keep out of reach|do not|prescription|schedule\s*h"),
    ("Storage", r"^\s*(storage|store)"),
    ("Ingredients", r"^\s*(ingredients?|composition)"),
    ("Other details", r"^\s*(marketed|manufactured|mfg|net quantity|mrp|batch)|use before|expiry"),
]
ICONS = {
    "About the product": "🧴", "Directions": "📋", "Warnings": "⚠️",
    "Storage": "🌡️", "Ingredients": "🧪", "Other details": "🏭",
}


def split_sections(good):
    sections = {}
    current = "About the product"
    for t, c in good:
        for name, pat in SECTION_HEADERS:
            if re.search(pat, t, re.I):
                current = name
                break
        sections.setdefault(current, []).append(t)
    return sections


def show_sections(sections, lang):
    st.subheader("Label sections")
    for name, lines in sections.items():
        body = " ".join(lines)
        with st.container(border=True):
            st.markdown(f"**{ICONS.get(name, '📄')} {name}**")
            st.write(body)
            if st.button(f"🔊 Read {name}", key=f"read_{name}"):
                try:
                    if lang == "हिन्दी":
                        spoken, code = to_hindi(body), "hi"
                    else:
                        spoken, code = body, "en"
                    st.audio(make_audio(spoken, code), format="audio/mp3")
                except Exception:
                    st.error("Audio is busy right now. Try again in a minute.")


# ---------- sidebar ----------
with st.sidebar:
    st.header("Scan settings")
    quality = st.select_slider(
        "Speed vs accuracy", options=["Fast", "Balanced", "Detailed"], value="Fast",
        help="Fast is quickest. Detailed reads smaller print but is slower.",
    )
    MAX_SIDE = {"Fast": 800, "Balanced": 1100, "Detailed": 1500}[quality]
    with_hindi = st.checkbox("Label also has Hindi text", value=False)
    use_clahe = st.checkbox(
        "Contrast boost", value=True,
        help="Turn this off for shiny foil strips if the text looks noisy.",
    )
    check_red = st.checkbox(
        "Look for red warning boxes", value=True,
        help="Reads light text on red boxes, like 'Schedule H prescription drug'.",
    )

# load the OCR model once, before any timing starts
load_reader(with_hindi)
if check_red:
    load_reader(False)

# ---------- app ----------
img = st.camera_input("Scan a label") or st.file_uploader(
    "Or upload a photo", type=["png", "jpg", "jpeg"]
)

if img:
    t0 = time.time()
    data = img.getvalue()
    results, processed, blur, ocr_secs = run_ocr(data, use_clahe, MAX_SIDE, with_hindi)
    results = list(results)

    red_results, red_count = [], 0
    if check_red:
        red_results, red_count = scan_red_boxes(data)
        red_results = drop_similar(red_results, [])     # top and bottom boxes repeat
        results = drop_similar(red_results, [t for t, _ in results]) + results

    good = [(t, c) for t, c in results if c >= HIDE_LIMIT]
    text = " ".join(t for t, c in good)

    # Rotated text is scanned only when the user asks for it
    if find_expiry(text)[1] != "labelled":
        flag = f"rot_{hash(data)}"
        if st.button("🔄 Look for rotated text: dates, batch, MRP (slower)"):
            st.session_state[flag] = True
        if st.session_state.get(flag):
            seen = {t.lower() for t, _ in results}
            for t, c in scan_rotated(data):
                if c >= 0.5 and len(t.strip()) >= 4 and t.lower() not in seen:
                    results.append((t, c))
                    seen.add(t.lower())
            good = [(t, c) for t, c in results if c >= HIDE_LIMIT]
            text = " ".join(t for t, c in good)

    if blur < BLUR_LIMIT:
        st.warning("📷 The photo looks blurry. Hold the phone steady, move slightly back, "
                   "use brighter light and retake it for a more accurate reading.")
    st.caption(f"Sharpness score: {blur:.0f} (warning below {BLUR_LIMIT})")
    st.caption(f"OCR itself took {ocr_secs:.1f} s (first run, model loading not included). "
               f"This page took {time.time() - t0:.1f} s.")

    if red_count and not red_results:
        st.warning("⚠️ A red box was found on the label (it may be a prescription or safety "
                   "warning) but it could not be read. Please check it on the pack.")
    elif red_results:
        st.warning("⚠️ Text from a red warning box on the label:")
        st.markdown(colourise(red_results), unsafe_allow_html=True)

    show_expiry(text)
    show_key_facts(text)

    lang = st.radio("Language", ["English", "हिन्दी"], horizontal=True)

    translated = False
    shown = text
    if lang == "हिन्दी" and text:
        try:
            shown = to_hindi(text)
            translated = True
        except Exception:
            st.error("Translation is busy right now. Showing English. Wait a minute and switch again.")

    st.subheader("Label text")
    if not text:
        st.write("No text detected, try a sharper photo.")
    else:
        st.caption("🟩 Green = confident and the word is recognised")
        st.caption("🟨 Yellow = low confidence or an unrecognised word. Please check it on the pack.")
        st.caption("Unclear lines are hidden. Even green text can be wrong, "
                   "so double-check doses and dates on the pack.")
        g, y = word_stats(results)
        if g + y:
            st.caption(f"{100 * g / (g + y):.0f}% of the words read are green ({g} green, {y} yellow).")
        if translated:
            c1, c2 = st.columns(2)
            c1.markdown("**Original (from label)**")
            c1.markdown(colourise(results), unsafe_allow_html=True)
            c2.markdown("**हिन्दी (translated)**")
            c2.write(shown)
        else:
            st.markdown(colourise(results), unsafe_allow_html=True)

    hidden = sum(1 for _, c in results if c < HIDE_LIMIT)
    if hidden:
        st.warning(f"{hidden} line(s) were too unclear to read and are hidden. "
                   "Please check the original packaging for those parts, "
                   "especially any warning boxes.")
    st.info("LabelLens only repeats what is printed on the label. "
            "Always double-check numbers (doses, dates) and verify with your "
            "pharmacist or the original packaging.")

    show_sections(split_sections(good), lang)

    if text and st.button("🔊 Read aloud"):
        audio = make_audio(shown, "hi" if translated else "en")
        st.audio(audio, format="audio/mp3")

    with st.expander("Raw OCR results (with confidence)"):
        for t, c in results:
            st.write(f"{t}  ({c:.2f})  [{level(t, c)}]")
    with st.expander("Processed image"):
        st.image(processed)