"""Generate the July 7 interim presentation deck (.pptx).

Run once:
    .venv\\Scripts\\python.exe -m pip install python-pptx
    .venv\\Scripts\\python.exe scripts/make_interim_pptx.py

Output: ``INTERIM_PRESENTATION_JULY7.pptx`` in the project root.

Content matches ``INTERIM_SCRIPT_JULY7.md`` slide-for-slide. Speaker
notes are attached to every slide so you can read them in Presenter
View while facing the panel.

Palette — Midnight Executive:
  Primary   navy    1E2761
  Secondary ice     CADCFC
  Accent    white   FFFFFF
  Highlight gold    FFC857   (used sparingly for the RO5 money slide)
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
except ImportError as e:                                        # noqa: BLE001
    print("python-pptx not installed. Run:")
    print("  .venv\\Scripts\\python.exe -m pip install python-pptx")
    sys.exit(1)


# --------------------------------------------------------------------- palette
NAVY   = RGBColor(0x1E, 0x27, 0x61)
ICE    = RGBColor(0xCA, 0xDC, 0xFC)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
GOLD   = RGBColor(0xFF, 0xC8, 0x57)
MUTED  = RGBColor(0x8A, 0x96, 0xA6)
DARK   = RGBColor(0x0B, 0x10, 0x15)
BULL   = RGBColor(0x26, 0xA6, 0x9A)
BEAR   = RGBColor(0xEF, 0x53, 0x50)
DIVIDE = RGBColor(0x2C, 0x37, 0x50)

# safe fonts (see pptx skill Typography notes)
HEAD_FONT = "Cambria"          # serif for headers
BODY_FONT = "Calibri"          # sans for body


# ============================================================ helper builders
SLIDE_W = Inches(10)
SLIDE_H = Inches(5.625)


def add_bg(slide, color):
    """Solid slide background covering the full 10" × 5.625" slide."""
    bg = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H,
    )
    bg.fill.solid(); bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    # Move behind everything else.
    slide.shapes._spTree.remove(bg._element); slide.shapes._spTree.insert(2, bg._element)
    return bg


def add_text(slide, x, y, w, h, text, *, font=BODY_FONT, size=14,
             color=WHITE, bold=False, italic=False, align="left",
             valign="top", margin=0.05):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.word_wrap = True
    tf.vertical_anchor = {
        "top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM,
    }[valign]
    para = tf.paragraphs[0]
    para.alignment = {
        "left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT,
    }[align]
    run = para.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    return box


def add_bullets(slide, x, y, w, h, items, *, font=BODY_FONT, size=14,
                color=WHITE, dot_color=None, space_after=6):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    dc = dot_color or color
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(space_after)
        # coloured bullet dot
        r = p.add_run(); r.text = "▪  "
        r.font.name = font; r.font.size = Pt(size)
        r.font.color.rgb = dc; r.font.bold = True
        r2 = p.add_run(); r2.text = item
        r2.font.name = font; r2.font.size = Pt(size); r2.font.color.rgb = color
    return box


def add_card(slide, x, y, w, h, *, fill=DIVIDE, shadow=True):
    """Rounded card. Returns the shape so you can drop text on top."""
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h),
    )
    card.fill.solid(); card.fill.fore_color.rgb = fill
    card.line.fill.background()
    if shadow:
        # python-pptx shadow support is limited; the default is fine.
        pass
    card.adjustments[0] = 0.08          # rounder corners
    return card


def add_footer(slide, page_num, total=12):
    add_text(slide, 0.3, 5.28, 6, 0.3,
             "Multimodal Crypto Direction Predictor · Interim 2 · July 7 2026",
             font=BODY_FONT, size=9, color=MUTED, italic=True)
    add_text(slide, 9.0, 5.28, 0.7, 0.3, f"{page_num} / {total}",
             font=BODY_FONT, size=9, color=MUTED, align="right")


# ============================================================ slide builders
def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)

    # decorative side band with soft tint
    band = s.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(6.5), Inches(0), Inches(3.5), Inches(5.63))
    band.fill.solid(); band.fill.fore_color.rgb = DIVIDE
    band.line.fill.background()

    # accent circle
    dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(8.15), Inches(1.4),
                              Inches(1.3), Inches(1.3))
    dot.fill.solid(); dot.fill.fore_color.rgb = GOLD
    dot.line.fill.background()

    add_text(s, 0.7, 1.05, 5.7, 0.4, "INTERIM PRESENTATION 2",
             font=BODY_FONT, size=14, color=GOLD, bold=True)
    add_text(s, 0.7, 1.55, 6.0, 1.9,
             "Multimodal Cryptocurrency\nDirection Prediction",
             font=HEAD_FONT, size=36, color=WHITE, bold=True)
    add_text(s, 0.7, 3.35, 6.0, 0.6,
             "with Calibrated Uncertainty",
             font=HEAD_FONT, size=22, color=ICE, italic=True)
    add_text(s, 0.7, 4.15, 6.0, 0.4,
             "Fusing numerical features · chart-CNN · news sentiment",
             font=BODY_FONT, size=13, color=ICE)
    add_text(s, 0.7, 4.55, 6.0, 0.4,
             "Wrapped in inductive conformal prediction · validated via Diebold-Mariano",
             font=BODY_FONT, size=13, color=ICE)

    add_text(s, 0.7, 5.05, 6.0, 0.4,
             "Pathum   ·   Sri Lanka Institute of Information Technology",
             font=BODY_FONT, size=12, color=MUTED, italic=True)

    s.notes_slide.notes_text_frame.text = (
        "Open with the aim in one sentence. Ten minutes total, twelve slides. "
        "Emphasise: what is already built, with measured evidence, and defence "
        "against the toughest questions I can anticipate."
    )
    return s


def header(s, title, subtitle=None):
    add_text(s, 0.5, 0.3, 9, 0.5, title, font=HEAD_FONT, size=26,
             color=WHITE, bold=True)
    if subtitle:
        add_text(s, 0.5, 0.85, 9, 0.35, subtitle, font=BODY_FONT, size=12,
                 color=ICE, italic=True)


def slide_problem(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "Problem", "single-modality crypto prediction has hit a measurable ceiling")

    # 3 cards
    cards = [
        ("~53 %", "accuracy plateau for numerical\nmodels on 1h BTC direction",
         "Livieris et al. 2020; Sezer & Ozbayoglu 2018"),
        ("rarely tested", "chart-vision models seldom\nDM-benchmarked vs strong\ntabular baselines", ""),
        ("no calibration", "practitioners cannot know\nwhen to trust the model —\nalmost no prior work quantifies it", ""),
    ]
    for i, (big, mid, cite) in enumerate(cards):
        x = 0.5 + i * 3.1
        add_card(s, x, 1.5, 2.9, 2.5, fill=DIVIDE)
        add_text(s, x + 0.15, 1.65, 2.6, 0.7, big, font=HEAD_FONT, size=24,
                 color=GOLD, bold=True)
        add_text(s, x + 0.15, 2.4, 2.6, 1.2, mid, font=BODY_FONT, size=13, color=ICE)
        add_text(s, x + 0.15, 3.6, 2.6, 0.4, cite, font=BODY_FONT, size=9,
                 color=MUTED, italic=True)

    add_text(s, 0.5, 4.35, 9, 0.4,
             "My contribution — measure whether fusing all three modalities produces "
             "statistically significant improvement,",
             font=BODY_FONT, size=13, color=WHITE, bold=True)
    add_text(s, 0.5, 4.7, 9, 0.4,
             "and quantify the model's confidence honestly with conformal prediction.",
             font=BODY_FONT, size=13, color=WHITE, bold=True)

    add_footer(s, 2)
    s.notes_slide.notes_text_frame.text = (
        "Deliver the ceiling number confidently. If asked why crypto specifically: 24/7 markets, "
        "5-10x the data of equities per pair, ideal for deep learning while adversarial. "
        "If asked if 53% is basically noise: crypto Sharpe-positive threshold is ~52%, so 1pp "
        "gain IS economically meaningful."
    )
    return s


def slide_aim_rqs(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "Aim & Research Questions")

    add_card(s, 0.5, 1.35, 9, 0.9, fill=DIVIDE)
    add_text(s, 0.7, 1.45, 8.6, 0.35, "AIM",
             font=BODY_FONT, size=10, color=GOLD, bold=True)
    add_text(s, 0.7, 1.75, 8.6, 0.55,
             "Design, implement, and evaluate a multimodal crypto direction-prediction system "
             "that fuses chart imagery, numerical features, and news sentiment, quantifying its "
             "own uncertainty via conformal prediction.",
             font=BODY_FONT, size=12, color=WHITE, italic=True)

    rqs = [
        ("RQ1", "Can a leak-free, reproducible pipeline\nbe built for crypto OHLCV + news?"),
        ("RQ2", "Do single-modality numerical models\nhit a measurable accuracy ceiling?"),
        ("RQ3", "Does chart-image CNN add statistically\nsignificant signal beyond numerical?"),
        ("RQ4", "Does fusing all three modalities with\ncalibrated uncertainty beat any single one?"),
    ]
    for i, (tag, body) in enumerate(rqs):
        col, row = i % 2, i // 2
        x = 0.5 + col * 4.6
        y = 2.55 + row * 1.4
        add_card(s, x, y, 4.4, 1.25, fill=DIVIDE)
        add_text(s, x + 0.2, y + 0.1, 1.0, 0.4, tag,
                 font=HEAD_FONT, size=18, color=GOLD, bold=True)
        add_text(s, x + 1.1, y + 0.15, 3.2, 1.0, body,
                 font=BODY_FONT, size=12, color=ICE)

    add_footer(s, 3)
    s.notes_slide.notes_text_frame.text = (
        "Read the aim once, slowly. Then rattle off the four RQs and note each maps "
        "to a phase. RQ4 is the headline."
    )
    return s


def slide_architecture(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "System architecture", "eight phases · each delivers a measurable artefact")

    phases = [
        ("1", "Data ingestion",
         "Binance OHLCV + 10 RSS feeds\nDual timestamps · leak-invariant"),
        ("2", "Numerical baselines",
         "Naive · XGBoost · LSTM · PatchTST\n6-fold walk-forward CV"),
        ("3", "Chart-CNN",
         "64×64 candle images\n+ GAF/MTF encoders"),
        ("4", "News sentiment",
         "CryptoBERT + FinBERT ensemble\n7 per-bar features"),
        ("5", "Multimodal fusion",
         "Concat-LR + inductive conformal\n90 % calibrated sets"),
        ("6", "Pattern engine",
         "FAISS + 3-state HMM\n+ Elastic Weight Consolidation"),
        ("7", "Dashboard",
         "FastAPI + TradingView\n+ Grad-CAM + range analysis"),
        ("8", "Self-sustaining",
         "Auto-persist + nightly refit\n+ live paper trading"),
    ]

    # 2 rows × 4 columns
    for i, (num, title, body) in enumerate(phases):
        col, row = i % 4, i // 4
        x = 0.3 + col * 2.4
        y = 1.4 + row * 1.9
        add_card(s, x, y, 2.25, 1.75, fill=DIVIDE)
        # numbered circle
        dot = s.shapes.add_shape(
            MSO_SHAPE.OVAL, Inches(x + 0.15), Inches(y + 0.15),
            Inches(0.5), Inches(0.5))
        dot.fill.solid(); dot.fill.fore_color.rgb = GOLD
        dot.line.fill.background()
        add_text(s, x + 0.15, y + 0.18, 0.5, 0.5, num,
                 font=HEAD_FONT, size=18, color=NAVY, bold=True, align="center")
        add_text(s, x + 0.75, y + 0.18, 1.5, 0.4, title,
                 font=HEAD_FONT, size=13, color=WHITE, bold=True)
        add_text(s, x + 0.15, y + 0.85, 2.05, 0.85, body,
                 font=BODY_FONT, size=10, color=ICE)

    add_footer(s, 4)
    s.notes_slide.notes_text_frame.text = (
        "Walk the panel through the phases. Emphasise: each PHASE has a deliverable, "
        "not a promise. Phase 5 is the headline. Phase 8 is the self-sustaining loop."
    )
    return s


def slide_ro1_4(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "RO1–RO4 · foundation evidence",
           "four objectives complete with measured evidence on disk")

    rows = [
        ("RO1", "Data pipeline · RQ1",
         "77,688 BTC 1h bars · 454 scored articles · 100 % published_at ≤ ingested_at"),
        ("RO2", "Numerical baselines · RQ2",
         "5 models × walk-forward · best (LSTM) 53.4 % · DM p<0.001 vs naive"),
        ("RO3", "Chart-CNN baselines · RQ3",
         "cnn_candle 54.0 % on BTC · DM p = 0.006 vs XGBoost — vision adds signal"),
        ("RO4", "News sentiment · RQ4 prep",
         "CryptoBERT + FinBERT ensemble · leakage invariant tests pass"),
    ]
    for i, (ro, sub, evidence) in enumerate(rows):
        y = 1.35 + i * 0.95
        add_card(s, 0.5, y, 9, 0.82, fill=DIVIDE)
        add_text(s, 0.75, y + 0.06, 0.9, 0.7, ro,
                 font=HEAD_FONT, size=20, color=GOLD, bold=True)
        add_text(s, 1.75, y + 0.07, 4.0, 0.35, sub,
                 font=BODY_FONT, size=12, color=WHITE, bold=True)
        add_text(s, 1.75, y + 0.4, 7.5, 0.4, evidence,
                 font=BODY_FONT, size=11, color=ICE)

    add_footer(s, 5)
    s.notes_slide.notes_text_frame.text = (
        "Deliver each row in one breath. Emphasise DM p-values — that is the "
        "statistical rigour bit the panel wants to hear."
    )
    return s


def slide_ro5(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "RO5 · headline result — fusion beats every baseline",
           "BTCUSDT · 1h · 6-fold walk-forward · n = 37,992 held-out bars")

    # 4 big metric cards
    metrics = [
        ("0.5492", "accuracy",        WHITE),
        ("0.1023", "MCC",             WHITE),
        ("0.0245", "Sharpe / bar",    WHITE),
        ("4.93",   "cumulative log-PnL", GOLD),
    ]
    for i, (val, lbl, col) in enumerate(metrics):
        x = 0.5 + i * 2.35
        add_card(s, x, 1.35, 2.15, 1.35, fill=DIVIDE)
        add_text(s, x, 1.45, 2.15, 0.8, val, font=HEAD_FONT, size=30,
                 color=col, bold=True, align="center")
        add_text(s, x, 2.28, 2.15, 0.35, lbl.upper(), font=BODY_FONT,
                 size=10, color=MUTED, align="center", bold=True)

    # DM p-values block
    add_card(s, 0.5, 2.9, 4.4, 2.05, fill=DIVIDE)
    add_text(s, 0.7, 3.0, 4.0, 0.35, "DIEBOLD–MARIANO p-values",
             font=BODY_FONT, size=10, color=GOLD, bold=True)
    add_text(s, 0.7, 3.35, 4.0, 0.35, "vs cnn_candle",
             font=BODY_FONT, size=12, color=WHITE)
    add_text(s, 3.5, 3.35, 1.3, 0.35, "p ≈ 0.001",
             font=HEAD_FONT, size=14, color=GOLD, bold=True, align="right")
    add_text(s, 0.7, 3.75, 4.0, 0.35, "vs XGBoost",
             font=BODY_FONT, size=12, color=WHITE)
    add_text(s, 3.5, 3.75, 1.3, 0.35, "p ≈ 0.000",
             font=HEAD_FONT, size=14, color=GOLD, bold=True, align="right")
    add_text(s, 0.7, 4.15, 4.0, 0.35, "vs LSTM · PatchTST · cnn_gaf",
             font=BODY_FONT, size=12, color=WHITE)
    add_text(s, 3.5, 4.15, 1.3, 0.35, "p ≈ 0.000",
             font=HEAD_FONT, size=14, color=GOLD, bold=True, align="right")
    add_text(s, 0.7, 4.6, 4.0, 0.3,
             "negative signed p ⇒ fusion beats the baseline",
             font=BODY_FONT, size=9, color=MUTED, italic=True)

    # Conformal card
    add_card(s, 5.1, 2.9, 4.4, 2.05, fill=DIVIDE)
    add_text(s, 5.3, 3.0, 4.0, 0.35, "CONFORMAL CALIBRATION",
             font=BODY_FONT, size=10, color=GOLD, bold=True)
    add_text(s, 5.3, 3.35, 4.0, 0.35, "empirical coverage",
             font=BODY_FONT, size=12, color=WHITE)
    add_text(s, 8.1, 3.35, 1.3, 0.35, "0.883",
             font=HEAD_FONT, size=14, color=GOLD, bold=True, align="right")
    add_text(s, 5.3, 3.75, 4.0, 0.35, "average set size",
             font=BODY_FONT, size=12, color=WHITE)
    add_text(s, 8.1, 3.75, 1.3, 0.35, "1.72",
             font=HEAD_FONT, size=14, color=GOLD, bold=True, align="right")
    add_text(s, 5.3, 4.15, 4.0, 0.35, "confident singletons",
             font=BODY_FONT, size=12, color=WHITE)
    add_text(s, 8.1, 4.15, 1.3, 0.35, "28.2 %",
             font=HEAD_FONT, size=14, color=GOLD, bold=True, align="right")
    add_text(s, 5.3, 4.6, 4.0, 0.3,
             "0.883 within 2 pp of nominal 0.90 → calibration is honest",
             font=BODY_FONT, size=9, color=MUTED, italic=True)

    add_footer(s, 6)
    s.notes_slide.notes_text_frame.text = (
        "THIS IS THE MONEY SLIDE. Deliver slowly and confidently. Pause after "
        "the 0.5492. Pause after 'p equals one in a thousand'. Emphasise that "
        "88% coverage vs 90% nominal is the calibration honesty proof."
    )
    return s


def slide_ro6(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "RO6 · pattern engine + continual learning",
           "FAISS retrieval · HMM regimes · EWC forgetting audit")

    add_card(s, 0.5, 1.35, 4.4, 3.5, fill=DIVIDE)
    add_text(s, 0.7, 1.45, 4.0, 0.35, "RETRIEVAL",
             font=BODY_FONT, size=10, color=GOLD, bold=True)
    add_text(s, 0.7, 1.85, 4.0, 0.4,
             "FAISS IndexFlatIP over 128-d L2-normalised CNN embeddings",
             font=BODY_FONT, size=12, color=WHITE)
    add_text(s, 0.7, 2.35, 4.0, 0.4,
             "→ sub-ms top-K on 77 K bars",
             font=BODY_FONT, size=12, color=GOLD, bold=True)

    add_text(s, 0.7, 3.0, 4.0, 0.35, "REGIMES",
             font=BODY_FONT, size=10, color=GOLD, bold=True)
    add_text(s, 0.7, 3.35, 4.0, 0.4,
             "3-state HMM on (log-return, 24h vol)",
             font=BODY_FONT, size=12, color=WHITE)
    add_text(s, 0.7, 3.75, 4.0, 0.4,
             "9.9 % bear · 58.2 % sideways · 31.9 % bull",
             font=BODY_FONT, size=12, color=ICE)
    add_text(s, 0.7, 4.2, 4.0, 0.4,
             "three well-populated states — no degeneracy",
             font=BODY_FONT, size=10, color=MUTED, italic=True)

    add_card(s, 5.1, 1.35, 4.4, 3.5, fill=DIVIDE)
    add_text(s, 5.3, 1.45, 4.0, 0.35, "ELASTIC WEIGHT CONSOLIDATION",
             font=BODY_FONT, size=10, color=GOLD, bold=True)
    add_text(s, 5.3, 1.85, 4.0, 0.4,
             "3 chronological phases · 6 epochs each · λ=5,000",
             font=BODY_FONT, size=12, color=WHITE)

    add_text(s, 5.3, 2.4, 2.4, 0.35, "naive sequential",
             font=BODY_FONT, size=12, color=WHITE)
    add_text(s, 7.7, 2.4, 1.6, 0.35, "0.5553",
             font=HEAD_FONT, size=14, color=WHITE, bold=True, align="right")

    add_text(s, 5.3, 2.85, 2.4, 0.35, "EWC sequential",
             font=BODY_FONT, size=12, color=WHITE)
    add_text(s, 7.7, 2.85, 1.6, 0.35, "0.5509",
             font=HEAD_FONT, size=14, color=WHITE, bold=True, align="right")

    add_text(s, 5.3, 3.3, 2.4, 0.35, "Δ (EWC − naive)",
             font=BODY_FONT, size=12, color=GOLD, bold=True)
    add_text(s, 7.7, 3.3, 1.6, 0.35, "−0.45 pp",
             font=HEAD_FONT, size=14, color=GOLD, bold=True, align="right")

    add_text(s, 5.3, 3.9, 4.0, 0.85,
             "Interpretation: catastrophic forgetting is NOT the binding\n"
             "constraint on BTC 1h. Framework in place to detect drift.",
             font=BODY_FONT, size=11, color=ICE, italic=True)

    add_footer(s, 7)
    s.notes_slide.notes_text_frame.text = (
        "The EWC delta being small IS a positive finding — it means the "
        "underlying data isn't drifting hard, and the framework is instrumented "
        "to detect it if it were."
    )
    return s


def slide_ro7(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "RO7 · interactive dashboard — LIVE DEMO",
           "explainable predictions in one screen")

    # feature list
    features = [
        "FastAPI backend · vanilla-JS frontend · TradingView Lightweight Charts",
        "Live SSE relay — backend fans Binance ticks to browser (no client-side geo issue)",
        "Click any candle → fused prediction + 90 % conformal set",
        "Top-5 similar historical patterns (regime-filtered) as mini candlestick charts",
        "Grad-CAM overlay — what the CNN attended to on that specific bar",
        "\"Analyze visible range\" — regime mix · return · P(up) histogram · news overlaps",
    ]
    add_bullets(s, 0.5, 1.3, 9, 3.6, features,
                font=BODY_FONT, size=13, color=WHITE, dot_color=GOLD,
                space_after=8)

    # demo tag
    demo = s.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(4.6), Inches(9), Inches(0.55))
    demo.fill.solid(); demo.fill.fore_color.rgb = GOLD
    demo.line.fill.background()
    demo.adjustments[0] = 0.5
    add_text(s, 0.7, 4.68, 8.6, 0.4,
             "🖥  Switch to http://127.0.0.1:8000 for a 30-second live demo",
             font=HEAD_FONT, size=16, color=NAVY, bold=True, align="center")

    add_footer(s, 8)
    s.notes_slide.notes_text_frame.text = (
        "If live demo works: switch tab, click Start on paper trader, click a candle, "
        "click Analyze visible range. If it fails: pivot to screenshots you took the "
        "night before. HAVE THE SCREENSHOTS READY."
    )
    return s


def slide_ro8(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "RO8 · self-sustaining live pipeline",
           "the system evaluates itself, forward, in real time")

    # 4 quadrants
    quads = [
        ("PAPER TRADER",
         "confidence-gated — trades ONLY on\nconformal singleton bars",
         GOLD),
        ("VOL-SCALED SIZING",
         "inverse-vol capped at 10-100 %\nof balance per trade",
         GOLD),
        ("AUTO-PERSIST",
         "every closed candle writes to the\nPhase-1 parquet schema",
         GOLD),
        ("AUTO-REFIT",
         "HMM + FAISS + fusion refit nightly\nwithout dashboard restart",
         GOLD),
    ]
    for i, (title, body, col) in enumerate(quads):
        cx, ry = i % 2, i // 2
        x = 0.5 + cx * 4.6
        y = 1.35 + ry * 1.55
        add_card(s, x, y, 4.4, 1.35, fill=DIVIDE)
        add_text(s, x + 0.2, y + 0.15, 4.0, 0.35, title,
                 font=BODY_FONT, size=10, color=col, bold=True)
        add_text(s, x + 0.2, y + 0.55, 4.0, 0.8, body,
                 font=BODY_FONT, size=13, color=WHITE)

    add_card(s, 0.5, 4.55, 9, 0.55, fill=GOLD)
    add_text(s, 0.7, 4.6, 8.6, 0.5,
             "confident-accuracy > uncertain-accuracy → live proof that conformal calibration is real",
             font=HEAD_FONT, size=13, color=NAVY, bold=True, align="center")

    add_footer(s, 9)
    s.notes_slide.notes_text_frame.text = (
        "The confidence-vs-uncertainty accuracy split is the LIVE VERSION of the "
        "conformal calibration proof. Emphasise: this is thesis gold."
    )
    return s


def slide_mapping(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "Objectives → evidence mapping",
           "every objective has measurable evidence on disk right now")

    rows = [
        ("RO1",  "Data pipeline",        "RQ1",  "77 K bars · 454 articles · invariants pass"),
        ("RO2",  "Numerical baselines",  "RQ2",  "25 experiments · DM p<0.001 vs naive"),
        ("RO3",  "Chart-CNN",            "RQ3",  "54.0 % · DM p=0.006 vs XGBoost"),
        ("RO4",  "News sentiment",       "RQ4",  "ensemble + leak-invariant tests"),
        ("RO5",  "Multimodal fusion",    "RQ4",  "0.549 acc · DM p≈0.001 · CP cov 0.88"),
        ("RO6",  "Pattern engine",       "RQ4",  "FAISS + 3-state HMM + EWC delta"),
        ("RO7",  "Dashboard",            "all",  "running at localhost:8000"),
        ("RO8",  "Self-sustaining",      "RQ4",  "persistor + retrainer live"),
    ]

    # table header
    add_card(s, 0.5, 1.3, 9, 0.4, fill=GOLD)
    hcolor = NAVY
    add_text(s, 0.7,  1.32, 0.9, 0.35, "RO",        font=BODY_FONT, size=11, color=hcolor, bold=True)
    add_text(s, 1.65, 1.32, 3.0, 0.35, "OBJECTIVE", font=BODY_FONT, size=11, color=hcolor, bold=True)
    add_text(s, 4.85, 1.32, 0.9, 0.35, "MAPS TO",   font=BODY_FONT, size=11, color=hcolor, bold=True)
    add_text(s, 5.85, 1.32, 3.5, 0.35, "EVIDENCE",  font=BODY_FONT, size=11, color=hcolor, bold=True)

    for i, (ro, obj, rq, ev) in enumerate(rows):
        y = 1.75 + i * 0.4
        fill = DIVIDE if i % 2 == 0 else NAVY
        add_card(s, 0.5, y, 9, 0.38, fill=fill)
        highlight = ro == "RO5"
        col = GOLD if highlight else WHITE
        add_text(s, 0.7,  y + 0.03, 0.9, 0.35, ro,  font=HEAD_FONT, size=13, color=col, bold=True)
        add_text(s, 1.65, y + 0.03, 3.0, 0.35, obj, font=BODY_FONT, size=12, color=col)
        add_text(s, 4.85, y + 0.03, 0.9, 0.35, rq,  font=BODY_FONT, size=12, color=col)
        add_text(s, 5.85, y + 0.03, 3.5, 0.35, ev,  font=BODY_FONT, size=11, color=ICE if not highlight else GOLD)

    add_footer(s, 10)
    s.notes_slide.notes_text_frame.text = (
        "This is what the panel should walk out remembering. Point at RO5 — 'that "
        "row is the direct answer to RQ4.' The mapping is what makes the objectives "
        "unfailable in the viva."
    )
    return s


def slide_defence(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "Defence playbook — toughest questions rehearsed")

    qas = [
        ("What if fusion fails at final?",
         "Objective is to MEASURE, not win. A negative result with a tight CI is a\n"
         "publishable contribution. Already measured — it did not fail."),
        ("Why 1 hour and not 1 day?",
         "Finest interval with ~77 K bars (enough for deep learning) where 24h news\n"
         "lookback is still relevant."),
        ("Only BTC in Phase 5?",
         "Phases 2 & 3 evaluated across 5 pairs. Phase 5 focused BTC first for depth;\n"
         "6–8 are pair-agnostic — expansion is config, not code."),
        ("Is 54.9 % economically meaningful?",
         "Sharpe/bar 0.025 and cumulative log-PnL 4.93 over 38 K bars —\n"
         "measurable positive expectancy before fees."),
        ("Overfitting?",
         "Walk-forward CV with 1-bar gap · lookahead-safe features · conformal cov\n"
         "0.88 near nominal 0.90 = honesty test passed."),
    ]

    for i, (q, a) in enumerate(qas):
        y = 1.35 + i * 0.75
        add_card(s, 0.5, y, 9, 0.68, fill=DIVIDE)
        add_text(s, 0.7,  y + 0.05, 4.0, 0.3,  q, font=BODY_FONT,
                 size=11, color=GOLD, bold=True, italic=True)
        add_text(s, 0.7,  y + 0.32, 8.6, 0.35, a, font=BODY_FONT,
                 size=10, color=ICE)

    add_footer(s, 11)
    s.notes_slide.notes_text_frame.text = (
        "Don't READ from the slide during Q&A — know these by heart. If you can "
        "deliver each answer without looking, the panel will assume the whole "
        "thesis is equally rehearsed."
    )
    return s


def slide_close(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "Timeline to final viva")

    # Gantt-style milestones
    milestones = [
        ("weeks 1-4",  "Extend Phase 5 fusion to all 5 pairs — configuration only"),
        ("weeks 4-8",  "Weekly full CNN retrain — closes the fallback banner"),
        ("weeks 8-10", "Multi-week live paper-trading evidence for RO8 verdict"),
        ("weeks 10-12", "Thesis write-up + final-viva rehearsal"),
    ]
    for i, (period, task) in enumerate(milestones):
        y = 1.4 + i * 0.6
        add_card(s, 0.5, y, 2.4, 0.48, fill=GOLD)
        add_text(s, 0.5, y + 0.08, 2.4, 0.35, period,
                 font=HEAD_FONT, size=13, color=NAVY, bold=True, align="center")
        add_text(s, 3.05, y + 0.08, 6.5, 0.4, task,
                 font=BODY_FONT, size=13, color=WHITE)

    # closing badge
    add_card(s, 0.5, 4.15, 9, 1.0, fill=DIVIDE)
    add_text(s, 0.7, 4.25, 8.6, 0.4,
             "The system is running. The evidence is on disk.",
             font=HEAD_FONT, size=18, color=GOLD, bold=True, align="center")
    add_text(s, 0.7, 4.7, 8.6, 0.4,
             "Thank you — questions?",
             font=HEAD_FONT, size=16, color=WHITE, italic=True, align="center")

    add_footer(s, 12)
    s.notes_slide.notes_text_frame.text = (
        "Close confidently. 'The system is running. The evidence is on disk.' "
        "Then pause. Then 'Thank you — questions?' — don't rush past it."
    )
    return s


# ============================================================ new research slides
def slide_related_work(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "Related work", "anchored to established prior methodology")

    rows = [
        ("Baltrušaitis et al. (2018)", "Late-fusion taxonomy for multimodal ML"),
        ("Livieris et al. (2020)",      "Numerical-only crypto direction baselines"),
        ("Sezer & Ozbayoglu (2018)",    "GAF-encoded chart-image CNN"),
        ("Vovk et al. (2005)",          "Inductive conformal prediction framework"),
        ("Kirkpatrick et al. (2017)",   "Elastic Weight Consolidation (continual learning)"),
        ("Hamilton (1989)",             "Regime-switching HMM in finance"),
        ("Diebold & Mariano (1995)",    "Forecast-loss significance test"),
    ]
    for i, (paper, use) in enumerate(rows):
        y = 1.3 + i * 0.52
        add_card(s, 0.5, y, 9, 0.45, fill=DIVIDE)
        add_text(s, 0.7, y + 0.06, 4.0, 0.35, paper,
                 font=HEAD_FONT, size=13, color=GOLD, bold=True)
        add_text(s, 4.8, y + 0.06, 4.6, 0.35, use,
                 font=BODY_FONT, size=12, color=ICE)

    add_footer(s, 3, total=20)
    s.notes_slide.notes_text_frame.text = (
        "Deliver the row headers with authority. If asked about any one, be ready "
        "with its one-sentence claim from the SAY block."
    )
    return s


def slide_walk_forward(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "Walk-forward cross-validation",
           "the harness that lets us claim statistical significance")

    # visual timeline
    add_card(s, 0.5, 1.3, 9, 2.5, fill=DIVIDE)
    # 4 folds bars
    fold_y = 1.55
    fold_h = 0.35
    fold_pad = 0.15
    label_x = 0.7
    train_x0 = 1.7
    total_w = 7.6
    for i in range(4):
        y = fold_y + i * (fold_h + fold_pad)
        add_text(s, label_x, y, 0.95, fold_h, f"fold {i}",
                 font=BODY_FONT, size=11, color=ICE)
        # train segment
        train_w = 2.5 + i * 1.0
        train_rect = s.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(train_x0), Inches(y),
            Inches(train_w), Inches(fold_h))
        train_rect.fill.solid(); train_rect.fill.fore_color.rgb = GOLD
        train_rect.line.fill.background()
        add_text(s, train_x0 + 0.05, y, train_w - 0.1, fold_h, "train",
                 font=BODY_FONT, size=10, color=NAVY, bold=True, valign="middle")
        # gap
        gap_w = 0.08
        gap_x = train_x0 + train_w
        gap_rect = s.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(gap_x), Inches(y),
            Inches(gap_w), Inches(fold_h))
        gap_rect.fill.solid(); gap_rect.fill.fore_color.rgb = BEAR
        gap_rect.line.fill.background()
        # test segment
        test_w = 1.0
        test_x = gap_x + gap_w
        test_rect = s.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(test_x), Inches(y),
            Inches(test_w), Inches(fold_h))
        test_rect.fill.solid(); test_rect.fill.fore_color.rgb = ICE
        test_rect.line.fill.background()
        add_text(s, test_x + 0.05, y, test_w - 0.1, fold_h, "test",
                 font=BODY_FONT, size=10, color=NAVY, bold=True, valign="middle")

    # legend
    add_text(s, 0.7, 3.15, 8.6, 0.4,
             "training window grows chronologically   ·   1-bar gap prevents label leak   ·   test size fixed",
             font=BODY_FONT, size=11, color=MUTED, italic=True)

    add_text(s, 0.5, 3.95, 9, 0.4,
             "Same harness across every model.  Diebold–Mariano test compares any two on the same test bars.",
             font=BODY_FONT, size=12, color=WHITE, bold=True)
    add_text(s, 0.5, 4.35, 9, 0.4,
             "This is the foundation that lets us claim statistical significance, not just accuracy numbers.",
             font=BODY_FONT, size=12, color=ICE, italic=True)

    add_footer(s, 5, total=20)
    s.notes_slide.notes_text_frame.text = (
        "Point at the expanding gold bar. Emphasise gap = red between train and test — "
        "'this is what prevents label leak'. Then Diebold-Mariano gives statistical rigour."
    )
    return s


def slide_numerical_features(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "Numerical feature engineering",
           "31 lookahead-safe features per bar")

    groups = [
        ("Returns", "6",
         "log-returns at lags 1, 2, 3, 5, 10, 20"),
        ("Rolling stats", "16",
         "mean, std, min, max over windows 5, 10, 20, 50"),
        ("Technical indicators", "4",
         "RSI · MACD · ATR · Bollinger %B"),
        ("Volume / microstructure", "2",
         "volume ratio · taker-buy share (Binance aggressive-buy)"),
        ("Time cyclic", "4",
         "sin / cos of hour-of-day + day-of-week"),
    ]
    for i, (name, cnt, desc) in enumerate(groups):
        y = 1.35 + i * 0.55
        add_card(s, 0.5, y, 9, 0.48, fill=DIVIDE)
        add_text(s, 0.7,  y + 0.06, 0.6, 0.35, cnt,
                 font=HEAD_FONT, size=20, color=GOLD, bold=True)
        add_text(s, 1.5,  y + 0.03, 3.2, 0.35, name,
                 font=BODY_FONT, size=13, color=WHITE, bold=True)
        add_text(s, 1.5,  y + 0.28, 3.2, 0.3, "features",
                 font=BODY_FONT, size=9, color=MUTED, italic=True)
        add_text(s, 4.75, y + 0.08, 4.7, 0.35, desc,
                 font=BODY_FONT, size=11, color=ICE)

    add_text(s, 0.5, 4.35, 9, 0.4,
             "Every feature uses only information available at or before bar t.  Warmup rows dropped.",
             font=BODY_FONT, size=11, color=WHITE, italic=True)
    add_text(s, 0.5, 4.7, 9, 0.4,
             "Enforced by keeping construction in one auditable file (src/features/numerical.py).",
             font=BODY_FONT, size=10, color=MUTED, italic=True)

    add_footer(s, 7, total=20)
    s.notes_slide.notes_text_frame.text = (
        "Speak the counts — 6, 16, 4, 2, 4 = 31 total. Emphasise lookahead-safe by construction."
    )
    return s


def slide_cnn_arch(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "Chart-CNN architecture",
           "compact ~250 K parameters · dual-head design")

    # architecture flow diagram — vertical blocks
    blocks = [
        ("input · (3, 64, 64)",   "RGB candle image  OR  GAF+MTF field"),
        ("Conv 3→32 · BN · ReLU · MaxPool",  "32 × 32"),
        ("Conv 32→64 · BN · ReLU · MaxPool", "16 × 16"),
        ("Conv 64→128 · BN · ReLU · MaxPool", "8 × 8"),
        ("Conv 128→128 · BN · ReLU · AdaptiveAvgPool", "128"),
    ]
    for i, (name, sz) in enumerate(blocks):
        y = 1.3 + i * 0.45
        add_card(s, 0.5, y, 5.5, 0.38, fill=DIVIDE)
        add_text(s, 0.7, y + 0.06, 4.5, 0.3, name,
                 font=BODY_FONT, size=12, color=WHITE)
        add_text(s, 5.15, y + 0.06, 0.8, 0.3, sz,
                 font=HEAD_FONT, size=12, color=GOLD, bold=True, align="right")

    # two-head fork
    add_card(s, 6.25, 1.75, 3.25, 1.35, fill=DIVIDE)
    add_text(s, 6.45, 1.85, 3.0, 0.3, "EMBEDDING HEAD",
             font=BODY_FONT, size=10, color=GOLD, bold=True)
    add_text(s, 6.45, 2.2, 3.0, 0.35, "Linear 128 → 128",
             font=BODY_FONT, size=12, color=WHITE)
    add_text(s, 6.45, 2.55, 3.0, 0.4, "reused by Phase 6 FAISS index",
             font=BODY_FONT, size=10, color=ICE, italic=True)

    add_card(s, 6.25, 3.2, 3.25, 1.35, fill=DIVIDE)
    add_text(s, 6.45, 3.3, 3.0, 0.3, "CLASSIFIER HEAD",
             font=BODY_FONT, size=10, color=GOLD, bold=True)
    add_text(s, 6.45, 3.65, 3.0, 0.35, "Linear 128 → 1",
             font=BODY_FONT, size=12, color=WHITE)
    add_text(s, 6.45, 4.0, 3.0, 0.4, "binary direction logit",
             font=BODY_FONT, size=10, color=ICE, italic=True)

    add_text(s, 0.5, 4.7, 9, 0.4,
             "Same architecture for both encoders · same walk-forward harness · DM-comparable with numerical",
             font=BODY_FONT, size=11, color=MUTED, italic=True, align="center")

    add_footer(s, 8, total=20)
    s.notes_slide.notes_text_frame.text = (
        "Highlight the dual heads: embedding for Phase 6 similarity, classifier for direction. "
        "'250 K parameters is deliberately small — CPU-trainable in ~15 min per fold.'"
    )
    return s


def slide_sentiment(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "News sentiment ensemble",
           "CryptoBERT + FinBERT · calibrated ensemble")

    # two models side by side
    add_card(s, 0.5, 1.35, 4.4, 1.7, fill=DIVIDE)
    add_text(s, 0.7,  1.45, 4.0, 0.35, "CRYPTOBERT",
             font=BODY_FONT, size=10, color=GOLD, bold=True)
    add_text(s, 0.7,  1.8, 4.0, 0.4, "ElKulako/cryptobert",
             font=BODY_FONT, size=12, color=WHITE)
    add_text(s, 0.7,  2.2, 4.0, 0.4, "crypto-domain text training",
             font=BODY_FONT, size=11, color=ICE)
    add_text(s, 0.7,  2.6, 4.0, 0.4, "output ∈ [−1, +1]",
             font=BODY_FONT, size=11, color=ICE)

    add_card(s, 5.1, 1.35, 4.4, 1.7, fill=DIVIDE)
    add_text(s, 5.3,  1.45, 4.0, 0.35, "FINBERT",
             font=BODY_FONT, size=10, color=GOLD, bold=True)
    add_text(s, 5.3,  1.8, 4.0, 0.4, "ProsusAI/finbert",
             font=BODY_FONT, size=12, color=WHITE)
    add_text(s, 5.3,  2.2, 4.0, 0.4, "general financial text training",
             font=BODY_FONT, size=11, color=ICE)
    add_text(s, 5.3,  2.6, 4.0, 0.4, "output ∈ [−1, +1]",
             font=BODY_FONT, size=11, color=ICE)

    # ensemble formula card
    add_card(s, 0.5, 3.2, 9.0, 1.05, fill=DIVIDE)
    add_text(s, 0.7,  3.3, 8.6, 0.35, "ENSEMBLE",
             font=BODY_FONT, size=10, color=GOLD, bold=True)
    add_text(s, 0.7,  3.6, 8.6, 0.35,
             "score = mean(cb, fb)   ·   confidence = 1 − |cb − fb|",
             font=HEAD_FONT, size=13, color=WHITE)
    add_text(s, 0.7,  3.95, 8.6, 0.3,
             "7 per-bar features over 24 h lookback:  sent_count · mean · std · min · max · last · conf_mean",
             font=BODY_FONT, size=10, color=ICE, italic=True)

    # leak guard
    add_card(s, 0.5, 4.35, 9.0, 0.55, fill=GOLD)
    add_text(s, 0.7,  4.42, 8.6, 0.4,
             "Leakage guard — hard assert per row:  published_at  ≤  bar_open_time",
             font=HEAD_FONT, size=12, color=NAVY, bold=True)

    add_footer(s, 9, total=20)
    s.notes_slide.notes_text_frame.text = (
        "Emphasise the ensemble is calibrated — score is agreement, confidence is 1 minus "
        "disagreement. The leak guard is critical for defending against panellist skepticism."
    )
    return s


def slide_fusion_arch(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "Late-fusion architecture",
           "concatenate then classify — 166-dim input")

    # three modality inputs on the left
    modalities = [
        ("Numerical",  "[31]", 1.35),
        ("CNN embed",  "[128]", 2.15),
        ("Sentiment",  "[7]", 2.95),
    ]
    for name, dim, y in modalities:
        add_card(s, 0.5, y, 2.0, 0.55, fill=DIVIDE)
        add_text(s, 0.6, y + 0.08, 1.4, 0.35, name,
                 font=BODY_FONT, size=12, color=WHITE, bold=True)
        add_text(s, 1.75, y + 0.08, 0.6, 0.35, dim,
                 font=HEAD_FONT, size=13, color=GOLD, bold=True, align="right")

    # concat / standardize block
    add_card(s, 3.05, 1.85, 2.15, 1.85, fill=GOLD)
    add_text(s, 3.1, 1.95, 2.0, 0.4, "CONCAT",
             font=HEAD_FONT, size=12, color=NAVY, bold=True, align="center")
    add_text(s, 3.1, 2.3, 2.0, 0.55, "[166]",
             font=HEAD_FONT, size=22, color=NAVY, bold=True, align="center")
    add_text(s, 3.1, 2.9, 2.0, 0.35, "standardize",
             font=BODY_FONT, size=11, color=NAVY, align="center", italic=True)
    add_text(s, 3.1, 3.25, 2.0, 0.3, "(train stats only)",
             font=BODY_FONT, size=9, color=NAVY, align="center")

    # logistic regression
    add_card(s, 5.55, 1.85, 2.0, 1.85, fill=DIVIDE)
    add_text(s, 5.6,  1.95, 1.9, 0.4, "L2 LOGISTIC",
             font=BODY_FONT, size=11, color=GOLD, bold=True, align="center")
    add_text(s, 5.6,  2.3, 1.9, 0.35, "P(up | x)",
             font=HEAD_FONT, size=16, color=WHITE, align="center")
    add_text(s, 5.6,  2.75, 1.9, 0.4, "166 coefficients",
             font=BODY_FONT, size=10, color=ICE, align="center", italic=True)
    add_text(s, 5.6,  3.15, 1.9, 0.4, "fully interpretable",
             font=BODY_FONT, size=10, color=ICE, align="center")

    # conformal wrapper
    add_card(s, 7.9, 1.85, 1.6, 1.85, fill=DIVIDE)
    add_text(s, 7.95, 1.95, 1.5, 0.4, "CONFORMAL",
             font=BODY_FONT, size=11, color=GOLD, bold=True, align="center")
    add_text(s, 7.95, 2.35, 1.5, 0.4, "90 %",
             font=HEAD_FONT, size=22, color=WHITE, bold=True, align="center")
    add_text(s, 7.95, 2.85, 1.5, 0.4, "prediction",
             font=BODY_FONT, size=10, color=ICE, align="center")
    add_text(s, 7.95, 3.15, 1.5, 0.4, "set",
             font=BODY_FONT, size=10, color=ICE, align="center")

    add_text(s, 0.5, 4.05, 9, 0.35,
             "Why simple? interpretable · ablation-friendly · refits in seconds → enables nightly auto-retrain",
             font=BODY_FONT, size=11, color=MUTED, italic=True, align="center")
    add_text(s, 0.5, 4.5, 9, 0.4,
             "This slide IS the answer to RQ4.",
             font=HEAD_FONT, size=13, color=GOLD, bold=True, align="center")

    add_footer(s, 10, total=20)
    s.notes_slide.notes_text_frame.text = (
        "Walk left to right. Three modality vectors → concat → standardise → LR → conformal. "
        "Emphasise: 166 coefficients tell you exactly which features matter — this is interpretable."
    )
    return s


def slide_conformal(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "Conformal prediction primer",
           "Vovk 2005 · inductive (split) classifier")

    # left: the algorithm
    add_card(s, 0.5, 1.3, 4.4, 3.4, fill=DIVIDE)
    add_text(s, 0.7, 1.4, 4.0, 0.35, "ALGORITHM",
             font=BODY_FONT, size=10, color=GOLD, bold=True)
    add_text(s, 0.7, 1.75, 4.0, 0.4,
             "1.  reserve last 20 % of train as calibration",
             font=BODY_FONT, size=11, color=WHITE)
    add_text(s, 0.7, 2.2, 4.0, 0.4,
             "2.  nonconformity  α_i = 1 − p(y_i | x_i)",
             font=BODY_FONT, size=11, color=WHITE)
    add_text(s, 0.7, 2.65, 4.0, 0.65,
             "3.  threshold τ = quantile(α,\n     ⌈(n+1)(1 − α)⌉ / n)",
             font=BODY_FONT, size=11, color=WHITE)
    add_text(s, 0.7, 3.4, 4.0, 0.4,
             "4.  C(x) = { c :  1 − p(c | x) ≤ τ }",
             font=BODY_FONT, size=11, color=WHITE)

    # right: the guarantee + result
    add_card(s, 5.1, 1.3, 4.4, 1.55, fill=GOLD)
    add_text(s, 5.3, 1.4, 4.0, 0.3, "COVERAGE GUARANTEE",
             font=BODY_FONT, size=10, color=NAVY, bold=True)
    add_text(s, 5.3, 1.75, 4.0, 0.5,
             "P( y ∈ C(x) )  ≥  1 − α",
             font=HEAD_FONT, size=16, color=NAVY, bold=True)
    add_text(s, 5.3, 2.3, 4.0, 0.5,
             "under exchangeability of calibration + test",
             font=BODY_FONT, size=10, color=NAVY, italic=True)

    add_card(s, 5.1, 3.05, 4.4, 1.65, fill=DIVIDE)
    add_text(s, 5.3, 3.15, 4.0, 0.3, "MEASURED ON BTC 1H",
             font=BODY_FONT, size=10, color=GOLD, bold=True)
    add_text(s, 5.3, 3.5, 4.0, 0.5,
             "empirical coverage 0.883",
             font=HEAD_FONT, size=15, color=WHITE, bold=True)
    add_text(s, 5.3, 4.0, 4.0, 0.35,
             "nominal α = 0.10  →  target 0.90",
             font=BODY_FONT, size=11, color=ICE)
    add_text(s, 5.3, 4.35, 4.0, 0.35,
             "Δ = 1.7 pp  →  calibration is honest",
             font=BODY_FONT, size=11, color=GOLD, italic=True)

    add_footer(s, 11, total=20)
    s.notes_slide.notes_text_frame.text = (
        "Deliver the four algorithm steps clearly. Then point at the coverage guarantee — "
        "'this is why 88% coverage vs 90% nominal is the calibration honesty proof.'"
    )
    return s


def slide_ablation(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    header(s, "RO5 · Ablation study",
           "each modality's marginal contribution isolated")

    # header row
    hy = 1.35
    add_card(s, 0.5, hy, 9, 0.42, fill=GOLD)
    add_text(s, 0.7,  hy + 0.05, 3.0, 0.35, "VARIANT",
             font=BODY_FONT, size=11, color=NAVY, bold=True)
    add_text(s, 4.0,  hy + 0.05, 1.3, 0.35, "Accuracy",
             font=BODY_FONT, size=11, color=NAVY, bold=True, align="center")
    add_text(s, 5.3,  hy + 0.05, 1.3, 0.35, "MCC",
             font=BODY_FONT, size=11, color=NAVY, bold=True, align="center")
    add_text(s, 6.6,  hy + 0.05, 1.4, 0.35, "Sharpe/bar",
             font=BODY_FONT, size=11, color=NAVY, bold=True, align="center")
    add_text(s, 8.0,  hy + 0.05, 1.4, 0.35, "PnL log",
             font=BODY_FONT, size=11, color=NAVY, bold=True, align="center")

    rows = [
        ("fusion_num",             "0.5374", "0.078", "0.003", "0.74", False),
        ("fusion_num_cnn",         "0.5492", "0.102", "0.024", "4.93", True),
        ("fusion_num_sent",        "0.5374", "0.078", "0.003", "0.74", False),
        ("fusion_num_cnn_sent",    "0.5492", "0.102", "0.024", "4.93", True),
    ]
    for i, (name, acc, mcc, sharpe, pnl, hi) in enumerate(rows):
        y = 1.85 + i * 0.45
        fill = DIVIDE if not hi else DIVIDE
        add_card(s, 0.5, y, 9, 0.42, fill=fill)
        col = GOLD if hi else WHITE
        add_text(s, 0.7, y + 0.05, 3.3, 0.35, name,
                 font="Consolas" if False else BODY_FONT,
                 size=11, color=col, bold=hi)
        for j, val in enumerate([acc, mcc, sharpe, pnl]):
            x = 4.0 + j * 1.35 if j < 3 else 4.0 + j * 1.4
            # simpler: predefined columns
        add_text(s, 4.0, y + 0.05, 1.3, 0.35, acc,
                 font=HEAD_FONT, size=12, color=col, bold=hi, align="center")
        add_text(s, 5.3, y + 0.05, 1.3, 0.35, mcc,
                 font=HEAD_FONT, size=12, color=col, bold=hi, align="center")
        add_text(s, 6.6, y + 0.05, 1.4, 0.35, sharpe,
                 font=HEAD_FONT, size=12, color=col, bold=hi, align="center")
        add_text(s, 8.0, y + 0.05, 1.4, 0.35, pnl,
                 font=HEAD_FONT, size=12, color=col, bold=hi, align="center")

    add_text(s, 0.5, 3.95, 9, 0.4,
             "Vision (CNN) is the marginal contributor  →  +1.2 pp accuracy, +4.2 PnL_log",
             font=BODY_FONT, size=12, color=WHITE, bold=True)
    add_text(s, 0.5, 4.35, 9, 0.4,
             "Sentiment adds zero  →  the regression correctly refused to overfit sparse features",
             font=BODY_FONT, size=12, color=ICE, italic=True)

    add_footer(s, 14, total=20)
    s.notes_slide.notes_text_frame.text = (
        "Read the gold rows (with CNN). Emphasise: identical acc for num_cnn and num_cnn_sent "
        "= sentiment weight is ZERO. That's the model correctly refusing to overfit."
    )
    return s


# =============================================================== assemble
def build():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625)

    # 20-slide expanded research edition
    slide_title(prs)             #  1
    slide_problem(prs)           #  2
    slide_related_work(prs)      #  3 ⚡
    slide_aim_rqs(prs)           #  4 ⭐
    slide_walk_forward(prs)      #  5 ⚡
    slide_architecture(prs)      #  6 ⭐
    slide_numerical_features(prs)#  7 ⚡
    slide_cnn_arch(prs)          #  8 ⚡
    slide_sentiment(prs)         #  9 ⚡
    slide_fusion_arch(prs)       # 10 ⭐
    slide_conformal(prs)         # 11 ⚡
    slide_ro1_4(prs)             # 12
    slide_ro5(prs)               # 13 ⭐
    slide_ablation(prs)          # 14 ⚡
    slide_ro6(prs)               # 15
    slide_ro7(prs)               # 16
    slide_ro8(prs)               # 17
    slide_mapping(prs)           # 18 ⭐
    slide_defence(prs)           # 19
    slide_close(prs)             # 20

    out = Path(__file__).resolve().parent.parent / "INTERIM_PRESENTATION_JULY7.pptx"
    prs.save(str(out))
    print(f"[OK] wrote {out}")
    print("20 slides — expanded research edition")
    print("Open it and rehearse the SAY blocks from INTERIM_SCRIPT_JULY7.md")


if __name__ == "__main__":
    build()
