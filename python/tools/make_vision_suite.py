"""Generates the image test suite of the Qwen3.5 image input: bench/raw/vision-suite/*.png + cases.json (deterministic, small, plain backgrounds).

    python python/tools/make_vision_suite.py [out dir]        (default bench/raw/vision-suite)

Every case: {id, image, question, expect_all: [substrings], expect_any: [substrings], expect_regex: [regexes], notes}; an answer PASSES when it contains every `expect_all` substring, at least one `expect_any`
substring (when the list is not empty) and matches every `expect_regex` (all case-insensitive). The same scoring is implemented by python/tools/qwen35_vision_quality.py and bench/raw/vision-suite.js.
Fonts: Arial (Windows) when present, else DejaVu / the scalable default font of Pillow; the shapes do not depend on a font, the text cases do a little (the pixels differ, the content does not).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
FONT_FILES = {"bold": ["arialbd.ttf", "DejaVuSans-Bold.ttf"], "regular": ["arial.ttf", "DejaVuSans.ttf"]}

RED, GREEN, BLUE, BLACK, WHITE = (220, 30, 30), (30, 160, 60), (30, 70, 220), (0, 0, 0), (255, 255, 255)


def font(size: int, weight: str = "regular"):
    for name in FONT_FILES[weight]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def canvas(width: int, height: int, background=WHITE):
    image = Image.new("RGB", (width, height), background)
    return image, ImageDraw.Draw(image)


def centered_text(draw, box, text, f, fill=BLACK):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=f)
    x = box[0] + (box[2] - box[0] - (right - left)) // 2 - left
    y = box[1] + (box[3] - box[1] - (bottom - top)) // 2 - top
    draw.text((x, y), text, font=f, fill=fill)


def large_text():
    image, draw = canvas(640, 384)
    centered_text(draw, (0, 0, 640, 384), "WebNN 2026", font(110, "bold"))
    return image


def small_text():
    image, draw = canvas(640, 480)
    f = font(30)
    for i, line in enumerate(["Order 48213", "Date 2026-11-26", "Item Widget x3", "Total 127.50 USD", "Thank you"]):
        draw.text((40, 40 + i * 80), line, font=f, fill=BLACK)
    return image


def red_circle():
    image, draw = canvas(384, 384)
    draw.ellipse((92, 92, 292, 292), fill=RED)
    return image


def five_squares():
    image, draw = canvas(640, 320)
    for i in range(5):
        x = 30 + i * 118
        draw.rectangle((x, 110, x + 80, 190), fill=BLUE)
    return image


def left_right():
    image, draw = canvas(512, 384)
    draw.rectangle((50, 120, 210, 280), fill=RED)
    draw.rectangle((302, 120, 462, 280), fill=GREEN)
    return image


def bar_chart():
    image, draw = canvas(512, 512)
    draw.line((60, 440, 480, 440), fill=BLACK, width=3)
    draw.line((60, 440, 60, 40), fill=BLACK, width=3)
    for i, (label, height) in enumerate(zip("ABCD", (150, 250, 190, 360))):   # D is the tallest
        x = 100 + i * 95
        draw.rectangle((x, 440 - height, x + 60, 438), fill=(60, 110, 200))
        centered_text(draw, (x, 448, x + 60, 490), label, font(36, "bold"))
    return image


def banner():
    image, draw = canvas(768, 256, (255, 235, 150))
    draw.rectangle((6, 6, 761, 249), outline=(160, 40, 40), width=6)
    centered_text(draw, (0, 0, 768, 256), "BANNER", font(120, "bold"), (160, 40, 40))
    return image


def tall_triangle():
    image, draw = canvas(384, 768)
    draw.polygon([(192, 200), (62, 560), (322, 560)], fill=GREEN)
    return image


CASES = [
    ("ocr-large", large_text, "What text is written in the image? Reply with the text only.", ["webnn", "2026"], [], [], "large printed text: the baseline OCR case"),
    ("ocr-small", small_text, "What is the order number and what is the total? Reply briefly.", ["48213", "127.50"], [], [], "small printed text lines: resolution test, a bigger grid helps (16x16 loses it)"),
    ("color-circle", red_circle, "What color is the circle?", ["red"], [], [], "one red circle on white"),
    ("count-squares", five_squares, "How many squares are there? Answer with a number.", [], ["5", "five"], [], "five blue squares in a row"),
    ("left-right", left_right, "Which color is on the left? Answer with one word.", ["red"], [], [], "red square left, green square right"),
    ("bar-chart", bar_chart, "Which bar is the tallest? Answer with the letter only.", [], [], ["\\bD\\b"], "bars A B C D, D is the tallest"),
    ("banner-wide", banner, "What word is written on the banner? Reply with the word only.", ["banner"], [], [], "3:1 wide banner: tests the aspect-ratio preset choice"),
    ("tall-triangle", tall_triangle, "What shape is this and what color is it?", ["triangle", "green"], [], [], "1:2 tall image with a green triangle"),
]


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "bench" / "raw" / "vision-suite"
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for case_id, make, question, expect_all, expect_any, expect_regex, notes in CASES:
        image = make()
        image.save(out / f"{case_id}.png", optimize=True)
        cases.append({"id": case_id, "image": f"{case_id}.png", "width": image.width, "height": image.height, "question": question, "expect_all": expect_all, "expect_any": expect_any,
                      "expect_regex": expect_regex, "notes": notes})
    (out / "cases.json").write_text(json.dumps({"cases": cases}, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
