#!/usr/bin/env python3
"""
Unified PDF form filler.
Detects whether a PDF has interactive AcroForm fields (easy) or is image-based (hard),
then uses the appropriate LLM strategy to fill it.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import anthropic
import fitz  # pymupdf
import pytesseract
import requests
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import BooleanObject, NameObject

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

DEFAULT_MODEL = "claude-sonnet-4-5"


# ---------- Shared utilities ----------

def fetch_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Expected top-level JSON object.")
    return data


def extract_pdf_attachment(payload: dict[str, Any]) -> dict[str, Any]:
    if "payload" not in payload or not isinstance(payload["payload"], dict):
        raise ValueError("Input JSON missing valid 'payload' object.")

    pdf_parts: list[dict[str, Any]] = []
    stack = [payload["payload"]]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        if "pdf" in str(node.get("mimeType", "")).lower():
            pdf_parts.append(node)
        parts = node.get("parts", [])
        if isinstance(parts, list):
            stack.extend(parts)

    if not pdf_parts:
        raise ValueError("No PDF attachment found in payload.")
    return next((p for p in pdf_parts if p.get("attachmentLink")), None) or pdf_parts[0]


def download_pdf_bytes(pdf_part: dict[str, Any]) -> bytes:
    link = pdf_part.get("attachmentLink")
    if link:
        response = requests.get(str(link), timeout=60)
        response.raise_for_status()
        return response.content
    content = pdf_part.get("content")
    if content:
        try:
            return base64.b64decode(content)
        except Exception as exc:
            raise ValueError("Failed to decode inline PDF content.") from exc
    raise ValueError("PDF part has neither attachmentLink nor inline content.")


def build_email_context(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "sender": source.get("sender", {}),
        "recipients": source.get("recipients", []),
        "cc": source.get("cc", []),
        "subject": source.get("subject", ""),
        "snippet": source.get("snippet", ""),
    }


def detect_pdf_type(pdf_bytes: bytes) -> str:
    """Returns 'form' if interactive AcroForm fields exist, 'image' otherwise."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    if reader.get_fields():
        return "form"
    for page in reader.pages:
        if page.extract_text().strip():
            return "form"
    return "image"


def parse_json_from_text(text: str) -> Any:
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting the largest valid JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Model response did not contain a JSON object.")
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Response was truncated — rescue complete field objects
        fields = re.findall(r'\{[^{}]*"label"[^{}]*"value"[^{}]*\}', candidate, re.DOTALL)
        if fields:
            complete = [json.loads(f) for f in fields if _try_parse(f)]
            if complete:
                return {"fields": complete}
        raise ValueError("Could not parse model response as JSON.")


def _try_parse(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def call_openrouter(messages: list[dict], model: str, max_tokens: int = 1500) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)

    # Convert OpenAI-style messages to Anthropic format
    system = ""
    anthropic_messages = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"] if isinstance(m["content"], str) else ""
        else:
            content = m["content"]
            if isinstance(content, str):
                anthropic_messages.append({"role": m["role"], "content": content})
            elif isinstance(content, list):
                # Vision message — convert image_url blocks to Anthropic format
                parts = []
                for block in content:
                    if block.get("type") == "image_url":
                        url = block["image_url"]["url"]
                        if url.startswith("data:"):
                            media_type, b64data = url.split(";base64,")
                            media_type = media_type.split(":")[1]
                            parts.append({
                                "type": "image",
                                "source": {"type": "base64", "media_type": media_type, "data": b64data},
                            })
                    elif block.get("type") == "text":
                        parts.append({"type": "text", "text": block["text"]})
                anthropic_messages.append({"role": m["role"], "content": parts})

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=anthropic_messages,
    )
    return response.content[0].text


# ---------- Easy path (AcroForm fields) ----------

def ask_for_field_values(
    field_names: list[str], email_context: dict[str, Any], model: str
) -> dict[str, str]:
    messages = [
        {
            "role": "system",
            "content": (
                "You fill PDF form fields using provided context and reasonable inferences. "
                "Return only a compact JSON object mapping exact field names to values. "
                "For fields inferable from the email (names, dates, addresses), use those values. "
                "For unknown fields invent plausible placeholder values. "
                "Never leave a field empty."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "email_context": email_context,
                "field_names": field_names,
                "requirements": [
                    "Use exact field names as JSON keys.",
                    "Fill ALL fields — no empty strings.",
                    "Infer seller/buyer names from the email sender/recipient.",
                    "Return only the JSON object, no markdown or explanation.",
                ],
            }),
        },
    ]
    raw = call_openrouter(messages, model)
    parsed = parse_json_from_text(raw)
    return {str(k): str(v) for k, v in parsed.items()}


def fill_acroform_pdf(pdf_bytes: bytes, field_values: dict[str, str], output_path: Path) -> None:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    writer.append(reader)
    for page in writer.pages:
        writer.update_page_form_field_values(page, field_values, auto_regenerate=False)
    if "/AcroForm" in writer._root_object:
        writer._root_object["/AcroForm"].update(
            {NameObject("/NeedAppearances"): BooleanObject(True)}
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        writer.write(f)


# ---------- Hard path (fixed-layout California Bill of Sale) ----------

# Field layout for California DMV Bill of Sale (REG 135), bottom copy.
# Positions are fractions of page width (x, w) and fractions of bottom-half height (y).
# y=0.0 = top of bottom copy, y=1.0 = bottom of page.
# Each entry: (field_key, label_for_model, x_frac, y_frac, w_frac)
FORM_LAYOUT = [
    ("vin",           "Vehicle Identification Number (VIN)",       0.145, 0.13, 0.125),
    ("year",          "Year Model",                                0.285, 0.13, 0.055),
    ("make",          "Make",                                      0.350, 0.13, 0.066),
    ("license",       "License Plate Number",                      0.428, 0.13, 0.097),
    ("engine",        "Motorcycle Engine Number (N/A if none)",    0.510, 0.13, 0.110),
    ("seller_name",   "Seller Full Name (print)",                  0.020, 0.225, 0.245),
    ("buyer_name",    "Buyer Full Name (print)",                   0.515, 0.225, 0.245),
    ("month",         "Sale Month (2 digits)",                     0.078, 0.320, 0.036),
    ("day",           "Sale Day (2 digits)",                       0.126, 0.320, 0.030),
    ("year2",         "Sale Year (2 digits, e.g. 23)",             0.165, 0.320, 0.030),
    ("price",         "Selling Price (numbers only)",              0.400, 0.320, 0.115),
    ("relationship",  "Relationship to buyer if gift, else N/A",   0.298, 0.400, 0.135),
    ("gift_value",    "Gift value if gift, else 0",                0.520, 0.400, 0.095),
    ("seller_print1", "Seller Print Name (signature line 1)",      0.020, 0.510, 0.165),
    ("seller_date1",  "Seller Date (signature line 1)",            0.213, 0.510, 0.088),
    ("seller_dl1",    "Seller DL or Dealer Number (line 1)",       0.332, 0.510, 0.105),
    ("seller_print2", "Seller Print Name (signature line 2)",      0.020, 0.610, 0.165),
    ("seller_date2",  "Seller Date (signature line 2)",            0.213, 0.610, 0.088),
    ("seller_dl2",    "Seller DL or Dealer Number (line 2)",       0.332, 0.610, 0.105),
    ("seller_addr",   "Seller Mailing Address",                    0.020, 0.710, 0.246),
    ("seller_city",   "Seller City",                               0.288, 0.710, 0.106),
    ("seller_state",  "Seller State (2-letter)",                   0.408, 0.710, 0.037),
    ("seller_zip",    "Seller ZIP Code",                           0.457, 0.710, 0.058),
    ("seller_phone",  "Seller Daytime Phone",                      0.567, 0.710, 0.106),
    ("buyer_print1",  "Buyer Print Name (line 1)",                 0.020, 0.800, 0.230),
    ("buyer_print2",  "Buyer Print Name (line 2)",                 0.020, 0.873, 0.230),
    ("buyer_addr",    "Buyer Mailing Address",                     0.020, 0.938, 0.246),
    ("buyer_city",    "Buyer City",                                0.288, 0.938, 0.106),
    ("buyer_state",   "Buyer State (2-letter)",                    0.408, 0.938, 0.037),
    ("buyer_zip",     "Buyer ZIP Code",                            0.457, 0.938, 0.058),
]


def ask_for_field_values_layout(
    email_context: dict[str, Any], model: str
) -> dict[str, str]:
    """Ask the model for values for all hardcoded layout fields."""
    labels = {key: label for key, label, *_ in FORM_LAYOUT}
    messages = [
        {
            "role": "system",
            "content": (
                "You fill a California DMV Bill of Sale form. "
                "Return ONLY a JSON object mapping each field key to its value. "
                "Use the email context for names/addresses. "
                "Invent plausible placeholders for unknown fields. Never use empty strings."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "email_context": email_context,
                "fields": labels,
                "instructions": "Return a flat JSON object: {field_key: value}. No markdown.",
            }),
        },
    ]
    raw = call_openrouter(messages, model, max_tokens=1500)
    parsed = parse_json_from_text(raw)
    return {str(k): str(v) for k, v in parsed.items()}


def ask_vision_for_fields(
    pdf_bytes: bytes, email_context: dict[str, Any], model: str
) -> tuple[list[dict[str, Any]], int, int]:
    """Build field list from hardcoded layout + model-supplied values."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    pw, ph = page.rect.width, page.rect.height

    values = ask_for_field_values_layout(email_context, model)

    fields = []
    for key, label, x_frac, y_frac, w_frac in FORM_LAYOUT:
        value = values.get(key, "").strip()
        if not value:
            continue
        # Convert fractions to absolute PDF points
        x = x_frac * pw
        y = ph / 2 + y_frac * (ph / 2)  # offset into bottom half
        w = w_frac * pw
        fields.append({"label": label, "value": value, "x": x, "y_baseline": y, "w": w})

    # img_w/img_h not used when coords are already in PDF points
    return fields, pw, ph


def overlay_text_on_pdf(
    pdf_bytes: bytes,
    fields: list[dict[str, Any]],
    img_w: float,
    img_h: float,
    output_path: Path,
) -> None:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    FIELD_H = 11.0  # points to white out above baseline

    for field in fields:
        value = str(field.get("value", "")).strip()
        if not value:
            continue
        x = field["x"]
        y = field["y_baseline"]
        w = field["w"]

        # White out the blank area
        page.draw_rect(fitz.Rect(x, y - FIELD_H, x + w, y + 1), color=(1, 1, 1), fill=(1, 1, 1))
        page.insert_text(fitz.Point(x + 1, y - 2), value, fontsize=8.5, color=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))


# ---------- Main ----------

def main() -> int:
    parser = argparse.ArgumentParser(description="Fill a PDF form using an LLM (auto-detects easy vs image-based).")
    parser.add_argument("--json-url", required=True, help="Source JSON URL containing the PDF attachment")
    parser.add_argument("--output", default="out/filled.pdf", help="Output path for filled PDF")
    parser.add_argument("--save-original", default="out/original.pdf", help="Path to save the original PDF")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model ID")
    parser.add_argument("--list-fields-only", action="store_true", help="Print discovered fields and exit (easy PDFs only)")
    args = parser.parse_args()

    try:
        source = fetch_json(args.json_url)
        pdf_part = extract_pdf_attachment(source)
        pdf_bytes = download_pdf_bytes(pdf_part)

        save_original_path = Path(args.save_original)
        save_original_path.parent.mkdir(parents=True, exist_ok=True)
        save_original_path.write_bytes(pdf_bytes)
        print(f"Original PDF saved to: {save_original_path}")

        pdf_type = detect_pdf_type(pdf_bytes)
        print(f"PDF type detected: {pdf_type}")

        email_context = build_email_context(source)
        output_path = Path(args.output)

        if pdf_type == "form":
            reader = PdfReader(io.BytesIO(pdf_bytes))
            fields = list(reader.get_fields().keys())
            if not fields:
                raise ValueError("No form fields discovered in PDF.")
            if args.list_fields_only:
                print("Discovered fields:")
                for name in fields:
                    print(f"  - {name}")
                return 0
            field_values = ask_for_field_values(fields, email_context, args.model)
            normalized = {name: str(field_values.get(name, "")) for name in fields}
            fill_acroform_pdf(pdf_bytes, normalized, output_path)
            print(f"Filled PDF saved to: {output_path}")
            print("Fields:")
            for name in fields:
                print(f"  - {name}: {normalized[name]}")

        else:
            if args.list_fields_only:
                print("Image-based PDF — use vision call to discover fields (omit --list-fields-only).")
                return 0
            vision_fields, img_w, img_h = ask_vision_for_fields(pdf_bytes, email_context, args.model)
            overlay_text_on_pdf(pdf_bytes, vision_fields, img_w, img_h, output_path)
            print(f"Filled PDF saved to: {output_path}")
            print("Fields overlaid:")
            for f in vision_fields:
                print(f"  - {f.get('label', '?')}: {f.get('value', '')}")

        import subprocess
        subprocess.run(["open", str(output_path)])

        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
