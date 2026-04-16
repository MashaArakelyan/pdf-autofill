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

import fitz  # pymupdf
import requests
from pypdf import PdfReader, PdfWriter
from pypdf.generic import BooleanObject, NameObject

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"


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
    # Strip markdown code fences (handles multiline content inside)
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Model response did not contain a JSON object.")
    return json.loads(match.group(0))


def call_openrouter(messages: list[dict], model: str, max_tokens: int = 1500) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    response = requests.post(
        OPENROUTER_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "pdf-autofill",
        },
        json={"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": messages},
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices", [])
    if not choices:
        raise ValueError("OpenRouter returned no choices.")
    content = choices[0].get("message", {}).get("content", "")
    if not content.strip():
        raise ValueError("OpenRouter returned empty message content.")
    return content


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


# ---------- Hard path (image-based PDF) ----------

def render_pdf_page_to_png(pdf_bytes: bytes, dpi: int = 150) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    return pix.tobytes("png")


def ask_vision_for_fields(
    pdf_bytes: bytes, email_context: dict[str, Any], model: str
) -> list[dict[str, Any]]:
    png = render_pdf_page_to_png(pdf_bytes)
    b64 = base64.b64encode(png).decode()

    messages = [
        {
            "role": "system",
            "content": (
                "You analyze a scanned PDF form image and return filled values with positions. "
                "Return only a JSON object with a 'fields' array. "
                "Each item must have: "
                "'label' (short field description), "
                "'value' (text to fill in — never empty, use plausible placeholders if unknown), "
                "'x' (0.0–1.0, normalized horizontal position where text should start), "
                "'y' (0.0–1.0, normalized vertical position — 0 is top of page). "
                "Use the email context to infer names, dates, and addresses."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {
                    "type": "text",
                    "text": json.dumps({
                        "email_context": email_context,
                        "instructions": (
                            "Identify every blank field in this form. "
                            "Return filled values and normalized x/y positions for each. "
                            "No markdown, no explanation — only the JSON object."
                        ),
                    }),
                },
            ],
        },
    ]

    raw = call_openrouter(messages, model, max_tokens=3000)
    parsed = parse_json_from_text(raw)
    fields = parsed.get("fields", [])
    if not isinstance(fields, list):
        raise ValueError("Vision model response missing 'fields' array.")
    return fields


def overlay_text_on_pdf(
    pdf_bytes: bytes, fields: list[dict[str, Any]], output_path: Path
) -> None:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    pw, ph = page.rect.width, page.rect.height

    for field in fields:
        value = str(field.get("value", "")).strip()
        if not value:
            continue
        x = float(field.get("x", 0)) * pw
        y = float(field.get("y", 0)) * ph
        page.insert_text(fitz.Point(x, y), value, fontsize=9, color=(0, 0, 0.7))

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
            vision_fields = ask_vision_for_fields(pdf_bytes, email_context, args.model)
            overlay_text_on_pdf(pdf_bytes, vision_fields, output_path)
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
