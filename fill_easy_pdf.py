#!/usr/bin/env python3
"""
End-to-end "easy form" pipeline:
1) Fetch challenge JSON
2) Download PDF attachment
3) Read PDF form fields
4) Ask OpenRouter to generate best-effort field values from email context
5) Save a filled PDF
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests
from pypdf import PdfReader, PdfWriter
from pypdf.generic import BooleanObject, NameObject


DEFAULT_JSON_URL = "https://interaction.co/assets/easy-pdf.json"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def fetch_json(url: str, verify_ssl: bool = True) -> dict[str, Any]:
    response = requests.get(url, timeout=30, verify=verify_ssl)
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
        mime_type = str(node.get("mimeType", "")).lower()
        if "pdf" in mime_type:
            pdf_parts.append(node)
        parts = node.get("parts", [])
        if isinstance(parts, list):
            stack.extend(parts)

    if not pdf_parts:
        raise ValueError("No PDF attachment found in payload.")

    linked_pdf = next((p for p in pdf_parts if p.get("attachmentLink")), None)
    return linked_pdf or pdf_parts[0]


def download_pdf_bytes(pdf_part: dict[str, Any], verify_ssl: bool = True) -> bytes:
    link = pdf_part.get("attachmentLink")
    if link:
        response = requests.get(str(link), timeout=60, verify=verify_ssl)
        response.raise_for_status()
        return response.content

    content = pdf_part.get("content")
    if content:
        import base64

        try:
            return base64.b64decode(content)
        except Exception as exc:
            raise ValueError("Failed to decode inline PDF content.") from exc

    raise ValueError("PDF part has neither attachmentLink nor inline content.")


def get_form_field_names(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    fields = reader.get_fields()
    if not fields:
        return []
    return list(fields.keys())


def build_email_context(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "sender": source.get("sender", {}),
        "recipients": source.get("recipients", []),
        "cc": source.get("cc", []),
        "subject": source.get("subject", ""),
        "snippet": source.get("snippet", ""),
    }


def parse_json_from_text(text: str) -> dict[str, str]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        return {str(k): str(v) for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Model response did not contain JSON object.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Model JSON response was not an object.")
    return {str(k): str(v) for k, v in parsed.items()}


def ask_openrouter_for_field_values(
    field_names: list[str],
    email_context: dict[str, Any],
    model: str = DEFAULT_OPENROUTER_MODEL,
    verify_ssl: bool = True,
) -> dict[str, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    system_prompt = (
        "You fill PDF form fields using provided context and reasonable inferences. "
        "Return only a compact JSON object mapping exact field names to values. "
        "For fields inferable from the email (names, dates, addresses), use those values. "
        "For fields with no context (IDs, plate numbers, prices), invent plausible placeholder values. "
        "Never leave a field empty — always provide a best-effort value."
    )

    user_prompt = {
        "email_context": email_context,
        "field_names": field_names,
        "requirements": [
            "Use exact field names as JSON keys.",
            "Fill ALL fields with best-effort or placeholder values — no empty strings.",
            "Infer seller/buyer names from the email sender/recipient where applicable.",
            "Do not include explanation, markdown, or extra text.",
        ],
    }

    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 1200,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional but recommended by OpenRouter for app identification.
        "HTTP-Referer": "https://localhost",
        "X-Title": "easy-pdf-autofill",
    }

    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
        verify=verify_ssl,
    )
    response.raise_for_status()
    body = response.json()

    choices = body.get("choices", [])
    if not choices:
        raise ValueError("OpenRouter returned no choices.")

    message_content = choices[0].get("message", {}).get("content", "")
    if not isinstance(message_content, str) or not message_content.strip():
        raise ValueError("OpenRouter returned empty message content.")

    return parse_json_from_text(message_content)


def fill_pdf_form(pdf_bytes: bytes, field_values: dict[str, str], output_path: Path) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill the easy PDF challenge form with OpenRouter.")
    parser.add_argument("--json-url", required=True, help="Source JSON URL")
    parser.add_argument(
        "--output",
        default="out/easy-filled.pdf",
        help="Output path for filled PDF",
    )
    parser.add_argument(
        "--save-original",
        default="out/easy-original.pdf",
        help="Optional path to save downloaded original PDF",
    )
    parser.add_argument("--model", default=DEFAULT_OPENROUTER_MODEL, help="OpenRouter model ID")
    parser.add_argument(
        "--list-fields-only",
        action="store_true",
        help="Download and print discovered field names, then exit",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS cert verification for debugging in constrained environments",
    )
    args = parser.parse_args()

    verify_ssl = not args.insecure

    try:
        source = fetch_json(args.json_url, verify_ssl=verify_ssl)
        pdf_part = extract_pdf_attachment(source)
        pdf_bytes = download_pdf_bytes(pdf_part, verify_ssl=verify_ssl)

        save_original_path = Path(args.save_original)
        save_original_path.parent.mkdir(parents=True, exist_ok=True)
        save_original_path.write_bytes(pdf_bytes)

        fields = get_form_field_names(pdf_bytes)
        if not fields:
            raise ValueError("No form fields discovered in PDF.")

        if args.list_fields_only:
            print(f"Original PDF saved to: {save_original_path}")
            print("Discovered fields:")
            for name in fields:
                print(f" - {name}")
            return 0

        email_context = build_email_context(source)
        field_values = ask_openrouter_for_field_values(
            fields,
            email_context,
            model=args.model,
            verify_ssl=verify_ssl,
        )

        # Keep only keys that exist in the PDF form.
        normalized_values = {name: str(field_values.get(name, "")) for name in fields}
        output_path = Path(args.output)
        fill_pdf_form(pdf_bytes, normalized_values, output_path)

        print("Success.")
        print(f"Original PDF saved to: {save_original_path}")
        print(f"Filled PDF saved to:   {output_path}")
        print("Discovered fields:")
        for name in fields:
            print(f" - {name}: {normalized_values.get(name, '')}")
        return 0
    except Exception as exc:  # broad catch for challenge scripting speed
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
