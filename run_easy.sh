#!/bin/bash
source .venv/bin/activate
export ANTHROPIC_API_KEY="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"
python3 fill_easy_pdf.py --json-url "https://interaction.co/assets/easy-pdf.json"
