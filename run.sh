#!/bin/bash
source .venv/bin/activate
export OPENROUTER_API_KEY="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"
python3 fill_pdf.py "$@"
