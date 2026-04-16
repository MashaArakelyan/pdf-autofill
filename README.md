# Easy PDF Autofill (OpenRouter)

Simple script for the first (easy) challenge:
- fetch JSON from Interaction
- download the PDF attachment
- read its form fields
- ask OpenRouter to propose values from email context
- save a filled PDF

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY="your_key_here"
```

## Run

```bash
python fill_easy_pdf.py
```

Outputs:
- `out/easy-original.pdf`
- `out/easy-filled.pdf`

## Useful options

```bash
python fill_easy_pdf.py --output out/my-filled.pdf --model anthropic/claude-3.5-sonnet
```

To point at a different challenge JSON:

```bash
python fill_easy_pdf.py --json-url "https://interaction.co/assets/easy-pdf.json"
```

Example OpenRouter model names:
- `anthropic/claude-3.5-sonnet`
- `openai/gpt-4o-mini`

If your machine has certificate-chain issues in the Python runtime, you can use:

```bash
python fill_easy_pdf.py --insecure
```
