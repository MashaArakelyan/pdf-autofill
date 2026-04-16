#!/bin/bash
# Paste your Anthropic API key below and run: bash set_key.sh
KEY="PASTE_YOUR_KEY_HERE"

sed -i '' "s|ANTHROPIC_API_KEY=\".*\"|ANTHROPIC_API_KEY=\"$KEY\"|" "$(dirname "$0")/run.sh"
echo "API key updated in run.sh"
