#!/bin/sh
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=sk-dummy

export OPENAI_DEFAULT_MODEL=${1:-aya-expanse:8b}
code .
