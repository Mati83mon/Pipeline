#!/usr/bin/env bash
#
# Deploy this repository to a Hugging Face Space.
#
#   ./deploy.sh <hf-username> [space-name] [flavor]
#
# Uses the modern `hf` CLI. The old `huggingface-cli repo create --hardware
# gpu-t4 --yes` invocation in the first draft used three flags that do not
# exist. Hardware here is a real `--flavor` value, and `--secrets HF_TOKEN`
# forwards the token from your local login into the Space, so no manual step
# is left over.
#
# FLAVOR defaults to zero-a10g, which is how ZeroGPU is named in the API.

set -euo pipefail

USERNAME="${1:-}"
SPACE_NAME="${2:-ai-media-pipeline}"
FLAVOR="${3:-zero-a10g}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "$USERNAME" ]]; then
    echo "usage: ./deploy.sh <hf-username> [space-name]" >&2
    echo "   eg: ./deploy.sh Mati83moni ai-media-pipeline" >&2
    exit 2
fi

REPO_ID="${USERNAME}/${SPACE_NAME}"

if ! command -v hf >/dev/null 2>&1; then
    echo "==> installing the Hugging Face CLI"
    pip install -U "huggingface_hub>=1.0" >/dev/null
fi

if ! hf auth whoami >/dev/null 2>&1; then
    echo "not logged in. run: hf auth login   (needs a WRITE token)" >&2
    exit 1
fi

echo "==> validating the app before uploading"
python -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from pipeline.config import load_config
config = load_config('${REPO_ROOT}/config.json')
print('    config ok; modules:', ', '.join(config.enabled_modules))
"

if command -v pytest >/dev/null 2>&1; then
    ( cd "$REPO_ROOT" && python -m pytest tests/ -q ) || {
        echo "tests failed; not deploying" >&2
        exit 1
    }
fi

echo "==> creating Space ${REPO_ID} on ${FLAVOR} (skipped if it already exists)"
hf repos create "$REPO_ID" \
    --type space \
    --sdk gradio \
    --flavor "$FLAVOR" \
    --secrets HF_TOKEN \
    --private \
    --exist-ok

echo "==> uploading"
hf upload "$REPO_ID" "$REPO_ROOT" . \
    --type space \
    --commit-message "Deploy AI Media Pipeline" \
    --exclude "*.pyc" \
    --exclude "__pycache__/*" \
    --exclude ".git/*" \
    --exclude "outputs/*" \
    --exclude "hf-cache/*" \
    --exclude "model_cache/*" \
    --exclude ".pytest_cache/*"

cat <<EOF

Deployed: https://huggingface.co/spaces/${REPO_ID}

Hardware (${FLAVOR}) and the HF_TOKEN secret were set at creation time. If the
Space already existed, those flags were ignored — check them under Settings.

HF_TOKEN only unlocks the gated default text model if the account behind it has
accepted that model's licence. Without it the Text tab falls back to an open
model and says so in the run info.

The first run of each tab downloads tens of gigabytes. Watch the Space logs.
EOF
