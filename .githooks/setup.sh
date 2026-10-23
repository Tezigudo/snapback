#!/usr/bin/env bash
# Run once after cloning: enables the repo-tracked git hooks.
set -euo pipefail
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
echo "✓ git hooks enabled (core.hooksPath = .githooks)"
