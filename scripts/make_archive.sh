#!/bin/sh
# Build a release archive from git, never from the working folder.
# git archive only contains tracked files, so the gitignored personal notes
# (OUTREACH.md, EMAILS.md, HANDOFF.md, PAPER.md, AGENTS.md) cannot end up in it.
set -e
cd "$(dirname "$0")/.."
ref="${1:-HEAD}"
name="Token-Thermodynamics-$(git describe --tags --always "$ref")"
git archive --format=zip --prefix="$name/" -o "$name.zip" "$ref"
if unzip -l "$name.zip" | grep -E "OUTREACH|EMAILS|HANDOFF|PAPER\.md|AGENTS\.md"; then
    echo "private file found in archive, aborting" >&2
    rm "$name.zip"
    exit 1
fi
echo "wrote $name.zip"
