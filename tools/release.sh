#!/usr/bin/env bash
# Release dev -> main without the author's private / AI-assistant working files.
#
#   bash tools/release.sh                 # update main from dev
#   bash tools/release.sh paper-2026-10   # ... and tag the release (the tag a paper cites)
#
# Works in a separate git worktree, so the working directory (and any training campaign running from
# it) is never touched. Pushes main (and the tag) to origin. docs/DEVELOPMENT.md describes the branches.
set -euo pipefail

# Paths that stay on dev but are never released to main (one list, edit here).
EXCLUDE=(CLAUDE.md NotesForAI EthanTravelDocs report.md)

TAG="${1:-}"
REPO="$(git rev-parse --show-toplevel)"
WT="$(mktemp -d)/release-main"

git -C "$REPO" fetch -q origin
git -C "$REPO" push -q origin dev
git -C "$REPO" worktree add -q "$WT" main
trap 'git -C "$REPO" worktree remove --force "$WT"' EXIT

cd "$WT"
git merge -q --no-ff --no-commit dev -m "Release dev to main" || true   # excluded paths may conflict
git rm -r -q --cached --ignore-unmatch -- "${EXCLUDE[@]}"
rm -rf -- "${EXCLUDE[@]}"
if git diff --name-only --diff-filter=U | grep -q .; then
    echo "Unresolved merge conflicts outside the excluded paths:" >&2
    git diff --name-only --diff-filter=U >&2
    exit 1
fi
git commit -q -m "Release dev to main (excluding: ${EXCLUDE[*]})" || echo "main already up to date"
git push -q origin main
if [ -n "$TAG" ]; then
    git tag -a "$TAG" -m "Release $TAG"
    git push -q origin "$TAG"
    echo "Tagged $TAG at $(git rev-parse --short HEAD)"
fi
echo "main is at $(git rev-parse --short HEAD)"
