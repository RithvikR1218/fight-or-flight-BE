#!/usr/bin/env bash
# Download the large data files that aren't in git (GitHub limits files to 100 MB)
# from this repo's GitHub Release into data/, and verify their checksums.
#
#   ./scripts/fetch_data.sh             # buildings source, prebuilt 3D tiles, risk model crime history
#   ./scripts/fetch_data.sh --no-tiles  # skip tiles (rebuild with build_buildings.sh)
#   ./scripts/fetch_data.sh --no-crime  # skip crime.db (rebuild with backfill_crime + export_crime_db)
set -euo pipefail

REPO="${DATA_RELEASE_REPO:-RithvikR1218/fight-or-flight-BE}"
TAG="${DATA_RELEASE_TAG:-data-v2}"   # data-v2: 5-mile tiles; data-v1: 10-mile
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TILES=1 CRIME=1
for arg in "$@"; do
  case "$arg" in
    --no-tiles) TILES=0 ;;
    --no-crime) CRIME=0 ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done

FILES=("data/raw/LARIAC6_Buildings_2020.gdb.zip")
[[ $TILES == 1 ]] && FILES+=("data/tiles/buildings.pmtiles")
# The risk model's LAPD 2020-2024 history (scripts/export_crime_db.py), mounted into the API at /app/algorithm/data.
[[ $CRIME == 1 ]] && FILES+=("risk-model/data/crime.db")

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
expected() { grep "  $1\$" data/checksums.sha256 | cut -d' ' -f1; }

for path in "${FILES[@]}"; do
  want="$(expected "$path")"
  if [[ -f "$path" && "$(sha256 "$path")" == "$want" ]]; then
    echo "ok       $path (already present)"
    continue
  fi
  url="https://github.com/$REPO/releases/download/$TAG/$(basename "$path")"
  echo "fetching $path"
  mkdir -p "$(dirname "$path")"
  curl -fL --retry 5 --retry-delay 3 -C - -o "$path.part" "$url"
  got="$(sha256 "$path.part")"
  if [[ -n "$want" && "$got" != "$want" ]]; then
    echo "checksum mismatch for $path (got $got, want $want); delete $path.part and retry" >&2
    exit 1
  fi
  mv "$path.part" "$path"
  echo "ok       $path"
done
