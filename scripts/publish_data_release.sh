#!/usr/bin/env bash
# Maintainers: upload the large data files to the GitHub Release and refresh
# data/checksums.sha256 (commit that file afterwards). Requires `gh auth login`.
#
#   ./scripts/publish_data_release.sh                           # every file, tag data-v2
#   ./scripts/publish_data_release.sh risk-model/data/crime.db  # only the files named
#   DATA_RELEASE_TAG=data-v2 ./scripts/publish_data_release.sh
set -euo pipefail

REPO="${DATA_RELEASE_REPO:-RithvikR1218/fight-or-flight-BE}"
TAG="${DATA_RELEASE_TAG:-data-v2}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ALL=("data/raw/LARIAC6_Buildings_2020.gdb.zip" "data/tiles/buildings.pmtiles" "risk-model/data/crime.db")
if [[ $# -gt 0 ]]; then FILES=("$@"); else FILES=("${ALL[@]}"); fi
for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }
  [[ " ${ALL[*]} " == *" $f "* ]] || { echo "$f is not a release file (${ALL[*]})" >&2; exit 1; }
done

# Refresh the checksum lines of the files being uploaded; keep the others.
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi; }
touch data/checksums.sha256
for f in "${FILES[@]}"; do
  grep -v "  $f\$" data/checksums.sha256 > data/checksums.sha256.tmp || true
  sha256 "$f" >> data/checksums.sha256.tmp
  mv data/checksums.sha256.tmp data/checksums.sha256
done
echo "updated data/checksums.sha256 (commit it)"

if gh release view "$TAG" -R "$REPO" >/dev/null 2>&1; then
  gh release upload "$TAG" "${FILES[@]}" -R "$REPO" --clobber
else
  gh release create "$TAG" "${FILES[@]}" -R "$REPO" --title "Data $TAG" --notes \
"Large data files for the backend (too big for git). Fetch with \`./scripts/fetch_data.sh\`.

- \`LARIAC6_Buildings_2020.gdb.zip\`: LA County Countywide Building Outlines (LARIAC6, 2020), source for \`build_buildings.sh\`
- \`buildings.pmtiles\`: prebuilt 3D building tiles for the USC study circle (zoom 10-16, property \`height_m\`)
- \`crime.db\`: the risk model's LAPD 2020-2024 history (\`scripts/export_crime_db.py\`), goes in \`risk-model/data/\`"
fi
echo "done: https://github.com/$REPO/releases/tag/$TAG"
