"""Unzip UK Data Service LFS downloads from Downloads into data/raw/<SN>/."""
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DOWNLOADS = Path.home() / "Downloads"

GROUPS = {
    "Quarterly LFS 2021-2026 (main sample)": {
        8806, 8826, 8872, 8915, 8957, 8999, 9027, 9052, 9097, 9131, 9225,
        9226, 9264, 9297, 9323, 9349, 9388, 9433, 9480, 9522, 9569, 9607},
    "Two-quarter longitudinal 2021-2026": {
        8827, 8871, 8917, 8958, 8998, 9028, 9053, 9099, 9132, 9302, 9301,
        9265, 9298, 9324, 9348, 9389, 9432, 9481, 9531, 9570},
    "Quarterly LFS 2016-2020 (extension)": {
        7985, 8039, 8104, 8145, 8195, 8235, 8292, 8326, 8343, 8381, 8407,
        8447, 8485, 8563, 8588, 8614, 8639, 8671, 8720, 8777},
}
ALL = set().union(*GROUPS.values())

found = set()
for z in sorted(DOWNLOADS.glob("*.zip")):
    m = re.search(r"(?<!\d)(\d{4})(?!\d)", z.name)
    if not m or int(m.group(1)) not in ALL:
        continue
    sn = int(m.group(1))
    dest = RAW / str(sn)
    with zipfile.ZipFile(z) as f:
        f.extractall(dest)
    found.add(sn)
    print(f"{z.name} -> data/raw/{sn}")

print()
for name, sns in GROUPS.items():
    missing = sorted(sns - found)
    print(f"{name}: {len(sns & found)} of {len(sns)} extracted")
    if missing and sns & found:
        print(f"  not found in Downloads: {missing}")

print("\nStata files found:")
for sn in sorted(found):
    files = sorted(p.name for p in (RAW / str(sn)).rglob("*.dta"))
    print(f"  {sn}: {files}")