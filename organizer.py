from pathlib import Path
import shutil

ROOT = Path("Europe")

for csv_file in ROOT.rglob("export*/*.csv"):
    export_dir = csv_file.parent
    target_dir = export_dir.parent  # move up one level

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / csv_file.name

    print(f"Moving: {csv_file} -> {target_path}")
    shutil.move(str(csv_file), str(target_path))

# cleanup empty export folders
for export_dir in sorted(ROOT.rglob("export*"), reverse=True):
    if export_dir.is_dir() and not any(export_dir.iterdir()):
        print(f"Removing empty folder: {export_dir}")
        export_dir.rmdir()