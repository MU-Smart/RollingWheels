import pandas as pd
from pathlib import Path

def filter_csv_by_timestamp(
    input_csv: Path,
    output_csv: Path,
    min_ts,
    max_ts
):
    try:
        df = pd.read_csv(input_csv)

        # Skip files without attr_time
        if 'attr_time' not in df.columns:
            return

        # Convert timestamp
        df['attr_time'] = pd.to_datetime(df['attr_time'].astype(float), unit='ms')

        filtered = df[
            (df['attr_time'] >= min_ts) &
            (df['attr_time'] <= max_ts)
        ].copy()

        if filtered.empty:
            print(f"No data in range for {input_csv}")
            return

        filtered['attr_time'] = (filtered['attr_time'].astype('int64') // 10**6)

        # Ensure output directory exists
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        filtered.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}")

    except Exception as e:
        print(f"Skipped {input_csv}: {e}")


def process_root_directory(
    place,
    input_root,
    output_root,
    min_target_timestamp,
    max_target_timestamp
):
    input_root = Path(input_root)
    output_root = Path(output_root)

    min_ts = pd.to_datetime(float(min_target_timestamp), unit='ms')
    max_ts = pd.to_datetime(float(max_target_timestamp), unit='ms')

    for csv_file in input_root.rglob("*.csv"):
        # Preserve relative path
        relative_path = csv_file.relative_to(input_root)
        
        output_csv = output_root / relative_path.parent / f"{place}_{relative_path.name}"

        filter_csv_by_timestamp(
            input_csv=csv_file,
            output_csv=output_csv,
            min_ts=min_ts,
            max_ts=max_ts
        )


if __name__ == "__main__":
    process_root_directory(
        place="Basteistrabe",
        input_root="./Dresden-July15-18",
        output_root="./splitted/Dresden-July15-18",
        min_target_timestamp="1531904720000.0",
        max_target_timestamp="1531915929000.0"
    )