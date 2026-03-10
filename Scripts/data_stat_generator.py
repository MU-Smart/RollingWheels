"""
Dataset Statistics Generator for RollingWheels Raw_Data
Covers both Labeled_Data_Without_GPS and Unlabeled_Data_With_GPS
"""

import os
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "Datasets" / "Raw_Data"
SURFACE_TYPES_CSV = Path(__file__).resolve().parent.parent / "Datasets" / "surface_types.csv"

# ── helpers ──────────────────────────────────────────────────────────────────

def fmt_duration(ms: float) -> str:
    """Format milliseconds as HH:MM:SS.mmm"""
    if ms <= 0:
        return "00:00:00.000"
    total_s = ms / 1000
    h = int(total_s // 3600)
    m = int((total_s % 3600) // 60)
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def load_surface_map() -> dict:
    if SURFACE_TYPES_CSV.exists():
        df = pd.read_csv(SURFACE_TYPES_CSV)
        return dict(zip(df["surface_id"].astype(str), df["surface_name"]))
    return {}


def duration_ms_from_timestamps(series: pd.Series) -> float:
    """Return duration in ms from a timestamp series (epoch ms).
    Uses consecutive-diff sum to be robust against corrupted outlier timestamps.
    Only counts gaps <= 10 seconds (10_000 ms) between consecutive readings.
    """
    ts = pd.to_numeric(series, errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return 0.0
    diffs = ts.diff().dropna()
    # only include plausible inter-sample gaps (up to 10 s)
    return float(diffs[diffs <= 10_000].sum())


def parse_timestamp_col(df: pd.DataFrame) -> pd.Series | None:
    for col in ["timestamp", "attr_time"]:
        if col in df.columns:
            return df[col]
    return None


# ── labeled data parser ───────────────────────────────────────────────────────

LABELED_FILENAME_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})_SurfaceTypeID_(?P<surf>\d+)_(?P<phone>[^_]+(?:_[^_]+)*)[-_]exp(?P<exp>\d+)[-_](?P<subject_or_sensor>.+)\.csv",
    re.IGNORECASE,
)


def parse_labeled_filename(fname: str) -> dict:
    """Extract metadata from labeled CSV filenames."""
    m = LABELED_FILENAME_RE.match(fname)
    if not m:
        return {}
    info = m.groupdict()
    tail = info.pop("subject_or_sensor", "")
    # tail is either "subjectN" or a sensor chip name (e.g. ICM20607)
    sub_match = re.match(r"subject(\d+)", tail, re.IGNORECASE)
    if sub_match:
        info["subject"] = sub_match.group(1)
        info["sensor_chip"] = None
    else:
        info["subject"] = None
        info["sensor_chip"] = tail
    return info


def collect_labeled_stats(surface_map: dict) -> dict:
    """Crawl Labeled_Data_Without_GPS and aggregate statistics."""
    labeled_root = ROOT / "Labeled_Data_Without_GPS"

    def labeled_entry() -> dict:
        return {
            "files": 0,
            "total_frames": 0,
            "total_duration_ms": 0.0,
            "sensors": defaultdict(int),
            "phones": defaultdict(int),
            "subjects": set(),
            "experiments": set(),
            "dates": set(),
        }

    # per-country × surface stats
    stats = defaultdict(labeled_entry)

    for country_dir in sorted(labeled_root.iterdir()):
        if not country_dir.is_dir():
            continue
        country = country_dir.name

        for surf_dir in sorted(country_dir.iterdir()):
            if not surf_dir.is_dir():
                continue
            surf_id = surf_dir.name.replace("SurfaceTypeID_", "")
            surf_name = surface_map.get(surf_id, f"SurfaceTypeID_{surf_id}")
            key = (country, surf_id, surf_name)

            for csv_file in sorted(surf_dir.glob("*.csv")):
                meta = parse_labeled_filename(csv_file.name)
                try:
                    df = pd.read_csv(csv_file, low_memory=False)
                except Exception as e:
                    print(f"  [WARN] Could not read {csv_file}: {e}")
                    continue

                ts_col = parse_timestamp_col(df)
                duration = duration_ms_from_timestamps(ts_col) if ts_col is not None else 0.0

                s = stats[key]
                s["files"] += 1
                s["total_frames"] += len(df)
                s["total_duration_ms"] += duration

                # sensor names from column if present
                if "sensorName" in df.columns:
                    for sensor, cnt in df["sensorName"].value_counts().items():
                        s["sensors"][sensor] += cnt

                if meta.get("phone"):
                    s["phones"][meta["phone"]] += 1
                if meta.get("subject"):
                    s["subjects"].add(meta["subject"])
                if meta.get("exp"):
                    s["experiments"].add(meta["exp"])
                if meta.get("date"):
                    s["dates"].add(meta["date"])
                if meta.get("sensor_chip"):
                    s["sensors"][meta["sensor_chip"]] += len(df)

    return stats


# ── unlabeled data parser ─────────────────────────────────────────────────────

def collect_unlabeled_stats() -> dict:
    """Crawl Unlabeled_Data_With_GPS and aggregate statistics."""
    unlabeled_root = ROOT / "Unlabeled_Data_With_GPS"

    def unlabeled_entry() -> dict:
        return {
            "files": 0,
            "total_frames": 0,
            "total_duration_ms": 0.0,
            "sensor_types": defaultdict(int),
            "locations": set(),
        }

    # key = (continent, country, phone)
    stats = defaultdict(unlabeled_entry)

    for continent_dir in sorted(unlabeled_root.iterdir()):
        if not continent_dir.is_dir():
            continue
        continent = continent_dir.name

        for country_dir in sorted(continent_dir.iterdir()):
            if not country_dir.is_dir():
                continue
            country = country_dir.name

            for phone_dir in sorted(country_dir.iterdir()):
                if not phone_dir.is_dir():
                    continue
                phone = phone_dir.name
                key = (continent, country, phone)

                for csv_file in sorted(phone_dir.glob("*.csv")):
                    fname = csv_file.stem  # no extension
                    # detect sensor type from filename suffix
                    if fname.endswith("AccelerometerData"):
                        sensor_type = "Accelerometer"
                        location = fname.replace("_AccelerometerData", "")
                    elif fname.endswith("GyroscopeData"):
                        sensor_type = "Gyroscope"
                        location = fname.replace("_GyroscopeData", "")
                    elif fname.endswith("GPSData"):
                        sensor_type = "GPS"
                        location = fname.replace("_GPSData", "")
                    else:
                        sensor_type = "Unknown"
                        location = fname

                    try:
                        df = pd.read_csv(csv_file, low_memory=False)
                    except Exception as e:
                        print(f"  [WARN] Could not read {csv_file}: {e}")
                        continue

                    ts_col = parse_timestamp_col(df)
                    duration = duration_ms_from_timestamps(ts_col) if ts_col is not None else 0.0

                    s = stats[key]
                    s["files"] += 1
                    s["total_frames"] += len(df)
                    s["total_duration_ms"] += duration
                    s["sensor_types"][sensor_type] += len(df)
                    s["locations"].add(location.replace("_", " "))

    return stats


# ── pretty printing ───────────────────────────────────────────────────────────

SEP = "=" * 80
SEP2 = "-" * 80


def print_labeled_report(stats: dict):
    print(f"\n{SEP}")
    print("  LABELED DATA  (Labeled_Data_Without_GPS)")
    print(SEP)

    # group by country first
    by_country: dict[str, list] = defaultdict(list)
    for (country, surf_id, surf_name), s in sorted(stats.items(), key=lambda x: (x[0][0], int(x[0][1]))):
        by_country[country].append((surf_id, surf_name, s))

    grand_frames = 0
    grand_dur = 0.0

    for country, entries in sorted(by_country.items()):
        country_frames = sum(s["total_frames"] for _, _, s in entries)
        country_dur = sum(s["total_duration_ms"] for _, _, s in entries)
        print(f"\n  Country: {country}")
        print(f"  {'Surface':<45} {'Files':>6}  {'Frames':>10}  {'Duration (hh:mm:ss)':>20}")
        print(f"  {SEP2}")

        for surf_id, surf_name, s in entries:
            label = f"[{surf_id:>2}] {surf_name}"
            print(f"  {label:<45} {s['files']:>6}  {s['total_frames']:>10,}  {fmt_duration(s['total_duration_ms']):>20}")

            # phones
            if s["phones"]:
                phones_str = ", ".join(f"{p}({n})" for p, n in sorted(s["phones"].items()))
                print(f"       Phones   : {phones_str}")
            # sensors
            if s["sensors"]:
                sensors_str = ", ".join(f"{sn}({cnt:,})" for sn, cnt in sorted(s["sensors"].items(), key=lambda x: -x[1]))
                print(f"       Sensors  : {sensors_str}")
            if s["subjects"]:
                print(f"       Subjects : {len(s['subjects'])} unique")
            if s["experiments"]:
                print(f"       Exps     : {len(s['experiments'])} unique")
            if s["dates"]:
                date_range = f"{min(s['dates'])} → {max(s['dates'])}"
                print(f"       Dates    : {date_range}")

        print(f"  {SEP2}")
        print(f"  {'Country total':<45} {'':>6}  {country_frames:>10,}  {fmt_duration(country_dur):>20}")
        grand_frames += country_frames
        grand_dur += country_dur

    print(f"\n  {SEP2}")
    print(f"  {'LABELED GRAND TOTAL':<45} {'':>6}  {grand_frames:>10,}  {fmt_duration(grand_dur):>20}")


def print_unlabeled_report(stats: dict):
    print(f"\n{SEP}")
    print("  UNLABELED DATA  (Unlabeled_Data_With_GPS)")
    print(SEP)

    grand_frames = 0
    grand_dur = 0.0

    for (continent, country, phone), s in sorted(stats.items()):
        grand_frames += s["total_frames"]
        grand_dur += s["total_duration_ms"]

    # group by continent → country
    by_cont_country: dict = defaultdict(lambda: defaultdict(list))
    for (continent, country, phone), s in sorted(stats.items()):
        by_cont_country[continent][country].append((phone, s))

    for continent, countries in sorted(by_cont_country.items()):
        print(f"\n  Continent: {continent}")
        for country, phones in sorted(countries.items()):
            print(f"\n    Country: {country}")
            for phone, s in sorted(phones):
                print(f"\n      {phone}")
                print(f"        Files    : {s['files']}")
                print(f"        Frames   : {s['total_frames']:,}")
                print(f"        Duration : {fmt_duration(s['total_duration_ms'])}")
                if s["sensor_types"]:
                    for stype, cnt in sorted(s["sensor_types"].items(), key=lambda x: -x[1]):
                        print(f"        {stype:<14}: {cnt:,} frames")
                if s["locations"]:
                    locs = sorted(s["locations"])
                    print(f"        Locations: {', '.join(locs)}")

    print(f"\n  {SEP2}")
    print(f"  UNLABELED GRAND TOTAL  frames={grand_frames:,}  duration={fmt_duration(grand_dur)}")


def print_overall_summary(labeled_stats: dict, unlabeled_stats: dict):
    labeled_frames = sum(s["total_frames"] for s in labeled_stats.values())
    labeled_dur = sum(s["total_duration_ms"] for s in labeled_stats.values())
    unlabeled_frames = sum(s["total_frames"] for s in unlabeled_stats.values())
    unlabeled_dur = sum(s["total_duration_ms"] for s in unlabeled_stats.values())

    # unique phones
    labeled_phones: set = set()
    for s in labeled_stats.values():
        labeled_phones.update(s["phones"].keys())
    unlabeled_phones: set = set()
    for (_, _, phone), _ in unlabeled_stats.items():
        unlabeled_phones.add(phone)

    # unique surface types
    surface_ids = {surf_id for (_, surf_id, _) in labeled_stats.keys()}

    # countries
    labeled_countries = {c for (c, _, _) in labeled_stats.keys()}
    unlabeled_countries = {c for (_, c, _) in unlabeled_stats.keys()}
    all_countries = labeled_countries | unlabeled_countries

    print(f"\n{SEP}")
    print("  OVERALL SUMMARY")
    print(SEP)
    print(f"  {'Category':<35} {'Value'}")
    print(f"  {SEP2}")
    print(f"  {'Total frames':<35} {labeled_frames + unlabeled_frames:,}")
    print(f"  {'  → Labeled frames':<35} {labeled_frames:,}")
    print(f"  {'  → Unlabeled frames':<35} {unlabeled_frames:,}")
    print(f"  {'Total recording duration':<35} {fmt_duration(labeled_dur + unlabeled_dur)}")
    print(f"  {'  → Labeled duration':<35} {fmt_duration(labeled_dur)}")
    print(f"  {'  → Unlabeled duration':<35} {fmt_duration(unlabeled_dur)}")
    print(f"  {'Countries':<35} {sorted(all_countries)}")
    print(f"  {'  → Labeled':<35} {sorted(labeled_countries)}")
    print(f"  {'  → Unlabeled':<35} {sorted(unlabeled_countries)}")
    print(f"  {'Labeled surface types':<35} {len(surface_ids)} ({sorted(surface_ids, key=int)})")
    print(f"  {'Labeled phone models':<35} {sorted(labeled_phones)}")
    print(f"  {'Unlabeled phone identifiers':<35} {sorted(unlabeled_phones)}")
    print(SEP)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Root: {ROOT}")
    surface_map = load_surface_map()

    print("\nCollecting labeled data statistics...")
    labeled_stats = collect_labeled_stats(surface_map)

    print("Collecting unlabeled data statistics...")
    unlabeled_stats = collect_unlabeled_stats()

    print_overall_summary(labeled_stats, unlabeled_stats)
    print_labeled_report(labeled_stats)
    print_unlabeled_report(unlabeled_stats)


if __name__ == "__main__":
    main()
