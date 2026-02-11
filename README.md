# RollingWheels: Wheelchair IMU & GPS Dataset

The **RollingWheels Dataset** contains multimodal time-series data collected from smartphones attached to real manual wheelchairs. This dataset is designed to support research in surface type classification, wheelchair navigation, and activity recognition. It includes high-frequency inertial motion data (Accelerometer, Gyroscope), atmospheric pressure, and GPS trajectories.

## Index

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Sensor Specifications](#sensor-specifications)
- [File Naming Convention](#file-naming-convention)
- [Data Format](#data-format)
- [License](#license)

---

## Overview

The data was collected by mounting smartphones (e.g., Samsung Galaxy S7, Motorola moto G 7/8/9) to wheelchairs. The dataset is split into two primary categories to facilitate different machine learning tasks:

1. **Labeled Data (Without GPS):** Focused on surface type identification across different countries (USA, Vietnam). These sessions were conducted in controlled environments to ensure ground truth for surface types.
2. **Unlabeled Data (With GPS):** Focused on real-world navigation and route mapping in European cities, capturing naturalistic wheelchair usage.

---

## Directory Structure

The dataset is organized by labeling status, geography, and surface/device type as shown in the file hierarchy:

**RollingWheels/**

- **Datasets/**
  - **Labeled_Data_Without_GPS/**
    - **USA/**
      - `SurfaceTypeID_1/` ... `SurfaceTypeID_10/`
    - **Vietnam/**
      - `SurfaceTypeID_1/` ... `SurfaceTypeID_12/`
  - **Unlabeled_Data_With_GPS/**
    - **Europe/**
      - **Austria/**
        - `Phone 7/`
        - `Phone 8/` (e.g., `Mirabellplatz_GPSData.csv`)
        - `Phone 9/`
      - **France/**
      - **Germany/**

---

## Sensor Specifications

Data captured via smartphone onboard sensors include:

| Sensor            | Measurement                   | Typical Units      |
| :---------------- | :---------------------------- | :----------------- |
| **Accelerometer** | Linear Acceleration (3-axis)  | m/s²               |
| **Gyroscope**     | Angular Velocity (3-axis)     | rad/s or deg/s     |
| **Pressure**      | Atmospheric Pressure/Altitude | hPa                |
| **GPS**           | Geospatial Positioning        | Latitude/Longitude |

---

## File Naming Convention

### Labeled Data (USA & Vietnam)

Files in the labeled directories follow a structured naming convention to encode the date, surface, hardware, and subject:

`YYYY-MM-DD_SurfaceTypeID_X_DeviceModel_exp#_subject#.csv`

- **Example:** `2019-08-30_SurfaceTypeID_1_SamsungGalaxyS7_exp3_subject1.csv`

### Unlabeled Data (Europe)

Files are typically named based on the specific location or landmark where the trace was recorded:

`<LocationName>_GPSData.csv`

- **Example:** `Mirabellplatz_GPSData.csv`

---

## Data Format

All data is provided in `.csv` format. While column headers may vary slightly between Android and iOS collection tools, the standard structure includes:

- `timestamp`: Time of measurement (Unix epoch or relative elapsed time).
- `accel_x`, `accel_y`, `accel_z`: 3-axis acceleration.
- `gyro_x`, `gyro_y`, `gyro_z`: 3-axis rotation.
- `pressure`: Barometer data (useful for detecting ramps or elevators).
- `latitude`, `longitude`: GPS coordinates (available in "Unlabeled" subset only).

---

## Usage Example (Python)

To load and inspect a surface-specific file:

```python
import pandas as pd
import os

# Define path to a specific surface experiment
file_path = "Datasets/Labeled_Data_Without_GPS/USA/SurfaceTypeID_1/2019-08-30_SurfaceTypeID_1_SamsungGalaxyS7_exp3_subject1.csv"

# Load Dataset
df = pd.read_csv(file_path)

# Preview sensor data
print(df[['accel_x', 'accel_y', 'accel_z']].head())
```
