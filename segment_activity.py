import pandas as pd
import numpy as np

# Load your data
df = pd.read_csv('your_file.csv')

# 1. Calculate the Magnitude of acceleration to simplify the 3 axes into 1
df['magnitude'] = np.sqrt(df['attr_x']**2 + df['attr_y']**2 + df['attr_z']**2)

# 2. Calculate rolling standard deviation (window size depends on your sampling rate)
# A window of 50-100 points usually works well to smooth out noise
df['rolling_std'] = df['magnitude'].rolling(window=100, center=True).std()

# 3. Define a threshold (0.5 is a common starting point for m/s^2)
# Any segment with std > 0.5 is considered "Moving"
df['is_moving'] = df['rolling_std'] > 0.5

# 4. Find the start and end of each moving segment
df['group'] = (df['is_moving'] != df['is_moving'].shift()).cumsum()
moving_segments = df[df['is_moving'] == True].groupby('group')['attr_time'].agg(['min', 'max', 'count'])

# 5. Filter for segments that are long enough to be real activity (e.g., > 1 second)
# Assuming time is in milliseconds, count depends on frequency
significant_segments = moving_segments[moving_segments['count'] > 500].sort_values(by='min')

print("Detected Activity Windows:")
for i, (idx, row) in enumerate(significant_segments.iterrows(), 1):
    print(f"Part {i}: Start = {row['min']}, End = {row['max']}")
    
    # Optional: Save each part to a new CSV
    # part_df = df[(df['attr_time'] >= row['min']) & (df['attr_time'] <= row['max'])]
    # part_df.to_csv(f'activity_part_{i}.csv', index=False)