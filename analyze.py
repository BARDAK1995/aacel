import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid

# Read the data from data.txt
df = pd.read_csv('data.txt', sep=';')

# Calculate time in seconds (sampled at 30 Hz)
df['time'] = df['index'] / 30.0

# Filter data to start from 15 seconds
start_time = 16.4
df = df[df['time'] >= start_time]

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['x'], label='X acceleration')
plt.plot(df['time'], df['y'], label='Y acceleration')
plt.plot(df['time'], df['z'], label='Z acceleration')

#also plot the total acceleration so sum all squared then square root them
df['total_acceleration'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
plt.plot(df['time'], df['total_acceleration'], label='Total acceleration')

plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (g)')  # Assuming units are in g; adjust if needed
plt.title('Accelerometer Data Over Time')
plt.legend()
plt.grid(True)
plt.show()

# --- New analysis: Integration for speed and dual-axis plot ---

# It is assumed that the 'total_acceleration' is linear acceleration (gravity-compensated).
# If it includes gravity, this integration will not produce a correct speed reading.

# Convert total acceleration to m/s^2
g = 9.81 # m/s^2
df['total_acceleration_ms2'] = df['total_acceleration'] * g

# Integrate acceleration to get speed, assuming starting from rest at start_time.
# cumtrapz calculates the cumulative trapezoidal integral.
df['speed_ms'] = cumulative_trapezoid(df['total_acceleration_ms2'], df['time'], initial=0)

# Convert speed to mph
df['speed_mph'] = df['speed_ms'] * 2.23694

# For reference, speed in km/h can be calculated as:
# df['speed_kmh'] = df['speed_ms'] * 3.6

# Calculate a 1-second rolling average of acceleration for smoothing
window_size = 10  # 30 samples = 1 second at 30Hz
df['avg_acceleration_ms2'] = df['total_acceleration_ms2'].rolling(window=window_size, center=True).mean()

# Create a new plot for acceleration and speed with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot acceleration on the left y-axis
color = 'tab:red'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Acceleration (m/s^2)', color=color)
ax1.plot(df['time'], df['total_acceleration_ms2'], color=color, alpha=0.3, label='Total Acceleration')
ax1.plot(df['time'], df['avg_acceleration_ms2'], color=color, linestyle='-', label='Averaged Acceleration')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, which='both', axis='y')

# Create a second y-axis that shares the same x-axis for speed
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Speed (mph)', color=color)
ax2.plot(df['time'], df['speed_mph'], color=color, label='Speed (mph)')
ax2.tick_params(axis='y', labelcolor=color)

# Add a title and a combined legend to the plot
plt.title('Total Acceleration and Integrated Speed Over Time')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')
fig.tight_layout()  # Adjust plot to prevent right y-label from being clipped
plt.show()