from tkinter import X
from suspension_analysis import load_accel_data, compute_total_acceleration_g, select_time_window, estimate_dominant_frequency_hz, estimate_damping_ratio, plot_window_analysis
import matplotlib.pyplot as plt
df = load_accel_data("leftrear.txt", sample_rate_hz=30.0)
df = compute_total_acceleration_g(df)
dwin = select_time_window(df, start_time_s=14.2, window_s=3)
t = dwin["time"].to_numpy()
a = dwin["a_total_g"].to_numpy()
x = dwin["x"].to_numpy()
f = estimate_dominant_frequency_hz(t, x, fmin_hz=0.1, fmax_hz=3.0)
print("frequency: ", f)
dwin = select_time_window(df, start_time_s=16.55, window_s=1.6)
t = dwin["time"].to_numpy()
a = dwin["a_total_g"].to_numpy()
x = dwin["x"].to_numpy()
damp = estimate_damping_ratio(t, x,known_fd_hz=f, fmin_hz=0.01, fmax_hz=5.0)
plot_window_analysis(t, x, fmin_hz=0.1, fmax_hz=5.0, damping=damp)
dwin = select_time_window(df, start_time_s=14.2, window_s=3)
t = dwin["time"].to_numpy()
a = dwin["a_total_g"].to_numpy()
x = dwin["x"].to_numpy()
plot_window_analysis(t, x, fmin_hz=0.1, fmax_hz=5.0, damping=damp)


df = load_accel_data("rightrear.txt", sample_rate_hz=30.0)
df = compute_total_acceleration_g(df)
dwin = select_time_window(df, start_time_s=29.55, window_s=3.2)
t = dwin["time"].to_numpy()
a = dwin["a_total_g"].to_numpy()
x = dwin["x"].to_numpy()
f = estimate_dominant_frequency_hz(t, x, fmin_hz=0.1, fmax_hz=5.0)
print("frequency: ", f)


dwin = select_time_window(df, start_time_s=23.69, window_s=1.7)
t = dwin["time"].to_numpy()
a = dwin["a_total_g"].to_numpy()
x = dwin["x"].to_numpy()

damp = estimate_damping_ratio(t, x,known_fd_hz=f, fmin_hz=0.01, fmax_hz=5.0)
plot_window_analysis(t, x, fmin_hz=0.1, fmax_hz=5.0, damping=damp)



dwin = select_time_window(df, start_time_s=29.55, window_s=3.2)
t = dwin["time"].to_numpy()
a = dwin["a_total_g"].to_numpy()
x = dwin["x"].to_numpy()

damp = estimate_damping_ratio(t, x,known_fd_hz=f, fmin_hz=0.01, fmax_hz=5.0)
plot_window_analysis(t, x, fmin_hz=0.1, fmax_hz=5.0, damping=damp)