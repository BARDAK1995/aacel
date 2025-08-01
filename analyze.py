import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid


def load_data(filename):
    """
    Reads accelerometer data from a CSV file and adds a time column.

    Args:
        filename (str): The path to the data file.

    Returns:
        pd.DataFrame: DataFrame with a 'time' column added.
    """
    df = pd.read_csv(filename, sep=';')
    # Calculate time in seconds (sampled at 30 Hz)
    df['time'] = df['index'] / 30.0
    return df


def process_data(df, start_time, end_time=None):
    """
    Filters data, calculates accelerations, and integrates to find speed.

    Args:
        df (pd.DataFrame): The input DataFrame with time data.
        start_time (float): The time in seconds to start the analysis from.
        end_time (float, optional): The time to end analysis. Defaults to None.

    Returns:
        pd.DataFrame: The processed DataFrame with acceleration and speed columns.
    """
    # Filter data to start from the specified time
    df_processed = df[df['time'] >= start_time].copy()
    if end_time is not None:
        df_processed = df_processed[df_processed['time'] <= end_time]

    # Calculate total acceleration
    df_processed['total_acceleration'] = np.sqrt(df_processed['x']**2 + df_processed['y']**2 + df_processed['z']**2)
    
    # Convert total acceleration to m/s^2
    g = 9.81  # m/s^2
    df_processed['total_acceleration_ms2'] = df_processed['total_acceleration'] * g

    # Integrate acceleration to get speed, assuming starting from rest at start_time.
    df_processed['speed_ms'] = cumulative_trapezoid(
        df_processed['total_acceleration_ms2'], df_processed['time'], initial=0
    )
    
    # Convert speed to mph
    df_processed['speed_mph'] = df_processed['speed_ms'] * 2.23694

    # Calculate a rolling average of acceleration for smoothing
    window_size = 10  # ~1/3 second rolling average at 30Hz
    df_processed['avg_acceleration_g'] = df_processed['total_acceleration'].rolling(
        window=window_size, center=True
    ).mean()
    
    return df_processed


def plot_raw_accelerations(df):
    """
    Plots the raw x, y, z, and total acceleration data.

    Args:
        df (pd.DataFrame): DataFrame containing acceleration data.
    """
    font_size = 16
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'].to_numpy(), df['x'].to_numpy(), label='X acceleration')
    plt.plot(df['time'].to_numpy(), df['y'].to_numpy(), label='Y acceleration')
    plt.plot(df['time'].to_numpy(), df['z'].to_numpy(), label='Z acceleration')
    plt.plot(df['time'].to_numpy(), df['total_acceleration'].to_numpy(), label='Total acceleration')
    
    plt.xlabel('Time (seconds)', fontsize=font_size)
    plt.ylabel('Acceleration (g)', fontsize=font_size)
    plt.title('Accelerometer Data Over Time', fontsize=font_size + 2)
    plt.legend(fontsize=font_size - 2)
    plt.tick_params(axis='both', which='major', labelsize=font_size - 2)
    plt.grid(True)


def plot_speed_analysis(df):
    """
    Plots smoothed acceleration and integrated speed on a dual-axis plot.

    Args:
        df (pd.DataFrame): DataFrame with processed acceleration and speed data.
    """
    font_size = 16
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot acceleration on the left y-axis
    color = 'tab:red'
    ax1.set_xlabel('Time (s)', fontsize=font_size)
    ax1.set_ylabel('Acceleration (g)', color=color, fontsize=font_size)
    ax1.plot(df['time'].to_numpy(), df['total_acceleration'].to_numpy(), color=color, alpha=0.3, label='Total Acceleration (g)')
    ax1.plot(df['time'].to_numpy(), df['avg_acceleration_g'].to_numpy(), color=color, linestyle='-', label='Averaged Acceleration (g)')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=font_size - 2)
    ax1.tick_params(axis='x', labelsize=font_size - 2)
    ax1.grid(True, which='both', axis='y')

    # Create a second y-axis for speed
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Speed (mph)', color=color, fontsize=font_size)
    ax2.plot(df['time'].to_numpy(), df['speed_mph'].to_numpy(), color=color, label='Speed (mph)')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=font_size - 2)

    # Add a title and a combined legend
    plt.title('Total Acceleration and Integrated Speed Over Time', fontsize=font_size + 2)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=font_size - 2)
    fig.tight_layout()


def analyze_ride(filename, start_time, end_time=None):
    """
    Master function to load, process, and plot accelerometer data.

    Args:
        filename (str): The path to the data file.
        start_time (float): The time in seconds to start the analysis.
        end_time (float, optional): The time to end analysis. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data for the specified range.
    """
    # It is assumed that the 'total_acceleration' is linear acceleration (gravity-compensated).
    # If it includes gravity, this integration will not produce a correct speed reading.
    print("NOTE: Speed calculation assumes 'total_acceleration' is gravity-compensated.")

    # Load and process data
    raw_df = load_data(filename)
    processed_df = process_data(raw_df, start_time, end_time)

    # Plot results for this specific ride
    plot_raw_accelerations(processed_df)
    plot_speed_analysis(processed_df)
    
    plt.show()

    return processed_df


def calculate_vehicle_dynamics(df, specs):
    """
    Calculates drive unit torque and wheel horsepower based on acceleration data and vehicle specs.

    Args:
        df (pd.DataFrame): DataFrame containing processed data, including speed and acceleration.
        specs (dict): Dictionary of vehicle specifications.

    Returns:
        pd.DataFrame: The input DataFrame with new 'drive_unit_torque_nm' and 'wheel_horsepower' columns.
    """
    # Constants
    g_ms2 = 9.81
    watts_to_hp = 745.7

    # Unpack specs and calculate constants
    total_mass_kg = specs['vehicle_mass_kg'] + specs['driver_mass_kg']
    wheel_radius_m = specs['wheel_diameter_m'] / 2.0
    
    # Force for linear acceleration (F=ma), using smoothed acceleration in m/s^2
    force_accel = total_mass_kg * df['avg_acceleration_g'] * g_ms2
    
    # Total tractive force is based purely on acceleration for this analysis
    total_tractive_force = force_accel
    
    # Torque at the wheels
    wheel_torque_nm = total_tractive_force * wheel_radius_m
    
    # Torque at the drive unit, accounting for gear ratio and efficiency
    # A small epsilon is added to the denominator to avoid division by zero if efficiency is 0.
    drive_unit_torque_nm = (wheel_torque_nm / specs['final_drive_ratio']) / (specs['drivetrain_efficiency'] + 1e-6)
    
    df['drive_unit_torque_nm'] = drive_unit_torque_nm
    
    # Calculate power and horsepower at the wheels
    wheel_angular_velocity_rad_s = df['speed_ms'] / wheel_radius_m
    wheel_power_watts = wheel_torque_nm * wheel_angular_velocity_rad_s
    df['wheel_horsepower'] = wheel_power_watts / watts_to_hp

    return df


if __name__ == '__main__':
    # --- Vehicle & Run Configuration ---
    # NOTE: These parameters are estimates for a 2024 Tesla Model 3 RWD.
    # Torque calculations are highly sensitive to these values.
    tesla_model_3_rwd_2024_specs = {
        'vehicle_mass_kg': 1765,      # Curb weight
        'driver_mass_kg': 95,
        'final_drive_ratio': 9.0,     # Approximate final drive ratio
        'wheel_diameter_m': 0.6687,   # For standard 235/45R18 tires
        'drag_coefficient': 0.22,
        'frontal_area_m2': 2.22,
        'rolling_resistance_coefficient': 0.012,
        'drivetrain_efficiency': 0.90,  # Assumed efficiency from motor to wheels
    }

    # A list of dictionaries, where each dictionary defines a ride to analyze.
    rides_to_analyze = [
        {'file': 'data.txt', 'start': 16.2, 'end': 23.0},
#        {'file': 'data_corner.txt', 'start': 12, 'end': 30.0},
        #{'file': 'data3.txt', 'start': 15.7, 'end': 23.0},  # No end time, will go to end of file
        {'file': 'data4.txt', 'start': 0.5, 'end': 5.5}, 
        {'file': 'data5.txt', 'start': 0.7, 'end': 5.3}, 
       # {'file': 'data6.txt', 'start': 0.2}, 
       # {'file': 'data7.txt', 'start': 3.7}, 
       # {'file': 'data8.txt', 'start': 30.1},
    ]
    rides_to_analyze = [
#        {'file': 'data.txt', 'start': 16.2, 'end': 23.0},
        {'file': 'data_corner.txt', 'start': 5, 'end': 40.0},
 #       {'file': 'data3.txt', 'start': 15.7, 'end': 23.0},  # No end time, will go to end of file
  #      {'file': 'data4.txt', 'start': 0.5, 'end': 5.5}, 
   #     {'file': 'data5.txt', 'start': 0.7}, 
    #    {'file': 'data6.txt', 'start': 0.2}, 
     #   {'file': 'data7.txt', 'start': 3.7}, 
      #  {'file': 'data8.txt', 'start': 30.1},
    ]
    all_ride_data = {}

    for ride in rides_to_analyze:
        try:
            # Create a unique name for each ride analysis for the legend
            ride_name = f"{ride['file']} (start: {ride['start']})"
            print(f"\n--- Analyzing {ride_name} ---")

            ride_data = analyze_ride(
                ride['file'], 
                ride['start'], 
                ride.get('end', None)  # Use .get() for safe access to 'end'
            )
            
            # Add torque and horsepower calculations to the dataframe
            ride_data_with_dynamics = calculate_vehicle_dynamics(ride_data, tesla_model_3_rwd_2024_specs)
            all_ride_data[ride_name] = ride_data_with_dynamics

        except FileNotFoundError:
            print(f"ERROR: Could not find file {ride['file']}. Skipping.")

    # --- Overlay Plot of Accelerations ---
    if all_ride_data:
        font_size = 16
        plt.figure(figsize=(14, 7))
        
        for name, data in all_ride_data.items():
            if not data.empty:
                # Normalize time to start from 0 for comparison
                time_normalized = data['time'].to_numpy() - data['time'].to_numpy()[0]
                plt.plot(time_normalized, data['avg_acceleration_g'].to_numpy(), label=name)

        plt.xlabel('Time From Analysis Start (s)', fontsize=font_size)
        plt.ylabel('Averaged Acceleration (g)', fontsize=font_size)
        plt.title('Comparison of Smoothed Acceleration Across Rides', fontsize=font_size + 2)
        plt.legend(fontsize=font_size - 2)
        plt.tick_params(axis='both', which='major', labelsize=font_size - 2)
        plt.grid(True)
        plt.show()

    # --- Overlay Plot of Speeds ---
    if all_ride_data:
        font_size = 16
        plt.figure(figsize=(14, 7))
        
        for name, data in all_ride_data.items():
            if not data.empty:
                # Normalize time to start from 0 for comparison
                time_normalized = data['time'].to_numpy() - data['time'].to_numpy()[0]
                plt.plot(time_normalized, data['speed_mph'].to_numpy(), label=name)

        plt.xlabel('Time From Analysis Start (s)', fontsize=font_size)
        plt.ylabel('Speed (mph)', fontsize=font_size)
        plt.title('Comparison of Integrated Speed Across Rides', fontsize=font_size + 2)
        plt.legend(fontsize=font_size - 2)
        plt.tick_params(axis='both', which='major', labelsize=font_size - 2)
        plt.grid(True)
        plt.show()

    # --- Overlay Plot of Torque ---
    if all_ride_data:
        font_size = 16
        plt.figure(figsize=(14, 7))
        
        for name, data in all_ride_data.items():
            if not data.empty:
                # Normalize time to start from 0 for comparison
                time_normalized = data['time'].to_numpy() - data['time'].to_numpy()[0]
                plt.plot(time_normalized, data['drive_unit_torque_nm'].to_numpy(), label=name)

        plt.xlabel('Time From Analysis Start (s)', fontsize=font_size)
        plt.ylabel('Drive Unit Torque (Nm)', fontsize=font_size)
        plt.title('Comparison of Calculated Drive Unit Torque Across Rides', fontsize=font_size + 2)
        plt.legend(fontsize=font_size - 2)
        plt.tick_params(axis='both', which='major', labelsize=font_size - 2)
        plt.grid(True)
        plt.show()

    # --- Overlay Plot of Horsepower ---
    if all_ride_data:
        font_size = 16
        plt.figure(figsize=(14, 7))
        
        for name, data in all_ride_data.items():
            if not data.empty:
                # Normalize time to start from 0 for comparison
                time_normalized = data['time'].to_numpy() - data['time'].to_numpy()[0]
                plt.plot(time_normalized, data['wheel_horsepower'].to_numpy(), label=name)

        plt.xlabel('Time From Analysis Start (s)', fontsize=font_size)
        plt.ylabel('Effective Wheel Horsepower (hp)', fontsize=font_size)
        plt.title('Comparison of Calculated Wheel Horsepower Across Rides', fontsize=font_size + 2)
        plt.legend(fontsize=font_size - 2)
        plt.tick_params(axis='both', which='major', labelsize=font_size - 2)
        plt.grid(True)
        plt.show()

