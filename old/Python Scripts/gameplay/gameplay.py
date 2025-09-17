# ITP 116, Spring 2025
# Final Project
# Name: Naazneen Shafeer Vemmerath Kulangara
# Email: vemmerat@usc.edu
# Description: This python file is a gameplay file taken from lab 7

#original code taken from lab 7 and then modifying with three new functions

import random as rand
import helper
import functions

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def find_latest_simulation_files(folder_path="simulation_data"):
    """Find the most recent simulation files in the specified folder"""
    try:
        # Get all pickle files in the directory
        files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
        
        if not files:
            raise FileNotFoundError("No simulation files found in the directory")
        
        # Extract timestamps from filenames (assuming format with timestamps)
        file_info = []
        for f in files:
            try:
                # Extract timestamp from filename (adjust pattern as needed)
                if 'no_df' in f.lower():
                    prefix = 'no_df'
                elif 'with_df' in f.lower():
                    prefix = 'with_df'
                else:
                    continue
                    
                # Find timestamp in filename (assuming format YYYYMMDD_HHMMSS)
                parts = f.split('_')
                for part in parts:
                    if len(part) == 15 and part[:8].isdigit() and part[9:].isdigit():
                        timestamp = datetime.strptime(part, "%Y%m%d_%H%M%S")
                        file_info.append((f, timestamp, prefix))
                        break
            except:
                continue
        
        if not file_info:
            raise ValueError("No valid simulation files found with timestamp patterns")
        
        # Find most recent pair
        latest_time = max(t for (f, t, p) in file_info)
        latest_files = {
            'no_df': next(f for f, t, p in file_info if t == latest_time and p == 'no_df'),
            'with_df': next(f for f, t, p in file_info if t == latest_time and p == 'with_df')
        }
        
        return {
            'no_df': os.path.join(folder_path, latest_files['no_df']),
            'with_df': os.path.join(folder_path, latest_files['with_df'])
        }
        
    except Exception as e:
        print(f"Error finding simulation files: {str(e)}")
        raise

def load_simulation_data(file_path):
    """Load simulation data from pickle file"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        raise

def plot_orbits(data_nodf, data_df):
    """Plot the orbital paths from simulation data"""
    plt.figure(figsize=(12, 12))
    

    r1_nodf = data_nodf['r1']
    r2_nodf = data_nodf['r2']
    r1_df = data_df['r1']
    r2_df = data_df['r2']
    

    plt.plot(r1_nodf[:, 0], r1_nodf[:, 1], 'b-', label='BH1 - No DF', linewidth=1.5)
    plt.plot(r2_nodf[:, 0], r2_nodf[:, 1], 'r-', label='BH2 - No DF', linewidth=1.5)
    plt.plot(r1_df[:, 0], r1_df[:, 1], 'c--', label='BH1 - With DF', linewidth=1.5)
    plt.plot(r2_df[:, 0], r2_df[:, 1], 'm--', label='BH2 - With DF', linewidth=1.5)
    
    # Start
    plt.scatter(r1_nodf[0, 0], r1_nodf[0, 1], c='blue', marker='o', s=100, edgecolor='black', zorder=5)
    plt.scatter(r2_nodf[0, 0], r2_nodf[0, 1], c='red', marker='o', s=100, edgecolor='black', zorder=5)
    
    # End
    plt.scatter(r1_nodf[-1, 0], r1_nodf[-1, 1], c='blue', marker='*', s=200, edgecolor='black', zorder=5)
    plt.scatter(r2_nodf[-1, 0], r2_nodf[-1, 1], c='red', marker='*', s=200, edgecolor='black', zorder=5)
    plt.scatter(r1_df[-1, 0], r1_df[-1, 1], c='cyan', marker='*', s=200, edgecolor='black', zorder=5)
    plt.scatter(r2_df[-1, 0], r2_df[-1, 1], c='magenta', marker='*', s=200, edgecolor='black', zorder=5)
    
    plt.xlabel('X [pc]', fontsize=12)
    plt.xlim(-1000, 1000)
    plt.ylim(-1000, 1000)
    plt.ylabel('Y [pc]', fontsize=12)
    plt.title('Black Hole Binary Orbits\nWith and Without Dynamical Friction', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='upper right')
    plt.gca().set_aspect('equal', adjustable='box')
    
    text_str = (
        f"Simulation Time: {data_nodf['t'][-1]:.1f} yr\n"
        f"Final Separation (No DF): {np.linalg.norm(data_nodf['r1'][-1]-data_nodf['r2'][-1]):.1f} pc\n"
        f"Final Separation (With DF): {np.linalg.norm(data_df['r1'][-1]-data_df['r2'][-1]):.1f} pc"
    )
    plt.text(0.02, 0.98, text_str, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ic = data_df['initial_conditions']
    
    ic_text = (
        f"Initial Conditions:\n"
        f"Mass: {ic['mass']}\n"
        f"Separation: {ic['sep']} pc\n"
        f"Angle: {ic['angle']}°\n"
        f"Velocity: {ic['velocity']} km/s"
    )
    
    plt.text(0.02, 0.02, ic_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def plot_velocity(data_nodf, data_df):
    """Plot the velocity magnitude vs. time from simulation data"""
    plt.figure(figsize=(12, 12))
    
    v1_nodf = data_nodf['v1']
    v2_nodf = data_nodf['v2']
    v1_df = data_df['v1']
    v2_df = data_df['v2']
    t = data_nodf['t']  # Time array (same for both df and nodf)
    
    # might change this if vel magnitude looks weird
    v1_nodf_mag = np.linalg.norm(v1_nodf, axis=1)
    v2_nodf_mag = np.linalg.norm(v2_nodf, axis=1)
    v1_df_mag = np.linalg.norm(v1_df, axis=1)
    v2_df_mag = np.linalg.norm(v2_df, axis=1)
    
    plt.plot(t, v1_nodf_mag, 'b-', label='BH1 - No DF', linewidth=1.5)
    plt.plot(t, v2_nodf_mag, 'r-', label='BH2 - No DF', linewidth=1.5)
    plt.plot(t, v1_df_mag, 'c--', label='BH1 - With DF', linewidth=1.5)
    plt.plot(t, v2_df_mag, 'm--', label='BH2 - With DF', linewidth=1.5)
    

    plt.scatter(t[0], v1_nodf_mag[0], c='blue', marker='o', s=100, edgecolor='black', zorder=5)
    plt.scatter(t[0], v2_nodf_mag[0], c='red', marker='o', s=100, edgecolor='black', zorder=5)
    

    plt.scatter(t[-1], v1_nodf_mag[-1], c='blue', marker='*', s=200, edgecolor='black', zorder=5)
    plt.scatter(t[-1], v2_nodf_mag[-1], c='red', marker='*', s=200, edgecolor='black', zorder=5)
    plt.scatter(t[-1], v1_df_mag[-1], c='cyan', marker='*', s=200, edgecolor='black', zorder=5)
    plt.scatter(t[-1], v2_df_mag[-1], c='magenta', marker='*', s=200, edgecolor='black', zorder=5)
    
    plt.xlabel('Time [yr]', fontsize=12)
    plt.ylabel('Velocity Magnitude [km/s]', fontsize=12)
    plt.title('Black Hole Velocity vs. Time\nWith and Without Dynamical Friction', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='upper right')
    
    text_str = (
        f"Final Vel BH1 (No DF): {v1_nodf_mag[-1]:.1f} km/s\n"
        f"Final Vel BH1 (With DF): {v1_df_mag[-1]:.1f} km/s\n"
        f"Final Vel BH2 (No DF): {v2_nodf_mag[-1]:.1f} km/s\n"
        f"Final Vel BH2 (With DF): {v2_df_mag[-1]:.1f} km/s"
    )
    plt.text(0.02, 0.98, text_str, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ic = data_df['initial_conditions']
    ic_text = (
        f"Initial Conditions:\n"
        f"Mass: {ic['mass']}\n"
        f"Separation: {ic['sep']} pc\n"
        f"Angle: {ic['angle']}°\n"
        f"Velocity: {ic['velocity']} km/s"
    )
    
    plt.text(0.02, 0.02, ic_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_gravitational_acceleration_vs_time(data_df):
    """Plot gravitational acceleration vs. time for both BHs with DF"""
    plt.figure(figsize=(12, 6))

    t = data_df['t']
    a1_mag = np.linalg.norm(data_df['a_grav1'], axis=1)
    a2_mag = np.linalg.norm(data_df['a_grav2'], axis=1)

    plt.plot(t, a1_mag, 'b-', label='Gravitational Acceleration - BH1', linewidth=1.5)
    plt.plot(t, a2_mag, 'r-', label='Gravitational Acceleration - BH2', linewidth=1.5)

    plt.xlabel('Time [yr]', fontsize=12)
    plt.ylabel('Acceleration [km/s²]', fontsize=12)
    plt.title('Gravitational Acceleration vs Time (With DF)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    
def plot_position_vs_time(data_df):
    """Plot position magnitude vs. time for both BHs with DF"""
    plt.figure(figsize=(12, 6))

    t = data_df['t']
    r1_mag = np.linalg.norm(data_df['r1'], axis=1)
    r2_mag = np.linalg.norm(data_df['r2'], axis=1)

    plt.plot(t, r1_mag, 'b-', label='Position Magnitude - BH1', linewidth=1.5)
    plt.plot(t, r2_mag, 'r-', label='Position Magnitude - BH2', linewidth=1.5)

    plt.xlabel('Time [yr]', fontsize=12)
    plt.ylabel('Distance from Origin [pc]', fontsize=12)
    plt.ylim(0, 1000)
    plt.title('Position vs Time (With DF)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_df_acceleration_vs_time(data_df):
    """Plot DF acceleration vs. time for both BHs"""
    plt.figure(figsize=(12, 6))

    t = data_df['t']
    df1_mag = np.linalg.norm(data_df['a_df1'], axis=1)
    df2_mag = np.linalg.norm(data_df['a_df2'], axis=1)

    plt.plot(t, df1_mag, 'c--', label='DF Acceleration - BH1', linewidth=1.5)
    plt.plot(t, df2_mag, 'm--', label='DF Acceleration - BH2', linewidth=1.5)

    plt.xlabel('Time [yr]', fontsize=12)
    plt.ylabel('DF Acceleration [km/s²]', fontsize=12)
    plt.title('Dynamical Friction Acceleration vs Time', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()



def plot_separation_vs_time_bh1(data_df):
    """Plot BH1's separation from BH2 vs. time with DF and annotations"""
    plt.figure(figsize=(12, 6))

    t = data_df['t']
    r1 = data_df['r1']
    r2 = data_df['r2']
    sep = np.linalg.norm(r1 - r2, axis=1)

    plt.plot(t, sep, 'b-', label='Separation (BH1 to BH2)', linewidth=1.5)

    plt.xlabel('Time [yr]', fontsize=12)
    plt.ylabel('Separation [pc]', fontsize=12)
    plt.title('Separation vs Time (BH1 to BH2, With DF)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)


    sep_init = sep[0]
    sep_final = sep[-1]
    sep_text = (
        f"Initial Separation: {sep_init:.2f} pc\n"
        f"Final Separation: {sep_final:.2f} pc"
    )
    plt.text(0.02, 0.98, sep_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


    ic = data_df['initial_conditions']
    ic_text = (
        f"Initial Conditions:\n"
        f"Mass: {ic['mass']}\n"
        f"Separation: {ic['sep']} pc\n"
        f"Angle: {ic['angle']}°\n"
        f"Velocity: {ic['velocity']} km/s"
    )
    plt.text(0.02, 0.02, ic_text, transform=plt.gca().transAxes,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def plot_separation_vs_time_bh2(data_df):
    """Plot BH2's separation from BH1 vs. time with DF and annotations"""
    plt.figure(figsize=(12, 6))

    t = data_df['t']
    r1 = data_df['r1']
    r2 = data_df['r2']
    sep = np.linalg.norm(r2 - r1, axis=1)

    plt.plot(t, sep, 'r-', label='Separation (BH2 to BH1)', linewidth=1.5)

    plt.xlabel('Time [yr]', fontsize=12)
    plt.ylabel('Separation [pc]', fontsize=12)
    plt.title('Separation vs Time (BH2 to BH1, With DF)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)


    sep_init = sep[0]
    sep_final = sep[-1]
    sep_text = (
        f"Initial Separation: {sep_init:.2f} pc\n"
        f"Final Separation: {sep_final:.2f} pc"
    )
    plt.text(0.02, 0.98, sep_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


    ic = data_df['initial_conditions']
    ic_text = (
        f"Initial Conditions:\n"
        f"Mass: {ic['mass']}\n"
        f"Separation: {ic['sep']} pc\n"
        f"Angle: {ic['angle']}°\n"
        f"Velocity: {ic['velocity']} km/s"
    )
    plt.text(0.02, 0.02, ic_text, transform=plt.gca().transAxes,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

def plot_df_grav_vel_components_vs_time(data_df):
    """
    Plot x, y, z components vs. time for:
      1) DF acceleration
      2) Gravitational acceleration
      3) Velocity
    Each in its own subplot with consistent coloring.
    """

    t = data_df['t']

    # Convert to arrays for easier slicing
    a_df = np.array(data_df['a_df1'])     # shape: (N, 3)
    a_grav = np.array(data_df['a_grav1']) # shape: (N, 3)
    v = np.array(data_df['v1'])           # shape: (N, 3)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- 1) DF Acceleration ---
    axes[0].plot(t, a_df[:, 0], 'r-', label='X')
    axes[0].plot(t, a_df[:, 1], 'g-', label='Y')
    axes[0].plot(t, a_df[:, 2], 'b-', label='Z')
    axes[0].set_ylabel('a_DF [pc/yr²]', fontsize=12)
    axes[0].set_title('Dynamical Friction Acceleration Components', fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend(fontsize=10)

    # --- 2) Gravitational Acceleration ---
    axes[1].plot(t, a_grav[:, 0], 'r-', label='X')
    axes[1].plot(t, a_grav[:, 1], 'g-', label='Y')
    axes[1].plot(t, a_grav[:, 2], 'b-', label='Z')
    axes[1].set_ylabel('a_grav [pc/yr²]', fontsize=12)
    axes[1].set_title('Gravitational Acceleration Components', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # --- 3) Velocity ---
    axes[2].plot(t, v[:, 0], 'r-', label='X')
    axes[2].plot(t, v[:, 1], 'g-', label='Y')
    axes[2].plot(t, v[:, 2], 'b-', label='Z')
    axes[2].set_ylabel('Velocity [pc/yr]', fontsize=12)
    axes[2].set_xlabel('Time [yr]', fontsize=12)
    axes[2].set_title('Velocity Components', fontsize=14)
    axes[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
