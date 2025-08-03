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
