#!/usr/bin/env python3
"""
Point trajectory generation demo.
Visualizes multiple samples comparing predicted 4 points with reference points (3D + XY projection).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec

from utils.dataset import TrajectoryDataset
from utils.points_evaluation import PointEvaluator 

np.random.seed(int(1))

def create_simplified_points_demo(model_path="checkpoints/Polynomial-DiT-local/strike_checkpoint_latest_4_points.pth", 
                              processed_data_file="datasets/strike_processed.txt",
                              save_dir="logs",
                              fig_width=15,
                              fig_height=5,
                              left_margin=0.06,
                              right_margin=0.96,
                              top_margin=0.85,
                              bottom_margin=0.15,
                              wspace=0.3,
                              dpi=300,
                              legend_y=1.0):
    """
    Create a simplified 2D demo with 3 subplots.
    Each subplot shows the predicted 4 points and the input point,
    with an inclined plane box and slope angle annotation.

    Args:
        model_path: Path to the trained model checkpoint.
        processed_data_file: Path to the preprocessed dataset file.
        save_dir: Directory to save the generated figures.
        Other args: Layout / plotting parameters.
    """
    print("=== Creating simplified 4-point prediction demo (predicted points + input point + slope angle) ===")
    os.makedirs(save_dir, exist_ok=True)
    
    font_path = './utils/Times New Roman.ttf'
    prop = fm.FontProperties(fname=font_path)
    font_name = prop.get_name()
    fm.fontManager.addfont(font_path)
    
    plt.rcParams.update({
        'font.family': font_name,
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 20,
        'axes.linewidth': 2.0,
        'grid.linewidth': 0.8,
        'lines.linewidth': 5.0
    })
    
    morandi_colors = {
        'pred_points': '#D4A5A5',
        'input_point': '#C4A484',
        'slope_box': '#A8B5A8',
        'slope_edge': '#7B8A8B'
    }
    
    evaluator = PointEvaluator(model_path)
    
    dataset = TrajectoryDataset(processed_data_file, normalize=True, target_type="points")
    condition_scaler, point_scaler = dataset.get_scalers()
    
    selected_indices = np.random.choice(len(dataset), 3, replace=False)
    print(f"Selected sample indices: {selected_indices}")
    
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
    
    all_x_coords = []
    all_y_coords = []
    all_trajectory_data = []
    
    for fig_idx, data_idx in enumerate(selected_indices):
        condition, true_points = dataset[data_idx]
        condition_np = condition.numpy()
        
        print(f"Figure {fig_idx+1}, sample index {data_idx}")
        print(f"  Condition: [{condition_np[0]:.3f}, {condition_np[1]:.3f}, {condition_np[2]:.3f}]")
        
        condition_batch = condition.unsqueeze(0)
        pred_points = evaluator.predict_points(condition_batch)
        pred_points_denorm = point_scaler.inverse_transform(pred_points.cpu().numpy())[0]
        pred_points_reshaped = pred_points_denorm.reshape(4, 3)
        
        condition_denorm = condition_scaler.inverse_transform(condition_np.reshape(1, -1))[0]
        
        all_x_coords.extend(pred_points_reshaped[:, 0])
        all_x_coords.append(condition_denorm[0])
        all_y_coords.extend(pred_points_reshaped[:, 1])
        all_y_coords.append(condition_denorm[1])
        
        all_trajectory_data.append({
            'pred_points': pred_points_reshaped,
            'condition': condition_denorm,
            'slope_angle': condition_denorm[2]
        })
        print(f"  Slope angle: {condition_denorm[2]:.3f}")
    
    x_min, x_max = min(all_x_coords), max(all_x_coords)
    y_min, y_max = min(all_y_coords), max(all_y_coords)
    
    x_margin = (x_max - x_min) * 0.15
    y_margin = (y_max - y_min) * 0.15
    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin
    
    for fig_idx, data in enumerate(all_trajectory_data):
        ax = axes[fig_idx]
        
        ax.plot(data['pred_points'][:, 0], data['pred_points'][:, 1], 
                color=morandi_colors['pred_points'], linewidth=4, alpha=0.9,
                marker='o', markersize=12, label='Predicted 4 Points' if fig_idx == 0 else '')
        
        ax.scatter([data['condition'][0]], [data['condition'][1]], 
                   c=morandi_colors['input_point'], s=600, marker='*', 
                   edgecolors='black', linewidth=2, zorder=10,
                   label='Input Point' if fig_idx == 0 else '')
        
        box_width = 0.8
        box_height = 0.1
        box_x = -box_width/2
        box_y = -box_height/2 + 0.05
        from matplotlib.patches import Rectangle
        slope_box = Rectangle((box_x, box_y), box_width, box_height,
                             facecolor=morandi_colors['slope_box'], 
                             edgecolor=morandi_colors['slope_edge'],
                             linewidth=2, alpha=0.8)
        ax.add_patch(slope_box)
        
        text_x = box_x + box_width / 2
        text_y =  box_height/2 + 0.25
        ax.text(text_x, text_y, f'Inclined: {data["slope_angle"]:.3f}', 
                horizontalalignment='center', verticalalignment='bottom',
                fontsize=16, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('X (m)', fontsize=18)
        ax.set_ylabel('Y (m)', fontsize=18)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.locator_params(nbins=5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    
    legend_elements = [
        plt.Line2D([0], [0], color=morandi_colors['pred_points'], linewidth=4, label='Predicted 4 Points'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=morandi_colors['input_point'], 
                  markersize=18, label='Input Point', markeredgecolor='black', markeredgewidth=1.5),
        plt.Rectangle((0, 0), 1, 1, facecolor=morandi_colors['slope_box'], 
                      edgecolor=morandi_colors['slope_edge'], label='Inclined Plane')
    ]
    fig.legend(handles=legend_elements, 
              loc='upper center',
              bbox_to_anchor=(0.35, legend_y),
              ncol=3,
              fontsize=18,
              frameon=True,
              framealpha=0.95,
              fancybox=True,
              shadow=True,
              edgecolor='gray',
              facecolor='white',
              columnspacing=2.0,
              handlelength=2.5,
              handletextpad=0.8)
    plt.subplots_adjust(left=left_margin, right=right_margin, 
                       top=top_margin, bottom=bottom_margin, 
                       wspace=wspace)
    save_path = os.path.join(save_dir, 'simplified_points_demo.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    print("Simplified points demo figure saved to:")
    print(f"  PNG: {save_path}")
    print(f"Figure size: {fig_width:.1f} x {fig_height:.1f} inches")
    plt.rcdefaults()
    return save_path


if __name__ == "__main__":
    create_simplified_points_demo() 