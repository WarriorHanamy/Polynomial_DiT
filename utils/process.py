import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import os
from pathlib import Path

def generate_xy_distribution_plot(condition_xy_data, trajectory_xy_data, condition_data, output_file):
    """
    Generate a triple distribution plot:
    - Left: condition x-y distribution
    - Middle: 40-point trajectory x-y distribution
    - Right: theta distribution

    Args:
        condition_xy_data: list of x-y coordinates for conditions.
        trajectory_xy_data: list of x-y coordinates for 40 trajectory points.
        condition_data: full condition vectors (including theta) list.
        output_file: output file name (used to derive plot filename).
    """
    if not condition_xy_data or not trajectory_xy_data or not condition_data:
        print("Not enough data to generate distribution plot.")
        return
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    condition_xy = np.array(condition_xy_data)
    trajectory_xy = np.array(trajectory_xy_data)
    condition_data = np.array(condition_data)
    theta_data = condition_data[:, 3]  # 4th dim is theta
    
    condition_total = len(condition_xy)
    trajectory_total = len(trajectory_xy)
    condition_has_nan = np.isnan(condition_xy).any()
    condition_has_inf = np.isinf(condition_xy).any()
    trajectory_has_nan = np.isnan(trajectory_xy).any()
    trajectory_has_inf = np.isinf(trajectory_xy).any()
    theta_has_nan = np.isnan(theta_data).any()
    theta_has_inf = np.isinf(theta_data).any()
    
    print("\n=== Data quality check ===")
    print(f"Num condition samples: {condition_total:,}")
    print(f"Num trajectory points: {trajectory_total:,}")
    print(f"Conditions - NaN: {condition_has_nan}, Inf: {condition_has_inf}")
    print(f"Trajectories - NaN: {trajectory_has_nan}, Inf: {trajectory_has_inf}")
    print(f"Theta - NaN: {theta_has_nan}, Inf: {theta_has_inf}")
    
    if condition_has_nan or condition_has_inf:
        valid_mask = np.isfinite(condition_xy).all(axis=1)
        condition_xy = condition_xy[valid_mask]
        theta_data = theta_data[valid_mask]
        print(f"Conditions after filtering: {len(condition_xy):,}")
    
    if trajectory_has_nan or trajectory_has_inf:
        valid_mask = np.isfinite(trajectory_xy).all(axis=1)
        trajectory_xy = trajectory_xy[valid_mask]
        print(f"Trajectories after filtering: {len(trajectory_xy):,}")
    
    if len(condition_xy) == 0 or len(trajectory_xy) == 0:
        print("No valid data left to plot.")
        return
        
    print("✅ Data quality OK: invalid values filtered.")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Distribution Analysis: Conditions, Trajectory Points and Theta', fontsize=16, fontweight='bold')
    
    condition_color = '#0066FF'
    trajectory_color = '#FF1A1A'
    theta_color = '#00CC00'
    
    scatter1 = ax1.scatter(condition_xy[:, 0], condition_xy[:, 1], 
                          c=condition_color, s=15, alpha=0.6, marker='o')
    ax1.set_xlabel('X Coordinate (m)')
    ax1.set_ylabel('Y Coordinate (m)')
    ax1.set_title('Condition Data (X-Y Coordinates)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    cond_stats_text = f"Samples: {len(condition_xy):,}\n"
    cond_stats_text += f"X: [{condition_xy[:, 0].min():.2f}, {condition_xy[:, 0].max():.2f}]\n"
    cond_stats_text += f"Y: [{condition_xy[:, 1].min():.2f}, {condition_xy[:, 1].max():.2f}]\n"
    cond_stats_text += f"X: μ={condition_xy[:, 0].mean():.2f}, σ={condition_xy[:, 0].std():.2f}\n"
    cond_stats_text += f"Y: μ={condition_xy[:, 1].mean():.2f}, σ={condition_xy[:, 1].std():.2f}"
    
    ax1.text(0.02, 0.98, cond_stats_text, transform=ax1.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#99CCFF", alpha=0.8, edgecolor=condition_color))
    
    scatter2 = ax2.scatter(trajectory_xy[:, 0], trajectory_xy[:, 1], 
                          c=trajectory_color, s=8, alpha=0.4, marker='.')
    ax2.set_xlabel('X Coordinate (m)')
    ax2.set_ylabel('Y Coordinate (m)')
    ax2.set_title('Trajectory Points (40 Points × N Samples)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    traj_stats_text = f"Points: {len(trajectory_xy):,}\n"
    traj_stats_text += f"X: [{trajectory_xy[:, 0].min():.2f}, {trajectory_xy[:, 0].max():.2f}]\n"
    traj_stats_text += f"Y: [{trajectory_xy[:, 1].min():.2f}, {trajectory_xy[:, 1].max():.2f}]\n"
    traj_stats_text += f"X: μ={trajectory_xy[:, 0].mean():.2f}, σ={trajectory_xy[:, 0].std():.2f}\n"
    traj_stats_text += f"Y: μ={trajectory_xy[:, 1].mean():.2f}, σ={trajectory_xy[:, 1].std():.2f}"
    
    ax2.text(0.02, 0.98, traj_stats_text, transform=ax2.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFCCCC", alpha=0.8, edgecolor=trajectory_color))
    
    ax3.hist(np.rad2deg(theta_data), bins=50, color=theta_color, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Theta (degrees)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Theta Distribution')
    ax3.grid(True, alpha=0.3)
    
    theta_stats_text = f"Samples: {len(theta_data):,}\n"
    theta_stats_text += f"Range: [{np.rad2deg(theta_data.min()):.2f}°, {np.rad2deg(theta_data.max()):.2f}°]\n"
    theta_stats_text += f"Mean: {np.rad2deg(theta_data.mean()):.2f}°\n"
    theta_stats_text += f"Std: {np.rad2deg(theta_data.std()):.2f}°"
    
    ax3.text(0.02, 0.98, theta_stats_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#CCFFCC", alpha=0.8, edgecolor=theta_color))
    
    plt.tight_layout()
    
    base_name = Path(output_file).stem
    plot_filename = logs_dir / f"{base_name}_triple_distribution.png"
    
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # Close figure to free memory
    
    print(f"\nTriple distribution plot saved to: {plot_filename}")
    
    cond_x_range = condition_xy[:, 0].max() - condition_xy[:, 0].min()
    cond_y_range = condition_xy[:, 1].max() - condition_xy[:, 1].min()
    traj_x_range = trajectory_xy[:, 0].max() - trajectory_xy[:, 0].min()
    traj_y_range = trajectory_xy[:, 1].max() - trajectory_xy[:, 1].min()
    theta_range = np.rad2deg(theta_data.max() - theta_data.min())
    
    print(f"Condition range: ΔX={cond_x_range:.2f} m, ΔY={cond_y_range:.2f} m")
    print(f"Trajectory range: ΔX={traj_x_range:.2f} m, ΔY={traj_y_range:.2f} m")
    print(f"Theta range: {theta_range:.2f}°")
    
    expected_points_per_sample = 40
    num_samples = len(condition_xy)
    actual_trajectory_points = len(trajectory_xy)
    expected_trajectory_points = num_samples * expected_points_per_sample
    
    print(f"Trajectory points: expected {expected_trajectory_points:,} (num_samples×40), "
          f"actual {actual_trajectory_points:,}")

def least_squares_fit_and_sample_around_reference(points, reference_point_index, poly_degree=4, 
                                                 front_start_pct=0.05, front_end_pct=0.3, 
                                                 back_start_pct=0.3, back_end_pct=0.05,
                                                 front_samples=3, back_samples=3):
    """
    Fit a polynomial curve with least squares and resample points around a reference,
    with configurable exclusion regions.
    
    Args:
        points: array of 3D points (length ≥ 2).
        reference_point_index: index of the reference point in the original points.
        poly_degree: polynomial degree.
        front_start_pct: start exclusion percentage on the front half (near 0).
        front_end_pct: end exclusion percentage on the front half (near reference).
        back_start_pct: start exclusion percentage on the back half (near reference).
        back_end_pct: end exclusion percentage on the back half (near 1).
        front_samples: number of samples in the front valid region.
        back_samples: number of samples in the back valid region.
    
    Returns:
        resampled_points: concatenated front+back sampled points.
        fitted_coeffs: fitted polynomial coefficients for x/y/z.
        sampling_info: detailed sampling information dictionary.
    """
    points = np.array(points)
    n_points = len(points)
    
    if n_points < 2:
        return points, [], {}
    
    t = np.linspace(0, 1, n_points)
    
    fitted_coeffs = []
    
    for coord_idx in range(3):  # x, y, z
        coord_values = points[:, coord_idx]
        
        effective_degree = min(poly_degree, n_points - 1)
        
        try:
            coeffs = np.polyfit(t, coord_values, effective_degree)
            fitted_coeffs.append(coeffs)
        except np.linalg.LinAlgError:
            print(f"Coordinate {coord_idx} fitting failed, using linear fit instead.")
            coeffs = np.polyfit(t, coord_values, 1)
            fitted_coeffs.append(coeffs)
    
    reference_t = t[reference_point_index]
    
    # Front half: 0 → reference_t
    front_region_start = 0
    front_region_end = reference_t
    front_length = front_region_end - front_region_start
    
    front_exclude_start1 = front_region_start
    front_exclude_end1 = front_region_start + front_length * front_start_pct
    front_exclude_start2 = front_region_end - front_length * front_end_pct
    front_exclude_end2 = front_region_end
    
    front_valid_start = front_exclude_end1
    front_valid_end = front_exclude_start2
    
    # Back half: reference_t → 1
    back_region_start = reference_t
    back_region_end = 1.0
    back_length = back_region_end - back_region_start
    
    back_exclude_start1 = back_region_start
    back_exclude_end1 = back_region_start + back_length * back_start_pct
    back_exclude_start2 = back_region_end - back_length * back_end_pct
    back_exclude_end2 = back_region_end
    
    back_valid_start = back_exclude_end1
    back_valid_end = back_exclude_start2
    
    front_valid_length = max(0, front_valid_end - front_valid_start)
    back_valid_length = max(0, back_valid_end - back_valid_start)
    
    resampled_points = []
    t_front = []
    t_back = []
    
    if front_valid_length > 0 and front_samples > 0:
        if front_samples == 1:
            t_front = [(front_valid_start + front_valid_end) / 2]
        else:
            t_front = np.linspace(front_valid_start, front_valid_end, front_samples).tolist()
    
    if back_valid_length > 0 and back_samples > 0:
        if back_samples == 1:
            t_back = [(back_valid_start + back_valid_end) / 2]
        else:
            t_back = np.linspace(back_valid_start, back_valid_end, back_samples).tolist()
    
    all_t_values = t_front + t_back
    
    for t_val in all_t_values:
        point = []
        for coord_idx in range(3):
            coord_value = np.polyval(fitted_coeffs[coord_idx], t_val)
            point.append(coord_value)
        resampled_points.append(point)
    
    sampling_info = {
        'reference_t': reference_t,
        'original_t': t,
        
        'params': {
            'front_start_pct': front_start_pct,
            'front_end_pct': front_end_pct,
            'back_start_pct': back_start_pct,
            'back_end_pct': back_end_pct,
            'front_samples': front_samples,
            'back_samples': back_samples
        },
        
        'front_region': (front_region_start, front_region_end),
        'excluded_front': (front_exclude_start1, front_exclude_end1, front_exclude_start2, front_exclude_end2),
        'front_valid_region': (front_valid_start, front_valid_end),
        't_front': t_front,
        
        'back_region': (back_region_start, back_region_end),
        'excluded_back': (back_exclude_start1, back_exclude_end1, back_exclude_start2, back_exclude_end2),
        'back_valid_region': (back_valid_start, back_valid_end),
        't_back': t_back,
        
        # Backward-compatible keys (kept for legacy code)
        'before_region': (front_valid_start, front_valid_end),
        'after_region': (back_valid_start, back_valid_end),
        't_before': t_front,
        't_after': t_back
    }
    
    return np.array(resampled_points), fitted_coeffs, sampling_info

def calculate_rmse(original_points, fitted_coeffs, reference_index=3):
    """
    Compute RMSE between original data points and the fitted curve
    using the true shortest distance for each point.
    
    Args:
        original_points: original points (including the reference point).
        fitted_coeffs: polynomial coefficients [x_coeffs, y_coeffs, z_coeffs].
        reference_index: index of the reference point to exclude from error.
    
    Returns:
        rmse: root-mean-square error.
    """
    original_points = np.array(original_points)
    n_points = len(original_points)
    
    mask = np.ones(n_points, dtype=bool)
    mask[reference_index] = False
    data_points = original_points[mask]
    
    def curve_point(t):
        point = []
        for coord_idx in range(3):
            coord_value = np.polyval(fitted_coeffs[coord_idx], t)
            point.append(coord_value)
        return np.array(point)
    
    def distance_to_curve(data_point):
        """Compute minimal distance from a data point to the fitted curve."""
        t_values = np.linspace(0, 1, 1000)
        distances = []
        
        for t in t_values:
            curve_pt = curve_point(t)
            dist = np.linalg.norm(data_point - curve_pt)
            distances.append(dist)
        
        return min(distances)
    
    errors = []
    for point in data_points:
        min_distance = distance_to_curve(point)
        errors.append(min_distance)
    
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    return rmse


def process_entire_dataset(input_file, output_file,
                          poly_degree=4, front_start_pct=0.05, front_end_pct=0.3, 
                          back_start_pct=0.3, back_end_pct=0.05,
                          front_samples=3, back_samples=3,
                          augment_data=True, augment_factor=3,
                          xy_noise_std=0.01, theta_noise_std=np.deg2rad(1)):
    """
    Process the entire dataset:
    - Replace original 40 points with 6 resampled points.
    - Append polynomial coefficients.
    - Support data augmentation.
    Uses the indexed data format consistently.
    
    Args:
        input_file: input file path.
        output_file: output file path.
        poly_degree: polynomial degree.
        front_start_pct, front_end_pct, back_start_pct, back_end_pct: exclusion region config.
        front_samples, back_samples: number of samples in front/back regions.
        augment_data: whether to perform data augmentation.
        augment_factor: how many augmented variants per original row.
        xy_noise_std: std of xy noise (meters).
        theta_noise_std: std of theta noise (radians).
    
    Returns:
        Number of successfully processed original lines.
    """
    print(f"\nStart processing dataset: {input_file}")
    print(f"Output file: {output_file}")
    print("Input format: index + 4 condition dims + 40 points (120 dims) = 125 dims total")
    print("Output format: index + 4 condition dims + 6 points (18 dims) + 15 polynomial coeffs = 38 dims total")
    if augment_data:
        print(f"Data augmentation: ON, {augment_factor} variants per line.")
        print(f"XY noise std: ±{xy_noise_std:.3f} m")
        print(f"Theta noise std: ±{np.rad2deg(theta_noise_std):.1f}°")
    else:
        print("Data augmentation: OFF")
    print("="*60)
    
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: cannot find file {input_file}")
        return 0
    
    processed_lines = []
    successful_count = 0
    failed_count = 0
    total_generated = 0
    
    # For collecting x-y distribution data.
    condition_xy_data = []   # condition x-y
    trajectory_xy_data = []  # 40-point x-y
    condition_data = []      # full condition (4 dims)
    
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        try:
            data_parts = line.split(',')
            
            if len(data_parts) != 125:  # index + 124 dims
                print(f"Line {line_idx+1}: invalid length ({len(data_parts)} != 125)")
                failed_count += 1
                continue
            
            data_index = int(data_parts[0])
            data = [float(x) for x in data_parts[1:]]
            
            if any(not math.isfinite(x) for x in data):
                print(f"Line {line_idx+1}: contains non-finite values.")
                failed_count += 1
                continue
            
            original_prefix_data = data[:4]
            point_coords = data[4:]
            
            points = []
            valid_data = True
            
            for i in range(40):
                x = point_coords[i*3]
                y = point_coords[i*3 + 1] 
                z = point_coords[i*3 + 2]
                    
                points.append([x, y, z])
            
            augment_count = augment_factor if augment_data else 1
            line_success = False
            
            for aug_idx in range(augment_count):
                try:
                    if aug_idx == 0:
                        prefix_data = original_prefix_data.copy()
                    else:
                        prefix_data = original_prefix_data.copy()
                        
                        xy_noise = np.random.normal(0, xy_noise_std, 2)
                        prefix_data[0] += xy_noise[0]
                        prefix_data[1] += xy_noise[1]
                        
                        theta_noise = np.random.normal(0, theta_noise_std)
                        prefix_data[3] += theta_noise
                    
                    # Drop first/last 8 points in each half to reduce edge noise.
                    front_points = points[8:20]   # 12 points
                    back_points = points[20:32]   # 12 points
                    
                    # Reference point: average of the last front and first back point.
                    front_last_point = points[19]
                    back_first_point = points[20]
                    
                    calculated_point = [
                        (front_last_point[0] + back_first_point[0]) / 2,
                        (front_last_point[1] + back_first_point[1]) / 2,
                        (front_last_point[2] + back_first_point[2]) / 2
                    ]
                    
                    ordered_points = front_points + [calculated_point] + back_points
                    
                    fitted_points, coeffs, sampling_info = least_squares_fit_and_sample_around_reference(
                        ordered_points, 12, poly_degree=poly_degree,
                        front_start_pct=front_start_pct, front_end_pct=front_end_pct,
                        back_start_pct=back_start_pct, back_end_pct=back_end_pct,
                        front_samples=front_samples, back_samples=back_samples
                    )
                    
                    if len(fitted_points) != 6:
                        assert False, "fitted_points length is not 6."
                    
                    new_data = prefix_data.copy()
                    
                    for point in fitted_points:
                        new_data.extend([point[0], point[1], point[2]])
                    
                    for coord_coeffs in coeffs:
                        for coeff in coord_coeffs:
                            new_data.append(float(coeff))
                    
                    if len(new_data) != 37:
                        print(f"Data length error: expected 37 dims, got {len(new_data)}.")
                        continue
                    
                    data_str = ','.join([f'{x:.6f}' for x in new_data])
                    new_line = f"{data_index},{data_str}"
                    processed_lines.append(new_line)
                    total_generated += 1
                    line_success = True
                    
                    # Collect data for distribution plot (only from first augmented variant).
                    if aug_idx == 0:
                        condition_xy = [prefix_data[0], prefix_data[1]]
                        condition_xy_data.append(condition_xy)
                        
                        for point in points:
                            trajectory_xy_data.append([point[0], point[1]])
                        
                        condition_data.append(prefix_data)
                    
                except Exception as e:
                    if aug_idx == 0:
                        print(f"Line {line_idx+1}: fitting failed - {str(e)}")
                    continue
            
            if line_success:
                successful_count += 1
            else:
                failed_count += 1
                
            if (line_idx + 1) % 100 == 0:
                print(f"Processed {line_idx + 1} lines, success: {successful_count}, "
                      f"failed: {failed_count}, generated: {total_generated}")
                    
        except Exception as e:
            print(f"Line {line_idx+1}: processing failed - {str(e)}")
            failed_count += 1
            continue
    
    try:
        with open(output_file, 'w') as f:
            for line in processed_lines:
                f.write(line + '\n')
        
        print("\nProcessing finished!")
        print(f"Total original lines: {len(lines)}")
        print(f"Successfully processed: {successful_count}")
        print(f"Failed: {failed_count}")
        print(f"Success rate: {successful_count/(successful_count+failed_count)*100:.1f}%")
        if augment_data:
            print(f"Augmentation factor: {augment_factor}x")
            print(f"Total generated lines: {total_generated}")
            print(f"Avg variants per successful line: {total_generated/successful_count:.1f}" if successful_count > 0 else "N/A")
        print(f"Output file: {output_file}")
        print("Output format: index + 4 conditions + 18 sampled point dims + 15 poly coeffs = 38 dims")
        
        generate_xy_distribution_plot(condition_xy_data, trajectory_xy_data, condition_data, output_file)
        
        return successful_count
        
    except Exception as e:
        print(f"Failed to write output file: {str(e)}")
        return 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process indexed trajectory dataset.')
    parser.add_argument('--input', type=str, default="datasets/grasp_saw.txt", 
                       help='Input file path (indexed raw data).')
    parser.add_argument('--output', type=str, default="datasets/tmp.txt", 
                       help='Output file path.')
    parser.add_argument('--augment_factor', type=int, default=5, 
                       help='Data augmentation factor.')
    parser.add_argument('--xy_noise', type=float, default=0.01, 
                       help='XY coordinate noise std (meters).')
    parser.add_argument('--theta_noise', type=float, default=1.0, 
                       help='Theta noise std (degrees).')
    
    args = parser.parse_args()
    
    input_file = args.input
    if not os.path.exists(input_file):
        print(f"Error: input file does not exist: {input_file}")
        exit(1)
    
    print(f"Using indexed data file: {input_file}")
    print("Starting processing of dataset with index preservation...")
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    if not lines:
        print(f"Error: input file is empty: {input_file}")
        exit(1)
        
    indices = []
    for line in lines:
        try:
            parts = line.strip().split(',')
            if not parts:
                continue
            index = int(parts[0])
            indices.append(index)
        except (ValueError, IndexError) as e:
            print(f"Warning: skipping invalid line - {str(e)}")
            continue
            
    if not indices:
        print("Error: no valid index data found.")
        exit(1)
        
    indices = sorted(indices)
    expected_indices = list(range(min(indices), max(indices)+1))
    
    if indices != expected_indices:
        print("\n⚠️ Warning: non-contiguous indices found!")
        missing = set(expected_indices) - set(indices)
        if missing:
            print(f"Missing indices: {sorted(list(missing))}")
    else:
        print("\n✅ Index check passed: all indices are contiguous.")
        
    print(f"Index range: {min(indices)} - {max(indices)}")
    print(f"Total lines: {len(indices)}")
    
    print("\n" + "="*60)
    print("Start processing entire dataset...")
    successful_count = process_entire_dataset(
        input_file=input_file,
        output_file=args.output,
        poly_degree=4,
        front_start_pct=0.05,
        front_end_pct=0.4,
        back_start_pct=0.4,
        back_end_pct=0.05,
        front_samples=3,
        back_samples=3,
        augment_data=True,
        augment_factor=args.augment_factor,
        xy_noise_std=args.xy_noise,
        theta_noise_std=np.deg2rad(args.theta_noise)
    )
    
    if successful_count > 0:
        print(f"\n✅ Dataset processing finished. Successfully processed {successful_count} lines.")
        print(f"New data file generated: {args.output}")
    else:
        print("\n❌ Dataset processing failed.")