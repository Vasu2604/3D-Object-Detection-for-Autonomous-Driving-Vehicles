import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os

def visualize_boxes_simple(lidar_points, pred_boxes, gt_boxes, output_path, sample_idx):
    """Simple bird's eye view visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot predictions
    ax1.scatter(lidar_points[:, 0], lidar_points[:, 1], c=lidar_points[:, 2], 
                s=0.1, cmap='viridis', alpha=0.3)
    
    if len(pred_boxes) > 0:
        for box in pred_boxes:
            corners = box[:4]  # x, y, z, l, w, h, yaw
            # Draw simple rectangle for bird's eye view
            x, y = corners[0], corners[1]
            l, w = corners[3], corners[4]
            ax1.plot([x-l/2, x+l/2, x+l/2, x-l/2, x-l/2],
                    [y-w/2, y-w/2, y+w/2, y+w/2, y-w/2], 'r-', linewidth=2)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'Predicted 3D Boxes (Sample {sample_idx})')
    ax1.axis('equal')
    ax1.grid(True)
    
    # Plot ground truth
    ax2.scatter(lidar_points[:, 0], lidar_points[:, 1], c=lidar_points[:, 2], 
                s=0.1, cmap='viridis', alpha=0.3)
    
    if len(gt_boxes) > 0:
        for box in gt_boxes:
            corners = box[:4]
            x, y = corners[0], corners[1]
            l, w = corners[3], corners[4]
            ax2.plot([x-l/2, x+l/2, x+l/2, x-l/2, x-l/2],
                    [y-w/2, y-w/2, y+w/2, y+w/2, y-w/2], 'g-', linewidth=2)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Ground Truth 3D Boxes (Sample {sample_idx})')
    ax2.axis('equal')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# Load validation data
data_root = 'data/nuscenes/'
val_pkl = 'data/nuscenes/nuscenes_infos_val.pkl'

print("Loading validation data...")
with open(val_pkl, 'rb') as f:
    val_data = pickle.load(f)

print(f"Total samples: {len(val_data['data_list'])}")

# Create output directory
os.makedirs('qualitative_results', exist_ok=True)

# Visualize first 3 samples
for idx in [0, 10, 20]:  # Different samples
    if idx >= len(val_data['data_list']):
        break
        
    sample_info = val_data['data_list'][idx]
    lidar_path = os.path.join(data_root, sample_info['lidar_points']['lidar_path'])
    
    # Load point cloud
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    
    # Sample points for visualization (too many points slow down rendering)
    if len(points) > 10000:
        indices = np.random.choice(len(points), 10000, replace=False)
        points = points[indices]
    
    # Get GT boxes
    if 'instances' in sample_info:
        gt_boxes = sample_info['instances']['gt_bboxes_3d']
    else:
        gt_boxes = sample_info.get('gt_bboxes_3d', np.array([]))
    
    # For now, use GT as "predictions" since we need model inference
    pred_boxes = gt_boxes  # Placeholder
    
    output_path = f'qualitative_results/visualization_{idx:03d}.png'
    visualize_boxes_simple(points, pred_boxes, gt_boxes, output_path, idx)

print("\nVisualization complete! Check 'qualitative_results/' directory.")
