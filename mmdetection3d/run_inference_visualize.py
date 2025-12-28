import numpy as np
import pickle
import torch
from mmengine.config import Config, DictAction
from mmdet3d.apis import init_model, inference_detector
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

def corners_3d(bbox_3d):
    """Get 8 corners of a 3D bounding box."""
    # bbox_3d: [x, y, z, l, w, h, yaw]
    x, y, z, l, w, h, yaw = bbox_3d
    
    # Create corners in local coordinate system
    x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
    y_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
    z_corners = np.array([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2])
    
    corners = np.vstack([x_corners, y_corners, z_corners])
    
    # Rotate around z-axis
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    corners = rotation_matrix @ corners
    
    # Translate
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z
    
    return corners.T

def visualize_detection(points, pred_boxes, pred_labels, pred_scores, output_path):
    """Visualize 3D detection results."""
    fig = plt.figure(figsize=(20, 10))
    
    # Bird's eye view
    ax1 = fig.add_subplot(121)
    
    # Sample points for visualization
    if len(points) > 10000:
        indices = np.random.choice(len(points), 10000, replace=False)
        vis_points = points[indices]
    else:
        vis_points = points
    
    ax1.scatter(vis_points[:, 0], vis_points[:, 1], c=vis_points[:, 2], 
                s=0.5, cmap='viridis', alpha=0.3)
    
    # Define colors for different classes
    class_colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'brown', 'pink']
    class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    
    # Draw predicted boxes
    for bbox, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if score > 0.3:  # Confidence threshold
            x, y, z, l, w, h, yaw = bbox[:7]
            color = class_colors[int(label) % len(class_colors)]
            
            # Get corners for bird's eye view
            corners = corners_3d(bbox[:7])
            # Draw box footprint
            footprint = corners[:4]  # bottom 4 corners
            footprint = np.vstack([footprint, footprint[0]])  # close the loop
            ax1.plot(footprint[:, 0], footprint[:, 1], color=color, linewidth=2, alpha=0.8)
            
            # Add label
            label_name = class_names[int(label)] if int(label) < len(class_names) else f'cls_{int(label)}'
            ax1.text(x, y, f'{label_name}\n{score:.2f}', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
    
    ax1.set_xlabel('X (meters)', fontsize=12)
    ax1.set_ylabel('Y (meters)', fontsize=12)
    ax1.set_title('Bird\'s Eye View - Predicted 3D Bounding Boxes', fontsize=14, fontweight='bold')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-50, 50)
    
    # 3D perspective view
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Sample points for 3D view
    sample_indices = np.random.choice(len(points), min(5000, len(points)), replace=False)
    ax2.scatter(points[sample_indices, 0], 
               points[sample_indices, 1], 
               points[sample_indices, 2], 
               c=points[sample_indices, 2], s=1, cmap='viridis', alpha=0.2)
    
    # Draw 3D boxes
    for bbox, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if score > 0.3:
            color = class_colors[int(label) % len(class_colors)]
            corners = corners_3d(bbox[:7])
            
            # Define the 12 edges of a box
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # top face
                [4, 5], [5, 6], [6, 7], [7, 4],  # bottom face
                [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
            ]
            
            for edge in edges:
                points_edge = corners[edge]
                ax2.plot3D(*points_edge.T, color=color, linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.set_zlabel('Z (m)', fontsize=10)
    ax2.set_title('3D Perspective View', fontsize=14, fontweight='bold')
    ax2.set_xlim(-50, 50)
    ax2.set_ylim(-50, 50)
    ax2.set_zlim(-2, 5)
    
    plt.suptitle('Real-Time Inference: Fine-Tuned CenterPoint Model on NuScenes Sample', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()

# Configuration
config_file = 'configs/centerpoint/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
checkpoint_file = 'work_dirs/centerpoint_mini_finetune/epoch_20.pth'

print("Loading model...")
model = init_model(config_file, checkpoint_file, device='cuda:0')
print("Model loaded successfully!")

# Load validation data
val_pkl = 'data/nuscenes/nuscenes_infos_val.pkl'
print(f"Loading validation data from {val_pkl}...")
with open(val_pkl, 'rb') as f:
    val_data = pickle.load(f)

print(f"Total validation samples: {len(val_data['data_list'])}")

# Select a sample (use sample index 0)
sample_idx = 0
sample_info = val_data['data_list'][sample_idx]
lidar_path = os.path.join('data/nuscenes/', sample_info['lidar_points']['lidar_path'])

print(f"\nRunning inference on sample {sample_idx}...")
print(f"LiDAR file: {lidar_path}")

# Check if file exists
if not os.path.exists(lidar_path):
    print(f"ERROR: LiDAR file not found: {lidar_path}")
    print("Available lidar path in sample_info:")
    print(sample_info['lidar_points'])
    exit(1)

# Run inference
result = inference_detector(model, lidar_path)

print("\nInference completed!")
print(f"Detected objects: {len(result.pred_instances_3d.bboxes_3d)}")

# Load point cloud
points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
print(f"Point cloud shape: {points.shape}")

# Extract predictions
pred_boxes = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
pred_labels = result.pred_instances_3d.labels_3d.cpu().numpy()
pred_scores = result.pred_instances_3d.scores_3d.cpu().numpy()

print(f"\nPredictions:")
for i, (label, score) in enumerate(zip(pred_labels, pred_scores)):
    if score > 0.3:
        print(f"  Object {i}: Label={label}, Score={score:.3f}")

# Create output directory
os.makedirs('inference_results', exist_ok=True)
output_path = 'inference_results/real_inference_visualization.png'

# Visualize
print(f"\nGenerating visualization...")
visualize_detection(points, pred_boxes, pred_labels, pred_scores, output_path)

print("\nDone! Check 'inference_results/real_inference_visualization.png'")
