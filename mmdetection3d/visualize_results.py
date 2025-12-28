import os
import numpy as np
import torch
from mmengine.config import Config
from mmdet3d.apis import init_model, inference_detector
from mmdet3d.visualization import Det3DLocalVisualizer
import mmcv
from mmengine.structures import InstanceData
import matplotlib.pyplot as plt
from pathlib import Path

# Load config and checkpoint
config_file = 'configs/centerpoint/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
checkpoint_file = 'work_dirs/custom_centerpoint_mini/epoch_20.pth'

# Initialize model
model = init_model(config_file, checkpoint_file, device='cuda:0')

# Get some validation samples
data_root = 'data/nuscenes/'
val_pkl = 'data/nuscenes/nuscenes_infos_val.pkl'

import pickle
with open(val_pkl, 'rb') as f:
    val_data = pickle.load(f)

print(f"Total validation samples: {len(val_data['data_list'])}")

# Create output directory
output_dir = 'qualitative_results'
os.makedirs(output_dir, exist_ok=True)

# Visualize first 5 samples
num_samples = min(5, len(val_data['data_list']))
print(f"Generating visualizations for {num_samples} samples...")

for idx in range(num_samples):
    sample_info = val_data['data_list'][idx]
    lidar_path = os.path.join(data_root, sample_info['lidar_points']['lidar_path'])
    
    # Run inference
    result = inference_detector(model, lidar_path)
    
    # Visualize
    visualizer = Det3DLocalVisualizer()
    
    # Load point cloud
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    
    # Save visualization
    output_file = os.path.join(output_dir, f'sample_{idx:03d}.png')
    
    visualizer.set_points(points)
    visualizer.draw_bboxes_3d(result.pred_instances_3d.bboxes_3d)
    visualizer.show(save_path=output_file)
    
    print(f"Saved visualization to {output_file}")

print("\nQualitative visualizations completed!")
