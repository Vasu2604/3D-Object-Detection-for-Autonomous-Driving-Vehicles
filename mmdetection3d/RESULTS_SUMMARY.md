# CenterPoint 3D Detection on nuScenes - Results Summary

## Current Setup
- **Dataset**: nuScenes-mini (323 train, 81 val samples)
- **Model**: CenterPoint (SECOND + SECFPN, voxel 0.1m)
- **Pretrained Checkpoint**: Official nuScenes CenterPoint (trained on full 700-scene trainval)

## Results

### Baseline (Training from Scratch on Mini)
- mAP: ~0.15
- NDS: ~0.21

### Fine-Tuned from Pretrained Checkpoint  
- **mAP: 0.3206** (↑113%)
- **NDS: 0.4176** (↑101%)

### Per-Class Performance on Mini Val
| Class | AP | Trans Err | Scale Err |
|-------|-----|-----------|----------|
| car | 0.749 | 0.221 | 0.173 |
| pedestrian | 0.840 | 0.145 | 0.289 |
| bus | 0.790 | 0.342 | 0.172 |
| truck | 0.522 | 0.134 | 0.174 |
| motorcycle | 0.160 | 0.237 | 0.341 |
| traffic_cone | 0.145 | 0.091 | 0.354 |
| trailer | 0.000 | 1.000 | 1.000 |
| construction_vehicle | 0.000 | 1.000 | 1.000 |
| bicycle | 0.000 | 0.206 | 0.399 |
| barrier | 0.000 | 1.000 | 1.000 |

## Analysis

### Why NDS is 0.42 instead of 0.65?
1. **Evaluation on mini val**: Only 81 samples vs thousands in full val
2. **Extreme class imbalance**: 4 classes have zero or near-zero AP
3. **Domain**: Mini is a tiny subset, not representative of full nuScenes

### Key Insight
The **pretrained checkpoint encodes knowledge from full nuScenes (700 scenes)**, so we indirectly benefit from the full dataset even though we only fine-tune on mini.

## Checkpoint Used
```
checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth
```
Official MMDetection3D model trained on full nuScenes trainval.

