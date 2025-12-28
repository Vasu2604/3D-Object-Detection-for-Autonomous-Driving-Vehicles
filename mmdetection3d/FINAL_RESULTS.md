# CenterPoint 3D Object Detection - Final Results

## ✅ COMPLETED WORK

### Setup
- **Dataset**: nuScenes-mini (323 train, 81 val samples)
- **Model**: CenterPoint (SECOND backbone + SECFPN neck, voxel size 0.1m)
- **Pretrained**: Official MMDetection3D CenterPoint trained on full nuScenes (700 scenes)
- **Training**: Fine-tuned pretrained model on nuScenes-mini for 20 epochs

### Results

**Baseline (from scratch on mini)**:
- mAP: ~0.15
- NDS: ~0.21

**Fine-tuned from Pretrained Checkpoint**:
- **mAP: 0.3206** (↑113% improvement)
- **NDS: 0.4176** (↑101% improvement)

### Per-Class Performance
| Class | AP | mATE | mASE | mAOE |
|-------|-----|------|------|------|
| car | 0.749 | 0.221 | 0.173 | 0.305 |
| pedestrian | 0.840 | 0.145 | 0.289 | 0.305 |
| bus | 0.790 | 0.342 | 0.172 | 0.032 |
| truck | 0.522 | 0.134 | 0.174 | 0.220 |
| motorcycle | 0.160 | 0.237 | 0.341 | 0.694 |
| traffic_cone | 0.145 | 0.091 | 0.354 | nan |
| trailer | 0.000 | 1.000 | 1.000 | 1.000 |
| construction_vehicle | 0.000 | 1.000 | 1.000 | 1.000 |
| bicycle | 0.000 | 0.206 | 0.399 | 1.821 |
| barrier | 0.000 | 1.000 | 1.000 | 1.000 |

## Analysis

### Why NDS is 0.42 vs Official 0.65?

1. **Evaluation dataset**: mini (81 samples) vs full val (6019 samples)
2. **Class imbalance**: 4 classes have zero/near-zero examples in mini
3. **Domain**: mini is NOT representative of full nuScenes distribution

### Key Achievements

✅ **Doubled performance** by using pretrained weights
✅ **Strong results** on frequent classes (car, pedestrian, bus)
✅ **Complete pipeline**: data prep → training → evaluation
✅ **Proper comparison**: from-scratch vs pretrained fine-tuning

## Files & Checkpoints

- Config: `configs/centerpoint/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py`
- Pretrained: `checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth`
- Fine-tuned: `work_dirs/centerpoint_mini_finetune/epoch_20.pth`
- Results: `work_dirs/centerpoint_mini_finetune/results_eval/metrics_summary.json`

