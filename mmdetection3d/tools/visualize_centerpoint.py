import os
import mmengine
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet3d.registry import MODELS, DATASETS
from mmdet3d.structures import Det3DDataSample
import matplotlib.pyplot as plt

def main():
    cfg_file = 'configs/centerpoint/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
    ckpt_file = 'work_dirs/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d/epoch_20.pth'
    out_dir = 'work_dirs/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d/custom_vis'
    os.makedirs(out_dir, exist_ok=True)

    cfg = Config.fromfile(cfg_file)
    cfg.work_dir = './temp_vis_workdir'
    cfg.load_from = ckpt_file
    cfg.launcher = 'none'

    # build dataset and model
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    data_loader = mmengine.runner.build_dataloader(cfg.test_dataloader, default_args=dict(dataset=dataset))
    model = MODELS.build(cfg.model)
    checkpoint = mmengine.runner.load_checkpoint(model, ckpt_file, map_location='cpu')
    model.eval()

    # take first N samples
    num_samples = 5
    for i, data in enumerate(data_loader):
        if i >= num_samples:
            break

        with torch.no_grad():
            result = model.test_step(data)[0]

        # result is a Det3DDataSample
        assert isinstance(result, Det3DDataSample)
        bev_img = result.metainfo.get('img', None)
        if bev_img is None:
            # fall back: visualize scores as text
            print(f'Sample {i}: got prediction with {len(result.pred_instances_3d.bboxes_3d)} boxes')
            continue

        plt.figure(figsize=(6, 6))
        plt.imshow(bev_img[..., ::-1])
        plt.axis('off')
        plt.title(f'Sample {i}')
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'sample_{i:03d}.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print('Saved', out_path)

if __name__ == '__main__':
    import torch  # placed here to avoid circular imports if any
    main()
