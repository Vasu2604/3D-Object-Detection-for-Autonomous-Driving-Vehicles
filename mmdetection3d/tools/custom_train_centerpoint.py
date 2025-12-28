#!/usr/bin/env python

import os
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmengine.config import Config
from mmengine.runner import set_random_seed
from mmengine.dist import get_rank

# IMPORTANT: import MMDet3D modules so that datasets & transforms register
import mmdet3d  # noqa: F401
import mmdet3d.datasets  # noqa: F401
import mmdet3d.datasets.transforms  # noqa: F401
import mmdet3d.models  # noqa: F401

from mmdet3d.registry import DATASETS, MODELS
from mmengine.dataset import default_collate




def parse_args():
    parser = argparse.ArgumentParser(
        description='Custom training script for CenterPoint on nuScenes-mini')
    parser.add_argument(
        '--config',
        default='configs/centerpoint/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py',
        help='Path to config file')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/custom_centerpoint_mini',
        help='Directory to save logs and checkpoints')
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (override config if given)')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed')
    args = parser.parse_args()
    return args


def build_dataloader_from_cfg(cfg_dataloader, dataset, shuffle=True) -> DataLoader:
    """Simple dataloader builder using cfg fields."""
    batch_size = cfg_dataloader.get('batch_size', 1)
    num_workers = cfg_dataloader.get('num_workers', 4)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=default_collate,
        pin_memory=True,
        drop_last=False)
    return loader


def move_to_device(data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Recursively move tensors in a nested dict/list/tuple to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    return data


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    if args.epochs is not None:
        max_epochs = args.epochs
    else:
        # Fallback: try train_cfg or default to 20
        max_epochs = getattr(cfg, 'max_epochs', 20)
        if hasattr(cfg, 'train_cfg') and isinstance(cfg.train_cfg, dict):
            max_epochs = cfg.train_cfg.get('max_epochs', max_epochs)

    # Prepare work_dir
    work_dir = Path(args.work_dir)
    ckpt_dir = work_dir / 'ckpts'
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = work_dir / 'train_log.json'

    # Save used config for reproducibility
    cfg.dump(str(work_dir / 'config_used.py'))

    # Set seed
    set_random_seed(args.seed, deterministic=False)
    rank = get_rank()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from mmengine.registry import TRANSFORMS
    print('Known transforms:', [k for k in TRANSFORMS.module_dict.keys()])

    # Build dataset & dataloader (train only for now)
    train_dataset = DATASETS.build(cfg.train_dataloader.dataset)
    train_loader = build_dataloader_from_cfg(cfg.train_dataloader, train_dataset, shuffle=True)

    # Build model
    model = MODELS.build(cfg.model)
    model.to(device)
    model.train()

    # Build optimizer from cfg.optim_wrapper.optimizer
    optim_cfg = cfg.optim_wrapper.optimizer
    optim_type = optim_cfg.get('type', 'AdamW')
    lr = optim_cfg.get('lr', 1e-3)
    weight_decay = optim_cfg.get('weight_decay', 0.01)

    if optim_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {optim_type} not implemented in this script.')

    # Simple scheduler: cosine over epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Load existing log if any
    if train_log_path.exists():
        with open(train_log_path, 'r') as f:
            train_log = json.load(f)
    else:
        train_log = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_iters = 0

        if rank == 0:
            pbar = tqdm(total=len(train_loader), desc=f'Epoch [{epoch}/{max_epochs}]', ncols=100)
        else:
            pbar = None

        for data_batch in train_loader:
            # MMDet3D models expect data dict with specific keys; we pass as-is
            data_batch = move_to_device(data_batch, device)

            # Some versions use model.train_step(); else use forward + loss
            if hasattr(model, 'train_step'):
                # train_step is usually used by Runner, but we can reuse it here
                optimizer.zero_grad()
                outputs = model.train_step(data_batch, optimizer=optimizer)
                # outputs is usually a dict with 'loss', 'log_vars', 'num_samples'
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            else:
                # Fallback: assume forward returns loss dict
                optimizer.zero_grad()
                out = model(**data_batch)
                if isinstance(out, dict) and 'loss' in out:
                    loss = out['loss']
                else:
                    # Or sum all loss terms if needed
                    loss = 0
                    for v in out.values():
                        if isinstance(v, torch.Tensor):
                            loss = loss + v

                loss.backward()
                optimizer.step()

            # If train_step already stepped optimizer, we only need loss value
            if isinstance(loss, torch.Tensor):
                loss_val = loss.item()
            else:
                loss_val = float(loss)

            epoch_loss += loss_val
            num_iters += 1

            if pbar is not None:
                pbar.set_postfix(loss=f'{loss_val:.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.2e}')
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        # Epoch average loss
        avg_loss = epoch_loss / max(1, num_iters)

        # Step scheduler
        scheduler.step()

        # Save checkpoint every epoch
        ckpt_path = ckpt_dir / f'epoch_{epoch}.pth'
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   ckpt_path)

        # Log metrics
        epoch_log = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'lr': optimizer.param_groups[0]['lr']
        }
        train_log.append(epoch_log)

        with open(train_log_path, 'w') as f:
            json.dump(train_log, f, indent=2)

        if rank == 0:
            print(f'End of epoch {epoch}: avg_loss={avg_loss:.4f}, ckpt={ckpt_path}')

        # Optional: here you can hook in a call to tools/test.py to evaluate
        # and then parse metrics_summary.json to add val_mAP / val_NDS to epoch_log.

    if rank == 0:
        print('Training completed.')


if __name__ == '__main__':
    main()
