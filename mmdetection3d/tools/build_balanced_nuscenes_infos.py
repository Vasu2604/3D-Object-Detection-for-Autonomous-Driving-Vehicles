#!/usr/bin/env python
import argparse
import os
import random
from collections import defaultdict
import mmengine
from mmengine.logging import print_log

def parse_args():
    parser = argparse.ArgumentParser(
        description='Build a class-balanced subset of nuScenes train infos.')
    parser.add_argument('--src', default='data/nuscenes/nuscenes_infos_train.pkl')
    parser.add_argument('--dst', default='data/nuscenes/nuscenes_infos_train_balanced.pkl')
    parser.add_argument('--max-samples-per-class', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    
    print_log(f'Loading infos from {args.src}', logger='current')
    infos = mmengine.load(args.src)
    
    if isinstance(infos, dict) and 'data_list' in infos:
        data_list = infos['data_list']
    elif isinstance(infos, list):
        data_list = infos
    else:
        raise TypeError('Unsupported infos format')
    
    print_log(f'Loaded {len(data_list)} samples', logger='current')
    
    class_to_indices = defaultdict(list)
    for idx, sample in enumerate(data_list):
        gt_names = sample.get('gt_names', sample.get('ann_infos', {}).get('gt_names', []))
        for cls_name in set(gt_names):
            class_to_indices[cls_name].append(idx)
    
    print_log('Class distribution:', logger='current')
    for cls_name, idxs in sorted(class_to_indices.items()):
        print_log(f'  {cls_name}: {len(idxs)} samples', logger='current')
    
    selected_indices = set()
    max_k = args.max_samples_per_class
    
    for cls_name, idxs in class_to_indices.items():
        chosen = random.sample(idxs, min(len(idxs), max_k))
        selected_indices.update(chosen)
        print_log(f'{cls_name}: selected {len(chosen)}/{len(idxs)}', logger='current')
    
    selected_indices = sorted(list(selected_indices))
    print_log(f'Total unique samples: {len(selected_indices)}', logger='current')
    
    balanced_data_list = [data_list[i] for i in selected_indices]
    
    if isinstance(infos, dict) and 'data_list' in infos:
        balanced_infos = dict(infos)
        balanced_infos['data_list'] = balanced_data_list
    else:
        balanced_infos = balanced_data_list
    
    mmengine.dump(balanced_infos, args.dst)
    print_log(f'Saved to {args.dst}', logger='current')

if __name__ == '__main__':
    main()
