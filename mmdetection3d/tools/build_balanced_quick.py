import mmengine
import random
from collections import defaultdict

random.seed(42)
infos = mmengine.load('data/nuscenes/nuscenes_infos_train.pkl')
data_list = infos['data_list'] if isinstance(infos, dict) else infos

print(f'Loaded {len(data_list)} samples')

class_to_indices = defaultdict(list)
for idx, sample in enumerate(data_list):
    gt_names = sample.get('gt_names', sample.get('ann_infos', {}).get('gt_names', []))
    for cls_name in set(gt_names):
        class_to_indices[cls_name].append(idx)

print('\nClass distribution:')
for cls_name in sorted(class_to_indices.keys()):
    print(f'  {cls_name}: {len(class_to_indices[cls_name])} samples')

max_per_class = 200
selected = set()
for cls_name, idxs in class_to_indices.items():
    chosen = random.sample(idxs, min(len(idxs), max_per_class))
    selected.update(chosen)
    print(f'{cls_name}: selected {len(chosen)}/{len(idxs)}')

selected = sorted(list(selected))
print(f'\nTotal balanced samples: {len(selected)}')

balanced_data = [data_list[i] for i in selected]
out = dict(infos) if isinstance(infos, dict) else {}
out['data_list'] = balanced_data
out['metainfo'] = infos.get('metainfo', {})

mmengine.dump(out, 'data/nuscenes/nuscenes_infos_train_balanced.pkl')
print('Saved!')
