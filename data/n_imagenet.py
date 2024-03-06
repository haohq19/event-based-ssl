import os
import numpy as np
from torchvision.datasets import DatasetFolder

def load_event_npz(path): 
    data = np.load(path)
    return data['event_data']

class NImageNet(DatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            ):
        self.root = root
        self.train = train

        events_np_root = os.path.join(root, 'events_np')

        if train:
            _root = os.path.join(events_np_root, 'train')
        else:
            _root = os.path.join(events_np_root, 'validation')
        _loader = load_event_npz
        _transform = transform
        _target_transform = target_transform

        super(NImageNet, self).__init__(
            root=_root,
            loader=_loader,
            extensions=('.npz'),
            transform=_transform,
            target_transform=_target_transform
            )
        
if __name__ == "__main__":
    dataset = NImageNet(root='/home/haohq/event-based-ssl/datasets/NImageNet', train=True)
    x = dataset.__getitem__(0)
    print(x)