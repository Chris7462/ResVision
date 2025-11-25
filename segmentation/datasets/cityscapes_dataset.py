"""
Cityscapes Dataset for PyTorch
Semantic segmentation with 19 classes (official 34 -> 19 mapping)
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class CityscapesDataset(Dataset):
    """
    Cityscapes Dataset for Semantic Segmentation

    Args:
        split_file: Path to train.txt, val.txt, or test.txt
        leftimg_dir: Directory containing leftImg8bit images (e.g., leftImg8bit/train)
        gtfine_dir: Directory containing gtFine label images (e.g., gtFine/train)
        dataset_info_path: Path to dataset_info.json
        transform: Albumentations transform pipeline
    """

    def __init__(self, split_file, leftimg_dir, gtfine_dir, dataset_info_path, transform=None):
        self.leftimg_dir = leftimg_dir
        self.gtfine_dir = gtfine_dir
        self.transform = transform

        # Load file list (format: city/basename)
        with open(split_file, 'r') as f:
            self.file_list = [line.strip() for line in f if line.strip()]

        # Load dataset info
        with open(dataset_info_path, 'r') as f:
            info = json.load(f)

        self.num_classes = info['num_classes']
        self.class_names = info['class_names']
        self.ignore_index = info['ignore_index']
        self.label_id_to_train_id = {int(k): v for k, v in info['label_id_to_train_id'].items()}

        print(f"Loaded {len(self.file_list)} images for this split")

    def convert_label_id_to_train_id(self, label_id_mask):
        """Convert labelId mask to trainId mask"""
        train_id_mask = np.full(label_id_mask.shape, self.ignore_index, dtype=np.int64)

        for label_id, train_id in self.label_id_to_train_id.items():
            train_id_mask[label_id_mask == label_id] = train_id

        return train_id_mask

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Get city/basename (e.g., 'aachen/aachen_000000_000019')
        file_path = self.file_list[idx]
        city, basename = file_path.split('/')

        # Load raw image
        img_filename = f'{basename}_leftImg8bit.png'
        img_path = os.path.join(self.leftimg_dir, city, img_filename)
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load label image (labelIds)
        label_filename = f'{basename}_gtFine_labelIds.png'
        label_path = os.path.join(self.gtfine_dir, city, label_filename)
        label_id_mask = np.array(Image.open(label_path))

        # Convert labelId to trainId
        target = self.convert_label_id_to_train_id(label_id_mask)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=target)
            image = transformed['image']
            target = transformed['mask']

        # Ensure target is Long type for PyTorch
        target = target.long()

        return {
            'image': image,
            'target': target,
        }

    def __repr__(self):
        return (
            f"CityscapesDataset(\n"
            f"  Number of samples: {len(self.file_list)}\n"
            f"  Number of classes: {self.num_classes}\n"
            f"  Ignore index: {self.ignore_index}\n"
            f")"
        )


if __name__ == '__main__':
    """Test the dataset"""
    import argparse
    from torch.utils.data import DataLoader
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    parser = argparse.ArgumentParser(description='Test CityscapesDataset')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory of Cityscapes dataset')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split (default: train)')
    args = parser.parse_args()

    # ImageNet normalization (for pretrained ResNet101)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Simple transform for testing (resize to 1024×512)
    transform = A.Compose([
        A.Resize(height=512, width=1024),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    # Create dataset
    dataset = CityscapesDataset(
        split_file=os.path.join(args.data_root, 'splits', f'{args.split}.txt'),
        leftimg_dir=os.path.join(args.data_root, 'leftImg8bit', args.split),
        gtfine_dir=os.path.join(args.data_root, 'gtFine', args.split),
        dataset_info_path=os.path.join(args.data_root, 'splits', 'dataset_info.json'),
        transform=transform
    )

    print(f"\n{dataset}")

    # Test loading one sample
    if len(dataset) > 0:
        print("\nLoading first sample...")
        sample = dataset[0]
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Target shape: {sample['target'].shape}")
        print(f"  Unique target values: {torch.unique(sample['target']).numpy()}")

        # Test dataloader
        print("\nTesting DataLoader...")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Images shape: {batch['image'].shape}")
            print(f"  Targets shape: {batch['target'].shape}")
            break

        print("\nCityscapesDataset test passed!")
