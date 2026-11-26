import os
from enum import Enum
from pathlib import Path

import PIL
import torch
from torchvision import transforms

_CLASSNAMES_AD2 = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test_public"
    EVAL_ALL = "eval_all"  # Gộp validation + test_public phục vụ đánh giá pilot


def extract_lighting(stem: str) -> str:
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit() and parts[1]:
        return parts[1]
    return "chua_xac_dinh"


class MVTecAD2Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset cho MVTec AD 2 tương thích hoàn toàn với pipeline của patchcore-inspection.
    Hỗ trợ cấu trúc thư mục MVTec AD 2:
      <source>/<classname>/train/good/*.png
      <source>/<classname>/validation/good/*.png
      <source>/<classname>/test_public/good/*.png
      <source>/<classname>/test_public/bad/*.png
      <source>/<classname>/test_public/ground_truth/bad/*_mask.png
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        square_resize=True,
        **kwargs,
    ):
        super().__init__()
        self.source = str(source)
        self.split = split if isinstance(split, DatasetSplit) else DatasetSplit(split)
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES_AD2
        self.resize = resize
        self.imagesize_val = imagesize
        self.square_resize = square_resize

        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD

        resize_target = (resize, resize) if square_resize else resize

        self.transform_img = transforms.Compose([
            transforms.Resize(resize_target, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(resize_target, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ])

        self.imagesize = (3, imagesize, imagesize)
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path, split_name, image_id, lighting = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if mask_path is not None and os.path.exists(mask_path):
            mask = PIL.Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros([1, *image.size()[1:]], dtype=torch.float32)

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": image_id,
            "image_id": image_id,
            "split": split_name,
            "lighting": lighting,
            "image_path": image_path,
            "mask_path": mask_path if mask_path is not None else "chua_xac_dinh",
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        data_to_iterate = []

        if self.split == DatasetSplit.TRAIN:
            sub_splits = [("train", ["good"])]
        elif self.split == DatasetSplit.VAL:
            sub_splits = [("validation", ["good"])]
        elif self.split == DatasetSplit.TEST:
            sub_splits = [("test_public", ["good", "bad"])]
        elif self.split == DatasetSplit.EVAL_ALL:
            sub_splits = [("validation", ["good"]), ("test_public", ["good", "bad"])]
        else:
            raise ValueError(f"Unsupported split: {self.split}")

        for classname in self.classnames_to_use:
            imgpaths_per_class[classname] = {}
            cat_root = Path(self.source) / classname

            for split_name, defect_types in sub_splits:
                for anomaly in defect_types:
                    anomaly_dir = cat_root / split_name / anomaly
                    if not anomaly_dir.exists():
                        continue
                    files = sorted([p for p in anomaly_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
                    key = f"{split_name}/{anomaly}"
                    imgpaths_per_class[classname][key] = [p.resolve().as_posix() for p in files]

                    for img_p in files:
                        stem = img_p.stem
                        image_id = f"{classname}/{split_name}/{anomaly}/{stem}"
                        lighting = extract_lighting(stem)
                        if split_name == "test_public" and anomaly != "good":
                            m_p = cat_root / "test_public" / "ground_truth" / anomaly / f"{stem}_mask{img_p.suffix}"
                            mask_str = m_p.resolve().as_posix() if m_p.exists() else None
                        else:
                            mask_str = None

                        data_to_iterate.append([
                            classname,
                            anomaly,
                            img_p.resolve().as_posix(),
                            mask_str,
                            split_name,
                            image_id,
                            lighting,
                        ])

        return imgpaths_per_class, data_to_iterate
