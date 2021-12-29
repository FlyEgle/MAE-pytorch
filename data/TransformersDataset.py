"""
-*- coding: utf-8 -*-
@datetime: 2021-06-28
@author  : jiangmingchao@joyy.sg
@describe: Imagenet dataset
"""
from math import e
import torch
import random
import numpy as np
import urllib.request as urt

from PIL import Image
from io import BytesIO
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms as imagenet_transforms
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.transforms import _pil_interp
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


def build_transform(is_train, args):
    resize_im = args.crop_size > 32
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        # horizion flip, color_jitter, randaug, bicubic interpolation, randomearse
        transform = create_transform(
            input_size=args.crop_size,
            is_training=is_train,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = imagenet_transforms.RandomCrop(args.crop_size, padding=4)
        # return compose transform
        return transform


class ImageDataset(Dataset):
    def __init__(self,
                 image_file,
                 train_phase,
                 crop_size,
                 shuffle,
                 args,
                 ) -> None:
        super(ImageDataset, self).__init__()
        
        self.image_file = image_file
        self.image_list = [x.strip() for x in open(self.image_file).readlines()]
        self.length = [x for x in range(len(self.image_list))]
        self.train_phase = train_phase
        self.crop_size = crop_size
        self.shuffle = shuffle
        self.args = args
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if self.shuffle and self.train_phase:
            for _ in range(10):
                random.shuffle(self.image_list)

        # train
        if self.train_phase:
            self.data_aug = build_transform(
                is_train = self.train_phase,
                args = self.args
            )
            print(self.data_aug)

        # test
        else:
            self.data_aug = imagenet_transforms.Compose([
                imagenet_transforms.Resize(int(256 / 224 * self.crop_size)),
                imagenet_transforms.CenterCrop(
                    (self.crop_size, self.crop_size)),
                imagenet_transforms.ToTensor(),
                imagenet_transforms.Normalize(
                    mean=self.mean,
                    std=self.std
                )
            ])

    def _decode_image(self, image_path):
        if "http" in image_path:
            image = Image.open(BytesIO(urt.urlopen(image_path).read()))
        else:
            image = Image.open(image_path)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def __getitem__(self, index):
        for _ in range(10):
            try:
                line = self.image_list[index]
                image_path, image_label = line.split(',')[0], line.split(',')[1]
                image = self._decode_image(image_path)
                image = self.data_aug(image)
                label = torch.from_numpy(np.array(int(image_label))).long()
                return image, label, image_path
            except Exception as e:
                index = random.choice(self.length)
                print(f"The exception is {e}, image path is {image_path}!!!")

    def __len__(self):
        return len(self.image_list)


class ImageDatasetTNF(Dataset):
    def __init__(self,
                 image_file,
                 train_phase,
                 crop_size,
                 shuffle=True,
                 interpolation='random',
                 auto_augment="rand",
                 color_prob=0.4,
                 hflip_prob=0.5,

                 ) -> None:
        super(ImageDatasetTNF, self).__init__()
        
        self.image_file = image_file
        self.image_list = [x.strip() for x in open(self.image_file).readlines()]
        self.length = [x for x in range(len(self.image_list))]
        self.train_phase = train_phase
        self.crop_size = crop_size
        self.shuffle = shuffle
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.hflip_prob = hflip_prob

        if self.shuffle and self.train_phase:
            for _ in range(10):
                random.shuffle(self.image_list)

        self.colorjitter_prob = None if color_prob ==0.0 else (color_prob, )*3
        self.auto_augment = auto_augment
        self.interpolation = interpolation

        # train
        if self.train_phase:
            basic_tf = [
                imagenet_transforms.RandomResizedCrop(
                    (self.crop_size, self.crop_size), interpolation=3),
                imagenet_transforms.RandomHorizontalFlip(self.hflip_prob),
            ]

            auto_tf = []
            if self.auto_augment:
                assert isinstance(auto_augment, str)
                if isinstance(self.crop_size, (tuple, list)):
                    img_size_min = min(self.crop_size)
                else:
                    img_size_min = self.crop_size

                aa_params = dict(
                    translate_dict=int(img_size_min * 0.45),
                    img_mean=tuple([min(255, round(255 * x))
                                    for x in self.mean])
                )
                if self.interpolation and self.interpolation != "random":
                    aa_params['interpolation'] = _pil_interp(
                        self.interpolation)
                # rand aug
                if auto_augment.startswith('rand'):
                    auto_tf += [rand_augment_transform(
                        auto_augment, aa_params)]
                # augmix
                elif auto_augment.startswith('augmix'):
                    aa_params['translate_pct'] = 0.3
                    auto_tf += [augment_and_mix_transform(
                        auto_augment, aa_params)]
                # auto aug
                else:
                    auto_tf += [auto_augment_transform(
                        auto_augment, aa_params)]

            if self.colorjitter_prob is not None:
                auto_tf += [
                    imagenet_transforms.ColorJitter(*self.colorjitter_prob)
                ]

            final_tf = [
                imagenet_transforms.ToTensor(),
                imagenet_transforms.Normalize(
                    mean=self.mean,
                    std=self.std
                )
            ]
            self.data_aug = imagenet_transforms.Compose(
                basic_tf + auto_tf + final_tf
            )

            print(self.data_aug)

        # test
        else:
            self.data_aug = imagenet_transforms.Compose([
                imagenet_transforms.Resize(int(256 / 224 * self.crop_size)),
                imagenet_transforms.CenterCrop(
                    (self.crop_size, self.crop_size)),
                imagenet_transforms.ToTensor(),
                imagenet_transforms.Normalize(
                    mean=self.mean,
                    std=self.std
                )
            ])

    def _decode_image(self, image_path):
        if "http" in image_path:
            image = Image.open(BytesIO(urt.urlopen(image_path).read()))
        else:
            image = Image.open(image_path)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def __getitem__(self, index):
        for _ in range(10):
            try:
                line = self.image_list[index]
                image_path, image_label = line.split(',')[0], line.split(',')[1]
                image = self._decode_image(image_path)
                image = self.data_aug(image)
                label = torch.from_numpy(np.array(int(image_label))).long()
                return image, label, image_path
            except Exception as e:
                index = random.choice(self.length)
                print(f"The exception is {e}, image path is {image_path}!!!")

    def __len__(self):
        return len(self.image_list)



# val
class ImageDatasetTest(Dataset):
    def __init__(self,
                 image_file,
                 train_phase,
                 input_size,
                 crop_size,
                 shuffle=True,
                 mode="cnn"
                 ) -> None:
        super(ImageDatasetTest, self).__init__()
        self.image_file = image_file
        self.image_list = [x.strip()
                           for x in open(self.image_file).readlines()]
        self.length = [x for x in range(len(self.image_list))]
        self.train_phase = train_phase
        self.input_size = input_size
        self.crop_size = crop_size
        self.shuffle = shuffle
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.mode = mode
        if self.shuffle and self.train_phase:
            for _ in range(10):
                random.shuffle(self.image_list)

        if self.mode == "cnn":
            self.data_aug = imagenet_transforms.Compose(
                [
                    imagenet_transforms.Resize(int(256 / 224 * self.crop_size)),
                    imagenet_transforms.CenterCrop(self.crop_size),
                    imagenet_transforms.ToTensor(),
                    imagenet_transforms.Normalize(
                        mean=self.mean,
                        std=self.std
                    )
                ]
            )
        elif self.mode == "transformers":
            self.data_aug = imagenet_transforms.Compose(
                [
                    imagenet_transforms.Resize(
                        (self.crop_size, self.crop_size)),
                    imagenet_transforms.ToTensor(),
                    imagenet_transforms.Normalize(
                        mean=self.mean,
                        std=self.std
                    )
                ]
            )

    def _decode_image(self, image_path):
        if "http" in image_path:
            image = Image.open(BytesIO(urt.urlopen(image_path).read()))
        else:
            image = Image.open(image_path)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def __getitem__(self, index):
        for _ in range(10):
            try:
                line = self.image_list[index]
                if len(line.split(',')) >= 2:
                    image_path, image_label = line.split(
                        ',')[0], line.split(',')[1]
                    label = torch.from_numpy(np.array(int(image_label))).long()
                else:
                    image_path = line
                    label = torch.from_numpy(np.array(0)).long()

                image = self._decode_image(image_path)
                image = self.data_aug(image)

                return image, label, image_path

            except Exception as e:
                index = random.choice(self.length)
                print(f"The exception is {e}, image path is {image_path}!!!")

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    train_file = "/data/jiangmingchao/data/dataset/imagenet/val_oss_imagenet_128w.txt"
    train_dataset = ImageDataset(
        image_file=train_file,
        train_phase=True,
        input_size=224,
        crop_size=224,
        shuffle=True,
        interpolation='bilinear',
        auto_augment="rand"
    )
    print(train_dataset)
    print(len(train_dataset))
    for idx, data in enumerate(train_dataset):
        print(f"{idx}", data[0].shape, data[1])
        break