import os
import argparse
import math
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import torch
import cv2
import model.model as module_arch
from data_loader.data_loader import CustomDataLoader
from utils.util import denormalize
from torch import Tensor
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms

def main(blurred_image, deblurred_dir, resume):
    # load checkpoint
    checkpoint = torch.load(resume)
    config = checkpoint['config']

    # setup data_loader instances
    #data_loader = CustomDataLoader(data_dir=blurred_dir)

    # build model architecture
    generator_class = getattr(module_arch, config['generator']['type'])
    generator = generator_class(**config['generator']['args'])

    # prepare model for deblurring
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    if config['n_gpu'] > 1:
        generator = torch.nn.DataParallel(generator)

    generator.load_state_dict(checkpoint['generator'])

    generator.eval()

    # start to deblur
    with torch.no_grad():
        blurred = Image.open(blurred_image).convert('RGB')
        h = blurred.size[1]
        w = blurred.size[0]
        new_h = h - h % 4 + 4 if h % 4 != 0 else h
        new_w = w - w % 4 + 4 if w % 4 != 0 else w
        blurred = transforms.Resize([new_h, new_w], Image.BICUBIC)(blurred)
        transform = transforms.Compose([
            transforms.ToTensor(),  # convert to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])
        blurred = transform(blurred)
        blurred.unsqueeze_(0)
        print(blurred.shape)
        blurred = blurred.to(device)
        deblurred = generator(blurred)
        deblurred_img = to_pil_image(denormalize(deblurred).squeeze().cpu())

        deblurred_img.save("./deblurred.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deblur your own image!')

    parser.add_argument('-b', '--blurred', required=True, type=str, help='image_path')
    parser.add_argument('-d', '--deblurred', required=True, type=str, help='dir to save deblurred images')
    parser.add_argument('-r', '--resume', required=True, type=str, help='path to latest checkpoint')
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(args.blurred, args.deblurred, args.resume)
