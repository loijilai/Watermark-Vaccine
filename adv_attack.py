import torch
from torchvision import transforms
from PIL import Image
import argparse
import os

from utils import image_save
from model.WDNet import generator
from model.SplitNet import models
from model.BVMR import load_globals, init_nets, BVMR_WV

def main():
    # mean = (0.0, 0.0, 0.0)
    # std = (1.0, 1.0, 1.0)
    # # This reshaping allows std and mean to be used in normalization across three channels.
    # mu = torch.tensor(mean).view(3, 1, 1).cuda()
    # std = torch.tensor(std).view(3, 1, 1).cuda()

    # upper_limit = ((1 - mu) / std)
    # lower_limit = ((0 - mu) / std)

    # epsilon = (args.epsilon / 255.) / std
    # start_epsilon = (args.start_epsilon / 255.) / std
    # step_alpha = (args.step_alpha / 255.) / std

    # np.random.seed(123)
    original_model = args.input_path.split('/')[-2]
    secondary_model = args.model
    print(f'\n Noise by: {original_model} \n Attack on: {secondary_model}\n')
    if args.model == 'WDNet':
        model = generator(3, 3)
        model.load_state_dict(torch.load('./checkpoints/WDNet_G.pkl', map_location='cpu'))
        model.cuda().eval()
    elif args.model == 'SplitNet':
        model = models.__dict__['vvv4n']().cuda()
        model.load_state_dict(torch.load('./checkpoints/SplitNet/splitnet.pth.tar')['state_dict'])
        model.eval()
    elif args.model == 'BVMR':
        _opt = load_globals(nets_path='./checkpoints/BVMR', globals_dict={}, override=False)
        _device = torch.device('cuda')
        model = init_nets(opt=_opt, net_path='./checkpoints/BVMR', device=_device, tag=11).eval()
    
    print(f'Loading data from {args.input_path}...')
    transform_norm = transforms.Compose([transforms.ToTensor()])
    imageJ_names = os.listdir(args.input_path)
    for imageJ_name in imageJ_names:
        imageJ = Image.open(os.path.join(args.input_path, imageJ_name))
        image = transform_norm(imageJ).cuda()
        image = torch.unsqueeze(image, dim=0)
        if args.model == 'WDNet':
            model_output, _, _, _, _ = model(image)
        elif args.model == 'SplitNet':
            mid_output, mid_mask, _ = model(image)
            model_output = mid_output[0] * mid_mask + image * (1 - mid_mask)
        elif args.model == 'BVMR':
            img = image * 255
            mid_adv1 = (img) / 127.5 - 1
            mid_output = model(mid_adv1)
            mid_guess_images, mid_guess_mask = mid_output[0], mid_output[1]
            mid_expanded_guess_mask = mid_guess_mask.repeat(1, 3, 1, 1)
            mid_reconstructed_pixels = mid_guess_images * mid_expanded_guess_mask
            mid_reconstructed_images = mid_adv1 * (1 - mid_expanded_guess_mask) + mid_reconstructed_pixels
            model_output = mid_reconstructed_images / 2 + 0.5

        image_save(model_output, os.path.join(args.output_path, imageJ_name.split('.')[0]+f'-{original_model}-{secondary_model}.png'))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, help='Watermark Removal Network (WDNet,SplitNet,BVMR)', default='WDNet')
    argparser.add_argument('--input_path', type=str,default='./dataset/experiment/vaccine_wm/Ensemble/dwv')
    # argparser.add_argument('--output_path', type=str,default='./output/experiment/WDNet/iwv')
    argparser.add_argument('--output_path', type=str,default='./trash')
    args = argparser.parse_args()
    main()