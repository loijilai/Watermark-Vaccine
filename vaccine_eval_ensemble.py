import torch
from torchvision import transforms
from PIL import Image
import argparse
import os

from utils import image_save, create_image_grid, natural_sort_key, image_concat
from model.WDNet import generator
from model.SplitNet import models
from model.BVMR import load_globals, init_nets, BVMR_WV

def main():
    # Load target models
    WDNet_model = generator(3, 3)
    WDNet_model.load_state_dict(torch.load('./checkpoints/WDNet_G.pkl', map_location='cpu'))
    WDNet_model.cuda().eval()

    SplitNet_model = models.__dict__['vvv4n']().cuda()
    SplitNet_model.load_state_dict(torch.load('./checkpoints/SplitNet/splitnet.pth.tar')['state_dict'])
    SplitNet_model.eval()

    _opt = load_globals(nets_path='./checkpoints/BVMR', globals_dict={}, override=False)
    _device = torch.device('cuda')
    BVMR_model = init_nets(opt=_opt, net_path='./checkpoints/BVMR', device=_device, tag=11).eval()
    
    
    # Load source datasets
    dwv_path = os.path.join(args.input_path, 'Ensemble', 'dwv', args.attack_type)
    iwv_path = os.path.join(args.input_path, 'Ensemble', 'iwv', args.attack_type)

    transform_norm = transforms.Compose([transforms.ToTensor()])
    tensor_to_pil = transforms.ToPILImage()

    grids = [[] for _ in range(args.num_of_images)]
    image_names = sorted(os.listdir(dwv_path), key=natural_sort_key)[:args.num_of_images]
    for i, image_name in enumerate(image_names):
        imagePIL = Image.open(os.path.join(dwv_path, image_name))
        imageTensor = transform_norm(imagePIL).cuda()
        imageTensor = torch.unsqueeze(imageTensor, dim=0)

        # inference all three models
        model_output, _, _, _, _ = WDNet_model(imageTensor)
        pil_image = tensor_to_pil(model_output.squeeze(0))
        grids[i].append(pil_image)

        mid_output, mid_mask, _ = SplitNet_model(imageTensor)
        model_output = mid_output[0] * mid_mask + imageTensor * (1 - mid_mask)
        pil_image = tensor_to_pil(model_output.squeeze(0))
        grids[i].append(pil_image)

        img = imageTensor * 255
        mid_adv1 = (img) / 127.5 - 1
        mid_output = BVMR_model(mid_adv1)
        mid_guess_images, mid_guess_mask = mid_output[0], mid_output[1]
        mid_expanded_guess_mask = mid_guess_mask.repeat(1, 3, 1, 1)
        mid_reconstructed_pixels = mid_guess_images * mid_expanded_guess_mask
        mid_reconstructed_images = mid_adv1 * (1 - mid_expanded_guess_mask) + mid_reconstructed_pixels
        model_output = mid_reconstructed_images / 2 + 0.5
        pil_image = tensor_to_pil(model_output.squeeze(0))
        grids[i].append(pil_image)

    for i, pil_images in enumerate(grids):
        grid_image = image_concat(pil_images)
        grid_image.save(os.path.join(args.output_path, f'{i+1}-dwv-{args.attack_type}-ensemble.png'))
    

    grids = [[] for _ in range(args.num_of_images)]
    image_names = sorted(os.listdir(iwv_path), key=natural_sort_key)[:args.num_of_images]
    for i, image_name in enumerate(image_names):
        imagePIL = Image.open(os.path.join(iwv_path, image_name))
        imageTensor = transform_norm(imagePIL).cuda()
        imageTensor = torch.unsqueeze(imageTensor, dim=0)

        # inference all three models
        model_output, _, _, _, _ = WDNet_model(imageTensor)
        pil_image = tensor_to_pil(model_output.squeeze(0))
        grids[i].append(pil_image)

        mid_output, mid_mask, _ = SplitNet_model(imageTensor)
        model_output = mid_output[0] * mid_mask + imageTensor * (1 - mid_mask)
        pil_image = tensor_to_pil(model_output.squeeze(0))
        grids[i].append(pil_image)

        img = imageTensor * 255
        mid_adv1 = (img) / 127.5 - 1
        mid_output = BVMR_model(mid_adv1)
        mid_guess_images, mid_guess_mask = mid_output[0], mid_output[1]
        mid_expanded_guess_mask = mid_guess_mask.repeat(1, 3, 1, 1)
        mid_reconstructed_pixels = mid_guess_images * mid_expanded_guess_mask
        mid_reconstructed_images = mid_adv1 * (1 - mid_expanded_guess_mask) + mid_reconstructed_pixels
        model_output = mid_reconstructed_images / 2 + 0.5
        pil_image = tensor_to_pil(model_output.squeeze(0))
        grids[i].append(pil_image)

    for i, pil_images in enumerate(grids):
        grid_image = image_concat(pil_images)
        grid_image.save(os.path.join(args.output_path, f'{i+1}-iwv-{args.attack_type}-ensemble.png'))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path', type=str,default='./dataset/experiment/vaccine_wm')
    argparser.add_argument('--output_path', type=str,default='./output/experiment/Ensemble')
    argparser.add_argument('--num_of_images', type=int,default=5)
    argparser.add_argument('--attack_type', type=str, help='(Original, MIFGSM)', default='Original')
    argparser.add_argument('--show', action='store_true', help='output or not')
    # argparser.add_argument('--output_path', type=str,default='./trash')
    args = argparser.parse_args()
    main()