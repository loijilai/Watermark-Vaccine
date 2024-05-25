import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os

from utils import image_save
from model.WDNet import generator, WDNet_WV
from model.BVMR import load_globals, init_nets, BVMR_WV
from model.SplitNet import models, SplitNet_WV
from model.Ensemble_WV import Ensemble, Ensemble_WV
from tqdm import tqdm

def main():

    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)
    # This reshaping allows std and mean to be used in normalization across three channels.
    mu = torch.tensor(mean).view(3, 1, 1).cuda()
    std = torch.tensor(std).view(3, 1, 1).cuda()

    upper_limit = ((1 - mu) / std)
    lower_limit = ((0 - mu) / std)

    epsilon = (args.epsilon / 255.) / std
    start_epsilon = (args.start_epsilon / 255.) / std
    step_alpha = (args.step_alpha / 255.) / std

    np.random.seed(args.seed)

    if args.model == 'WDNet':
        model = generator(3, 3)
        model.eval()
        optimizer = WDNet_WV(model,args,epsilon,start_epsilon,step_alpha,upper_limit,lower_limit)
        model.load_state_dict(torch.load('./checkpoints/WDNet_G.pkl', map_location='cpu'))
        model.cuda()
    elif args.model == 'BVMR':
        _opt = load_globals('./checkpoints/BVMR', {}, override=False)
        _device = torch.device('cuda')
        model = init_nets(_opt, './checkpoints/BVMR', _device, 11).eval()
        optimizer = BVMR_WV(model,args,epsilon,start_epsilon,step_alpha,upper_limit,lower_limit)
    elif args.model == 'SplitNet':
        model = models.__dict__['vvv4n']().cuda()
        model.load_state_dict(torch.load('./checkpoints/SplitNet/splitnet.pth.tar')['state_dict'])
        model.eval()
        optimizer = SplitNet_WV(model, args, epsilon, start_epsilon, step_alpha, upper_limit, lower_limit)
    elif args.model == 'Ensemble':
        model = Ensemble()
        optimizer = Ensemble_WV(model, args, epsilon, start_epsilon, step_alpha, upper_limit, lower_limit)


    transform_norm = transforms.Compose([transforms.ToTensor()])



    for t in tqdm(range(args.num_img)):
        t += 1
        logo_index = np.random.randint(1, 161)
        imageJ_path = os.path.join(args.image_path ,f'{t}.jpg')
        logo_path = os.path.join(args.logo_path ,f'{logo_index}.png')

        img_J = Image.open(imageJ_path)
        img_source = transform_norm(img_J)
        img_source = torch.unsqueeze(img_source.cuda(), 0)


        # 初始化扰动
        seed = np.random.randint(0, 1000)

        # Clean output
        # wm, clean_pred, clean_mask = optimizer.Clean(img_source,logo_path,seed)
        # image_save(wm, os.path.join(args.save_path, f'{args.model}', 'clean', f'{t}-clean.png'))

        # Random Noise
        # random, random_pred, random_mask = optimizer.RN(img_source,logo_path,seed)
        # image_save(random, os.path.join(args.save_path, f'{args.model}', 'rn', f'{t}-random.png'))

        # Disrupting Watermark Vaccine
        if args.model == 'Ensemble':
            adv1 = optimizer.DWV(img_source,logo_path,seed,attack_type=args.attack_type)
        else:
            adv1, adv1_pred, adv1_mask = optimizer.DWV(img_source,logo_path,seed,attack_type=args.attack_type)
        image_save(adv1, os.path.join(args.save_path, f'{args.model}', 'dwv', f'{args.attack_type}', f'{t}-dwv.png'))
        
        # Inerasable Wateramark Vaccine
        if args.model == 'Ensemble':
            adv2 = optimizer.IWV(img_source,logo_path,seed,attack_type=args.attack_type)
        else:
            adv2, adv2_pred, adv2_mask = optimizer.IWV(img_source,logo_path,seed,attack_type=args.attack_type)
        image_save(adv2, os.path.join(args.save_path, f'{args.model}', 'iwv', f'{args.attack_type}', f'{t}-iwv.png'))



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, help='Watermark Removal Network (WDNet,SplitNet,BVMR,Ensemble)', default='WDNet')
    argparser.add_argument('--image_path', type=str,default='./dataset/CLWD/test/Watermark_free_image/')
    argparser.add_argument('--logo_path', type=str,default='./dataset/CLWD/watermark_logo/train_color')
    argparser.add_argument('--epsilon', type=int, help='the bound of perturbation', default=8)
    argparser.add_argument('--start_epsilon', type=int, help='the bound of random noise', default=8)
    argparser.add_argument('--step_alpha', type=int, help='step size', default=2)
    argparser.add_argument('--seed', type=int, help='random seed', default=160)
    argparser.add_argument('--num_img', type=int, help='imgsz', default=10)
    argparser.add_argument('--attack_iter', type=int, default=50)
    argparser.add_argument('--attack_type', type=str, help='Algorithm (Original, MIGFSM)', default='Original')
    argparser.add_argument('--save_path', type=str,default='./dataset/experiment/vaccine_wm')

    args = argparser.parse_args()

    main()
