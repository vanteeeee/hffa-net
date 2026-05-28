import os

import argparse
import numpy as np
import json
import cv2
import torch

import torch.nn as nn

import torch.nn.functional as F

from matplotlib import pyplot as plt

from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from tqdm import tqdm

import torchvision.utils as utils

from util import AverageMeter

from datasets.loader import OhazeLoaderTest, OhazeLoaderT

from models import *

from metrics import ssim

from util.CR import ContrastLoss

from util.CR_res import ContrastLoss_res

from models.extractor import VitExtractor

from torchvision import transforms as T

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='DehazeFFA', type=str, help='model name')

parser.add_argument('--num_workers', default=16, type=int, help='number of workers')

parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')

parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')

parser.add_argument('--data_dir', default='./data/RESIDE-IN/', type=str, help='path to dataset')

parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')

parser.add_argument('--dataset', default='indoor', type=str, help='dataset name')

parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')

parser.add_argument('--gpu', default='5,7', type=str, help='GPUs used for training')

args = parser.parse_args()



os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

#CUDA_VISIBLE_DEVICES='6' python3.8 train_gl.py

dino_preprocess = T.Compose([T.Resize(224),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

vit_extractor = VitExtractor('dino_vitb8', "cuda")

def extract_feature(x):

  f = vit_extractor.get_feature_from_input(dino_preprocess(x))[11][:, 0, :]

  return f

   

def train(train_loader, network, criterion, optimizer, scaler):



	losses = AverageMeter()

	torch.cuda.empty_cache()	

	network.train()



	for batch in train_loader:

		source_img = batch['source'].cuda()

		target_img = batch['target'].cuda()
		 
		

   



		with torch.no_grad():

			ref_feature = extract_feature(target_img)

			ref_feature = torch.squeeze(ref_feature)
			#print(ref_feature.shape)

		with autocast(args.no_autocast):

			output,g = network(source_img)

			g = torch.squeeze(g)
			#print(g.shape)

      

			

			loss = criterion[0](output, target_img)+0.1*criterion[0](g, ref_feature)
			#print(loss)

			# ablation-base

# 			loss = criterion[0](output, target_img)





		losses.update(loss.item())



		optimizer.zero_grad()

		scaler.scale(loss).backward()

		scaler.step(optimizer)

		scaler.update()



	return losses.avg




def save_image(dehaze, image_name, category):
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)

    for ind in range(batch_num):
        utils.save_image(dehaze_images[ind], './{}_results/{}'.format(category, image_name[ind][:-3] + 'png'))
        
        
        
def cam(x):
    #print(x.shape)
    x=x[0]
    x = x - np.min(x)
    cam_img = x / np.max(x)
    x=np.sum(x, axis=0)
    
    
    cam_img = np.uint8(255 * cam_img[0])
    #print(cam_img)
    #cam_img=cam_img[0]
    #print(cam_img)
    #cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img 

def calc_ssim(im1, im2):
	im1 = im1[0].view(im1.shape[2],im1.shape[3],3).detach().cpu().numpy()
	im2 = im2[0].view(im2.shape[2],im2.shape[3],3).detach().cpu().numpy()

	im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
	im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
	ans = [compare_ssim(im1_y, im2_y)]
	return ans

def save_image(dehaze, image_name, category):
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)

    for ind in range(batch_num):
        utils.save_image(dehaze_images[ind], './{}/{}'.format(category, image_name))

save=True

def valid(val_loader, network):

	PSNR = AverageMeter()
	ssim_list  = []


	torch.cuda.empty_cache()



	network.eval()



	for batch in val_loader:

		source_img = batch['source'].cuda()

		target_img = batch['target'].cuda()



		with torch.no_grad():							# torch.no_grad() may cause warning

			output,_ = network(source_img)

			output=output.clamp_(-1, 1)		

		if save==True:
			print(batch['filename'])
			filename=batch['filename'][0].split('/')[-1]	
			print(filename)		
			save_image(output* 0.5 + 0.5,filename,'FFA_imp')
		ssim_tmp=ssim(output * 0.5 + 0.5, target_img * 0.5 + 0.5).item()
		ssim_list.append(ssim_tmp)
    		
    
		mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))

		psnr = 10 * torch.log10(1 / mse_loss).mean()

		PSNR.update(psnr.item(), source_img.size(0))



	return PSNR.avg,sum(ssim_list) / len(ssim_list)





if __name__ == '__main__':

	setting_filename = os.path.join('configs', args.exp, args.model+'.json')

	print(setting_filename	) 

	if not os.path.exists(setting_filename):

		setting_filename = os.path.join('configs', args.exp, 'default.json')

	with open(setting_filename, 'r') as f:

		setting = json.load(f)



    #   Start training model from checkpoint

	checkpoint=torch.load("./saved_models/indoor/DehazeFFAwo_CR_2.pth", map_location=torch.device('cpu'))

	#checkpoint=None

	#   Start training model from NULL

	#checkpoint=None

	network = eval(args.model.replace('-', '_'))(3,3)

	network = nn.DataParallel(network).cuda()

 

	ckp=torch.load('ots_train_ffa_3_19.pk',map_location='cuda')

 

	parameters=ckp['model']

	parameters_new={}

	for key_i in parameters.keys():

		parameters_new[key_i[7:]] = parameters[key_i] 

    

	del parameters

        #self.ffanet=nn.DataParallel(self.ffanet)

	network.module.ffanet.load_state_dict(parameters_new)

	if checkpoint is not  None:

		network.load_state_dict(checkpoint['state_dict'])



	criterion = []

	criterion.append(nn.L1Loss())

	criterion.append(ContrastLoss_res(ablation=False).cuda())



	if setting['optimizer'] == 'adam':

		optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])

	elif setting['optimizer'] == 'adamw':

		optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])

	else:

		raise Exception("ERROR: unsupported optimizer")





	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)

	scaler = GradScaler()



	if checkpoint is not None:

		optimizer.load_state_dict(checkpoint['optimizer'])

		scheduler.load_state_dict(checkpoint['lr_scheduler'])

		scaler.load_state_dict(checkpoint['scaler'])

		best_psnr = checkpoint['best_psnr']
		start_epoch=0

		#start_epoch = checkpoint['epoch'] + 1

	else:

		best_psnr = 0

		start_epoch = 0

	dataset_dir='/data/train/'
	train_dataset = OhazeLoaderT(dataset_dir,  'train', 
								setting['patch_size'],
							    setting['edge_decay'],
							    setting['only_h_flip'])
	train_loader = DataLoader(train_dataset,
                              batch_size=2,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
	data_dir='/data/test/'
	val_dataset = OhazeLoaderTest(data_dir, setting['valid_mode'], 
							  setting['patch_size'])
	val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=args.num_workers,
                            pin_memory=True)





	save_dir = os.path.join(args.save_dir, args.exp)

	os.makedirs(save_dir, exist_ok=True)



	# if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):

	print('==> Start training, current model name: ' + args.model)

	# print(network)



	# writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))



	train_ls, test_ls, idx = [], [], []
	avg_psnr, avg_ssim = valid(val_loader, network)
	print(avg_psnr, avg_ssim)
	avg_psnr=0 
	best_psnr=0


	for epoch in tqdm(range(start_epoch,setting['epochs'] + 1)):

		loss = train(train_loader, network, criterion, optimizer, scaler)

		

		#train_ls.append(loss)

		#idx.append(epoch)



		# writer.add_scalar('train_loss', loss, epoch)



		scheduler.step()





		if epoch % setting['eval_freq'] == 0:

			print(epoch)
			avg_psnr, avg_ssim = valid(val_loader, network)
			#print(avg_psnr, avg_ssim)


			# writer.add_scalar('valid_psnr', avg_psnr, epoch)



			if avg_psnr > best_psnr:

				best_psnr = avg_psnr

				print(avg_psnr)



				torch.save({'state_dict': network.state_dict(),

							'optimizer':optimizer.state_dict(),

							'lr_scheduler':scheduler.state_dict(),

							'scaler':scaler.state_dict(),

							'epoch':epoch,

							'best_psnr':best_psnr

							},

						   os.path.join(save_dir, args.model+'wo_CR_2.pth'))



			# writer.add_scalar('best_psnr', best_psnr, epoch)



