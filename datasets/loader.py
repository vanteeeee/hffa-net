import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from util import hwc_to_chw, read_img
from random import randrange 
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	# simple re-weight for the edge
	if random.random() < Hc / H * edge_decay:
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	# horizontal flip
	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)

	if not only_h_flip:
		# bad data augmentations for outdoor
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs


def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	Hs = (H - Hc) // 2
	Ws = (W - Wc) // 2
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	return imgs


class PairLoader(Dataset):
	def __init__(self, data_dir, mode, size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test', 'crop']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		self.root_dir = data_dir
		self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'hazy')))
		self.img_num = len(self.img_names)
		self.clean_names=[]
   
   
		self.haze_dir = os.path.join(self.root_dir, 'hazy')
		
		self.clear_dir = os.path.join(self.root_dir, 'clear')
   
   
		for img_name in self.img_names:
				name=img_name.split('_')[0]	+'.png'		
				self.clean_names.append(name)   
				   

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		source_img = read_img(os.path.join(self.root_dir, 'hazy', img_name)) * 2 - 1
		img_name = self.clean_names[idx]
		target_img = read_img(os.path.join(self.root_dir, 'clear', img_name)) * 2 - 1
		
		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

		if self.mode == 'valid':
			[source_img, target_img] = align([source_img, target_img], self.size)
		if self.mode == 'crop':
			[source_img, target_img] = align([source_img, target_img], self.size)
		#print(source_img)   
		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}
   
class OhazeLoaderTest(Dataset):
	def __init__(self, data_dir, mode='test', size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test', 'crop']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		haze_imgsname=[]
		clear_imgsname=[]
		self.haze_dir = os.path.join(data_dir, 'hazy')
		self.clear_dir=os.path.join(data_dir, 'GT')
		for imgname in os.listdir(self.haze_dir):
			haze_imgsname.append(os.path.join(self.haze_dir,imgname))
			clearname=imgname.replace('_hazy.jpg','_GT.jpg')
			clear_imgsname.append(os.path.join(self.clear_dir,clearname))
   
   
    
		self.haze_imgsname= haze_imgsname
		self.clear_imgsname= clear_imgsname
		self.img_num = len(self.haze_imgsname)
   
   
		
				   

	def __len__(self):
		return int(self.img_num)

	def __getitem__(self, index):
		haze=self.haze_imgsname[index]
		clear=self.clear_imgsname[index]
   
		haze_img=cv2.imread(haze)
        #print(os.path.join(self.clear_dir,clear_name))
		gt_img=cv2.imread(clear)
		haze_img=cv2.cvtColor(haze_img,cv2.COLOR_BGR2RGB)
        
        
		

		haze_img=haze_img/255.0
		gt_img=cv2.cvtColor(gt_img,cv2.COLOR_BGR2RGB)/255.0
        
       
		gt_crop_img=gt_img.astype(np.float32)
		haze_crop_img=haze_img.astype(np.float32)
        
        # --- Transform to tensor --- #
		transform_all = Compose([ToTensor(), Resize((512,512)), Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])


		img_lq = transform_all(haze_crop_img)
		img_gt = transform_all(gt_crop_img)

		
		#print(source_img)   
		return {'source': img_lq, 'target': img_gt, 'filename': haze}
    
    
       
   
class OhazeLoaderT(Dataset):
	def __init__(self, data_dir, mode='train', size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test', 'crop']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		haze_imgsname=[]
		clear_imgsname=[]
		self.haze_dir = os.path.join(data_dir, 'hazy')
		self.clear_dir=os.path.join(data_dir, 'GT')
		for imgname in os.listdir(self.haze_dir):
			haze_imgsname.append(os.path.join(self.haze_dir,imgname))
			clearname=imgname.replace('_hazy.jpg','_GT.jpg')
			clear_imgsname.append(os.path.join(self.clear_dir,clearname))
   
   
    
		self.haze_imgsname= haze_imgsname
		self.clear_imgsname= clear_imgsname
		self.img_num = len(self.haze_imgsname)
   
   
   
		
				   

	def __len__(self):
		return int(self.img_num)

	def __getitem__(self, index):
		haze=self.haze_imgsname[index]
		clear=self.clear_imgsname[index]
   
		haze_img=cv2.imread(haze)
        #print(os.path.join(self.clear_dir,clear_name))
		gt_img=cv2.imread(clear)
		haze_img=cv2.cvtColor(haze_img,cv2.COLOR_BGR2RGB)
        
		[width,height] = haze_img.shape[:2]
		remaxw=int(width*1.0/512)
		remaxh=int(height*1.0/512)
        
		minre=min(remaxw*11,remaxh*11)
		minre=randrange(10, minre)/10
		crop_width=int(width/minre)
		crop_height=int(height/minre)    
		x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
		haze_img = haze_img[x:x + crop_width, y:y + crop_height]
		gt_img = gt_img[x:x + crop_width, y:y + crop_height]
		haze_img = cv2.resize(haze_img, (512, 512), interpolation=cv2.INTER_AREA)
		gt_img = cv2.resize(gt_img, (512, 512), interpolation=cv2.INTER_AREA)
		crop_width=256
		crop_height=256
		x, y = randrange(0, 512 - crop_width + 1), randrange(0, 512 - crop_height + 1)   
		haze_img = haze_img[x:x + crop_width, y:y + crop_height]
		gt_img = gt_img[x:x + crop_width, y:y + crop_height]
            
		if randrange(0,10)<3:
            
			haze_img=cv2.flip(haze_img, 1)
			gt_img=cv2.flip(gt_img, 1)

		if randrange(0,10)<3:
            
			haze_img=cv2.flip(haze_img, 0)
			gt_img=cv2.flip(gt_img, 0)
		if randrange(0,10)<3:
            
			haze_img=cv2.flip(haze_img, -1)
			gt_img=cv2.flip(gt_img, -1) 

		haze_img=haze_img/255.0
		gt_img=cv2.cvtColor(gt_img,cv2.COLOR_BGR2RGB)/255.0
        
       
		gt_crop_img=gt_img.astype(np.float32)
		haze_crop_img=haze_img.astype(np.float32)
        
        # --- Transform to tensor --- #
		transform_all = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])


		img_lq = transform_all(haze_crop_img)
		img_gt = transform_all(gt_crop_img)

		
		#print(source_img)   
		return {'source': img_lq, 'target': img_gt, 'filename': haze}


class SingleLoader(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(self.root_dir))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

		return {'img': hwc_to_chw(img), 'filename': img_name}
