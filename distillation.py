import os
import torch 
import numpy as np
from networks import get_network
from stupid_diffaug import ParamDiffAug, DiffAugment

class Distillation:
    def __init__(self, model, init_synth, img_loader):
        self.model = model 
        self.init_synth = init_synth
        self.img_loader = img_loader

    def synthesize(self, synth_init, num_iters, ipc, lr):
        optimizer = None 
        img_synth = synth_init
        for iter in range(num_iters):
            for batch_idx, (imgs, labels) in enumerate(self.img_loader):
                print("Do training step")
        pass
            
class DM:
    def __init__(self, img_synth, label_synth, dataset, args):
        self.img = img_synth
        self.label = label_synth

        m = len(dataset['trainset'])
        self.indices_class = [[] for c in range(dataset['num_classes'])]
        for i in range(m):
            _, lab = dataset['trainset'][i]          
            self.indices_class[lab].append(i)

        self.optimizer = torch.optim.Adam([self.img], lr=args.lr_img, betas=(0.5, 0.999)) 
        self.optimizer.zero_grad()

        self.dsa_param = ParamDiffAug()
        self.dsa_strategy = 'color_crop_cutout_flip_scale_rotate'

        self.seed = int.from_bytes(os.urandom(4), 'little') 
        self.net = get_network(args.synth_model, dataset['channel'], dataset['num_classes'], dataset['img_size'], self.seed).to(args.device)
        self.net.train()
        for param in list(self.net.parameters()):
            param.requires_grad = False
        self.embed = self.net.module.embed if torch.cuda.device_count() > 1 else self.net.embed 

    def synthesize(self, dataset, args): 
        def get_images(c, n):
            idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
            imgs = [dataset['trainset'][i][0].to(args.device) for i in idx_shuffle]
            return torch.stack(imgs, dim=0)  

        # assuming ConvNet I think?
        loss = torch.tensor(0.0).to(args.device)
        for c in range(dataset['num_classes']):
            img_real_by_class = get_images(c, args.batch_real)
            img_synth_by_class = self.img[c * args.ipc : (c + 1) * args.ipc].reshape((args.ipc, dataset['channel'], *dataset['img_size']))


            seed = int.from_bytes(os.urandom(4), 'little') 
            img_real_by_class = DiffAugment(img_real_by_class, self.dsa_param, self.dsa_strategy, seed=seed)
            img_synth_by_class = DiffAugment(img_synth_by_class, self.dsa_param, self.dsa_strategy, seed=seed)

            output_real = self.embed(img_real_by_class).detach()
            output_synth = self.embed(img_synth_by_class)

            loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_synth, dim=0))**2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

# JUST EXAMPLE

class GLaD:
    def __init__(self, synth_img):
        self.synth_img = synth_img     # pass by reference
        self.initialized_latent_whatever = None
        pass
    
    def pre_trained_generator_G(self):
        pass 

    def synthesize(self, args):
        # compute loss L
        self.initialized_latent_whatever = 42 # update by optimization on L 
        self.synth_img = self.pre_trained_generator_G(self.initialized_latent_whatever)




    