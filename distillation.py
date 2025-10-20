import os
import torch 
import numpy as np
from utils import get_network
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
    def __init__(self, trainset, img_synth, label_synth, num_classes, channel, img_size, args):
        self.img = img_synth
        self.label = label_synth

        self.trainset = trainset

        N = len(trainset)
        self.indices_class = [[] for c in range(num_classes)]
        for i in range(N):
            _, lab = trainset[i]           # (transforms applied if dataset has transform)
            self.indices_class[lab].append(i)

        self.num_classes = num_classes
        self.channel = channel
        self.img_size = img_size 
        self.optimizer = torch.optim.Adam([self.img], lr=args.lr_img, betas=(0.5, 0.999)) 
        self.optimizer.zero_grad()

        self.dsa_param = ParamDiffAug()
        self.dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
        self.dsa = True

    def synthesize(self, args): 
        net = get_network(args.train_model, self.channel, self.num_classes, self.img_size).to(args.device)
        net.train()
        for param in list(net.parameters()):
            param.requires_grad = False

        embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed 


        def get_images(c, n):
            idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
            imgs = [self.trainset[i][0].to(args.device) for i in idx_shuffle]
            return torch.stack(imgs, dim=0)  

        # assuming ConvNet I think?
        loss = torch.tensor(0.0).to(args.device)
        for c in range(self.num_classes):
            img_real = get_images(c, args.batch_real)
            img_syn = self.img[c * args.ipc : (c + 1) * args.ipc].reshape((args.ipc, self.channel, *self.img_size))

            if args.dsa:
                seed = int.from_bytes(os.urandom(4), 'little') 
                img_real = DiffAugment(img_real, self.dsa_param, self.dsa_strategy, seed=seed)
                img_syn = DiffAugment(img_syn, self.dsa_param, self.dsa_strategy, seed=seed)

            output_real = embed(img_real).detach()
            output_syn = embed(img_syn)

            loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        




    