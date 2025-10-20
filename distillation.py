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
            
class DM:
    def __init__(self):
        pass




    