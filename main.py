import copy
import time 
import torch
import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from utils import get_dataset, get_network, get_method, train_using_synth, train_using_real, test


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
	                        conflict_handler='resolve')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--train_model', type=str, default='ConvNet', help='model used for distillation')
    parser.add_argument('--test_model', type=str, default='ConvNet', help='model used to test distillation')
    parser.add_argument('--method', type=str, default='DM', help="distillation method")
    parser.add_argument('--ipc', type=int, default=50, help='images per class')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--train_iters', type=int, default=1000, help='epochs to train a model with synthetic images') 
    parser.add_argument('--synth_iters', type=int, default=20000, help='iterations to syntheize images')
    parser.add_argument('--synth_checks', type=int, default=1, help='number of checkpoints during synthesis')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--in', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


def main(args):
    # check if args.dataset is already processed and ready in a .pt file 
    # if not the proceed to generate 
    (   channel,        # int of number of channels
        img_size,        # (width, height)
        num_classes,    # int of number of classes
        class_names,    # list of class labels
        mean,           # list of normalization means, one per channel
        std,            # list of normalization std's, one per channel
        trainset,       # object from torchvision.datasets; think of as big list of tensors
        testset,        # object from torchvision.datasets; think of as big list of tensors
        trainloader,    # object torch.utils.data.DataLoader of trainset with batch size idk
        testloader      # object torch.utils.data.DataLoader of testset with batch size 256
    ) = get_dataset(args.dataset, args.data_path)


    distillation = get_method(args.method)

    test_model_real = get_network(args.test_model, channel, num_classes, img_size)

    img_synth = torch.randn(size=(num_classes * args.ipc, channel, *img_size), 
                            dtype=torch.float, requires_grad=True, device=args.device)
    label_synth = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    
    synth_results = [0] * args.synth_checks
    check_iters = [(i + 1) * (args.iterations / args.synth_checks) - 1 for i in range(args.synth_checks)]

    # shoul technically add outter loop here for repeating synthesis process 
    # i.e. under same intialize network, repear distillation 10 times 
    # each time see what is the average performance 
    # this is because distillation is a randomized process
    # I think?????

    for iters in range(args.iterations): 
        running_synth_time = 0.0
        start_synth_time = time.time()

        distillation.synthesize(img_synth, label_synth, args.ipc, args.lr_img)

        stop_synth_time = time.time()
        running_synth_time += (stop_synth_time - start_synth_time)
        
        if iters in check_iters:
            test_model_synth = copy.deepcopy(test_model_real)
            train_using_synth(test_model_synth, img_synth, label_synth, args)
            test_loss, test_acc = test(test_model_synth, testloader, args) 
            synth_results[iters] = test_loss, test_acc, running_synth_time

    train_using_real(test_model_real, trainloader, args)  
    real_results = test(test_model_real, testloader, args)

    write_output('./output.txt', real_results, synth_results, args)
    return (real_results, synth_results) 

def write_output(file_path, real_results, synth_results, args):
    s1 = f'''
    ====== Experiment Output ======\n
    Specs:\n
        \tDistillation method: {args.method}\n
        \tDataset: {args.dataset}\n
        \tIPC: {args.ipc}\n
        \tNetwork used for distillation: {args.train_model}\n
        \tNetwork trained on distilled data: {args.test_model}\n
        \tDistillation Iterations: {args.synth_iters}\n
    Performance:\n
        \tReal dataset: Loss {real_results[0] : .6f}, Accuracy {real_results[1] : .6f}\n 
        \tSynthetic dataset:\n
    '''
    [(i + 1) * (args.iterations / args.synth_checks) - 1 for i in range(args.synth_checks)]
    for i in range(args.synth_checks):
        check = (i + 1) * (args.iterations / args.synth_checks)
        loss, acc, synth_time = synth_results[i] 
        s = f"\t\tEpoch {check}: Loss {loss : .6f}, Accuracy {acc : .6f}, Time {synth_time}\n"
        s1 += s
    
    with open(file_path, 'w') as file:
        file.write(s1)


if __name__ == '__main__':
    main(parse_args())

