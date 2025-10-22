import os
import time 
import torch
import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from utils import get_dataset, get_method, train_using_synth, train_using_real, test
from networks import get_network


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
	                        conflict_handler='resolve')
    parser.add_argument('--dataset_name', type=str, default='FashionMNIST', help='dataset')
    parser.add_argument('--data_path', type=str, default='./temp_data_path', help='dataset path')

    parser.add_argument('--method', type=str, default='DM', help="distillation method")
    parser.add_argument('--ipc', type=int, default=3, help='images per class')

    parser.add_argument('--synth_model', type=str, default='ConvNet', help='model used to synthesize')
    parser.add_argument('--train_model', type=str, default='ConvNet', help='model that is trained on real/synth data')
    
    parser.add_argument('--num_tests', type=int, default=10, help='number of times to train a model on real/synth data')
    # parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--train_epochs', type=int, default=10, help='epochs to train a model on data') 
    parser.add_argument('--synth_iters', type=int, default=10, help='iterations to syntheize images')
    parser.add_argument('--synth_checks', type=int, default=1, help='number of checkpoints during synthesis')

    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=186, help='batch size for real data')
    parser.add_argument('--batch_synth', type=int, default=64, help='batch size for synthetic data')

    # parser.add_argument('--in', type=int, default=256, help='batch size for training networks')
    
    parser.add_argument('--output_path', type=str, default='./output.txt', help='path to write results')

    args = parser.parse_args()

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.seed = int.from_bytes(os.urandom(4), 'little')    # seed for the model that gets trained, so all training starts with same init param.

    return args


def test_synth(img_synth, label_synth, dataset, args):
    results = []
    for _ in range(args.num_tests):
        model = get_network(args.train_model, dataset['channel'], dataset['num_classes'], dataset['img_size'], args.seed).to(args.device)
        train_using_synth(model, img_synth, label_synth, args) 
        results.append(
            test(model, dataset['testloader'], args)
        )
    return results 

def test_real(dataset, args):
    results = []
    for _ in range(args.num_tests):
        model = get_network(args.train_model, dataset['channel'], dataset['num_classes'], dataset['img_size'], args.seed).to(args.device)
        train_using_real(model, dataset['trainloader'], args)
        results.append(
            test(model, dataset['testloader'], args)
        )
    return results


def main(args):
    device = torch.device(args.device) 
    dataset = get_dataset(args.dataset_name, args.data_path)

    img_synth = torch.randn(size=(dataset['num_classes'] * args.ipc, dataset['channel'], *dataset['img_size']), 
                            dtype=torch.float, requires_grad=True, device=device)
    label_synth = torch.repeat_interleave(
        torch.arange(dataset['num_classes'], device=device), 
        repeats=args.ipc
    ) # [0,0,0, 1,1,1, ..., 9,9,9]
    
    distillation = get_method(img_synth, label_synth, dataset, args)

    synth_results = []
    check_iters = [(i + 1) * (args.synth_iters / args.synth_checks) - 1 for i in range(args.synth_checks)]

    for iter in range(args.synth_iters): 
        time_elapsed = 0.0
        time_start = time.time()

        distillation.synthesize(dataset, args)

        time_stop = time.time()
        time_elapsed += time_stop - time_start
        
        if iter in check_iters:
            results = test_synth(img_synth, label_synth, dataset, args) 
            synth_results.append((results, time_elapsed))


    real_results = test_real(dataset, args) 

    # write_output(args.output_path, real_results, synth_results, args)
    return (real_results, synth_results) 





# I think this is currently broken after I changed stuff

def write_output(file_path, real_results, synth_results, args):
    s1 = f'''
    ====== Experiment Output ======\n
    Specs:\n
        \tDistillation method: {args.method}\n
        \tDataset: {args.dataset_name}\n
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

