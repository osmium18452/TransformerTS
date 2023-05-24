import argparse
import gc
import os.path
import pickle
import platform
import random

from DataPreprocessor import DataPreprocessor, TSDataset
from Models.Transformer import Transformer, Transformer_v1
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# TODO: draw pic
# TODO: GRID SEARCH
# TODO: POSITIONAL ENCODING
# TODO: MULTI LAYER LSTM
# TODO: MULTI HEAD GNN

# torchrun --nproc_per_node=6 --nnodes=1 main.py -GMN std -c 6 -a 0.1 -t 2 -m I -e 3 --stride 1 -C 2,3,4,5,6,7
# torchrun --nproc_per_node=8 --nnodes=1 main.py -GMfN minmax -c 6 -a 0.1 -t 2 -m I -e 3 --stride 1
# torchrun --nproc_per_node=8 --nnodes=1 main.py -GfMN std -a 0.1 -b 6 -s save/23.04.21 -t 4 -m L -e 60 -i 48 -p 400

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alt_learning_rate', type=float, default=None)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-C', '--CUDA_VISIBLE_DEVICES', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('-c', '--cuda_device', type=int, default=0)
    parser.add_argument('-d', '--dataset', type=str, default='wht')
    parser.add_argument('-D', '--d_model', type=int, default=512)
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-F', '--fixed_seed', action='store_true')
    parser.add_argument('-f', '--fudan', action='store_true')
    parser.add_argument('-m', '--gnn_map', type=str, default='L', help='L: laplacian, I')
    parser.add_argument('-G', '--gpu', action='store_true')
    parser.add_argument('-H', '--hidden_size', type=int, default=40)
    parser.add_argument('-i', '--input_window', type=int, default=60)
    parser.add_argument('-l', '--lr', type=float, default=0.001)
    parser.add_argument('-M', '--multiGPU', action='store_true')
    parser.add_argument('-o', '--model', type=str, default='v1', help='v1, v2')
    parser.add_argument('-n', '--nhead', type=int, default=8)
    parser.add_argument('-N', '--normalize', type=str, default=None, help='std, minmax, zeromean')
    parser.add_argument('-P', '--positional_encoding', type=str, default='sinusodial', help='zero,sin,sinusodial')
    parser.add_argument('-p', '--predict_window', type=int, default=30)
    parser.add_argument('-s', '--save_path', type=str, default='save')
    parser.add_argument('-t', '--transformer_layers', type=int, default=6)
    parser.add_argument('-V', '--dont_validate', action='store_true')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--validate_ratio', type=float, default=0.1)
    args = parser.parse_args()

    use_cuda = args.gpu
    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else -1
    if args.multiGPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda:' + str(args.cuda_device) if use_cuda else 'cpu')

    batch_size = args.batch_size
    input_time_window = args.input_window
    pred_time_window = args.predict_window
    nhead = args.nhead
    hidden_size = args.hidden_size * nhead
    dataset = args.dataset
    total_epoch = args.epoch
    lr = args.lr
    transformer_layers = args.transformer_layers
    normalize = args.normalize
    train_ratio = args.train_ratio
    validate_ratio = args.validate_ratio
    cuda_device = args.cuda_device
    test_ratio = 1 - train_ratio - validate_ratio
    alt_learning_rate = args.alt_learning_rate
    step_size = args.step_size
    gnn_map = args.gnn_map
    stride = args.stride
    multiGPU = args.multiGPU
    fixed_seed = args.fixed_seed
    positional_encoding = args.positional_encoding
    save_path = args.save_path
    fudan = args.fudan
    validate = not args.dont_validate
    which_model = args.model
    d_model = args.d_model

    if (multiGPU and local_rank == 0) or not multiGPU:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(args)

    # fix random seed
    if multiGPU or fixed_seed:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if multiGPU:
        import torch.distributed as dist

        dist.init_process_group(backend="nccl")

    if platform.system() == 'Windows':
        # data_dir = 'C:\\Users\\17110\\Desktop\\ts forecasting\\dataset\\pkl'
        # map_dir = 'C:\\Users\\17110\\Desktop\\ts forecasting\\dataset\\map'
        data_dir = 'E:\\forecastdataset\\pkl'
        map_dir = 'E:\\forecastdataset\\map'
    else:
        if fudan:
            data_dir = '/remote-home/liuwenbo/pycproj/forecastdata//pkl/'
            map_dir = '/remote-home/liuwenbo/pycproj/forecastdata//map/'
        else:
            data_dir = '/home/icpc/pycharmproj/forecast.dataset/pkl/'
            map_dir = '/home/icpc/pycharmproj/forecast.dataset/map/'
    if dataset == 'wht':
        dataset = pickle.load(open(os.path.join(data_dir, 'wht.pkl'), 'rb'))
        causal_map = pickle.load(open(os.path.join(map_dir, 'wht.map.pkl'), 'rb'))
    else:
        dataset = None
        causal_map = None
        print('dataset not found')
        exit()
    # if multiGPU and local_rank !=0:
    #     dist.barrier()
    data_preprocessor = DataPreprocessor(dataset=dataset, input_time_window=input_time_window,
                                         output_time_window=pred_time_window, normalize=normalize,
                                         train_ratio=train_ratio, valid_ratio=validate_ratio, stride=stride,
                                         positional_encoding=positional_encoding)
    ramdisk_path = '/home/icpc/ramdisk'
    # if multiGPU and local_rank==0:
    #     dist.barrier()

    causal_map = torch.Tensor(causal_map)
    L = causal_map
    for i in range(L.shape[0]):
        L[i][i] += torch.sum(causal_map, dim=1)[i]
    if gnn_map == 'I':
        causal_map = torch.eye(causal_map.shape[0])
    elif gnn_map == 'L':
        causal_map = L
    elif gnn_map == 'T':
        causal_map = L.transpose(0, 1)
    else:
        print('gnn map not found')
        exit()

    if use_cuda:
        causal_map = causal_map.to(device)
    sensors = data_preprocessor.load_num_sensors()
    if which_model == 'v1':
        model = Transformer_v1(sensors, sensors, n_heads=nhead)
    else:
        model = Transformer(sensors, sensors, d_model=d_model, n_heads=nhead)
    if use_cuda:
        model.to(device)

    train_input, train_tgt, train_gt = data_preprocessor.load_train_data()
    test_input, test_tgt, test_gt = data_preprocessor.load_test_data()
    valid_input, valid_tgt, valid_gt = data_preprocessor.load_validate_data()
    std, mean = None, None
    valid_original_gt, test_original_gt = None, None
    maxx, minn = None, None
    if normalize == 'std':
        std, mean = data_preprocessor.load_std_and_mean()
    elif normalize == 'minmax':
        minn, maxx = data_preprocessor.load_min_max()
    elif normalize == 'zeromean':
        std, mean = data_preprocessor.load_std_and_mean()
    elif normalize is None:
        pass
    else:
        print('invalid normalize option')
        exit()
    del data_preprocessor
    gc.collect()
    # print(test_original_gt.shape,valid_original_gt.shape)
    # exit()
    train_set = TSDataset(train_input, train_tgt, train_gt)
    test_set = TSDataset(test_input, test_tgt, test_gt)
    valid_set = TSDataset(valid_input, valid_tgt, valid_gt)
    num_train_samples = train_input.shape[0]
    num_test_samples = test_input.shape[0]
    num_valid_samples = valid_input.shape[0]
    del train_input, train_tgt, train_gt, test_input, test_tgt, valid_input, valid_tgt, valid_gt
    gc.collect()
    train_loader = DataLoader(train_set, sampler=DistributedSampler(train_set) if multiGPU else None,
                              batch_size=batch_size, shuffle=False if multiGPU else True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    validate_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if alt_learning_rate is not None:
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=alt_learning_rate)
    loss_fn = torch.nn.MSELoss()

    if (multiGPU and local_rank == 0) or not multiGPU:
        pbar_epoch = tqdm(total=total_epoch, ascii=True, dynamic_ncols=True)
    for epoch in range(total_epoch):
        # train
        model.train()
        total_iters = len(train_loader)
        if (multiGPU and local_rank == 0) or not multiGPU:
            pbar_iter = tqdm(total=total_iters, ascii=True, leave=False, dynamic_ncols=True)
            pbar_iter.set_description('training')
        for input_batch, tgt, gt in train_loader:
            if use_cuda:
                input_batch = input_batch.to(device)
                tgt = tgt.to(device)
                gt = gt.to(device)
            optimizer.zero_grad()
            output = model(input_batch, tgt)
            # print('output shape', output.shape,tgt.shape,input_batch.shape)
            # print("output,gt shape", output.shape, gt.shape)
            loss = loss_fn(output, gt)
            loss.backward()
            optimizer.step()
            if (multiGPU and local_rank == 0) or not multiGPU:
                pbar_iter.set_postfix_str('loss: %.4f' % (loss.item()))
                pbar_iter.update(1)
        if (multiGPU and local_rank == 0) or not multiGPU:
            pbar_iter.close()
        # dist.barrier()

        # validate
        gc.collect()
        model.eval()
        if ((multiGPU and local_rank == 0) or not multiGPU) and validate:
            with torch.no_grad():
                total_iters = len(validate_loader)
                pbar_iter = tqdm(total=total_iters, ascii=True, leave=False, dynamic_ncols=True)
                pbar_iter.set_description('validating')
                output_list = []
                gt_list = []
                for input_batch, tgt, gt in validate_loader:
                    if use_cuda:
                        input_batch = input_batch.to(device)
                        tgt = tgt.to(device)
                    output_list.append(model(input_batch, tgt).cpu())
                    gt_list.append(gt)
                    pbar_iter.update(1)
                pbar_iter.close()
                output = torch.cat(output_list, dim=0)
                ground_truth = torch.cat(gt_list, dim=0)
                if normalize == 'std':
                    output_original = output.transpose(1, 2) * std + mean
                    gt_original = ground_truth.transpose(1, 2) * std + mean
                    loss_original = loss_fn(output_original, gt_original)
                elif normalize == 'minmax':
                    output_original = output.transpose(1, 2) * (maxx - minn) + minn
                    gt_original = ground_truth.transpose(1, 2) * (maxx - minn) + minn
                    loss_original = loss_fn(output_original, gt_original)
                elif normalize == 'zeromean':
                    output_original = output.transpose(1, 2) + mean
                    gt_original = ground_truth.transpose(1, 2) + mean
                    loss_original = loss_fn(output_original, gt_original)
                else:
                    print('invalid normalize option')
                    exit()
                loss = loss_fn(output, ground_truth)
                if normalize is not None:
                    pbar_epoch.set_postfix_str(
                        'valid loss: %.4f, valid loss original: %.4f' % (loss.item(), loss_original.item()))
                else:
                    pbar_epoch.set_postfix_str('valid loss: %.4f' % (loss.item()))
                # print('pbar epoch upd')
                pbar_epoch.update()
        if multiGPU:
            dist.barrier()
    if (multiGPU and local_rank == 0) or not multiGPU:
        pbar_epoch.close()
    gc.collect()

    # test
    if (multiGPU and local_rank == 0) or not multiGPU:
        total_iters = len(test_loader)
        pbar_iter = tqdm(total=total_iters, ascii=True, dynamic_ncols=True)
        pbar_iter.set_description('testing')
        output_list = []
        gt_list = []
        with torch.no_grad():
            model.eval()
            for input_batch, tgt, gt in test_loader:
                if use_cuda:
                    input_batch = input_batch.to(device)
                    tgt = tgt.to(device)
                output_list.append(model(input_batch, tgt).cpu())
                gt_list.append(gt)
                pbar_iter.update(1)
        pbar_iter.close()
        output = torch.cat(output_list, dim=0)
        ground_truth = torch.cat(gt_list, dim=0)
        # print(output.shape)
        # exit()
        if normalize == 'std':
            output_original = output.transpose(1, 2) * std + mean
            ground_truth = ground_truth.transpose(1, 2) * std + mean
            loss_original = loss_fn(output_original, ground_truth)
        elif normalize == 'minmax':
            output_original = output.transpose(1, 2) * (maxx - minn) + minn
            ground_truth = ground_truth.transpose(1, 2) * (maxx - minn) + minn
            loss_original = loss_fn(output_original, ground_truth)
        elif normalize == 'zeromean':
            output_original = output.transpose(1, 2) + mean
            ground_truth = ground_truth.transpose(1, 2) + mean
            loss_original = loss_fn(output_original, ground_truth)

        loss = loss_fn(output, test_gt)
        if multiGPU:
            dist.destroy_process_group()
        if normalize:
            print('\033[32mtest loss: %.4f, test loss original: %.4f\033[0m' % (loss.item(), loss_original.item()))
            f = open(os.path.join(save_path, 'test_loss.txt'), 'a')
            print(args, file=f)
            print('test loss: %.4f, test loss original: %.4f' % (loss.item(), loss_original.item()), file=f)
            f.close()
        else:
            print('\033[32mtest loss: %.4f\033[0m' % (loss.item()))
            f = open(os.path.join(save_path, 'test_loss.txt'), 'a')
            print(args, file=f)
            print('\033[31mtest loss: %.4f\033[0m' % (loss.item()), file=f)
            f.close()
