import numpy as np
import h5py
from hyperparams import Hyperparams as hp
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import os
from unet.unet_student import UNet as unet_s
from unet.unet_teacher import UNet as unet_t
from tqdm import tqdm
import random
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='UNet network')
    parser.add_argument('--model_name', type=str,
                        help='teacher model name')
    return parser.parse_args()


class audioDatasets(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        super().__init__()
        print("h5 path:{}".format(h5_path))
        with h5py.File(h5_path, 'r') as hf:
            print('List of arrays in input file:', hf.keys())
            self.x = torch.from_numpy(np.array(hf.get('data')).reshape(-1, 1, 1, hp.win_length))
            self.y = torch.from_numpy(np.array(hf.get('label')).reshape(-1, 1, 1, hp.win_length))
        print('shape of x:', self.x.size())
        print('shape of y:', self.y.size())

    def __getitem__(self, index):
        data_pair = (self.x[index], self.y[index])
        # print(index)
        return data_pair

    def __len__(self):
        return len(self.x)


def reduce_mean(x, axis):
    sorted(axis)
    axis = list(reversed(axis))
    for d in axis:
        x = torch.mean(x, dim=d)
    return x


def get_loss(x, y, y_pred):
    n_filters = hp.n_filters_teacher
    unet = unet_t(n_filters)
    model_name = '/model_teacher.pt'
    # print("Loading model {}".format(hp.model_path + model_name))
    unet.cuda()
    unet = nn.DataParallel(unet)
    unet.load_state_dict(torch.load(hp.model_path + model_name))
    y_t = unet(x.to(device='cuda', dtype=torch.float))

    def rmse(y, y_pred):
        sqrt_l2_loss = torch.sqrt(reduce_mean((y_pred - y)**2, [1, 2, 3]))
        avg_sqrt_l2_loss = torch.mean(sqrt_l2_loss, 0)
        return avg_sqrt_l2_loss

    t_loss = rmse(y, y_t)
    s_loss = rmse(y, y_pred)
    return 0.9 * t_loss + 0.1 * s_loss


def snr(y, y_pred, device):
    sqrt_l2_loss = torch.sqrt(reduce_mean((y_pred - y)**2, [1, 2, 3]))
    sqrt_l2_norm = torch.sqrt(reduce_mean(y**2, axis=[1, 2, 3]))
    snr = 20 * torch.log(sqrt_l2_norm / (sqrt_l2_loss + 1e-8) + 1e-8) / torch.log(torch.tensor(10.).to(device))
    avg_snr = reduce_mean(snr, axis=[0])
    return avg_snr


def LSDBase(input1, input2):
    x = torch.squeeze(input1)
    x = torch.stft(x, hp.win_length, hp.win_length)
    x = torch.log(torch.abs(x)**2 + 1e-8)
    x_hat = torch.squeeze(input2)
    x_hat = torch.stft(x_hat, hp.win_length, hp.win_length)
    x_hat = torch.log(torch.abs(x_hat)**2 + 1e-8)
    lsd = reduce_mean(torch.sqrt(reduce_mean(torch.mul(x-x_hat, x-x_hat), axis=[2, 3]))+1e-8, axis=[0])

    return lsd


def train(dataloader, model, optimizer, device, scheduler, epoch, step_per_epoch):
    loss_all = []
    snr_all = []
    lsd_all = []
    out = []
    desc = "ITERATION - loss: {:.4f} - snr: {:.4f} - lsd: {:.4f} "
    t = tqdm(dataloader, total=step_per_epoch, desc=desc.format(0, 0, 0))

    for (x, y) in t:
        x, y = x.to(device), y.to(device)
        model = model.train()
        # model.to(device)
        output = model(x)

        # print("oooooooo:", output.grad_fn)
        loss = get_loss(x, y, output).cuda()
        snr_metrics = snr(y, output, device).cuda()
        lsd_metrics = LSDBase(y, output).cuda()
        loss_all.append(loss.item())
        snr_all.append(snr_metrics.item())
        lsd_all.append(lsd_metrics.item())
        t.set_description("ITERATION - loss: {:.4f} - snr: {:.4f} - lsd: {:.4f}".format(loss.item(),
                                                                                        snr_metrics.item(),
                                                                                        lsd_metrics.item()))
        t.refresh()
        optimizer.zero_grad()
        loss.backward()
        scheduler.step()
        optimizer.step()
        # out = np.append(out, output.cpu().data.numpy().flatten())
    loss_mean = np.mean(loss_all)
    snr_mean = np.mean(snr_all)
    lsd_mean = np.mean(lsd_all)
    print("training epoch:{}, global_step:{}, loss:{:.6f}, snr:{:.6f}, lsd:{:.6f}".format(
        epoch, epoch*step_per_epoch, loss_mean, snr_mean, lsd_mean))
    return loss_mean


if __name__ == "__main__":
    try:
        os.mkdir(hp.model_path)
    except:
        pass
    torch.manual_seed(7)  # cpu
    torch.cuda.manual_seed(7)  # gpu
    np.random.seed(7)  # numpy
    random.seed(7)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn

    def worker_init_fn(worker_id):
        np.random.seed(7 + worker_id)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    n_filters = hp.n_filters_student
    trainset = audioDatasets(hp.h5_path_train_unet)
    # kwargs = {'num_workers': 4} if use_cuda else {}
    trainloader = data.DataLoader(trainset, batch_size=hp.batch_size, shuffle=True,
                                  num_workers=0, worker_init_fn=worker_init_fn)
    step_per_epoch = len(trainloader)
    print("step_per_epoch:", step_per_epoch)
    unet = unet_s(n_filters)
    unet.cuda()
    print(unet)
    print("id: ", os.getpid())
    if hp.multi_gpu:
        unet = nn.DataParallel(unet)
    optimizer = optim.Adam(unet.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_per_epoch, gamma=0.1)
    best_train_loss = None
    try:
        for epoch in range(1, 1000):
            train_loss = train(trainloader, unet, optimizer, device, scheduler, epoch, step_per_epoch)
            if not best_train_loss or train_loss < best_train_loss:
                torch.save(unet.state_dict(), os.path.join(hp.model_path, 'model_student_dis.pt'.format(epoch)))
                best_train_loss = train_loss
                print('saved model in epoch:{}'.format(epoch))
            # if hp.save_real_time_result:
            #     librosa.output.write_wav(os.path.join(hp.realtime_output_dir, '{}_epoch.wav'.format(epoch)),
            #                              output, 44100)
    except KeyboardInterrupt:
        print('interrupt by self')
