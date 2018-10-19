import torch
from hyperparams import Hyperparams as hp
import librosa
import os
import numpy as np
import torch.nn as nn
from torchsummary import summary


def reduce_mean(x, axis):
    sorted(axis)
    axis = list(reversed(axis))
    for d in axis:
        x = torch.mean(x, dim=d)
    return x


def high_emphasisV4(x, sr, emphasis_freq_from, emphasis_freq_to, dB):
    def _SNR_db_to_linear(signal_amplitude):
        after_div_signal = 0
        return after_div_signal * signal_amplitude

    rate_low = emphasis_freq_from / sr
    rate_high = emphasis_freq_to / sr
    magnitude, phase = librosa.magphase(librosa.stft(x, n_fft=128))
    S = magnitude
    r, c = S.shape
    cut_low = int(r * rate_low)
    cut_high = int(r * rate_high)

    S_high = S[cut_low:cut_high, :]
    signal_amplitude = np.max(S_high) - np.min(S_high)
    rate = _SNR_db_to_linear(signal_amplitude) / signal_amplitude
    S[cut_low:cut_high, :] += S_high * (rate - 1)

    x = librosa.core.istft(S * phase)
    return x


def snr(y, y_pred):
    sqrt_l2_loss = torch.sqrt(reduce_mean((y_pred - y)**2, [1, 2, 3]))
    sqrt_l2_norm = torch.sqrt(reduce_mean(y**2, axis=[1, 2, 3]))
    snr = 20 * torch.log(sqrt_l2_norm / (sqrt_l2_loss + 1e-8) + 1e-8) / torch.log(torch.tensor(10.))
    avg_snr = reduce_mean(snr, axis=[0])
    return avg_snr


def LSDBase(input1, input2):
    x = torch.squeeze(input1)
    x = torch.stft(x, hp.win_length, hp.win_length)
    x = torch.log(torch.abs(x)**2 + 1e-8)
    x_hat = torch.squeeze(input2)
    x_hat = torch.stft(x_hat, hp.win_length, hp.win_length)
    x_hat = torch.log(torch.abs(x_hat)**2 + 1e-8)
    lsd = reduce_mean(torch.sqrt(reduce_mean(torch.mul(x-x_hat, x-x_hat), axis=[1, 2]))+1e-8, axis=[0])

    return lsd


def load_test_data(path, sr):
    x_hr, _ = librosa.load(path, sr=sr)
    return x_hr


teacher = False
if teacher:
    from unet.unet_teacher import UNet
    n_filters = hp.n_filters_teacher
    model_name = '/model_teacher.pt'.format(3)
else:
    from unet.unet_student import UNet
    n_filters = hp.n_filters_student
    model_name = '/model_student_dis.pt'.format(41)
    # model_name = '/model_student_song_8-16_{}.pt'.format(31)


unet = UNet(n_filters)
print("Loading model {}".format(hp.model_path + model_name))

unet.cuda()
unet = nn.DataParallel(unet)
unet.load_state_dict(torch.load(hp.model_path + model_name))
print('MODEL LOADED')
# summary(unet, (1, 1, 8192))
print("id: ", os.getpid())


test_wav = os.listdir(hp.test_data_path)

for i, _ in enumerate(test_wav):
    name = test_wav[i].split('.')[0]
    x_lq = load_test_data(hp.test_data_path + test_wav[i], sr=44100)
    x_hq = load_test_data(hp.groundtruth_data_path + test_wav[i], sr=44100)
    x_0_5k = librosa.resample(x_lq, 44100, 11200)
    x_hq_0_5k = librosa.resample(x_hq, 44100, 11200)
    output_5_16k = high_emphasisV4(x_lq, sr=44100, emphasis_freq_from=0, emphasis_freq_to=11200, dB=0)  # 4-1

    x_pq = []
    length = hp.win_length
    segments = int(np.ceil(len(x_0_5k) / length))
    x_0_5k_pad = np.zeros(shape=(segments * length, ), dtype=np.float32)
    x_hq_pad = np.zeros(shape=(segments * length, ), dtype=np.float32)

    x_0_5k_pad[:len(x_0_5k), ] = x_0_5k
    x_hq_pad[:len(x_0_5k), ] = x_hq_0_5k
    x_0_5k_pad = x_0_5k_pad.reshape((segments, 1, length))
    x_hq_pad = x_hq_pad.reshape((segments, 1, length))
    loss, snr_matric, LSDBase_matric = [], [], []

    for j in range(segments):
        tmp_lq = []
        tmp_hp = []
        zero_pad = np.zeros((1, 1, length))
        if j == 0:
            tmp_lq = np.append(tmp_lq, zero_pad)
            tmp_lq = np.append(tmp_lq, x_0_5k_pad[j])
            tmp_lq = np.append(tmp_lq, x_0_5k_pad[j + 1])

            tmp_hp = np.append(tmp_hp, zero_pad)
            tmp_hp = np.append(tmp_hp, x_hq_pad[j])
            tmp_hp = np.append(tmp_hp, x_hq_pad[j + 1])
        elif j == segments - 1:
            tmp_lq = np.append(tmp_lq, x_0_5k_pad[j - 1])
            tmp_lq = np.append(tmp_lq, x_0_5k_pad[j])
            tmp_lq = np.append(tmp_lq, zero_pad)

            tmp_hp = np.append(tmp_hp, x_hq_pad[j - 1])
            tmp_hp = np.append(tmp_hp, x_hq_pad[j])
            tmp_hp = np.append(tmp_hp, zero_pad)
        else:
            tmp_lq = np.append(tmp_lq, x_0_5k_pad[j - 1])
            tmp_lq = np.append(tmp_lq, x_0_5k_pad[j])
            tmp_lq = np.append(tmp_lq, x_0_5k_pad[j + 1])

            tmp_hp = np.append(tmp_hp, x_hq_pad[j - 1])
            tmp_hp = np.append(tmp_hp, x_hq_pad[j])
            tmp_hp = np.append(tmp_hp, x_hq_pad[j + 1])
        input = torch.from_numpy(np.reshape(tmp_lq, (1, 1, 1, length * 3)))
        output_0_5k = unet(input.to(device='cuda', dtype=torch.float32))
        tmp_hp = np.reshape(tmp_hp, (1, 1, 1, length * 3))
        output_0_5k = output_0_5k.cpu()

        snr_1 = snr(torch.from_numpy(np.array(tmp_hp, dtype=np.float32)), output_0_5k)
        lsd = LSDBase(torch.from_numpy(np.array(tmp_hp, dtype=np.float32)), output_0_5k)

        snr_matric.append(snr_1.data.numpy())
        LSDBase_matric.append(lsd.data.numpy())

    snr_ = np.mean(np.array(snr_matric))
    lsd_ = np.mean(np.array(LSDBase_matric))
    print("{}: snr:{:.4f}, lsd:{:.4f}".format(name, snr_, lsd_))
