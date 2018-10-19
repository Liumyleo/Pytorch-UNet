import librosa
import numpy as np
import os
from hyperparams import Hyperparams as hp
import torch
import torch.nn as nn
from torchsummary import summary
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='UNet network')
    parser.add_argument('--mode', type=str, default='student',
                        help='Which model will be predicted?')
    return parser.parse_args()


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


def load_test_data(path, sr):
    x_hr, _ = librosa.load(path, sr=sr)
    return x_hr


if __name__ == "__main__":
    args = get_arguments()
    test_wav = os.listdir(hp.test_data_path)
    print('restoring weights')
    if hp.output_dir[-1] == '/':
        hp.output_dir = hp.output_dir[:-1]
    output_dir = hp.output_dir + '/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.mode == 'teacher':
        from unet.unet_teacher import UNet
        n_filters = hp.n_filters_teacher
        model_name = '/model_teacher.pt'
    elif args.mode == 'student':
        from unet.unet_student import UNet
        n_filters = hp.n_filters_student
        model_name = '/model_student.pt'

    elif args.mode == 'student_dis':
        from unet.unet_student import UNet
        n_filters = hp.n_filters_student
        model_name = '/model_student_dis.pt'

    unet = UNet(n_filters)
    print("Loading model {}".format(os.path.join(hp.model_path, model_name)))
    unet.cuda()
    unet = nn.DataParallel(unet)
    unet.load_state_dict(torch.load(os.path.join(hp.model_path, model_name)))
    print('MODEL LOADED')
    summary(unet, (1, 1, 8192))
    print("id: ", os.getpid())

    for i, _ in enumerate(test_wav):
        name = test_wav[i].split('.')[0]
        print(name)
        x_lq = load_test_data(hp.test_data_path + test_wav[i], sr=hp.sr)
        x_pq = []
        length = 480
        segments = int(np.ceil(len(x_lq) / length))
        x_lq_pad = np.zeros(shape=(segments * length, 1), dtype=np.float32)
        x_lq_pad[:len(x_lq), 0] = x_lq
        x_lq_pad = x_lq_pad.reshape((segments, 1, length))

        for j in range(segments):
            tmp = []
            zero_pad = np.zeros((1, 1, length))
            if j == 0:
                tmp = np.append(tmp, zero_pad)
                tmp = np.append(tmp, x_lq_pad[j])
                tmp = np.append(tmp, x_lq_pad[j+1])
            elif j == segments-1:
                tmp = np.append(tmp, x_lq_pad[j-1])
                tmp = np.append(tmp, x_lq_pad[j])
                tmp = np.append(tmp, zero_pad)
            else:
                tmp = np.append(tmp, x_lq_pad[j-1])
                tmp = np.append(tmp, x_lq_pad[j])
                tmp = np.append(tmp, x_lq_pad[j+1])
            input = torch.from_numpy(np.reshape(tmp, (1, 1, 1, length*3)))
            output_0_5k = unet(input.to(device='cuda', dtype=torch.float))
            output_0_5k = output_0_5k.cpu()
            output_0_5k = output_0_5k.data.numpy()
            output_0_5k = output_0_5k.flatten()
            x_pq = np.concatenate((x_pq, output_0_5k[length:length*2]))

        x_pq = x_pq.flatten()

        librosa.output.write_wav(output_dir + name + "_output_{}.wav".format(args.mode), x_pq, hp.sr)
