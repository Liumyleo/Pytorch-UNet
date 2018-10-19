class Hyperparams:

    # h5_path_train_rdn = './lists-h5/rdn/yy_record_data_600min_rdn-train.h5'     #600min
    # h5_path_train_unet = './lists-h5/unet/yy_record_data_600min_unet-train.h5'     #600min
    # h5_path_train_unet = '../datasets/8k-16k_h5/2048-all/lists-h5/yy_record_data_600min-train.h5' # 2048
    # h5_path_train_unet = '../99-14_keras_merge_model/lists-h5/yy_record_data_600min-train.h5' # 8192
    # h5_path_train_unet = '/data/share/liumingyu/lists-h5_lizhi/yy_record_data_600min-train.h5' # 8192
    h5_path_train_unet='../1-4_leo_wav/lists-h5/yy_record_data_8-16-train.h5'
    # h5_path_train_unet = '../datasets/5k-5k/lists-h5/yy_record_data_600min-train.h5'
    win_length = 8192
    sr = 32000
    n_filters_student = [16, 32, 64, 32, 16]
    n_filters_teacher = [128, 256, 512, 512, 512, 256, 128]
    batch_size = 64

    model_path = './model'
    test_data_path = './data/test-data/'
    groundtruth_data_path = './data/groundtruth/'
    output_dir = './model_output/'
    realtime_output_dir = './realtime_output'

    multi_gpu = True
    save_real_time_result = True
