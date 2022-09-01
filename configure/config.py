import argparse

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', default=False, help='Whether model is training')
    parser.add_argument('--save_model_freq', default=100, type=int, help='How many epochs to save model')
    parser.add_argument('--model_path', default='saved/models')
    parser.add_argument('--log_path', default='saved/logs/')

    parser.add_argument('--root_train_dir', default='')
    parser.add_argument('--s', '--server', default='/data1/weipeng/Data/ISIC2019/ISIC_2019_Training_Input/')

    parser.add_argument('--batch_size', default=36, type=int, help='Mini batch size for stochastic gradient descent algorithms.')
    parser.add_argument('--epochs', default=5000, type=int, help='Number of epochs')
    parser.add_argument('--lr_rate', default=1e-3, type=float, help='Learning rate for training process')
    parser.add_argument('--lr_decay', default=1e-5, type=float, help='Learning rate decay for training process')
    parser.add_argument('--optimizer', default='sgd', type=str, help='Optimizer type')

    parser.add_argument('--is_gpu', default=True, help='Whether use gpu')
    parser.add_argument('--start_gpu_idx', default=0, help='Used gpu start index')
    parser.add_argument('--end_gpu_idx', default=0, help='Used gpu end index')
    parser.add_argument('--gpu_cnt', default=2, help='Used the count of gpus ')

    parser.add_argument('--class_nums', default=7, type=int, help='The number of class types')
    args = parser.parse_args()
    return args

