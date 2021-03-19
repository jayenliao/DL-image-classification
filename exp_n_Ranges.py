# Experiment of number of bars in the global color histogram¶

import pickle
import numpy as np
import matplotlib.pyplot as plt
from args import init_arguments
from utils import smooth_curve
from main import main

if __name__ == '__main__':
    model = 'TwoLayerNet'
    feature_type = 'Histogram'

    parser = init_arguments()
    parser.add_argument('--n_Ranges_list', nargs='+', type=int, default=[1, 5, 10, 20])
    parser.add_argument('--plot_figsize_', nargs='+', type=int, default=(12, 10))
    args = parser.parse_args()
    args.models = [model]
    
    d_train_loss = {}
    for n_Ranges in args.n_Ranges_list:
        print('----> n_Ranges =', n_Ranges, '\n')
        args.n_Ranges = n_Ranges
        trainer = main(args, feature_type=feature_type, return_trainer=True)[model]
        d_train_loss[n_Ranges] = trainer.train_loss

    fn = trainer.fn.replace('Accuracy', 'dict_loss_n_Ranges').replace('.txt', '.pkl')
    with open(fn, 'wb') as f:
        pickle.dump(d_train_loss, f, pickle.HIGHEST_PROTOCOL)
    

    plt.figure(figsize=args.plot_figsize_)
    for n_Ranges in args.n_Ranges_list:
        x = np.arange(len(d_train_loss[n_Ranges]))
        plt.plot(x, smooth_curve(d_train_loss[n_Ranges]), label=n_Ranges)
    plt.title('Plot of Training Loss of ' + model + ' with Different Numbers of Bin Cuts')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(fn.replace('.pkl', '.png'))
    print('The plot of training loss is saved as', fn)
    