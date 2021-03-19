# Experiment for optimizer comparison

import pickle
import numpy as np
import matplotlib.pyplot as plt
from args import init_arguments
from utils import smooth_curve
from main import main

if __name__ == '__main__':
    model = 'TwoLayerNet'
    args = init_arguments().parse_args()
    args.models = [model]
    
    d_train_loss = {}
    for optimizer in ['SGD', 'Momentum', 'AdaGrad', 'Adam']:
        args.optimizer = optimizer
        print('------------------', args.optimizer, '------------------\n')
        trainer = main(args, feature_type='SURF', return_trainer=True)[model]
        d_train_loss[optimizer] = trainer.train_loss
    
    fn = trainer.fn.replace('Accuracy', 'loss_optimizers').replace('.txt', '.pkl')
    with open(fn, 'wb') as f:
        pickle.dump(fn, f, pickle.HIGHEST_PROTOCOL)

    plt.figure(figsize=(8,6))
    for optimizer in ['SGD', 'Momentum', 'AdaGrad', 'Adam']:
        x = np.arange(len(d_train_loss[optimizer]))
        plt.plot(x, smooth_curve(d_train_loss[optimizer]), label=optimizer)
    plt.title('Plot of Training Loss of ' + model + ' with Different Optimizers')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(fn.replace('.pkl', '.png'))
    print('The plot of training loss is saved as', fn)