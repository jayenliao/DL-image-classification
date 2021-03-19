'''
Deep Learning - HW1: Image Classification
Jay Liao (re6094028@gs.ncku.edu.tw)

----------------------------------------

Procedures:

1. Load in the files
2. Extract features
3. Train and evaluate models
    - rf
    - xgb
    - perceptron
    - NN
4. Test models and compare

----------------------------------------
'''
import time
import numpy as np
import pandas as pd
from args import init_arguments
from utils import *
from feature_extraction import *
from trainers import *

def main(args, feature_type, return_trainer=False):
    PATH = args.dataPATH if args.dataPATH[-1] == '/' else args.dataPATH + '/'
    fn = PATH + 'X_' + feature_type + '_'
    if feature_type == 'Histogram':
        fn += str(args.n_Ranges) + '_'
    print('======= Feature type:', feature_type, '======= ')
    try:
        print('Loading preprocessed datasets ...')
        X_tr, y_tr = np.load(fn + 'tr.npy', allow_pickle=True), load_original_data('train', PATH)[1]
        X_va, y_va = np.load(fn + 'va.npy', allow_pickle=True), load_original_data('val', PATH)[1]
        X_te, y_te = np.load(fn + 'te.npy', allow_pickle=True), load_original_data('test', PATH)[1]
    
    except:
        print('Loading the original images ...')
        t0 = time.time()
        img_list_tr, fn_list_tr, y_tr = load_images('train', PATH)
        img_list_va, fn_list_va, y_va = load_images('val', PATH)
        img_list_te, fn_list_te, y_te = load_images('test', PATH)
        tdiff = time.time() - t0
        print('Time cost of loading: %8.2fs' % tdiff)
        
        print('\nExtracting features ...')
        t0 = time.time()
        X_tr = get_features_for_images(img_list_tr, feature_type, args.n_Ranges)
        X_va = get_features_for_images(img_list_va, feature_type, args.n_Ranges)
        X_te = get_features_for_images(img_list_te, feature_type, args.n_Ranges)
        tdiff = time.time() - t0
        print('Time cost of feature extraction: %8.2fs' % tdiff)
        
        if feature_type == 'Histogram':
            fn += str(args.n_Ranges) + '_'
        np.save(fn + 'tr.npy', X_tr)
        np.save(fn + 'va.npy', X_va)
        np.save(fn + 'te.npy', X_te)

    print('\nShapes of feature matrices (Train | Val | Test):', end='  ')
    print(X_tr.shape, X_va.shape, X_te.shape)
    y_onehot_te = one_hot_transformation(y_te)

    d_trainers = {}
    for model in args.models:
        if model in ['rf', 'xgb']:
            args.save_trainer = False
            trainer = NonNNtrainer(X_tr=X_tr, y_tr=y_tr, feature_type=feature_type, model_=model, n_jobs=args.n_jobs, random_state=args.seed)
            trainer.train(save_model=args.save_models, savePATH=args.savePATH, pretrained_model=args.pretrained_model)
            print('\nEvaluate the trained model on the testing set ...')
            print_accuracy(y_onehot_te, trainer.model.predict_proba(X_te))

        else:
            y_onehot_tr, y_onehot_va = one_hot_transformation(y_tr), one_hot_transformation(y_va)
            print('Shapes of y label matrices (Train | Val | Test):', end='  ')
            print(y_onehot_tr.shape, y_onehot_va.shape, y_onehot_te.shape)
            
            trainer = NNtrainer(X_tr, y_onehot_tr, feature_type, model, args.hidden_size, args.hidden_layer_act, args.output_layer_act, args.optimizer, args.lr, args.seed)
            trainer.train(batch_size=args.batch_size, epochs=args.epochs, save_trainer=args.save_trainer, save_model=args.save_models, savePATH=args.savePATH, X_va=X_va, y_va=y_onehot_va, print_result_per_epochs=args.print_result_per_epochs, pretrained_model=args.pretrained_model)
            
            print('\nEvaluate the trained model on the testing set ...')
            print_accuracy(y_onehot_te, trainer.model.predict(X_te))

            if trainer.trained:
                if model != 'NaivePerceptron':
                    trainer.plot_training('loss', args.plot_figsize)
                trainer.plot_training('accuracy', args.plot_figsize)
        print()

        if args.save_trainer:
            fn = trainer.fn.replace('Accuracy', 'Trainer').replace('.txt', '.pkl')
            trainer.X_tr, trainer.y_tr = None, None   # Remove the training data from the trainer to avoid saving a heavy file. 
            try:
                with open(fn, 'wb') as f:
                    pickle.dump(trainer, f, pickle.HIGHEST_PROTOCOL)
                print('\nThe trainer is saved as', fn)
            except:
                print('\nThe trainer fails to be saved!!!!! (；ﾟДﾟ；)')

        if return_trainer:
            d_trainers[model] = trainer

    if return_trainer:
        return d_trainers

if __name__ == '__main__':
    args = init_arguments().parse_args()
    for feature_type in args.feature_types:
        main(args, feature_type)