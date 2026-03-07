from sklearn import metrics as skmetrics
from IPython.display import clear_output
from map_tool_box.modules import Utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch
import copy
import math

# splits into train, validation, and test sets for each fold for n-fold cross-validation
# train_sample_names will never be in the test fold and only used for trainin/validation
# test_sample_names will be mixed into folds for training/validation
# truths_dictionary is a map of {sample_name: [truth_1, ..., truth_n]} 
# n_folds is the number of folds to evenly split test sample names up
    # these will be attempted to be split enenly amoung the truths, not guarenteed for multilabel
# shuffle=True will randomize order
# validation_percent will be the percent of train and mixed in test samples used for early stopping 
    # validation_percent=0 to not use validation
# skip_truths is a list of truths to not put in validation set (in case of sparse classes)
def cross_validation_folds(all_train_sample_names, all_test_sample_names, truths_dictionary, n_folds,
               shuffle=False, validation_percent=0, skip_truths=[]):

    # copy lists for safe keeping
    all_train_sample_names = all_train_sample_names.copy()
    all_test_sample_names = all_test_sample_names.copy()
    
    # make folds dictionary to fill with sample names
    folds_dictionary = {fold:{'train_sample_names':[], 'validation_sample_names':[], 'test_sample_names':[]} for fold in range(n_folds)}

    # get our unique set of positive ground truth labels
    unique_truths = get_unique_truths(all_test_sample_names, truths_dictionary)
    
    # get the label distribution 
    truth_counts = {truth:get_truth_count(truth, all_test_sample_names, truths_dictionary) for truth in unique_truths}
    
    # sort in ascending order of label count
    truth_counts = dict(sorted(truth_counts.items(), key=lambda item: item[1]))

    # add all sample names to each train set
    for fold in range(n_folds):
        folds_dictionary[fold]['train_sample_names'] = all_train_sample_names.copy() + all_test_sample_names.copy()

    # randomize order
    if shuffle:
        random.shuffle(all_test_sample_names)
        
    # fill the list of test_sample names for each fold
    # start with the lowest label count
    for truth in truth_counts:

        # get all sample names positive for this label
        positive_sample_names = []
        temp_sample_names = all_test_sample_names.copy()
        for sample_name in temp_sample_names:
            truths = truths_dictionary[sample_name]
            if truth in truths:
                positive_sample_names.append(sample_name)
                all_test_sample_names.remove(sample_name)

        # add one test sample at a time to each fold until no more left
        # and remove from the train set at that fold
        idx = 0
        while(idx < len(positive_sample_names)):
            for fold in range(n_folds):
                if idx >= len(positive_sample_names):
                    break
                sample_name = positive_sample_names[idx]
                folds_dictionary[fold]['test_sample_names'].append(sample_name)
                folds_dictionary[fold]['train_sample_names'].remove(sample_name)
                idx += 1

    # now seperate validation 
    if validation_percent > 0:
        for fold in range(n_folds):
            fold_train_sample_names = folds_dictionary[fold]['train_sample_names'].copy()

            # randomize order
            if shuffle:
                random.shuffle(fold_train_sample_names)

            # max number of validation sample names
            max_val = max(1, int(validation_percent*len(fold_train_sample_names)))
        
            # get the label distribution 
            truth_counts = {truth:get_truth_count(truth, fold_train_sample_names, truths_dictionary) for truth in unique_truths}

            # randomly sample from training names while meeting criteria
            N_val = 0
            max_truths = {truth:max(1, int(validation_percent*truth_counts[truth])) for truth in unique_truths}
            active_truths = {truth:0 for truth in unique_truths}
            for i in range(len(fold_train_sample_names)):
                sample_name = fold_train_sample_names[i]
                truths = truths_dictionary[sample_name]
                valid = True
                for truth in truths:
                    if active_truths[truth] >= max_truths[truth] or truth in skip_truths:
                        valid = False
                        break
                if valid:
                    folds_dictionary[fold]['validation_sample_names'].append(sample_name)
                    folds_dictionary[fold]['train_sample_names'].remove(sample_name)
                    for truth in truths:
                        active_truths[truth] += 1
                    N_val += 1
                    if N_val >=  max_val:
                        break

    return folds_dictionary


# splits into train, validation, and test sets for each fold for leave-one-out
# train_sample_names will never be in the test fold and only used for trainin/validation
# test_sample_names will be mixed into folds for training/validation
# truths_dictionary is a map of {sample_name: [truth_1, ..., truth_n]} 
# the number of folds is equal to the size of the test set -- placeing one test sample in each test fold
# shuffle=True will randomize order
# validation_percent will be the percent of train and mixed in test samples used for early stopping 
    # validation_percent=0 to not use validation
# skip_truths is a list of truths to not put in validation set (in case of sparse classes)
def leave_one_out_folds(all_train_sample_names, all_test_sample_names, truths_dictionary,
               shuffle=False, validation_percent=0, skip_truths=[]):

    # copy lists for safe keeping
    all_train_sample_names = all_train_sample_names.copy()
    all_test_sample_names = all_test_sample_names.copy()

    # get number of folds in leave-one-out
    n_folds = len(all_test_sample_names)
    
    # make folds dictionary to fill with sample names
    folds_dictionary = {fold:{'train_sample_names':[], 'validation_sample_names':[], 'test_sample_names':[]} for fold in range(n_folds)}

    # get our unique set of positive ground truth labels
    unique_truths = get_unique_truths(all_test_sample_names, truths_dictionary)

    # add all sample names to each train set
    for fold in range(n_folds):
        folds_dictionary[fold]['train_sample_names'] = all_train_sample_names.copy() + all_test_sample_names.copy()
        
    # fill the list of test_sample names for each fold
    for fold in range(n_folds):
        sample_name = all_test_sample_names[fold]
        folds_dictionary[fold]['test_sample_names'].append(sample_name)
        folds_dictionary[fold]['train_sample_names'].remove(sample_name)

    # now seperate validation 
    if validation_percent > 0:
        for fold in range(n_folds):
            fold_train_sample_names = folds_dictionary[fold]['train_sample_names'].copy()

            # randomize order
            if shuffle:
                random.shuffle(fold_train_sample_names)

            # max number of validation sample names
            max_val = max(1, int(validation_percent*len(fold_train_sample_names)))
        
            # get the label distribution 
            truth_counts = {truth:get_truth_count(truth, fold_train_sample_names, truths_dictionary) for truth in unique_truths}

            # randomly sample from training names while meeting criteria
            N_val = 0
            max_truths = {truth:max(1, int(validation_percent*truth_counts[truth])) for truth in unique_truths}
            active_truths = {truth:0 for truth in unique_truths}
            for i in range(len(fold_train_sample_names)):
                sample_name = fold_train_sample_names[i]
                truths = truths_dictionary[sample_name]
                valid = True
                for truth in truths:
                    if active_truths[truth] >= max_truths[truth] or truth in skip_truths:
                        valid = False
                        break
                if valid:
                    folds_dictionary[fold]['validation_sample_names'].append(sample_name)
                    folds_dictionary[fold]['train_sample_names'].remove(sample_name)
                    for truth in truths:
                        active_truths[truth] += 1
                    N_val += 1
                    if N_val >=  max_val:
                        break

    return folds_dictionary

def get_truth_count(truth, sample_names, truths_dictionary):
    truth_count = 0
    for sample_name in sample_names:
        truths = truths_dictionary[sample_name]
        if truth in truths:
            truth_count += 1
    return truth_count

def get_unique_truths(sample_names, truths_dictionary):
    unique_truths = []
    for sample_name in sample_names:
        truths = truths_dictionary[sample_name]
        for truth in truths:
            if truth not in unique_truths:
                unique_truths.append(truth)
    return unique_truths

def summarize_data(sample_names, truths_dictionary, mode_data_dictionary):
    mode_names = list(mode_data_dictionary.keys())
    mode_keys = [f'# {mode_name} observations' for mode_name in mode_names]
    data_dict = {'':['# samples']+mode_keys,'total':[0]*(len(mode_names)+1)}
    for sample_name in sample_names:
        data_dict['total'][0] += 1
        for idx, mode_name in enumerate(mode_names):
            if sample_name not in mode_data_dictionary[mode_name]:
                continue
            data_dict['total'][idx+1] += len(mode_data_dictionary[mode_name][sample_name])
        truths = truths_dictionary[sample_name]
        for truth in truths:
            if truth not in data_dict:
                data_dict[truth] = [0]*(len(mode_names)+1)
            data_dict[truth][0] += 1
            for idx, mode_name in enumerate(mode_names):
                if sample_name not in mode_data_dictionary[mode_name]:
                    continue
                data_dict[truth][idx+1] += len(mode_data_dictionary[mode_name][sample_name])
    dataframe = pd.DataFrame.from_dict(data_dict)
    dataframe.set_index('', inplace=True)
    display(dataframe)

# summarizes the counts of data based on truths train/val/test
def summarize_split(data_dictionary, truths_dictionary, train_sample_names, validation_sample_names, test_sample_names):
    set_names = ['all', 'train', 'validation', 'test']
    data_dict = {'data split':set_names, 'sample count':[0,0,0,0]}
    all_sample_names = train_sample_names + validation_sample_names + test_sample_names
    for i, sample_names in enumerate([all_sample_names, train_sample_names, validation_sample_names, test_sample_names]):
        set_name = set_names[i]
        data_dict['sample count'][i] = len(sample_names)
        for sample_name in sample_names:
            truths = truths_dictionary[sample_name]
            for truth in truths:
                if truth not in data_dict:
                    data_dict[truth] = [0, 0, 0, 0]
                data_dict[truth][i] += 1
    dataframe = pd.DataFrame.from_dict(data_dict)
    dataframe.set_index('data split', inplace=True)
    display(dataframe)

# PyTorch custom MyDataset needed for dataloaders to feed data into neural network models during both training and inference
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y=None, sample_size=None):
        self.X = X
        self.Y = Y
        if sample_size is None:
            sample_size = len(X)
        self.sample_size = sample_size
    def __getitem__(self, index):
        x = self.X[index].astype(np.float32)
        if self.Y is None:
            return x
        y = self.Y[index].astype(np.float32)
        return x, y
    def __len__(self):
        return self.sample_size

# make PytTorch DataLoader object
def make_dataloader(X, Y=None, batch_size=32, shuffle=False, drop_last=False, num_workers=0, pin_memory=False, sample_size=None):
    # convert to torch DataLoader
    data_set = MyDataset(X, Y, sample_size)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, 
                                              num_workers=num_workers, pin_memory=pin_memory)
    return data_loader

# makes a pytorch MLP block  (a list of modules)
    # layers is the number of nodes in [input, hidden1, ..., hiddenN, output] so must have a length of atleast 2
    # dropout is None to not use or a list with nodes to drop in each layer above (not including output)
def mlp_modules(layers, dropout=None, hid_activation=torch.nn.ELU, out_activation=torch.nn.Sigmoid, with_bias=True):
    modules = []
    for idx in range(len(layers)-1):
        if dropout is not None and dropout[idx] > 0:
            modules.append(torch.nn.Dropout(dropout[idx]))
        modules.append(torch.nn.Linear(layers[idx], layers[idx + 1], bias=with_bias))
        if idx < len(layers)-2:
            modules.append(hid_activation())
    if out_activation is not None:
        modules.append(out_activation())
       
    return modules

# creates a pytorch MLP model
def mlp(layers, dropout=None, hid_activation=torch.nn.ELU, out_activation=torch.nn.Sigmoid, with_bias=True):
    modules = mlp_modules(layers, dropout, hid_activation, out_activation, with_bias)
    model = torch.nn.Sequential(*modules)
    
    return model

# creates several blocks of conv>activation>pool, followed by fully connected MLP
def cnn(block_layers):
    
    # bulid CNN blocks
    blocks = []
    for i in range(len(block_layers)):
        block = []
        for j in range(len(block_layers[i])):
            layer_func = block_layers[i][j][0]
            if layer_func is None:
                continue
            if len(block_layers[i][j]) == 1:
                layer = layer_func()
            else:
                layer_params = block_layers[i][j][1]
                layer = layer_func(**layer_params)
            if isinstance(layer, list):
                block = block + layer
            else:
                block.append(layer)
        blocks.append(torch.nn.Sequential(*block))

    # make model
    model = torch.nn.Sequential(*blocks)
    
    return model
    
# scale between zero and one
# standardize to zero mean, unit variance
def preprocess_RGB(X):
    X = X/255
    X = (X - 0.449)/0.226
    return X
    
# scale between zero and one
def preprocess_grey(X):
    X = X/255
    return X
    
# scale between zero and 255 and cast as uint8
def postprocess_grey(X):
    X = X*255
    X = X.astype(np.uint8)
    return X

def add_gaussian_noise(x, percent_level):
    with_noise = torch.normal(x, torch.abs(percent_level*x))
    return with_noise

# forward pass through a PyTorch neural network
# n_iterations=0 will go through all iterations otherwise will stop at the given number
def forward(model, data_loader, device, n_iterations=0,
            criterion=None, with_grad=False, memory_saver=True, return_predictions=False, return_losses=False, 
            x_preproc_funcs=None, x_preproc_paramss=None, y_preproc_funcs=None, y_preproc_paramss=None):
    predictions = []
    losses = []

    # mini-batch iterations
    for iteration, data in enumerate(data_loader):
        if n_iterations > 0 and iteration >= n_iterations:
            break
        x, y = data

        # process x
        if x_preproc_funcs is not None:
            for i in range(len(x_preproc_funcs)):
                x_preproc_func = x_preproc_funcs[i]
                x_preproc_params = x_preproc_paramss[i]
                x = x_preproc_func(x, **x_preproc_params)

        # process y
        if y_preproc_funcs is not None:
            for i in range(len(y_preproc_funcs)):
                y_preproc_func = y_preproc_funcs[i]
                y_preproc_params = y_preproc_paramss[i]
                y = y_preproc_func(y, **y_preproc_params)

        # forward prop while calculating gradient
        if with_grad:
            model.optimizer.zero_grad()
            p = model(x.to(device=device))
            loss = criterion(p, y.to(device=device))
            loss.backward()
            model.optimizer.step()
            realized_loss = float(loss.detach().cpu())

        # forward prop with no gradient
        else:
            with torch.no_grad():
                p = model(x.to(device=device))
                if return_losses:
                    loss = criterion(p, y.to(device=device))
                    realized_loss = float(loss.detach().cpu())

        # save ouputs as requested
        if return_predictions:
            predictions.append(p.detach().cpu().numpy())
        if return_losses:
            losses.append(realized_loss)
        if memory_saver:
            del x, y, p # clear mem from gpu

    # aggregate outputs as requested
    if return_predictions and return_losses:
        return np.vstack(predictions), losses
    if return_predictions:
        return np.vstack(predictions)
    if return_losses:
        return losses

# read and write torch model training checkpoints
def read_checkpoint(input_dir, model, lr_scheduler):
    checkpoint = torch.load(input_dir + 'torch_ckpt.pt')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    model.optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    best_weights = checkpoint['best_weights']
    wait = checkpoint['wait']
    best_loss = checkpoint['best_loss']
    train_times = checkpoint['train_times']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    return start_epoch, best_weights, wait, best_loss, train_times, train_losses, val_losses
def write_checkpoint(output_dir, epoch, model, lr_scheduler, best_weights, wait, best_loss, train_times, train_losses, eval_losses):
    torch.save({
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer' : model.optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
        'best_weights': best_weights,
        'wait': wait,
        'best_loss': best_loss,
        'train_times': train_times,
        'train_losses': train_losses,
        'eval_losses': eval_losses,
    }, output_dir + 'torch_ckpt.pt')

# plt.plot() learning curve that shows training/validation losses against epoch
def show_learning_curve(train_losses, eval_losses, clear_screen=False, 
                        train_label='train', ylabel='Loss', xlabel='Iteration'):
    if clear_screen:
        clear_output()
    plt.plot(train_losses, label=train_label)
    for key in eval_losses:
        plt.plot(eval_losses[key], label=key)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# casts a list of strings into 0s and 1s for false or true for each string
def one_hot_encode(list_of_strings, all_possible_strings):
    one_hot_encoded = [(1 if value in list_of_strings else 0) for value in all_possible_strings]
    return one_hot_encoded

# training loop for neural network model
# dataloaders is a dictionary of data to evaluate at each epoch
    # this requires an entry dataloaders[train_dataloader] which will be used for the gradient during backprop
    # specify check_dataloader=key to check for stopping criteria at each epoch, assumes 'validation' key
# evaluate_each_epoch=True will evaluate on each epoch (fully iterated through training) by default, False will evaluate on each mini batch iteration
# train and eval x/y funcs and perams are optional lists of functions to call during forward prop through network
def train(model_func, model_params, optimizier_func, optimizer_params, dataloaders, 
        train_dataloader='train', check_dataloader='validation',
        device='cpu', criterion=torch.nn.MSELoss(), evaluate_each_epoch=True,
        return_best_weights=True, stopping_criteria='early', eval_loss_tolerance=1e-4, patience=10, max_epochs=10_000, 
        show_curve_freq=10, continue_training=False,
        pytorch_threads=1, checkpoint_freq=0, input_dir=None, output_dir=None, print_progress=False,
        lr_scheduler_func=None, lr_scheduler_params={},
        train_x_preproc_funcs=None, train_x_preproc_paramss=None, train_y_preproc_funcs=None, train_y_preproc_paramss=None,
        eval_x_preproc_funcs=None, eval_x_preproc_paramss=None, eval_y_preproc_funcs=None, eval_y_preproc_paramss=None,
        forward_function=forward, forward_kwargs={},
     ):
    
    # forward pass of neural network with training data (update gradient)
    def _forward_train(dataloader, update_gradient=True):
        model.train()
        if evaluate_each_epoch:
            n_iterations = 0
        else:
            n_iterations = 1
        losses = forward_function(model, dataloader, device, n_iterations=n_iterations, criterion=criterion, with_grad=update_gradient, return_losses=True,
                         x_preproc_funcs=train_x_preproc_funcs, x_preproc_paramss=train_x_preproc_paramss,
                         y_preproc_funcs=train_y_preproc_funcs, y_preproc_paramss=train_y_preproc_paramss,
                         **forward_kwargs)
        return np.mean(losses)

    # forward pass of neural network with validation data (do not update gradient)
    def _forward_eval(dataloader):
        model.eval()
        losses = forward_function(model, dataloader, device, criterion=criterion, return_losses=True,
                         x_preproc_funcs=eval_x_preproc_funcs, x_preproc_paramss=eval_x_preproc_paramss,
                         y_preproc_funcs=eval_y_preproc_funcs, y_preproc_paramss=eval_y_preproc_paramss,
                         **forward_kwargs)
        return np.mean(losses)

    # set torch number of threads
    torch.set_num_threads(pytorch_threads)

    # make model and other init vars
    model = model_func(**model_params)
    model.to(device)

    # make optimizer
    optimizer_params['params'] = model.parameters()
    model.optimizer = optimizier_func(**optimizer_params)
    
    # make lr scheduler
    lr_scheduler = None
    if lr_scheduler is not None:
        lr_scheduler_params['optimizer'] = model.optimizer
        lr_scheduler = lr_scheduler_func(**lr_scheduler_params)
    
    # load state_dict or reset training?
    if continue_training:
       start_epoch, best_weights, wait, best_loss, train_times, train_losses, eval_losses = read_checkpoint(input_dir, model, lr_scheduler)
    else:
        best_weights = copy.deepcopy(model.state_dict())
        wait = 0
        start_epoch = 1
        train_losses = [_forward_train(dataloaders[train_dataloader], update_gradient=False)]
        eval_losses = {key:[] for key in dataloaders if key not in [train_dataloader]}
        for key in eval_losses:
            eval_losses[key].append(_forward_eval(dataloaders[key]))
        best_loss = eval_losses[check_dataloader][0]
        train_times = []

    # training loop
    if start_epoch < max_epochs:
        for epoch in range(start_epoch, max_epochs+1):
            if stopping_criteria == 'early' and wait > patience:
                break
            sw = Utils.Stopwatch()

            # forward passes of neural network
            train_loss = _forward_train(dataloaders[train_dataloader], update_gradient=True)
            train_losses.append(train_loss)
            for key in eval_losses:
                eval_losses[key].append(_forward_eval(dataloaders[key]))

            # check best weights and learning convergence 
            eval_loss = eval_losses[check_dataloader][-1]
            
            if eval_loss + eval_loss_tolerance < best_loss:
                best_loss = eval_loss
                best_weights = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1

            # benchmark execution time for this epoch
            epoch_time = sw.stop()
            train_times.append(epoch_time)

            # log progress
            progress_str = f'epoch {epoch} time {epoch_time:0.2} best_loss {best_loss:0.4}'
            if print_progress:
               print(progress_str)
            if show_curve_freq > 0 and epoch % show_curve_freq == 0:
                show_learning_curve(train_losses, eval_losses=eval_losses, clear_screen=True, 
                                    xlabel=('Epoch' if evaluate_each_epoch else 'Iteration') )

            # checkpoint
            if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
                write_checkpoint(output_dir, epoch, model, lr_scheduler, best_weights, wait, best_loss, train_times, train_losses, eval_losses)

            # advance learning rate scheduler
            if lr_scheduler is not None:
                lr_scheduler.step()

    # load best weights
    if return_best_weights:
        model.load_state_dict(best_weights)

    return model, train_losses, eval_losses, train_times

def get_predictions(model, dataloader, device='cpu', 
                    x_preproc_funcs=None, x_preproc_paramss=None, 
                    y_postproc_funcs=None, y_postproc_paramss=None,
                   forward_func=forward, forward_kwargs={}):

    # propigate X forward through neural network
    predictions = forward_func(model, dataloader, device, return_predictions=True, 
                          x_preproc_funcs=x_preproc_funcs, x_preproc_paramss=x_preproc_paramss,
                         **forward_kwargs)
    
    # process predictions
    predictions = np.array(predictions)
    if y_postproc_funcs is not None:
        for i in range(len(y_postproc_funcs)):
            y_postproc_func = y_postproc_funcs[i]
            y_postproc_params = y_postproc_paramss[i]
            predictions = y_postproc_func(predictions, **y_postproc_params)
            
    return predictions

def optimize_cutoffs(predictions, truths, metric=skmetrics.balanced_accuracy_score, show_cutoffs=False, class_names=None):
    optimized_cutoffs = []
    for c in range(predictions.shape[1]):
        class_predictions = predictions[:, c].copy()
        class_predictions_list = class_predictions.tolist()
        class_predictions_list.sort()
        class_predictions = np.expand_dims(class_predictions, axis=1)
        class_truths = np.expand_dims(truths[:, c], axis=1)
        class_cutoffs = {}
        best_cutoff = 0
        best_cutoff_eval = 0
        for idx in range(len(class_predictions_list)-1):
            cutoff = (class_predictions_list[idx+1] + class_predictions_list[idx]) / 2
            classifications = get_classifications(class_predictions, [cutoff])
            evaluation = metric(class_truths, classifications)
            class_cutoffs[cutoff] = evaluation
            if evaluation > best_cutoff_eval:
                best_cutoff = cutoff
                best_cutoff_eval = evaluation
        optimized_cutoffs.append(best_cutoff)
        if show_cutoffs:
            plt.plot([cutoff for cutoff in class_cutoffs], [class_cutoffs[cutoff] for cutoff in class_cutoffs])
            if class_names is None:
                plt.title(f'varying cutoff values for class #{c}')
            else:
                plt.title(f'varying cutoff values for class {class_names[c]}')
            plt.xlabel('cutoff')
            plt.ylabel('metric')
            plt.show()
    return optimized_cutoffs

def get_classifications(predictions, cutoffs):
    classifications = []
    for y in predictions:
        classification = [0 for c in cutoffs]
        for c in range(len(cutoffs)):
            cutoff = cutoffs[c]
            if y[c] >= cutoff:
                classification[c] = 1
        classifications.append(classification)
    return np.array(classifications)
    
# return both the total metric and that for each seperate label
def multilabel_evaluations(truths, classifications, class_names, metric):
    evaluations = {}
    evaluations['total'] = metric(truths.flatten(), classifications.flatten())
    for c in range(truths.shape[1]):
        evaluations[class_names[c]] = metric(truths[:, c].flatten(), classifications[:, c].flatten())
    return evaluations
    
# shows to screen the accuracies object returned from multilabel_accuracy()
def display_multilabel_accuracy(accuracies):
    data_dict = {name:[accuracies[name]] for name in accuracies}
    pdf = pd.DataFrame.from_dict(data_dict)
    pdf = pdf.style.hide(axis='index')
    display(pdf)