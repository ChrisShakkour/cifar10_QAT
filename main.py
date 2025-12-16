import logging
from pathlib import Path
import gc

import torch as t
import yaml

import process
import quan
import util
from model import create_model
from quan.quantizer import lsq
import matplotlib.pyplot as plt
import numpy as np
from quan.quantizer.lsq import *
from functools import partial
import math
import random
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from packaging.version import parse, Version
from util.gdtuo import *
# dump_once_per_run_py37_logs.py
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

Number = Union[int, float]

# ---------------------------------------------------------------------------
# Globals that make the “one file per Python run” rule work
_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
_FILE_HANDLE: Optional[Path] = None
# ---------------------------------------------------------------------------

def save_and_plot_losses(losses, filepath="losses.npy", png_path="loss_plot.png", y_label="Loss", title="Training Loss"):
    """
    Saves loss values to .npy file and plots the curve.
    """
    # save the list
    np.save(filepath, np.array(losses))
    # plot
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.show()

def plot_4_arrays_save(arr1, arr2, arr3, arr4, save_path="grid_plot.png", cmap="viridis", titles=None):
    """
    Plots 4 list-based arrays on a 2x2 grid and saves the plot.
    
    Args:
        arr1, arr2, arr3, arr4: 2D lists or numpy-like nested lists
        save_path: path to save the resulting plot (e.g., 'plot.png')
        cmap: matplotlib colormap
        titles: optional list of 4 strings for subplot titles
    """

    # Convert to numpy arrays
    arrays = [np.array(a) for a in [arr1, arr2, arr3, arr4]]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for ax, arr, idx in zip(axes.flatten(), arrays, range(4)):
        im = ax.imshow(arr, cmap=cmap)
        if titles:
            ax.set_title(titles[idx])
        ax.axis("off")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {save_path}")

def dump_three_lists(
    a: List[Number],
    b: List[Number],
    c: List[Number],
    name_a: str,
    name_b: str,
    name_c: str,
    path: Union[str, Path],
    base_name: str = "data",
    log_params: Optional[Dict[str, Any]] = None,  # <‑‑ NEW
) -> Path:
    """
    Write three numeric lists to *one* TXT file per Python run.
    Subsequent calls in the same run overwrite the file.

    Parameters
    ----------
    log_params : dict, optional
        If given, each key–value pair is written at the top of the file
        as "key: value" before the column headers.
    """
    global _FILE_HANDLE

    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)

    if _FILE_HANDLE is None:                           # first call this run
        _FILE_HANDLE = path / f"{base_name}_{_RUN_ID}.txt"

    if _FILE_HANDLE.exists():                          # overwrite on repeat
        _FILE_HANDLE.unlink()

    # Normalise list lengths
    max_len = max(len(a), len(b), len(c))
    pad = lambda lst: lst + [""] * (max_len - len(lst))
    a_p, b_p, c_p = map(pad, (a, b, c))

    # ---------------- write file -----------------
    with _FILE_HANDLE.open("w", encoding="utf-8") as f:
        # 1. Optional run‑level parameters
        if log_params:
            for k, v in log_params.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")                # blank line before the table

        # 2. Column headers
        f.write(f"{name_a}\t{name_b}\t{name_c}\n")

        # 3. Rows
        for x, y, z in zip(a_p, b_p, c_p):
            f.write(f"{x}\t{y}\t{z}\n")

    return _FILE_HANDLE

from pathlib import Path
from datetime import datetime

# -------------------------------------------------------------------
# Globals for one-file-per-run
_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
# -------------------------------------------------------------------

def init_results_file(
    path: Union[str, Path],
    base_name: str = "data_repeat",
    log_params: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Create/truncate one TXT file for this run, write run-level params
    and the three-column header.
    Returns the full path to the file.
    """
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / f"{base_name}_{_RUN_ID}.txt"
    with file_path.open("w", encoding="utf-8") as f:
        # 1) run-level parameters
        if log_params:
            for k, v in log_params.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
        # 2) column headers
        f.write("train_top1\ttest_top1\tC\n")

    return file_path

# ------------------------------------------------------------------
# Append one row of results to the file created by init_results_file.
def append_result(
    file_path: Union[str, Path],
    train_val: float,
    test_val: float,
    c_value: Any
) -> None:
    """
    Append one row of results to the file created by init_results_file.
    `c_value` can be a list, tuple, scalar, etc (it will be str()-ed).
    """
    with Path(file_path).open("a", encoding="utf-8") as f:
        f.write(f"{train_val}\t{test_val}\t{c_value}\n")

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    t.use_deterministic_algorithms(True)



seed = 0

#Which solution to use for the problem of depending on the future:
# -1 - regular STE 
# 0 - analytical soltuion for gSTE
# 1 - Delayed Updates
#1.5 Delayed Updates for Meta Network
# 2 - Updating all times together
# 3 - analytical solution for MetaNetwork
# 4 - STE towers
# 5 - analytical soltuion for gSTE in test framework
# 6 - Meta network with delayed updates in test framework
# 7 - Meta network with all times together
# 8 - check baseline which is initial quantization and then full precision training
# 9 - Training all times together with MAD instead of STE
#10 - Training all times together and then taking the last trained a value and keep training with its value without learning a
#11 - Training all times together for x epochs then taking the learned values and training them again all times together to find the following x epochs
#12 - less_greedy_Updates which is the method proposed in the Test by Daniel
num_solution = 11

#The learning rate   set used to train the a parameters
a_lr = 1e5
backward_for_test = "1 bit DoReFa with DualPWL"
#Decides how many diffrent a parameters for each weight
#0 - a per element, every element in the weight gets a repective a parameter
#1 - a per layer, every weight gets one a parameter
#2 - a per chnnel, every channel of the weight gets a respective a parameter
a_per=0

#if grouping together multiple a values to be a shared parameter to reduce memory consuption this sets the amount of parameter together each time else set 1
num_share_params=1

#In all time todgether training this sets the amount of epochs learned before startin learning from screatch
num_of_epochs_each_time = 1
num_epochs_div_repeats=1
def main():
    print("Num solution is : ",num_solution)
    set_random_seed(seed)
    script_dir = Path.cwd()
    args = util.get_config(default_file=script_dir / 'config.yaml')

    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    log_dir = util.init_logger(args.name, output_dir, script_dir / 'logging.conf')
    logger = logging.getLogger()

    with open(log_dir / "args.yaml", "w") as yaml_file:  # dump experiment config
        yaml.safe_dump(args, yaml_file)

    pymonitor = util.ProgressMonitor(logger)
    tbmonitor = util.TensorBoardMonitor(logger, log_dir)
    monitors = [pymonitor, tbmonitor]

    if args.device.type == 'cpu' or not t.cuda.is_available() or args.device.gpu == []:
        args.device.gpu = []
    else:
        available_gpu = t.cuda.device_count()
        for dev_id in args.device.gpu:
            if dev_id >= available_gpu:
                logger.error('GPU device ID {0} requested, but only {1} devices available'
                             .format(dev_id, available_gpu))
                exit(1)
        # Set default device in case the first one on the list
        t.cuda.set_device(args.device.gpu[0])
        # Enable the cudnn built-in auto-tuner to accelerating training, but it
        # will introduce some fluctuations in a narrow range.
        t.backends.cudnn.benchmark = True
        t.backends.cudnn.deterministic = False

    # Initialize data loader
    train_loader, val_loader, test_loader = util.load_data(args.dataloader)
    logger.info('Dataset `%s` size:' % args.dataloader.dataset +
                '\n          Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
                '\n        Validation Set = %d (%d)' % (len(val_loader.sampler), len(val_loader)) +
                '\n              Test Set = %d (%d)' % (len(test_loader.sampler), len(test_loader)))

    # Create the model

    import os, torch
    print("PID:", os.getpid(),
      " CUDA device:", torch.cuda.current_device(),
      " Name:", torch.cuda.get_device_name(0))
    
    print("torch.cuda.current_device()",torch.cuda.current_device())
    print("t.cuda.memory_summary",t.cuda.memory_summary(device=None, abbreviated=False))
    print("torch.cuda.mem_get_info()",torch.cuda.mem_get_info())
    model = create_model(args)
    

    T = len(train_loader)*num_of_epochs_each_time# A vector length for All times together solution (# of learning steps before update)
    list_for_lsq=[T, a_per,num_share_params]
    modules_to_replace = quan.find_modules_to_quantize(model, args.quan,num_solution, list_for_lsq)
    modules_to_replace_temp=dict(modules_to_replace)
    model = quan.replace_module_by_names(model, modules_to_replace)
    logger.info('Inserted quantizers into the original model')

    if args.device.gpu and not args.dataloader.serialized:
        model = t.nn.DataParallel(model, device_ids=args.device.gpu)

    model.to(args.device.type)

    start_epoch = 0
    if args.resume.path:
        model, start_epoch, _ = util.load_checkpoint(
            model, args.resume.path, args.device.type, lean=args.resume.lean)

    # Define loss function (criterion) and optimizer
    criterion = t.nn.CrossEntropyLoss().to(args.device.type)

    if num_solution == 0:
        pass
    #    main_analyticalgSTE(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir)
    elif  num_solution == -1:
        main_original(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir)
    #elif num_solution == 1 or num_solution == 1.5:
    #    main_DelayedUpdates(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir)
    elif num_solution == 2 or num_solution==7 or num_solution == 8 or num_solution == 9 or num_solution == 10 :
        main_all_times(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T,num_of_epochs_each_time)
    #elif num_solution == 5:
    #    main_analytical_all_time(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T)
    #elif num_solution == 6:
    #    main_DelayedUpdates_meta_network_all_time(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir)
    elif num_solution == 11:
        main_all_times_repeat(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T,num_of_epochs_each_time)
    elif num_solution == 12:
        main_all_times_less_greedy(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T,num_of_epochs_each_time)

def main_original(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir):

    # optimizer = t.optim.Adam(model.parameters(), lr=args.optimizer.learning_rate)
    optimizer = t.optim.SGD(model.parameters(),
                            lr=args.optimizer.learning_rate,
                            momentum=args.optimizer.momentum,
                            weight_decay=args.optimizer.weight_decay)
    lr_scheduler = util.lr_scheduler(optimizer,
                                     batch_size=train_loader.batch_size,
                                     num_samples=len(train_loader.sampler),
                                     **args.lr_scheduler)
    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % lr_scheduler)

    perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)
    t_top1_list=[]
    v_top1_list=[]
    test_list=[]
    if args.eval:
        process.validate(test_loader, model, criterion, -1, monitors, args)
    else:  # training
        if args.resume.path or args.pre_trained:
            logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
            top1, top5, _ = process.validate(test_loader, model, criterion,
                                             start_epoch - 1, monitors, args)
            perf_scoreboard.update(top1, top5, start_epoch - 1)
        temp_seed=0
        set_random_seed(temp_seed)

        for epoch in range(start_epoch, args.epochs):
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            t_top1, t_top5, t_loss = process.train(train_loader, model, criterion, optimizer,
                                                   lr_scheduler, epoch, monitors, args)
            random_seed=0
            #torch.manual_seed(random_seed)
            #recalibrate BN
            with t.no_grad():
                for inputs, _ in train_loader: 
                    outputs = model.forward(inputs)

            v_top1, v_top5, v_loss = process.validate(train_loader, model, criterion, epoch, monitors, args)
            #torch.manual_seed(random_seed)

            v_top1_2, v_top5_2, v_loss_2 = process.validate(test_loader, model, criterion, epoch, monitors, args)
            print("two vals : ",v_top1,v_top1_2)
            test_list.append(v_top1_2)
            t_top1_list.append(t_top1)
            v_top1_list.append(v_top1)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

            perf_scoreboard.update(v_top1, v_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)
            util.save_checkpoint(epoch, args.arch, model, {'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir)
            print("vont_top1_list : ",v_top1_list)
            print("t_top1_list : ",t_top1_list)
            print("test list is : ",test_list)
        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, model, criterion, -1, monitors, args)

    
    tbmonitor.writer.close()  # close the TensorBoard
    logger.info('Program completed successfully ... exiting ...')
    logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')


list_train = []
list_vont_top1 = []
last_train=None
prev_list_train=[]
def non_zero_statistics(tensor):
    # Filter out non-zero elements
    non_zero_elements = tensor[tensor != 0]
    
    # Calculate statistics
    mean = non_zero_elements.mean().item() if non_zero_elements.numel() > 0 else None
    std = non_zero_elements.std().item() if non_zero_elements.numel() > 0 else None
    min_val = non_zero_elements.min().item() if non_zero_elements.numel() > 0 else None
    max_val = non_zero_elements.max().item() if non_zero_elements.numel() > 0 else None
    count = non_zero_elements.numel()

    return {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'count': count
    }

loss_list = []

def main_all_times_repeat(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T,num_of_epochs_each_time):
    torch.backends.cudnn.deterministic = True
    
    v_top1_list=[]
    stats_list = []
    a_lr_t=a_lr
    load_from_prev = False
    if load_from_prev == True:
        model, start_epoch, _ = util.load_checkpoint(
            model, "/home/gild/Lsq_with_gSTE/out/MyProject_20241029-172911/MyProject_best.pth.tar",num_solution,modules_to_replace_temp,args.quan.excepts, args.device.type)
    if num_solution == 11:
        set_random_seed(seed)
        temp_seed=0
        start_epoch=0
        train_top1_list=[]
        test_top1_list = []
        train_loss = []
        test_loss = []

        run_cfg = {
        "a_learning_rate": a_lr,
        "num_solution": num_solution
        }
        results_file = init_results_file(
            path="/home/gild/Lsq_with_gSTE/out",         # or wherever you want to write
            base_name="data_repeat",
            log_params=run_cfg
        )

        for seg in range(0,num_epochs_div_repeats):
            print(seg," Segment of training, using the optimal weights we found for previous segment")
            model_copy=None
            model_copy = copy.deepcopy(model)
            # if seg==1:
            #     a_lr_t=a_lr_t*2
            # if seg>1:
            #     a_lr_t*1.1
            #if seg>=1:
            #    a_lr_t=a_lr_t*2
            optim = SGD_Delayed_Updates(args.optimizer.learning_rate,0.0,a_lr_t)
            mw = ModuleWrapper(model, optim, modules_to_replace_temp,args.quan.excepts)
            mw.initialize()

            perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)
            counter=0

            if args.eval:
                process.validate(test_loader, mw, criterion, -1, monitors, args)
            else:  # training
                #print("wpwpwpw",args.epochs)
                take_Best=True
                best_dict={}
                best_acc=0
                for times in range(start_epoch, args.epochs):
                    #print("beg :",t.cuda.memory_summary(device=None, abbreviated=False))
                    set_random_seed(temp_seed)

                    print(times ," time of finding the optimal weights for this segment")                    

                    train_a_all_times(times,val_loader,train_loader,start_epoch,T,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw,num_solution,num_of_epochs_each_time,seg,test_loader)
                    #recalibrate BN
                    with t.no_grad():
                        for inputs, _ in train_loader: 
                            outputs = mw.forward(inputs)
                    v_top1, v_top5, v_loss = process.validate(test_loader, mw, criterion, start_epoch, monitors, args)
                    v_top1_list.append(v_top1)
                    test_loss.append(v_loss)
                        
                    t_top1, t_top5, t_loss =process.validate(train_loader, mw, criterion, start_epoch, monitors, args)
                    # v_top1, v_top5, v_loss = process.validate(test_loader, mw, criterion, start_epoch, monitors, args)
                    train_top1_list.append(t_top1)
                    test_top1_list.append(v_top1)
                    train_loss.append(t_loss)

                    run_cfg = {
                            "a_learning_rate": a_lr,
                            "num_solution": num_solution
                            ,"Backward": backward_for_test
                    }
                    #first version:
                    dump_three_lists(train_top1_list, test_top1_list, list_train, "train_top1", "test_top1", "C",path="/home/gild/Lsq_with_gSTE/out", log_params=run_cfg  )
                    append_result(
                    file_path=results_file,
                    train_val=t_top1,
                    test_val=v_top1,
                    c_value=list_train[-1]
                )
                    #second version:
                    #dump_three_lists(list_train, test_top1_list, [6], "train_top1", "test_top1", "C",path="/home/gild/Lsq_with_gSTE/out", log_params=run_cfg  )


                    if v_top1>best_acc and take_Best:
                        print("am here er er er @!$%!#%!#%@!#$@!#@!$!@%$!@%!@")
                        best_acc=v_top1
                        best_dict=model.state_dict().copy()
                    torch.save(model.state_dict(), '/home/gild/Lsq_with_gSTE/models_saved/num_sol_'+str(num_solution)+'_lr_'+str(a_lr_t)+"_each_time_"+str(num_of_epochs_each_time)+".pth")
                    
                    prev_model = model.state_dict()
                    
                    model_new= copy.deepcopy(model_copy)
                    
                    
                    with torch.no_grad():#saving trained a values between iterations
                        flag=0
                        
                        for name, param in model_new.named_parameters():
                            if counter ==args.epochs-1:
                                print("aofbfdalnb;skdaf!@#!@#@!#!@#@!#@!#@!#@!#@!#!@#@!#@!")
                                restart_a_each_time=True
                                if restart_a_each_time==True:
                                    if name.endswith('.a') or name.endswith('.a_n'):
                                        param.data.fill_(1.0)
                                    else:
                                        if take_Best:
                                            param.copy_(best_dict[name].detach())
                                        else:
                                            param.copy_(prev_model[name].detach())
                                else:
                                    if take_Best:
                                            param.copy_(best_dict[name].detach())
                                    else:
                                        param.copy_(prev_model[name].detach())


                            else:
                                
                                if name.endswith('.a') or name.endswith('.a_n'):
                                    #torch.where(param.data-prev_model[name].data!=0,,)
                                    diff=param.data-prev_model[name].data
                                    #print("diffrence between previous a and current a: ",diff[diff!=0])
                                    #if non_zero_statistics(diff)['count'] != 0:
                                        #print("Statistics of the change in a : ",non_zero_statistics(diff))
                                        #print("Statistics of a : ",non_zero_statistics(param.data))
                                        #stats_list.append(non_zero_statistics(diff))
                                    param.copy_(prev_model[name].detach())
                                    assert torch.equal(model_new.state_dict()[name].detach(), prev_model[name].detach())
                                    if flag==1:
                                        
                                        tensor_histogram(param.cpu())
                                    flag+=1
                                else:
                                    reset_weights=True
                                    if reset_weights==False:
                                        param.copy_(prev_model[name].detach())
                    
                    #v_top1, v_top5, v_loss = process.validate(test_loader, mw, criterion, start_epoch, monitors, args)

                    counter+=1
                    #print("mid :",t.cuda.memory_summary(device=None, abbreviated=False))

                    model=None
                    model=model_new
                    
                    
                    optim = SGD_Delayed_Updates(args.optimizer.learning_rate,0.0,a_lr_t)
                    
                    mw = ModuleWrapper(model_new, optim, modules_to_replace_temp,args.quan.excepts)

                    mw.initialize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    print("v_top1_list on train : ",v_top1_list)
                    logger.info(("v_top1_list on train : "+str(v_top1_list)))
                    #print("stats list ",stats_list)
                    #print("end",t.cuda.memory_summary(device=None, abbreviated=False))
            temp_seed+=1

        save_and_plot_losses(loss_list       , str(log_dir) + "/overall_loss.npy" , str(log_dir) + "/overall_loss.png" , "loss" , "Overall Loss")
        save_and_plot_losses(test_loss       , str(log_dir) + "/test_loss.npy"    , str(log_dir) + "/test_loss.png"    , "loss" , "Test Loss")
        save_and_plot_losses(train_loss      , str(log_dir) + "/train_loss.npy"   , str(log_dir) + "/train_loss.png"   , "loss" , "Train Loss")
        save_and_plot_losses(train_top1_list , str(log_dir) + "/train_top1.npy"   , str(log_dir) + "/train_top1.png"   , "top1" , "Train Top1 Accuracy")
        save_and_plot_losses(test_top1_list  , str(log_dir) + "/test_top1.npy"    , str(log_dir) + "/test_top1.png"    , "top1" , "Test Top1 Accuracy")
        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, mw, criterion, -1, monitors, args)
        tbmonitor.writer.close()  # close the TensorBoard
        logger.info('Program completed successfully ... exiting ...')

def main_all_times(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T,num_of_epochs_each_time):
    torch.backends.cudnn.deterministic = True
    set_random_seed(seed)  
    model_copy = copy.deepcopy(model)
    compare_models(model_copy, model)

    optim = SGD_Delayed_Updates(args.optimizer.learning_rate,0.0,a_lr)
    mw = ModuleWrapper(model, optim, modules_to_replace_temp,args.quan.excepts)
    mw.initialize()

    perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)
    counter=0
    v_top1_list=[]
    
    if args.eval:
        process.validate(test_loader, mw, criterion, -1, monitors, args)
    else:  # training
        
        for times in range(start_epoch, args.epochs):
            #set_random_seed(seed)  

            v_top1, v_top5, v_loss = process.validate(test_loader, mw, criterion, start_epoch, monitors, args)

            train_a_all_times(times,val_loader,train_loader,start_epoch,T,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw,num_solution,num_of_epochs_each_time,0)
            
            v_top1, v_top5, v_loss = process.validate(test_loader, mw, criterion, start_epoch, monitors, args)
            
            v_top1_list.append(v_top1)
            torch.save(model.state_dict(), '/home/gild/Lsq_with_gSTE/models_saved/num_sol_'+str(num_solution)+'_lr_'+str(a_lr)+"_each_time_"+str(num_of_epochs_each_time)+".pth")
            
            prev_model = model.state_dict()
            
            model_new= copy.deepcopy(model_copy)
            
            
            
            if num_solution == 8:
                model_new.load_state_dict(prev_model)
            else:
                with torch.no_grad():#saving trained a values between iterations
                    flag=0
                    for name, param in model_new.named_parameters():
                        if name.endswith('.a') or name.endswith('.a_n') or name.endswith('.a_p'):
                            param.copy_(prev_model[name])
                            assert torch.equal(model_new.state_dict()[name], prev_model[name])
                            if flag==1:
                                tensor_histogram(param.cpu())
                            flag+=1
            
            model=None
            model=model_new
            
            
            optim = SGD_Delayed_Updates(args.optimizer.learning_rate,0.0,a_lr)
            
            mw = ModuleWrapper(model_new, optim, modules_to_replace_temp,args.quan.excepts)

            mw.initialize()

            v_top1, v_top5, v_loss = process.validate(test_loader, mw, criterion, start_epoch, monitors, args)
            gc.collect()
            torch.cuda.empty_cache()
            
            print("v_top1_list : ",v_top1_list)        
            
        

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, mw, criterion, -1, monitors, args)

        tbmonitor.writer.close()  # close the TensorBoard
        logger.info('Program completed successfully ... exiting ...')
        logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')


def train_a_all_times(times,val_loader,train_loader,start_epoch,T,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw,num_solution,num_of_epochs_each_time,base,test_loader):
    print(" Starting ",times," time of updating a")
    mw.begin()
    #if num_solution == 12:
    #    mw.zero_grad_less_greedy()
    mw.zero_grad()
    
    this_training_list=[]
    count=0
    prev_last=[]
    last_train=[]
    train_top1_list=[]
    test_top1_list=[]
    for epoch in range(num_of_epochs_each_time):
        
        logger.info('>>>>>>>> Epoch %3d' % (base*num_of_epochs_each_time+epoch))
        
        t_top1, t_top5, t_loss = process.train_all_times(train_loader, mw,num_solution,T, criterion, epoch, monitors, args,base*num_of_epochs_each_time)
        # v_top1, v_top5, v_loss = process.validate(test_loader, mw, criterion, start_epoch, monitors, args)
        loss_list.append(t_loss)

        # train_top1_list.append(t_top1)
        # test_top1_list.append(v_top1)
        # run_cfg = {
        #         "a_learning_rate": a_lr,
        #         "num_solution": num_solution
                
        # }
        # dump_three_lists(train_top1_list, test_top1_list, [6], "train_top1", "test_top1", "C",path="/home/gild/Lsq_with_gSTE/out", log_params=run_cfg  )
        
        prev_last=last_train
        last_train=[times,t_top1]

        this_training_list.append(t_top1)
        count+=1

        if num_of_epochs_each_time == count:
            rng = random.Random()
            
            num=rng.randint(1, 100)
            print("num is : ",num)
            #recalibrate BN
            with t.no_grad():
                for inputs, _ in train_loader: 
                    outputs = mw.forward(inputs)

            set_random_seed(num)  
            vont_top1, _, _ = process.validate(train_loader, mw, criterion, start_epoch, monitors, args)
            list_vont_top1.append([times,vont_top1])

            mw.step_a()
            mw.zero_grad()
            count=0
    
    print("list vont_top1 is : ",list_vont_top1)
    logger.info(("list vont_top1 is : "+str(list_vont_top1)))
    print("Current : ",this_training_list)
    logger.info(("Current : "+str(this_training_list)))
    #prev_list_train.append(prev_last)
    #print("prev list train is : ",prev_list_train)
    list_train.append(last_train)
    print("list train is : ",list_train) 
    logger.info(("list train is : "+str(list_train)))

def main_all_times_less_greedy(model,args,modules_to_replace_temp,train_loader,logger,test_loader,criterion,monitors,val_loader,start_epoch,tbmonitor,log_dir,T,num_of_epochs_each_time):
    torch.backends.cudnn.deterministic = True
    
    v_top1_list=[]
    stats_list = []
    a_lr_t=a_lr
    if num_solution == 12:
        set_random_seed(seed)
        temp_seed=0
        start_epoch=0
        train_top1_list=[]
        test_top1_list = []
        run_cfg = {
        "a_learning_rate": a_lr,
        "num_solution": num_solution
        }
        results_file = init_results_file(
            path="/home/gild/Lsq_with_gSTE/out",         # or wherever you want to write
            base_name="data_repeat",
            log_params=run_cfg
        )

        for seg in range(0,num_epochs_div_repeats):
            print(seg," Segment of training, using the optimal weights we found for previous segment")
            model_copy=None
            model_copy = copy.deepcopy(model)

            optim = SGD_less_greedy_Updates(args.optimizer.learning_rate,0.0,a_lr_t)
            mw = ModuleWrapper(model, optim, modules_to_replace_temp,args.quan.excepts)
            mw.initialize()

            perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)
            counter=0
            
            if args.eval:
                process.validate(test_loader, mw, criterion, -1, monitors, args)
            else:  # training
                for times in range(start_epoch, args.epochs):
                    #print("beg :",t.cuda.memory_summary(device=None, abbreviated=False))
                    set_random_seed(temp_seed)

                    print(times ," time of finding the optimal weights for this segment")
                    

                    train_a_all_times(times,val_loader,train_loader,start_epoch,T,criterion,monitors,args,logger,perf_scoreboard,tbmonitor,mw,num_solution,num_of_epochs_each_time,seg, test_loader)
                    #recalibrate BN
                    with t.no_grad():
                        for inputs, _ in train_loader: 
                            outputs = mw.forward(inputs)
                    v_top1, v_top5, v_loss = process.validate(test_loader, mw, criterion, start_epoch, monitors, args)
                    v_top1_list.append(v_top1)
                    
                    t_top1, t_top5, t_loss =process.validate(train_loader, mw, criterion, start_epoch, monitors, args)
                    # v_top1, v_top5, v_loss = process.validate(test_loader, mw, criterion, start_epoch, monitors, args)
                    train_top1_list.append(t_top1)
                    test_top1_list.append(v_top1)
                    run_cfg = {
                            "a_learning_rate": a_lr,
                            "num_solution": num_solution
                            ,"Backward": backward_for_test
                    }
                    #first version:
                    dump_three_lists(train_top1_list, test_top1_list, list_train, "train_top1", "test_top1", "C",path="/home/gild/Lsq_with_gSTE/out", log_params=run_cfg  )
                    append_result(
                    file_path=results_file,
                    train_val=t_top1,
                    test_val=v_top1,
                    c_value=list_train[-1]
                    )
                    #second version:
                    #dump_three_lists(list_train, test_top1_list, [6], "train_top1", "test_top1", "C",path="/home/gild/Lsq_with_gSTE/out", log_params=run_cfg  )


                    torch.save(model.state_dict(), '/home/gild/Lsq_with_gSTE/models_saved/num_sol_'+str(num_solution)+'_lr_'+str(a_lr_t)+"_each_time_"+str(num_of_epochs_each_time)+".pth")
                    
                    prev_model = model.state_dict()
                    #print("befdeepcopy :",t.cuda.memory_summary(device=None, abbreviated=False))

                    model_new= copy.deepcopy(model_copy)
                    #print("aftdeepcopy :",t.cuda.memory_summary(device=None, abbreviated=False))

                    
                    with torch.no_grad():#saving trained a values between iterations
                        flag=0
                        
                        for name, param in model_new.named_parameters():
                            if counter ==args.epochs-1:
                                
                                restart_a_each_time=True
                                if restart_a_each_time==True:
                                    if name.endswith('.a') or name.endswith('.a_n'):
                                        param.data.fill_(1.0)
                                    else:
                                        param.copy_(prev_model[name].detach())
                                else:
                                    param.copy_(prev_model[name].detach())

                            else:
                                if name.endswith('.a') or name.endswith('.a_n'):
                                    #torch.where(param.data-prev_model[name].data!=0,,)
                                    diff=param.data-prev_model[name].data
                                    #print("diffrence between previous a and current a: ",diff[diff!=0])
                                    #if non_zero_statistics(diff)['count'] != 0:
                                        #print("Statistics of the change in a : ",non_zero_statistics(diff))
                                        #print("Statistics of a : ",non_zero_statistics(param.data))
                                        #stats_list.append(non_zero_statistics(diff))
                                    param.copy_(prev_model[name].detach())
                                    #print("model_new.state_dict()[name].detach() ",model_new.state_dict()[name].detach().sum()," prev_model[name].detach() ", prev_model[name].detach().sum())
                                    #diff = torch.abs(model_new.state_dict()[name].detach() - prev_model[name].detach())
                                    #print("Max difference:", torch.max(diff), " name is : ",name)
                                    assert torch.allclose(model_new.state_dict()[name].detach().to(prev_model[name].dtype), prev_model[name].detach())
                                    if flag==1:
                                        
                                        tensor_histogram(param.cpu())
                                    flag+=1
                                else:
                                    reset_weights=True
                                    if reset_weights==False:
                                        param.copy_(prev_model[name].detach())
                    
                    #v_top1, v_top5, v_loss = process.validate(test_loader, mw, criterion, start_epoch, monitors, args)

                    counter+=1
                    #print("mid :",t.cuda.memory_summary(device=None, abbreviated=False))

                    model=None
                    model=model_new
                    
                    
                    optim = SGD_less_greedy_Updates(args.optimizer.learning_rate,0.0,a_lr_t)
                    
                    mw = ModuleWrapper(model_new, optim, modules_to_replace_temp,args.quan.excepts)

                    mw.initialize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    print("v_top1_list on train : ",v_top1_list)
                    logger.info(("v_top1_list on train : "+str(v_top1_list)))
                    #print("stats list ",stats_list)
                    #print("end",t.cuda.memory_summary(device=None, abbreviated=False))
            temp_seed+=1

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, mw, criterion, -1, monitors, args)

        tbmonitor.writer.close()  # close the TensorBoard
        logger.info('Program completed successfully ... exiting ...')

def tensor_histogram(tensor):
    # Convert the tensor to a numpy array
    plt.close('all')

    tensor_np = tensor.numpy()
    
    # Flatten the tensor to 1D for the histogram
    tensor_np_flat = tensor_np.flatten()
    
    # Plot the histogram
    plt.ion()  # Enable interactive mode
    plt.hist(tensor_np_flat, bins=100,range=(0,5), edgecolor='black')
    plt.title('Histogram of Tensor Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.draw()
    plt.pause(0.001)  # Allow the plot to update
    
def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

if __name__ == "__main__":
    main()
