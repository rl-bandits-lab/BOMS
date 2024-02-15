import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker
import pandas as pd
import re
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def merge_csv(csv1, csv2, result_path):
    df1 = pd.read_csv(result_path+'/Trained_Policy/'+csv1)
    df2 = pd.read_csv(result_path+'/Trained_Policy/'+csv2)
    df2['Policy'] = df2['Policy'].apply(str)
    
    merge_df = pd.merge(df1, df2, on='Policy', how='outer')
    merge_df.to_csv(result_path+'/Trained_Policy/Policy Performance.csv')


def plot_average_reward(x, y, x_label, y_label, y_min, y_max, title, path):
    plt.figure(figsize=(15,9))
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(y_min, y_max)
    plt.title(title)
    plt.savefig(path)


def get_mean_std(method, max_exp_return, result_path):
    all_regrets = []

    if method== 'MOPO':
        with open(result_path+'/Results/Baseline - MOPO.txt', 'r', newline='') as f:
            for line in f:
                if 'Trajectory Optimal Returns for Episode: ' in line:
                    result = re.findall("\d+\.\d+", line)      
                    returns = list(map(float, result))
                    regrets = [max_exp_return-r for r in returns]
                    all_regrets.append(regrets)
        regret_mean = np.average(np.array(all_regrets), axis=0)
        regret_std = np.std(np.array(all_regrets), axis=0)

    else:
        for i in range(10):
            with open(result_path+'/Results/'+method+' {}.txt'.format(i+1), 'r', newline='') as f:
                for line in f:
                    if 'Trajectory Optimal Returns for Episode: ' in line:
                        result = re.findall("\d+\.\d+", line)   
                        returns = list(map(float, result))
                        regrets = [max_exp_return-r for r in returns]
                        all_regrets.append(regrets)
        regret_mean = np.average(np.array(all_regrets), axis=0)
        regret_std = np.std(np.array(all_regrets), axis=0)

    return regret_mean, regret_std

# +
def plot_baseline_regrets(args, policy_df, bo_update_times, result_path):
    ope_exp_return = (policy_df[policy_df['OPE'] == policy_df['OPE'].max()]).iloc[0]['True_Returns']
    max_exp_return = policy_df['True_Returns'].max()
    
    if 'hopper' in args.task:
        y_high = 2000
        y_low = 0
    elif 'walker' in args.task:
        y_high = 5000
        y_low = 0
    elif 'halfcheetah' in args.task:
        y_high = 3000
        y_low = 0
    elif 'pen' in args.task:
        y_high = 1500
        y_low = 0
    

    boms_regret_mean, boms_regret_std = get_mean_std('BOMS', max_exp_return, result_path)
    urms_regret_mean, urms_regret_std = get_mean_std('Baseline - Random Selection', max_exp_return, result_path)
    mopo_regret_mean, mopo_regret_std = get_mean_std('MOPO', max_exp_return, result_path)
    
    ope_regrets = np.ones(bo_update_times) * (max_exp_return - ope_exp_return)
    
    
    x = np.linspace(1, bo_update_times, bo_update_times)
    fig, ax = plt.subplots(1, figsize=[6,4])
    ax.plot(x, boms_regret_mean, c="#940303", label="BOMS")
    ax.fill_between(x, boms_regret_mean-boms_regret_std, boms_regret_mean+boms_regret_std, facecolor="#94030322")

    ax.plot(x, urms_regret_mean, c="#125183", label="Random")
    ax.fill_between(x, urms_regret_mean-urms_regret_std, urms_regret_mean+urms_regret_std, facecolor="#12518322")
    
    ax.plot(x, mopo_regret_mean, c="#117506", label="MOPO")
    ax.fill_between(x, mopo_regret_mean-mopo_regret_std, mopo_regret_mean+mopo_regret_std, facecolor="#11750611")

    ax.plot(x, ope_regrets, c="#9c8c05", label="OPE")
    
    
    xlocator = matplotlib.ticker.MultipleLocator(5)
    plt.gca().xaxis.set_major_locator(xlocator)
    ylocator = matplotlib.ticker.MultipleLocator(y_high/4)
    plt.gca().yaxis.set_major_locator(ylocator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel('updated epochs')
    ax.set_ylabel('regret')
    plt.title(args.task[:-3], fontsize=16)

#     axins = zoomed_inset_axes(ax, 3, loc='upper right', borderpad=2)
#     mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
#     axins.set_xlim(16, 20)
#     axins.set_ylim(150, 250)
    
#     axins.plot(x, boms_regret_mean, c="#940303", label="BOMS")
#     axins.fill_between(x, boms_regret_mean-boms_regret_std, boms_regret_mean+boms_regret_std, facecolor="#94030322")

#     axins.plot(x, urms_regret_mean, c="#125183", label="Random")
#     axins.fill_between(x, urms_regret_mean-urms_regret_std, urms_regret_mean+urms_regret_std, facecolor="#12518322")
    
#     axins.plot(x, mopo_regret_mean, c="#117506", label="MOPO")
#     axins.fill_between(x, mopo_regret_mean-mopo_regret_std, mopo_regret_mean+mopo_regret_std, facecolor="#11750611")

#     axins.plot(x, ope_regrets, c="#9c8c05", label="OPE")

#     plt.legend(fontsize="15")
    plt.savefig(result_path+'/Results/Regrets Comparison with Baselines.png')


# +
def plot_ablation_regrets(args, policy_df, bo_update_times, result_path):
    max_exp_return = policy_df['True_Returns'].max()
    
    if 'hopper' in args.task:
        y_high = 2000
        y_low = 0
    elif 'walker' in args.task:
        y_high = 5000
        y_low = 0
    elif 'halfcheetah' in args.task:
        y_high = 3000
        y_low = 0
    elif 'pen' in args.task:
        y_high = 1500
        y_low = 0

    boms_regret_mean, boms_regret_std = get_mean_std('BOMS', max_exp_return, result_path)
    explo_regret_mean, explo_regret_std = get_mean_std('Ablation - Explo Policy', max_exp_return, result_path)
    behavior_regret_mean, behavior_regret_std = get_mean_std('Ablation - Behavior Policy', max_exp_return, result_path)
    trained_regret_mean, trained_regret_std = get_mean_std('Ablation - Trained Policy', max_exp_return, result_path)
    weight_regret_mean, weight_regret_std = get_mean_std('Ablation - Weight Bias', max_exp_return, result_path)
    
    
    x = np.linspace(1, bo_update_times, bo_update_times)
    fig, ax = plt.subplots(1, figsize=[6,4])
    ax.plot(x, boms_regret_mean, c="#940303", label="BOMS")
    ax.fill_between(x, boms_regret_mean-boms_regret_std, boms_regret_mean+boms_regret_std, facecolor="#94030322")

    ax.plot(x, explo_regret_mean, c="#383838", label="Exploration Policy")
    ax.fill_between(x, explo_regret_mean-explo_regret_std, explo_regret_mean+explo_regret_std, facecolor="#38383822")
    
    ax.plot(x, trained_regret_mean, c="#0e8072", label="Model-based Policy")
    ax.fill_between(x, trained_regret_mean-trained_regret_std, trained_regret_mean+trained_regret_std, facecolor="#0e807222")
    
    ax.plot(x, behavior_regret_mean, c="#a19316", label="Model-free Policy")
    ax.fill_between(x, behavior_regret_mean-behavior_regret_std, behavior_regret_mean+behavior_regret_std, facecolor="#a1931622")
    
    ax.plot(x, weight_regret_mean, c="#800ba3", label="Weight Bias")
    ax.fill_between(x, weight_regret_mean-weight_regret_std, weight_regret_mean+weight_regret_std, facecolor="#800ba322")
    
    
    xlocator = matplotlib.ticker.MultipleLocator(5)
    plt.gca().xaxis.set_major_locator(xlocator)
    ylocator = matplotlib.ticker.MultipleLocator(y_high/4)
    plt.gca().yaxis.set_major_locator(ylocator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel('updated epochs')
    ax.set_ylabel('regret')
    plt.title(args.task[:-3], fontsize=16)
    
    
    
#     axins = zoomed_inset_axes(ax, 3, loc='upper right', borderpad=1.5)
#     mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
#     axins.set_xlim(16, 20)
#     axins.set_ylim(80, 280)
    
#     axins.plot(x, boms_regret_mean, c="#940303", label="BOMS")
#     axins.fill_between(x, boms_regret_mean-boms_regret_std,, boms_regret_mean+boms_regret_std,, facecolor="#94030322")

#     axins.plot(x, explo_regret_mean, c="#383838", label="Explo Policy")
#     axins.fill_between(x, explo_regret_mean-explo_regret_std, explo_regret_mean+explo_regret_std, facecolor="#38383822")
    
#     axins.plot(x, trained_regret_mean, c="#0e8072", label="Trained Policy")
#     axins.fill_between(x, trained_regret_mean-trained_regret_std, trained_regret_mean+trained_regret_std, facecolor="#0e807222")
    
#     axins.plot(x, behavior_regret_mean, c="#a19316", label="Behavior Policy")
#     axins.fill_between(x, behavior_regret_mean-behavior_regret_std, behavior_regret_mean+behavior_regret_std, facecolor="#a1931622")
    
#     axins.plot(x, weight_regret_mean, c="#800ba3", label="Weight Bias")
#     axins.fill_between(x, weight_regret_mean-weight_regret_std, weight_regret_mean+weight_regret_std, facecolor="#800ba322")
    
    
#     plt.legend(fontsize="15")
    plt.savefig(result_path+'/Results/Regrets Comparison with Ablations.png')


