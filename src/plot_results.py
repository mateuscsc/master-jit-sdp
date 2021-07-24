# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#############
# Auxiliary #
#############


def create_plots(list_filenames, list_legend_names, metric, y_axis_low=0.0, step=0.2, loc='lower right', out_name = '', flag_legend=1):
    size = 7
    size_legend = 7
    fig = plt.figure(figsize=(9, 6))

    # markers = ['o', 'x', 'd', 's', '^', 'v', '>', '<']
    max_len = 0
    for i in range(len(list_filenames)):
        arr = np.loadtxt(list_filenames[i], delimiter=', ')         # load data

        means = np.mean(arr, axis=0)                                # y-axis values
        print('shape',means.shape)
        x_axis = np.arange(means.shape[0])                          # x-axis values
        se = np.std(arr, ddof=1, axis=0) / np.sqrt(arr.shape[0])    # standard error (ddof=1 for sample)

        # markers_on = range(0, arr.shape[1], 500)
        # plt.plot(x_axis, means, marker=markers[i], markevery=markers_on, label=str(list_legend_names[i]))
        plt.plot(x_axis, means, label=str(list_legend_names[i]))
        plt.fill_between(x_axis, means - se, means + se, alpha=0.2)
        max_len = arr.shape[1]
        # x-axis
        plt.xlim(0, arr.shape[1])

    # y-axis
    if metric == 'gmeans':
        y_axis_label = 'G-mean'
    elif metric == 'recalls':
        y_axis_label = 'Recall'
    elif metric == 'specificities':
        y_axis_label = 'Specificity'

    plt.xlabel('Time Step', fontsize=size, weight='bold')
    plt.xticks(np.arange(0, 55000 , 10000), fontsize=size)

    plt.ylabel(y_axis_label, fontsize=size, weight='bold')
    plt.yticks(np.arange(y_axis_low, 1.000001, step), fontsize=size)
    plt.ylim(y_axis_low, 1.0)

    # legend
    if flag_legend:
        leg = plt.legend(ncol=1, loc=loc, fontsize=size_legend)
        leg.get_frame().set_alpha(0.9)

    # grid
    plt.grid(linestyle='dotted')

    # plot
    plt.show()

    # save
    out_name = metric + out_name
    # fig.savefig(out_dir + 'plot_' + out_name +'.pdf', bbox_inches='tight')


def calc_mean_se(files, idx):
    means = []
    ses = []

    for f in files:
        arr = np.loadtxt(f, delimiter=', ')  # load data
        arr = arr[:, idx]  # last column

        mean = np.mean(arr, axis=0)  # mean
        se = np.std(arr, ddof=1, axis=0) / np.sqrt(arr.shape[0])  # standard error (ddof=1 for sample)

        means.append(mean)
        ses.append(se)

    means = np.asarray(means)
    ses = np.asarray(ses)

    return means, ses

########
# plot #
########

out_dir_main = "../exps/"
metrics = ['gmeans', 'recalls', 'specificities']
for m in metrics:
    out_dir = out_dir_main + '/'#comparison/'
    filenames = [
        #out_dir + 'areba20' + '_preq_' + m + '.txt',
        # out_dir + 'areba2' + '_preq_' + m + '.txt',
        # out_dir + 'oob' + '_preq_' + m + '.txt',
        #out_dir + 'oob_single' + '_preq_' + m + '.txt',
        # out_dir + 'adaptive_cs' + '_preq_' + m + '.txt',
        out_dir + 'oob_pool_single' + '_preq_' + m + '.txt',
        # out_dir + 'baseline' + '_preq_' + m + '.txt',
    ]
    create_plots(filenames, ['oob_pool_single', 'baseline'], metric=m, flag_legend=0, loc="lower left")
