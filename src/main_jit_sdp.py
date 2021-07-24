
import sys
sys.path.append('../src_oob_pool')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
tf.disable_v2_behavior()

from latency_data import LatencyData
import numpy as np
import pandas as pd

from run_main_jit_sdp_oob_pool import run
from class_nn_standard import NN_standard

from time import sleep

###########################################################################################
#                                     Auxiliary functions                                 #
###########################################################################################


# Create text file
def create_file(filename):
    f = open(filename, 'w')
    f.close()

# Write array to a row in the given file
def write_to_file(filename, arr):
    with open(filename, 'a') as f:
        np.savetxt(f, [arr], delimiter=', ', fmt='%1.6f')

###########################################################################################
#                                           Main                                          #
###########################################################################################

def main():

    print('Starting process ...')
    sleep(2)
    print('#'*60)

    # Reproducibility
    seed = 0
    random_state = np.random.RandomState(seed)

    ########################
    # Settings A: Scenario #
    ########################

    # time steps & repetitions
    repeats = 3  # number of repetitions
    times = 30_000  # time steps per repetition

    # Dataset
    dataset = 'nova'  # 
    target = 'class'

    # class imbalanace method
    method = 'oob_pool_single'  # 

    # fixed - do not alter the following

    # Prequential evaluation
    preq_fading_factor = 0.99  # 0 << f < 1.0 - typically, >= 0.8

    # Delayed size metric
    delayed_forget_rate = preq_fading_factor

    

    # store results
    flag_store = 1
    #############################
    # Settings B: parameters ~ model:dataset:baseline {nova:mlp:} # from dissertation {Dinaldo}
    #############################

    # waiting time wt << days to verification latency |  borb-waiting-time = (90, 91, 180)
    waiting_time = 91   
    #############################
    #  #
    #############################

    # fixed - do not alter the following

    # Baseline
    learning_rate = 0.01
    output_activation = 'sigmoid'
    loss_function = 'binary_crossentropy'
    weight_init = "he"
    class_weights = {0: 1.0, 1: 1.0}
    num_epochs = 1
    minibatch_size = 1
    layer_dims = [14, 8, 1] # features shape (1,14)


    # OOB: number of classifiers
    ensemble_size = 20

    # safety check
    if method == 'oob_pool_single':
        ensemble_size = 1


    ################
    # Output files #
    ################

    # output directory
    out_dir = '../exps/'

    # output filenames
    out_name = method
    if method == 'borb' :
        out_name += str(borb_window_size)

    filename_recalls = out_name + '_preq_recalls' + '.txt'
    filename_specificities = out_name + '_preq_specificities' + '.txt'
    filename_gmeans = out_name + '_preq_gmeans' + '.txt'

    # Create output files
    if flag_store:
        create_file(out_dir + filename_recalls)
        create_file(out_dir + filename_specificities)
        create_file(out_dir + filename_gmeans)

    ##############
    # Input data #
    ##############


    BASE_PATH_DIR = '../input/'
    

    # Dataset dirs
    dataset_dir = ''    # init
    if dataset == 'nova':
        dataset_dir = os.path.join(BASE_PATH_DIR,"nova_preprocess.csv")
    elif dataset == 'nova':
        dataset_dir = os.path.join(BASE_PATH_DIR,"nova_preprocess.csv")


    #############################################
    # Generate stream with verification latency #           
    #############################################
    stream = LatencyData(dataset_dir,91,96,True) 

    df = stream.load_data_with_latency()

    times = df[df.code==0].shape[0]# time steps per repetition. Number of instances 
    print('Numero de Time steps',times)

    #########
    # Start #
    #########

    for r in range(repeats):
        print('Repetition: ', r)

        # NN
        nn_standard = NN_standard(
            layer_dims=layer_dims,
            learning_rate=learning_rate,
            output_activation=output_activation,
            loss_function=loss_function,
            num_epochs=num_epochs,
            weight_init=weight_init,
            class_weights=class_weights,
            minibatch_size=minibatch_size)

        # model(s)
        models = [nn_standard]
        if method == 'oob':
            for i in range(ensemble_size - 1):
                models.append(
                    NN_standard(
                        layer_dims=layer_dims,
                        learning_rate=learning_rate,
                        output_activation=output_activation,
                        loss_function=loss_function,
                        num_epochs=num_epochs,
                        weight_init=weight_init,
                        class_weights=class_weights,
                        minibatch_size=minibatch_size,
                        seed=seed)
                )

        # start
        recall, specificity, gmean = run(random_state,times, df, models, method, preq_fading_factor, layer_dims,delayed_forget_rate,target)

        # store
        if flag_store:
            write_to_file(out_dir + filename_recalls, recall)
            write_to_file(out_dir + filename_specificities, specificity)
            write_to_file(out_dir + filename_gmeans, gmean)

        print('#'*60)        


if __name__ == "__main__":
    main()