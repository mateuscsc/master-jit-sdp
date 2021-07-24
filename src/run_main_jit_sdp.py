import numpy as np
from class_models import Baseline ,OOB_POOL

from time import sleep
from copy import copy
###########################################################################################
#                                     Auxiliary functions                                 #
###########################################################################################

# Update prequential evaluation metric (recall / specificity) using fading factor
def update_preq_metric(s_prev, n_prev, correct, fading_factor):
    s = correct + fading_factor * s_prev
    n = 1.0 + fading_factor * n_prev
    metric = s / n

    return s, n, metric


# Update delayed evaluation metric (size, recall / specificity)
def update_delayed_metric(prev, flag, forget_rate):
    return (1.0 - forget_rate) * flag + forget_rate * prev

# Update prequential evaluation metric (recall / specificity) using fading factor
def update_preq_metric(s_prev, n_prev, correct, fading_factor):
    s = correct + fading_factor * s_prev
    n = 1.0 + fading_factor * n_prev
    metric = s / n

    return s, n, metric

def read_new_instance(df):#,time_elapsed):
    
    next = df.head(1).copy()
    df_new = copy(df.drop(0, axis=0).reset_index(drop=True))

    return df_new, next[next.columns[:-1]] , next[next.columns[-1]].values[0]

###########################################################################################
#                                           Run                                           #
###########################################################################################


def run(random_state, time_steps, df, models, method, preq_fading_factor, layer_dims,delayed_forget_rate,target):

    
    
    ############################
    # Init prequential metrics #
    ############################

    preq_recalls = np.zeros(time_steps)
    preq_specificities = np.zeros(time_steps)
    preq_gmeans = np.zeros(time_steps)

    preq_recall, preq_specificity = (1.0,) * 2  # NOTE: init to 1.0 not 0.0
    preq_recall_s, preq_recall_n = (0.0,) * 2
    preq_specificity_s, preq_specificity_n = (0.0,) * 2

    ########################
    # Init delayed metrics #
    ########################

    # size
    delayed_size_neg, delayed_size_pos = (0.0,) * 2


    ################
    # Init methods #
    ################

    technique = Baseline(models[0])     # Baseline init 

    # State-of-the-art
    if method == 'oob_pool_single' or method == 'oob_pool':
        technique = OOB_POOL(models)

    
   
    #########
    # Start #
    #########

    #for t in range(time_steps):
    t = 0
    
    while t < (time_steps -1)  :    
        
        


        ####################
        # Get next example #
        ####################

        df, next, code = read_new_instance(df)

        ##############
        # Prediction verification #
        ##############

        # get x
        x = next.copy().values[:,:-1]

        # get ground truth
        y = next.copy().values[:,-1].reshape(1,1) 

        if y == 0:
            example_neg = True
        else:
            example_neg = False  


        

        if code == 0 :
            if t % 1000 == 0 :
                print('Time step: ', t)
            #time_steps -=1
            t += 1
            y_hat_score, y_hat_class = technique.predict(x)


            #######################
            # Update preq metrics #
            #######################

            # check if misclassification
            correct = 0
            if y == y_hat_class:
                correct = 1

            # update preq. recall / specificity
            if example_neg:
                preq_specificity_s, preq_specificity_n, preq_specificity = update_preq_metric(preq_specificity_s,
                                                                                              preq_specificity_n, correct,
                                                                                              preq_fading_factor)
            else:
                preq_recall_s, preq_recall_n, preq_recall = update_preq_metric(preq_recall_s, preq_recall_n, correct,
                                                                               preq_fading_factor)

            preq_gmean = np.sqrt(preq_recall * preq_specificity)

            # append to results
            preq_recalls[t] = preq_recall
            preq_specificities[t] = preq_specificity
            preq_gmeans[t] = preq_gmean

            ##########################
            # Update delayed metrics #
            ##########################
            delayed_size_neg = update_delayed_metric(delayed_size_neg, example_neg, delayed_forget_rate)
            delayed_size_pos = update_delayed_metric(delayed_size_pos, not example_neg, delayed_forget_rate)    


        elif code == 1 :
            if  example_neg:
                technique.train(x, y)
                
                
            else:
                technique.append_to_pool_defect_induncing(next)   

                # Calculate class imbalance rate
                imbalance_rate = 1.0
                if (not example_neg) and (delayed_size_pos < delayed_size_neg) and (delayed_size_pos != 0.0):
                    imbalance_rate = delayed_size_neg / delayed_size_pos
                elif example_neg and (delayed_size_neg < delayed_size_pos) and (delayed_size_neg != 0.0):
                    imbalance_rate = delayed_size_pos / delayed_size_neg

                #print('delayed_size_pos',delayed_size_pos,'   delayed_size_neg',delayed_size_neg)    

                # OOBPool oversample
                technique.oob_oversample(random_state, imbalance_rate) 

                technique.train(x, y)
                
                

        
       


    return preq_recalls, preq_specificities, preq_gmeans
