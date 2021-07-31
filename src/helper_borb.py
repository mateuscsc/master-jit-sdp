import numpy as np
from class_models import Baseline 
from class_borb import BORB
from time import sleep
from copy import copy
###########################################################################################
#                                     Auxiliary functions                                 #
###########################################################################################


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


def run(random_state, time_steps, df, models, method, preq_fading_factor, layer_dims,delayed_forget_rate,target,params):

    
    
    ############################
    # Init prequential metrics #
    ############################
    # FIX len time_steps -1
    preq_recalls = np.zeros(time_steps-1)
    preq_specificities = np.zeros(time_steps-1)
    preq_gmeans = np.zeros(time_steps-1)

    preq_recall, preq_specificity = (0.0,1.0) #* 2  # NOTE: init to 1.0 not 0.0

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
    
    if method == 'borb' :
        technique = BORB(models[0],params["borb_window_size"], params["borb_lo"], params["borb_l1"],
                        params["borb_fr1"], params["borb_m"], params["borb_sample_size"] , params["borb_ps_size"])    

    
   
    #########
    # Start #
    #########
    first_loop = True

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
            if t % 500 == 0 :
                print('Time step: ', t)
                
            #time_steps -=1
            
            y_hat_score, y_hat_class = technique.predict(x)

            #update scores 'C' | BORB
            technique.update_scores_predict(y_hat_score)

            # update threshold | BORB
            technique.update_threshold()

            # predict | BORB
            y_hat_class = technique.compute_predict_score(y_hat_score,th=True)

            # While model not trainable baseline predict always zero
            if  (technique.is_trainable(t) == False) and first_loop:
                y_hat_class = y_hat_class * 0

            elif (technique.is_trainable(t) == True):
                first_loop = False


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

            print('G-mean',preq_gmean)

            ##########################
            # Update delayed metrics #
            ##########################
            delayed_size_neg = update_delayed_metric(delayed_size_neg, example_neg, delayed_forget_rate)
            delayed_size_pos = update_delayed_metric(delayed_size_pos, not example_neg, delayed_forget_rate)    

            t += 1
        
        ###############
        # BORB #
        ###############

        if method == 'borb' :
            technique.append_to_hist_data(next)


        

        if code == 1 :
            # Add hist data
            #technique.append_to_hist_data(next)

            if technique.is_trainable(t): #  :: time elapsed at the moment (instance)
                # Recovery buffer score {C}
                scores = technique.get_buffer_scores()
                df_pos, df_neg = technique.get_hist_instances()

                technique = BORB(models[0],params["borb_window_size"], params["borb_lo"], params["borb_l1"],
                        params["borb_fr1"], params["borb_m"], params["borb_sample_size"] , params["borb_ps_size"],
                        df_pos=df_pos,df_neg=df_neg,scores=scores, restart_model=True)

                del scores, df_pos, df_neg

                #technique.restart_model()

                for i in range(51):
                    # Generate Weighted Sample
                    x, y = technique.get_training_set()

                    ####################
                    # Train classifier #
                    ####################
                    technique.train(x, y)

                    # predict
                    y_hat_score, y_hat_class = technique.predict(x)

                    #update scores 'C'
                    technique.update_scores_predict(y_hat_score)

                    # update ir1
                    technique.update_induced_defect_predict_rate()

                    # update boosting factor
                    technique.update_boosting_factors()

                    # update probs class
                    technique.update_class_probability()
                

        
       


    return preq_recalls, preq_specificities, preq_gmeans
