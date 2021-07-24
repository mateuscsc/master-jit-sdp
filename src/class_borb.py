import numpy as np
import pandas as pd
from collections import deque
from class_models import Baseline
from time import sleep

###########################################################################################
#                                Batch Oversampling Rate Boosting (BORB)                  #
###########################################################################################


class BORB(Baseline):
	###############
    # Constructor #
    ###############
    def __init__(self, model, borb_window_size,borb_lo,borb_l1,borb_fr1,borb_m,borb_sample_size,borb_ps_size):
        Baseline.__init__(self, model)
        #save model
        self.model = model
        # init
        self.window_size = borb_window_size
        # buffer to store last (window_size) predictions
        self.scores = deque(maxlen=self.window_size)

        # Set  borb induced defect prediction rate
        self.borb_ir1 = borb_fr1

        # Set  borb fixed defect prediction rate
        self.borb_fr1 = borb_fr1

        # Init scores with fr1 value
        self._init_score()

        # Init dataframe empty <positive instances>
        self.df_pos = pd.DataFrame()

        # Init dataframe empty <negative instances>
        self.df_neg= pd.DataFrame()

        # Init oversampling boosting factors 𝑜𝑏𝑓0 and 𝑜𝑏𝑓1
        self.obf_neg = 1
        self.obf_pos = 1

        # and 𝑙0 and 𝑙1 control the maximum boosting factor values
        self.borb_lo = borb_lo
        self.borb_l1 = borb_l1

        # Init threshold baseline
        self.threshold = 0

        

        self.update_threshold()

        # set parameter m 
        self.m = borb_m

        # Init probability of class
        self.prob_neg = 0.5
        self.prob_pos = 0.5

        # flag enable train
        self.enable_train = False

        # set borb_sample_size
        self.borb_sample_size = borb_sample_size

        # set pull request size 
        self.borb_ps_size = borb_ps_size

        # auxiliary to pull request step
        self.ps_step = 0

    #############
    # Auxiliary #
    #############
    def _init_score(self):
        for c in range(self.window_size):
            self.scores.append(self.borb_fr1)


    #######
    # API #
    #######

    def restart_model(self):
        Baseline.__init__(self, self.model)
        # Restart oversampling boosting factors
        self.obf_neg = 1
        self.obf_pos = 1

    # Weighted.Sample(𝑇, 𝑜𝑏𝑓0, 𝑜𝑏𝑓1, 𝑠);
    def get_training_set(self):

        
        sample_neg_vector = [0] * self.df_neg.shape[0]
        sample_pos_vector = [1] * self.df_pos.shape[0]
        total_sample = sample_neg_vector + sample_pos_vector

        # current total instances
        train_set = self.df_neg.shape[0] + self.df_pos.shape[0]
        if self.borb_sample_size > train_set:
            s = train_set
        else:
            s =	self.borb_sample_size

        prob_per_neg_exemple = [self.prob_neg / self.df_neg.shape[0] ] * self.df_neg.shape[0]
        prob_per_pos_exemple = 	[self.prob_pos / self.df_pos.shape[0] ] * self.df_pos.shape[0]
        total_probs = prob_per_neg_exemple + prob_per_pos_exemple

        sample = np.random.choice(total_sample,s,True,p=total_probs)
        n_samples_pos = sum(sample)
        n_samples_neg = len(sample) - n_samples_pos
        _df_neg = self.df_neg.sample(n_samples_neg,replace=True).copy()
        _df_pos = self.df_pos.sample(n_samples_pos,replace=True).copy()

        # merge instances
        _df_merge = pd.concat([_df_neg,_df_pos],axis=0)
        n_features = _df_merge.shape[1] - 1
        size       = _df_merge.shape[0]# current train_set size
        
        # convert merged instances (dataframe) to np arrays
        x = _df_merge.values[:,:-1].copy().reshape(size,n_features)
        y = _df_merge.values[:,-1].copy().reshape(size, 1)
        # batch GD
        self.model.change_minibatch_size(size)
        return x, y

    def append_to_hist_data(self, new_instance):
        check_inst_neg = new_instance['class'] == 0
        check_inst_pos = new_instance['class'] == 1
        if len(new_instance[check_inst_neg].copy()) > 0:
            self.df_neg = pd.concat([self.df_neg,new_instance[check_inst_neg].copy()],axis=0)
        elif len(new_instance[check_inst_pos].copy()) > 0:
            self.df_pos = pd.concat([self.df_pos,new_instance[check_inst_pos].copy()],axis=0)

    def update_scores_predict(self, predicts):
        for inst in predicts:
            self.scores.append(inst[0])

    def update_boosting_factors(self):

        if self.borb_ir1 > self.borb_fr1:
            factor = ((( self.m ** self.borb_ir1) -  (self.m ** self.borb_fr1)) / (self.m - (self.m ** self.borb_fr1)))
            self.obf_neg = factor * self.borb_lo + 1
            self.obf_pos = 1
        else:
            exp = self.borb_fr1 - self.borb_ir1
            factor = ((( self.m ** exp) - 1 ) / ((self.m ** self.borb_fr1) -1 ))
            self.obf_pos = factor * self.borb_l1 + 1
            self.obf_neg = 1

    def update_class_probability(self):
        self.prob_neg = self.obf_neg / (self.obf_neg + self.obf_pos)
        self.prob_pos = self.obf_pos / (self.obf_neg + self.obf_pos)

    def update_threshold(self):
        if len(self.scores) > 1:
            self.threshold = np.quantile(self.scores,self.borb_fr1)

    def compute_predict_score(self,predicts,th=True):
        if th:
            return np.greater(predicts, self.threshold).astype(np.float16)
        else:
            return np.greater(predicts, 0.5).astype(np.float16)

    def update_induced_defect_predict_rate(self):
        if len(self.scores) > 0:
            self.borb_ir1 = np.mean((np.greater(self.scores, 0.5).astype(np.float16) )) # Threshold = 0.5 to compute ir1

    def is_trainable(self,time_elapsed):
        if ( (self.df_neg.shape[0] >1) and (self.df_pos.shape[0] >1) ):
            if (time_elapsed // self.borb_ps_size ) > self.ps_step :
                self.ps_step += 1
                self.enable_train = True
                return self.enable_train
            else:
                self.enable_train = False
                return self.enable_train
    				


    			
    	

    			
