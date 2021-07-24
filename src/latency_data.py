import numpy as np
import pandas as pd
from copy import copy
from gc import collect
import os,gc
class LatencyData(object):
    """ StreamManager

    Manage instances that are related to a timestamp and a delay.

    Parameters
    ----------
    timestamp: numpy.datetime64
        Current timestamp of the stream. This timestamp is always
        updated as the stream is processed to simulate when
        the labels will be available for each sample.

    """

    #_COLUMN_NAMES = ["X", "y_true", "arrival_time", "available_time"]
    
    _COLUMN_FEATURES = ['fix', 'ns', 'nd', 'nf', 'entrophy', 'la', 'ld', 'lt', 'ndev', 'age',
                       'nuc', 'exp', 'rexp', 'sexp']
    _COLUMN_LABEL = ['class']
    _COLUMN_FLAG_TREINO = ['code']
    

    def __init__(self, path_data,wt=None,pull_request_size=None,inst_alvo_orig = True):
        # set dataset path
        self.data_dir = path_data
        # set waiting time value
        self.waiting_time = wt 
        
        #set pull request size
        self._PS = pull_request_size
        
        if self.waiting_time == None:
            raise ValueError("Waiting time not defined")
            
        if self._PS == None:
            raise ValueError("Pull request size not defined")
            
        # Init dataframe empty
        self._new_df = pd.DataFrame()    
        # Init dataframe empty historical data current
        self._hist_data = pd.DataFrame()
        
        # Init all historical data - varification latency
        self._create_new_data_stream(inst_alvo_orig)
        
        # get initial timestamp
        self.timestamp = self._new_df['arrival_time'][0]
     
        
    def _create_new_data_stream(self,inst_alvo_orig):
        
        """
        Create historical data with verification latency
        """
        #load dataset
        df = pd.read_csv(self.data_dir)
        # cria coluna de arrival_time
        # tira o offset do primeiro valor. O primeiro valor passa a ser o dia zero
        # transforma o timestamp para dia(divide por 86400 segundos)
        df['arrival_time'] = (df.arrival_time - df.arrival_time[0]) / 86400
        # cria coluna de flag para treino ou teste (predict) com valores iniciais zero
        df['code'] = np.zeros(df.shape[0])
        # cria coluna de flag para treino ou teste (predict) com valores iniciais zero
        df['available_time'] =  np.zeros(df.shape[0])
        
        # collect trash
        _ = gc.collect()

        # verifica quais instâncias são limpas. coluna 'class' ==> 'containsbug'
        check_inst_limpas = df['class'] == 0

        # verifica quais instâncias são alvo. coluna 'class' ==> 'containsbug'
        check_inst_alvo = df['class'] == 1

        # Inicio do  PseudoCodigo Cabral ####################################
        # Regras para instâncias limpas
        # if(x.contains_bug == false): 
        #	geraCopia(w_dias, false ,x, 1)  # linha 1.1      
        #	x.code = 0 						# linha 1.2
        #####################################################################
        #
        # Gerando copia das instâncias <linha 1.1>
        instancias_copia = df[check_inst_limpas].copy()
        # available_time <--  w_dias + arrival_time(timestamp)
        instancias_copia['available_time'] = instancias_copia['arrival_time'] + self.waiting_time
        # x.code <-- 1
        instancias_copia['code'] = 1
        # Adiciona copias na base gerada
        self._new_df = pd.concat([self._new_df,instancias_copia.copy()],axis=0)
        # 
        del instancias_copia # remove da memoria

        # Captura das instancias limpas <linha 1.2>
        instancias_orig = df[check_inst_limpas].copy()
        # available_time <--  arrival_time(timestamp)
        instancias_orig['available_time'] = instancias_orig['arrival_time'] 
        # x.code <-- 0 . 
        instancias_orig['code'] = 0

        # Adiciona instancias originais na base gerada
        self._new_df = pd.concat([self._new_df,instancias_orig.copy()],axis=0)
        #
        del instancias_orig # remove da memoria
        #######################################################################
        # Regras para instâncias alvo
        # if(x.contains_bug == true):   
        #
        # add original_timestamp
        #
        #    geraCopia(days_to_fix, true, x, 1) # linha 2.1
        #
        #    if(x.days_to_fix > w_dias):        # linha 2.2
        #
        #                geraCopia(w_dias, false, x, 1)                     

        #    x.code = 1                         # linha 2.3
        # Gerando copias das instâncias alvo <linha 2.1>
        instancias_copia = df[check_inst_alvo].copy()
        # available_time <--  daystofix + arrival_time(timestamp)
        instancias_copia['available_time'] = instancias_copia['arrival_time'] + instancias_copia['daystofix']
        # Set label True
        instancias_copia['class'] = 1
        # x.code <-- 1
        instancias_copia['code'] = 1
        # Adiciona copias na base gerada
        self._new_df = pd.concat([self._new_df,instancias_copia.copy()],axis=0)
        # 
        del instancias_copia # remove da memoria

        # Recaptura das instâncias alvo 
        instancias_copia = df[check_inst_alvo].copy()
        # Gerando copias na condição de days_to_fix > w_dias <linha 2.2>
        instancias_copia_days_maior =  instancias_copia[instancias_copia['daystofix'] > self.waiting_time].copy()
        # available_time <--  w_dias + arrival_time(timestamp)
        instancias_copia_days_maior['available_time'] = instancias_copia_days_maior['arrival_time'] + self.waiting_time
        # Set label False
        instancias_copia_days_maior['class'] = 0
        # x.code <-- 1 . 
        instancias_copia_days_maior['code'] = 1
        # Adiciona copias na base gerada
        self._new_df = pd.concat([self._new_df,instancias_copia_days_maior.copy()],axis=0)
        del instancias_copia_days_maior, instancias_copia # remove da memoria

        #### Réplicas das instãncias alvos com available_time igual ao arrival_time

        if inst_alvo_orig:

	        # Captura das instancias alvo <linha 2.3>
	        instancias_orig = df[check_inst_alvo].copy()
	        # available_time <--  arrival_time(timestamp)
	        instancias_orig['available_time'] = instancias_orig['arrival_time'] 
	        # x.code <-- 1 . 
	        instancias_orig['code'] = 0
	        # Adiciona instancias originais na base gerada
	        self._new_df = pd.concat([self._new_df,instancias_orig.copy()],axis=0)
	        #
	        del instancias_orig # remove da memoria
	    ############################## Fim do  PseudoCodigo Cabral ############

        # Sort values by available_time values
        self._new_df.sort_values(by = 'available_time',ignore_index = True,inplace = True)
        # move class column to last index
        self._adjust_colum(['class'],self._new_df.shape[1]-1)
        # move class column to last index
        self._adjust_colum(['code'],self._new_df.shape[1]-1)
        del df 




    
    def enable_to_train(self):
        return self.timestamp % self._PS == 0
                

    def update_timestamp(self, timestamp):
        """ update_timestamp

        Update current timestamp of the stream.

        Parameters
        ----------
        timestamp: datetime64
            Current timestamp of the stream. This timestamp is always
            updated as the stream is processed to simulate when
            the labels will be available for each sample.

        """

        self.timestamp = timestamp

    def get_available_samples(self):
        """ get_available_samples

        Get available samples of the stream, i.e., samples that have
        their labels available (available_time <= timestamp).

        Returns
        -------
        tuple
            A tuple containing the data, their true labels and predictions.

        """

        # get samples that have label available
        self._hist_data = self._new_df[self._new_df['available_time'] < self.timestamp].copy()
        # remove these samples from queue
        #self.queue = self.queue[self.queue['available_time'] > self.timestamp]
        # get X, y_true and y_pred
        X = np.array(self._hist_data[self._COLUMN_FEATURES].values.tolist())
        y_true = np.array(self._hist_data[self._COLUMN_LABEL].values.tolist())
        #y_pred = np.array(samples["y_pred"].values.tolist())
        # return X, y_true and y_pred for the dequeued samples
        return X, y_true

    def load_data_with_latency(self):
        """ get_available_samples

        Get available samples of the stream, i.e., samples that have
        their labels available (available_time <= timestamp).

        Returns
        -------
        tuple
            A tuple containing the data, their true labels and predictions.

        """

        # get samples that have label available
        #self._hist_data = self._new_df[self._new_df['available_time'] < self.timestamp].copy()
        # remove these samples from queue
        #self.queue = self.queue[self.queue['available_time'] > self.timestamp]
        # get X, y_true and y_pred
        #X = np.array(self._hist_data[self._COLUMN_FEATURES].values.tolist())
        #y_true = np.array(self._hist_data[self._COLUMN_LABEL].values.tolist())
        #y_pred = np.array(samples["y_pred"].values.tolist())
        # return X, y_true and y_pred for the dequeued samples
        return self._new_df[self._COLUMN_FEATURES + self._COLUMN_LABEL + self._COLUMN_FLAG_TREINO]    


    def get_time_elapsed(self):
    
        

        return self._new_df.available_time.to_list().copy()   


    
    def _adjust_colum(self,cols_to_move,new_index):
        """
        This method re-arranges the columns in a dataframe to place the desired columns at the desired index.
        ex Usage: df = move_columns(df, ['Rev'], 2)   
        :param df:
        :param cols_to_move: The names of the columns to move. They must be a list
        :param new_index: The 0-based location to place the columns.
        :return: Return a dataframe with the columns re-arranged
        """    
        other = [c for c in self._new_df if c not in cols_to_move]
        start = other[0:new_index]
        end = other[new_index:]
        self._new_df = self._new_df[start + cols_to_move + end]

        
if __name__ == "__main__":
    
    BASE_PATH_DIR = '../data_raw'
    SAVE_PATH_DIR = '../input/'
    dataset_dir = os.path.join(SAVE_PATH_DIR,"nova_preprocess.csv")
    stream = StreamManager(dataset_dir,90,90)
    
    print(stream._new_df['class'].value_counts())
    print(stream._new_df.shape)
    # stream.update_timestamp(90)
    # stream.get_available_samples()
    # print(stream._hist_data)
    # print('#'*40)
    # stream.update_timestamp(91)
    # stream.get_available_samples()
    # print(stream._hist_data)