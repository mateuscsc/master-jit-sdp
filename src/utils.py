import mpu.io

def save_params(name,data):
	mpu.io.write(name+'.json', data)

def load_params(name):
	if name.split('.')[1] == 'json':
		return mpu.io.read(name)
	else:
		return mpu.io.read(name+'.json')
			










if __name__ == "__main__":


	model_params = {
	    "borb_params": [ 
	        {
	            "dataset": "npm",
	            "waiting_time": 91,
	            "borb_lo": 2.02,
	            "borb_l1": 1.24 ,
	            "borb_m": 2.03 ,
	            "borb_window_size": 142,
	            "borb_ps_size": 96,
	            "borb_sample_size": 1207,
	            "borb_fr1": 0.38,
	            "mlp_batch_size": 131,
	            "mlp_dropout_h_layer": 0.42,
	            "mlp_dropout_input_layer": 0.14,
	            "mlp_h_layers_size": 14,
	            "mlp_learning_rate": 0.0008,
	            "mlp_log_transformation": True,
	            "mlp_n_epochs": 51,
	            "mlp_nhidden_layers": 3,
	            "output_activation": 'sigmoid',
	            "loss_function": 'binary_crossentropy',
	            "weight_init": "he",
	            "layer_dims": [14, 8, 1],
	            "class_weights" :{0: 1.0, 1: 1.0}
	        },

	    ]
	}
	save_params("borb_params",model_params)
