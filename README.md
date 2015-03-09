Source code for, AutoRec, an autoencoder based model for collaborative filtering. This package also includes implementation of
RBM based collaborative filtering model(RBM-CF).


Dependencies
============
* Cython
* Progressbar
* Envoy
* climin


Configuration
=============
All the models are defined in yaml configuration file. Configuraiton file consists of three section
* **data**:
	In this section, we define the path of train/test file and path to store the model
	- **train** : path of the training file
	- **test** : path of the test file
	- **save** : path for saving the model
* **param**:
	In this section, we define parameters for the network
	- **lamda**: list of regularization paramter per each layer
	- **max_iter**: maximum number of iteration
	- **batch_size**: size of the batch
	- **optimizer**: Choice of the optimiezer (lbfgs, rprop, rmsprop)
	- **reg_bias**:  whether to regularise bias or not
	- **beta**: sparsity control parameter
	- **num_threads**: maximum number of threads to be used while doing some of the matrix operations (set it to number of CPU cores)
* **layer**:
	In this section, we define the network architecture. Layers are defined by layer index(starting from 1).
	Note that, layers index should be in ascending order (For eg: 1, 2, 3).
	Each layers is defined as 
	- Layer index:
		+ **activation**: Type of activation function (identity, sigmoid, relu, nrelu, tanh)
		+ **num_nodes**: number of nodes in the given layer
		+ **type**: layer type (input, hidden, output)
		+ **partial**: Whether the layer data is partially observed  or needs partial computation (applicable only to input/output nodes)
		+ **binary** : whether to enforce binary coding in the layer or not

Installation
============

First you will need to build the cython modules. Build cython modules 
$ bash buildCython.sh 

Running Code
=============

Running Autorec model
* cd nn/autorec
* PYTHONPATH=<path_to_NNRec_folder> python learner.py -c <path_to_configuration_file>

Running RBMCF model
* cd nn/cfrbm
* PYTHONPATH=<path_to_NNRec_folder> python learner.py -c <path_to_configuration_file>





