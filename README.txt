1. Dataset
	Dataset:
	http://tcd.miovision.com/static/dataset/MIO-TCD-Localization.tar
	
	Trained Models:
	VGG: https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
	ResNet: https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5
	
2. SourceCode
	- train_frcnn_resnet.py: Implementation of training procedure when using ResNet architecture.
	- test_frcnn_resnet.py: Implementation of testing procedure when using ResNet architecture.
	- train_frcnn_vgg.py: Implementation of training procedure when using VGG architecture.
	- train_frcnn_vgg.py: Implementation of testing procedure when using VGG architecture.
	- map.py: Procedure that calculates the mean average precision score.
	- faster_rcnn:
		- data_augment: Implementation for data augmentation
		- data_generators: Functions for using ground truth bounding boxes
		- fixed_batch_normalisation.py: Functions and classes for batch normalization
		- intersection_over_union.py: Functions for calculating IoU values
		- losses.py: Functions for calculating the bounding box regression and classification losses
		- parser.py: Implementation for parsing the image files and ground truth labels
		- resnet.py: Functions for creating and using the ResNet architecture
		- roi_helpers.py: Helper functions for ROI pooling
		- roi_pooling_conv.py: Functions implementing ROI pooling.
		- vgg.py: Functions for creating and using the VGG architecture
		- visualize: Contains function that helps draw bounding boxes on the images

3. Output
	This folder contains the two sub-folders of each respective method implemented:
	a. sample_resnet_results:  Contains 20 sample images that were obtained when ResNet architecture was used.
	b. sample_vgg_results:  Contains 20 sample images that were obtained when VGG architecture was used.

4. References
	- https://github.com/yhenon/keras-frcnn
	- https://github.com/keras-team/keras/tree/master/keras/applications
	- https://github.com/jinfagang/keras_frcnn
	- prakhardogra921 
