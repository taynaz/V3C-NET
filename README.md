# V3C-NET
Vehicle volume estimation using 3d CNN

We are working towards solving a spatial temporal problem. We want to identify a vehicle only out of a real-world environment having other objects as well and we want to count the number of vehicles for a certain time frame as well. 3D CNN seems the best solution as it will help preserve both the spatial as well as temporal data.  
We use a complete 3d Convolutional Neural Network (See figure 8). One half of the network extracts the vehicular features from the video and the latter half counts the vehicles from the vehicular features. 
we designed the network to have 3x3x3 kernel size. We also did brief experimentations with varying input sizes and with varying convolutional neural network architecture designs.

We named the network V3C net; V standing for vehicles, 3 for 3d convolutional network and C for count. The network has eight convolution layers responsible for finding the required vehicular features, five pooling layers to minimize the size of features to make computations faster and seven fully connected layer in end to create a fully connected neural network to count the vehicles from the extracted vehicular features.  
The input is of 3d shape . Spatial domain dimensions are 84x84. The images are really huge in size, and they are resized to accommodate our required dimensions. Each input sample contains 250 frames extracted from 25 fps videos. To make up for 25 temporal dimension, every tenth frame is chosen sequentially. All of the input sample is reshaped and send to network as the network’s input. 
The first convolution layer outputs sixty-four feature maps. The layer has 5248 trainable parameters. The next layer is pooling layer, a max pooling is carried out spatially only on the input from the previous first convolutional layer. The spatial features aren’t pooled as we want to preserve the temporal data at this point and not lose it too soon hence affecting the count result.  
This Layer is next followed by convolution layer with a 3x3x3 kernel size outputting 256 feature maps. It has 221312 learnable parameters. From this layer onwards, the rest of the pooling layers have the same 2x2x2 kernel size. The 3a and 3b have 884992 and 1769728 learnable parameters respectively. The next convolution layer after pooling has 3539456 trainable parameters. The next three convolution layers have 7078400 parameters. The feature extractor part of our network has a total parameter of 27,655,936 that can be trained. The fully connected or dense are all interconnected with each. This causes a huge increase in the number of parameters. The last seven fully connected layers have total parameters of 102,781,953 trainable parameters.  
Before the last pooling layer, zero padding is applied spatially only. Adding it to the temporal data might cause redundancy in final vehicular count.  
The fully connected first layer eventually receives a feature map sized 8192. The output dimension from the last layer of dimension 1x4x4x512 is flattened before passing it to the fully connected layers. This network of fully connected layers will learn to identify vehicular features and count them. The complete architecture is shown in the table (Table number)
Figure 8 V3CNet Architecture

Layer 	Kernel Size (depth, height,  width)	Output shape (depth, height,  height, width, output  channels)
Conv-1 	3x3x3 	25,84,84,64
Max-pool-1 	1x2x2 	25,42,42,64
Conv-2 	3x3x3 	25,42,42,128
Max-pool-2 	2x2x2 	12,21,21,256
Conv-3 	3x3x3 	12,21,21,256
Conv-4 	3x3x3 	12,21,21,256
Max-pool-3 	2x2x2 	6,10,10,256
Conv-5 	3x3x3 	6,10,10,512
Conv-6 	3x3x3 	6,10,10,512
Max-pool-4 	2x2x2 	3,5,5,512
Conv-7 	3x3x3 	3,5,5,512
Conv-8 	3x3x3 	3,5,5,512
Zero padding 	0x2x2 	3,9,9,512
Max-pool-5 	2x2x2 	1,4.4.512
Flatten 	~ 	~
Fully-connected 1 	~ 	8192
Fully-connected 2 	~ 	4096
Fully-connected 3 	~ 	4096
Fully-connected 4 	~ 	4096
Fully-connected 5 	~ 	4096
Fully-connected 6 	~ 	512
Fully-connected 7 	~ 	1

