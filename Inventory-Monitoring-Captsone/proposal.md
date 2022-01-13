# AWS Machine Learning Engineer Capstone Proposal
## INVENTORY MONITORING AT DISTRIBUTION CENTERS
                                                                - by Raphael Blankson

## Domain Background
In recent times, Artificial Intelligence (AI) has come roaring out of high-tech labs to become something that people use every day without even realizing it. AI has brought great benefits to all industries, including supply chain and logistics. The volumes of data in supply chain and logistics is ever-growing daily and thus more sophisticated solutions are urgently needed.
One of the main engines of supply-chain and logistics is distribution centers. A distribution center is a specialized warehouse that serves as a hub to strategically store finished goods, streamline the picking and packing process, and ship goods out to another location or final destination [1]. Monitoring inventory is critical for the success of all distribution centers. This is because without the proper monitoring, there will be shortage in inventory which can lead to backorders and customers not receiving the expected number of products or no product at all. To achieve this, companies usually use robots to move items in bins.
This project focuses on building a machine learning (computer vision) model that can be used in the inventory monitoring process (counting the number of items in the bins robots carry). Using the model for this process speed up distribution process and also keep customers happy.


## Problem Statement
Amazon fulfillment centers are bustling hubs of innovation the company to deliver millions of products all over the world. These products are randomly placed in bins which are carried by robots. There are often issues of misplaced items which results in mismatch. The recorded bin inventory differs from its actual content [6].
The goal of this project is to create a deep learning model that can identify the content of the items in the bin and provide the item count. This can be achieved by attaching a camera to the robot and feeding the images from the camera to the deep learning model which will then identify the content and provide the item count.
The tasks involved may be summarized as follows:
* Download and preprocess the [Amazon Bin Image Dataset](https://registry.opendata.aws/amazon-bin-imagery/) [2].
* Upload data to S3.
* Train model that can identify and show item count.
* Deploy model to sagemaker endpoint.
* Use hyper-parameter tuning to select best hyperparameters.
* Train model with spot instances to reduce cost.
* Use multi-instance training to distribute training across multiple instances.

By providing the correct count of items using artificial intelligence will help serve its customers better.

## Dataset and Inputs
The original [Amazon Bin Image Dataset](https://registry.opendata.aws/amazon-bin-imagery/) [2] contains 500,000 images (jpg format) from bins (containing one or more objects) of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations. For each image there is a metadata file (json format) containing information about the image like the number of objects, it's dimension and the type of object. For this task, only a subset of the dataset used to classify the number of objects in each bin in order to avoid excess cost and the item count from the metadata is used as the label to train the model.

Images are located in the bin-images directory, and metadata for each image is located in the metadata directory. Images and their associated metadata share simple numerical unique identifiers. Below is an example image and its associated metadata:

![sample image](https://i.imgur.com/HN3akuE.jpg)
![sample metadata](https://i.imgur.com/Sg1EvIE.png)

As can be seen from the image, there are tapes in front of the bin that prevents items from falling and sometimes can make the objects in the bin unclear. From the metadata, we can see that this particular bin shown in the image contains 2 different object categories as show in `"EXPECTED_QUANTITY": 2` and for each category, the `quantity` field is 1. Each object category has a unique id called `asin`.


<!-- For example, the metadata for the image at https://aft-vbi-pds.s3.amazonaws.com/bin-images/523.jpg is found at https://aft-vbi-pds.s3.amazonaws.com/metadata/523.json. -->

<!-- If you use the AWS Command Line Interface, you can list images in the bucket with the "ls" command:

aws s3 ls s3://aft-vbi-pds/bin-images/

To download data using the AWS Command Line Interface, you can use the "cp" command. For instance, the following command will copy the image named 523.jpg to your local directory:

aws s3 cp s3://aft-vbi-pds/bin-images/523.jpg 523.jpg -->

## Solution Statement
Since the task at hand is an image classification task, the ideal model to choose is a convolutional neural network. Training a convolutional neural network from scratch is expensive and thus a method called transfer learning which is the improvement of learning a new task through the transfer of knowledge from a related task that has already been learned[4]. A pre-trained convolutional neural network called ResNet [3] will be used. One advantage of using the ResNet architecture is that it solves the vanishing gradient problem. The pre-trained network as a feature extractor and then replace its final layer with a fully connected neural network for fine-tuning on our task at hand.
The solution will be implemented in a sagemaker notebook instance. I will perform hyperparameter tuning to search for the best hyperparameters for training the model. The model will be deployed to a sagemaker endpoint and also create a lambda function to test how the model can be accessed by external clients. The model will also be tested on an EC2 instance to check how we can reduce the cost of running our model in the cloud. 
Finally, multi-instance training will be done. Accuracy will be the metric used in accessing the model.


## BenchMark Model
The bench model is taken from [5]. The model from [5] achieved an overall accuracy of 55.67. We will have to improve the accuracy of this model and also train using sagemaker best practices as learnt from the nanodegree.
Below is the loss and accuracy plot from the benchmark training.
![benchmark loss](https://i.imgur.com/1CPnYMp.png)
![benchmark accuracy](https://i.imgur.com/AHdtKgG.png)



## Evaluation Metrics
We will use accuracy as the evaluation metrics for this project.  The formula for accuracy is given by:
$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}$$

## Project Design
The project will be run in sagemaker, thus we have to create a sagemaker instance. The main steps involved in the project design are listed below:
* Download the dataset.
* Perform data preprocessing if necessary.
* Upload data to an S3 bucket.
* Perform hyperparameter tuning to select the best hyperparameters.
* Create and train the model using sagemaker estimator.
* Use model debugging and profiling to monitor and debug the training job in sagemaker.
* Deploy the model to a sagemaker endpoint and make prediction on the endpoint.
* Create lamda function to allow external clients to access the model.
* Perform cheaper training using EC2 instance.
* Train the model on multiple instances.


## References
1. https://www.shipbob.com/blog/distribution-center/
2. Amazon Bin Image Dataset was accessed on 03-01-2022 from https://registry.opendata.aws/amazon-bin-imagery.
3. He, K., Zhang, X., Ren, S. and Sun, J., 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
4. Torrey, L. and Shavlik, J., 2010. Transfer learning. In Handbook of research on machine learning applications and trends: algorithms, methods, and techniques (pp. 242-264). IGI global.
5. [Amazon Bin Image Dataset Challenge](https://github.com/silverbottlep/abid_challenge) by silverbottlep.
6. [Amazon Inventory Reconciliation using AI](https://github.com/OneNow/AI-Inventory-Reconciliation) by Pablo Rodriguez Bertorello, Sravan Sripada, Nutchapol Dendumrongsup.


