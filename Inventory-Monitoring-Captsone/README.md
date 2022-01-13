# Inventory Monitoring at Distribution Centers

This is an end-to-end machine learning project using a pre-trained ResNet model on the [Amazon Bin Image Dataset](https://registry.opendata.aws/amazon-bin-imagery/).

This project focuses on building a machine learning (computer vision) model that can
be used in the inventory monitoring process (counting the number of items in the bins
robots carry).

## Project Set Up and Installation

In aws sagemaker, create a notebook instance and and select any instance type of your choice. I used **ml.t2.medium** in this project because most code in the project notebook do not requre GPU and also to avoid extra cost in running in the project since GPU instances are more expensive and this instance type costs $0.05 per hour. Once the notebook is launched, open the jupyter notebook and choose a kernel of your choice. It is advisable to choose the `conda_pytorch_latest_p36` for this project since it comes pre-installed with most of the packages needed to run a pytorch project. After this, everything is else is same as running any python jupyter notebook.

## Dataset

### Overview

The original [Amazon Bin Image Dataset](https://registry.opendata.aws/amazon-bin-imagery/) contains 500,000 images (jpg format) from
bins (containing one or more objects) of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations. For each image there is a metadata file (json format) containing information about the image like the number of objects, it’s
dimension and the type of object. For this task, only a subset of the dataset used to classify the number of objects in each bin in order to avoid excess cost and the item count from the metadata is used as the label to train the model.
Images are located in the bin-images directory, and metadata for each image is located in the metadata directory. Images and their associated metadata share simple numerical unique identifiers. Below is an example image and its associated metadata:

---

![sample image](https://i.imgur.com/py6Ro43.jpg)

---

![sample metadata](https://i.imgur.com/3LYS0Lg.png)

In the `sagemaker.ipynb` notebook, the function `download_and_arrange_data` uses the file `file_list.json` to download the data from [Amazon Bin Image Dataset](https://registry.opendata.aws/amazon-bin-imagery/).

However, in order to have train and test data, I created a function called `random_split` which shuffles the `file_list.json` and splits it into 80% for train and 20% test. Another function called `save_json` saves the splitted data into **train_list.json** and **test_list.json** but maintains the structure just as `file_list.json`. I modified the `download_and_arrange_data` function to accept a file and thus can be called separately on train_list.json and test_list.json.
I programmatically create s3 bucket in the same notebook with a function called `create_bucket` which accepts the name of the bucket as input. The data is then uploaded the data to the S3 bucket using the command `!aws s3 cp train_data s3://inventorybin/train_data/ --recursive` for train data and `!aws s3 cp train_data s3://inventorybin/test_data/ --recursive` for the test data.

## Python script files

- `hpo.py` - This is used for hyperparameter tuning. The model is trained with different hyperparameters and the best combination is selected based on the criteria given. In this case we use the loss.
- `train.py` - This is used for both single instance training and multi-intance training. The best hyperparameters from the tuning job is used for training the model. We also add hooks for proper debugging purposes.
- `inference.py` - This file is used to enable us make inference with the deployed endpoint.
- `ec2train.py` - The code here is used for training on the EC2 spot instance to reduce cost.

## Model Training

In this project, a pre-trained convolutional neural network (CNN) specifically the `ResNet50` model was used. Although recently there have been the introduction of vision transformers, computer vision problems have predominantly been solved with convolutional neural networks thus the reason to opt for convolutional neural network (CNN). One advantage of using the ResNet architecture is that it solves the vanishing gradient problem. The pre-trained network as a feature extractor and then replace its final layer with a fully connected neural network for fine-tuning on our task at hand. The `ResNet50` have performed very well on most computer vision tasks that is why it was selected.

### Hyperparameter Tuning

Hyperparameters are the settings that can be tuned before running a training job in order to control the behaviour of a machine learning model. They can impact the training time, model convergence and the accuracy of the model.
To select the best hyperparameter, I used the following 3 parameters in my hyperparameter search space:

- `learning_rate` - determines the step size at each epoch whiles moving towawrd the minimum of a loss function.Learning rate influences the extent to which a model can acquire new information. Generally, during training, the model starts at some random point and sample different weights. It is extremely crucial to set the right learning rate for any training job because a larger learning rate may cause the model to overshoot the optimal solution and smaller learning rate will result in longer training time to find the optimal solution.
- `batch_size` - shows the number of training samples that will be used in each epoch. This is necessary because smaller batch sizes help to easily get out "local minima" and also utilize less resource power whilst larger batch sizes require more resource power and may get stuck in the wrong solution.
- `epoch` - refers one complete pass of the training dataset through the model. It is important to set the right number of epochs because this can influence overfitting. For a small dataset as used in this project, setting a very large epoch may cause the model memorize the training data instead of generalizing and this may cause overfitting and setting it very small may also result in the model not learning.

---

<!-- ![hyperparameter jobs](https://i.imgur.com/Y1W0GRy.png) -->

![hyperparameter jobs](https://i.imgur.com/ue3tWl3.png)

---

![best hyperparameter](https://i.imgur.com/hNfHWUB.png)

### Model Performance

The loss (cross entropy loss) decreased as expected, however since the dataset is small as compared to the size of the dataset used in the benchmark model, the accuracy of the model was impressive although we could not beat that of the benchmark model. This model achieved an average test accuracy of 33% as compared to the benchmark model which achieved an accuracy of 55.67%. This is because we used a very small dataset. Below is a table showing the difference in datasets and and the accuracy achieved.

The bench model used taken from the [Amazon Bin Image Dataset Challenge](https://github.com/silverbottlep/abid_challenge) by [silverbottlep](https://github.com/silverbottlep).

| Description      | Total size | Train size | Test size | Accuracy |
| ---------------- | ---------- | ---------- | --------- | -------- |
| Project image    | 10,441     | 8,352      | 2,089     | 33%      |
| Benchmark images | 535,234    | 481,711    | 53,523    | 55.67%   |

## Standout Suggestions

### Model Deployment

During training the model was saved to S3. We first get the saved model from S3 using `estimator.model_location`.
We pass in the model_location, the sagemaker role, and use the script `inference.py` as entry point.
We created a class called `ImagePredictor` that inherits the sagemaker `predictor` class that deserializes the image because we read the image as a json object, which is what the sagemaker endpoint expects as input. We use the class as the `predictor_cls` parameter for the sagemaker pytorch model.
We then deploy to the sagemaker endpoint using a single instance count and the `ml.m5.large` instance type. Below is a sample code used to define the predictor and also deploy it.

`pytorch_model = PyTorchModel(model_data=model_location, role=role, entry_point='inference.py',py_version='py3', framework_version='1.4', predictor_cls=ImagePredictor)`

`predictor = pytorch_model.deploy(initial_instance_count=1, instance_type='ml.m5.large')`

To query the endpoint, we first use the python request library to download an image from the original datasource which is not part of our training and test dataset. We use the predict function from the sagemaker pytorch model to predict how many items are in the given image. The output is an array of numbers. Thus we use `numpy.argmax` to get the actual number of items (single number) in the given image.Below is an image of the created endpoint in sagemaker.

---

sample endpoint prediction
![endpoint query predict](https://i.imgur.com/2aoBilV.png)

---

running endpoint
![endpoint](https://i.imgur.com/YuqKHGt.png)

### Reduce Cost

To reduce the cost of training, I used EC2 spot instance for my training. I chose the **t2.medium** as instance type. This instance costs $0.0464 with **2 vCPUs** and **4 GB** of memory and it has low to moderate performance.
After creating and launching the spot instance, I activated the latest pytorch model using `source activate pytorch_latest_p37`. In the terminal I used **vim** to create a file called train.py using `vim train.py`.
In the opened vim editor, I used the command `:set paste` (this is done to keep the formatting of any pasted code) and also entered `i` to set it to insert mode. I pasted code from my `train.py` script and used the command `:wq!` to save and close the vim editor.
To easily have access to my training data, I connected my ec2 instance to my s3 bucket by first creating an **IAM** role for my ec2 instance and attached the **AmazonS3FullAccess** (although this is not the most secured option but for the purpose of this project) to allow the ec2 instance to read from S3.
First I installed the [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html) and verified if my ec2 instance has access to S3 using `aws s3 ls s3://<name-of-s3-bucket>`.
I used the command `aws s3 sync s3://<name-of-s3-bucket> <my-local-directory>` to get my training and test dataset into my ec2 instance.
To train the model, I use `python train.py`.
The ec2 spot instance training cost is lower than the sagemaker cost as evident in the images shown below. For about 7 hours, it only cost $0.35 whilst total cost of sagemaker is about $9.83 dollars. Even though sagemaker has a lot of different services, the main cost of running a sagemaker notebook is still more expensive than that of ec2 instance.
Another reason that also accounted for the high in using sagemaker is because I used the instance type `ml.g4dn.xlarge` which has a GPU.

---

running spot intance
![ec2 spot instance](https://i.imgur.com/lWLaTw6.png)

---

ec2 cost
![ec2 cost](https://i.imgur.com/mtXPnN4.png)

---

sagemaker cost
![sagemaker cost](https://i.imgur.com/i5fi1jq.png)

### Model Profiling and Debugging

The hook for debugging is set to record the loss in both training and testing.The profiler report can be found in the Profiler report folder. Below is the loss plot, however because I set the count too high the validation loss did not appear in the plot and I plotted it separately.

---

loss plot for training
![hook plot](https://i.imgur.com/Ns1iYOI.png)

---

validation plot
![validation loss plot](https://i.imgur.com/ncRssst.png)

### Multi-Instance Training

For multi-instance training, I used the `ml.g4dn.xlarge` instance. I selected this because it has **4 vCPU**s **16 GiB** of memory and also includes a **GPU** for faster and accelerated training.
For the actual training I used the `train.py` as entry point and used an **instance count of 2**. This will speed up training by training two instances at the same time. Below is the image showing the multi-instance jobs created in sagemaker.

---

![multi-instance job settings](https://i.imgur.com/Z2yeFzq.png)

---

![multi-instance cloud watch](https://i.imgur.com/x6jtTC7.png)
