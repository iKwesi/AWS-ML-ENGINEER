#1 Serialize Image lambda function

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event["s3_key"]
    bucket = event["s3_bucket"]

    # Download the data from s3 to /tmp/image.png
    ## TODO: fill in
    boto3.resource('s3').Bucket(bucket).download_file(key, "/tmp/image.png")

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }
    
#2 Image-classification-lambda-function

import os
import sys
import subprocess

# pip install custom package to /tmp/ and add to path
subprocess.call('pip3 install sagemaker -t /tmp/ --no-cache-dir'.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
sys.path.insert(1, '/tmp/')

import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer
# from sagemaker.predictor import Predictor

ENDPOINT = "image-classification-2021-11-24-12-26-57-245"

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))

    # Decode the image data
    # image = base64.b64decode(event["image_data"])
    image = base64.urlsafe_b64decode(event["image_data"])
    print(image)

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(ENDPOINT, sagemaker_session = sagemaker.Session())

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences = predictor.predict(image)

    # We return the data back to the Step Function    
    event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }


#3 filter low inference lambda function

import json


THRESHOLD = .93


def lambda_handler(event, context):

    # Grab the inferences from the event
    inferences = event["inferences"]
    # body = json.loads(event['body'])
    # inferences = json.loads(body['inferences'])

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = (True for i in inferences if i > THRESHOLD)

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
