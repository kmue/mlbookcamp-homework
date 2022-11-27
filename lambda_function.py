#!/usr/bin/env python
# coding: utf-8

# for array operations
import numpy as np

# for img download from url
from io import BytesIO
from urllib import request

# for img preprocessing
from PIL import Image

# for running tflite model
import tflite_runtime.interpreter as tflite

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# to replace
with Image.open('1280px-Smaug_par_David_Demaret.jpg') as img:
    img = img.resize((150, 150), Image.NEAREST)
    

interpreter = tflite.Interpreter(model_path = 'dino_dragon.tflite')

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

###
def predict(url):
    
    img = download_image(url)
    img = prepare_image(img, (150, 150))
    
    x = np.array(img, dtype='float32')
    X = np.array([x]) / 255
    
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)
    return(pred[0][0]) # returns probability value of "dino"

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return(result)
    
    
    
    
    
    



