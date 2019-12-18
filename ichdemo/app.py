import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, url_for, redirect
from magicsauce import * 
import keras
import tensorflow as tf
import cv2

model = keras.models.load_model(weights_file, custom_objects={'correct_positive_diagnoses': correct_positive_diagnoses})
model._make_predict_function()
app = Flask(__name__, static_url_path='/static')


@app.route("/")
def fileFrontPage():
    return render_template('base.html')

@app.route("/handleUpload", methods=['POST'])
def handleFileUpload():
    if 'brainscan' in request.files:
        dcm = request.files['brainscan']
        if dcm.filename != ' ':
            ds = pydicom.dcmread(dcm)

        dcmarray = ds.pixel_array

        img = window_and_scale_brain_subdural_soft(ds)
        img = cv2.resize(img, INPUT_SHAPE[:2], interpolation=cv2.INTER_LINEAR)
        tensor = np.expand_dims(img, axis=0)

        graph = tf.get_default_graph()
        with graph.as_default():
            p = model.predict(tensor)

        print(p)
        fname = dcm.filename
        max_intensity = np.max(dcmarray)
        plt.imshow(dcmarray, cmap=plt.cm.bone)
        plt.savefig('static/images/'+fname+'_raw_img.png')
        metadata = metaprint(ds)
        bsb= bsb_window(ds, fname)
    return render_template('index.html', max=max_intensity, metadata= metadata, 
        url = '/static/images/'+fname+'_raw_img.png', windowing = bsb, prediction=p)

def metaprint(dataset):
    metadata = []
    for data_element in dataset:
        if data_element.name == 'Pixel Data':
            name = data_element.name
            value = 'Array of 524288 elements'
        else:
            name = data_element.name
            value = data_element.value
        metadata.append({'name' : name, 'value': value})
    return metadata

def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000

def window_image(dcm, window_center, window_width):
    
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
    
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)

    return img

def bsb_window(dcm, filename):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    bone_img = window_image(dcm, 600, 2800)
    
    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    bone_img = (bone_img + 800) / 2800

    plt.imshow(brain_img, cmap=plt.cm.bone)
    plt.title('Brain Matter')
    bmpath = 'static/images/'+filename+'_brain_img.png'
    plt.savefig(bmpath)

    plt.imshow(subdural_img, cmap=plt.cm.bone)
    plt.title('Subdural/Blood')
    sdpath = 'static/images/'+filename+'_subdural_img.png'
    plt.savefig(sdpath)

    #plt.imshow(bone_img, cmap=plt.cm.bone)
    plt.imshow(bone_img, cmap=plt.cm.bone)
    plt.title('Bone')
    bopath = 'static/images/'+filename+'_bone.png'
    plt.savefig(bopath)

    windowing = [ 
    {'url': bmpath}, 
    {'url': sdpath},
    {'url': bopath}]

    #bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)

    return windowing

if __name__ == '__main__':
    app.run(debug=True, port=9999)     


