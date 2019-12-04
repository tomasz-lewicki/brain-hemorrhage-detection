import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, url_for, redirect

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
        max_intensity = np.max(dcmarray)
        plt.imshow(dcmarray, cmap=plt.cm.bone)
        plt.savefig('static/images/rawimg.png')
        metadata = metaprint(ds)
        bsb= bsb_window(ds)
            #dcm.save(os.path.join('C:/Users/Public/Pictures', dcm.filename))
    return render_template('index.html', max=max_intensity, metadata= metadata, 
        url = "/static/images/rawimg.png", windowing = bsb)

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

def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)
    
    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380

    plt.imshow(brain_img, cmap=plt.cm.bone)
    plt.title('Brain Matter')
    plt.savefig('static/images/brain_img.png')

    plt.imshow(subdural_img, cmap=plt.cm.bone)
    plt.title('Subdural/Blood')
    plt.savefig('static/images/subdural_img.png')

    plt.imshow(soft_img, cmap=plt.cm.bone)
    plt.title('Soft Tissue')
    plt.savefig('static/images/soft_img.png')

    windowing = [ 
    {'url': "/static/images/brain_img.png"}, 
    {'url': "/static/images/subdural_img.png"},
    {'url': "/static/images/soft_img.png"}]

    #bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)

    return windowing

if __name__ == '__main__':
    app.run(debug=True)     