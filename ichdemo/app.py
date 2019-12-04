import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, url_for, redirect

app = Flask(__name__, static_url_path='/static')


@app.route("/")
def fileFrontPage():
    return render_template('index.html')

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
            #dcm.save(os.path.join('C:/Users/Public/Pictures', dcm.filename))
    return render_template('index.html', max=max_intensity, metadata= metadata, 
        url = "/static/images/rawimg.png")

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

if __name__ == '__main__':
    app.run(debug=True)     