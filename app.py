import keras
import numpy as np
from keras.applications import mobilenet
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from flask import Flask, render_template, request
from werkzeug import secure_filename
app = Flask(__name__)
@app.route('/')
def test():
   return "hello"
@app.route('/upload')
def upload_file():
   return render_template('upload.html')
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file1():
   if request.method == 'POST':
      f = request.files['file']
      mobilenet_model = mobilenet.MobileNet(weights='mobilenet_1_0_224_tf.h5')
      original = load_img(f, target_size=(224, 224))
      numpy_image = img_to_array(original)
      image_batch = np.expand_dims(numpy_image, axis=0)
      processed_image = mobilenet.preprocess_input(image_batch)

      predictions = mobilenet_model.predict(processed_image)
      
      label = decode_predictions(predictions)
      
      return str(label)
      
		
if __name__ == '__main__':   
   app.run(debug = True)

