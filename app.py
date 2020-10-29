from flask import Flask, redirect, render_template, request, session, url_for
import flask
import logging as logger
logger.basicConfig(level="DEBUG")
import os
import smtplib
import config
import numpy as np
import tensorflow as tf
import cv2
from glob import glob
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
import shutil
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, patch_request_class, ALL         # import flask


app = Flask(__name__)             # create an app instance
dropzone = Dropzone(app)
# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*, .pdf, .txt'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'
app.config['SECRET_KEY'] = 'supersecretkeygoeshere'
app.config['DROPZONE_MAX_FILE_SIZE'] = 16 # set maximum file size, default is 3MB
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Uploads settings
app.config['UPLOADED_FILES_DEST'] = os.getcwd() + '/uploads'
files = UploadSet('files', ALL)
configure_uploads(app, files)
patch_request_class(app)

#########################################

def LeNet_build(numChannels, imgRows, imgCols, numClasses, weightsPath=None):
    # print width, height, depth
    model = Sequential()

    inputShape = (imgRows, imgCols, numChannels)

    # # if we are using "channels first", update the input shape
    # if K.image_data_format() == "channels_first":
    #     inputShape = (numChannels, imgRows, imgCols)
   
    # first set of CONV => RELU => POOL
    model.add(Conv2D(20, (5, 5), padding="same",
        input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # set of FullyConnected => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(numClasses))
    model.add(Activation("sigmoid"))

    # if a weights path is supplied (inicating that the model was
    # pre-trained), then load the weights
    if weightsPath is not None:
        model.load_weights(weightsPath)

    # return the constructed network architecture
    return model


########################################


#########################################
image_list = []
@app.route('/selectFiles', methods=['GET', 'POST'])
def selectFiles():
    global image_list
    
    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
    # list to hold our uploaded image urls
    file_urls = session['file_urls']
    # handle image upload from Dropzone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            
            # save the file with to our photos folder
            filename = files.save(
                file,
                name=file.filename    
            )
            print("################### FILE NAME TYPE", type(filename))
                
            if "txt" not in filename:
                fn = "./uploads/" + filename
                image_list.append(fn)
                
            # append image urls
            file_urls.append(files.url(filename))
            
        session['file_urls'] = file_urls
        return "uploading..."
    # return dropzone template on GET request    
    return render_template('indexOld.html')
#########################################
@app.route('/results')
def results():
    
    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('index'))
        
    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
    session.pop('file_urls', None)
    
    return render_template('results.html', file_urls=file_urls)


###########################
xRay_path = './uploads'
covid_dir = glob(xRay_path+'/*')

images=[]

@app.route("/getResults", methods=['GET', 'POST'])
def getResults():
    print("################# IN GET_RESULTS ##############")
    print("******** IMAGE_LIST*************")
    print(image_list)
    for i in range(len(image_list)):
        image = cv2.imread(image_list[i])
        image = cv2.resize(image,(50,50))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        
    print("*************************")
    print("type(images): ", type(images))
    print("len(images): ", len(images))
    print("images[0].shape: ", images[0].shape)
    print("*************************")
    
    
    test = np.expand_dims(images[0], axis=0)
    print("test.type: ", type(test))
    print("test.shape: ", test.shape)
    
    model = tf.keras.models.load_model('./model/model.h5')
    y_pred1 = model.predict(test)
    
    print("*************************")
    print("MODEL PREDICTION")
    print(y_pred1)
    print("*************************")

    y_pred_bin1 = np.argmax(y_pred1, axis=1)
    
    print("*************************")
    print("MODEL PREDICTION AFTER ARGMAX")
    print(y_pred_bin1)
    print("*************************")
    
    #result = '<body bgcolor="powderblue"><h1>COVID status:</h1> <div class="aligncenter"></div></body>'
    
    image_list.clear()
    images.clear()
        
    for f in glob('./uploads/*'):
        os.remove(f)
    
    status = "NULL"
    if y_pred_bin1[0] == 0:
        #status = result + "\n" + "Patient has Corona Virus symptoms!"
        return render_template('results.html', result = "Patient has Corona Virus symptoms!")
        
    elif y_pred_bin1[0] == 1:
        #status = result + "\n" + "Patient is Corona free!!! :)"
        return render_template('results.html', result = "Patient is Corona free!!! :)")
    
    return status
    

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')



@app.route("/contact", methods=['POST'])
def contact():
    try:
        fname = flask.request.form['fname']
        lname = flask.request.form['lname']
        mail = flask.request.form['email']
        sbjct = flask.request.form['subject']
        msg1 = flask.request.form['message']
        msg2 = "Name: " + fname + " " + lname + "\n" + "Email ID: " + mail + "\n" + "Message: " + msg1
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login(config.EMAIL_ADDRESS, config.PASSWORD)
        message = 'Subject: {}\n\n{}'.format(sbjct, msg2)
        server.sendmail(config.EMAIL_ADDRESS, config.EMAIL_ADDRESS, message)   #server.sendmail(fromaddr, toaddrs, msg)
        server.quit()
        #return("Your message has been sent. Thank you!")
        return render_template('index.html', messageMail = "Your message has been sent. Thank you!")
        
    except:
        #return("Something went wrong \N{worried face} Email failed to send.")
        return render_template('index.html', messageMail = "Something went wrong \N{worried face} Email failed to send.")
        
        

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if request.path == '/getResults':
        return response
    else:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    

if __name__ == "__main__":
    logger.debug("Starting Flask Server")
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run()
    shutil.rmtree('./uploads')
    del image_list
    del images
    print("bye")