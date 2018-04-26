import base64
import datetime
import io
import datetime
import cv2
import pandas as pd
import numpy as np
import inspect
import os
import os.path
import dash
import tensorflow as tf
import facenet2
from tensorflow.python.training import training
from tensorflow.python.platform import gfile
import sklearn
from sklearn import metrics
import math
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from PIL import Image
from scipy import misc
import datetime
import requests
import pickle

click = 0
click_cam = 0
test = [['N']]
corr1 = [0,0,0,0,0]
new1 = ''
new2 = ''
new3 = ''
this_string1 = ''
this_string2 = ''
this_string3 = ''
this_string4 = ''


setwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+'/'
prewd = os.path.abspath(os.path.join(setwd, os.pardir))

def test_sex_pred(test_sex_pred_global):
    if test_sex_pred_global == 1:
        sex_part = html.Div([
                html.Img(src=findsrc2(prewd+"/data/image/man.png"), style={ 'display': 'block',
                                        'marginLeft': 'auto', 
                                        'marginRight': 'auto'}),
                html.H2(children = 'man : ' + str(format(round(loaded_model_sex.predict_proba(test)[0][1],2), '.0%')),style={
                                         'textAlign': 'center',
                                        'color': colors['text']
                                        })
            
            ])
        return sex_part

    if test_sex_pred_global == -1:
        sex_part = html.Div([
                html.Img(src=findsrc2(prewd+"/data/image/woman.png"), style={ 'display': 'block',
                                        'marginLeft': 'auto', 
                                        'marginRight': 'auto'}),
                html.H2(children = 'woman : ' + str(format(round(loaded_model_sex.predict_proba(test)[0][0],2), '.0%')),style={
                                         'textAlign': 'center',
                                        'color': colors['text']
                                        })
            
            ])
        return sex_part

def test_chubby_pred(test_chubby_pred_global):
    if test_chubby_pred_global == 1:
        chubby_part = html.Div([
                html.Img(src=findsrc2(prewd+"/data/image/fat.jpeg"), style={ 'display': 'block',
                                        'marginLeft': 'auto', 
                                        'marginRight': 'auto'}),
                html.H2(children = 'chubby : ' + str(format(round(loaded_model_chubby.predict_proba(test)[0][1],2), '.0%')),style={
                                         'textAlign': 'center',
                                        'color': colors['text']
                                        })
            
            ])
        return chubby_part
    if test_chubby_pred_global == -1:
        chubby_part = html.Div([
                html.Img(src=findsrc2(prewd+"/data/image/thin.jpeg"), style={ 'display': 'block',
                                        'marginLeft': 'auto', 
                                        'marginRight': 'auto'}),
                html.H2(children = 'thin : ' + str(format(round(loaded_model_chubby.predict_proba(test)[0][0],2), '.0%')),style={
                                         'textAlign': 'center',
                                        'color': colors['text']
                                        })
            
            ])   
        return chubby_part
    
def test_attr_pred(test_attr_pred_global):
        attr_part = html.Div([
                html.Img(src=findsrc2(prewd+"/data/image/attractive.jpeg"), style={ 'width': '225px',
                                         'height': '225px',
                                         'display': 'block',
                                         'marginLeft': 'auto', 
                                         'marginRight': 'auto'}),
                html.H2(children = 'attractive : ' + str(format(round(loaded_model_attr.predict_proba(test)[0][1],2), '.0%')),style={
                                         'textAlign': 'center',
                                        'color': colors['text']
                                        })
            
            ])
        return attr_part

def face_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(prewd+"/lib/haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    )
    print("Found {0} faces!".format(len(faces)))
    if len(faces) == 0:
        return('N')
    if len(faces) > 1:
        faces = [faces[0]]
    for (x, y, w, h) in faces:
        tau = 0.4
        k = int(tau*h)
        t = int(tau*w)
        x1 = np.max([0,y-k])
        x2 = np.min([image.shape[0],y-k+int((1+2*tau)*h)])
        y1 = np.max([0,x-t])
        y2 = np.min([image.shape[1],x-t+int((1+2*tau)*w)])
        z = np.min([x2-x1,y2-y1])
        crop_img = image[x1:x1+z,y1:y1+z]
        img = Image.fromarray(crop_img, 'RGB')
        img.save('uploadpicture.jpg')
        img = Image.open('uploadpicture.jpg')
        img = img.resize((160,160),Image.ANTIALIAS)
        img.save('resized.jpg')
        img = misc.imread('resized.jpg')
        images = np.zeros((1, 160, 160,3))
        if img.ndim == 2:
            img = facenet2.to_rgb(img)
        if len(img[0][0]) == 4:
            img=img[:,:,0:3]
        img = facenet2.prewhiten(img)
        images[0,:,:,:] = img
        return(main(images))

def main(image):
    
    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    # Run forward pass to calculate embeddings
    print('Calculating features for images')
    emb_array = np.zeros((1, embedding_size))
    feed_dict = { images_placeholder:image, phase_train_placeholder:False }
    emb_array[0] = sess.run(embeddings, feed_dict=feed_dict)
    this = np.matrix.round(emb_array,4)
    this = pd.DataFrame(this)
    return(this)

def findsrc(this):
    if this == '':
        return None
    print(this)
    image_filename = prewd +'/data/img_align_celeba/' + this # replace with your own image
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    return('data:image/jpeg;base64,'+str(encoded_image)[2:-1])

def findsrc2(this):
    if this == '':
        return None
    encoded_image = base64.b64encode(open(this, 'rb').read())
    return('data:image/jpeg;base64,'+str(encoded_image)[2:-1])


def parse_contents(contents,local):
    global test
    global this_string1
    global this_string2
    global this_string3
    global this_string4
    global corr1
    global test_sex
    global test_chubby
    global test_attr
    
    corr1 = [0,0,0,0,0]
    if local:
        content_type, content_string = contents.split(',')
        contents = base64.b64decode(content_string)
    img = np.array(Image.open(io.BytesIO(contents)))
    test = face_detector(img)
    if test[0][0] == 'N':
        return None
    this = sklearn.metrics.pairwise.cosine_similarity(train, test, dense_output=True)
    test_sex_pred_global = loaded_model_sex.predict(test)
    test_sex = test_sex_pred(test_sex_pred_global)
    test_chubby_pred_global = loaded_model_chubby.predict(test)
    test_chubby = test_chubby_pred(test_chubby_pred_global)
    test_attr_pred_global = loaded_model_attr.predict(test) 
    test_attr = test_attr_pred(test_attr_pred_global)    
    flat_list = [item for sublist in this for item in sublist]
    print(flat_list[0])
    t1 = np.argsort(flat_list)[::-1][:1]
    t2 = np.argsort(flat_list)[::-1][1:2]
    t3 = np.argsort(flat_list)[::-1][2:3]
    t4 = np.argsort(flat_list)[::-1][3:4]
    this_string1 = '{0:06}'.format(t1[0]+1) + ".jpg"
    this_string2 = '{0:06}'.format(t2[0]+1) + ".jpg"
    this_string3 = '{0:06}'.format(t3[0]+1) + ".jpg"
    this_string4 = '{0:06}'.format(t4[0]+1) + ".jpg"


    corr1 = np.sort(flat_list)[::-1][0:4]





def get_banner():
    banner =  html.Div([
                        html.Img(src= "http://chuantu.biz/t6/293/1524541998x1822612335.jpg",height='100%', width = '100%')
                        ], className='banner', style={'background-size': 'cover'})
    return banner




def get_menu():
    menu = html.Div([
                     
                     dcc.Link('Similarity', href='/Similarity', className="tab first",
                              style = {'background-color': '#FFFFFF','font-size':'20px','color': colors['text'],
                                       'padding': '12px 25px','width':'28%','text-align': 'center',
                                       'text-decoration':'none','display': 'inline-block','borderWidth': '1px',
                                       'borderStyle': 'solid','borderRadius':'5px','textAlign':'center',
                                       'border-color': '#B8B8B8'}),
                     
                     dcc.Link('Attributes', href='/Attributes', className="tab",
                              style = {'background-color': '#FFFFFF','font-size':'20px','color': colors['text'],
                                       'padding': '12px 25px','width':'28%','text-align': 'center',
                                       'text-decoration':'none','display': 'inline-block','borderWidth': '1px',
                                       'borderStyle': 'solid','borderRadius':'5px','textAlign':'center',
                                       'border-color': '#B8B8B8'}),
                     
                     dcc.Link('Contact Info', href='/Contact',className="tab",
                              style = {'background-color': '#FFFFFF','font-size':'20px','color': colors['text'],
                                       'padding': '12px 25px','width':'28%','text-align': 'center',
                                       'text-decoration':'none','display': 'inline-block','borderWidth': '1px',
                                       'borderStyle': 'solid','borderRadius':'5px','textAlign':'center',
                                       'border-color': '#B8B8B8'})
                     
                     ], className="row ")
    return menu


def get_watermark():
    watermark = html.Div([
                         html.Br([]),
                         html.Br([]),
                         html.P(["@Spring2018-ADS-Project5-Group-3 "],
                                 style={'text-align': 'right', 'color':colors['text']})
                          ])
    return watermark

def get_contact():
    contact = html.Div([
                        dcc.Markdown('''
                                     **We are Columbia University students at Department of Statistics.**
                                     
                                     **Team members and email address are as follows:**
                                    
                                       - Guo, Du [dg2999@columbia.edu]()
                                       - Guo, Tao [tg2620@columbia.edu]()
                                       - Jiang, Yiran [yj2462@columbia.edu]()
                                       - Liu, Fangbing [fl2476@columbia.edu]()
                                       - Wang, Jingyi [jw3592@columbia.edu]()
                                     '''.replace('    ', '')
                                    )
                        ], style={'color':colors['text'], 'text-align':'center'})
    return contact



app = dash.Dash()

app.scripts.config.serve_locally = True

colors = {
    'background': '#F9E79F',
    'text': '#6E2C00'
}


noPage = html.Div([  # 404
                   
                   html.P(["404 Page not found"])
                   
                   ], className="no-page")



##Page1
def cal_Sim():
    if set(corr1) != set([0,0,0,0]):
        Similarity= html.Div([
                              # uploaded image display and output
                              
                              
                              # real-person image
                              
                              html.Div([
                                        # HTML images accept base64 encoded strings in the same format
                                        # that is supplied by the upload
                                        html.Div([html.Img(src=findsrc2('resized.jpg'))],
                                                 style={
                                                 'width': '160px',
                                                 'height': '160px',
                                                 'display': 'block',
                                                 'marginLeft': 'auto',
                                                 'marginRight': 'auto',
                                                 'borderWidth': '6px',
                                                 'borderStyle': 'solid',
                                                 'borderRadius': '50%',
                                                 'overflow': 'hidden',
                                                 'border-color': '#FFFFFF'}),
                                        html.Br(),
                                        html.Br(),
                                        html.Br(),
                                        
                                        html.Div([
                                                  html.Div([ html.H4('{:.0%}'.format(round(corr1[0],2)),style={'textAlign': 'center'}),
                                                            html.Img(src=findsrc(this_string1),style={'display': 'block','marginLeft': 'auto','marginRight': 'auto'})], className="three columns"),
                                                  html.Div([ html.H4('{:.0%}'.format(round(corr1[1],2)),style={'textAlign': 'center'}),
                                                            html.Img(src=findsrc(this_string2),style={'display': 'block','marginLeft': 'auto','marginRight': 'auto'})], className="three columns"),
                                                  html.Div([ html.H4('{:.0%}'.format(round(corr1[2],2)),style={'textAlign': 'center'}),
                                                            html.Img(src=findsrc(this_string3),style={'display': 'block','marginLeft': 'auto','marginRight': 'auto'})], className="three columns"),
                                                  html.Div([html.H4('{:.0%}'.format(round(corr1[3],2)),style={'textAlign': 'center'}),
                                                            html.Img(src=findsrc(this_string4),style={'display': 'block','marginLeft': 'auto','marginRight': 'auto'})], className="three columns"),
                                                  ], className="row", style={
                                                 'width': '900px',
                                                 'height': '300px',
                                                 'display': 'block',
                                                 'marginLeft': 'auto',
                                                 'marginRight': 'auto',
                                                 'borderWidth': '2px',
                                                 'borderStyle': 'solid',
                                                 #'borderRadius': '5px',
                                                 'overflow': 'hidden',
                                                 'backgroundColor':'#F8F8F8',
                                                 'color': colors['text']})
                                        ])
                              ])
        return Similarity
    else:
        Similarity =    html.Div([
                                  
                                  html.Div([
                                            html.P(["        "],style={'display': 'inline-block', 'float':'left'}),
                                            html.Div([html.Img(src=findsrc2(prewd+"/data/image/question2.jpg"))],
                                                     style={
                                                     'width': '160px',
                                                     'height': '160px',
                                                     'display': 'block',
                                                     'marginLeft': 'auto',
                                                     'marginRight': 'auto',
                                                     'borderWidth': '6px',
                                                     'borderStyle': 'solid',
                                                     'borderRadius': '50%',
                                                     'overflow': 'hidden',
                                                     'border-color': '#FFFFFF'}),
                                            
                                            html.Hr(),
                                            
                                            html.Div([
                                                      html.Img(src=findsrc2(prewd+"/data/image/question2.jpg")),
                                                      html.Hr(),
                                                      ],style={
                                                     'width': '160px',
                                                     'height': '160px',
                                                     'display': 'block',
                                                     'marginLeft': 'auto',
                                                     'marginRight': 'auto',
                                                     'borderWidth': '2px',
                                                     'borderStyle': 'solid',
                                                     'borderRadius': '50%',
                                                     'overflow': 'hidden',
                                                     'border-color': '#FFFFFF'})
                                    
                                            ])
                                  ])
        return Similarity



##Page2
def call_stat():
    if test[0][0] == 'N':
        Stat = html.Div([
                html.Hr(),
                html.Img(src=findsrc2(prewd+"/data/image/question2.jpg"),
                     style={
                     'width': '160px',
                     'height': '160px',
                     'display': 'block',
                     'marginLeft': 'auto',
                     'marginRight': 'auto',
                     'borderWidth': '6px',
                     'borderStyle': 'solid',
                     'borderRadius': '50%',
                     'overflow': 'hidden',
                     'border-color': '#FFFFFF'}),
                html.H2('Sorry, we can not detect an Alien',style={
                         'textAlign': 'center',
                        'color': colors['text']
                        })
                     ])
        return Stat

    Stat = html.Div([
        html.Hr(),
        html.Div([
                html.Div([test_sex], className = 'four columns'),
                html.Div([test_chubby], className = 'four columns'),
                html.Div([test_attr], className = 'four columns')                
                ], className = 'row')
    ])
    return Stat
 

##Page3
def cal_cont():
    contact = html.Div([
                       get_contact()
                       ])
    return contact





#camera
def get_camera():
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        cv2.imshow('frame', rgb)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            out = cv2.imwrite(setwd + 'capture.jpg', frame)
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    image_filename = setwd + 'capture.jpg'
    print('step4')
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    a = 'data:image/jpeg;base64,'+str(encoded_image)[2:-1]
    parse_contents(a, local = True)


app.layout = html.Div(style={'backgroundColor':colors['background']},
                      children=[
                                html.Div([
                                          # header
                                          get_banner(),
                                          html.Br([]),
                                          # image
                                          
                                          html.Div([dcc.Upload(
                                                               id='upload-image',
                                                               children=html.Div(['Drag or ',
                                                                                  html.A('Select Files')
                                                                                  ]),
                                                               style={
                                                               'display':'block',
                                                               'marginLeft': 'auto',
                                                               'marginRight': 'auto',
                                                               'width': '40%',
                                                               'height': '40px',
                                                               'lineHeight': '30px',
                                                               'borderWidth': '1px',
                                                               'borderStyle': 'solid',
                                                               'borderRadius':'5px',
                                                               'textAlign': 'center',
                                                               'border-color': '#B8B8B8',
                                                               'color':colors['text']
                                                               },
                                                               # Allow multiple files to be uploaded
                                                               multiple=True
                                                               )],style={'width': '33.33%', 'display': 'inline-block', 'float':'left'})
                                          ]),
                                
                                html.Div([
                                          dcc.Input(
                                                    placeholder='Image URL',
                                                    id='input-box',
                                                    type='text',
                                                    style={'display':'inline-block'}),
                                          
                                          html.Button(
                                                      children='Submit',
                                                      id='button',
                                                      n_clicks=0,
                                                      style={'display':'inline-block'})
                                          ],
                                         style={'width': '33.33%','display': 'inline-block'}),
                                
                                html.Div([
                                          html.Button('Connect Camera',
                                                      id='button-2',
                                                      n_clicks=0,
                                                      style={'display':'inline-block'})
                                          ],
                                         style={'width': '33.33%', 'float': 'right', 'display': 'inline-block'}),
                                
                                html.P(["* Image Format: jpg/png       * Better to use full face image with clean background        * P - Take Photo, Q - Quit Camera"], style={
                                       'display':'inline-block',
                                       'width': '100%',
                                       'height': '30px',
                                       'textAlign': 'center',
                                       'color':'#9494b8'
                                       }),
                                html.Br([]),
                                # menu
                                get_menu(),
                                html.Br([]),
                                dcc.Location(id='url', refresh=False),
                                html.Div(id = 'output-image-upload'),
                                
                                get_watermark()
                                ])



@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents'),
               dash.dependencies.Input('button','n_clicks'),
               dash.dependencies.Input('url', 'pathname'),
               dash.dependencies.Input('button-2','n_clicks')],  #camera
              [dash.dependencies.State('input-box', 'value')])

def update_output(contents,n_clicks,pathname,push,value):
    global click
    global click_cam #camera
    global corr1
    global test
    global this_string1
    global this_string2
    global this_string3
    global this_string4
    global new1
    global new2
    global new3
    
    for i in range(1):
        if pathname != new3:
            break
        if n_clicks != click:
            new1 = ''
            click = n_clicks
            new2 = value
            try:
                response = requests.get(value)
                parse_contents(response.content,False)
            except:
                corr1 = [0,0,0,0]
                pass
            if pathname == '/' or pathname == '/Similarity':
                new3 = pathname
                return cal_Sim()
            elif pathname == '/Attributes':
                new3 = pathname
                return call_stat()
            elif pathname == '/Contact':
                new3 = pathname
                return cal_cont()
            else:
                return noPage
        if push != click_cam:
            click_cam = push
            try:
                get_camera()
            except:
                print('error loading camera')
        
            
        if contents is not None or value is not None:
            if contents != new1:
                new2 = ''
                new1 = contents
                parse_contents(contents[0],True)
                if pathname == '/' or pathname == '/Similarity':
                    new3 = pathname
                    return cal_Sim()
                elif pathname == '/Attributes':
                    new3 = pathname
                    return call_stat()
                elif pathname == '/Contact':
                    new3 = pathname
                    return cal_cont()
                else:
                    new3 = pathname
                    return noPage


    if pathname == '/' or pathname == '/Similarity':
        new3 = pathname
        return cal_Sim()
    elif pathname == '/Attributes':
        new3 = pathname
        return call_stat()
    elif pathname == '/Contact':
        new3 = pathname
        return cal_cont()
    else:
        new3 = pathname
        return noPage
    



app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.Session() as sess:
            print('Loading Data')
            train = np.load(prewd+"/data/feature.npy")
            # Load the model
            loaded_model_sex = pickle.load(open(prewd+"/lib/logistic_sex.pkl", 'rb'))
            loaded_model_chubby = pickle.load(open(prewd+"/lib/logistic_chubby.pkl", 'rb'))
            loaded_model_attr = pickle.load(open(prewd+"/lib/logistic_attractive.pkl", 'rb'))
            print('Loading feature extraction model')
            facenet2.load_model(prewd+"/lib/20180402-114759.pb")
            app.run_server(debug=True,use_reloader=False)
