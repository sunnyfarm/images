#!/usr/bin/env python

from flask import Flask, Response, request, abort
from functools import wraps
import ssl
import json
import cv2
import numpy as np
import jsonpickle
import base64
from werkzeug import secure_filename

context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain('server.crt', 'server.key')

app = Flask(__name__)

def require_appkey(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        input_key = ""
        if request.args.get('key'):
            input_key = request.args.get('key')
        if request.headers.get('x-api-key'):
            input_key = request.headers.get('x-api-key')

        try:
            apikey = open('api.key', 'r')
            key=apikey.readline().replace('\n', '')
            
            while key:
                if input_key == key:
                    return view_function(*args, **kwargs)
                key = apikey.readline().replace('\n', '')
        finally:
            apikey.close()

        abort(401)
    return decorated_function

@app.route('/api/v1/query', methods=['POST'])
@require_appkey
def query():
    json_str = request.json
    json_out = json.loads(json_str)
    if json_out['image'] is not None:
        image = json_out['image']
        data = base64.b64decode(image)
        try:
            f = open('out.jpg', 'wb') 
            f.write(data)
        finally:
            f.close()
    #print(request.json, request.data, request.files)
    
    #image = f['files']
    #json = f['json']
    
    # decode image
    response = {'message': 'image received'}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/')
def index_page():
    return 'Test'

app.run(host='0.0.0.0',port='443', debug = False/True, ssl_context=context)
