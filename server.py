#!/usr/bin/env python

from flask import Flask
from functools import wraps
from flask import request, abort
import ssl
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
                print(key)
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
    return 'Good\n'

@app.route('/')
def index_page():
    return 'Test'

app.run(host='0.0.0.0',port='443', debug = False/True, ssl_context=context)
