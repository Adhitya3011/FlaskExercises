#imports
from flask import Flask, request

#init app
app = Flask(__name__)

#homepage 
@app.route('/')
def getIP():
    ip_add = request.remote_addr
    return '<h1> IP:  ' + ip_add

if __name__ == '__main__':
    app.run(debug=True)
