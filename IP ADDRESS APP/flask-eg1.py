from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/test2", methods=['GET','POST'])
def test2():
    if request.method() == 'GET':
        return "<p>Hello, now you hage a new function and the method is GET!</p>"
    else:
        return "<p>Hello, now you hage a new function and the method is POST!</p>"

if __name__ == '__main__':
    app.run(debug=True)