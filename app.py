from flask import Flask, request, render_template, request
import FakeNewsPrediction
import requests


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/linktest')
def linktest():
    return render_template("checkpage.html")

@app.route('/result')
def result():
    news = request.args['name']
    result  = FakeNewsPrediction.getprediction(news)
    return result
@app.route('/faq')
def faq():
    return render_template("faq.html")


if __name__ == "__main__":
    app.debug = True
    app.run()
