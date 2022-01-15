from flask import Flask, request, jsonify
from flask_cors import CORS
from module import Model
import pandas as pd
import requests as req
import random
import json
import os

def papago_api(url, text):
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Naver-Client-Id": "hSGgxvmA0A_pFdvv6yyw",
        "X-Naver-Client-Secret": "zhb3k_X5OQ"
    }

    params = {
        "source": "ko",
        "target": "en",
        "text": text
    }
    
    response = req.post(url, headers=headers, data=params)
    return response.json()["message"]["result"]["translatedText"]

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

@app.route("/predict", methods=["POST"])
def predict():
    req_text = json.loads(request.get_data())
    conn_papago = papago_api("https://openapi.naver.com/v1/papago/n2mt", req_text["text"])

    if Model(conn_papago).predict()[-1] > 0.5:
        return jsonify({
          "score" : str(Model(conn_papago).predict()[-1]),
          "res" : "positive",
          "translate" : conn_papago
        })
    else:
        return jsonify({
          "score" : str(Model(conn_papago).predict()[-1]),
          "res" : "negative",
          "translate" : conn_papago
        })

@app.route("/recommend", methods=["POST"])
def recommend():
    req_text = json.loads(request.get_data())

    if random.choice(["책", "노래"]) == "책":
        if req_text["res"] == "positive":
            joy_book = pd.read_csv(os.getcwd() + "/dataset/joy_book.csv")
            joy_book_idx = random.randrange(0, joy_book.__len__())
            dic = {
                "res" : "book",
                "title" : joy_book.iloc[joy_book_idx][0],
                "author" : joy_book.iloc[joy_book_idx][1],
                "link" : joy_book.iloc[joy_book_idx][-1],
                "emotion" : "positive",
            }
            return jsonify(dic)
        else:
            sad_book = pd.read_csv(os.getcwd() + "/dataset/sad_book.csv")
            sad_book_idx = random.randrange(0, sad_book.__len__())
            dic = {
                "res" : "book",
                "title" : sad_book.iloc[sad_book_idx][0],
                "author" : sad_book.iloc[sad_book_idx][1],
                "link" : sad_book.iloc[sad_book_idx][-1],
                "emotion" : "negative",
            }
            return jsonify(dic)
    else:
        if req_text["res"] == "positive":
            joy_song = pd.read_csv(os.getcwd() + "/dataset/joy_music.csv")
            joy_song_idx = random.randrange(0, joy_song.__len__())
            dic = {
                "res" : "music",
                "author" : joy_song.iloc[joy_song_idx][0],
                "title" : joy_song.iloc[joy_song_idx][1],
                "link" : joy_song.iloc[joy_song_idx, -1].split("/track/")[-1].split("?")[0],
                "emotion" : "positive",
            }
            return jsonify(dic)
        else:
            sad_song = pd.read_csv(os.getcwd() + "/dataset/sad_music.csv")
            sad_song_idx = random.randrange(0, sad_song.__len__())
            dic = {
                "res" : "music",
                "author" : sad_song.iloc[sad_song_idx][0],
                "title" : sad_song.iloc[sad_song_idx][1],
                "link" : sad_song.iloc[sad_song_idx, -1].split("/track/")[-1].split("?")[0],
                "emotion" : "negative",
            }
            return jsonify(dic)
app.run()