from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import DialogKoGPT2
from classification import ClassificationModel
from kogpt2_transformers import get_kogpt2_tokenizer
import torch
import pandas as pd
import requests as req
import random
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(
    f"{os.getcwd()}/chatbot_model.pth", map_location=device)

model = DialogKoGPT2()
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

tokenizer = get_kogpt2_tokenizer()


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
    conn_papago = papago_api(
        "https://openapi.naver.com/v1/papago/n2mt", req_text["text"])

    if ClassificationModel(conn_papago).predict()[-1] > 0.5:
        return jsonify({
            "score": str(ClassificationModel(conn_papago).predict()[-1]),
            "res": "positive",
            "translate": conn_papago
        })
    else:
        return jsonify({
            "score": str(ClassificationModel(conn_papago).predict()[-1]),
            "res": "negative",
            "translate": conn_papago
        })


@app.route("/chat", methods=["POST"])
def chat():
    print(request.get_data())
    req_text = json.loads(request.get_data())
    tokenized_indexs = tokenizer.encode(req_text["text"])

    input_ids = torch.tensor(
        [tokenizer.bos_token_id, ] + tokenized_indexs + [tokenizer.eos_token_id]).unsqueeze(0)
    sample_output = model.generate(
        input_ids=input_ids
    )

    return jsonify(tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs)+1:], skip_special_tokens=True))


@app.route("/recommend", methods=["POST"])
def recommend():
    req_text = json.loads(request.get_data())

    if random.choice(["책", "노래"]) == "책":
        if req_text["res"] == "positive":
            joy_book = pd.read_csv(os.getcwd() + "/dataset/joy_book.csv")
            joy_book_idx = random.randrange(0, joy_book.__len__())
            dic = {
                "res": "book",
                "title": joy_book.iloc[joy_book_idx][0],
                "author": joy_book.iloc[joy_book_idx][1],
                "link": joy_book.iloc[joy_book_idx][-1],
                "emotion": "positive",
            }
            return jsonify(dic)
        else:
            sad_book = pd.read_csv(os.getcwd() + "/dataset/sad_book.csv")
            sad_book_idx = random.randrange(0, sad_book.__len__())
            dic = {
                "res": "book",
                "title": sad_book.iloc[sad_book_idx][0],
                "author": sad_book.iloc[sad_book_idx][1],
                "link": sad_book.iloc[sad_book_idx][-1],
                "emotion": "negative",
            }
            return jsonify(dic)
    else:
        if req_text["res"] == "positive":
            joy_song = pd.read_csv(os.getcwd() + "/dataset/joy_music.csv")
            joy_song_idx = random.randrange(0, joy_song.__len__())
            dic = {
                "res": "music",
                "author": joy_song.iloc[joy_song_idx][0],
                "title": joy_song.iloc[joy_song_idx][1],
                "link": joy_song.iloc[joy_song_idx, -1].split("/track/")[-1].split("?")[0],
                "emotion": "positive",
            }
            return jsonify(dic)
        else:
            sad_song = pd.read_csv(os.getcwd() + "/dataset/sad_music.csv")
            sad_song_idx = random.randrange(0, sad_song.__len__())
            dic = {
                "res": "music",
                "author": sad_song.iloc[sad_song_idx][0],
                "title": sad_song.iloc[sad_song_idx][1],
                "link": sad_song.iloc[sad_song_idx, -1].split("/track/")[-1].split("?")[0],
                "emotion": "negative",
            }
            return jsonify(dic)


print("Flask Server On")
app.run()
