# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
from pathlib import Path
from flask import Flask, request
import tensorflow as tf

from clean_text import clean_texts


CWD = Path(__file__).parent.resolve()

model = tf.keras.models.load_model(CWD / 'models/cyberbullying-bdlstm.h5')

with open(CWD / 'models/tokenizer.json', encoding='UTF-8') as file:
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(file.read())

app = Flask(__name__)

@app.route('/cyberbullyingchatbotmsg', methods=['GET'])
def cyberbullyingchatbotmsg():
    messages = clean_texts([request.args.get("msg")], tokenizer)
    return {"score": model.predict(messages).tolist()[0][0]}

@app.route('/', methods=['GET'])
def main():
    return "Hello World"

@app.route('/echo', methods=['GET', 'POST'])
def echo():
    if 'echo' in request.args:
        return request.args.get('echo')
    else:
        return "Param not found", 404

app.run(debug=True)
