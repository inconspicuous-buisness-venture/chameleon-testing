from flask import Flask, render_template, request, jsonify
from model import *
from tools import *

from datetime import datetime
import json
import os

app = Flask(__name__)

LOG_DIR = './chats'
LOG_NAME = f'chat_{datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")}.json'
LOG_PATH = os.path.join(LOG_DIR, LOG_NAME)
CHAT_LOG = []


def update_chat_log(in_request, output):
    CHAT_LOG.append({"request": in_request, "response": output})
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    with open(LOG_PATH, 'w') as f:
        json.dump(CHAT_LOG, f, indent=2)


@app.route('/')
def index():
    return render_template('main.html')


@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    threshold_value = data['thresholdValue']
    temperature_value = data['temperatureValue']
    iterations_value = data['iterationsValue']
    duration_value = data['durationValue']
    text_content = data['text']
    prompt_content = data['prompt']
    selected_model = data['model']
    selected_algorithm = data['algorithm']
    
    response = gemini_request(temperature_value, duration_value, text_content, prompt_content)
    # detectability = asyncio.run(main(response["output"]))
    detectability = 100
    response['score'] = detectability
    print("SCORE: " + str(detectability))
    
    update_chat_log(data, response)
    
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
