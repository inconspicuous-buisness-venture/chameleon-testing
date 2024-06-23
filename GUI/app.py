from flask import Flask, render_template, request, jsonify
from model import *
from tools import *

app = Flask(__name__)

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
    detectability = asyncio.run(main(response["output"]))

    response['score'] = detectability
    print("SCORE: " + str(detectability))
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
