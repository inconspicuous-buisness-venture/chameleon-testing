from flask import Flask, render_template, request, jsonify
from model import ModelAPIs

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/app')
def application():
    return render_template('app.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    threshold_value = data['thresholdValue']
    temperature_value = data['temperatureValue']
    iterations_value = data['iterationsValue']
    duration_value = data['durationValue']
    text_content = data['text']
    selected_model = data['model']
    selected_algorithm = data['algorithm']
    
    response = ModelAPIs.gemini_request(temperature_value, duration_value, text_content)
    
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
