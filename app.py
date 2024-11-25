from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from neural_networks import visualize

app = Flask(__name__)

# Define the main route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle experiment parameters and trigger the experiment
@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    activation = request.json['activation']
    lr = float(request.json['lr'])
    step_num = int(request.json['step_num'])

    result_gif = f"results/visualize_{activation}.gif"

    # Run the experiment with the provided parameters
    visualize(activation, lr, step_num, result_gif)
    
    return jsonify({
        "result_gif": result_gif if os.path.exists(result_gif) else None,
    })

# Route to serve result images
@app.route('/results/<filename>')
def results(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(debug=True)