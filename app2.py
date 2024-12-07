from flask import Flask, request, jsonify, render_template
from setfit import SetFitModel  
from flask_cors import CORS

model = SetFitModel.from_pretrained("my_trained_setfit_model")

app = Flask(__name__)
CORS(app)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.get('feature1')
    
    predictions = model.predict([data])
    print("Prediction is: ", predictions)
    
    label = "Stressed" if predictions[0] == "negative" else "Not Stressed"
    
    return jsonify({"prediction": label})

# Route for HTML form
@app.route('/')
def index():
    return render_template('html/stressmeasure.html')  

if __name__ == "__main__":
    app.run(debug=True)
