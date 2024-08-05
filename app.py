from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Charger le modèle
model = joblib.load('meilleur_modele.pkl')

# Initialiser Flask
app = Flask(__name__)

# Point de terminaison pour les prédictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    
    # Faire une prédiction
    prediction = model.predict(df)
    
    # Retourner la prédiction comme JSON
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
