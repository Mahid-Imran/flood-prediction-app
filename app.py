from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model  = joblib.load('flood_model.pkl')
scaler = joblib.load('scaler.pkl')

FEATURES = [
    'MonsoonIntensity','TopographyDrainage','RiverManagement',
    'Deforestation','Urbanization','ClimateChange','DamsQuality',
    'Siltation','AgriculturalPractices','Encroachments',
    'IneffectiveDisasterPreparedness','DrainageSystems',
    'CoastalVulnerability','Landslides','Watersheds',
    'DeterioratingInfrastructure','PopulationScore','WetlandLoss',
    'InadequatePlanning','PoliticalFactors'
]

def risk_level(prob):
    if prob < 0.3:   return "Low Risk",      "#28a745"
    elif prob < 0.6: return "Medium Risk",   "#ffc107"
    elif prob < 0.8: return "High Risk",     "#fd7e14"
    else:            return "Critical Risk", "#dc3545"

@app.route('/')
def home():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    data   = request.get_json()
    values = [float(data[f]) for f in FEATURES]

    row = dict(zip(FEATURES, values))
    row['Rain_Urban']        = row['MonsoonIntensity'] * row['Urbanization']
    row['Drainage_Rain']     = row['DrainageSystems']  * row['MonsoonIntensity']
    row['Climate_Deforest']  = row['ClimateChange']    * row['Deforestation']
    row['Dam_Silt']          = row['DamsQuality']      * row['Siltation']
    row['EnvironmentalRisk'] = row['MonsoonIntensity'] + row['ClimateChange'] + row['Deforestation'] + row['Urbanization']

    all_cols   = FEATURES + ['Rain_Urban','Drainage_Rain','Climate_Deforest','Dam_Silt','EnvironmentalRisk']
    all_values = [row[c] for c in all_cols]

    scaled = scaler.transform([all_values])
    prob   = float(model.predict(scaled)[0])
    prob   = max(0.0, min(1.0, prob))

    risk, color = risk_level(prob)
    return jsonify({
        'probability': round(prob * 100, 1),
        'risk':  risk,
        'color': color
    })

if __name__ == '__main__':
    app.run(debug=True)