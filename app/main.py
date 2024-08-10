from flask import Blueprint, render_template, request
from .gemini_api import fetch_sustainable_materials, fetch_recommendations
from .model import train_model, make_predictions

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Fetch sustainable materials data
        materials_df = fetch_sustainable_materials()

        # Train the model and make predictions
        if materials_df is not None:
            model, predictions = train_model(materials_df)
            recommendations = []

            for index, row in materials_df.iterrows():
                rec = fetch_recommendations(row['id'])
                recommendations.append({
                    'name': row['name'],
                    'predicted_adoption_rate': predictions[index],
                    'description': rec['description'] if rec else 'No recommendations available'
                })

            return render_template('results.html', recommendations=recommendations)
    
    return render_template('index.html')
