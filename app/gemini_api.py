import requests
import pandas as pd

API_KEY = 'app/api.txt'  # Replace with your Gemini API key in api.txt file
BASE_URL = 'https://gemini.google.dev/api/v1/'

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

def fetch_sustainable_materials():
    """Fetch sustainable materials data from the Gemini API."""
    materials_endpoint = f'{BASE_URL}materials'
    response = requests.get(materials_endpoint, headers=headers)

    if response.status_code == 200:
        materials_data = response.json()
        return pd.DataFrame(materials_data['materials'])
    else:
        print(f"Failed to fetch materials: {response.status_code}")
        return None

def fetch_recommendations(material_id):
    """Fetch recommendations for a given material."""
    recommendations_endpoint = f'{BASE_URL}materials/{material_id}/recommendations'
    response = requests.get(recommendations_endpoint, headers=headers)

    if response.status_code == 200:
        recommendations_data = response.json()
        if recommendations_data['recommendations']:
            return recommendations_data['recommendations'][0]  # Return the first recommendation
        else:
            return None
    else:
        print(f"Failed to fetch recommendations: {response.status_code}")
        return None
