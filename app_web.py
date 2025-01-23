import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import os
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import nltk
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pydeck as pdk
import pydeck as pdk



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Charger le modèle SentenceTransformer
model = SentenceTransformer('Prashasst/anime-recommendation-model')

#Latitude and Longitude countries 

data = {
    "Country": ['Sri Lanka', 'Guatemala', 'Portugal', 'Suisse', 'Éthiopie', 'Bolivie', 'Namibie', 'France', 'Italie', 'Islande', 'Mexique', 'Argentine', 'Vietnam', 'Jamaïque', 'Colombie'],
    "Latitude": [7.8731, 15.7835, 39.3999, 46.8182, 9.145, -16.2902, -22.9576, 46.6034, 41.8719, 64.9631, 23.6345, -38.4161, 14.0583, 18.1096, 4.5709],
    "Longitude": [80.7718, -90.2308, -8.2245, 8.2275, 40.4897, -63.5887, 18.4904, 1.8883, 12.5674, -19.0208, -102.5528, -63.6167, 108.2772, -77.2975, -74.2973]
}

data = pd.DataFrame(data)

def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    
    lemmatizer = WordNetLemmatizer()
    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Retourner la liste de tokens
    return tokens

# Charger les données (assurez-vous que votre fichier ou dfFrame est correctement configuré)
@st.cache_data
def load_df():
    # Remplacez par votre méthode pour charger les données
    df = pd.read_csv("df_tourisme_translated.csv")  # Modifier avec le chemin de votre dfset
    df['Combined_description'] = df['description_translated'].fillna('') + " " + df['Detail_translated'].fillna('')
    df['Processed_Description'] = df['Combined_description'].apply(preprocess_text)
    df['Processed_Description'] = df['Processed_Description'].apply(lambda x: " ".join(x))
    df['Processed_Description'] = df['Processed_Description'].astype(str)
    df = df.merge(data, left_on="Pays", right_on="Country", how="left")
    return df

df = load_df()

# Titre de l'application
st.markdown("# 🌍 Destination Recommendations")
# Entrée utilisateur
st.sidebar.markdown("## 🧭 Filters")
query = st.sidebar.text_input("Enter your search:", "Looking for a place where I can do a roadtrip.")

# Générer l'embedding de la requête
query_embedding = model.encode([query])

# Générer les embeddings pour les descriptions
with st.spinner("Searching a place for you 🧑‍✈️🔍..."):
    embeddings = model.encode(df['Processed_Description'].tolist())
st.success("Places find 🎉!")
# Calculer les similarités cosinus
similarity_scores = cosine_similarity(query_embedding, embeddings)

df['Similarity_Score'] = similarity_scores[0]

# Trier les résultats
result_sorted = df[["Category",'Pays','Place', 'Hotel', 'Adresse', 'description_translated', 'Similarity_Score']].sort_values(by='Similarity_Score', ascending=False)

# Liste unique des pays et catégories disponibles
available_countries = sorted(df['Pays'].unique())
available_categories = sorted(df['Category'].unique())  

# Ajout des filtres
selected_countries = st.sidebar.multiselect("Filter by country :", available_countries, default=available_countries)
selected_categories = st.sidebar.multiselect("Filter by category :", available_categories, default=available_categories)

# Appliquer les filtres au DataFrame trié
filtered_results = result_sorted[
    (result_sorted['Pays'].isin(selected_countries)) &
    (result_sorted['Category'].isin(selected_categories))
]

# Afficher les meilleurs résultats filtrés
st.subheader("Your top 5 destinations 🛫")
st.table(filtered_results.head(5))

# Groupement des destinations par pays avec filtres
st.subheader("Recommendations by country 🛫")
destinations_par_pays = filtered_results.groupby('Pays')[['Category',"Place", 'description_translated', 'Similarity_Score']].apply(
    lambda x: x.sort_values('Similarity_Score', ascending=False)
)
st.write(destinations_par_pays.head(10))  # Afficher les 10 premières entrées


st.subheader("Interactive Map of Destinations 🗺️")
if "Latitude" in df.columns and "Longitude" in df.columns:
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=df["Latitude"].mean(),
            longitude=df["Longitude"].mean(),
            zoom=4,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'HexagonLayer',
                data=df,
                get_position='[Longitude, Latitude]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
            pdk.Layer(
                    'ScatterplotLayer',
                    data=df.head(5),  # Filtrer pour afficher seulement les pays sélectionnés
                    get_position='[Longitude, Latitude]',
                    get_color='[0, 255, 0, 160]',  # Couleur verte pour les pays sélectionnés
                    get_radius=300,
                    pickable=True
                ),
        ],
    ))

st.subheader("Interactive Map of Destinations 🗺️")
if "Latitude" in df.columns and "Longitude" in df.columns:
    # Filtrer les données pour la Colombie
    colombie_data = df[df["Country"] == "Colombie"]

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=4.5709	,
            longitude=-74.2973,
            zoom=4,
            pitch=50,
        ),
        layers=[
            # Couche pour tous les pays
            pdk.Layer(
                'HexagonLayer',
                data=df,
                get_position='[Longitude, Latitude]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
            # Couche pour les meilleurs résultats
            pdk.Layer(
                'ScatterplotLayer',
                data=df.head(5),  # Afficher les pays sélectionnés
                get_position='[Longitude, Latitude]',
                get_color='[0, 255, 0, 160]',  # Couleur verte
                get_radius=300,
                pickable=True,
            ),
            # Couche spécifique pour la Colombie
            pdk.Layer(
                'ScatterplotLayer',
                data=colombie_data,
                get_position='[Longitude, Latitude]',
                get_color='[0, 0, 255, 200]',  # Couleur bleue pour la Colombie
                get_radius=500,
                pickable=True,
            ),
        ],
    ))



st.download_button(
    label="Download results",
    data=filtered_results.head(5).to_csv(index=False),
    file_name='filtered_recommendations.csv',
    mime='text/csv',
)

#Proposer des restos: 
#le scrapping
from dataclasses import dataclass, asdict
import requests
import pandas as pd

@dataclass
class Item:
    Nom: str | None
    Address: str | None
    Note: float | None
    Telephone: str | None
    Serves_Vegetarian_Food: bool | None
    Price_Level: str | None
    Horaires: list[str] | None
    User_Rating_Count: int | None
    Country: str | None


def run_api(lat, lng):
    url = "https://places.googleapis.com/v1/places:searchNearby"
    api_key = "AIzaSyCBn-917H8fNTQ7grtFZ_nhfT0oiscCQ-8"

    payload = {
        "includedTypes": ["restaurant"],
        "maxResultCount": 20,
        "rankPreference": "DISTANCE",
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": lat,
                    "longitude": lng,
                },
                "radius": 1000
            }
        }
    }

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "places.displayName,places.formattedAddress,places.rating,"
            "places.nationalPhoneNumber,places.servesVegetarianFood,places.priceLevel,"
            "places.regularOpeningHours,places.userRatingCount"
        )
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"API call failed for coordinates ({lat}, {lng}) with status code {response.status_code}")
        return None

def parse_json(data, country):
    if not data or "places" not in data:
        return []
    places = data["places"]

    for place in places:
        yield asdict(Item(
            Nom=place.get("displayName", {}).get("text"),
            Address=place.get("formattedAddress"),
            Note=place.get("rating"),
            Telephone=place.get("nationalPhoneNumber"),
            Serves_Vegetarian_Food=place.get("servesVegetarianFood"),
            Price_Level=place.get("priceLevel"),
            Horaires=place.get("regularOpeningHours", {}).get("weekdayText"),
            User_Rating_Count=place.get("userRatingCount"),
            Country=country
        ))

def save_to_csv(leads, filename="output_restaurants.csv"):
    df = pd.DataFrame(columns=[
        "Nom", "Address", "Note", "Telephone", "Serves_Vegetarian_Food", 
        "Price_Level", "Horaires", "User_Rating_Count"
    ])
    for lead in leads:
        df = pd.concat([df, pd.DataFrame([lead])], ignore_index=True)

    # Suppression de la colonne 'Horaires'
    df.drop(columns=["Horaires"], inplace=True)

    # Suppression des lignes ayant des NaN dans 3 colonnes ou plus
    df = df.dropna(thresh=len(df.columns) - 2)

    # Enregistrer dans un fichier CSV
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def main():
    # Liste des coordonnées (latitude, longitude) avec leur pays
    locations = [
        ((7.8731, 80.7718), "Sri Lanka"),
        ((6.9271, 79.8612), "Sri Lanka"),
        ((8.3065, 81.0464), "Sri Lanka"),
        ((15.7835, -90.2308), "Guatemala"),
        ((14.6349, -90.5069), "Guatemala"),
        ((16.2500, -90.8103), "Guatemala"),
        ((39.3999, -8.2245), "Portugal"),
        ((38.7169, -9.1395), "Portugal"),
        ((41.1496, -8.6110), "Portugal"),
        ((46.8182, 8.2275), "Suisse"),
        ((47.3769, 8.5417), "Suisse"),
        ((46.9481, 7.4474), "Suisse"),
        ((9.1450, 40.4897), "Éthiopie"),
        ((9.0543, 38.7467), "Éthiopie"),
        ((14.3550, 38.2194), "Éthiopie"),
        ((-16.2902, -63.5887), "Bolivie"),
        ((-17.7833, -63.1810), "Bolivie"),
        ((-16.5000, -68.1193), "Bolivie"),
        ((-22.9576, 18.4904), "Namibie"),
        ((-22.5592, 17.0833), "Namibie"),
        ((-19.1791, 15.9579), "Namibie"),
        ((46.6034, 1.8883), "France"),
        ((48.8566, 2.3522), "France"),
        ((43.7102, 7.2620), "France"),
        ((41.8719, 12.5674), "Italie"),
        ((45.4642, 9.1900), "Italie"),
        ((41.9028, 12.4964), "Italie"),
        ((64.9631, -19.0208), "Islande"),
        ((64.1355, -21.8954), "Islande"),
        ((63.4389, -20.3093), "Islande"),
        ((-38.4161, -63.6167), "Argentine"),
        ((-34.6037, -58.3816), "Argentine"),
        ((-22.9035, -43.2096), "Argentine"),
        ((23.6345, -102.5528), "Mexique"),
        ((19.4326, -99.1332), "Mexique"),
        ((21.1619, -86.8515), "Mexique"),
        ((14.0583, 108.2772), "Vietnam"),
        ((21.0285, 105.8542), "Vietnam"),
        ((10.8231, 106.6297), "Vietnam"),
        ((18.1096, -77.2975), "Jamaïque"),
        ((17.9714, -76.7929), "Jamaïque"),
        ((18.3970, -77.0190), "Jamaïque"),
        ((4.5709, -74.2973), "Colombie"),
        ((4.7110, -74.0721), "Colombie"),
        ((6.2476, -75.5663), "Colombie")
    ]

    all_leads = []
    for (lat, lng), country in locations:
        print(f"Fetching data for coordinates: ({lat}, {lng}) in {country}")
        data = run_api(lat, lng)
        if data:
            leads = parse_json(data, country)
            all_leads.extend(leads)

    # Enregistrer dans un fichier CSV
    save_to_csv(all_leads)  # Correction ici

if __name__ == "__main__":
    main()


#Affichage
import pandas as pd
import random

# Charger les données
import os
@st.cache_data
def load_data(file_path):
    if not file_path.endswith('.csv'):
        raise ValueError("Le fichier fourni n'est pas au format .csv. Veuillez vérifier.")
    return pd.read_csv(file_path)

# Fonction pour recommander des restaurants
def recommend_restaurants(data, country, num_recommendations=3):
    country_data = data[data['Country'] == country]
    if len(country_data) < num_recommendations:
        return country_data
    return country_data.sample(num_recommendations)

# Charger les données des restaurants
data_file = "output_restaurants.csv"  # Chemin du fichier CSV généré
# data_file = "C:/Users/douni/Downloads/restaurants.csv" # Assurez-vous que ce fichier est à jour et correct
data = load_data(data_file)
# st.write(data)


# Interface utilisateur avec Streamlit
st.title("🍽️ Restaurant Recommendations")
st.write("Select the desired country to discover the best restaurants in that country! 🌟")

# Ajouter une barre déroulante pour sélectionner un pays
countries = data['Country'].unique()
selected_country = st.selectbox("Choose a country :", countries)

# Afficher les recommandations
if selected_country:
    st.write(f"Recommended restaurants for {selected_country}:")
    recommendations = recommend_restaurants(data, selected_country)
    for _, row in recommendations.iterrows():
        st.write(f"**Name**: {row['Nom']}")
        st.write(f"**Address**: {row['Address']}")
        st.write(f"**Rating**: {row['Note']}")
        st.write(f"**Number of reviews**: {row['User_Rating_Count']}")
        st.write(f"**Phone number**: {row['Telephone']}")
        st.write(f"**Price**: {row['Price_Level']}")
        st.write(f"**Vegetarian option**: {row['Serves_Vegetarian_Food']}")

        #st.write(f"**Prix**: {'$' * int(row['Price_Level']) if pd.notna(row['Price_Level']) else 'Non spécifié'}")
        st.write("---")
