import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("final_model.pkl")

# Sidebar for page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Aperçu Général", "Tester mon modele de Scoring"])

if page == "Aperçu Général":
    st.title("Scoring de crédit bancaire")

    st.write("""
    ---

    Je suis Presley Koyaweda, ravi de partager avec vous mon projet de scoring de crédit bancaire.

    ### 🎯 **Objectif**
    Ce projet visait à développer un modèle capable de prédire la probabilité de défaut de paiement des emprunteurs auprès des banques.
         
    ### 🚀 **Étapes du Projet**
    1. Exploration des Données.
    2. Discrétisation des Variables.
    3. Analyse des Correspondances Multiples (ACM)
    4. Calcul du V de Cramer.
    5. Modélisation
    6. Mise en production
         
    ### 📊 **Résultats**
    Les résultats de notre analyse montrent que l'Analyse Discriminante Linéaire (LDA) avec des caractéristiques polynomiales offre des performances solides parmi les modèles testés. Voici les mesures de qualité obtenues :

    - Précision (Precision) : 86.25%
    - Rappel (Recall) : 95.83%
    - Score F1 (F1 Score) : 86.25%
    - Exactitude (Accuracy) : 76.33%

    Ces résultats indiquent que l'Analyse Discriminante Linéaire est efficace pour prédire la solvabilité des emprunteurs dans ce projet.

    ### 🌟 **Conclusion**
    Ce projet permet de prédire efficacement la solvabilité des emprunteurs, offrant ainsi une aide précieuse pour les décisions de crédit des institutions financières. Nous avons particulièrement cherché à augmenter le rappel (recall) afin de minimiser les chances de refuser un crédit à ceux qui le méritent vraiment.
    Car Il est pire de classer un client comme bon alors qu'il est mauvais que de classer un client comme mauvais alors qu'il est bon.
    """)

    # Display images in a 2x2 grid
    col1, col2 = st.columns(2)

    with col1:
        st.image("learning_curve.png", caption="Learning Curve")
        st.image("plan_factoriel.png", caption="Plan Factoriel des Individus")
    
    with col2:
        st.image("v_de_cramer.png", caption="Pouvoir Discriminant des Variables (V de Cramer)")
        st.image("precision_recall_curve.png", caption="Precision and Recall Curve")

elif page == "Tester mon modele de Scoring":
    st.title("Scoring de crédit bancaire")

    st.write("""
    Testez ce modèle en remplissant le formulaire ci-dessous pour obtenir une évaluation de la solvabilité.

    ###### Veuillez renseigner tous les champs
    """)

    # Instructions for categorization in the sidebar
    st.header('Instructions de Catégorisation des Variables')
    st.write("""

Pour faciliter l'analyse des données, nous avons catégorisé les variables en fonction de leurs valeurs.

#### Account_Balance (Statut du compte courant existant)
- **Catégorie 1**: ... < 0 DM
- **Catégorie 2**: 0 <= ... < 200 DM
- **Catégorie 3**: ... >= 200 DM / salaire assigné pour au moins 1 an
- **Catégorie 4**: pas de compte courant

#### Payment_Status_of_Previous_Credit (Statut du paiement du crédit précédent)
- **Catégorie 1**: aucun crédit / tous les crédits remboursés à temps
- **Catégorie 2**: tous les crédits dans cette banque remboursés à temps
- **Catégorie 3**: crédits existants remboursés à temps jusqu'à maintenant
- **Catégorie 4**: retard dans le remboursement par le passé
- **Catégorie 5**: compte critique / autres crédits existants (pas dans cette banque)

#### Duration_Category (Catégorie de la durée de crédit)
- **Catégorie 1**: Durée < 12 mois
- **Catégorie 2**: 12 mois <= Durée < 18 mois
- **Catégorie 3**: 18 mois <= Durée < 24 mois
- **Catégorie 4**: Durée >= 24 mois

#### Value_Savings_Stocks (Compte d'épargne / obligations)
- **Catégorie 1**: ... < 100 DM
- **Catégorie 2**: 100 <= ... < 500 DM
- **Catégorie 3**: 500 <= ... < 1000 DM
- **Catégorie 4**: ... >= 1000 DM
- **Catégorie 5**: inconnu / pas de compte d'épargne

#### Purpose (But du crédit)
- **Catégorie 1**: voiture (neuve)
- **Catégorie 2**: voiture (d'occasion)
- **Catégorie 3**: meubles / équipements
- **Catégorie 4**: radio / télévision
- **Catégorie 5**: appareils domestiques
- **Catégorie 6**: réparations
- **Catégorie 7**: éducation
- **Catégorie 8**: (vacances - n'existe pas?)
- **Catégorie 9**: reconversion
- **Catégorie 10**: business
- **Catégorie 11**: autres

#### Mount_Category (Catégorie du montant de crédit)
- **Catégorie 1**: Montant < 1365 DM
- **Catégorie 2**: 1365 DM <= Montant < 2319 DM
- **Catégorie 3**: 2319 DM <= Montant < 3972 DM
- **Catégorie 4**: Montant >= 3972 DM

#### Most_valuable_available_asset (Propriété)
- **Catégorie 1**: immobilier
- **Catégorie 2**: si pas A121 : contrat d'épargne-logement / assurance vie
- **Catégorie 3**: si pas A121 / A122 : voiture ou autre, pas dans l'attribut 6
- **Catégorie 4**: inconnu / pas de propriété

#### Type_of_apartment (Logement)
- **Catégorie 1**: location
- **Catégorie 2**: propre
- **Catégorie 3**: gratuitement

#### Length_of_current_employment (Emploi actuel depuis)
- **Catégorie 1**: sans emploi
- **Catégorie 2**: ... < 1 an
- **Catégorie 3**: 1 <= ... < 4 ans
- **Catégorie 4**: 4 <= ... < 7 ans
- **Catégorie 5**: ... >= 7 ans

#### Age_category (Catégorie d'âge)
- **Catégorie 1**: Âge < 27 ans
- **Catégorie 2**: 27 ans <= Âge < 33 ans
- **Catégorie 3**: 33 ans <= Âge < 37 ans
- **Catégorie 4**: Âge >= 37 ans

#### Concurrent_Credits (Crédits concurrents)
- **Catégorie 1**: aucun
- **Catégorie 2**: co-demandeur
- **Catégorie 3**: garant

#### Sex_Marital_Status (Statut personnel et sexe)
- **Catégorie 1**: homme : divorcé / séparé
- **Catégorie 2**: femme : divorcée / séparée / mariée
- **Catégorie 3**: homme : célibataire
- **Catégorie 4**: homme : marié / veuf
- **Catégorie 5**: femme : célibataire                 


    """)

    # Create a form for user input
    st.sidebar.header('Input Parameters')

    def user_input_features():
        account_balance = st.sidebar.number_input('Account Balance', min_value=1, max_value=4, value=1)
        payment_status = st.sidebar.number_input('Payment Status of Previous Credit', min_value=1, max_value=5, value=1)
        duration_category = st.sidebar.number_input('Duration Category', min_value=1, max_value=4, value=1)
        value_savings_stocks = st.sidebar.number_input('Value Savings Stocks', min_value=1, max_value=5, value=1)
        purpose = st.sidebar.number_input('Purpose', min_value=1, max_value=11, value=1)
        mount_category = st.sidebar.number_input('Mount Category', min_value=1, max_value=4, value=1)
        most_valuable_available_asset = st.sidebar.number_input('Most Valuable Available Asset', min_value=1, max_value=4, value=1)
        type_of_apartment = st.sidebar.number_input('Type of Apartment', min_value=1, max_value=3, value=1)
        length_of_current_employment = st.sidebar.number_input('Length of Current Employment', min_value=1, max_value=5, value=1)
        age_category = st.sidebar.number_input('Age Category', min_value=1, max_value=4, value=1)
        concurrent_credits = st.sidebar.number_input('Concurrent Credits', min_value=1, max_value=3, value=1)
        sex_marital_status = st.sidebar.number_input('Sex Marital Status', min_value=1, max_value=5, value=1)

        data = {
            'Account_Balance': account_balance,
            'Payment_Status_of_Previous_Credit': payment_status,
            'Duration_Category': duration_category,
            'Value_Savings_Stocks': value_savings_stocks,
            'Purpose': purpose,
            'Mount_Category': mount_category,
            'Most_valuable_available_asset': most_valuable_available_asset,
            'Type_of_apartment': type_of_apartment,
            'Length_of_current_employment': length_of_current_employment,
            'Age_category': age_category,
            'Concurrent_Credits': concurrent_credits,
            'Sex_Marital_Status': sex_marital_status
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Display user input
    st.subheader('User Input parameters')
    st.write(input_df)

    # Prediction
    if st.button('Prédire'):
        prediction_proba = model.predict_proba(input_df)
        prediction = (prediction_proba[:, 1] > 0.5).astype(int)  # Assuming 0.5 as the threshold

        # Convert prediction to readable result
        if prediction[0] == 0:
            result = "Non Solvable"
        else:
            result = "Solvable"

        st.subheader('Prediction')
        st.write(result)
