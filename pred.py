# file_path: streamlit_app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt

# Streamlit configuration
st.set_page_config(page_title="Meal Order Prediction", layout="centered")

# Step 1: Load the dataset
@st.cache_data
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

# Step 2: Preprocess the dataset
def preprocess_data(df):
    # Drop columns not needed for training
    X = df.drop(['id', 'num_orders'], axis=1)
    y = df['num_orders']

    # Define categorical and numerical columns
    categorical_features = ['center_id', 'meal_id', 'city_code', 'region_code', 'center_type', 'category', 'cuisine']
    numerical_features = ['week', 'checkout_price', 'base_price', 'emailer_for_promotion', 'homepage_featured', 'op_area']

    # Apply transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create pipeline with preprocessing and regression model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    return model_pipeline, X, y

# Step 3: Train the model
@st.cache_data
def train_model(_model_pipeline, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    _model_pipeline.fit(X_train, y_train)
    return _model_pipeline

# Step 4: Predict number of orders
def predict_orders(model_pipeline, user_input):
    prediction = model_pipeline.predict(user_input)
    return int(prediction[0])

# Step 5: Visualization Function
def visualize_predictions(model_pipeline, user_input, df):
    # Extract historical data for the given center_id
    center_id = user_input['center_id'].iloc[0]
    historical_data = df[df['center_id'] == center_id].sort_values(by='week')

    # Plot historical data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(historical_data['week'], historical_data['num_orders'], marker='o', label='Historical Orders')

    # Predict orders for the next 10 weeks
    user_week = user_input['week'].iloc[0]
    weeks = np.arange(user_week, user_week + 10)
    predicted_orders = []

    for week in weeks:
        temp_input = user_input.copy()
        temp_input['week'] = week
        predicted_orders.append(predict_orders(model_pipeline, temp_input))

    # Plot predicted data
    ax.plot(weeks, predicted_orders, marker='o', linestyle='--', color='red', label='Predicted Orders')
    ax.set_xlabel('Week')
    ax.set_ylabel('Number of Orders')
    ax.set_title(f'Number of Orders for Center ID: {center_id} (Historical and Predicted)')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


#streamlit
def main():
    st.title("Warehouse Optimization")

    # File uploader for dataset
    uploaded_file = st.file_uploader("Choose a CSV file:", type="csv")

    if uploaded_file is not None:
        # Load and preprocess the dataset
        df = load_dataset(uploaded_file)
        
        # Drop 'lead_time' and 'current_stock' for model training
        df = df.drop(columns=['lead_time', 'current_stock'], errors='ignore')
        model_pipeline, X, y = preprocess_data(df)

        # Train the model
        trained_model = train_model(model_pipeline, X, y)
        st.success("Model trained successfully!")

        # User Input
        st.header("Enter Details to Predict Orders")
        week = st.number_input("Week (1-52):", min_value=1, max_value=52, value=1)
        center_id = st.text_input("Center ID:")
        meal_id = st.text_input("Meal ID:")
        checkout_price = st.number_input("Checkout Price:", min_value=0.0, value=0.0)
        base_price = st.number_input("Base Price:", min_value=0.0, value=0.0)
        emailer_for_promotion = st.selectbox("Emailer for Promotion:", [0, 1])
        homepage_featured = st.selectbox("Homepage Featured:", [0, 1])
        city_code = st.text_input("City Code:")
        region_code = st.text_input("Region Code:")
        center_type = st.text_input("Center Type:")
        op_area = st.number_input("Operational Area:", min_value=0.0, value=0.0)
        category = st.text_input("Meal Category:")
        cuisine = st.text_input("Cuisine:")
        lead_time = st.number_input("Lead Time (days):", min_value=1, value=1)
        current_stock = st.number_input("Current Stock:", min_value=0, value=0)

        # Create DataFrame for User Input
        user_input = pd.DataFrame({
            'week': [week],
            'center_id': [center_id],
            'meal_id': [meal_id],
            'checkout_price': [checkout_price],
            'base_price': [base_price],
            'emailer_for_promotion': [emailer_for_promotion],
            'homepage_featured': [homepage_featured],
            'city_code': [city_code],
            'region_code': [region_code],
            'center_type': [center_type],
            'op_area': [op_area],
            'category': [category],
            'cuisine': [cuisine]
        })

        if st.button("Predict Orders"):
            # Predict the number of orders based on user input
            predicted_orders = predict_orders(trained_model, user_input)
            st.write(f"Predicted Number of Orders: {predicted_orders}")

            # Perform calculations directly from user input
            average_daily_demand = predicted_orders / lead_time
            std_dev_demand = np.std([predicted_orders])  # Standard deviation of a single prediction is 0, so use a dummy value if needed
            service_level = 1.65  # 95% service level (Z-score for normal distribution)
            safety_stock = service_level * std_dev_demand * np.sqrt(lead_time)
            reorder_point = (average_daily_demand * lead_time) + safety_stock
            ordering_cost = 50  # Assume a constant ordering cost
            holding_cost_per_unit = 2  # Assume a constant holding cost per unit per year
            annual_demand = predicted_orders * 365 / lead_time
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
            stockout = 1 if current_stock < reorder_point else 0

            # Display calculations
            st.subheader("Calculated Values")
            st.write(f"Average Daily Demand: {average_daily_demand:.2f}")
            st.write(f"Safety Stock: {safety_stock:.2f}")
            st.write(f"Reorder Point: {reorder_point:.2f}")
            st.write(f"EOQ: {eoq:.2f}")
            st.write(f"Stockout Status: {'Stockout' if stockout == 1 else 'No Stockout'}")

            # Visualize the historical and predicted orders
            visualize_predictions(trained_model, user_input, df)

if __name__ == "__main__":
    main()
