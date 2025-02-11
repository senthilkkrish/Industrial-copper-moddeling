# -----------------------------------------------------------------------------------------------------#
# Run the below in the terminal
# -----------------------------------------------------------------------------------------------------#
# pip install google-api-python-client
# pip install isodate
# -----------------------------------------------------------------------------------------------------#

# -----------------------------------------------------------------------------------------------------#
# Importing required packages
# -----------------------------------------------------------------------------------------------------#
import streamlit as st

# from pymongo import MongoClient
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------------------------------------------------------------------------#
## Code for Streamlit page
# -----------------------------------------------------------------------------------------------------#
st.set_page_config(layout="wide")
header = st.header(
    "Industrial Copper Modeling",    divider="blue",    anchor="Industrial Copper Modeling",)
# -----------------------------------------------------------------------------------------------------#
## Code to read data from excel to dtaframe
# -----------------------------------------------------------------------------------------------------#
csv_name = "Copper_Set_csv.csv"
Master_data = pd.read_csv("Copper_Set_csv.csv", dtype={"id": "string", "material_ref": "string"}, low_memory=False)
Master_df = pd.DataFrame(Master_data)
df = Master_df

# st.write('selling_price before removing outliers')
# st.write(Master_df['selling_price'].skew())

# -----------------------------------------------------------------------------------------------------#
## Code to remove Outliers
# -----------------------------------------------------------------------------------------------------#

# st.write('Removing Outliers from selling_price...')

# Using Interquartile Range (IQR)
Q1 = df["selling_price"].quantile(0.25)
Q3 = df["selling_price"].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df = df[(df["selling_price"] >= lower_bound) & (df["selling_price"] <= upper_bound)]

# st.write('selling_price after removing outliers')
# st.write(df['selling_price'].skew())

df_cleaned = df.dropna()

# Boxplot
plt.figure(figsize=(15, 5))
sns.boxplot(x=df_cleaned["selling_price"])
plt.title("Boxplot of Selling Price")
# plt.show()
#st.pyplot(plt)
# Convert 'quantity tons' to numeric
df_cleaned.loc[:, "quantity tons"] = pd.to_numeric(df_cleaned["quantity tons"], errors="coerce")

# -----------------------------------------------------------------------------------------------------#
## Assign Features and Target
# -----------------------------------------------------------------------------------------------------#

X = df_cleaned[["quantity tons", "thickness", "width", "country"]]  # feature columns
y = df_cleaned["selling_price"]  # Target column

# -----------------------------------------------------------------------------------------------------#
# #Train-Test Split: Split the dataset into training and testing sets.
# -----------------------------------------------------------------------------------------------------#

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X.shape, y.shape
# X_train.shape, y_train.shape
# X_test.shape, y_test.shape
def ML_Predict():
    new_input = pd.DataFrame(
        [[54.151139, 2.00, 1500.0, 28.0]], columns=X_train.columns
    )  # Replace with actual feature values
    prediction = st.empty()
    prediction = st.session_state["model"].predict(new_input)
    st.write(f"Prediction for the input {new_input}: {prediction}")

# ----------------------------------------------------------------------------------------------------#
# Calling  ML  Model input (single instance with 2 features)
# ----------------------------------------------------------------------------------------------------#
def ML_Model(Model, Modeltype):
    if Model == "Regression":
        # -----------------------------------------------------------------------------------------------------#
        # Train the Linear Regression model
        # -----------------------------------------------------------------------------------------------------#
        if Modeltype == "Linear Regression":
            lr = LinearRegression()
            lr.fit(X_train, y_train)

            # Make predictions
            y_pred_lr = lr.predict(X_test)

            col1, col2 = st.columns([1,2])
            mse = mean_squared_error(y_test, y_pred_lr)
            r2 = r2_score(y_test, y_pred_lr)
            rmse = np.sqrt(mse)
            with col1:
                # Evaluate
                st.markdown("## Linear Regression:")
                #st.write("Linear Regression:")
                st.markdown(f"### Mean Squared Error: `{mse:.4f}`")
                st.markdown(f"### R2 Score: `{r2:.4f}`")
                st.markdown(f"### Root of Mean Squared Error: `{rmse:.4f}`")

            with col2:
                plt.figure(figsize=(18, 6))
                plt.scatter(y_test, y_pred_lr)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title("Actual vs. Predicted")
                plt.show()
                st.pyplot(plt)
            st.session_state["model"] = lr
        else:
            # ----------------------------------------------------------------------------------------------------#
            # Train the Random Forest model
            # ----------------------------------------------------------------------------------------------------#
            rf = RandomForestRegressor(random_state=42)
            rf.fit(X_train, y_train)

            # Make predictions
            y_pred_rf = rf.predict(X_test)
            mse = mean_squared_error(y_test, y_pred_rf)
            r2 = r2_score(y_test, y_pred_rf)
            rmse = np.sqrt(mse)
            col1, col2 = st.columns([1,2])
            with col1:
                # Evaluate
                st.markdown("## Random Forest Regressor:")
                #st.write("Random Forest Regressor:")
                st.markdown(f"### Mean Squared Error: `{mse:.4f}`")
                st.markdown(f"### R2 Score: `{r2:.4f}`")
                st.markdown(f"### Root of Mean Squared Error: `{rmse:.4f}`")
            with col2:
                plt.figure(figsize=(18, 6))
                plt.scatter(y_test, y_pred_rf)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title("Actual vs. Predicted")
                plt.show()
                st.pyplot(plt)
            st.session_state["model"] = rf
            # return rf
    else:
        # -----------------------------------------------------------------------------------------------------#
        # Train the Classification model
        # -----------------------------------------------------------------------------------------------------#
        st.write("Classification")
    # def predict(new_input):
    #     # Predict
    #     prediction = lr.predict(new_input)
    #     # Predict
    #     prediction2 = rf.predict(new_input)
    #     # Print the prediction
    #     st.write(f"Prediction for the input {new_input}: {prediction2}")
    #     # Print the prediction
    #     st.write(f"Prediction for the input {new_input}: {prediction}")


# trained_model = ML_Model("Regression", "Liner Regression")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ip_quantity_tons = st.text_input(
            "Input quantity in tons : ",
        )
    with col2:
        ip_thickness = st.text_input(
            "Input thickness : ",
        )
    with col3:
        ip_width = st.text_input(
            "Input width : ",
        )
    with col4:
        ip_country = st.selectbox(
            "Select country code : ",
            ("HONKONG", "PORTUGAL", "SPAIN", "TURKEY", "USA"),
        )
    if st.button("Predict"):
        ML_Predict()
    #st.button("Predict", on_click=ML_Predict)

#         new_input = pd.DataFrame([[54.151139, 2.00, 1500.0, 28.0]], columns=X_train.columns)  # Replace with actual feature values
#         prediction = trained_model.predict(new_input)

#     #if st.button("Predict"):
#         #predict(new_input)
#         # Predict
# # prediction = lr.predict(new_input)
# #     # Predict
# # prediction2 = rf.predict(new_input)
#     # Print the prediction
# #st.write(f"Prediction for the input {new_input}: {prediction2}")
#     # Print the prediction
# st.write(f"Prediction for the input {new_input}: {prediction}")

# with st.spinner("Wait for it..."):
#     time.sleep(1)
# st.success("Done!", icon="✅")
# -----------------------------------------------------------------------------------------------------#
## Code for st.sidebar in Streamlit page
# -----------------------------------------------------------------------------------------------------#

# with st.sidebar:
#     st.sidebar.image(
#         "C:/Users/Viney Acsa Sam/OneDrive/Desktop/Visual Studio/Industrial Copper Modeling/Industrial Copper.jpg",
#         use_container_width=False,
#     )
col1, col2 = st.columns(2)
with col1:
    # Display DataFrame in Streamlit
    Model = st.selectbox(
            "Select Model : ",
            ("Regression", "Classification"),
        )
with col2:
    if Model == "Regression":
        Modeltype = st.selectbox(
                "Select Modeltype : ",
                ("Linear Regression", "Random Forest"),
            )
    else:
        Modeltype = st.selectbox(
                "Select Modeltype : ",
                ("Logistic Regression", "Neural Networks"),
            )

#st.button("Build", on_click=ML_Model, args=[Model, Modeltype])
st.write("Click here to")
if st.button("Build"):
    ML_Model(Model, Modeltype)

# -----------------------------------------------------------------------------------------------------#
# End of Code - Industrial_Copper_Modeling.py
# -----------------------------------------------------------------------------------------------------#
