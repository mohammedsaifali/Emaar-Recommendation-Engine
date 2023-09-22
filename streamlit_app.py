import streamlit as st
import numpy as np
import pandas as pd
from lightfm import LightFM
import joblib

# Load pre-trained model
model = joblib.load("lightfm_model.pkl")

# Load data for mappings
customer_details = pd.read_excel("CustomerDetails.xlsx")
sales_register = pd.read_excel("SalesRegister.xlsx")

# Create a mapping from CustomerCode to Customer Name
customer_map = dict(zip(customer_details['Code'], customer_details['Name']))

# Create a mapping from ItemCode to ItemName
item_map = dict(zip(sales_register['ItemCode'], sales_register['ItemName']))

# Prepare data for LightFM to get the mapping
dataset = Dataset()
all_users = pd.concat([customer_details['Code'], sales_register['CustomerCode']]).unique()
dataset.fit(users=all_users, items=sales_register['ItemCode'].unique())

# Obtain the internal mapping of user and item IDs from the dataset
user_id_map, _, item_id_map, _ = dataset.mapping()

# Reverse the item_id_map dictionary to map from internal LightFM IDs to original item codes
reverse_item_map = {v: k for k, v in item_id_map.items()}

def main():
    st.title("Item Recommender")
    customer_name = st.selectbox("Select Customer Name:", sorted(list(customer_map.values())))
    customer_code_original = [code for code, name in customer_map.items() if name == customer_name][0]

    # Convert the customer_code to its internal LightFM ID
    customer_code_internal = user_id_map[customer_code_original]

    # Generate recommendations
    n_items = len(item_id_map)
    scores = model.predict(customer_code_internal, np.arange(n_items))
    
    # Fetch the original item codes using the reverse_item_map
    top_items = [reverse_item_map[i] for i in np.argsort(-scores)[:5]]

    # Map the top items to their names
    recommended_items = [item_map[item] for item in top_items]

    st.write("Recommended Items:")
    for item in recommended_items:
        st.write(item)

if __name__ == "__main__":
    main()
