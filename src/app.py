import os
import streamlit as st
from src.agent_tools import (
    assess_order_and_image,
    classify_product_image,
    predict_return_risk,
)

st.set_page_config(
    page_title="Flipkart Support & Return Risk Assessment", layout="wide"
)

st.title("Flipkart Order Return Risk & Catalog Assistant")

tab1, tab2 = st.tabs(["Order Risk Assesment", "Support & Policy Query"])

with tab1:
    st.header("Assess Order Return Risk")
    col1, col2 = st.columns(2)

    with col1:
        category= st.selectbox(
            "Declared Product Category",
            ["Apparel", "Electronics", "Home", "Footwear", "Beauty"],
        )
        payment = st.selectbox(
            "Payment Method", ["COD", "Prepaid_Card", "Prepaid_UPI", "Wallet"]
        )
        price= st.number_input("Price(INR)", min_value=100.0, value=2499.0)
        discount=st.slider("Discount (%)", 0.0, 75.0, 15.0)
        tenure=st.number_input("Customer Tenure (Days)", min_value=1, value=120)

    with col2:
        prev_orders = st.number_input(
            "Previous Orders", min_value=0, value=5
        )
        prev_returns = st.number_input(
            "Previous Returns", min_value=0, value=2
        )
        del_distance = st.number_input(
            "Delivery Distance (KM)", min_value=1.0, value=12.5
        )
        del_days=st.number_input(
            "Deliver Days", min_value=1, value=3
        )
        weekend=st.selectbox("Is Weekend Offer?", [0,1], index=1)
        rating=st.slider("Rating Given", 1.0, 5.0, 4.0)

        if st.button("Run Risk Assessment"):
            order_features = {
                "price_inr": price,
                "discount_pct": discount,
                "customer_tenure_days": tenure,
                "num_previous_orders": prev_orders,
                "num_previous_returns": prev_returns,
                "delivery_distance_km": del_distance,
                "delivery_days": del_days,
                "is_weekend_order": weekend,
                "rating_given": rating,
                "product_category": category,
                "payment_method": payment,
            }

            result= predict_return_risk(order_features)
            st.subheader("Results")
            st.metric(
                label="Return Probability",
                value=f"{result['return_probability']*100:.2f}%",
            )
            st.write(f"**Risk Bucket:**{result['risk_bucket']}")
            st.write(f"**Recommendation:** {result['action_recommendation']}")

with tab2:
    st.header("Ask Policy Question")
    query = st.text_input("Enter your support query:")
    if st.button("Submit Query"):
        if "return window" in query.lower():
            st.success(
                "Apparel and footwear items are eligible for return within a 14-day window."
            )
        elif "cod" in query.lower():
            st.success(
                "COD refunds are processed via bank transfer or Wallet within 3-5 business days."
            )
        else:
            st.warning("No direct policy found for this query.")

