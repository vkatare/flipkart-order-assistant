import os
import joblib
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd

# Dynamically resolve root directory relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_TABULAR_PATH = os.path.join(PROJECT_ROOT, "models", "return_risk_model.pkl")
MODEL_IMAGE_PATH = os.path.join(PROJECT_ROOT, "models", "product_classifier.pt")

T_STAR_RF = 0.5000

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

NUMERIC_FEATURES = [
    "price_inr", "discount_pct", "customer_tenure_days",
    "num_previous_orders", "num_previous_returns", "delivery_distance_km",
    "delivery_days", "is_weekend_order", "rating_given"
]
CATEGORICAL_FEATURES = ["product_category", "payment_method"]

# Lazy load model singletons
_rf_model=None
_image_model=None

def _get_tabular_model():
    """Lazy loader for saved Random Forest pipeline"""
    global _rf_model
    if _rf_model is None:
        if not os.path.exists(MODEL_TABULAR_PATH):
            raise FileNotFoundError(f"Model file {MODEL_TABULAR_PATH} not found")
        _rf_model = joblib.load(MODEL_TABULAR_PATH)
    return _rf_model

def _get_image_model():
    """Lazy loader for saved image classfier model"""
    global _image_model
    if _image_model is None:
        if not os.path.exists(MODEL_IMAGE_PATH):
            raise FileNotFoundError(f"Model file {MODEL_IMAGE_PATH} not found")

        # Reconstruct ResNet-18 feature extractor + linear head architecture
        weights =  models.ResNet18_Weights.DEFAULT
        resnet =  models.resnet18(weights=weights)
        for param in resnet.parameters():
            param.requires_grad = False

        backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        classifier_head = nn.Linear(512, 10)

        full_model = nn.Sequential(backbone, classifier_head)
        full_model.load_state_dict(torch.load(MODEL_IMAGE_PATH, map_location=torch.device("cpu")))
        full_model.eval()
        _image_model = full_model
    return _image_model

# Tabular return risk prediction
def  predict_return_risk(order_details: dict) -> dict:
    """
    Predicts the risk of an order being returned based on customer & order details.

    Parameters:
    order_details (dict): Dictionary containing tabular features:
        - price_inr (float)
        - discount_pct (float)
        - customer_tenure_days (int)
        - num_previous_orders (int)
        - num_previous_returns (int)
        - delivery_distance_km (float)
        - delivery_days (int)
        - is_weekend_order (int: 0 or 1)
        - rating_given (float)
        - product_category (str)
        - payment_method (str)

    Returns:
    dict: Risk probability, binary return decision at t*_rf, and risk tier bucket.
    """
    model = _get_tabular_model()

    #Convert dictionary input to Dataframe
    input_df = pd.DataFrame([order_details])

    #Extract estimated return probabililty for Class 1 (Returned)
    prob_return = float(model.predict_proba(input_df)[:, 1][0])

    #Binary prediction based on empirical threshold t*_rf = 0.5000
    will_return = bool(prob_return >= T_STAR_RF)

    #Risk Tier Bucketing
    if prob_return < 0.35:
        risk_bucket = "Low Risk"
        action_recommendation = "Standard Order Processing."
    elif prob_return < T_STAR_RF:
        risk_bucket = "Medium Risk"
        action_recommendation = "Send proactive order delivery confirmation SMS"
    else:
        risk_bucket = "High Risk"
        action_recommendation = "Flag for quality verification prior to dispatch"

    return {
        "return_probability":round(prob_return, 4),
        "t_star_threshold":T_STAR_RF,
        "predicted_return": will_return,
        "risk_bucket": risk_bucket,
        "action_recommendation":action_recommendation,
    }           

# Product Image Classification
def classify_product_image(image_path: str) -> dict:
    """
    Classifies a product image into one of 10 fashion categories using PyTorch model.

    Parameters:
    image_path (str): Path to local PNG/JPG image file.

    Returns:
    dict: Top predicted category, confidence score, and top-3 class probabilities.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path {image_path} does not exist")
    model = _get_image_model()

    # Define tranform(64x64, 3-channel, ImageNet normalization)
    transform =  transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    img= Image.open(image_path).convert("RGB")
    tensor_img = transform(img).unsqueeze(0) #Add batch dimension [1, 3, 64, 64]

    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.softmax(outputs, dim=1)[0].numpy()

    top_idx = int(np.argmax(probabilities))
    top_class = CLASS_NAMES[top_idx]
    confidence = float(probabilities[top_idx])

    #Top-3 predicions
    top3_indices = np.argsort(probabilities)[::-1][:3]
    top3_preds = [
        {
            "category": CLASS_NAMES[idx], "probability":round(float(probabilities[idx]), 4)
        }
        for idx in top3_indices
    ]

    return {
        "predicted_category": top_class,
        "confidence": round(confidence, 4),
        "top_3_predictions": top3_preds
    }

def assess_order_and_image(order_details: dict, image_path: str) -> dict:
    """
    Combines tabular risk scoring and visual product classification to assess
    order risk and detect potential product catalog mismatches.

    Parameters:
    order_details (dict): Tabular features for the order.
    image_path (str): Path to product image.

    Returns:
    dict: Consolidated assessment report with inconsistency flags.
    """
    risk_report= predict_return_risk(order_details)
    image_report= classify_product_image(image_path)

    declared_category = order_details.get("product_category", "").lower()
    predicted_category = image_report["predicted_category"].lower()

    # Mismatch check logic
    category_mismatch = (
        declared_category != "" and
        declared_category not in predicted_category and
        predicted_category not in declared_category
    )

    status = "ok"
    if category_mismatch:
        status = "FLAGGED_MISMATCH"
        risk_report["action_recommendation"]+="WARNING: Declared category does not match image classification"

    return {
        "status": status,
        "declared_category": order_details.get("product_category"),
        "image_predicted_category":image_report["predicted_category"],
        "category_mismatch_detected": category_mismatch,
        "return_risk_assessment": risk_report,
        "image_classification_assessment":image_report
    }

# Quick tool self-test execution

if __name__ == "__main__":
    print("==== Testing agent tools ====")

    #Sample order dict
    test_order={
        "price_inr": 2499.0,
        "discount_pct": 15.0,
        "customer_tenure_days": 120,
        "num_previous_orders": 5,
        "num_previous_returns": 2,
        "delivery_distance_km": 12.5,
        "delivery_days": 3,
        "is_weekend_order": 1,
        "rating_given": 4.0,
        "product_category": "Sneaker",
        "payment_method": "COD"
    }

    test_img_path = os.path.join(PROJECT_ROOT, "data", "sample_images", "03_sneaker.png")

    print("1. Testing tabular risk scoring")
    risk_out = predict_return_risk(test_order)
    print(risk_out)

    print("2. Testing image classfier tool:")
    img_out = classify_product_image(test_img_path)

    print("3. Testing combined assessment tool:")
    combined_out = assess_order_and_image(test_order,test_img_path)
    print(combined_out)

