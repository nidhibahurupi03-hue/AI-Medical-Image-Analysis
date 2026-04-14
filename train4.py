from src4.preprocessing4 import load_data
from src4.model4 import build_model

def train_model():
    train_data = load_data("data4/train")
    test_data = load_data("data4/test")

    model = build_model()

    model.fit(
        train_data,
        validation_data=test_data,
        epochs=5
    )

    model.save("models4/model.h5")

    print("✅ Model Trained Successfully")