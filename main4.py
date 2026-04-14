from src4.train4 import train_model
from src4.predict4 import predict_image

print("==== AI Medical Image Analysis ====")
print("1. Train Model")
print("2. Predict Image")

choice = input("Enter choice: ")

if choice == "1":
    train_model()

elif choice == "2":
    path = input("Enter image path: ")
    predict_image(path)

else:
    print("Invalid Choice")