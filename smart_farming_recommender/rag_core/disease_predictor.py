import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Define the model architecture.
# I am assuming a ResNet-18 architecture with a modified final layer.
# The user will likely need to change this to match their model.
def create_model(num_classes):
    model = torchvision.models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Define the class names.
# This is a placeholder. The user MUST provide the correct class names.
CLASS_NAMES = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Load the model
def load_model(model_path):
    num_classes = len(CLASS_NAMES)
    model = create_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Predict the disease
def predict_disease(image_path):
    """
    Predicts the disease from an image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The predicted disease name.
    """
    try:
        model_path = '/Users/ishantsingh/CPS_PROJECT/plant_disease_model_1_latest.pt'
        model = load_model(model_path)
        image_tensor = preprocess_image(image_path)
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            disease_name = CLASS_NAMES[predicted.item()]
        return disease_name
    except Exception as e:
        print(f"Error predicting disease: {e}")
        return "Could not predict the disease."

if __name__ == '__main__':
    # Example usage:
    # Create a dummy image for testing
    dummy_image = Image.new('RGB', (224, 224), color = 'red')
    dummy_image_path = 'dummy_image.png'
    dummy_image.save(dummy_image_path)
    
    # Since we don't have the actual model, we can't run the prediction.
    # The following line is for demonstration purposes.
    # predicted_disease = predict_disease(dummy_image_path)
    # print(f"Predicted disease: {predicted_disease}")
    
    print("Disease predictor script created.")
    print("NOTE: The model architecture and class names are placeholders.")
    print("The user needs to provide the correct information for the model to work.")
