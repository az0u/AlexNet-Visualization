import streamlit as st
import torch
from torchvision.models import alexnet, AlexNet_Weights
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from gradcam import GradCAM_, overlay_attribution, visualize_attribution
from hooks import get_activation, attach_hooks, activations, visualize_layers

# Device the model and the data is placed on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("🧩 AlexNet Visualization") # page title 
st.write('Upload an image and see how it the AlexNet model "sees" it!') 

# Upload image module 
up_image = st.file_uploader(
    "Upload an image", 
    type=("jpg", "jpeg", "png"))  

# Load pre-trained AlexNet
model = alexnet(weights=AlexNet_Weights.DEFAULT)
model.eval() # evalutation mode 
model.to(device)  # ensure model on same device as input

# Mapping layer names to more simple format
layer_names = {
    "features.0": "Convolutional 1",
    "features.1": "ReLU 1",
    "features.2": "MaxPool 1",
    "features.3": "Convolutional 2",
    "features.4": "ReLU 2",
    "features.5": "MaxPool 2",
    "features.6": "Convolutional 3",
    "features.7": "ReLU 3",
    "features.8": "Convolutional 4",
    "features.9": "ReLU 4", 
    "features.10": "Convolutional 5",
    "features.11": "ReLU 5",
    "features.12": "MaxPool 3",
    "avgpool": "AvgPool",
}

# process image 
def process_image(image):

    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")

    preprocess = AlexNet_Weights.DEFAULT.transforms()
    # add batch dim and move to device
    input_tensor = preprocess(image).unsqueeze(0).to(device)  
    return input_tensor


# Load class names into a list
with open("class_names_ImageNet.txt", "r") as file:
    class_names = [line.strip() for line in file.readlines()]


def predict(model, input): 
    
    # output 
    output = model(input)
    # confidence score 
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    pred_idx = torch.argmax(probabilities)   # index of highest probability
    confidence = probabilities[pred_idx].item() * 100  # convert to %
    # get class name from index
    predicted_class = class_names[pred_idx]
    return predicted_class, confidence


# Load image from uploader or output error
if up_image is not None:
    img = Image.open(up_image).convert("RGB")
    # show input image 
    st.image(img, caption="Input image", width='stretch')
    image = st.selectbox(
        "Layer activations visualization",
        ("No","Yes"))
    gradcam = st.selectbox(
        "Prediction heatmap",
        ("No","Yes"))
    
    image_input = process_image(img)
    
    if (image=="Yes"): 
        with st.spinner("⏳ Hold tight, your image is being processed..."):
            st.success("Using the uploaded image.")

            st.header("👀 AlexNet Layer Visualization") 
            st.write("See how your image activates different layers inside AlexNet!") 

            # Dropdown for layer selection by user 
            layer_type = st.selectbox(
            "Choose a layer type:", 
            ("All","Convolutional", "ReLU", "MaxPool")) 

            visualize_layers(model, layer_type, layer_names, image_input)
            predicted_class, confidence = predict(model, image_input)

            st.write(f"💭 Predicted Class: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}%")
            
        
    if (gradcam=="Yes"):
        with st.spinner("⏳ Hold tight, your image is being processed..."):
        # Load model
            model = alexnet(weights=AlexNet_Weights.DEFAULT)

            # Calculate prediction
            with torch.no_grad():
                prediction = model(image_input).argmax(dim=-1).item()

            # Use explainability method to get attribution
            attribution = GradCAM_(
                model=model,
                inputs=image_input,
                target=prediction
            )

            attribution_overlaid = overlay_attribution(attribution, image_input)

            # Visualize results
            visualize_attribution(
                attribution=attribution,
                original_image=image_input,
                image_std=[0.229, 0.224, 0.225],
                image_mean=[0.485, 0.456, 0.406],
            )

            st.header("🗺️ GradCAM Heatmap")
            st.write('Visualize the parts of the image that the model found "most important" for its prediction!')

            # Display images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(Image.open(up_image), width='stretch')
            with col2:
                st.subheader("Attribution Overlay")
                st.image("./attribution_overlay.png", width='stretch')

            predicted_class, confidence = predict(model, image_input)

            st.write(f"💭 Predicted Class: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}%")
            
else:
    st.info("Please upload an Image to proceed.")