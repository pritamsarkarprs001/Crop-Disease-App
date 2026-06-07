import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2


st.set_page_config(page_title="Agri-Scan AI", page_icon="🌿", layout="centered")

st.title("🌿 Agri-Scan AI: Crop Disease Predictor")
st.write("Upload a picture of a plant leaf, and the AI will predict if it is healthy or diseased.")





CLASSES = [
    "Cashew anthracnose", "Cashew gumosis", "Cashew healthy", "Cashew leaf miner", "Cashew red rust",
    "Cassava bacterial blight", "Cassava brown spot", "Cassava green mite", "Cassava healthy", "Cassava mosaic",
    "Maize fall armyworm", "Maize grasshoper", "Maize healthy", "Maize leaf beetle", "Maize leaf blight", "Maize leaf spot", "Maize streak virus",
    "Tomato healthy", "Tomato leaf blight", "Tomato leaf curl", "Tomato septoria leaf spot", "Tomato verticulium wilt"
]




@st.cache_resource
def load_model():
    
    model_path  = "crop_protection_efficientnet_b0.onnx"
    session     = ort.InferenceSession(model_path)
    return session



session  = load_model()



def process_image(image):
    
    img  = np.array(image)
    img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    
    clahe       = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    ycrcb       = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    channels    = list(cv2.split(ycrcb))
    channels[0] = clahe.apply(channels[0])
    ycrcb       = cv2.merge(channels)
    img         = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    


    img = cv2.resize(img, (256, 256))

    

    img  = img  / 255.0  
    mean = np.array([0.485, 0.456,0.406])
    std  = np.array([0.229,0.224, 0.225])
    img  = (img - mean) /std

    img = np.transpose(img, (2, 0, 1))
    
 


    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img



uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    
    

    with st.spinner('Analyzing plant biology...'):
        
        
        input_tensor = process_image(image)
        input_name   = session.get_inputs()[0].name
        outputs      = session.run(None, {input_name: input_tensor})
        
        
        logits        = outputs[0][0]
        exp_logits    = np.exp(logits  - np.max(logits)) 
        probabilities = exp_logits  /exp_logits.sum()
        
       
        predicted_idx   = np.argmax(probabilities)
        predicted_class = CLASSES[predicted_idx]
        confidence      = probabilities[predicted_idx] * 100

   


    st.success("Analysis Complete!")
    
   
    st.markdown(f"### Diagnosis: **{predicted_class}**")
    
    st.write(f"**Confidence Score:** {confidence:.2f}%")
    st.progress(int(confidence))
 
    if confidence < 60:
        st.warning("Confidence is somewhat low. Ensure the image is clear, well-lit, and focused on a single leaf.")