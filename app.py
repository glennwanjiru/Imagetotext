import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Define the Streamlit app
def main():
    st.title("BLIP Image Captioning")
    st.write("Upload an image and get a caption!")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Conditional captioning
        text = "a photography of"
        inputs = processor(images=image, text=text, return_tensors="pt")
        out = model.generate(**inputs)
        conditional_caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Unconditional captioning
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        unconditional_caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Display captions
        st.write(f"Conditional Caption: {conditional_caption}")
        st.write(f"Unconditional Caption: {unconditional_caption}")

if __name__ == "__main__":
    main()
