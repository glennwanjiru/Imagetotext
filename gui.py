import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import pyttsx3
from ttkthemes import ThemedTk
import threading
import time

# Initialize TTS engine
tts_engine = pyttsx3.init()

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def generate_caption(image_path, caption_type='conditional'):
    """Generate caption for the given image using BLIP model."""
    image = Image.open(image_path).convert('RGB')
    text = "At Camera One,there is " if caption_type == 'conditional' else ""
    inputs = processor(image, text, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def read_caption(caption):
    """Convert text to speech."""
    tts_engine.say(caption)
    tts_engine.runAndWait()

def update_progress(percentage):
    """Update the progress bar."""
    progress_bar['value'] = percentage
    root.update_idletasks()

def process_image(file_path, caption_type='conditional'):
    """Process the image and update the UI."""
    try:
        # Update progress bar to 50%
        update_progress(50)
        caption = generate_caption(file_path, caption_type)
        # Update progress bar to 100%
        update_progress(100)
        caption_label.config(text=caption)
        read_caption(caption)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
    finally:
        # Reset progress bar
        update_progress(0)

def open_file():
    """Open file dialog and process selected image."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    # Display selected image
    image = Image.open(file_path)
    image.thumbnail((300, 300))
    img = ImageTk.PhotoImage(image)
    panel.config(image=img)
    panel.image = img

    # Start processing in a new thread
    threading.Thread(target=process_image, args=(file_path,)).start()

def capture_photo():
    """Capture photo from webcam and process."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Error", "Could not capture image")
        return

    # Convert the frame to PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Display captured image
    image.thumbnail((300, 300))
    img = ImageTk.PhotoImage(image)
    panel.config(image=img)
    panel.image = img

    # Save the image to a temporary file
    temp_file_path = "temp_capture.jpg"
    image.save(temp_file_path)

    # Start processing in a new thread
    threading.Thread(target=process_image, args=(temp_file_path,)).start()

# Create main window with ThemedTk for modern themes
root = ThemedTk(theme="breeze")
root.title("Surroundings Description App")
root.geometry("600x700")
root.configure(bg="#e0e0e0")

# Add a title label
title_label = tk.Label(root, text="CCTV feed Description App", font=("Arial", 18, "bold"), bg="#e0e0e0")
title_label.pack(pady=20)

# Create and place widgets with modern styling
open_button = ttk.Button(root, text="Open Image", command=open_file)
open_button.pack(pady=15, fill='x', padx=20)

capture_button = ttk.Button(root, text="Capture Photo", command=capture_photo)
capture_button.pack(pady=15, fill='x', padx=20)

panel = tk.Label(root, bg="white", relief="solid")
panel.pack(padx=20, pady=20, fill='both', expand=True)

caption_label = tk.Label(root, text="Caption will appear here.", font=("Arial", 14), bg="#e0e0e0", wraplength=500)
caption_label.pack(pady=15)

# Add a progress bar
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=15)

# Add tooltips for buttons
def create_tooltip(widget, text):
    tooltip = tk.Toplevel(widget)
    tooltip.wm_overrideredirect(True)
    tooltip.wm_geometry(f"+{widget.winfo_rootx()}+{widget.winfo_rooty()+30}")
    label = tk.Label(tooltip, text=text, background="lightyellow", relief="solid", padx=5, pady=5)
    label.pack()
    widget.bind("<Enter>", lambda e: tooltip.lift())
    widget.bind("<Leave>", lambda e: tooltip.destroy())

create_tooltip(open_button, "Open an image file from your computer.")
create_tooltip(capture_button, "Capture a photo using your webcam.")

# Run the GUI event loop
root.mainloop()
