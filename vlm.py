import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq
import queue
import time
import os

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: {DEVICE}")

# Streamlit app layout
st.title("Live Webcam Image Description")
st.write("This app captures webcam frames and describes pairs of frames using SmolVLM.")

# Initialize session state
if "frames" not in st.session_state:
    st.session_state.frames = []
if "description" not in st.session_state:
    st.session_state.description = ""
if "frame_queue" not in st.session_state:
    st.session_state.frame_queue = queue.Queue(maxsize=2)
if "running" not in st.session_state:
    st.session_state.running = False

# Load model and processor
@st.cache_resource
def load_model():
    try:
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Base")
        model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-Base",
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
        ).to(DEVICE)
        return processor, model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None

processor, model = load_model()
if processor is None or model is None:
    st.stop()

# Function to find available webcam
def find_webcam(max_index=5):
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            st.write(f"Webcam found at index {index}")
            return cap, index
        cap.release()
    return None, -1

# Initialize webcam
cap, webcam_index = find_webcam()
if cap is None:
    st.error("No webcam found. Check connections, permissions, or try a different index.")
    st.stop()

# Placeholder for live feed
frame_placeholder = st.empty()

# Start/Stop button
if st.button("Start/Stop Webcam"):
    st.session_state.running = not st.session_state.running
    if not st.session_state.running:
        cap.release()
        frame_placeholder.empty()
        st.session_state.description = ""
        st.session_state.frames = []
        st.session_state.frame_queue = queue.Queue(maxsize=2)

# Main loop
while st.session_state.running and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame.")
        st.session_state.running = False
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (320, 240))
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Display live feed
    frame_placeholder.image(img_rgb, channels="RGB", caption="Live Webcam Feed")

    # Add frame to queue
    try:
        st.session_state.frame_queue.put_nowait(pil_img)
    except queue.Full:
        st.session_state.frame_queue.get()
        st.session_state.frame_queue.put(pil_img)

    # Process if 2 frames are available
    if st.session_state.frame_queue.qsize() == 2:
        frames = list(st.session_state.frame_queue.queue)
        try:
            # Create input messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": "Can you describe the two images?"}
                    ]
                },
            ]

            # Prepare inputs
            with st.spinner("Processing frames..."):
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=prompt, images=frames, return_tensors="pt")
                inputs = inputs.to(DEVICE)

                # Generate outputs
                generated_ids = model.generate(**inputs, max_new_tokens=100)  # Reduced for speed
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            # Update session state
            st.session_state.description = generated_texts[0]
            st.session_state.frames = frames

        except Exception as e:
            st.session_state.description = f"Error processing frames: {str(e)}"

    # Display description and frames
    if st.session_state.description:
        st.subheader("Generated Description")
        st.write(st.session_state.description)

    if st.session_state.frames:
        st.subheader("Processed Frames")
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.frames[0], caption="Frame 1", use_column_width=True)
        with col2:
            st.image(st.session_state.frames[1], caption="Frame 2", use_column_width=True)

    # Control frame rate
    time.sleep(0.3)  # Increased to reduce load

# Release webcam when stopped
if not st.session_state.running:
    cap.release()

# Instructions and troubleshooting
st.markdown("""
### Instructions
1. Ensure your webcam is connected and not used by other apps.
2. Click "Start/Stop Webcam" to begin capturing frames.
3. The app processes pairs of frames and displays descriptions.
4. Click the button again to stop.

### Troubleshooting
- **Webcam Not Found**:
  - Check if the webcam is connected: `ls /dev/video*` in terminal.
  - Ensure no other apps (e.g., Zoom) are using the webcam.
  - Verify permissions: Run `sudo usermod -a -G video $USER` and reboot.
  - Install V4L2: `sudo apt-get install v4l-utils` and test with `v4l2loopback`.
  - Try a USB webcam or different index (edit code to test indices 0-5).
- **Performance**: Use a GPU for faster processing. Increase `time.sleep` to 0.5 if laggy.
- **Runtime Errors**: If event loop errors persist, try Python 3.11 or downgrade Streamlit (`pip install streamlit==1.35.0`).
- **Model Download**: Install `hf_xet` for faster downloads: `pip install hf_xet`.

**Note**: This app works locally. For cloud deployment, contact the developer for WebRTC solutions.
""")
