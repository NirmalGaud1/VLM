import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import torch
from PIL import Image
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq
import av
import queue

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Streamlit app layout
st.title("Live Webcam Image Description with SmolVLM")
st.write("This app captures live webcam frames and generates descriptions for pairs of frames.")

# Initialize session state for storing frames and descriptions
if "frames" not in st.session_state:
    st.session_state.frames = []
if "description" not in st.session_state:
    st.session_state.description = ""

# Load model and processor (load once to avoid reloading)
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Base")
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-Base",
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)
    return processor, model

processor, model = load_model()

# Video processor class for webcam streaming
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=2)  # Store up to 2 frames

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Convert BGR (OpenCV) to RGB (PIL)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Add frame to queue
        try:
            self.frame_queue.put_nowait(pil_img)
        except queue.Full:
            self.frame_queue.get()  # Remove oldest frame
            self.frame_queue.put(pil_img)

        # If we have 2 frames, process them
        if self.frame_queue.qsize() == 2:
            frames = list(self.frame_queue.queue)
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
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=prompt, images=frames, return_tensors="pt")
                inputs = inputs.to(DEVICE)

                # Generate outputs
                generated_ids = model.generate(**inputs, max_new_tokens=500)
                generated_texts = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                # Update session state with description
                st.session_state.description = generated_texts[0]
                st.session_state.frames = frames  # Store frames for display

            except Exception as e:
                st.session_state.description = f"Error processing frames: {str(e)}"

        return frame

# Webcam streaming
st.subheader("Webcam Feed")
ctx = webrtc_streamer(
    key="webcam",
    video_processor_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

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

# Instructions
st.markdown("""
### Instructions
1. Allow webcam access when prompted.
2. The app captures live frames and processes pairs of frames when available.
3. View the generated description and the processed frames below.
4. Ensure good lighting and a clear view for better results.

**Note**: Processing may be slow on CPU. For optimal performance, use a GPU with CUDA.
""")
