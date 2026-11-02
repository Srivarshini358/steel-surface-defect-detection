import os
import gradio as gr
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(MODEL_PATH)

def detect_defect(image):
    if image is None:
        return None
    results = model.predict(source=image, conf=0.25, verbose=False)
    result_image = results[0].plot()
    return Image.fromarray(result_image[..., ::-1])

demo = gr.Interface(
    fn=detect_defect,
    inputs=gr.Image(type="numpy", label="Upload Steel Surface Image"),
    outputs=gr.Image(label="Defect Detection Result"),
    title="Steel Surface Defect Detection",
    description="Upload a steel surface image to detect and localize surface defects."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
