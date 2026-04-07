from notebook_utils import device_widget
from PIL import Image
import openvino as ov
from ov_paddleocr_vl import OVPaddleOCRVLForCausalLM
from gradio_helper import make_demo
from pathlib import Path
import time
import os


LLM_INT4_COMPRESS = False
LLM_INT8_COMPRESS = True
VISION_INT8_QUANT = False

#device = device_widget("CPU")

#OV_OUT_DIR = '/home/aistudio/openvino_notebooks/notebooks/paddleocr_vl/ov_paddleocr_vl_model'
OV_OUT_DIR = OV_OUT_DIR = os.path.abspath("/home/ippr/openvino_notebooks/notebooks/paddleocr_vl/ov_paddleocr_vl_model")

device = "CPU"
# Parameters
ov_model_path = str(OV_OUT_DIR)
task = "ocr"
max_new_tokens = 512

llm_infer_list = []
vision_infer = []
core = ov.Core()

paddleocr_vl_model = OVPaddleOCRVLForCausalLM(
    core=core,
    ov_model_path=ov_model_path,
    #device=device.value,
    device=device,
    llm_int4_compress=False,
    llm_int8_compress=True,
    vision_int8_quant=False,
    llm_int8_quant=True,
    llm_infer_list=llm_infer_list,
    vision_infer=vision_infer,
)

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

# Prepare a test image.
img = None
if img is None:
    img =  "/home/ippr/openvino_notebooks/notebooks/paddleocr_vl/test.png"

image_path = str(img)
image = Image.open(image_path).convert("RGB")
# image = image.resize((1200, 800), Image.Resampling.LANCZOS)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPTS[task]},
        ],
    }
]

print("\n" + "=" * 60)
print("🔄 Loading OpenVINO model...")
print("=" * 60)

# OpenVINO version print
try:
    print("OpenVINO version:\n", ov.get_version())
except Exception:
    print("OpenVINO version:\n", getattr(ov, "__version__", "unknown"))
print()

generation_config = {
    "bos_token_id": paddleocr_vl_model.tokenizer.bos_token_id,
    "eos_token_id": paddleocr_vl_model.tokenizer.eos_token_id,
    "pad_token_id": paddleocr_vl_model.tokenizer.pad_token_id,
    "max_new_tokens": int(max_new_tokens),
    "do_sample": False,
}

start = time.perf_counter()
response, history = paddleocr_vl_model.chat(messages=messages, generation_config=generation_config)
chat_time = time.perf_counter() - start


print("\n" + "=" * 60)
#print(f"📄 {device} OpenVINO {task} result:")
print(f"📄 {device} OpenVINO {task} result:")
print("=" * 60)
print(response)
print("=" * 60)

# Parameters
ov_model_path = str(OV_OUT_DIR)
task = "ocr"
max_new_tokens = 512

llm_infer_list = []
vision_infer = []
core = ov.Core()

paddleocr_vl_model = OVPaddleOCRVLForCausalLM(
    core=core,
    ov_model_path=ov_model_path,
    device=device,
    llm_int4_compress=False,
    llm_int8_compress=True,
    vision_int8_quant=False,
    llm_int8_quant=True,
    llm_infer_list=llm_infer_list,
    vision_infer=vision_infer,
)

demo = make_demo(paddleocr_vl_model)

# 最终稳定配置
demo.launch(
    server_name="0.0.0.0",
    server_port=8899,
    debug=False,
    share=True,
    height=900
)