import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import tempfile
import torch
import pickle

# Tiêu đề app
st.set_page_config(page_title="Phân loại trái mít với YOLOv8")
st.title("🍈 Phân loại trái mít: chín / sống / non bằng YOLOv8")
st.write("Tải ảnh chứa trái mít để mô hình nhận dạng và phân loại độ chín.")

# Load model
@st.cache_resource
def load_model():
    model_path = "best.pt"  
    try:
        return YOLO(model_path)
    except (RuntimeError, pickle.UnpicklingError) as e:
        st.error("❌ Không thể load model. Có thể do phiên bản Torch không tương thích.")
        st.stop()

model = load_model()

# Upload ảnh
uploaded_file = st.file_uploader("📤 Tải ảnh lên", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Ảnh đã tải", use_column_width=True)

    # Lưu ảnh tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Dự đoán
    st.write("🔍 Đang phân tích...")
    try:
        results = model.predict(temp_path, conf=0.4)
    except Exception as e:
        st.error(f"❌ Lỗi khi dự đoán ảnh: {str(e)}")
        os.remove(temp_path)
        st.stop()

    # Hiển thị ảnh có bounding box
    for r in results:
        im_array = r.plot()  # vẽ bbox
        im = Image.fromarray(im_array[..., ::-1])
        st.image(im, caption="📌 Kết quả nhận dạng", use_column_width=True)

    # Hiển thị nhãn và độ tin cậy
    st.subheader("📋 Kết quả chi tiết:")
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            st.write(f"- **{class_name}** ({conf:.2f})")

    # Xoá ảnh tạm
    os.remove(temp_path)
