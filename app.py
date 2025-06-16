import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import tempfile
import torch
import pickle
import numpy as np
import time

# Tiêu đề app
st.set_page_config(page_title="Phân loại trái mít", page_icon="🍈", layout="centered")
st.title("🍈 Ứng dụng phân loại trái mít bằng YOLOv8")
st.markdown(
    '''
    ### 📘 Hướng dẫn sử dụng:
    1. Nhấn **"Tải ảnh lên"** để chọn ảnh trái mít từ máy tính.
    2. Hệ thống sẽ **tự động phân tích ảnh** và phân loại trái mít thành:
        - 🟡 **Mít chín** – vỏ vàng, có mùi thơm  
        - 🟢 **Mít sống** – vỏ xanh đậm, chưa chín  
        - 🟤 **Mít non** – quả nhỏ, gai mịn
    3. Kết quả sẽ hiển thị ngay bên dưới kèm hình ảnh minh hoạ.
    '''
)

# Load model
@st.cache_resource
def load_model():
    model_path = "best.pt"
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Không thể load model. Lỗi thực tế: {str(e)}")
        st.stop()

model = load_model()

# Từ điển tên lớp → tiếng Việt
label_map = {
    "jackfruit_ripe": "🟡 Mít chín – vỏ vàng, có mùi thơm",
    "jackfruit_unripe": "🟢 Mít sống – vỏ xanh đậm, chưa chín",
    "jackfruit_young": "🟤 Mít non – quả nhỏ, gai mịn"
}

# Upload ảnh
uploaded_file = st.file_uploader("📤 Tải ảnh trái mít (jpg, png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="📷 Ảnh bạn đã chọn", use_column_width=True)

    # Lưu ảnh tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Dự đoán
    st.markdown("⏳ **Đang phân tích ảnh...**")
    start_time = time.time()
    try:
        results = model.predict(temp_path, conf=0.4, imgsz=640)
    except Exception as e:
        st.error(f"❌ Lỗi khi phân tích ảnh: {str(e)}")
        os.remove(temp_path)
        st.stop()
    end_time = time.time()

    # Hiển thị ảnh có kết quả
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        st.image(im, caption="📌 Kết quả nhận dạng", use_column_width=True)

    # Hiển thị nhãn và độ tin cậy
    st.subheader("📋 Kết quả phân loại:")
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name_en = model.names[cls_id]
            class_label = label_map.get(class_name_en, class_name_en)
            st.write(f"- **{class_label}**  \n  → Xác suất: `{conf:.2%}`")

    st.success(f"✅ Phân tích hoàn tất sau {end_time - start_time:.2f} giây.")
    os.remove(temp_path)
