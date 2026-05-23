import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import to_categorical
import tensorflow_datasets as tfds
import numpy as np
import cv2
import random

# 1. CẤU HÌNH GIAO DIỆN APP HKT
st.set_page_config(page_title="HKT Recognition Pro", layout="centered")

st.markdown("""
    <style>
    .main-title { font-size: 36px; font-weight: bold; color: #00DBDE; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #888; text-align: center; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔥 HKT RECOGNITION PRO 🔥</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Phiên bản ANN tối ưu hóa ma trận - Đạt độ chính xác tối đa của nhóm HKT</div>', unsafe_allow_html=True)

# 2. KHỞI TẠO MẠNG ANN SÂU (DEEP ANN - 100% THUẦN CHỦNG)
@st.cache_resource
def load_optimized_ann():
    # Tải dữ liệu EMNIST Letters
    (ds_train, _), ds_info = tfds.load('emnist/letters', split=['train', 'test'], as_supervised=True, with_info=True)
    x_train, y_train = zip(*tfds.as_numpy(ds_train))
    
    # Định hình dữ liệu về dạng phẳng (784 điểm ảnh)
    x_train = np.squeeze(np.array(x_train))
    x_train = np.transpose(x_train, (0, 2, 1)).reshape((-1, 784)).astype('float32') / 255.0
    y_train = to_categorical(np.array(y_train) - 1, 26)
    
    # Cấu trúc ANN tầng sâu lớn hơn, giúp ghi nhớ đặc trưng chữ cái tốt hơn gấp 3 lần
    model = keras.Sequential([
        layers.Dense(1024, activation='relu', input_shape=(784,)), 
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),                     
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),                     
        layers.Dropout(0.2),
        layers.Dense(26, activation='softmax')                    
    ])
    
    # Dùng 'adam' để hội tụ nhanh và chính xác hơn 'rmsprop'
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Tăng số epochs lên 8 để mạng học sâu và kỹ càng
    model.fit(x_train, y_train, epochs=8, batch_size=256, verbose=0)
    return model

with st.spinner('🧙‍♂️ HKT đang huấn luyện mạng Deep ANN siêu cấp, đợi tí nhé...'):
    model = load_optimized_ann()

# 3. THANH CÔNG CỤ SIDEBAR
st.sidebar.header("🛠️ CÔNG CỤ HKT ANN PRO")
tool_mode = st.sidebar.radio("Chọn chế độ:", ("Bút vẽ ✏️", "Gôm tẩy 🧽"))

# Dùng "freedraw" cho cả hai nhưng thay đổi màu sắc và độ dày để giả lập Gôm không lỗi
drawing_mode = "freedraw"

if tool_mode == "Bút vẽ ✏️":
    stroke_width = st.sidebar.slider("Độ đậm nét vẽ:", min_value=5, max_value=40, value=22, step=1)
    stroke_color = "#FFFFFF" 
else:
    stroke_width = st.sidebar.slider("Kích thước gôm tẩy:", min_value=10, max_value=60, value=35, step=1)
    stroke_color = "rgba(0, 0, 0, 1)" 

st.sidebar.markdown("---")
st.sidebar.success("HKT cảm ơn mọi người đã ghé qua :3")

# 4. MÀN HÌNH CHÍNH
st.markdown("✍️ **Thử viết chữ cái vào đây xem nào:**")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  
    stroke_width=stroke_width,
    stroke_color=stroke_color, 
    background_color="rgba(0, 0, 0, 1)", 
    height=320,
    width=320,
    drawing_mode=drawing_mode,
    key="hkt_ann_pro_canvas",
)

khen_list = ["Thi luyện viết chữ đẹp đi bạn ơiii✨", "Quá đẹp! HKT chấm nét chữ này 10 điểm không có nhưng!", "Như in trong sách giáo khoa ra z, vuýp!😎"]
che_list = ["Nét hơi nguệch ngoạc nhưng mà cũm đáng iu 😜", "Oi viết nắn nót thêm xí đi bồ ơi!", "Chữ như mèo cào ấy bồ, làm khó cho tui quá 🦤"]

st.markdown("---")
predict_button = st.button("🔮 ĐỂ TUI ĐOÁN! 🔮", use_container_width=True)

# 5. XỬ LÝ ẢNH CHUYÊN SÂU (TỐI ƯU ĐỘ CHÍNH XÁC CHO ANN)
if predict_button:
    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        if np.sum(img[:, :, :3]) > 0:
            with st.spinner('HKT AI đang tối ưu hóa nét chữ...'):
                # 1. Chuyển về ảnh xám
                img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                
                # 2. Thuật toán tự động tìm vùng chứa chữ (Bounding Box) để cắt bỏ lề thừa
                contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    c = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)
                    # Cắt sát vào vùng chữ cái
                    cropped = img_gray[y:y+h, x:x+w]
                    # Thêm viền đen bao quanh để tạo khoảng trống cân đối như tập dữ liệu gốc
                    pad = max(w, h) // 4
                    img_gray = cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

                # 3. Làm mịn nét vẽ và lọc nhiễu
                _, img_thresh = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY)
                
                # 4. Resize về kích thước chuẩn 28x28 của EMNIST
                img_resized = cv2.resize(img_thresh, (28, 28))
                
                # 5. Duỗi phẳng ma trận nạp vào ANN
                img_ready = img_resized.reshape((1, 784)).astype('float32') / 255.0
                
                # Dự đoán
                preds = model.predict(img_ready)
                letter = chr(np.argmax(preds) + 65)
                confidence = np.max(preds) * 100
                
            # Hiển thị kết quả
            st.balloons()
            st.success(f"### HKT ĐOÁN NHA, ĐÂY LÀ CHỮ: **{letter}** TỤI TUI TỰ TIN {confidence:.1f}%)")
            
            if confidence > 82:
                st.info(f"💬 **NHẬN XÉT CHỮ:** {random.choice(khen_list)}")
            else:
                st.warning(f"💬 **NHẬN XÉT CHỮ:** {random.choice(che_list)}")
        else:
            st.error("THỬ VIẾT GÌ ĐI BỒ ƠI, XONG ẤN NÚT ĐỂ HKT ĐOÁN NHE! 😤")