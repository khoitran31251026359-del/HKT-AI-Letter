import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import random
import tensorflow as tf
from tensorflow.keras import layers, models


st.set_page_config(page_title="HKT Recognition Pro", layout="centered")
st.markdown("""
    <style>
    .main-title { font-size: 36px; font-weight: bold; color: #00DBDE; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #888; text-align: center; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔥 HKT RECOGNITION PRO 🔥</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Phiên bản ANN tối ưu hóa ma trận - Đạt độ chính xác tối đa của nhóm HKT</div>', unsafe_allow_html=True)

@st.cache_resource
def load_optimized_ann():
    x_train = []
    y_train = []
    
    for i in range(65, 91):
        letter = chr(i)
        for font_scale in [0.7, 1.0, 1.3]:
            for thickness in [1, 2, 3]:
                for dx in [-3, 0, 3]:
                    for dy in [-3, 0, 3]:

                        blank = np.zeros((50, 50), dtype=np.uint8)
                        cv2.putText(blank, letter, (13 + dx, 35 + dy), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness)
                        
                        for angle in [-15, 0, 15]:
                            if angle != 0:
                                M = cv2.getRotationMatrix2D((25, 25), angle, 1.0)
                                rotated = cv2.warpAffine(blank, M, (50, 50))
                            else:
                                rotated = blank.copy()

                            contours, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if len(contours) > 0:
                                c = max(contours, key=cv2.contourArea)
                                x, y, w, h = cv2.boundingRect(c)
                                cropped = rotated[y:y+h, x:x+w]
                                resized = cv2.resize(cropped, (28, 28))
                            else:
                                resized = cv2.resize(rotated, (28, 28))
                            
                            x_train.append(resized.flatten())
                            y_train.append(i - 65)
                        
    x_train = np.array(x_train).astype('float32') / 255.0
    y_train = np.array(y_train)
    
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dense(26, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    model.fit(x_train, y_train, epochs=25, batch_size=128, shuffle=True, verbose=0)
    return model

with st.spinner('🧙‍♂️ HKT đang huấn luyện mạng Deep ANN siêu cấp, đợi tí nhé...'):
    model = load_optimized_ann()

st.sidebar.header("🛠️ CÔNG CỤ HKT ANN PRO")
tool_mode = st.sidebar.radio("Chọn chế độ:", ("Bút vẽ ✏️", "Gôm tẩy 🧽"))

drawing_mode = "freedraw"

if tool_mode == "Bút vẽ ✏️":
    stroke_width = st.sidebar.slider("Độ đậm nét vẽ:", min_value=5, max_value=40, value=22, step=1)
    stroke_color = "#FFFFFF" 
else:
    stroke_width = st.sidebar.slider("Kích thước gôm tẩy:", min_value=10, max_value=60, value=35, step=1)
    stroke_color = "rgba(0, 0, 0, 1)" 

st.sidebar.markdown("---")
st.sidebar.success("HKT cảm ơn mọi người đã ghé qua :3")

st.markdown("✍️ **Thử viết chữ cái vào đây xem nào:**")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  
    stroke_width=stroke_width,
    stroke_color=stroke_color, 
    background_color="rgba(0, 0, 0, 1)", 
    height=320,
    width=320,
    drawing_mode=drawing_mode,
    update_streamlit=True,
    key="hkt_ann_pro_canvas",
)

khen_list = ["Thi luyện viết chữ đẹp đi bạn ơiii✨", "Quá đẹp! HKT chấm nét chữ này 10 điểm không có nhưng!", "Như in trong sách giáo khoa ra z, vuýp!😎"]
che_list = ["Nét hơi nguệch ngoạc nhưng mà cũm đáng iu 😜", "Oi viết nắn nót thêm xí đi bồ ơi!", "Chữ như mèo cào ấy bồ, làm khó cho tui quá 🦤"]

st.markdown("---")
predict_button = st.button("🔮 ĐỂ TUI ĐOÁN! 🔮", use_container_width=True)

if predict_button:
    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        if np.sum(img[:, :, :3]) > 0:
            with st.spinner('HKT AI đang quét ma trận đặc trưng...'):
                img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                
                contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    c = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)
                    cropped = img_gray[y:y+h, x:x+w]
                    
                    pad = max(w, h) // 4
                    img_gray = cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

                _, img_thresh = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY)
                img_resized = cv2.resize(img_thresh, (28, 28))

                img_ready = img_resized.reshape((1, 784)).astype('float32') / 255.0
                
                preds = model.predict(img_ready)
                letter = chr(np.argmax(preds) + 65)
                confidence = np.max(preds) * 100
                
            st.balloons()
            st.success(f"### HKT ĐOÁN NHA, ĐÂY LÀ CHỮ: **{letter}** (TỤI TUI TỰ TIN {confidence:.1f}%)")
            
            if confidence > 82:
                st.info(f"💬 **NHẬN XÈT CHỮ:** {random.choice(khen_list)}")
            else:
                st.warning(f"💬 **NHẬN XÉT CHỮ:** {random.choice(che_list)}")
        else:
            st.error("THỬ VIẾT GÌ ĐI BỒ ƠI, XONG ẤN NÚT ĐỂ HKT ĐOÁN NHE! 😤")
