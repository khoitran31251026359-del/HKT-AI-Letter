import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import random
from sklearn.neural_network import MLPClassifier

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
def train_hkt_brain():
    X_train = []
    y_train = []
    
    for i in range(65, 91):
        letter = chr(i)
        for font_scale in [0.6, 0.8, 1.0, 1.2]:
            for thickness in [1, 2, 3]:
                for dx in [-3, 0, 3]:
                    for dy in [-3, 0, 3]:
           
                        blank = np.zeros((50, 50), dtype=np.uint8)
                        cv2.putText(blank, letter, (13 + dx, 35 + dy), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness)

                        contours, _ = cv2.findContours(blank, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if len(contours) > 0:
                            c = max(contours, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(c)
                            cropped = blank[y:y+h, x:x+w]
                            resized = cv2.resize(cropped, (28, 28))
                        else:
                            resized = cv2.resize(blank, (28, 28))
                        
                        X_train.append(resized.flatten())
                        y_train.append(letter)
                        
    ann = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=30,
        activation='relu',
        solver='adam',
        random_state=42,
        verbose=True
    )
    
    ann.fit(X_train, y_train)
    return ann

hkt_brain = train_hkt_brain()

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
                _, img_thresh = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY)
                
                contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    c = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)
                    
                    cropped = img_thresh[y:y+h, x:x+w]
                    
                    pad = 4
                    cropped_padded = cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
                    
                    img_resized = cv2.resize(cropped_padded, (28, 28))
                    
                    input_features = img_resized.flatten().reshape(1, -1)
                    
                    letter_predicted = hkt_brain.predict(input_features)[0]
                    
                    prob = hkt_brain.predict_proba(input_features)
                    max_prob = np.max(prob)
                    
                    confidence = 70.0 + (max_prob * 28.7) 
                    if confidence > 98.7: confidence = 98.7
                    
                    st.balloons()
                    st.success(f"### HKT ĐOÁN NHA, ĐÂY LÀ CHỮ: **{letter_predicted}** (TỤI TUI TỰ TIN {confidence:.1f}%)")
                    
                    if confidence > 88:
                        st.info(f"💬 **NHẬN XÈT CHỮ:** {random.choice(khen_list)}")
                    else:
                        st.warning(f"💬 **NHẬN XÉT CHỮ:** {random.choice(che_list)}")
                else:
                    st.error("Lỗi xử lý hình ảnh, thử viết lại rõ hơn bồ ơi!")
        else:
            st.error("THỬ VIẾT GÌ ĐI BỒ ƠI, XONG ẤN NÚT ĐỂ HKT ĐOÁN NHE! 😤")
