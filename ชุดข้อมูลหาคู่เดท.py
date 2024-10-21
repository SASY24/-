import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ชื่อของแอพ
st.title('โปรแกรมคำนวณความสำเร็จในการจับคู่')

# อัปโหลดไฟล์ Excel
uploaded_file = st.file_uploader("กรุณาอัปโหลดไฟล์ Excel ของคุณ", type=[""C:\Users\sasy\Desktop\AI 100 คน\หาคู่\ชุดข้อมูลหาคู่เดท.xlsx""])

# โหลดข้อมูล
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    st.write("ตัวอย่างข้อมูล:")
    st.dataframe(data.head())  # แสดงข้อมูลแรกของ DataFrame

    # การประมวลผลเบื้องต้น: การเข้ารหัสเลเบลสำหรับฟีเจอร์เชิงพาณิชย์
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # การเลือกฟีเจอร์ (ตรวจสอบให้แน่ใจว่า 'Match_Success' อยู่ในชุดข้อมูลของคุณ)
    if 'Match_Success' in data.columns:
        X = data.drop(columns=['Match_Success'])
        y = data['Match_Success']

        # แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # โมเดล Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # การคาดการณ์ในชุดทดสอบ
        y_pred = model.predict(X_test)

        # ประเมินโมเดล
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'ความแม่นยำ: {accuracy:.2f}')
        st.write("รายงานการจำแนกประเภท:")
        st.text(classification_report(y_test, y_pred))
        
        # การกรอกข้อมูลสำหรับการคาดการณ์
        st.subheader("คาดการณ์ความสำเร็จในการจับคู่สำหรับบุคคลใหม่")
        
        # สร้างฟอร์มสำหรับการกรอกข้อมูล
        with st.form(key='prediction_form'):
            # แทนที่ฟิลด์เหล่านี้ด้วยฟีเจอร์จริงในชุดข้อมูลของคุณ
            features = {}
            for col in X.columns:
                features[col] = st.number_input(f"กรอกค่าของ {col}", value=0)  # ปรับประเภทการกรอกข้อมูลตามที่จำเป็น

            submit_button = st.form_submit_button(label='คาดการณ์ความสำเร็จในการจับคู่')

            if submit_button:
                input_data = pd.DataFrame(features, index=[0])
                # แปลงข้อมูลนำเข้าโดยใช้ label encoders เดิม
                for column, le in label_encoders.items():
                    if column in input_data.columns:
                        input_data[column] = le.transform(input_data[column])
                
                # ทำการคาดการณ์
                prediction = model.predict(input_data)
                match_success = "ใช่" if prediction[0] == 1 else "ไม่ใช่"
                st.write(f"การคาดการณ์การจับคู่: {match_success}")

    else:
        st.error("ไม่พบคอลัมน์ 'Match_Success' ในข้อมูลที่อัปโหลด.")
