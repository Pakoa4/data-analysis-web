import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# --- ส่วนที่แก้ไขใหม่: ตั้งค่าฟอนต์ภาษาไทยแบบครอบคลุม ---
# กำหนดรายชื่อฟอนต์ภาษาไทยยอดฮิตเผื่อไว้หลายๆ ตัว (รองรับทั้ง Windows และ Mac)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Leelawadee UI', 'Leelawadee', 'Thonburi', 'Cordia New', 'Arial Unicode MS']
plt.rcParams['figure.dpi'] = 200 # เพิ่มความคมชัด (Resolution)
# --------------------------------------------------------

# (โค้ดส่วนอื่นๆ ตั้งแต่ st.set_page_config... คงไว้เหมือนเดิมทั้งหมดได้เลยครับ)
# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ระบบวิเคราะห์ชุดข้อมูล", layout="wide")
st.title("📊 ระบบวิเคราะห์ชุดข้อมูลและสถิติ")

# 1. ส่วนรับไฟล์จากผู้ใช้งาน (File Uploader)
st.sidebar.header("นำเข้าข้อมูล")
uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ (CSV หรือ Excel)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # --- ระบบดักจับและแปลงการอ่านภาษาไทย (Encoding) ---
        if uploaded_file.name.endswith('csv'):
            try:
                # ลองอ่านแบบมาตรฐานสากล (UTF-8) ก่อน
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                # ถ้าอ่านภาษาไทยไม่ออก (Error) ให้ใช้การเข้ารหัสแบบ Windows Thai (TIS-620)
                uploaded_file.seek(0) # รีเซ็ตไฟล์กลับไปจุดเริ่มต้นก่อนอ่านใหม่
                df = pd.read_csv(uploaded_file, encoding='tis-620')
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success(f"อัปโหลดไฟล์ {uploaded_file.name} สำเร็จ! (จำนวน {df.shape[0]} แถว, {df.shape[1]} คอลัมน์)")

        # แบ่งหน้าจอเป็น 2 ฝั่ง
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 ตัวอย่างชุดข้อมูล (Raw Data)")
            
            # สร้างช่องให้พิมพ์ตัวเลข (ค่าเริ่มต้นคือ 100 แถว)
            row_count = st.number_input("จำนวนแถวที่ต้องการแสดง", min_value=1, max_value=len(df), value=100)
            
            # แสดงข้อมูลตามจำนวนที่พิมพ์
            st.dataframe(df.head(row_count))

        with col2:
            st.subheader("🗂️ การแบ่งหมวดหมู่ข้อมูล (Data Types)")
            # แยกหมวดหมู่ว่าเป็นตัวเลข (Numerical) หรือข้อความ (Categorical)
            type_df = pd.DataFrame(df.dtypes, columns=['ประเภทข้อมูล']).astype(str)
            st.dataframe(type_df)

        st.markdown("---")

        # 2. การวิเคราะห์สถิติเบื้องต้น
        st.subheader("📈 สถิติเชิงพรรณนา (Descriptive Statistics)")
        # เลือกเฉพาะคอลัมน์ที่เป็นตัวเลขเพื่อหาค่าเฉลี่ย, สูงสุด, ต่ำสุด
        st.dataframe(df.describe())

        st.markdown("---")

        # 3. การวิเคราะห์ความน่าจะเป็นและกราฟ
        st.subheader("📊 วิเคราะห์เจาะลึกและกราฟ")
        
        # --- ระบบกรองคอลัมน์ที่ไม่ต้องการ (เพิ่ม วันที่, เวลา, Date) ---
        ignore_keywords = ['id', 'รหัส', 'ลำดับ', 'จำนวน', 'no.', 'index', 'วันที่', 'date', 'เวลา', 'time', 'เดือน', 'ปี', 'year']
        
        # กรองคอลัมน์ โดยแปลงชื่อคอลัมน์เป็นตัวพิมพ์เล็กก่อนเพื่อเช็คคำต้องห้าม
        valid_columns = [
            col for col in df.columns 
            if not any(keyword in str(col).lower() for keyword in ignore_keywords)
        ]
        
        # ป้องกัน Error กรณีที่ไฟล์มีแต่คอลัมน์ที่โดนกรองทิ้งหมด
        if len(valid_columns) == 0:
            valid_columns = df.columns
            
        # ใช้คอลัมน์ที่ผ่านการกรองแล้วมาแสดงเป็นตัวเลือก
        selected_col = st.selectbox("เลือกตัวแปร (Column) ที่ต้องการวิเคราะห์:", valid_columns)

        # --- การตั้งค่ากราฟความละเอียดสูง ---
        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

        # ตรวจสอบประเภทข้อมูลเพื่อวาดกราฟ
        if df[selected_col].dtype == 'object' or df[selected_col].dtype == 'bool':
            # ข้อมูลแบบหมวดหมู่ (Categorical) -> สร้าง Bar Chart
            value_counts = df[selected_col].value_counts()
            probabilities = df[selected_col].value_counts(normalize=True) * 100
            
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette="viridis")
            
            ax.set_title(f"ความถี่ของข้อมูล: {selected_col}", fontsize=14, fontweight='bold')
            ax.set_ylabel("ความถี่ (Count)", fontsize=12)
            ax.set_xlabel(selected_col, fontsize=12)
            
            # หมุนข้อความแกน X เพื่อไม่ให้ตัวหนังสือทับกัน กรณีชื่อหมวดหมู่ยาว
            plt.xticks(rotation=45, ha='right')
            
            # เพิ่มตัวเลขบอกจำนวนกำกับไว้ด้านบนของกราฟแท่งแต่ละอัน
            for i, v in enumerate(value_counts.values):
                ax.text(i, v + (max(value_counts.values) * 0.02), str(v), ha='center', fontsize=10)

            st.pyplot(fig)
            
            st.write("**สัดส่วน / ความน่าจะเป็น (Probability):**")
            st.dataframe(probabilities.rename("ความน่าจะเป็น (%)"))
            
        else:
            # ข้อมูลแบบตัวเลข (Numerical) -> สร้าง Histogram
            sns.histplot(df[selected_col], kde=True, ax=ax, color="blue", bins=20)
            
            ax.set_title(f"การแจกแจงความน่าจะเป็นของ: {selected_col}", fontsize=14, fontweight='bold')
            ax.set_ylabel("ความถี่", fontsize=12)
            ax.set_xlabel(selected_col, fontsize=12)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
else:
    st.info("กรุณาอัปโหลดไฟล์ข้อมูลที่แถบด้านซ้ายเพื่อเริ่มต้นการวิเคราะห์")