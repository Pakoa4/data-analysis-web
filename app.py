import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ระบบวิเคราะห์ชุดข้อมูล", layout="wide")
st.title("📊 ระบบวิเคราะห์ชุดข้อมูลและสถิติ")

# 1. ส่วนรับไฟล์จากผู้ใช้งาน (File Uploader)
st.sidebar.header("นำเข้าข้อมูล")
uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ (CSV หรือ Excel)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # อ่านไฟล์ข้อมูล
        if uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success(f"อัปโหลดไฟล์ {uploaded_file.name} สำเร็จ! (จำนวน {df.shape[0]} แถว, {df.shape[1]} คอลัมน์)")

        # แบ่งหน้าจอเป็น 2 ฝั่ง
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 ตัวอย่างชุดข้อมูล (Raw Data)")
            st.dataframe(df.head(10))

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
        
        # ให้ผู้ใช้เลือกคอลัมน์ที่ต้องการวิเคราะห์
        selected_col = st.selectbox("เลือกตัวแปร (Column) ที่ต้องการวิเคราะห์:", df.columns)

        fig, ax = plt.subplots(figsize=(8, 4))

        # ตรวจสอบประเภทข้อมูลเพื่อวาดกราฟที่เหมาะสม
        if df[selected_col].dtype == 'object' or df[selected_col].dtype == 'bool':
            # ข้อมูลแบบหมวดหมู่ (Categorical) -> สร้าง Bar Chart แสดงความถี่และความน่าจะเป็น
            value_counts = df[selected_col].value_counts()
            probabilities = df[selected_col].value_counts(normalize=True) * 100
            
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette="viridis")
            ax.set_title(f"ความถี่ของ {selected_col}")
            ax.set_ylabel("จำนวน (Count)")
            st.pyplot(fig)
            
            st.write("**สัดส่วน / ความน่าจะเป็น (Probability):**")
            st.dataframe(probabilities.rename("ความน่าจะเป็น (%)"))
            
        else:
            # ข้อมูลแบบตัวเลข (Numerical) -> สร้าง Histogram พร้อมเส้นโค้งการแจกแจงความน่าจะเป็น (KDE)
            sns.histplot(df[selected_col], kde=True, ax=ax, color="blue")
            ax.set_title(f"การแจกแจงความน่าจะเป็นของ {selected_col}")
            ax.set_ylabel("ความถี่")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
else:
    st.info("กรุณาอัปโหลดไฟล์ข้อมูลที่แถบด้านซ้ายเพื่อเริ่มต้นการวิเคราะห์")