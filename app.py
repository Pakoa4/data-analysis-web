import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import urllib.request
import matplotlib.font_manager as fm

# --- ส่วนดาวน์โหลดและบังคับใช้ฟอนต์ภาษาไทย 100% ---
font_url = "https://github.com/google/fonts/raw/main/ofl/prompt/Prompt-Regular.ttf"
font_path = "Prompt-Regular.ttf"

if not os.path.exists(font_path):
    urllib.request.urlretrieve(font_url, font_path)

fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Prompt' 
plt.rcParams['figure.dpi'] = 200
# --------------------------------------------------------

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ระบบวิเคราะห์ชุดข้อมูล", layout="wide")
st.title("📊 ระบบวิเคราะห์ชุดข้อมูลและสถิติ")

# 1. ส่วนรับไฟล์จากผู้ใช้งาน (File Uploader)
st.sidebar.header("นำเข้าข้อมูล")
uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ (CSV หรือ Excel)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # ระบบดักจับและแปลงการอ่านภาษาไทย
        if uploaded_file.name.endswith('csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='tis-620')
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success(f"อัปโหลดไฟล์ {uploaded_file.name} สำเร็จ! (จำนวน {df.shape[0]:,} แถว, {df.shape[1]:,} คอลัมน์)")

        # แบ่งหน้าจอเป็น 2 ฝั่ง
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 ตัวอย่างชุดข้อมูล (Raw Data)")
            row_count = st.number_input("จำนวนแถวที่ต้องการแสดง", min_value=1, max_value=len(df), value=100)
            st.dataframe(df.head(row_count))

        with col2:
            st.subheader("🗂️ การแบ่งหมวดหมู่ข้อมูล (Data Types)")
            type_df = pd.DataFrame(df.dtypes, columns=['ประเภทข้อมูล']).astype(str)
            st.dataframe(type_df)

        st.markdown("---")

        # 2. การวิเคราะห์สถิติเบื้องต้น
        st.subheader("📈 สถิติเชิงพรรณนา (Descriptive Statistics)")
        
        desc_df = df.describe()
        if not desc_df.empty:
            index_mapping = {
                'count': 'จำนวนข้อมูล (Count)',
                'mean': 'ค่าเฉลี่ย (Mean)',
                'std': 'ค่าเบี่ยงเบนมาตรฐาน (SD)',
                'min': 'ค่าน้อยที่สุด (Min)',
                '25%': 'เปอร์เซ็นไทล์ที่ 25 (Q1)',
                '50%': 'ค่ากึ่งกลาง/มัธยฐาน (Median)',
                '75%': 'เปอร์เซ็นไทล์ที่ 75 (Q3)',
                'max': 'ค่ามากที่สุด (Max)'
            }
            desc_df = desc_df.rename(index=index_mapping)
            st.dataframe(desc_df.style.format("{:,.2f}"), use_container_width=True)
        else:
            st.info("ไม่พบคอลัมน์ที่เป็นตัวเลขสำหรับคำนวณสถิติ")

        st.markdown("---")

        # 3. การวิเคราะห์เจาะลึกและกราฟ (ปรับโฉมให้อ่านง่าย)
        st.subheader("📊 วิเคราะห์เจาะลึกและกราฟ")
        
        ignore_keywords = ['id', 'รหัส', 'ลำดับ', 'จำนวน', 'no.', 'index', 'วันที่', 'date', 'เวลา', 'time', 'เดือน', 'ปี', 'year']
        
        valid_columns = [
            col for col in df.columns 
            if not any(keyword in str(col).lower() for keyword in ignore_keywords)
        ]
        
        if len(valid_columns) == 0:
            valid_columns = df.columns
            
        selected_col = st.selectbox("เลือกตัวแปร (Column) ที่ต้องการวิเคราะห์:", valid_columns)

        # แปลงชื่อคอลัมน์ให้อ่านง่ายขึ้น (ตัด _ ออก และพิมพ์ใหญ่ตัวหน้า)
        readable_col = str(selected_col).replace('_', ' ').title()

        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

        # ตรวจสอบประเภทข้อมูล
        if str(df[selected_col].dtype) in ['object', 'bool', 'category', 'string']:
            # --- สร้าง Bar Chart สำหรับข้อมูลแบบข้อความ/หมวดหมู่ ---
            value_counts = df[selected_col].value_counts()
            probabilities = df[selected_col].value_counts(normalize=True) * 100
            
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette="viridis")
            
            ax.set_title(f"สัดส่วนของข้อมูล: {readable_col}", fontsize=14, fontweight='bold')
            ax.set_ylabel("จำนวน (ครั้ง)", fontsize=12)
            ax.set_xlabel(readable_col, fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.5) # เพิ่มเส้นตารางบางๆ ให้อ่านค่าแนวนอนง่ายขึ้น
            
            plt.xticks(rotation=45, ha='right')
            
            # ใส่ตัวเลขบนกราฟ
            for i, v in enumerate(value_counts.values):
                ax.text(i, v + (max(value_counts.values) * 0.02), f"{v:,}", ha='center', va='bottom', fontsize=10)

            st.pyplot(fig)
            
            # เพิ่มคำอธิบายภาษาคน
            top_1 = value_counts.index[0]
            top_1_val = value_counts.values[0]
            top_1_pct = probabilities.values[0]
            
            st.markdown("### 💡 สรุปสิ่งที่พบ (Insight)")
            st.success(f"**{top_1}** คือข้อมูลที่พบมากที่สุดเป็นอันดับ 1 ในหมวดหมู่นี้ โดยพบทั้งหมด **{top_1_val:,} ครั้ง** (คิดเป็น **{top_1_pct:,.2f}%** ของข้อมูลทั้งหมด)")
            
            st.write("**ตารางสัดส่วน / ความน่าจะเป็น (Probability):**")
            st.dataframe(probabilities.rename("สัดส่วน (%)").to_frame().style.format("{:,.2f}%"))
            
        else:
            # --- สร้าง Histogram สำหรับข้อมูลแบบตัวเลข ---
            # ปรับ bins='auto' เพื่อให้ระบบแบ่งจำนวนแท่งให้พอดีกับสายตา
            sns.histplot(df[selected_col], kde=True, ax=ax, color="blue", bins='auto')
            
            ax.set_title(f"การกระจายตัวของข้อมูล: {readable_col}", fontsize=14, fontweight='bold')
            ax.set_ylabel("จำนวนครั้งที่พบ (Frequency)", fontsize=12)
            ax.set_xlabel(f"ค่าของ {readable_col}", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.5) # เพิ่มเส้นตารางบางๆ
            
            # ใส่ตัวเลขบนกราฟ
            max_height = max([p.get_height() for p in ax.patches]) if ax.patches else 1
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.text(p.get_x() + p.get_width() / 2., height + (max_height * 0.02),
                            f"{int(height):,}",
                            ha='center', va='bottom', fontsize=9)
            
            st.pyplot(fig)

            # เพิ่มคำอธิบายภาษาคน
            mean_val = df[selected_col].mean()
            median_val = df[selected_col].median()
            min_val = df[selected_col].min()
            max_val = df[selected_col].max()

            st.markdown("### 💡 สรุปคำอธิบายกราฟ (Insight)")
            st.info(f"""
            - **ภาพรวม:** ข้อมูลชุดนี้มีค่าเริ่มต้นตั้งแต่ **{min_val:,.2f}** และสูงสุดอยู่ที่ **{max_val:,.2f}**
            - **ค่าเฉลี่ย (Mean):** โดยเฉลี่ยแล้วข้อมูลจะมีค่าอยู่ที่ประมาณ **{mean_val:,.2f}**
            - **ค่ากึ่งกลาง (Median):** ข้อมูลกว่าครึ่งหนึ่งมีค่าน้อยกว่าหรือเท่ากับ **{median_val:,.2f}**
            - 📈 **วิธีอ่านกราฟ:** แท่งที่สูงที่สุด คือช่วงตัวเลขที่พบได้บ่อยที่สุดในข้อมูลชุดนี้ (ส่วนเส้นโค้งสีน้ำเงินคือตัวช่วยดูแนวโน้มว่าข้อมูลส่วนใหญ่ไปกระจุกตัวอยู่ที่ค่าน้อยหรือค่ามาก)
            """)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
else:
    st.info("กรุณาอัปโหลดไฟล์ข้อมูลที่แถบด้านซ้ายเพื่อเริ่มต้นการวิเคราะห์")