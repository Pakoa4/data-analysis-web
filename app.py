import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import urllib.request
import matplotlib.font_manager as fm
import google.generativeai as genai 

# --- ดาวน์โหลดและบังคับใช้ฟอนต์ภาษาไทย 100% ---
font_url = "https://github.com/google/fonts/raw/main/ofl/prompt/Prompt-Regular.ttf"
font_path = "Prompt-Regular.ttf"

if not os.path.exists(font_path):
    urllib.request.urlretrieve(font_url, font_path)

fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Prompt' 
plt.rcParams['figure.dpi'] = 200
# ----------------------------------------

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ระบบวิเคราะห์ชุดข้อมูลด้วย AI", layout="wide")
st.title("📊 ระบบวิเคราะห์ชุดข้อมูลและสถิติ")

# 1. ส่วนรับไฟล์จากผู้ใช้งาน
st.sidebar.header("นำเข้าข้อมูล")
uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ (CSV หรือ Excel)", type=['csv', 'xlsx'])

st.sidebar.markdown("---")
st.sidebar.header("🤖 ตั้งค่า AI (Gemini)")
api_key = st.sidebar.text_input("🔑 ใส่ API Key ของคุณที่นี่:", type="password", help="รับฟรีได้ที่ aistudio.google.com")

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='tis-620')
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success(f"อัปโหลดไฟล์ {uploaded_file.name} สำเร็จ! (จำนวน {df.shape[0]:,} แถว, {df.shape[1]:,} คอลัมน์)")

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
        
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            desc_df = numeric_df.describe()
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

        # 3. การวิเคราะห์เจาะลึกและกราฟ
        st.subheader("🔍 วิเคราะห์เจาะลึกและกราฟ (Deep Insights)")
        
        ignore_keywords = ['id', 'รหัส', 'ลำดับ', 'จำนวน', 'no.', 'index', 'วันที่', 'date', 'เวลา', 'time', 'เดือน', 'ปี', 'year']
        valid_columns = [col for col in df.columns if not any(keyword in str(col).lower() for keyword in ignore_keywords)]
        if len(valid_columns) == 0:
            valid_columns = df.columns
            
        selected_col = st.selectbox("📌 เลือกตัวแปร (Column) ที่ต้องการวิเคราะห์แบบเจาะลึก:", valid_columns)
        readable_col = str(selected_col).replace('_', ' ').title()

        # --- ระบบตรวจจับประเภทข้อมูลฉบับใหม่ล่าสุด (แม่นยำ 100%) ---
        is_numeric = pd.api.types.is_numeric_dtype(df[selected_col])
        unique_count = df[selected_col].nunique()
        
        if unique_count == 0:
            st.warning("⚠️ คอลัมน์นี้ไม่มีข้อมูล (หรือเป็นค่าว่างทั้งหมด) ไม่สามารถวิเคราะห์ได้ครับ")
        else:
            # ตัดสินใจเลือกรูปแบบการแสดงผล
            view_mode = "text"
            if is_numeric:
                if unique_count <= 15:
                    view_mode = "numeric_categorical" # ตัวเลขแต่น้อยกลุ่ม (โชว์ 4 แท็บ)
                else:
                    view_mode = "numeric" # ตัวเลขหลากหลาย (โชว์ 2 แท็บ)

            if view_mode == "text":
                # ================= กรณีที่ 1: ข้อมูลแบบข้อความ (แสดง 3 แท็บ) =================
                value_counts = df[selected_col].value_counts()
                probabilities = df[selected_col].value_counts(normalize=True) * 100
                
                tab1, tab2, tab3 = st.tabs(["📊 กราฟแท่ง (Bar Chart)", "🍩 กราฟวงกลม (Pie Chart)", "🧠 สรุปข้อมูลเชิงลึก (AI Insights)"])
                
                with tab1:
                    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
                    # แปลงแกน x เป็นข้อความเสมอ ป้องกันกราฟเบี้ยว
                    x_labels = value_counts.index.astype(str)
                    sns.barplot(x=x_labels, y=value_counts.values, ax=ax, palette="viridis")
                    
                    ax.set_title(f"จำนวนข้อมูลแต่ละหมวดหมู่: {readable_col}", fontsize=14, fontweight='bold')
                    ax.set_ylabel("จำนวน (ครั้ง)", fontsize=12)
                    ax.set_xlabel(readable_col, fontsize=12)
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    plt.xticks(rotation=45, ha='right')
                    
                    for i, v in enumerate(value_counts.values):
                        ax.text(i, v + (max(value_counts.values) * 0.02), f"{v:,}", ha='center', va='bottom', fontsize=10)
                    st.pyplot(fig)

                with tab2:
                    fig_pie, ax_pie = plt.subplots(figsize=(9, 9), dpi=200)
                    if len(value_counts) > 10:
                        top_10 = value_counts[:10]
                        others = pd.Series([value_counts[10:].sum()], index=['อื่นๆ (Others)'])
                        pie_data = pd.concat([top_10, others])
                    else:
                        pie_data = value_counts

                    # สร้างชื่อป้ายกำกับแบบละเอียด (มีชื่อ + จำนวน)
                    custom_labels = [f"{str(idx)} \n({val:,} รายการ)" for idx, val in pie_data.items()]

                    wedges, texts, autotexts = ax_pie.pie(
                        pie_data.values, labels=custom_labels, autopct='%1.1f%%', 
                        startangle=90, colors=sns.color_palette("pastel", len(pie_data)),
                        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
                    )
                    
                    for text in texts: text.set_fontsize(10)
                    for autotext in autotexts:
                        autotext.set_fontsize(10)
                        autotext.set_fontweight('bold')
                    ax_pie.set_title(f"สัดส่วนเปอร์เซ็นต์แบบละเอียด: {readable_col}", fontsize=14, fontweight='bold')
                    st.pyplot(fig_pie)
                    
                with tab3:
                    st.markdown(f"### 💡 วิเคราะห์เจาะลึกหมวดหมู่: {readable_col}")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("ความหลากหลายทั้งหมด", f"{len(value_counts)} รูปแบบ")
                    col_b.metric("กลุ่มที่พบมากที่สุด", f"ชื่อ: {value_counts.index[0]}", f"{value_counts.values[0]:,} รายการ ({probabilities.values[0]:.2f}%)")
                    col_c.metric("กลุ่มที่พบน้อยที่สุด", f"ชื่อ: {value_counts.index[-1]}", f"{value_counts.values[-1]:,} รายการ ({probabilities.values[-1]:.2f}%)")
                    
                    if st.button("✨ ให้ AI ช่วยวิเคราะห์ข้อมูลนี้", key="btn_cat"):
                        if api_key == "": st.warning("⚠️ กรุณาใส่ API Key ที่แถบด้านซ้ายมือก่อนครับ")
                        else:
                            with st.spinner("🤖 AI กำลังคิดและประมวลผล..."):
                                try:
                                    genai.configure(api_key=api_key)
                                    model = genai.GenerativeModel('gemini-pro')
                                    prompt = f"สวมบทบาทเป็นนักวิเคราะห์ข้อมูลมืออาชีพ ช่วยวิเคราะห์ข้อมูลคอลัมน์ชื่อ '{readable_col}' ให้หน่อย ข้อมูลมีความหลากหลาย {len(value_counts)} รูปแบบ. หมวดหมู่ที่พบมากที่สุดคือชื่อ '{value_counts.index[0]}' มีจำนวน {value_counts.values[0]} รายการ (คิดเป็น {probabilities.values[0]:.2f}%) และหมวดหมู่ที่พบน้อยที่สุดคือชื่อ '{value_counts.index[-1]}' มีจำนวน {value_counts.values[-1]} รายการ. ช่วยเขียนอธิบายสั้นๆ 3-4 บรรทัด (ใช้ Bullet) ระบุชื่อข้อมูลและตัวเลขประกอบอย่างชัดเจน ให้คนทั่วไปอ่านแล้วเข้าใจง่ายๆ ว่าข้อมูลนี้บอกเทรนด์หรือพฤติกรรมอะไร"
                                    response = model.generate_content(prompt)
                                    st.success("**การวิเคราะห์จาก AI:**")
                                    st.write(response.text)
                                except Exception as e:
                                    st.error(f"เกิดข้อผิดพลาดในการเรียก AI: {e}")
                    
                    st.write("**ตารางรายละเอียดทุกหมวดหมู่:**")
                    detail_df = pd.DataFrame({"จำนวน (Count)": value_counts.values, "สัดส่วน (Percentage)": probabilities.values}, index=value_counts.index)
                    st.dataframe(detail_df.style.format({"จำนวน (Count)": "{:,.0f}", "สัดส่วน (Percentage)": "{:,.2f}%"}), use_container_width=True)

            elif view_mode == "numeric_categorical":
                # ================= กรณีที่ 2: ตัวเลขที่มีความหลากหลายน้อย (แสดง 4 แท็บ) =================
                value_counts = df[selected_col].value_counts()
                probabilities = df[selected_col].value_counts(normalize=True) * 100
                
                tab1, tab2, tab3, tab4 = st.tabs(["📊 กราฟแท่ง (Bar)", "🍩 กราฟวงกลม (Pie)", "📈 กราฟการกระจายตัว (Hist)", "🧠 สรุปข้อมูลเชิงลึก (AI Insights)"])
                
                with tab1:
                    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
                    x_labels = value_counts.index.astype(str)
                    sns.barplot(x=x_labels, y=value_counts.values, ax=ax, palette="viridis")
                    ax.set_title(f"จำนวนข้อมูลแต่ละกลุ่มตัวเลข: {readable_col}", fontsize=14, fontweight='bold')
                    ax.set_ylabel("จำนวน (ครั้ง)", fontsize=12)
                    ax.set_xlabel(readable_col, fontsize=12)
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    plt.xticks(rotation=45, ha='right')
                    for i, v in enumerate(value_counts.values):
                        ax.text(i, v + (max(value_counts.values) * 0.02), f"{v:,}", ha='center', va='bottom', fontsize=10)
                    st.pyplot(fig)
                    
                with tab2:
                    fig_pie, ax_pie = plt.subplots(figsize=(9, 9), dpi=200)
                    if len(value_counts) > 10:
                        top_10 = value_counts[:10]
                        others = pd.Series([value_counts[10:].sum()], index=['อื่นๆ (Others)'])
                        pie_data = pd.concat([top_10, others])
                    else:
                        pie_data = value_counts

                    custom_labels = [f"ค่าตัวเลข: {str(idx)} \n({val:,} รายการ)" for idx, val in pie_data.items()]

                    wedges, texts, autotexts = ax_pie.pie(
                        pie_data.values, labels=custom_labels, autopct='%1.1f%%', 
                        startangle=90, colors=sns.color_palette("pastel", len(pie_data)),
                        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
                    )
                    for text in texts: text.set_fontsize(10)
                    for autotext in autotexts:
                        autotext.set_fontsize(10)
                        autotext.set_fontweight('bold')
                    ax_pie.set_title(f"สัดส่วนเปอร์เซ็นต์แบบละเอียด: {readable_col}", fontsize=14, fontweight='bold')
                    st.pyplot(fig_pie)
                
                with tab3:
                    fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)}, figsize=(10, 6), dpi=200)
                    sns.boxplot(x=df[selected_col], ax=ax_box, color="lightblue")
                    ax_box.set_title(f"การกระจายตัวและค่าผิดปกติ (Outliers) ของ: {readable_col}", fontsize=14, fontweight='bold')
                    ax_box.set_xlabel('')
                    
                    sns.histplot(df[selected_col], kde=True, ax=ax_hist, color="blue", bins='auto')
                    ax_hist.set_ylabel("จำนวนครั้งที่พบ (Frequency)", fontsize=12)
                    ax_hist.set_xlabel(f"ค่าของ {readable_col}", fontsize=12)
                    ax_hist.grid(axis='y', linestyle='--', alpha=0.5)
                    
                    max_height = max([p.get_height() for p in ax_hist.patches]) if ax_hist.patches else 1
                    for p in ax_hist.patches:
                        height = p.get_height()
                        if height > 0:
                            ax_hist.text(p.get_x() + p.get_width() / 2., height + (max_height * 0.02), f"{int(height):,}", ha='center', va='bottom', fontsize=9)
                    st.pyplot(fig)
                    
                with tab4:
                    mean_val = df[selected_col].mean()
                    median_val = df[selected_col].median()
                    
                    st.markdown(f"### 💡 วิเคราะห์เจาะลึกตัวเลขแบบกลุ่ม: {readable_col}")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("ความหลากหลาย", f"{len(value_counts)} รูปแบบ")
                    col_b.metric("กลุ่มที่พบมากสุด", f"ค่า: {value_counts.index[0]}", f"{value_counts.values[0]:,} รายการ")
                    col_c.metric("ค่าเฉลี่ย (Mean)", f"{mean_val:,.2f}")
                    col_d.metric("ค่ากึ่งกลาง (Median)", f"{median_val:,.2f}")
                    
                    if st.button("✨ ให้ AI ช่วยวิเคราะห์ข้อมูลนี้", key="btn_both"):
                        if api_key == "": st.warning("⚠️ กรุณาใส่ API Key ที่แถบด้านซ้ายมือก่อนครับ")
                        else:
                            with st.spinner("🤖 AI กำลังคิดและประมวลผล..."):
                                try:
                                    genai.configure(api_key=api_key)
                                    model = genai.GenerativeModel('gemini-pro')
                                    prompt = f"สวมบทบาทเป็นนักวิเคราะห์ข้อมูลมืออาชีพ ช่วยอธิบายข้อมูลคอลัมน์ '{readable_col}' ซึ่งเป็นตัวเลขที่มีทั้งหมด {len(value_counts)} รูปแบบ. ค่าเฉลี่ยภาพรวมคือ {mean_val:.2f}, และค่าตัวเลขที่คนตอบมากที่สุดคือ '{value_counts.index[0]}' มีจำนวน {value_counts.values[0]} รายการ (คิดเป็น {probabilities.values[0]:.2f}%). ช่วยสรุปเป็น Bullet 3-4 ข้อ ระบุค่าตัวเลขประกอบอย่างชัดเจน ให้อ่านง่ายๆ ว่าตัวเลขพวกนี้บอกพฤติกรรมหรือเทรนด์อะไรได้บ้าง"
                                    response = model.generate_content(prompt)
                                    st.success("**การวิเคราะห์จาก AI:**")
                                    st.write(response.text)
                                except Exception as e:
                                    st.error(f"เกิดข้อผิดพลาดในการเรียก AI: {e}")
                    
                    st.write("**ตารางรายละเอียดทุกหมวดหมู่:**")
                    detail_df = pd.DataFrame({"จำนวน (Count)": value_counts.values, "สัดส่วน (Percentage)": probabilities.values}, index=value_counts.index)
                    st.dataframe(detail_df.style.format({"จำนวน (Count)": "{:,.0f}", "สัดส่วน (Percentage)": "{:,.2f}%"}), use_container_width=True)

            elif view_mode == "numeric":
                # ================= กรณีที่ 3: ข้อมูลแบบตัวเลขล้วนๆ (แสดง 2 แท็บ) =================
                tab1, tab2 = st.tabs(["📊 มุมมองกราฟ (Visualizations)", "🧠 สรุปข้อมูลเชิงลึก (AI Insights)"])
                
                with tab1:
                    fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)}, figsize=(10, 6), dpi=200)
                    sns.boxplot(x=df[selected_col], ax=ax_box, color="lightblue")
                    ax_box.set_title(f"การกระจายตัวและค่าผิดปกติ (Outliers) ของ: {readable_col}", fontsize=14, fontweight='bold')
                    ax_box.set_xlabel('')
                    
                    sns.histplot(df[selected_col], kde=True, ax=ax_hist, color="blue", bins='auto')
                    ax_hist.set_ylabel("จำนวนครั้งที่พบ (Frequency)", fontsize=12)
                    ax_hist.set_xlabel(f"ค่าของ {readable_col}", fontsize=12)
                    ax_hist.grid(axis='y', linestyle='--', alpha=0.5)
                    
                    max_height = max([p.get_height() for p in ax_hist.patches]) if ax_hist.patches else 1
                    for p in ax_hist.patches:
                        height = p.get_height()
                        if height > 0:
                            ax_hist.text(p.get_x() + p.get_width() / 2., height + (max_height * 0.02), f"{int(height):,}", ha='center', va='bottom', fontsize=9)
                    st.pyplot(fig)

                with tab2:
                    mean_val = df[selected_col].mean()
                    median_val = df[selected_col].median()
                    min_val = df[selected_col].min()
                    max_val = df[selected_col].max()
                    
                    st.markdown(f"### 💡 วิเคราะห์เจาะลึกตัวเลข: {readable_col}")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("ค่าเฉลี่ย (Mean)", f"{mean_val:,.2f}")
                    col_b.metric("ค่ากึ่งกลาง (Median)", f"{median_val:,.2f}")
                    col_c.metric("ค่าน้อยที่สุด (Min)", f"{min_val:,.2f}")
                    col_d.metric("ค่ามากที่สุด (Max)", f"{max_val:,.2f}")
                    
                    if st.button("✨ ให้ AI ช่วยวิเคราะห์ข้อมูลนี้", key="btn_num"):
                        if api_key == "": st.warning("⚠️ กรุณาใส่ API Key ที่แถบด้านซ้ายมือก่อนครับ")
                        else:
                            with st.spinner("🤖 AI กำลังคิดและประมวลผล..."):
                                try:
                                    genai.configure(api_key=api_key)
                                    model = genai.GenerativeModel('gemini-pro')
                                    prompt = f"สวมบทบาทเป็นนักวิเคราะห์ข้อมูลมืออาชีพ ช่วยอธิบายข้อมูลคอลัมน์ '{readable_col}' ให้คนทั่วไปฟังเป็นภาษาไทยแบบเข้าใจง่าย: ค่าเฉลี่ยคือ {mean_val:.2f}, ค่ากึ่งกลางคือ {median_val:.2f}, ต่ำสุดคือ {min_val:.2f}, และสูงสุดคือ {max_val:.2f}. ช่วยเขียนสรุปเป็น Bullet 3-4 ข้อ ระบุตัวเลขประกอบให้อ่านง่ายๆ ว่าตัวเลขพวกนี้บอกเทรนด์หรือพฤติกรรมอะไรได้บ้าง"
                                    response = model.generate_content(prompt)
                                    st.success("**การวิเคราะห์จาก AI:**")
                                    st.write(response.text)
                                except Exception as e:
                                    st.error(f"เกิดข้อผิดพลาดในการเรียก AI: {e}")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
else:
    st.info("กรุณาอัปโหลดไฟล์ข้อมูลที่แถบด้านซ้ายเพื่อเริ่มต้นการวิเคราะห์")