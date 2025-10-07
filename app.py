import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ----------------------------------------------
# 1. إعداد وتحميل البيانات
# ----------------------------------------------

# اسم الملف المرفق (يجب أن يكون الملف في نفس المجلد على GitHub)
FILE_PATH = "مؤشرات الاداء-Grid view (18).csv"

@st.cache_data
def load_data(path):
    # قراءة ملف CSV مع محاولة معالجة الترميز (Encoding) الشائع للغة العربية
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='cp1256')
        
    # تنظيف وتسمية الأعمدة لتسهيل التعامل معها
    df.rename(columns={
        'نسبة الانجاز الفعلية': 'Completion_Rate',
        'نسبة المدة المنقضية': 'Elapsed_Time_Rate',
        'نسبة الإنحراف الفعلية': 'Actual_Deviation_Rate',
        'قيمة العقد (ريال)': 'Contract_Value',
        'المهندس المشرف': 'Supervisor_Engineer',
        'التصنيف (انارة - طرق )': 'Category',
        'المقاول': 'Contractor',
        'المشروع': 'Project_Name',
        'عقد رقم': 'Contract_ID'
    }, inplace=True)
    
    # تحويل الأعمدة الرقمية الهامة إلى float
    # نحتاج لمعالجة أعمدة النسبة المئوية إذا كانت تحتوي على علامة %
    
    def clean_percentage(series):
        # نحاول تحويلها مباشرة، وفي حالة الخطأ نعتبرها رقمية (لأن بعض الأعمدة قد تكون نظيفة أصلاً)
        if series.dtype == 'object':
            return series.astype(str).str.replace('%', '', regex=False).str.replace(',', '').apply(pd.to_numeric, errors='coerce') / 100
        return series / 100 # إذا كانت رقمية أصلاً (أرقام صحيحة لنسبة مئوية) نحولها لنسبة عشرية
        
    # الأعمدة التي تمثل نسب مئوية (نحولها إلى قيمة عشرية، مثل 0.45 بدلاً من 45%)
    for col in ['Completion_Rate', 'Elapsed_Time_Rate']:
         if col in df.columns:
            # البيانات في ملف الـ CSV هي بالفعل قيم عشرية (مثل 49.07)، لكن بعضها قد يكون نصي بسبب الفواصل أو الرموز.
            # سنقوم بتحويلها مباشرة إلى أرقام ونتعامل معها كنسبة مئوية (49.07% = 0.4907)
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100

    # معالجة قيمة العقد (نحولها إلى أرقام، قد تحتوي على فواصل)
    if 'Contract_Value' in df.columns:
        df['Contract_Value'] = pd.to_numeric(df['Contract_Value'], errors='coerce')
        
    # حساب حالة الانحراف (زمنياً) بناءً على الفرق بين الإنجاز والمدة المنقضية
    # سنستخدم نسبة الإنحراف الفعلية الموفرة في الملف مباشرةً
    
    if 'Actual_Deviation_Rate' in df.columns:
        # تحويل عمود الانحراف إلى رقمي
        df['Actual_Deviation_Rate'] = pd.to_numeric(df['Actual_Deviation_Rate'], errors='coerce') / 100
        
        # تصنيف حالة المشروع
        def get_deviation_status(rate):
            if rate > 0.05:
                return 'متقدم عن الجدول'
            elif rate < -0.05:
                return 'متأخر عن الجدول'
            else:
                return 'في الموعد المحدد'
        
        df['Deviation_Status'] = df['Actual_Deviation_Rate'].apply(get_deviation_status)

    # إزالة الصفوف التي لا تحتوي على بيانات أساسية صالحة
    df.dropna(subset=['Completion_Rate', 'Elapsed_Time_Rate', 'Category', 'Contract_Value'], inplace=True)
    
    return df

df = load_data(FILE_PATH)

# ----------------------------------------------
# 2. تصميم لوحة المعلومات (Streamlit Dashboard)
# ----------------------------------------------

st.set_page_config(layout="wide", page_title="لوحة متابعة أداء عقود التشغيل والصيانة")
st.title("📊 لوحة متابعة أداء عقود التشغيل والصيانة")
st.markdown("---")

# ----------------- الفلاتر الجانبية -----------------
st.sidebar.header("خيارات الفلترة")

# فلتر المهندس المشرف
supervisor_options = df['Supervisor_Engineer'].unique().tolist()
selected_supervisor = st.sidebar.multiselect(
    "المهندس المشرف:",
    options=supervisor_options,
    default=supervisor_options
)

# فلتر التصنيف (إنارة/طرق)
category_options = df['Category'].unique().tolist()
selected_category = st.sidebar.multiselect(
    "تصنيف المشروع:",
    options=category_options,
    default=category_options
)

# تطبيق الفلاتر
df_filtered = df[
    (df['Supervisor_Engineer'].isin(selected_supervisor)) & 
    (df['Category'].isin(selected_category))
]

# ----------------- 3. عرض المؤشرات الرئيسية (KPIs) -----------------

col1, col2, col3, col4 = st.columns(4)

# KPI 1: متوسط نسبة الإنجاز
avg_completion = df_filtered['Completion_Rate'].mean() * 100
col1.metric("متوسط نسبة الإنجاز الفعلي", f"{avg_completion:.1f}%")

# KPI 2: متوسط نسبة المدة المنقضية
avg_elapsed = df_filtered['Elapsed_Time_Rate'].mean() * 100
col2.metric("متوسط نسبة المدة المنقضية", f"{avg_elapsed:.1f}%")

# KPI 3: إجمالي المؤشر المالي (قيمة العقود في الفلتر)
total_contract_value = df_filtered['Contract_Value'].sum() / 1000000 # تحويل إلى مليون ريال
col3.metric("إجمالي قيمة العقود (مليون ريال)", f"{total_contract_value:,.2f}M")

# KPI 4: إجمالي عدد المشاريع المتأخرة
late_projects = df_filtered[df_filtered['Deviation_Status'] == 'متأخر عن الجدول'].shape[0]
col4.metric("إجمالي المشاريع المتأخرة", late_projects)

st.markdown("---")

# ----------------- 4. الرسوم البيانية والتصنيفات -----------------

col5, col6 = st.columns(2)

# Chart 1: الأداء حسب المقاول (Contractor Performance)
if 'Actual_Deviation_Rate' in df_filtered.columns:
    performance_by_contractor = df_filtered.groupby('Contractor')['Actual_Deviation_Rate'].mean().sort_values(ascending=True).reset_index()
    fig1 = px.bar(
        performance_by_contractor, 
        x='Contractor', 
        y='Actual_Deviation_Rate', 
        color=np.where(performance_by_contractor['Actual_Deviation_Rate'] < 0, 'متأخر', 'متقدم/في الموعد'), # تمييز الانحراف السلبي
        color_discrete_map={'متأخر': 'red', 'متقدم/في الموعد': 'green'},
        title='متوسط معدل الانحراف الفعلي حسب المقاول',
        labels={'Actual_Deviation_Rate': 'متوسط الانحراف (أفضل للأعلى)', 'Contractor': 'المقاول', 'color': 'الحالة'},
        height=500
    )
    fig1.update_layout(xaxis={'categoryorder':'total ascending'})
    col5.plotly_chart(fig1, use_container_width=True)


# Chart 2: تصنيف حالة المشاريع حسب الانحراف الزمني
deviation_counts = df_filtered.groupby('Deviation_Status').size().reset_index(name='العدد')
# تحديد الألوان للحالة: متأخر=أحمر، متقدم=أخضر، في الموعد=أصفر
color_map = {'متأخر عن الجدول': 'red', 'متقدم عن الجدول': 'green', 'في الموعد المحدد': 'yellow'}
fig2 = px.pie(
    deviation_counts, 
    names='Deviation_Status', 
    values='العدد', 
    title='تصنيف حالة المشاريع حسب الانحراف الزمني',
    color='Deviation_Status',
    color_discrete_map=color_map,
    height=500
)
col6.plotly_chart(fig2, use_container_width=True)

# ----------------- 5. الجدول التفصيلي (سجل العقود) -----------------
st.subheader("سجل العقود التفصيلي")
st.dataframe(
    df_filtered,
    column_config={
        "Contract_ID": "رقم العقد",
        "Project_Name": "اسم المشروع",
        "Contractor": "المقاول",
        "Supervisor_Engineer": "المهندس المشرف",
        "Category": "التصنيف",
        "Completion_Rate": st.column_config.ProgressColumn(
            "نسبة الإنجاز الفعلي",
            format="%.1f%%",
            min_value=0,
            max_value=1
        ),
        "Elapsed_Time_Rate": st.column_config.ProgressColumn(
            "نسبة المدة المنقضية",
            format="%.1f%%",
            min_value=0,
            max_value=1
        ),
        "Actual_Deviation_Rate": st.column_config.NumberColumn("معدل الانحراف (زمنياً)", format="%.2f"),
        "Deviation_Status": "حالة الانحراف",
        "Contract_Value": st.column_config.NumberColumn("قيمة العقد (ريال)", format="%.0f")
    },
    hide_index=True,
    use_container_width=True
)
