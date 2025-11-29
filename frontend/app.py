import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import io

BACKEND_URL = "http://backend:8000"

st.set_page_config(page_title="Анализ тональности — Москва", layout="wide")
st.title("🔍 Анализ тональности отзывов горожан")

uploaded_file = st.file_uploader("Загрузите CSV с колонкой 'text'", type="csv")

if uploaded_file and 'result_df' not in st.session_state:
    with st.spinner("Анализирую..."):
        res = requests.post(f"{BACKEND_URL}/predict", files={"file": uploaded_file})
        if res.status_code == 200:
            st.session_state.result_df = pd.read_csv(io.BytesIO(res.content))
            st.session_state.edited = False
        else:
            st.error(f"Ошибка: {res.text}")

if 'result_df' in st.session_state:
    df = st.session_state.result_df.copy()

    # Фильтрация
    if 'src' in df.columns:
        sources = st.multiselect("Источник", df['src'].unique())
        if sources: df = df[df['src'].isin(sources)]
    labels = st.multiselect("Тональность", [0,1,2], format_func={0:"Negative",1:"Neutral",2:"Positive"}.get)
    if labels: df = df[df['label'].isin(labels)]

    # Поиск
    search = st.text_input("Поиск по тексту")
    if search: df = df[df['text'].str.contains(search, case=False, na=False)]

    # Ручная корректировка
    st.subheader("✏️ Редактирование разметки")
    edited_df = st.data_editor(
        df,
        column_config={"label": st.column_config.SelectboxColumn("Тональность", options=[0,1,2])},
        num_rows="dynamic"
    )

    # Сохранение правок
    if not df.equals(edited_df):
        full_df = st.session_state.result_df
        full_df.update(edited_df)
        st.session_state.result_df = full_df
        st.session_state.edited = True
        st.success("✅ Изменения применены!")

    # Визуализация
    st.subheader("📊 Распределение")
    fig = px.histogram(edited_df, x='label', category_orders={"label":[0,1,2]},
                       color='label', color_discrete_map={0:'red',1:'gray',2:'green'})
    st.plotly_chart(fig, use_container_width=True)

    # Экспорт
    st.download_button("📥 Скачать исправленный CSV",
                      st.session_state.result_df.to_csv(index=False),
                      "sentiment_corrected.csv", "text/csv")

    # Оценка качества
    st.subheader("📏 Оценка качества (macro-F1)")
    gt_file = st.file_uploader("Загрузите экспертную разметку", type="csv")
    if gt_file:
        with st.spinner("Вычисляю..."):
            pred_csv = st.session_state.result_df.to_csv(index=False).encode()
            files = {'predictions_file': ('pred.csv', pred_csv, 'text/csv'),
                     'ground_truth_file': ('gt.csv', gt_file.getvalue(), 'text/csv')}
            res = requests.post(f"{BACKEND_URL}/evaluate", files=files)
            if res.ok:
                st.metric("Macro-F1", f"{res.json()['macro_f1']:.4f}")
