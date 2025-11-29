import streamlit as st
import pandas as pd
import plotly.express as px
import io
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Надёжная настройка сессии с retry и большим таймаутом
def create_robust_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Используем host.docker.internal для macOS/Windows
BACKEND_URL = "http://host.docker.internal:8000"
session = create_robust_session()

st.set_page_config(page_title="Анализ тональности — Москва", layout="wide")
st.title("🔍 Анализ тональности отзывов горожан")

uploaded_file = st.file_uploader("Загрузите CSV с колонкой 'text'", type="csv")

if uploaded_file and 'result_df' not in st.session_state:
    with st.spinner("Анализирую тексты... (может занять до 60–90 сек для большой модели)"):
        try:
            # Увеличенный таймаут для "холодного" запуска модели
            response = session.post(
                f"{BACKEND_URL}/predict",
                files={"file": uploaded_file},
                timeout=180
            )
            if response.status_code == 200:
                st.session_state.result_df = pd.read_csv(io.BytesIO(response.content))
                st.session_state.edited = False
                st.success("✅ Анализ завершён!")
            else:
                st.error(f"Ошибка бэкенда: {response.status_code} — {response.text}")
        except requests.exceptions.Timeout:
            st.error("⏳ Превышено время ожидания. Модель загружается — повторите попытку через минуту.")
        except requests.exceptions.ConnectionError as e:
            st.error(f"🔌 Не удалось подключиться к бэкенду: {e}")
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")

if 'result_df' in st.session_state:
    df = st.session_state.result_df.copy()

    # Фильтрация по источнику (если есть)
    if 'src' in df.columns:
        sources = st.multiselect("Фильтр по источнику", df['src'].unique())
        if sources:
            df = df[df['src'].isin(sources)]

    # Фильтрация по тональности
    labels = st.multiselect(
        "Фильтр по тональности",
        options=[0, 1, 2],
        format_func={0: "Negative", 1: "Neutral", 2: "Positive"}.get
    )
    if labels:
        df = df[df['label'].isin(labels)]

    # Поиск по тексту
    search = st.text_input("Поиск по тексту")
    if search:
        df = df[df['text'].str.contains(search, case=False, na=False)]

    # === РУЧНАЯ КОРРЕКТИРОВКА ===
    st.subheader("✏️ Редактирование разметки")
    edited_df = st.data_editor(
        df,
        column_config={
            "label": st.column_config.SelectboxColumn(
                "Тональность",
                options=[0, 1, 2],
                required=True,
            ),
        },
        num_rows="dynamic",
        key="editor"
    )

    # Применение изменений
    if not df.equals(edited_df):
        full_df = st.session_state.result_df
        for idx, row in edited_df.iterrows():
            orig_idx = df.index[df.index == idx][0]  # сохраняем исходный индекс
            full_df.at[orig_idx, 'label'] = row['label']
        st.session_state.result_df = full_df
        st.session_state.edited = True
        st.success("✅ Изменения сохранены!")

    # Визуализация
    st.subheader("📊 Распределение тональности")
    fig = px.histogram(
        edited_df,
        x='label',
        category_orders={"label": [0, 1, 2]},
        labels={"label": "Класс"},
        color='label',
        color_discrete_map={0: 'red', 1: 'gray', 2: 'green'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Экспорт
    st.download_button(
        "📥 Скачать исправленный CSV",
        st.session_state.result_df.to_csv(index=False),
        "sentiment_corrected.csv",
        "text/csv"
    )

    # Оценка качества
    st.subheader("📏 Оценка качества (macro-F1)")
    gt_file = st.file_uploader("Загрузите экспертно размеченную выборку", type="csv")
    if gt_file:
        with st.spinner("Вычисляю macro-F1..."):
            try:
                pred_csv = st.session_state.result_df.to_csv(index=False).encode()
                files = {
                    'predictions_file': ('pred.csv', pred_csv, 'text/csv'),
                    'ground_truth_file': ('gt.csv', gt_file.getvalue(), 'text/csv')
                }
                eval_res = session.post(f"{BACKEND_URL}/evaluate", files=files, timeout=30)
                if eval_res.status_code == 200:
                    macro_f1 = eval_res.json().get("macro_f1", 0)
                    st.metric("Macro-F1", f"{macro_f1:.4f}")
                else:
                    st.error(f"Ошибка оценки: {eval_res.text}")
            except Exception as e:
                st.error(f"Ошибка при оценке: {e}")
