import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pypdf import PdfReader
from docx import Document
from model_hr import predict_candidate_fit, ROLE_CONFIG

# настройки страницы
st.set_page_config(
    page_title="Интеллектуальная HR-система",
    page_icon="📄",
    layout="wide"
)


# Проверяем наличие переменных в session_state.
# Если переменные отсутствуют, создаем их и присваиваем пустые значения.
# Это нужно для сохранения введенных данных
# (резюме, вакансии) между перезапусками приложения.

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "vacancy_text" not in st.session_state:
    st.session_state.vacancy_text = ""

if "vacancies_text" not in st.session_state:
    st.session_state.vacancies_text = ""

# DEMO-данные
DEMO_CASES = {
    "Junior Data Analyst — сильный кандидат": {
        "resume": """
Есть высшее образование в области прикладной информатики.
Опыт работы 1 год.
Навыки: python, sql, excel, pandas, data analysis, power bi, statistics.
Участвовал в аналитических проектах, строил отчеты и визуализации.
""",
        "vacancy": """
Требуется Junior Data Analyst.
Обязательные требования: python, sql, excel, pandas, data analysis.
Будет плюсом: power bi, tableau, visualization, git, statistics.
Опыт работы от 1 года.
Высшее образование обязательно.
"""
    },
    "Middle Data Analyst — частично подходящий кандидат": {
        "resume": """
Есть высшее образование.
Опыт работы 2 года.
Навыки: python, sql, excel, pandas, power bi.
Работал с аналитическими отчетами и дашбордами.
""",
        "vacancy": """
Требуется Middle Data Analyst.
Обязательные требования: python, sql, excel, pandas, data analysis.
Будет плюсом: power bi, tableau, statistics, visualization, product metrics.
Опыт работы от 2 лет.
Высшее образование обязательно.
"""
    },
    "ML Engineer —  слабый кандидат": {
        "resume": """
Есть высшее образование.
Опыт работы 1 год.
Навыки: excel, powerpoint, communication, reporting.
""",
        "vacancy": """
Требуется ML Engineer.
Обязательные требования: python, machine learning, pytorch, tensorflow.
Будет плюсом: deployment, pipelines, git, linux, airflow.
Опыт работы от 3 лет.
Высшее образование обязательно.
"""
    },
    "Backend Developer — сильный кандидат": {
        "resume": """
Есть высшее образование по информатике.
Опыт работы 3 года.
Навыки: python, api, sql, backend, docker, git, linux.
Разрабатывал backend-сервисы и интеграции.
""",
        "vacancy": """
Требуется Backend Developer.
Обязательные требования: python, api, sql, backend.
Будет плюсом: docker, git, linux, etl, integration.
Опыт работы от 2 лет.
Высшее образование будет плюсом.
"""
    },
}
# Вспомогательные функции
def read_uploaded_file(uploaded_file):
    # Функция читает загруженный файл и возвращает его текст
    # Поддерживаются форматы: .txt, .pdf, .docx
    if uploaded_file is None:

        return ""

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if file_name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()

        # DOCX
    if file_name.endswith(".docx"):
        doc = Document(uploaded_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()

    # Если формат не поддерживается
    return ""


def read_multiple_vacancy_files(uploaded_files):
    # Читает несколько файлов вакансий и возвращает список текстов
    texts = []

    for file in uploaded_files:
        text = read_uploaded_file(file)
        if text.strip():
            texts.append(text.strip())

    return texts

def calculate_match_percent(features, final_class):
    # Рассчитывает итоговый процент соответствия кандидата вакансии
    # Используются бизнес-правила и метрики модели
    score = 0
    # Вес обязательных навыков
    score += features["must_ratio"] * 35
    score += features["total_ratio"] * 20
    # Учет образования
    if features["education_match"] == 1:
        score += 10
    # Учет опыта работы
    if features["exp_diff"] >= 0:
        score += 15
    else:
        score -= 10
    # Учет текстового сходства
    score += float(features["tfidf_score"]) * 10
    score += float(features["bert_score"]) * 10
    # Ограничение до 100%
    score = min(round(score), 100)
    #Приведение к диапазону класса модели
    if final_class == 0:
        score = min(score, 54)
    elif final_class == 1:
        score = min(max(score, 55), 79)
    else:
        score = max(score, 80)

    return score


def make_recommendations(features):
    # Формирует рекомендации по улучшению резюме
    recommendations = []
    # Проверка обязательных навыков
    if features["missing_skills"]:
        recommendations.append(
            "Добавить или усилить обязательные навыки: " + ", ".join(features["missing_skills"])
        )
    # Проверка опыта
    if features["exp_diff"] < 0:
        recommendations.append("Добавить более релевантный опыт или подробнее описать выполненные задачи.")

    if features["education_match"] == 0:
        recommendations.append("Указать образование, курсы или релевантные сертификаты.")
    # Проверка TF-IDF
    if float(features["tfidf_score"]) < 0.45:
        recommendations.append("Сделать формулировки резюме ближе к тексту вакансии.")
    # Проверка BERT
    if float(features["bert_score"]) < 0.55:
        recommendations.append("Добавить проекты, технологии и реальные кейсы, связанные с вакансией.")
    # Если всё хорошо
    if not recommendations:
        recommendations.append("Профиль выглядит сильным, серьёзных улучшений не требуется.")

    return recommendations


def explain_rejection(features):
    # Определяет причины, по которым кандидат не подходит
    reasons = []

    if features["must_ratio"] < 0.4:
        reasons.append("слишком низкое совпадение обязательных навыков")

    if features["exp_diff"] < 0:
        reasons.append("опыт не соответствует требованиям вакансии")

    if features["education_match"] == 0:
        reasons.append("не выполнено требование по образованию")

    if not reasons:
        reasons.append("кандидат не прошёл жёсткий фильтр")

    return reasons

def show_status_card(label, match_percent):
    # Отображает итоговый статус кандидата с цветовой индикацией
    if label == "Не подходит":
        bg = "#fdeaea"
        border = "#e74c3c"
        text = "#8e2b23"
    elif label == "Частично подходит":
        bg = "#fff4df"
        border = "#f39c12"
        text = "#8a5a00"
    else:
        bg = "#eaf8ee"
        border = "#27ae60"
        text = "#1e6b3b"

    st.markdown(
        f"""
        <div style="
            background-color:{bg};
            border-left:8px solid {border};
            padding:18px;
            border-radius:12px;
            margin-bottom:18px;
        ">
            <h3 style="margin:0; color:{text};">Итог: {label}</h3>
            <p style="margin:8px 0 0 0; font-size:18px; color:{text};">
                Общий уровень соответствия: <b>{match_percent}%</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Блок объяснения результата оценки кандидата.
# Логика:
# - Если кандидат НЕ подходит (final_class == 0),
#   выводятся причины отказа (например: нет навыков, опыта или образования).
# - Если кандидат прошёл отбор,
#   показываются сильные стороны и зоны для улучшения:
#   • что подходит (навыки, опыт, образование, смысловое сходство)
#   • чего не хватает (опыт, обязательные навыки)
# Каждый пункт выводится в виде цветной карточки: причины отказа, сильные стороны и зоны для улучшения.

def show_reason_cards(features, final_class):
    # Показывает причины результата (плюсы и минусы кандидата)
    st.markdown("### Почему система дала такой результат")

    bad_bg, bad_border = "#fdeaea", "#f1b5ae"
    warn_bg, warn_border = "#fff4df", "#f2d29b"
    good_bg, good_border = "#eaf8ee", "#b8e0c7"

    if final_class == 0:
        reasons = explain_rejection(features)

        for reason in reasons:
            st.markdown(
                f"""
                <div style="
                    background:{bad_bg};
                    border:1px solid {bad_border};
                    padding:12px;
                    border-radius:10px;
                    margin-bottom:10px;
                    color:#222222;
                ">
                    ❌ {reason}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        good_points = []

        if features["must_ratio"] >= 0.7:
            good_points.append("хорошее совпадение обязательных навыков")

        if features["exp_diff"] >= 0:
            good_points.append("опыт соответствует требованиям вакансии")

        if features["education_match"] == 1:
            good_points.append("образование соответствует требованиям")

        if float(features["bert_score"]) >= 0.6:
            good_points.append("тексты резюме и вакансии близки по смыслу")

        for point in good_points:
            st.markdown(
                f"""
                <div style="
                    background:{good_bg};
                    border:1px solid {good_border};
                    padding:12px;
                    border-radius:10px;
                    margin-bottom:10px;
                    color:#222222;
                ">
                    ✅ {point}
                </div>
                """,
                unsafe_allow_html=True
            )

        if features["exp_diff"] < 0:
            st.markdown(
                f"""
                <div style="
                    background:{warn_bg};
                    border:1px solid {warn_border};
                    padding:12px;
                    border-radius:10px;
                    margin-bottom:10px;
                    color:#222222;
                ">
                    ⚠️ Опыт не соответствует требованиям вакансии (не хватает {abs(features["exp_diff"])} г.)
                </div>
                """,
                unsafe_allow_html=True
            )

        if features["missing_skills"]:
            st.markdown(
                f"""
                <div style="
                    background:{warn_bg};
                    border:1px solid {warn_border};
                    padding:12px;
                    border-radius:10px;
                    margin-bottom:10px;
                    color:#222222;
                ">
                    ⚠️ Не хватает обязательных навыков: {", ".join(features["missing_skills"])}
                </div>
                """,
                unsafe_allow_html=True
            )


def show_pie_chart(match_percent):
    # Строит круговую диаграмму общего соответствия
    fig, ax = plt.subplots(figsize=(4, 4))
    values = [match_percent, 100 - match_percent]
    labels = ["Совпадение", "Остальное"]
    ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=90)
    ax.set_title("Общее соответствие")
    st.pyplot(fig)


def show_line_chart(features):
# Строит график основных метрик (навыки и сходство)
    chart_df = pd.DataFrame({
        "Показатель": ["Must skills", "Total skills", "TF-IDF", "BERT"],
        "Значение": [
            float(features["must_ratio"]),
            float(features["total_ratio"]),
            float(features["tfidf_score"]),
            float(features["bert_score"])
        ]
    })
    st.line_chart(chart_df.set_index("Показатель"))

# Функция показывает, сколько навыков совпало:
# отдельно по обязательным навыкам
# и отдельно по дополнительным навыкам
def show_skill_match_summary(features):
    # общее количество обязательных навыков вакансии
    must_total = len(features["vacancy_must_skills"])

    # сколько обязательных навыков реально совпало
    must_matched = len(features["matched_skills"])

    # общее количество дополнительных навыков вакансии
    optional_total = len(features["vacancy_optional_skills"])

    # сколько дополнительных навыков совпало
    optional_matched = len(
        set(features["vacancy_optional_skills"]) & set(features["resume_skills"])
    )

    # выводим результат на экран
    st.markdown("### Совпадение навыков")
    st.write(f"**Обязательные навыки:** совпало {must_matched} из {must_total}")
    st.write(f"**Дополнительные навыки:** совпало {optional_matched} из {optional_total}")


# Функция показывает детальные progress bar по каждому важному критерию оценки кандидата
def show_detailed_progress(features):
    st.markdown("### Детальная оценка")

    # шкала по обязательным навыкам
    st.write("Обязательные навыки")
    st.progress(float(features["must_ratio"]))

    # шкала по общему совпадению навыков
    st.write("Общее совпадение навыков")
    st.progress(float(features["total_ratio"]))

    # шкала по текстовому сходству TF-IDF
    st.write("Текстовое сходство")
    st.progress(float(features["tfidf_score"]))

    # шкала по смысловому сходству BERT
    st.write("Смысловое сходство")
    st.progress(float(features["bert_score"]))

    # шкала по образованию:
    # если образование подходит → 1
    # если не подходит → 0
    st.write("Образование")
    st.progress(1.0 if features["education_match"] == 1 else 0.0)

    # отдельная шкала по опыту:
    # если опыт полностью подходит - 1.0, если не хватает 1 года - 0.5, если не хватает 2 лет и больше - 0.0
    if features["exp_diff"] >= 0:
        exp_score = 1.0
    elif features["exp_diff"] == -1:
        exp_score = 0.5
    else:
        exp_score = 0.0

    # вывод шкалы по опыту
    st.write("Соответствие опыта")
    st.progress(exp_score)

# Определяет наиболее подходящую роль кандидата на основе совпадения навыков
def recommend_role_by_resume(resume_text):
    text_lower = resume_text.lower()
    role_scores = {}

    for role_name, cfg in ROLE_CONFIG.items():
        must_skills = cfg["must_have_skills"]
        matched = 0

        for skill in must_skills:
            if skill.lower() in text_lower:
                matched += 1

        role_scores[role_name] = matched / len(must_skills)

    best_role = max(role_scores, key=role_scores.get)
    best_score = round(role_scores[best_role], 2)

    return best_role, best_score

# Формирует текстовый отчет по результатам анализа
def build_mini_report(result, match_percent):
    feature_values = result["features"]

    report = f"""
ИНТЕЛЛЕКТУАЛЬНАЯ HR-СИСТЕМА ДЛЯ IT ВАКАНСИЙ

Итоговый статус: {result["label"]}
Общий процент соответствия: {match_percent}%

Основные метрики:
- Совпадение обязательных навыков: {round(feature_values["must_ratio"], 2)}
- Совпадение всех навыков: {round(feature_values["total_ratio"], 2)}
- Разница опыта: {feature_values["exp_diff"]}
- Совпадение образования: {feature_values["education_match"]}
- TF-IDF similarity: {round(float(feature_values["tfidf_score"]), 2)}
- BERT similarity: {round(float(feature_values["bert_score"]), 2)}

Совпавшие обязательные навыки:
{", ".join(feature_values["matched_skills"]) if feature_values["matched_skills"] else "Нет"}

Недостающие обязательные навыки:
{", ".join(feature_values["missing_skills"]) if feature_values["missing_skills"] else "Нет"}

Рекомендации:
"""

    for recommendation in make_recommendations(feature_values):
        report += f"\n- {recommendation}"

    return report.strip()

# Подставляет демонстрационные данные в поля
def apply_demo_case(demo_name):
    st.session_state.resume_text = DEMO_CASES[demo_name]["resume"]
    st.session_state.vacancy_text = DEMO_CASES[demo_name]["vacancy"]

# Очищает все введенные пользователем данные
def clear_inputs():
    st.session_state.resume_text = ""
    st.session_state.vacancy_text = ""
    st.session_state.vacancies_text = ""

# Боковая панель: выбор режима работы приложения
st.sidebar.header("Режим работы")
mode = st.sidebar.radio(
    "Выберите режим",
    ["Быстрая проверка", "Подробный анализ", "Сравнение вакансий"]
)

# Demo-сценарии для быстрого тестирования системы
st.sidebar.markdown("---")
st.sidebar.subheader("Demo-сценарии")

# Кнопки для подстановки готовых примеров резюме и вакансий
for demo_name in DEMO_CASES:
    if st.sidebar.button(demo_name):
        apply_demo_case(demo_name) # загружаем данные
        st.rerun() # обновляем интерфейс
# Кнопка очистки всех полей
if st.sidebar.button("Очистить поля"):
    clear_inputs()
    st.rerun()

# Информация о модели
st.sidebar.markdown("---")
st.sidebar.write("Основная модель:")
st.sidebar.write("Logistic Regression")

# основной интерфейс
st.title("Интеллектуальная HR-система для IT вакансий")
# Краткое описание системы
st.markdown(
    "Гибридная система анализа резюме и вакансий на основе "
    "бизнес-правил, TF-IDF, BERT и машинного обучения."
)

st.subheader("Загрузка или ввод данных")
# Загрузка файла резюме
resume_file = st.file_uploader(
    "Загрузить резюме (.txt, .pdf или .docx)",
    type=["txt", "pdf", "docx"],
    key="resume_file"
)
# Если файл загружен — читаем его
if resume_file is not None:
    uploaded_resume_text = read_uploaded_file(resume_file)
    if uploaded_resume_text.strip():
        st.session_state.resume_text = uploaded_resume_text

# Поле для ручного ввода резюме
resume_text = st.text_area(
    "Текст резюме",
    key="resume_text",
    height=220,
    placeholder="Вставьте сюда текст резюме..."
)
# Определяем наиболее подходящую роль по резюме
best_role, best_role_score = recommend_role_by_resume(resume_text) if resume_text.strip() else ("—", 0)
# Вывод результаты
st.markdown("### Рекомендуемая роль кандидата")
st.info(f"Наиболее вероятная роль: **{best_role}** (уровень совпадения по навыкам: **{best_role_score}**)")

vacancy_text = ""
vacancies_text = ""

# Загрузка одной вакансии
if mode != "Сравнение вакансий":
    vacancy_file = st.file_uploader(
        "Загрузить вакансию (.txt, .pdf или .docx)",
        type=["txt", "pdf", "docx"],
        key="vacancy_file"
    )
    # Поле ввода вакансии
    if vacancy_file is not None:
        uploaded_vacancy_text = read_uploaded_file(vacancy_file)
        if uploaded_vacancy_text.strip():
            st.session_state.vacancy_text = uploaded_vacancy_text

# Загрузка нескольких вакансий
    vacancy_text = st.text_area(
        "Текст вакансии",
        key="vacancy_text",
        height=220,
        placeholder="Вставьте сюда текст вакансии..."
    )
else: #Загрузка нескольких вакансий
    vacancy_files = st.file_uploader(
        "Загрузить несколько вакансий (.txt, .pdf или .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="vacancy_files"
    )

    if vacancy_files:
        vacancy_texts_from_files = read_multiple_vacancy_files(vacancy_files)
        if vacancy_texts_from_files:
            st.session_state.vacancies_text = "\n===\n".join(vacancy_texts_from_files)

    vacancies_text = st.text_area(
        "Несколько вакансий (разделяйте вакансии строкой ===)",
        key="vacancies_text",
        height=300,
        placeholder="Вставьте несколько вакансий, разделяя их строкой ==="
    )

analyze_button = st.button("Проанализировать")


# одна вакансия
if analyze_button and mode != "Сравнение вакансий":
    # Проверка заполненности данных
    if not resume_text.strip():
        st.warning("Нужно заполнить текст резюме.")
    elif not vacancy_text.strip():
        st.warning("Нужно заполнить текст вакансии.")
    else: # Запуск модели
        result = predict_candidate_fit(resume_text, vacancy_text)
        features = result["features"]
        # Расчет процента соответствия
        match_percent = calculate_match_percent(features, result["final_class"])
        st.markdown("---")
        # Вывод итогового статуса
        show_status_card(result["label"], match_percent)

        # Основные метрики
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Процент соответствия %", f"{match_percent}%")
        with col2:
            st.metric("Обязательные навыки", round(features["must_ratio"], 2))
        with col3:
            st.metric("Общее совпадение", round(features["total_ratio"], 2))
        with col4:
            st.metric("Смысловое сходство", round(float(features["bert_score"]), 2))

        # режим анализа
        if mode == "Быстрая проверка":
            st.progress(match_percent / 100, text="Общий уровень соответствия")
            show_reason_cards(features, result["final_class"])

        if mode == "Подробный анализ":
            st.markdown("### HR-dashboard")

            d1, d2 = st.columns(2)

            with d1:
                show_pie_chart(match_percent)

            with d2:
                show_line_chart(features)

            # Подробный анализ по вкладкам
            tab1, tab2, tab3, tab4 = st.tabs(["Метрики", "Объяснение", "Навыки", "Рекомендации"])

            with tab1:
                st.write(f"**Совпадение обязательных навыков:** {round(features['must_ratio'], 2)}")
                st.write(f"**Совпадение всех навыков:** {round(features['total_ratio'], 2)}")
                st.write(f"**Разница опыта:** {features['exp_diff']}")
                st.write(f"**Совпадение образования:** {features['education_match']}")
                st.write(f"**Текстовое сходство:** {round(float(features['tfidf_score']), 2)}")
                st.write(f"**Смысловое сходство:** {round(float(features['bert_score']), 2)}")

                # показываем, сколько навыков совпало
                show_skill_match_summary(features)

                # показываем подробные progress bar по критериям
                show_detailed_progress(features)

            with tab2:
                show_reason_cards(features, result["final_class"])

            with tab3:
                st.markdown("**Совпавшие обязательные навыки**")
                st.write(", ".join(features["matched_skills"]) if features["matched_skills"] else "Нет")

                st.markdown("**Недостающие обязательные навыки**")
                st.write(", ".join(features["missing_skills"]) if features["missing_skills"] else "Нет")

                st.markdown("**Навыки из резюме**")
                st.write(", ".join(features["resume_skills"]) if features["resume_skills"] else "Не найдены")

                st.markdown("**Обязательные навыки вакансии**")
                st.write(
                    ", ".join(features["vacancy_must_skills"]) if features["vacancy_must_skills"] else "Не найдены")

                st.markdown("**Дополнительные навыки вакансии**")
                st.write(", ".join(features["vacancy_optional_skills"]) if features[
                    "vacancy_optional_skills"] else "Не найдены")

            with tab4:
                for recommendation in make_recommendations(features):
                    st.write(f"- {recommendation}")

        # Скачивание отчета
        report_text = build_mini_report(result, match_percent)
        st.download_button(
            label="Скачать мини-отчёт",
            data=report_text,
            file_name="mini_report.txt",
            mime="text/plain"
        )


# сравнение нескольких вакансий
if analyze_button and mode == "Сравнение вакансий":
    # Проверка данных
    if not resume_text.strip():
        st.warning("Нужно заполнить текст резюме.")
    elif not st.session_state.vacancies_text.strip():
        st.warning("Нужно вставить или загрузить хотя бы одну вакансию.")
    else:
        # Разделение вакансий
        vacancy_blocks = [block.strip() for block in st.session_state.vacancies_text.split("===") if block.strip()]

        if len(vacancy_blocks) == 0:
            st.warning("Не удалось выделить вакансии.")
        else:
            results_list = []
            # Анализ каждой вакансии
            for i, vacancy in enumerate(vacancy_blocks, start=1):
                result = predict_candidate_fit(resume_text, vacancy)
                features = result["features"]
                match_percent = calculate_match_percent(features, result["final_class"])
                results_list.append({
                    "№": i,
                    "Статус": result["label"],
                    "Процент соответствия": match_percent,
                    "Обязательные навыки": round(features["must_ratio"], 2),
                    "Общее совпадение": round(features["total_ratio"], 2),
                    "Смысловое сходство": round(float(features["bert_score"]), 2),
                    "Недостающие навыки": ", ".join(features["missing_skills"]) if features[
                        "missing_skills"] else "Нет",
                    "Соответствие образованию": features["education_match"],
                    "Разница опыта": features["exp_diff"]
                })
            # Формирование таблицы результатов

            results_df = pd.DataFrame(results_list).sort_values(by="Процент соответствия", ascending=False).reset_index(drop=True)
            results_df["№"] = range(1, len(results_df) + 1)
            st.markdown("---")
            st.subheader("Общий HR-dashboard")
            # Вывод полной таблицы
            st.dataframe(results_df, use_container_width=True)

            top3_df = results_df.head(3)
            st.markdown("### Top-3 вакансии")
            st.dataframe(top3_df, use_container_width=True)
            # Лучшая вакансия
            best_row = results_df.iloc[0]

            st.markdown("### Лучшая вакансия")
            show_status_card(best_row["Статус"], int(best_row["Процент соответствия"]))

            st.success(
                f"""
            Лучший вариант: вакансия №{best_row["№"]}
            Статус: {best_row["Статус"]}
            Процент соответствия: {best_row["Процент соответствия"]}%
            """
            )

            # Объяснение выбора лучшей вакансии

            reasons = []

            if best_row["Обязательные навыки"] >= 0.7:
                reasons.append("хорошее совпадение обязательных навыков")
            else:
                reasons.append("низкое совпадение обязательных навыков")

            if best_row["Общее совпадение"] >= 0.6:
                reasons.append("высокое общее совпадение навыков")

            if best_row["Смысловое сходство"] >= 0.6:
                reasons.append("высокое смысловое сходство")

            if best_row["Недостающие навыки"] != "Нет":
                reasons.append(f"не хватает обязательных навыков: {best_row['Недостающие навыки']}")

            if best_row["Соответствие образованию"] == 0:
                reasons.append("образование не соответствует требованиям вакансии")
            else:
                reasons.append("образование соответствует требованиям вакансии")

            if best_row["Разница опыта"] < 0:
                reasons.append(f"не хватает опыта: {abs(best_row['Разница опыта'])} г.")
            else:
                reasons.append("опыт соответствует требованиям вакансии")

            # Итоговый вывод
            st.markdown("**Почему выбрана эта вакансия:**")
            for reason in reasons:
                st.write(f"- {reason}")

            # Итоговый вывод
            st.markdown("### Итог по кандидату")

            improvement_points = []

            if best_row["Недостающие навыки"] != "Нет":
                improvement_points.append(f"добавить или усилить навыки: {best_row['Недостающие навыки']}")

            if best_row["Разница опыта"] < 0:
                improvement_points.append(f"не хватает опыта: {abs(best_row['Разница опыта'])} г.")

            if best_row["Соответствие образованию"] == 0:
                improvement_points.append("не соответствует требованиям по образованию")

            if best_row["Общее совпадение"] < 0.6:
                improvement_points.append("нужно усилить общее совпадение навыков с вакансией")

            if best_row["Смысловое сходство"] < 0.6:
                improvement_points.append("резюме стоит сделать ближе по формулировкам к вакансии")

            if best_row["Процент соответствия"] >= 80:
                st.success("Кандидат отлично подходит под лучшую из предложенных вакансий.")
            elif best_row["Процент соответствия"] >= 60:
                st.warning("Кандидат подходит частично. Ниже показано, чего не хватает.")
            else:
                st.error("Кандидат слабо подходит под предложенные вакансии.")

            if improvement_points:
                st.markdown("**Зоны для улучшения:**")
                for point in improvement_points:
                    st.write(f"- {point}")
            else:
                st.markdown("**Зоны для улучшения:**")
                st.write("- существенных недостатков не обнаружено")

            # Скачивание результатов
            csv_data = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Скачать сравнение вакансий (CSV)",
                data=csv_data,
                file_name="vacancy_comparison.csv",
                mime="text/csv"
            )