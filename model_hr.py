import re
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sentence_transformers import SentenceTransformer

'''
Список ключевых навыков, которые система ищет в резюме и вакансии.

Используется для:
- извлечения навыков из текста
- сравнения кандидата с требованиями вакансии
- расчёта совпадения (обязательных и всех навыков)
'''

SKILLS = [
    "python", "sql", "excel", "power bi", "tableau",
    "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn",
    "machine learning", "deep learning", "nlp", "bert",
    "pytorch", "tensorflow", "statistics", "data analysis",
    "data visualization", "visualization", "dashboard",
    "git", "github", "linux", "spark", "hadoop", "etl",
    "airflow", "big data", "a/b testing", "hypothesis testing",
    "regression", "classification", "clustering", "time series",
    "feature engineering", "business intelligence", "analytics",
    "reporting", "data cleaning", "product metrics",
    "business processes", "requirements", "uml", "bpmn", "api",
    "documentation", "integration", "backend", "html", "css",
    "javascript", "react", "docker"
]

# Словарь синонимов навыков.
# Нужен для того, чтобы система понимала одинаковые навыки
# на русском и английском, а также частые разговорные варианты.

SKILL_ALIASES = {
    "python": [
        "python", "питон", "пайтон"
    ],

    "sql": [
        "sql", "sql server", "postgresql", "mysql"
    ],

    "excel": [
        "excel", "эксель", "microsoft excel"
    ],

    "power bi": [
        "power bi", "powerbi", "power-bi",
        "пауэр би", "павер би", "павер биай"
    ],

    "tableau": [
        "tableau", "табло"
    ],

    "pandas": [
        "pandas", "пандас"
    ],

    "numpy": [
        "numpy", "нумпай", "нампай"
    ],

    "scikit-learn": [
        "scikit-learn", "sklearn",
        "сайкит лерн", "склерн"
    ],

    "machine learning": [
        "machine learning", "ml",
        "машинное обучение"
    ],

    "deep learning": [
        "deep learning",
        "глубокое обучение"
    ],

    "nlp": [
        "nlp", "нлп", "нлпи",
        "natural language processing",
        "обработка текста",
        "обработка естественного языка"
    ],

    "bert": [
        "bert", "берт"
    ],

    "pytorch": [
        "pytorch", "torch", "пайторч"
    ],

    "tensorflow": [
        "tensorflow", "тензорфлоу"
    ],

    "statistics": [
        "statistics", "статистика"
    ],

    "data analysis": [
        "data analysis",
        "анализ данных",
        "аналитика данных"
    ],

    "data visualization": [
        "data visualization",
        "визуализация данных"
    ],

    "visualization": [
        "visualization",
        "визуализация"
    ],

    "dashboard": [
        "dashboard", "дашборд"
    ],

    "git": [
        "git", "гит"
    ],

    "github": [
        "github", "git hub", "гитхаб", "джитхаб"
    ],

    "linux": [
        "linux", "линукс"
    ],

    "etl": [
        "etl"
    ],

    "airflow": [
        "airflow"
    ],

    "big data": [
        "big data",
        "биг дата",
        "большие данные"
    ],

    "a/b testing": [
        "a/b testing", "ab testing", "a b testing",
        "a b тестирование", "а б тестирование",
        "а/б тестирование", "аб тестирование",
        "a/b тестирование"
    ],

    "hypothesis testing": [
        "hypothesis testing",
        "проверка гипотез"
    ],

    "regression": [
        "regression", "регрессия"
    ],

    "classification": [
        "classification", "классификация"
    ],

    "clustering": [
        "clustering", "кластеризация"
    ],

    "time series": [
        "time series", "временные ряды"
    ],

    "feature engineering": [
        "feature engineering",
        "инженерия признаков"
    ],

    "business intelligence": [
        "business intelligence", "bi"
    ],

    "analytics": [
        "analytics", "аналитика"
    ],

    "reporting": [
        "reporting", "отчеты", "отчетность"
    ],

    "data cleaning": [
        "data cleaning",
        "очистка данных"
    ],

    "product metrics": [
        "product metrics",
        "метрики продукта"
    ],

    "business processes": [
        "business processes",
        "бизнес процессы",
        "бизнес-процессы"
    ],

    "requirements": [
        "requirements", "требования"
    ],

    "uml": [
        "uml"
    ],

    "bpmn": [
        "bpmn"
    ],

    "api": [
        "api", "апи"
    ],

    "documentation": [
        "documentation", "документация"
    ],

    "integration": [
        "integration", "интеграция"
    ],

    "backend": [
        "backend", "бэкенд"
    ],

    "html": [
        "html"
    ],

    "css": [
        "css"
    ],

    "javascript": [
        "javascript", "js", "джаваскрипт"
    ],

    "react": [
        "react", "реакт"
    ],

    "docker": [
        "docker", "докер"
    ]
}

# Ключевые слова для определения упоминания образования в тексте.
# Используются как общий индикатор (есть ли вообще образование)
EDUCATION_WORDS = [
    # базовые формулировки
    "высшее образование", "бакалавр", "магистр", "магистратура",
    "бакалавриат", "высшее", "диплом", "окончил", "закончил",

    # учебные заведения
    "университет", "вуз", "институт", "академия",

    # английские варианты
    "bachelor", "master", "degree", "bs", "ms",
    "university"
]

#Фразы, указывающие, что высшее образование ОБЯЗАТЕЛЬНО.
#Если найдено — требование считается строгим (required)
EDUCATION_REQUIRED_KEYWORDS = [
    "обязательно высшее образование",
    "обязателен диплом",
    "обязательна высшее образование",
    "обязательное высшее образование",

    "требуется высшее образование",
    "требуется диплом",
    "требование высшее образование",
    "требование к образованию",

    "необходимо высшее образование",
    "необходима высшее образование",

    "нужно высшее образование",
    "нужно наличие высшего образования",
    "нужен диплом",
    "нужна высшее образование",

    "наличие высшего образования",

    "только с высшим",
    "только с высшим образованием",

    "высшее образование обязательно",
    "высшее образование требуется",
    "высшее образование необходимо"
]


#Фразы, указывающие, что образование желательно, но не обязательно.
#Используются для определения optional-требования.

EDUCATION_OPTIONAL_KEYWORDS = [
    "будет плюсом",
    "будет преимуществом",
    "как преимущество",
    "преимуществом будет",
    "преимуществом считается",
    "желательно",
    "желателен",
    "желательна",
    "желательно наличие",
    "будет хорошо",
    "хорошо если есть",
    "плюсом будет",
    "будет дополнительным плюсом",
    "приветствуется",
    "будет приветствоваться",
    "не обязательно, но желательно",
    "необязательно, но желательно",
    "не обязательно",
    "необязательно"
]
# Возможные уровни кандидатов и вакансий
LEVELS = ["junior", "middle", "senior"]

LEVEL_TO_NUM = {
    "junior": 1,
    "middle": 2,
    "senior": 3
}

# Для каждой роли задаем уровень, опыт, образование,
# обязательные и дополнительные навыки
ROLE_CONFIG = {
    "Junior Data Analyst": {
        "level": "junior",
        "experience_options": [0, 1],
        "education_options": [0, 1],
        "must_have_skills": ["python", "sql", "excel", "pandas", "data analysis"],
        "optional_skills": ["power bi", "tableau", "visualization", "git", "statistics"]
    },

    "Middle Data Analyst": {
        "level": "middle",
        "experience_options": [2, 3],
        "education_options": [0, 1],
        "must_have_skills": ["python", "sql", "excel", "pandas", "data analysis"],
        "optional_skills": ["power bi", "tableau", "statistics", "visualization", "product metrics"]
    },

    "Senior Data Analyst": {
        "level": "senior",
        "experience_options": [3, 4, 5],
        "education_options": [0, 1],
        "must_have_skills": ["python", "sql", "excel", "pandas", "data analysis"],
        "optional_skills": ["statistics", "product metrics", "visualization", "machine learning", "git"]
    },

    "BI Analyst": {
        "level": "middle",
        "experience_options": [1, 2, 3],
        "education_options": [0, 1],
        "must_have_skills": ["sql", "excel", "power bi", "tableau", "dashboard"],
        "optional_skills": ["python", "pandas", "data visualization", "analytics", "reporting"]
    },

    "Product Analyst": {
        "level": "middle",
        "experience_options": [1, 2, 3],
        "education_options": [0, 1],
        "must_have_skills": ["sql", "python", "data analysis", "a/b testing", "statistics"],
        "optional_skills": ["product metrics", "visualization", "dashboard", "hypothesis testing", "pandas"]
    },

    "Business Analyst": {
        "level": "middle",
        "experience_options": [1, 2, 3],
        "education_options": [0, 1],
        "must_have_skills": ["sql", "excel", "analytics", "business processes", "requirements"],
        "optional_skills": ["documentation", "visualization", "communication", "api", "reporting"]
    },

    "System Analyst": {
        "level": "middle",
        "experience_options": [1, 2, 3],
        "education_options": [0, 1],
        "must_have_skills": ["sql", "uml", "bpmn", "requirements", "api"],
        "optional_skills": ["documentation", "integration", "analytics", "business processes", "git"]
    },

    "Junior Data Scientist": {
        "level": "junior",
        "experience_options": [0, 1],
        "education_options": [0, 1],
        "must_have_skills": ["python", "pandas", "numpy", "machine learning"],
        "optional_skills": ["statistics", "scikit-learn", "data analysis", "git", "visualization"]
    },

    "Middle Data Scientist": {
        "level": "middle",
        "experience_options": [1, 2, 3],
        "education_options": [1],
        "must_have_skills": ["python", "pandas", "numpy", "machine learning", "statistics"],
        "optional_skills": ["deep learning", "pytorch", "scikit-learn", "feature engineering", "git"]
    },

    "Senior Data Scientist": {
        "level": "senior",
        "experience_options": [3, 4, 5],
        "education_options": [1],
        "must_have_skills": ["python", "machine learning", "statistics", "feature engineering", "data analysis"],
        "optional_skills": ["deep learning", "pytorch", "tensorflow", "git", "deployment"]
    },

    "ML Engineer": {
        "level": "middle",
        "experience_options": [2, 3, 4],
        "education_options": [1],
        "must_have_skills": ["python", "machine learning", "pytorch", "tensorflow"],
        "optional_skills": ["deployment", "pipelines", "git", "linux", "airflow"]
    },

    "NLP Engineer": {
        "level": "middle",
        "experience_options": [2, 3, 4],
        "education_options": [1],
        "must_have_skills": ["python", "nlp", "bert", "pytorch", "machine learning"],
        "optional_skills": ["deep learning", "transformers", "git", "tensorflow", "data analysis"]
    },

    "Backend Developer": {
        "level": "middle",
        "experience_options": [1, 2, 3],
        "education_options": [0, 1],
        "must_have_skills": ["python", "api", "sql", "backend"],
        "optional_skills": ["docker", "git", "linux", "etl", "integration"]
    },

    "Frontend Developer": {
        "level": "middle",
        "experience_options": [0, 1, 2],
        "education_options": [0, 1],
        "must_have_skills": ["html", "css", "javascript"],
        "optional_skills": ["react", "git", "visualization", "dashboard", "api"]
    },

    "Fullstack Developer": {
        "level": "middle",
        "experience_options": [1, 2, 3],
        "education_options": [0, 1],
        "must_have_skills": ["python", "javascript", "api", "sql"],
        "optional_skills": ["react", "docker", "git", "linux", "backend"]
    }
}

# Эта функция создает одну вакансию по выбранной роли
def generate_vacancy(role_name, vacancy_id):
    config = ROLE_CONFIG[role_name]

    level = config["level"]
    required_experience = random.choice(config["experience_options"])
    education_required = random.choice(config["education_options"])

    must_have_skills = config["must_have_skills"]

    optional_count = random.randint(2, min(4, len(config["optional_skills"])))
    optional_skills = random.sample(config["optional_skills"], optional_count)

    education_text = "Высшее образование обязательно." if education_required == 1 else "Высшее образование будет плюсом."
    experience_text = f"Опыт работы от {required_experience} лет." if required_experience > 0 else "Опыт работы не обязателен."

    vacancy_text = f"""
    Требуется {role_name}.
    Уровень позиции: {level}.
    Обязательные требования: {", ".join(must_have_skills)}.
    Будет плюсом: {", ".join(optional_skills)}.
    {experience_text}
    {education_text}
    """

    return {
        "vacancy_id": vacancy_id,
        "title": role_name,
        "level": level,
        "education_required": education_required,
        "required_experience": required_experience,
        "must_have_skills": must_have_skills,
        "optional_skills": optional_skills,
        "text": vacancy_text
    }

# На каждую роль делаем по 10 разных вакансий
vacancies_data = []
vacancy_id = 1

for role_name in ROLE_CONFIG.keys():
    for _ in range(10):
        vacancy = generate_vacancy(role_name, vacancy_id)
        vacancies_data.append(vacancy)
        vacancy_id += 1

vacancies_df = pd.DataFrame(vacancies_data)
vacancies_df.head(3)

# Проверяем общее количество вакансий
print("Всего вакансий:", len(vacancies_df))
# Смотрим распределение вакансий по ролям
print(vacancies_df["title"].value_counts())

# Эта функция определяет уровень кандидата, в зависимости от уровня роли и типа кандидата
def generate_candidate_level(role_level, candidate_type):

    # Если роль senior — не занижаем сильно уровень
    if role_level == "senior":
        if candidate_type == "strong":
            return "senior"
        elif candidate_type == "medium":
            return random.choice(["senior", "middle"])
        else:
            return random.choice(["middle", "junior"])

    # Для остальных ролей
    if candidate_type == "strong":
        return role_level

    if candidate_type == "medium":
        return role_level

    if candidate_type == "weak":
        if role_level == "middle":
            return "junior"
        return role_level

    return role_level

# Эта функция генерирует опыт кандидата, в зависимости от уровня роли и типа кандидата

def generate_candidate_experience(role_level, candidate_type):
    if role_level == "junior":
        if candidate_type == "strong":
            return random.choice([0, 1])
        elif candidate_type == "medium":
            return 0
        else:
            return 0

    if role_level == "middle":
        if candidate_type == "strong":
            return random.choice([2, 3])
        elif candidate_type == "medium":
            return random.choice([1, 2])
        else:
            return random.choice([0, 1])

    if role_level == "senior":
        if candidate_type == "strong":
            return random.choice([3, 4, 5])
        elif candidate_type == "medium":
            return random.choice([2, 3])
        else:
            return random.choice([0, 1])

    return 0

# Эта функция определяет, есть ли у кандидата образование
def generate_candidate_education(candidate_type):
    if candidate_type == "strong":
        return random.choice([1, 1, 1, 0])
    elif candidate_type == "medium":
        return random.choice([1, 0])
    else:
        return random.choice([0, 0, 1])

# генерирует навыки кандидата.
# strong - все обязательные навыки
# medium - все обязательные или не хватает 1–2
# weak -только часть обязательных навыков

def generate_candidate_skills(must_have_skills, optional_skills, candidate_type):

    if candidate_type == "strong":
        selected_must = must_have_skills.copy()

        optional_count = min(len(optional_skills), random.randint(2, 4))
        selected_optional = random.sample(optional_skills, optional_count) if optional_count > 0 else []

    elif candidate_type == "medium":
        if len(must_have_skills) <= 2:
            missing_count = 1
        else:
            missing_count = random.choice([0, 1, 2])

        must_count = max(1, len(must_have_skills) - missing_count)
        selected_must = random.sample(must_have_skills, must_count)

        optional_count = min(len(optional_skills), random.randint(1, 3))
        selected_optional = random.sample(optional_skills, optional_count) if optional_count > 0 else []

    else:
        must_count = max(1, int(len(must_have_skills) * 0.3))
        selected_must = random.sample(must_have_skills, must_count)

        optional_count = min(len(optional_skills), random.randint(0, 2))
        selected_optional = random.sample(optional_skills, optional_count) if optional_count > 0 else []

    all_candidate_skills = list(set(selected_must + selected_optional))
    return all_candidate_skills

# собирает текст резюме из параметров кандидата
def generate_resume_text(role_name, level, education, experience, skills):
    education_text = "Есть высшее образование." if education == 1 else "Высшее образование не указано."
    experience_text = f"Опыт работы {experience} лет." if experience > 0 else "Коммерческий опыт работы отсутствует."

    text = f"""
    Кандидат рассматривает позицию {role_name}.
    Уровень кандидата: {level}.
    {education_text}
    {experience_text}
    Навыки: {", ".join(skills)}.
    Имеет опыт работы с инструментами и технологиями из своего профессионального направления.
    """
    return text

# создаем одно резюме
def generate_resume(res_id, role_key, candidate_type):
    config = ROLE_CONFIG[role_key]

    role_level = config["level"]
    candidate_level = generate_candidate_level(role_level, candidate_type)
    candidate_experience = generate_candidate_experience(role_level, candidate_type)
    candidate_education = generate_candidate_education(candidate_type)

    candidate_skills = generate_candidate_skills(
        config["must_have_skills"],
        config["optional_skills"],
        candidate_type
    )

    resume_text = generate_resume_text(
        role_key,
        candidate_level,
        candidate_education,
        candidate_experience,
        candidate_skills
    )

    return {
        "resume_id": res_id,
        "candidate_name": f"Кандидат {res_id}",
        "target_role": role_key,
        "level": candidate_level,
        "education": candidate_education,
        "experience": candidate_experience,
        "skills": candidate_skills,
        "text": resume_text
    }

# Создаем 400 резюме.
# Увеличиваем выборку и немного повышаем долю сильных кандидатов,
# чтобы данные были более сбалансированными и модель обучалась стабильнее.

resumes_data = []
resume_id = 1

role_names = list(ROLE_CONFIG.keys())

for _ in range(400):
    role_key = random.choice(role_names)

    candidate_type = random.choices(
        ["strong", "medium", "weak"],
        weights=[0.45, 0.35, 0.20],
        k=1
    )[0]

    resume = generate_resume(resume_id, role_key, candidate_type)
    resumes_data.append(resume)

    resume_id += 1

resumes_df = pd.DataFrame(resumes_data)
resumes_df.head()

""" 
Добавляем текстовые признаки: очистку текста , TF-IDF similarity, BERT similarity
"""

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Очищаем тексты заранее
resumes_df["clean_text"] = resumes_df["text"].apply(clean_text)
vacancies_df["clean_text"] = vacancies_df["text"].apply(clean_text)

"""
Считаем TF-IDF один раз:
- для всех резюме
- для всех вакансий

Потом для каждой пары берем уже готовые векторы,
"""

# Собираем все тексты в один список
all_texts = pd.concat(
    [resumes_df["clean_text"], vacancies_df["clean_text"]],
    ignore_index=True
)

# Обучаем TF-IDF один раз на всех текстах
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

# Создаем словари с готовыми TF-IDF-векторами
resume_tfidf_vectors = {}
vacancy_tfidf_vectors = {}

# Первые строки матрицы относятся к резюме
for i, (_, row) in enumerate(resumes_df.iterrows()):
    resume_tfidf_vectors[row["resume_id"]] = tfidf_matrix[i]

# Следующие строки относятся к вакансиям
offset = len(resumes_df)

for j, (_, row) in enumerate(vacancies_df.iterrows()):
    vacancy_tfidf_vectors[row["vacancy_id"]] = tfidf_matrix[offset + j]

import os
import warnings
import logging
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    os.environ["HF_TOKEN"] = hf_token

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# загружаем BERT
bert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Считаем эмбеддинги один раз
resume_embeddings = {}
vacancy_embeddings = {}

for _, row in resumes_df.iterrows():
    resume_embeddings[row["resume_id"]] = bert_model.encode(row["clean_text"])

for _, row in vacancies_df.iterrows():
    vacancy_embeddings[row["vacancy_id"]] = bert_model.encode(row["clean_text"])

# Проверяем общее количество резюме
print("Всего резюме:", len(resumes_df))

# Смотрим распределение по ролям
print(resumes_df["target_role"].value_counts())

# Смотрим распределение по уровням
print(resumes_df["level"].value_counts())

"""Расчет признаков
Функция принимает: резюме кандидата и вакансию
И возвращает числовые признаки:
- доля совпадения обязательных навыков
- доля совпадения всех навыков
- разница опыта
- совпадение образования
- TF-IDF similarity
- BERT similarity
"""

def calculate_features(resume, vacancy):
    # Берем навыки кандидата
    resume_skills = set(resume["skills"])

    # Берем навыки вакансии
    must_skills = set(vacancy["must_have_skills"])
    optional_skills = set(vacancy["optional_skills"])

    # Считаем совпадения обязательных навыков
    must_match = len(must_skills & resume_skills)
    must_total = len(must_skills)

    # Считаем совпадения всех навыков
    all_skills = must_skills | optional_skills
    total_match = len(all_skills & resume_skills)
    total_skills = len(all_skills)

    # Переводим в доли
    must_ratio = must_match / must_total if must_total > 0 else 0
    total_ratio = total_match / total_skills if total_skills > 0 else 0

    # Разница опыта
    exp_diff = resume["experience"] - vacancy["required_experience"]

    # Проверка образования
    education_match = int(resume["education"] >= vacancy["education_required"])

    tfidf_score = cosine_similarity(
        resume_tfidf_vectors[resume["resume_id"]],
        vacancy_tfidf_vectors[vacancy["vacancy_id"]])[0][0]

    # BERT similarity считаем по уже готовым embedding
    emb_resume = resume_embeddings[resume["resume_id"]]
    emb_vacancy = vacancy_embeddings[vacancy["vacancy_id"]]
    bert_score = cosine_similarity([emb_resume], [emb_vacancy])[0][0]

    return {
        "must_ratio": must_ratio,
        "total_ratio": total_ratio,
        "exp_diff": exp_diff,
        "education_match": education_match,
        "tfidf_score": tfidf_score,
        "bert_score": bert_score
    }

"""
Расчет таргета 
Логика:
0 — кандидат не подходит
1 — подходит частично
2 — идеально подходит

Изначально жесткий фильтр,если кандидат не проходит базовые требования, система сразу считает его неподходящим.
"""

def hard_filter(features):
    # слишком низкое совпадение обязательных навыков
    if features["must_ratio"] < 0.4:
        return 0

    # если опыта не хватает на 2 года и более — сразу отказ
    if features["exp_diff"] <= -2:
        return 0

    # образование не соответствует требованиям вакансии
    if features["education_match"] == 0:
        return 0

    return 1


"""
Мягкая оценка для кандидатов, которые уже прошли фильтр
1 — частично подходит
2 — хорошо подходит
"""
def calculate_soft_target(features):
    score = 0

    # Обязательные навыки
    if features["must_ratio"] >= 0.9:
        score += 2
    elif features["must_ratio"] >= 0.7:
        score += 1

    # Все навыки
    if features["total_ratio"] >= 0.75:
        score += 2
    elif features["total_ratio"] >= 0.5:
        score += 1

    # Опыт
    if features["exp_diff"] >= 0:
        score += 2
    else:
        score -= 2

    # Образование
    if features["education_match"] == 1:
        score += 1

    # Текстовые признаки
    if features["tfidf_score"] >= 0.55:
        score += 1
    if features["bert_score"] >= 0.65:
        score += 1

    if score >= 6 and features["exp_diff"] >= 0:
        return 2

    return 1

"""
Делим отдельно:
- резюме
- вакансии

Это нужно, чтобы одни и те же резюме и вакансии
не попадали одновременно и в train, и в test.
"""

resumes_train, resumes_test = train_test_split(
    resumes_df,
    test_size=0.2,
    random_state=42
)

vacancies_train, vacancies_test = train_test_split(
    vacancies_df,
    test_size=0.2,
    random_state=42
)

print("Train resumes:", resumes_train.shape)
print("Test resumes:", resumes_test.shape)
print("Train vacancies:", vacancies_train.shape)
print("Test vacancies:", vacancies_test.shape)


"""
Создаем пары резюме и вакансий

Сохраняем:
- полный таргет для всей системы
- отметку, прошел ли кандидат жесткий фильтр
"""
def build_pairs(resumes_part, vacancies_part):
    pairs = []

    for _, resume in resumes_part.iterrows():
        for _, vacancy in vacancies_part.iterrows():

            features = calculate_features(resume, vacancy)

            passed_filter = hard_filter(features)

            if passed_filter == 0:
                final_target = 0
            else:
                final_target = calculate_soft_target(features)

            pair = {
                "resume_id": resume["resume_id"],
                "vacancy_id": vacancy["vacancy_id"],
                "rule_pass": passed_filter,
                **features,
                "target": final_target
            }

            pairs.append(pair)

    return pd.DataFrame(pairs)

"""
Train пары строятся только из train-резюме и train-вакансий.
Test пары строятся только из test-резюме и test-вакансий.
"""

train_pairs_df = build_pairs(resumes_train, vacancies_train)
test_pairs_df = build_pairs(resumes_test, vacancies_test)

"""
Для модели берем только тех кандидатов,
которые прошли жесткий фильтр.

Модель учится различать:
1 — частично подходит
2 — хорошо подходит
"""

train_model_df = train_pairs_df[train_pairs_df["rule_pass"] == 1].copy()
test_model_df = test_pairs_df[test_pairs_df["rule_pass"] == 1].copy()

print("Train model pairs:", train_model_df.shape)
print("Test model pairs:", test_model_df.shape)

print("\nTrain model target:")
print(train_model_df["target"].value_counts())
print(train_model_df["target"].value_counts(normalize=True))

print("\nTest model target:")
print(test_model_df["target"].value_counts())
print(test_model_df["target"].value_counts(normalize=True))

print("Train pairs:", train_pairs_df.shape)
print("Test pairs:", test_pairs_df.shape)

print("\nTrain target:")
print(train_pairs_df["target"].value_counts())
print(train_pairs_df["target"].value_counts(normalize=True))

print("\nTest target:")
print(test_pairs_df["target"].value_counts())
print(test_pairs_df["target"].value_counts(normalize=True))

#Подготавливаем данные только для ML-модели
X_train = train_model_df.drop(columns=["target", "resume_id", "vacancy_id", "rule_pass"])
y_train = train_model_df["target"]

X_test = test_model_df.drop(columns=["target", "resume_id", "vacancy_id", "rule_pass"])
y_test = test_model_df["target"]

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

"""
Обучаем 3 модели:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
"""

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Логистическая регрессия
log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    ))
])

# Случайный лес
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

# Градиентный бустинг
gb_model = GradientBoostingClassifier(
    random_state=42
)

# Обучаем модели на тренировочной выборке
log_reg.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

#предсказание на тестовой выборке
y_pred_log = log_reg.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)

#оценка моделей
print("Logistic Regression Accuracy:", round(accuracy_score(y_test, y_pred_log), 4))
print(classification_report(y_test, y_pred_log))

print("Random Forest Accuracy:", round(accuracy_score(y_test, y_pred_rf), 4))
print(classification_report(y_test, y_pred_rf))

print("Gradient Boosting Accuracy:", round(accuracy_score(y_test, y_pred_gb), 4))
print(classification_report(y_test, y_pred_gb))

# сравнение моделей
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred_log),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_gb)
    ]
})

print("\nСравнение моделей:")
print(results)

"""
Функции для предсказания
Назначение: принять сырой текст резюме и вакансии
  извлечь признаки (навыки, опыт, образование, similarity)
  применить бизнес-правила
  применить ML-модель
  вернуть результат + объяснение
"""

def extract_skills_from_text(text, skills_list=SKILLS):
    # извлекаем навыки из текста с учетом русских/английских синонимов
    text_clean = clean_text(text)
    found_skills = []

    for skill in skills_list:
        aliases = SKILL_ALIASES.get(skill, [skill])

        for alias in aliases:
            if clean_text(alias) in text_clean:
                found_skills.append(skill)
                break

    return sorted(list(set(found_skills)))

def extract_experience_years(text):
    # извлекаем опыт работы в годах
    # поддерживаются форматы:
    # 2 года
    # 1.5 года
    # 2 года 6 месяцев
    # 6 месяцев
    # полгода
    text_clean = clean_text(text)

    # если явно указано отсутствие опыта
    if "без опыта" in text_clean:
        return 0

    # формат: 2 года 6 месяцев
    match_combo = re.search(r"(\d+)\s*(год|года|лет)\s*(\d+)\s*(месяц|месяца|месяцев)", text_clean)
    if match_combo:
        years = int(match_combo.group(1))
        months = int(match_combo.group(3))
        return years + months / 12

    # формат: 1.5 года / 1,5 года
    match_float = re.search(r"(\d+[.,]\d+)\s*(год|года|лет)", text_clean)
    if match_float:
        return float(match_float.group(1).replace(",", "."))

    # формат: 6 месяцев
    match_months = re.search(r"(\d+)\s*(месяц|месяца|месяцев)", text_clean)
    if match_months:
        months = int(match_months.group(1))
        return months / 12

    # формат: полгода
    if "полгода" in text_clean:
        return 0.5

    # обычный формат: 2 года
    matches = re.findall(r"(\d+)\s*(год|года|лет)", text_clean)
    if matches:
        return max(float(m[0]) for m in matches)

    return 0


def has_higher_education(text):
    text_clean = clean_text(text)

    '''
    Функция определяет наличие высшего образования в тексте резюме.

    Логика работы:
    1. Сначала проверяются отрицательные и неполные формулировки
       (например: "нет образования", "неполное высшее", "студент").
       Если найдено — возвращается 0 (образования нет).

    2. Если отрицаний нет — проверяются положительные формулировки
       (например: "имею высшее образование", "бакалавр", "магистр", "окончил университет").
       Если найдено — возвращается 1 (образование есть).

    3. Если ничего не найдено — считается, что образование отсутствует (return 0).

    Важно:
    Приоритет всегда у отрицательных формулировок, чтобы избежать ошибок
    (например, строка "нет образования" не считалась как наличие образования).
    '''

    negative_patterns = [
        "нет высшего образования",
        "без высшего образования",
        "высшее образование отсутствует",
        "не имеет высшего образования",
        "не получал высшее образование",
        "неоконченное высшее",
        "незаконченное высшее",
        "нет образования",
        "образование отсутствует",
        "без образования",
        "не окончил институт",
        "не закончил институт",
        "неполное высшее",
        "не учился в институте",
        "не окончил вуз",
        "не закончил вуз",
        "не училась в вузе",
        "не училась в институте",
        "не окончила вуз",
        "не закончила вуз",
        "не учился в вузе",
        "студент",
        "в процессе обучения",
        "учусь в университете",
        "учусь в вузе",
        "высшее образование не указано",
        "образование не указано",
        "не указано высшее образование",
    ]

    # сначала проверяем отрицание
    for pattern in negative_patterns:
        if pattern in text_clean:
            return 0

        #  высшее
    positive_patterns = [
        "есть высшее образование",
        "имею высшее образование",
        "у меня высшее образование",
        "имеется высшее образование",
        "высшее образование имеется",
        "получил высшее образование",
        "окончил университет",
        "закончил университет",
        "окончила университет",
        "закончила университет",
        "окончил институт",
        "окончила институт",
        "закончил институт",
        "закончила институт",
        "высшее образование",
        "бакалавр",
        "магистр",
        "бакалавриат",
        "магистратура",
        "специалитет",
        "университет",
        "институт"
    ]

    for pattern in positive_patterns:
        if pattern in text_clean:
            return 1

    return 0

'''
Функция определяет, является ли высшее образование в вакансии обязательным,
желательным или вообще не упоминается.

Раньше проблема была в том, что анализировался весь текст вакансии целиком.
Из-за этого фраза "будет плюсом" из блока дополнительных навыков могла
ошибочно влиять на блок образования.

Теперь логика доработана:
1. Текст вакансии разбивается на отдельные части.
2. Берутся только те части, где есть слова про образование или диплом.
3. Сначала проверяется, является ли образование обязательным.
4. Потом проверяется, является ли оно только плюсом.
5. Если просто есть упоминание "высшее образование" без уточнений,
   оно считается обязательным.

Функция возвращает:
"required"  - образование обязательно
"optional"  - образование желательно / будет плюсом
"none"      - требований к образованию нет
'''
def get_education_requirement(vacancy_text):
    text_clean = clean_text(vacancy_text)

    # Разбиваем текст на предложения / строки
    parts = re.split(r"[.\n]+", text_clean)

    education_parts = [
        part.strip() for part in parts
        if "образован" in part or "диплом" in part
    ]

    if not education_parts:
        return "none"

    # Сначала проверяем обязательность
    for part in education_parts:
        for phrase in EDUCATION_REQUIRED_KEYWORDS:
            if clean_text(phrase) in part:
                return "required"

    # Потом проверяем, является ли образование только плюсом
    for part in education_parts:
        for phrase in EDUCATION_OPTIONAL_KEYWORDS:
            if clean_text(phrase) in part:
                return "optional"

    # Если просто сказано "высшее образование", считаем обязательным
    for part in education_parts:
        if "высшее образование" in part:
            return "required"

    return "none"


def extract_vacancy_requirements(vacancy_text):
    # парсим требования вакансии
    text_clean = clean_text(vacancy_text)

    must_skills = []
    optional_skills = []

    # обязательные навыки
    must_match = re.search(
        r"обязательные требования\s*(.*?)(будет плюсом|опыт работы|высшее образование|$)",
        text_clean
    )
    if must_match:
        must_part = must_match.group(1)
        for skill in SKILLS:
            if clean_text(skill) in must_part:
                must_skills.append(skill)

    # дополнительные навыки
    optional_match = re.search(
        r"будет плюсом\s*(.*?)(опыт работы|высшее образование|$)",
        text_clean
    )
    if optional_match:
        optional_part = optional_match.group(1)
        for skill in SKILLS:
            if clean_text(skill) in optional_part:
                optional_skills.append(skill)

    # если не нашли блок обязательных — берем все навыки
    if not must_skills:
        must_skills = extract_skills_from_text(vacancy_text)

    required_experience = extract_experience_years(vacancy_text)
    education_requirement = get_education_requirement(vacancy_text)

    if education_requirement == "required":
        education_required = 1
    else:
        education_required = 0

    return {
        "must_have_skills": sorted(list(set(must_skills))),
        "optional_skills": sorted(list(set(optional_skills))),
        "required_experience": required_experience,
        "education_required": education_required,
        "education_requirement": education_requirement,
        "text": vacancy_text
    }


def prepare_resume_from_text(resume_text):
    # преобразуем резюме в структуру
    return {
        "skills": extract_skills_from_text(resume_text),
        "experience": extract_experience_years(resume_text),
        "education": has_higher_education(resume_text),
        "text": resume_text
    }


def calculate_features_for_new_texts(resume_text, vacancy_text):
    # считаем признаки
    resume_data = prepare_resume_from_text(resume_text)
    vacancy_data = extract_vacancy_requirements(vacancy_text)

    resume_skills = set(resume_data["skills"])
    must_skills = set(vacancy_data["must_have_skills"])
    optional_skills = set(vacancy_data["optional_skills"])

    # совпадение обязательных навыков
    must_match = len(must_skills & resume_skills)
    must_total = len(must_skills)

    # совпадение всех навыков
    all_skills = must_skills | optional_skills
    total_match = len(all_skills & resume_skills)
    total_skills = len(all_skills)

    must_ratio = must_match / must_total if must_total > 0 else 0
    total_ratio = total_match / total_skills if total_skills > 0 else 0

    # опыт и образование
    exp_diff = resume_data["experience"] - vacancy_data["required_experience"]
    education_bonus = 0

    if vacancy_data["education_requirement"] == "required":
        education_match = int(resume_data["education"] == 1)
    elif vacancy_data["education_requirement"] == "optional":
        education_match = 1
        if resume_data["education"] == 1:
            education_bonus = 0.05
    else:
        education_match = 1


    # текстовые признаки
    resume_clean = clean_text(resume_text)
    vacancy_clean = clean_text(vacancy_text)

    tfidf_score = cosine_similarity(
        tfidf_vectorizer.transform([resume_clean]),
        tfidf_vectorizer.transform([vacancy_clean])
    )[0][0]

    bert_score = cosine_similarity(
        [bert_model.encode(resume_clean)],
        [bert_model.encode(vacancy_clean)]
    )[0][0]

    # explainability
    matched_skills = sorted(list(must_skills & resume_skills))
    missing_skills = sorted(list(must_skills - resume_skills))

    return {
        "must_ratio": must_ratio,
        "total_ratio": total_ratio,
        "exp_diff": exp_diff,
        "education_match": education_match,
        "education_bonus": education_bonus,
        "education_requirement": vacancy_data["education_requirement"],
        "resume_education": resume_data["education"],
        "tfidf_score": tfidf_score,
        "bert_score": bert_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "resume_skills": resume_data["skills"],
        "vacancy_must_skills": vacancy_data["must_have_skills"],
        "vacancy_optional_skills": vacancy_data["optional_skills"]
    }

def predict_candidate_fit(resume_text, vacancy_text):
    # считаем признаки
    features = calculate_features_for_new_texts(resume_text, vacancy_text)

    # жесткий фильтр
    if hard_filter(features) == 0:
        return {
            "final_class": 0,
            "label": "Не подходит",
            "features": features
        }

    #  формируем вход для модели
    model_features = pd.DataFrame([{
        "must_ratio": features["must_ratio"],
        "total_ratio": features["total_ratio"],
        "exp_diff": features["exp_diff"],
        "education_match": features["education_match"],
        "tfidf_score": features["tfidf_score"],
        "bert_score": features["bert_score"]
    }])

    model_features = model_features[X_train.columns]

    #  предсказание модели
    pred = int(log_reg.predict(model_features)[0])

    #  дополнительное бизнес-правило:
    # если образование в вакансии не обязательное, но является плюсом,
    # и у кандидата оно есть — повышаем оценку с 1 до 2
    if features["education_bonus"] > 0 and pred == 1:
        pred = 2
    # если кандидату не хватает опыта,
    # он не может получить итоговую оценку "Хорошо подходит"
    if features["exp_diff"] < 0 and pred == 2:
        pred = 1

    # интерпретация
    if pred == 2:
        label = "Хорошо подходит"
    else:
        label = "Частично подходит"

    return {
        "final_class": pred,
        "label": label,
        "features": features
    }




