# 🏦 Credit Card Fraud Detection

> Сравнительный анализ моделей машинного обучения для детекции мошеннических транзакций

---

## 🛠️ Стек технологий

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189FDD?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-4C72B0?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## 📋 О проекте

Проект посвящён решению задачи **бинарной классификации** для обнаружения мошеннических операций по кредитным картам. Основная сложность задачи — **сильный дисбаланс классов**: мошеннические транзакции составляют менее 0.2% от всех операций.

### 🎯 Цель

Построить и сравнить несколько моделей машинного обучения, подобрав оптимальные гиперпараметры и порог классификации, чтобы максимизировать качество обнаружения фрода при приемлемом уровне ложных срабатываний.

---

## 📊 Данные

Используется датасет **Credit Card Fraud Detection** (Kaggle).

| Параметр | Значение |
|---|---|
| Всего транзакций | 284 807 |
| Мошеннических (Class=1) | 492 (≈ 0.17%) |
| Легитимных (Class=0) | 284 315 (≈ 99.83%) |
| Признаки | 30 (V1–V28 — PCA-компоненты, `Time`, `Amount`) |

### 📁 Схема разбиения

Данные разделяются **стратифицированно** на три части:

```
┌─────────────┐
│   Train     │  60% — обучение моделей
├─────────────┤
│ Validation  │  20% — подбор гиперпараметров и порога
├─────────────┤
│   Test      │  20% — финальная оценка
└─────────────┘
```

> ⚠️ `StandardScaler` обучается **только на train**, чтобы избежать утечки данных.

---

## 🤖 Модели

Были обучены и сравнены 4 модели. Для каждой подбирался **оптимальный порог** по максимуму F1 на валидационной выборке.

| # | Модель | Ключевые гиперпараметры | Дисбаланс |
|---|--------|-------------------------|-----------|
| 1 | Logistic Regression | `solver='lbfgs'`, `max_iter=1000` | — |
| 2 | Decision Tree | `max_depth`, `min_samples_leaf` (grid) | `class_weight='balanced'` |
| 3 | Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf` | `class_weight='balanced'` |
| 4 | **XGBoost** | `max_depth`, `learning_rate`, `n_estimators` | `scale_pos_weight` |

---

## 🏆 Результаты на тестовой выборке

| Модель | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|--------|-----------|-----------|--------|-----|---------|--------|
| Logistic Regression | 0.1 | 0.7009 | 0.7653 | 0.7317 | 0.9605 | 0.7414 |
| Decision Tree | 0.9 | 0.2401 | 0.8061 | 0.3700 | 0.9122 | 0.5956 |
| Random Forest | 0.5 | 0.8020 | 0.8265 | 0.8141 | 0.9719 | 0.8543 |
| 🥇 **XGBoost** | **0.9** | **0.9091** | **0.8163** | **0.8602** | **0.9838** | **0.8729** |

### 🔑 Выводы

- **XGBoost** показал наилучший результат: **F1 = 0.8602**, **PR-AUC = 0.8729**, **ROC-AUC = 0.9838**.
- **Random Forest** — сильный «серебряный призёр» с F1 = 0.8141.
- **Logistic Regression** — достойный baseline, но уступает ансамблям деревьев.
- **Одиночное Decision Tree** сильно переобучается и демонстрирует низкую Precision.

---

## 📈 Визуализации

Для финальной модели (XGBoost) в блокноте `04_XGBoost.ipynb` построены:

- **Confusion Matrix** — с переворотом осей для наглядности (Positive сверху)
- **ROC-кривая** — с AUC
- **Precision-Recall кривая** — с baseline (доля положительного класса)
- **Топ-15 важных признаков** — по `feature_importances_`

Также для каждой модели (за искл. регрессии) строились **тепловые карты PR-AUC** по сетке гиперпараметров.

---

## 🚀 Запуск

```bash
# 1. Клонировать репозиторий
git clone https://github.com/yourusername/telco-customer-churn.git
cd telco-customer-churn

# 2. Создать и активировать виртуальное окружение (рекомендуется)
python -m venv .venv
# Для Windows:
.venv\Scripts\activate
# Для macOS/Linux:
source .venv/bin/activate

# 3. Установить зависимости
pip install -r requirements.txt
```

4. Скачать датасет `creditcard.csv` с [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) и положить в корень проекта.
5. Открыть Jupyter Notebook и запустить блокноты по порядку:
   - `01_logistic_regression.ipynb`
   - `02_decision_tree.ipynb`
   - `03_random_forest.ipynb`
   - `04_XGBoost.ipynb`

---

## 📂 Структура проекта

```
telco-customer-churn/
│
├── .venv/                       # Виртуальное окружение (не коммитится)
├── 01_logistic_regression.ipynb # Логистическая регрессия
├── 02_decision_tree.ipynb       # Дерево решений
├── 03_random_forest.ipynb       # Случайный лес
├── 04_XGBoost.ipynb             # XGBoost + финальные графики
├── creditcard.csv               # Исходные данные (не коммитится)
├── requirements.txt             # Зависимости
├── .gitignore                   # Игнорирование .venv, .csv и т.д.
└── README.md                    # Описание проекта
```

---

## 💡 Ключевые технические решения

| Решение | Обоснование |
|---------|-------------|
| **Стратифицированное разбиение** | Сохраняет долю фрода во всех выборках |
| **Scaler fit только на train** | Предотвращает data leakage |
| **`class_weight='balanced'`** | Компенсация дисбаланса для деревьев |
| **`scale_pos_weight`** | Аналог для XGBoost |
| **Подбор порога на val** | Оптимизация F1 вместо стандартного 0.5 |
| **PR-AUC как главная метрика** | ROC-AUC завышен при сильном дисбалансе |

---

## 📌 Основные метрики

- **Precision** — доля реальных фродов среди предсказанных (важно снижать ложные тревоги)
- **Recall** — доля найденных фродов (важно не пропустить мошенничество)
- **F1-score** — баланс между Precision и Recall
- **PR-AUC** — площадь под PR-кривой, устойчива к дисбалансу
- **ROC-AUC** — общая разделяющая способность модели

---

## 👤 Автор

**Ваше Имя**  
📧 your.email@example.com  
🔗 [GitHub](https://github.com/yourusername)

---

## 📄 Лицензия

Проект распространяется под лицензией **MIT**. Данные — под лицензией Kaggle (ODbL).

---

<div align="center">

⭐ **Если проект был полезен — поставьте звезду!** ⭐

</div>
