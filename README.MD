# 🇷🇺 Детекция аномалий в медицинских данных (Thyroid Disease)

## 📌 О проекте
Данный проект посвящен разработке системы детекции аномалий для табличных медицинских данных на примере датасета **Thyroid Disease**. 
Основная задача — выявление пациентов с отклонениями (гипотиреоз, гипертиреоз) в условиях сильного дисбаланса классов (аномалии составляют около **7.4%**).

**Ключевая особенность:** Система построена на принципах **Semi-Supervised Anomaly Detection**. Базовые модели обучаются исключительно на нормальных данных (класс `0`), что позволяет им выучить идеальное распределение "здорового" пациента, не искажая границы выбросами.

## 🏗 Архитектура пайплайна

1. **Feature Engineering (без утечек данных):**
   *   Применение `QuantileTransformer` к непрерывным признакам (TSH, TT4, T4U, FTI) для нормализации распределений.
   *   Создание клинических отношений (`TSH/TT4`, `TSH/FTI`) и логарифмических разностей.
   *   Расчет робастного Z-score на основе медианы и MAD (Median Absolute Deviation), что устойчиво к выбросам.
   *   Бинарные флаги (например, `is_tsh_extreme_high`) на основе квантилей тренировочной выборки.
   *   *Важно:* Все статистики (медианы, MAD, квантили) вычисляются строго внутри кросс-валидационных фолдов на `train`-части для предотвращения Data Leakage.

2. **Базовые модели (One-Class Classifiers):**
   Для получения Out-Of-Fold (OOF) скоров аномальности были обучены и протюнены 9 моделей (только на `y == 0`):
   *   **Deep Learning:** Deep SVDD (на PyTorch), Autoencoder.
   *   **Статистические:** COPOD, ECOD, HBOS.
   *   **Геометрические:** GMM (Gaussian Mixture Model), LOF, One-Class SVM, Isolation Forest.

3. **Ансамблирование (Ensemble):**
   *   Первоначально протестированы Rank Averaging, Rank Product и генерация синтетических аномалий (через перемешивание и гауссов шум).
   *   **Финальное решение:** Semi-Supervised Stacking. Скоры 4 лучших и наиболее разнообразных моделей (Deep SVDD, COPOD, GMM, LOF) были объединены с использованием весов, оптимизированных через Optuna (максимизация PR-AUC на реальных OOF-скорах).

## 📊 Результаты
*   **PR-AUC:** ~0.77
*   **F1-score:** ~0.72 (при оптимальном подобранном пороге)
*   *Для сравнения:* Лучшая одиночная модель (Deep SVDD) дает PR-AUC 0.72. Случайное угадывание — 0.074.

## 🛠 Стек технологий
*   Python 3.10+
*   PyTorch (для Deep SVDD и Autoencoder)
*   Scikit-learn, PyOD (для базовых алгоритмов)
*   Optuna (для гиперпараметрического тюнинга и подбора весов)
*   Pandas, NumPy, SciPy

---

# 🇬🇧 Anomaly Detection in Medical Data (Thyroid Disease)

## 📌 Overview
This project is dedicated to building an Anomaly Detection system for tabular medical data, using the **Thyroid Disease** dataset as a case study. 
The primary challenge is identifying patients with abnormalities (hypothyroidism, hyperthyroidism) under severe class imbalance (anomalies make up about **7.4%** of the data).

**Key Highlight:** The system is built upon **Semi-Supervised Anomaly Detection**. Base estimators are trained exclusively on normal data (class `0`), allowing them to learn the optimal distribution of a "healthy" patient without distortion from outliers.

## 🏗 Pipeline Architecture

1. **Feature Engineering (Leakage-Free):**
   *   Applied `QuantileTransformer` to continuous features (TSH, TT4, T4U, FTI) for distribution normalization.
   *   Created clinical ratios (`TSH/TT4`, `TSH/FTI`) and log-differences.
   *   Calculated Robust Z-scores using Median and MAD (Median Absolute Deviation) to handle outliers gracefully.
   *   Generated binary flags (e.g., `is_tsh_extreme_high`) based on training set quantiles.
   *   *Crucial:* All statistics (medians, MAD, quantiles) are computed strictly within cross-validation folds on the `train` set to prevent Data Leakage.

2. **Base Estimators (One-Class Classifiers):**
   To obtain Out-Of-Fold (OOF) anomaly scores, 9 models were trained and tuned (on `y == 0` only):
   *   **Deep Learning:** Deep SVDD (PyTorch), Autoencoder.
   *   **Statistical:** COPOD, ECOD, HBOS.
   *   **Geometrical:** GMM (Gaussian Mixture Model), LOF, One-Class SVM, Isolation Forest.

3. **Ensemble Strategy:**
   *   Initially, Rank Averaging, Rank Product, and Synthetic Anomaly Generation (via permutation and Gaussian noise) were tested.
   *   **Final Solution:** Semi-Supervised Stacking. The scores of the top 4 most diverse and performant models (Deep SVDD, COPOD, GMM, LOF) were combined using weights optimized via Optuna (maximizing PR-AUC on real OOF scores).

## 📊 Results
*   **PR-AUC:** ~0.77
*   **F1-score:** ~0.72 (at the optimal tuned threshold)
*   *For reference:* The best single model (Deep SVDD) yields a PR-AUC of 0.72. Random guessing yields 0.074.

## 🛠 Tech Stack
*   Python 3.10+
*   PyTorch (for Deep SVDD and Autoencoder)
*   Scikit-learn, PyOD (for base algorithms)
*   Optuna (for hyperparameter tuning and weight optimization)
*   Pandas, NumPy, SciPy
