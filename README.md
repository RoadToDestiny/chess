# Chess Blunder Prediction (Прогнозирование шахматных ошибок)

Исследовательский проект по прогнозированию вероятности совершения ошибок шахматистами на следующем ходу на основе градиентного бустинга деревьев решений (**XGBoost**) и автоматической разметки движком **Stockfish**.

---

## 1. Архитектура репозитория

```text
Chess/
├── README.md                         # Документация проекта и инструкция по запуску
├── .gitignore                        # Исключения Git для данных, кэша и весов
├── .env.example                      # Переменные окружения (STOCKFISH_PATH, DATA_DIR)
├── requirements.txt                  # Зависимости Python
├── pyproject.toml                    # Конфигурация сборки пакета chess_error_prediction
├── docs/
│   └── journal_methodology_guide.md  # Единое научно-методологическое руководство (Этапы 1–6)
│
├── config/
│   └── config.yaml                   # Конфигурация параметров Stockfish, фильтрации, признаков и сплитов
│
├── data/                             # Директории данных исследования
│   ├── raw/                          # Исходный отфильтрованный корпус (filtered_games.pgn, 500 партий)
│   ├── interim/                      # Промежуточные результаты и эталонные выборки (labeled_sample.csv)
│   ├── processed/                    # Массив признаков (final_dataset.csv)
│   └── splits/                       # Групповые сплиты партий (train/val/test_game_ids.csv)
│
├── src/
│   └── chess_error_prediction/       # Основной Python-пакет
│       ├── schemas.py                # Спецификация контракта данных (Pydantic / Data Contract)
│       ├── engine/                   # Интеграция со Stockfish 19 UCI и анализ партий
│       │   ├── stockfish_client.py
│       │   ├── evaluation.py
│       │   ├── game_analysis.py
│       │   └── stage3_eval_verifier.py
│       ├── data/                     # Загрузка, чтение и фильтрация PGN
│       │   ├── download.py
│       │   ├── pgn_reader.py
│       │   ├── game_filter.py
│       │   └── validation.py
│       ├── labeling/                 # Centipawn loss, асимметричные маты и разметка ошибок
│       │   ├── error_labeler.py
│       │   └── stage5_target_verifier.py
│       ├── features/                 # Экстракторы 14 признаков до хода и аудит утечек
│       │   ├── extractors.py
│       │   ├── feature_spec.py
│       │   └── stage6_feature_verifier.py
│       ├── dataset/                  # Групповое разбиение (GroupKFold) и проверка утечек
│       │   ├── split.py
│       │   └── checks.py
│       ├── model/                    # Обучение XGBoost и калибровка (Этап 7)
│       └── experiments/              # Проведение абляционных экспериментов (Этап 8)
│
├── scripts/                          # Скрипты запуска пайплайнов и верификации
│   ├── download_and_filter_stage2.py
│   ├── verify_stage5_labeling.py
│   ├── verify_stage6_features.py
│   └── mwe.py                        # Minimal Working Example сквозного пайплайна
│
├── tests/                            # Набор автоматических тестов (Этапы 1–6, 31 тест)
│   ├── test_engine.py                # Проверка Stockfish UCI, детерминизма и таймаутов
│   ├── test_data.py                  # Чтение PGN, фильтрация и схема данных
│   ├── test_labeling.py              # Centipawn loss, пороги и асимметричная обработка матов
│   ├── test_schemas.py               # Валидация контракта признаков и защита от утечек
│   └── test_backend3_stages3_6.py    # Интеграционные тесты сплитов, GPU XGBoost и признаков
│
├── artifacts/                        # Артефакты моделей, весов и запусков
└── reports/                          # Таблицы, логи и графики исследования
    └── tables/
        ├── env_setup.log
        ├── stage2_filtering_funnel.csv
        ├── stage5_threshold_distribution.csv
        └── stage6_feature_audit.json
```

---

## 2. Итоги выполненных этапов (Этапы 1–6)

### Этап 1. Подготовка окружения и параметров Stockfish
- Зафиксирована точная версия движка для исследования: **Stockfish 19** (`bin/stockfish-windows-x86-64-universal.exe`).
- Зафиксирован строгий протокол детерминизма: `depth = 18`, `threads = 1`, `hash = 128 MB`, вызов `ucinewgame` перед каждой партией для очистки хэш-таблицы транспозиций.
- Аппаратное окружение: **AMD Ryzen 5 5600 6-Core Processor** (6 ядер, 12 потоков), **32 GB RAM**, **NVIDIA RTX 5060 Laptop GPU** (CUDA 13.4, `tree_method='hist'`), **Microsoft Windows 11 Pro 64-bit**, **Python 3.14.0**. Полный системный лог зафиксирован в `reports/tables/env_setup.log`.
- Зафиксированы точные версии зависимостей в `requirements.txt` (`xgboost==3.4.1`, `scikit-learn==1.9.1`, `pandas==3.1.1`, `numpy==2.5.5`, `python-chess==1.11.2`).

### Этап 2. Сбор, фильтрация данных и спецификация датасета
- Реализован потоковый пайплайн фильтрации PGN из базы Lichess (2026-09) без распаковки архивов в RAM.
- Сформирован чистый корпус из **500 партий** (35 550 прочитанных полуходов) со 100% соблюдением правил FIDE, начальным контролем времени $t_{\text{base}} \ge 600$ с (98% рапид, 2% классика), квалификационным диапазоном обоих игроков 1600–2800 Glicko-2 и длиной 10–120 ходов.
- Обеспечено сохранение меток времени `[%clk]`, исключение серверных оценок `[%eval]` и деперсонализация игроков через SHA-256. Артефакты: `data/raw/filtered_games.pgn`, `reports/tables/stage2_filtering_funnel.csv`.

### Этап 3. MVP пайплайна и разметка ошибок
- Реализована функция Centipawn Loss: $\Delta \text{Eval} = \text{Eval}_{\text{before}} - \text{Eval}_{\text{after}}$ с проекцией на перспективу активного игрока (POV).
- Внедрен **исчерпывающий протокол обработки матовых позиций**:
  - позиции с форсированным матом до хода изолируются из обучения (420 позиций, 1.18%);
  - зевок в форсированный мат соперника размечается как ошибка ($y=1$, `is_mate_error=True`, loss=2500 cp);
  - форсирование собственного мата — успешный ход ($y=0$, loss=0 cp);
  - упущенный соперником мат выбывает из сквозного датасета из-за фильтрации позиции до хода.
- Сформирован эталонный датасет `data/interim/labeled_sample.csv` (15 верифицированных позиций для качественной иллюстрации таксономии ошибок).

### Этап 4. Формирование признакового пространства и калибровка
- Разработан набор признаков строго до совершения хода (Zero Feature Leakage) в рамках трёхступенчатого сравнительного анализа:
  - **Вариант А (11 признаков позиции):** `eval_before`, `eval_diff_prev_move`, `has_prev_move`, `material_balance`, `piece_count`, `pawn_count`, `legal_checks_count`, `legal_captures_count`, `attacked_pieces_count` (включая пешки, без короля), `move_number` (полный ход), `game_phase`.
  - **Вариант Б (16 признаков, основной):** Вариант А + метаданные `player_rating`, `opponent_rating`, `rating_diff`, `time_control_base`, `time_control_inc`.
  - **Вариант В (18 признаков, расширенный):** Вариант Б + динамические часы `clock_remaining` и `time_spent_prev_move`.
- Сквозной пайплайн подтвержден в `scripts/mwe.py` с выводом полной строки датасета со всеми метаданными, оценками и целевой меткой.
- Реализована калибровка вероятностей (`calibration.py`): групповые OOF-прогнозы по партиям (`GroupKFold`), обучение калибраторов Платта и Isotonic, оценка на изолированном Test. Основной режим — `scale_pos_weight = 1.0`.

### Этап 5. Вычислительные затраты и баланс классов
- Сформирован полный отчет о вычислительных затратах (`reports/tables/benchmark.json`): 36 050 вызовов Stockfish, 7 062 с время движка (195.9 мс/вызов), 7 135 с общее время сборки, 342.5 МБ пик RAM.
- Рассчитано распределение классов при основном операционном пороге 200 cp: 31 040 обычных ходов (88.36%), 4 090 ошибок (11.64%), референсный `scale_pos_weight = 7.58` (рассчитан строго на Train).
- Сформирована таблица чувствительности порогов (150, 200, 300 cp) в `reports/tables/stage5_threshold_distribution.csv`.

### Этап 6. Подготовка выборок, аудит утечек и baseline LogisticRegression
- Проведено разделение по независимым партиям (`game_id`) в пропорции 70% / 15% / 15% с строгим разделением сырых полуходов и валидных строк (`reports/tables/dataset_row_counts.csv`):
  - Train: 350 партий, 25 253 полухода $\to$ **24 955 валидных строк** (`data/splits/train_game_ids.csv`).
  - Validation: 75 партий, 5 095 полуходов $\to$ **5 034 валидные строки** (`data/splits/val_game_ids.csv`).
  - Test: 75 партий, 5 202 полухода $\to$ **5 141 валидная строка** (`data/splits/test_game_ids.csv`).
- Доказано отсутствие пересечения множеств партий между сплитами ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
- Пройден аудит признаков на утечки (`reports/tables/stage6_feature_audit.json`).
- Реализован модуль baseline-модели `LogisticRegression` (`model/baseline_lr.py`): One-Hot для `game_phase`, `StandardScaler` строго на Train, исключение коллинеарного `rating_diff`, L2-регуляризация.

Полный пересчёт запускается командой `python scripts/build_dataset.py`. Скрипт формирует `data/processed/final_dataset.csv`, `reports/tables/benchmark.json`, `reports/tables/dataset_row_counts.csv` и обновляет таблицу порогов. Требуется Stockfish 19 в `bin/` или переменная `STOCKFISH_PATH`.

---

## 3. Установка и запуск

### 3.1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3.2. Запуск автоматических тестов
```bash
python -m pytest
```

### 3.3. Запуск минимального рабочего примера (MWE)
```bash
python scripts/mwe.py
```
