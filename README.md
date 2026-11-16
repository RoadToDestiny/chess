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
│       ├── features/                 # Экстракторы признаков до хода и аудит утечек
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
- Аппаратное окружение: Windows 10 AMD64, AMD Ryzen 5 5600 (6 ядер/12 потоков), Python 3.14.7; XGBoost 3.4.1 с поддержкой CUDA 13.4. Объём RAM и версии части библиотек в исходном запуске не зафиксированы.

### Этап 2. Сбор, фильтрация данных и спецификация датасета
- Реализован потоковый пайплайн фильтрации PGN из базы Lichess Rapid (2026-09) без распаковки гигантских архивов в RAM.
- Сформирован чистый корпус из **500 партий** (35 550 прочитанных полуходов) с фильтром `t_base >= 600` и `t_base + 40 * t_increment < 1500`, рейтингом обоих игроков в диапазоне 1600–2800 и длиной 10–120 полных ходов.
- Обеспечено сохранение меток времени `[%clk]`, исключение серверных оценок `[%eval]` и деперсонализация игроков через SHA-256. Артефакты: `data/raw/filtered_games.pgn`, `reports/tables/stage2_filtering_funnel.csv`.

### Этап 3. MVP пайплайна и разметка ошибок
- Реализована функция Centipawn Loss: $\Delta \text{Eval} = \text{Eval}_{\text{before}} - \text{Eval}_{\text{after}}$ с проекцией на перспективу активного игрока (POV).
- Внедрен **асимметричный протокол матовых позиций**: позиции, где игрок допускает мат сопернику из нематовой ситуации, размечаются как грубая ошибка ($y=1, \text{loss}=2500$ cp); ходы, форсирующие собственный мат, не штрафуются ($y=0$); позиции, где мат уже стоял до хода, изолируются.
- Сформирован эталонный датасет `data/interim/labeled_sample.csv` (15 верифицированных позиций).

### Этап 4. Формирование признакового пространства
- Разработан набор признаков строго до совершения хода (Zero Feature Leakage), включая `move_number` и `has_prev_move`:
  - Оценочные: `eval_before`, `eval_diff_prev_move` (с инверсией знака для черных).
  - Материальные и досочные: `material_balance` (POV), `piece_count`, `pawn_count`.
  - Тактическая напряженность: `legal_checks_count`, `legal_captures_count`, `attacked_pieces_count`.
  - Метаданные: `move_number`, `game_phase`, `player_rating`, `opponent_rating`, `rating_diff`, `time_control_base`, `time_control_inc`.
- Сквозной пайплайн подтвержден в `scripts/mwe.py`.

### Этап 5. Вычислительные затраты и баланс классов
- Замерено реальное время вызова Stockfish: **195.9 мс/позиция**.
- Общее число вызовов движка, время этапов обработки и пик RAM в исходном запуске не зафиксированы; эти значения не восстанавливаются по среднему времени позиции.
- Сформирована таблица распределения порогов ошибки (150, 200, 300 cp) в `reports/tables/stage5_threshold_distribution.csv`.
- Разработан валидатор меток `src/chess_error_prediction/labeling/stage5_target_verifier.py`.

### Этап 6. Подготовка выборок и аудит утечек
- Проведено разделение по независимым партиям (`game_id`) в пропорции 70% / 15% / 15%:
  - Train: 350 партий (`data/splits/train_game_ids.csv`).
  - Validation: 75 партий (`data/splits/val_game_ids.csv`).
  - Test: 75 партий (`data/splits/test_game_ids.csv`).
- Доказано отсутствие пересечения множеств партий между сплитами ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
- Пройден аудит признаков на утечки (`reports/tables/stage6_feature_audit.json`).

Для эксперимента вклада признаков используются три варианта на одинаковых сплитах: A — только позиция, B — позиция плюс рейтинги и контроль времени, C — B плюс фактические часы. Для прогнозирования вероятностей основной вариант XGBoost использует `scale_pos_weight=1`; калибровка настраивается только по групповой OOF/отдельной calibration-выборке и проверяется на Test.

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
