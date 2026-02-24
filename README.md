# Flash-RL Quantization Setup

Этот репозиторий содержит Docker-окружение и скрипты для запуска PPO обучения с квантизацией через Flash-RL на математическом датасете размером 8k.

## Структура

- `Dockerfile.A100` - Docker образ на основе SafeLLM с установленным Flash-RL
- `build.sh` - Скрипт для сборки Docker образа
- `start.sh` - Скрипт для запуска Docker контейнера
- `setup_verl.sh` - Скрипт для установки verl и Flash-RL
- `run_ppo_math8k_quantized.sh` - Скрипт для запуска PPO обучения с квантизацией

## Установка и использование

### 1. Сборка Docker образа (опционально)

Если нужно пересобрать образ с Flash-RL:

```bash
cd /home/gorbov_gv/quantization
bash build.sh
```

**Примечание:** Скрипт `start.sh` использует существующий образ `safe_llm_img`, поэтому сборка не обязательна. Flash-RL можно установить внутри контейнера при первом запуске.

### 2. Запуск Docker контейнера

```bash
cd /home/gorbov_gv/quantization
bash start.sh [gpu_id] [container_postfix]
```

Например:
```bash
bash start.sh 0,1 my_experiment
```

**Важно:** Скрипт использует существующий образ `safe_llm_img`, поэтому не требует пересборки.

### 3. Установка зависимостей в контейнере

Внутри контейнера запустите скрипт установки:

```bash
bash setup_verl.sh
```

Этот скрипт установит:
- **verl** из репозитория `yaof20/verl` (ветка `flash-rl`)
- **Flash-RL** (`flash-llm-rl`)

### 4. Запуск обучения PPO с квантизацией

Внутри Docker контейнера или на хосте (если все зависимости установлены):

```bash
# С FP8 квантизацией (рекомендуется для больших моделей)
bash run_ppo_math8k_quantized.sh fp8 Qwen/Qwen2.5-1.5B-Instruct

# С INT8 квантизацией
bash run_ppo_math8k_quantized.sh int8 Qwen/Qwen2.5-1.5B-Instruct

# С дополнительными параметрами Hydra
bash run_ppo_math8k_quantized.sh fp8 Qwen/Qwen2.5-32B-Instruct trainer.total_epochs=1000
```

## Параметры скрипта обучения

- `QUANTIZATION_TYPE` (первый аргумент): `fp8` или `int8` (по умолчанию: `fp8`)
- `MODEL_NAME` (второй аргумент): путь к модели (по умолчанию: `Qwen/Qwen2.5-1.5B-Instruct`)
- Дополнительные аргументы: передаются напрямую в Hydra конфигурацию

## Переменные окружения Flash-RL

Скрипт автоматически устанавливает:
- `FLASHRL_CONFIG` - тип квантизации (fp8/int8)
- `FLASHRL_LOGGING_LEVEL` - уровень логирования (INFO)
- `VLLM_ATTENTION_BACKEND` - XFORMERS

Дополнительные переменные (опционально):
- `FLASHRL_LMHEAD_FP32=1` - принудительное использование bf16 для lm head
- `FLASHRL_LOGGING_LEVEL=DEBUG` - детальное логирование
- `FLASHRL_LOGGING_FILE=/path/to/log` - сохранение логов в файл

## Конфигурация датасета

Скрипт настроен на работу с датасетом размером 8k:
- `train_data_size=8000`
- `val_data_size=1000`

Убедитесь, что данные находятся в:
- `$HOME/data/verl-agent/text/train.parquet`
- `$HOME/data/verl-agent/text/test.parquet`

Или измените пути в скрипте `run_ppo_math8k_quantized.sh`.

## Важные замечания

1. **Квантизация рекомендуется** для больших моделей (14B+, предпочтительно 32B+) и длинных CoT генераций
2. **Не изменяются зависимости**: vllm, torch, flash-attention остаются без изменений
3. **Flash-RL автоматически патчит** vLLM при импорте, если установлена переменная `FLASHRL_CONFIG`
4. **Используется существующий образ**: `start.sh` использует `safe_llm_img`, поэтому не требуется пересборка
5. **Установка зависимостей**: перед запуском обучения необходимо выполнить `bash setup_verl.sh`
6. **verl устанавливается из**: `https://github.com/yaof20/verl` (ветка `flash-rl`)

## Ссылки

- [Flash-RL GitHub](https://github.com/yaof20/Flash-RL)
- [Flash-RL Blog](https://fengyao.notion.site/flash-rl)
