# Этап 1: Установка зависимостей
FROM python:3.11-slim AS builder

WORKDIR /usr/src/app/bot

COPY requirements.txt /usr/src/app/bot
RUN pip install --no-cache-dir -r requirements.txt

# Этап 2: Запуск приложения
FROM python:3.11-slim

WORKDIR /usr/src/app/bot

# Копируем установленные зависимости из предыдущего этапа
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Копируем все остальные файлы приложения
COPY . .

# Запускаем ваше приложение
CMD ["python", "your_script.py"]