FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm

EXPOSE 8080

ENV PORT 8080

CMD ["python", "app.py"]
