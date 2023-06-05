FROM python:3.10

WORKDIR /APP

COPY Pipfile* ./
COPY data_colector.py .
COPY classifier.py .
COPY entrypoint.sh .

RUN pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile

ENTRYPOINT ["./entrypoint.sh"]