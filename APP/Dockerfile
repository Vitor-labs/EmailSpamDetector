FROM python:3.10

WORKDIR /APP

COPY Pipfile* ./

RUN pip install pipenv

RUN pipenv install --system --deploy --ignore-pipfile

COPY . .

CMD [ "python", "data_colector.py" ]
