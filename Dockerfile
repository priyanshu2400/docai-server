
FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./app/lung_disease_model.h5 /code/lung_disease_model.h5
COPY ./app/disease_prediction_model.pkl /code/app/disease_prediction_model.pkl

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD ["fastapi", "run", "app/main.py", "--port", "80"]

# ENV API_KEY=AIzaSyDglAXjJWGXJ3ggOSMv25jXfZP5VqdUK-U