FROM python:3.9-slim

ARG BASIC_AUTH_USERNAME
ARG BASIC_AUTH_PASSWORD

ENV BASIC_AUTH_USERNAME=$BASIC_AUTH_USERNAME
ENV BASIC_AUTH_PASSWORD=$BASIC_AUTH_PASSWORD

COPY ./requirements.txt /usr/requirements.txt

WORKDIR /usr

RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./src /usr/src
COPY ./models /usr/models

ENTRYPOINT ["python"]

CMD ["src/app/main.py"]