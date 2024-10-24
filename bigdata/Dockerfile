FROM python:3.9

RUN apt-get update && apt-get install -y default-jdk

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64
ENV PATH $JAVA_HOME/bin:$PATH

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]
