FROM apache/zeppelin:0.12.0

# 1. Spark 및 시스템 환경 변수
ENV SPARK_VERSION=4.1.1
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

# 2. PySpark 관련 환경 변수 
ENV PYSPARK_PYTHON=/opt/conda/envs/python_3_with_R/bin/python
ENV PYSPARK_DRIVER_PYTHON=/opt/conda/envs/python_3_with_R/bin/python

# $PYTHONPATH가 정의되지 않았을 때 발생하는 경고를 방지하기 위해 단독 선언
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.9-src.zip

USER root

# 3. Java 17 및 필수 패키지 설치
# box2d-utils -> libbox2d-dev 로 변경되었습니다.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    openjdk-17-jdk \
    krb5-user libpam-krb5 curl unzip wget libbox2d-dev vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# 이 옵션이 없으면 Spark 내부 객체 접근 시 InaccessibleObjectException으로 인터프리터가 죽습니다.
ENV SPARK_SUBMIT_OPTS="--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED --add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"

# 4. Java 환경 변수 설정
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# 5. Spark 다운로드 및 설치
RUN wget https://downloads.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -O /tmp/spark.tgz && \
    tar -zxvf /tmp/spark.tgz -C /opt/ && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm /tmp/spark.tgz

# 6. [추가] Zeppelin Spark 인터프리터 바이너리 동기화
# Spark 버전이 크게 바뀌었으므로 인터프리터를 재설치해주는 것이 안정적입니다.
RUN /opt/zeppelin/bin/install-interpreter.sh --name spark

# 7. 권한 설정
RUN chown -R 1000:1000 /opt/spark /opt/zeppelin

# 8. 실행 환경 복구 및 사용자 전환
WORKDIR /opt/zeppelin
USER 1000