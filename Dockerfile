FROM python:3.10.13-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*


RUN pip install pandas
RUN pip install streamlit
RUN pip install seaborn
RUN pip install matplotlib
RUN pip install xlsxwriter
RUN pip install protobuf
RUN pip install scikit-learn
RUN pip install gower
RUN pip install scipy


WORKDIR /home/app


EXPOSE 8501


ENTRYPOINT ["streamlit", "run", "/home/app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.runOnSave=true", "--server.fileWatcherType=poll"]
