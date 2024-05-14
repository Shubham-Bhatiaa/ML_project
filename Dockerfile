FROM python:3.8

WORKDIR /WORK

COPY . /WORK

RUN pip install streamlit
RUN pip install numpy
RUN pip install pandas
RUN pip install scikit-Learn

CMD ["streamlit","run","main.py"]