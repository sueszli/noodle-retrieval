FROM --platform=linux/amd64 python:3.6.15-slim-bullseye

RUN apt-get update && apt-get install -y make build-essential
RUN pip install jsonnet --no-build-isolation  
RUN pip install --upgrade pip
RUN pip install allennlp==1.2.2

RUN pip install blingfire==0.1.7
RUN pip install PyYAML==5.4
RUN pip install transformers==3.5.1
RUN pip install --find-links https://download.pytorch.org/whl/torch_stable.html torch==1.6.0

RUN pip install overrides

# convenience
RUN apt-get install -y git
RUN pip install numpy pandas matplotlib seaborn

CMD ["tail", "-f", "/dev/null"]
