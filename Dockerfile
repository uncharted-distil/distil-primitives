FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.5.8-20190530-050006

ENV PYTHONPATH=$PYTHONPATH:/app

RUN pip install -e .
