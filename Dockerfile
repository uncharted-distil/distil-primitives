FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.4.4-20190509-004421

RUN pip3 install -e git+https://github.com/uncharted-distil/distil-primitives.git#egg=DistilPrimitives
