FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-artful-python36-v2019.2.18-20190228-054439

RUN pip3 install --process-dependency-links git+https://github.com/uncharted-distil/distil-primitives.git@a58ba5536a6cd3c1f562d81c71c18f272803209f#egg=DistilPrimitives