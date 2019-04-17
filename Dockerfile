FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-artful-python36-v2019.2.18-20190228-054439

RUN pip3 install -e git+https://github.com/uncharted-distil/distil-primitives.git@d2e557148a9c8d5ef55eba74024083fd6516e6f3#egg=DistilPrimitives --process-dependency-links
