FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.5.8-20190530-050006

ENV PYTHONPATH=$PYTHONPATH:/app
ENV TORCH_MODEL_ZOO=/app

RUN curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
COPY . .
RUN mv vggish_model.ckpt /distil/third_party/audioset/vggish_model.ckpt
RUN pip install -e .
