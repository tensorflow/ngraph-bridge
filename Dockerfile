FROM tensorflow/tensorflow:devel-py3

RUN apt-get update && apt-get install -y \
  vim \
  virtualenv \
  git \
  wget \
  unzip \
  sudo \
  zlib1g-dev \
  bash-completion \
  cmake \
  libtinfo-dev \
  zip \
  golang-go \
  locate \
  curl \
  clang-format-3.9 \
  gcc-4.8 \
  g++-4.8 \
  openjdk-8-jdk

RUN ln -f -s /usr/bin/gcc-4.8 /usr/bin/gcc || true
RUN ln -f -s /usr/bin/g++-4.8 /usr/bin/g++ || true
RUN which gcc && gcc --version || true
RUN which c++ && c++ --version || true

RUN updatedb
RUN pip3 install --upgrade pip setuptools virtualenv==16.1.0
RUN pip3 install --upgrade pytest
RUN pip3 install yapf==0.26.0
ARG BAZEL_VERSION=0.24.1
RUN wget --no-verbose -c https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel_${BAZEL_VERSION}-linux-x86_64.deb
RUN dpkg -i bazel_${BAZEL_VERSION}-linux-x86_64.deb || true

ENTRYPOINT ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]
