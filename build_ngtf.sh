#!/usr/bin/env bash
apt-get update && apt-get install -y \
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

ln -f -s /usr/bin/gcc-4.8 /usr/bin/gcc || true
ln -f -s /usr/bin/g++-4.8 /usr/bin/g++ || true
which gcc && gcc --version || true
which c++ && c++ --version || true

updatedb
pip3 install --upgrade pip setuptools virtualenv==16.1.0
pip3 install --upgrade pytest
BAZEL_VERSION=0.24.1
wget --no-verbose -c https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel_${BAZEL_VERSION}-linux-x86_64.deb
dpkg -i bazel_${BAZEL_VERSION}-linux-x86_64.deb || true
echo ngtf_base completed
