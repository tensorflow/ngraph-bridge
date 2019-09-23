# How to use this Dockerfile (Dockerfile.clang-8.ubuntu18.04)
1. Create a dir where you want to build eg: docker_demo/build
2. Download ngraph-bridge repo : git clone https://github.com/tensorflow/ngraph-bridge.git
3. docker build -t <ngtf_build> -f Dockerfile.clang-8.ubuntu18.04 .
4. docker run -it -v ${PWD}:/<workspace> -w /<workspace> <ngtf_build>
5. Create a virtual env : virtualenv -p python3 <venv3>
6. Activate the virtual env : source venv3/bin/activate
7. pip install future
8. cd ngraph-bridge
9. ./configure_bazel.sh
10. BAZEL_LINKOPTS=-lc++ BAZEL_CXXOPTS=-stdlib=libc++ bazel build :hello_tf