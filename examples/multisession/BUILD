load("//:cxx_abi_option.bzl", "CXX_ABI")

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")
cc_binary(
    name = 'sample',
    srcs = [
        "sample.cpp",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "//:ngraph_bridge_lib"
    ],
    copts = [
        "-pthread", 
        "-std=c++11", 
        # "-D_FORTIFY_SOURCE=2",
        # "-Wformat",
        # "-Wformat-security",
        # "-Wformat",
        # "-fstack-protector-all",
        "-D NDEBUG",
        # "-I src",
        # "-I external/ngraph/src",
        # "-I logging",
    ] + CXX_ABI,
    linkstatic=False,
    visibility = ["//visibility:public"],
)