# ==============================================================================
#  Copyright 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
load("//:cxx_abi_option.bzl", "CXX_ABI")

cc_library(
    name = "ngraph_bridge_headers",
    hdrs = glob([
        "src/ngraph_api.h",
        "src/ngraph_assign_clusters.h",
        "src/ngraph_builder.h",
        "src/ngraph_backend_manager.h",
        "src/ngraph_capture_variables.h",
        "src/ngraph_cluster_manager.h",
        "src/ngraph_conversions.h",
        "src/ngraph_deassign_clusters.h",
        "src/ngraph_encapsulate_clusters.h",
        "src/ngraph_freshness_tracker.h",
        "src/ngraph_mark_for_clustering.h",
        "src/ngraph_rewrite_for_tracking.h",
        "src/ngraph_timer.h",
        "src/ngraph_utils.h",
        "src/ngraph_version_utils.h",
        "src/tf_deadness_analysis.h",
        "src/tf_graphcycles.h",
        "src/version.h",
        "src/grappler/ngraph_optimizer.h",
        "src/grappler/ngraph_add_identityn.h",
        "logging/ngraph_log.h",
        "logging/tf_graph_writer.h",
    ]),
    copts = [
        "-I external/ngraph_bridge/src",
        "-I external/ngraph_bridge/logging",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = 'ngraph_bridge_lib',
    srcs = [
        "src/ngraph_api.cc",
        "src/ngraph_assign_clusters.cc",
        "src/ngraph_assign_clusters.h",
        "src/ngraph_builder.cc",
        "src/ngraph_backend_manager.cc",
        "src/ngraph_capture_variables.cc",
        "src/ngraph_cluster_manager.cc",
        "src/ngraph_deassign_clusters.cc",
        "src/ngraph_encapsulate_clusters.cc",
        "src/ngraph_encapsulate_op.cc",
        "src/ngraph_freshness_tracker.cc",
        "src/ngraph_mark_for_clustering.cc",
        "src/ngraph_rewrite_for_tracking.cc",
        "src/ngraph_tracked_variable.cc",
        "src/ngraph_utils.cc",
        "src/tf_deadness_analysis.cc",
        "src/tf_graphcycles.cc",
        "src/grappler/ngraph_optimizer.cc",
        "src/grappler/ngraph_add_identityn.cc",
        "logging/ngraph_log.cc",
        "logging/tf_graph_writer.cc",
    ],
    deps = [
        ":ngraph_bridge_headers",
        # "@org_tensorflow//tensorflow/core:framework",
        # "@org_tensorflow//tensorflow/core:lib",
        # "@org_tensorflow//tensorflow/core:protos_all_cc",
        # "@org_tensorflow//tensorflow/core:tensorflow",

        # "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",

        "@ngraph//:ngraph_headers",
        "@ngraph//:ngraph_core",
        "@ngraph//:ngraph_version",
    ],
    copts = [
        "-pthread", 
        "-std=c++11", 
        "-D_FORTIFY_SOURCE=2",
        "-Wformat",
        "-Wformat-security",
        "-Wformat",
        "-fstack-protector-all",
        "-D NDEBUG",
        '-D SHARED_LIB_PREFIX=\\"lib\\"',
        '-D SHARED_LIB_SUFFIX=\\".so\\"',
        '-D NGRAPH_TF_USE_GRAPPLER_OPTIMIZER',
        "-I src",
        "-I logging",
        "-I external/ngraph/src",
    ] + CXX_ABI,
    linkopts = [
        "-Wl,-z,noexecstack",
        "-Wl,-z,relro",
        "-Wl,-z,now",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = 'ngraph_bridge_version',
    srcs = [
        "src/version.h",
        "src/version.cc",
    ],
    deps = [
        "@ngraph//:ngraph_headers",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-D_FORTIFY_SOURCE=2",
        "-Wformat",
        "-Wformat-security",
        "-Wformat",
        "-fstack-protector-all",
        "-D NDEBUG",
        '-D SHARED_LIB_PREFIX=\\"lib\\"',
        '-D SHARED_LIB_SUFFIX=\\".so\\"',
        '-D NGRAPH_TF_USE_GRAPPLER_OPTIMIZER',
        "-I src",
        "-I logging",
        "-I external/ngraph/src",
    ] + CXX_ABI,
    linkopts = [
        "-Wl,-z,noexecstack",
        "-Wl,-z,relro",
        "-Wl,-z,now",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = 'libngraph_bridge.so',
    linkshared = 1,
    deps = [
        ":ngraph_bridge_headers",
        ":ngraph_bridge_lib",
        ":ngraph_bridge_version",
        # # Known to work
        # "@org_tensorflow//tensorflow/core:framework",
        # "@org_tensorflow//tensorflow/core:lib",
        # "@org_tensorflow//tensorflow/core:protos_all_cc",
        # "@org_tensorflow//tensorflow/core:tensorflow",

        #"@org_tensorflow//tensorflow:libtensorflow_framework.so",
        "@local_config_tf//:libtensorflow_framework",
        # "@local_config_tf//:tf_header_lib",
        "@ngraph//:ngraph_headers",
        "@ngraph//:ngraph_core",
        "@ngraph//:ngraph_version",
    ],
    copts = [
        "-pthread", 
        "-std=c++11", 
        "-D_FORTIFY_SOURCE=2",
        "-Wformat",
        "-Wformat-security",
        "-Wformat",
        "-fstack-protector-all",
        "-D NDEBUG",
        '-D SHARED_LIB_PREFIX=\\"lib\\"',
        '-D SHARED_LIB_SUFFIX=\\".so\\"',
        '-D NGRAPH_TF_USE_GRAPPLER_OPTIMIZER',
        "-I src",
        "-I logging",
        "-I external/ngraph/src",
    ] + CXX_ABI,
    linkopts = [
        "-Wl,-z,noexecstack",
        "-Wl,-z,relro",
        "-Wl,-z,now",
    ],
    visibility = ["//visibility:public"],
)

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")
tf_cc_binary(
    name = 'hello_tf',
    srcs = [
        "examples/tf_cpp_examples/hello_tf.cpp",
    ],
    deps = [
        ":ngraph_bridge_headers",
        ":ngraph_bridge_lib",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        # "@org_tensorflow//tensorflow/core:core_cpu",
        # "@org_tensorflow//tensorflow/core:framework",
        # "@org_tensorflow//tensorflow/core:lib",
        # "@org_tensorflow//tensorflow/core:protos_all_cc",
        # "@org_tensorflow//tensorflow/core:tensorflow",

        "@ngraph//:ngraph_headers",
    ],
    copts = [
        "-pthread", 
        "-std=c++11", 
        "-D_FORTIFY_SOURCE=2",
        "-Wformat",
        "-Wformat-security",
        "-Wformat",
        "-fstack-protector-all",
        "-D NDEBUG",
        "-I src",
        "-I external/ngraph/src",
        "-I logging",
    ] + CXX_ABI,
    linkstatic=False,
    visibility = ["//visibility:public"],
)
