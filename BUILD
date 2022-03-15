# [internal] load cc_fuzz_target.bzl
# [internal] load cc_proto_library.bzl
# [internal] load android_cc_test:def.bzl

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = [":__subpackages__"])

licenses(["notice"])

# To run all cc_tests in this directory:
# bazel test //:all

# [internal] Command to run dsp_util_android_test.

# [internal] Command to run lyra_integration_android_test.

exports_files(
    srcs = [
        "wavegru_mod.cc",
    ],
)

pybind_extension(
    name = "wavegru_mod",  # This name is not actually created!
    srcs = ["wavegru_mod.cc"],
    deps = [
        "//sparse_matmul",
    ],
)

py_library(
    name = "wavegru_mod",
    data = [":wavegru_mod.so"],
)

py_binary(
    name = "wavegru",
    srcs = ["wavegru.py"],
    deps = [
        ":wavegru_mod"
    ],
)

