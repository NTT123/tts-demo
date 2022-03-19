---
title: WaveGRU Text To Speech
emoji: 🌍
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 2.8.10
app_file: app.py
pinned: false
license: mit
---


## Build wavenet-cpp


    ./bazelisk-linux-amd64 build wavegru_mod -c opt --copt=-march=native
    cp -f bazel-bin/wavegru_mod.so .

