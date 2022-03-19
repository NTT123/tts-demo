/*
WaveGRU:
> Embed > GRU > O1 > O2 > Sampling > ...
*/

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <random>
#include <vector>

#include "sparse_matmul/sparse_matmul.h"
namespace py = pybind11;
using namespace std;

using fvec = std::vector<float>;
using ivec = std::vector<int>;
using fndarray = py::array_t<float>;
using indarray = py::array_t<int>;
using mat = csrblocksparse::CsrBlockSparseMatrix<float, float, int16_t>;
using vec = csrblocksparse::CacheAlignedVector<float>;
using masked_mat = csrblocksparse::MaskedSparseMatrix<float>;

mat create_mat(int h, int w) {
  auto m = masked_mat(w, h, 0.90, 4, 4, 0.0, true);
  auto a = mat(m);
  return a;
}

struct WaveGRU {
  int hidden_dim;
  int repeat_factor;
  mat m;
  vec b;
  vec z, r, hh, zrh;
  vec fco1, fco2;
  vec o1b, o2b;
  vec t;
  vec h;
  vec logits;
  mat o1, o2;
  std::vector<vec> embed;

  WaveGRU(int hidden_dim, int repeat_factor)
      : hidden_dim(hidden_dim),
        repeat_factor(repeat_factor),
        b(3*hidden_dim),
        t(3*hidden_dim),
        zrh(3*hidden_dim),
        z(hidden_dim),
        r(hidden_dim),
        hh(hidden_dim),
        fco1(hidden_dim),
        fco2(256),
        h(hidden_dim),
        o1b(hidden_dim),
        o2b(256),
        logits(256) {
    m = create_mat(hidden_dim, 3*hidden_dim);
    o1 = create_mat(hidden_dim, hidden_dim);
    o2 = create_mat(hidden_dim, 256);
    embed = std::vector<vec>();
    for (int i = 0; i < 256; i++) {
      embed.emplace_back(hidden_dim * 3);
      embed[i].FillRandom();
    }
  }

  void load_embed(fndarray embed_weights) {
    auto a_embed = embed_weights.unchecked<2>();
    for (int i = 0; i < 256; i++) {
      for (int j = 0; j < hidden_dim * 3; j++) embed[i][j] = a_embed(i, j);
    }
  }

  mat load_linear(vec& bias, fndarray w, indarray mask, fndarray b) {
    auto w_ptr = static_cast<float*>(w.request().ptr);
    auto mask_ptr = static_cast<int*>(mask.request().ptr);
    auto rb = b.unchecked<1>();
    // load bias, scale by 1/4
    for (int i = 0; i < rb.shape(0); i++) bias[i] = rb(i) / 4;
    // load weights
    masked_mat mm(w.shape(0), w.shape(1), mask_ptr, w_ptr);
    mat mmm(mm);
    return mmm;
  }

  void load_weights(fndarray m, indarray m_mask, fndarray b,
                    fndarray o1, indarray o1_mask, 
                    fndarray o1b, fndarray o2,
                    indarray o2_mask, fndarray o2b) {
    this->m = load_linear(this->b, m, m_mask, b);
    this->o1 = load_linear(this->o1b, o1, o1_mask, o1b);
    this->o2 = load_linear(this->o2b, o2, o2_mask, o2b);
  }

  std::vector<int> inference(fndarray ft, float temperature) {
    auto rft = ft.unchecked<2>();
    int value = 127;
    std::vector<int> signal(rft.shape(0) * repeat_factor);
    h.FillZero();
    for (int index = 0; index < signal.size(); index++) {
      m.SpMM_bias(h, b, &zrh, false);

      for (int i = 0; i < 3 * hidden_dim; i++) t[i] = embed[value][i] + rft(index / repeat_factor, i);
      for (int i = 0; i < hidden_dim; i++) {
        z[i] = zrh[i] + t[i];
        r[i] = zrh[hidden_dim + i] + t[hidden_dim + i];
      }

      z.Sigmoid();
      r.Sigmoid();

      for (int i = 0; i < hidden_dim; i++) {
        hh[i] = zrh[hidden_dim * 2 + i]  * r[i] + t[hidden_dim * 2 + i];
      }
      hh.Tanh();
      for (int i = 0; i < hidden_dim; i++) {
        h[i] = (1. - z[i]) * h[i] + z[i] * hh[i];
      }
      o1.SpMM_bias(h, o1b, &fco1, true);
      o2.SpMM_bias(fco1, o2b, &fco2, false);
      // auto max_logit = fco2[0];
      // for (int i = 1; i <= 255; ++i) {
      //   max_logit = max(max_logit, fco2[i]);
      // }
      // float total = 0.0;
      // for (int i = 0; i <= 255; ++i) {
      //   logits[i] = csrblocksparse::fast_exp(fco2[i] - max_logit);
      //   total += logits[i];
      // }
      // for (int i = 0; i <= 255; ++i) {
      //   if (logits[i] < total / 1024.0) fco2[i] = -1e9;
      // }
      value = fco2.Sample(temperature);
      signal[index] = value;
    }
    return signal;
  }
};

PYBIND11_MODULE(wavegru_mod, m) {
  py::class_<WaveGRU>(m, "WaveGRU")
      .def(py::init<int, int>())
      .def("load_embed", &WaveGRU::load_embed)
      .def("load_weights", &WaveGRU::load_weights)
      .def("inference", &WaveGRU::inference);
}
