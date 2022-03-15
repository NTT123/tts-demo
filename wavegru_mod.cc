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
  int input_dim;
  int embed_dim;
  int hidden_dim;
  mat m1, m2, m3;
  vec b1, b2, b3;
  vec z, r, hh;
  vec fco1, fco2;
  vec o1b, o2b;
  vec t;
  vec h;
  mat o1, o2;
  std::vector<vec> embed;

  WaveGRU(int input_dim, int embed_dim, int hidden_dim)
      : input_dim(input_dim),
        embed_dim(embed_dim),
        hidden_dim(hidden_dim),
        b1(hidden_dim),
        b2(hidden_dim),
        b3(hidden_dim),
        z(hidden_dim),
        r(hidden_dim),
        hh(hidden_dim),
        fco1(hidden_dim),
        fco2(256),
        t(hidden_dim + input_dim + embed_dim),
        h(hidden_dim),
        o1b(hidden_dim),
        o2b(256) {
    m1 = create_mat(input_dim + hidden_dim + embed_dim, hidden_dim);
    m2 = create_mat(input_dim + hidden_dim + embed_dim, hidden_dim);
    m3 = create_mat(input_dim + hidden_dim + embed_dim, hidden_dim);
    o1 = create_mat(hidden_dim, hidden_dim);
    o2 = create_mat(hidden_dim, 256);
    embed = std::vector<vec>();
    for (int i = 0; i < 256; i++) {
      embed.emplace_back(embed_dim);
      embed[i].FillRandom();
    }
  }

  void load_embed(fndarray embed_weights) {
    auto a_embed = embed_weights.unchecked<2>();
    for (int i = 0; i < 256; i++) {
      for (int j = 0; j < embed_dim; j++) embed[i][j] = a_embed(i, j);
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

  void load_weights(fndarray m1, indarray m1_mask, fndarray b1, fndarray m2,
                    indarray m2_mask, fndarray b2, fndarray m3,
                    indarray m3_mask, fndarray b3, fndarray o1,
                    indarray o1_mask, fndarray o1b, fndarray o2,
                    indarray o2_mask, fndarray o2b) {
    this->m1 = load_linear(this->b1, m1, m1_mask, b1);
    this->m2 = load_linear(this->b2, m2, m2_mask, b2);
    this->m3 = load_linear(this->b3, m3, m3_mask, b3);
    this->o1 = load_linear(this->o1b, o1, o1_mask, o1b);
    this->o2 = load_linear(this->o2b, o2, o2_mask, o2b);
  }

  std::vector<int> inference(fndarray ft, float temperature) {
    auto rft = ft.unchecked<2>();
    std::vector<vec> xs;
    for (int i = 0; i < rft.shape(0); i++) {
      xs.emplace_back(input_dim);
      for (int j = 0; j < input_dim; j++) xs[i][j] = rft(i, j);
    }

    int value = 127;
    std::vector<int> signal(xs.size());
    h.FillZero();
    for (int index = 0; index < xs.size(); index++) {
      for (int i = 0; i < embed_dim; i++) t[i] = embed[value][i];
      for (int i = 0; i < input_dim; i++) t[embed_dim + i] = xs[index][i];
      for (int i = 0; i < hidden_dim; i++) t[embed_dim + input_dim + i] = h[i];
      m1.SpMM_bias(t, b1, &z, false);
      m2.SpMM_bias(t, b2, &r, false);
      z.Sigmoid();
      r.Sigmoid();

      for (int i = 0; i < hidden_dim; i++) {
        t[embed_dim + input_dim + i] = h[i] * r[i];
      }

      m3.SpMM_bias(t, b3, &hh, false);
      hh.Tanh();
      for (int i = 0; i < hidden_dim; i++) {
        h[i] = (1. - z[i]) * h[i] + z[i] * hh[i];
      }
      o1.SpMM_bias(h, o1b, &fco1, true);
      o2.SpMM_bias(fco1, o2b, &fco2, false);
      value = fco2.Sample(temperature);
      signal[index] = value;
    }
    return signal;
  }
};

PYBIND11_MODULE(wavegru_mod, m) {
  py::class_<WaveGRU>(m, "WaveGRU")
      .def(py::init<int, int, int>())
      .def("load_embed", &WaveGRU::load_embed)
      .def("load_weights", &WaveGRU::load_weights)
      .def("inference", &WaveGRU::inference);
}
