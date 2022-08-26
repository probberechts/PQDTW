# Elastic Product Quantization for Time Series

This repository contains a Cython implementation of the [PQDTW algorithm][1],
which adapts the [product quantization technique][2] to be used with elastic
distance measures (e.g., DTW). This enables real-time similarity search on
large in-memory data collections under time warping. The idea is to first
compress the data by partitioning the time series into equal length
sub-sequences which are represented by a short code. The distance between two
time series can then be efficiently approximated by pre-computed elastic
distances between their codes.

## Getting Started

```sh
# Clone the repository
git clone --recurse-submodules https://github.com/probberechts/pqdtw

# Compile the wavelib library
cd pqdtw/lib/wavelib/src
gcc -fPIC -c *.c
ar rcs libwavelib.a *.o

# Install and test
pip install .
pytest tests
```

## Research

If you make use of this package in your research, please consider citing the
following paper:

```bibtex
@article{robberechts2022elastic,
  title={Elastic Product Quantization for Time Series},
  author={Robberechts, Pieter and Meert, Wannes and Davis, Jesse},
  journal={arXiv preprint arXiv:2201.01856},
  year={2022}
}
```

[1]: <https://arxiv.org/abs/2201.01856>
[2]: <https://mccormickml.com/2017/10/13/product-quantizer-tutorial-part-1/>
