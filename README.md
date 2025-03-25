# ProbETA

This repository contains the code used in our paper: [Link Representation Learning for Probabilistic Travel Time Estimation](https://arxiv.org/abs/2407.05895).

## Requirements

To run the code, ensure your system meets the following requirements:

- **Operating System**: Ubuntu (tested on versions 16.04 and 18.04)
- **Programming Languages**:
  - [Julia](https://julialang.org/downloads/) >= 1.0
  - Python >= 3.6
- **Deep Learning Framework**:
  - PyTorch >= 0.4 (tested on versions 0.4 and 1.0)

To install the required Julia packages, run the following command in your terminal:

```bash
julia -e 'using Pkg; Pkg.add(["HDF5", "CSV", "DataFrames", "Distances", "StatsBase", "JSON", "Lazy", "JLD2", "ArgParse"])'
```

Python dependencies can be installed by referring to the source code.

## Dataset

The dataset consists of over **1 million trips** collected by **13,000+ taxis** over a **5-day period**. While this dataset is a subset of the one used in our paper, it is sufficient to reproduce results that closely match those reported in our research.

### Download Dataset

Download the dataset from the following link:

[Download Dataset](https://drive.google.com/open?id=1tdgarnn28CM01o9hbeKLUiJ1o1lskrqA)

Extract the `*.h5` files and place them in the following directory:

```
deepgtt/data/h5path
```

### Data Format

Each `.h5` file contains multiple trips recorded on a given day. Each trip consists of three fields:

- `lon` (longitude)
- `lat` (latitude)
- `tms` (timestamp)

To read `.h5` files, use the [`readtripsh5`](https://github.com/boathit/deepgtt/blob/master/harbin/julia/Trip.jl#L28) function in Julia. If using your own dataset, refer to `readtripsh5` to format your trajectories correctly into `.h5` files.

## Preprocessing

### 1. Map Matching

Before training, trips must be map-matched using the [Barefoot](https://github.com/boathit/barefoot) matching server. Follow the instructions in the Barefoot repository to set up the required servers.

Once the servers are running, execute the following command to match trips:

```bash
cd ProbETA/julia
julia -p 6 mapmatch.jl --inputpath ../data/h5path --outputpath ../data/jldpath
```

Here, `6` represents the number of available CPU cores.

### 2. Generating Training, Validation, and Test Data

Run the following command to process and split the dataset:

```bash
julia gentraindata.jl --inputpath ../data/jldpath --outputpath ../data/trainpath

cd .. && mv data/trainpath/150106.h5 data/validpath && mv data/trainpath/150107.h5 data/testpath
```

## Training the Model

Before training, ensure the road network PostgreSQL server is set up by following the instructions in [Barefoot](https://github.com/boathit/barefoot). The road network server (referenced in [db_utils.py](https://github.com/boathit/deepgtt/blob/master/harbin/python/db_utils.py#L8)) provides road segment features required by the model.

To train the model, navigate to the `Model` directory and run:

```bash
cd ProbETA/Model
python train.py -trainpath ../data/datapath
```

## Citation

If you use this repository in your research, please cite our paper:

```bibtex
@article{xu2024link,
  title={Link representation learning for probabilistic travel time estimation},
  author={Xu, Chen and Wang, Qiang and Sun, Lijun},
  journal={arXiv preprint arXiv:2407.05895},
  year={2024}
}
```

---

Our data preprocessing is partially based on the following work, and we sincerely appreciate their contribution:

```bibtex
@inproceedings{www19xc,
  author    = {Xiucheng Li and Gao Cong and Aixin Sun and Yun Cheng},
  title     = {Learning Travel Time Distributions with Deep Generative Model},
  booktitle = {Proceedings of the 2019 World Wide Web Conference on World Wide Web, {WWW} 2019, San Francisco, California, May 13-17, 2019},
  year      = {2019},
}
```



