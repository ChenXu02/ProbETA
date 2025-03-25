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

Harbin: The dataset consists of over **1 million trips** collected by **13,000+ taxis** over a **5-day period**.

Chengdu: The dataset consists of over **1.4 million taxis** collecting more than **1.4 billion GPS records** over a **28-day period** (from **August 3, 2014, to August 30, 2014**).

### Download Dataset

Download the dataset from the following link:

Harbin: [Download Dataset](https://drive.google.com/open?id=1tdgarnn28CM01o9hbeKLUiJ1o1lskrqA)

Chengdu: [Download Dataset](https://challenge.datacastle.cn/v3/cmptDetail.html?id=175)

Extracting data from OpenStreetMap can be used to obtain the road network. You can download the road network data for Chengdu from [OpenStreetMap](https://www.openstreetmap.org/) and use it for map matching.


### Data Format

Each `.h5` file contains multiple trips recorded on a given day. Each trip consists of three fields:

- `lon` (longitude)
- `lat` (latitude)
- `tms` (timestamp)

To read `.h5` files, use the [`readtripsh5`](https://github.com/Chenxu02/ProbETA/julia/Trip.jl#L28) function in Julia. If using your own dataset, refer to `readtripsh5` to format your trajectories correctly into `.h5` files.

## Preprocessing

### Map Matching

Before training, trips must be map-matched using the [Barefoot](https://github.com/boathit/barefoot) matching server. Follow the instructions in the Barefoot repository to set up the required servers.

Once the servers are running, execute the following command to match trips:

```bash
cd ProbETA/julia
julia -p 6 mapmatch.jl --inputpath ../data/h5path --outputpath ../data/jldpath
```

Here, `6` represents the number of available CPU cores.

## Training the Model

Before training, ensure the road network PostgreSQL server is set up by following the instructions in [Barefoot](https://github.com/boathit/barefoot).

To train the model, navigate to the `Model` directory and run:

```bash
cd ProbETA/Model
python train.py -trainpath ../data/datapath
```
The dataset does not need to be manually split beforehand; it will be randomly divided into training and test sets automatically.
Once training is complete, the model will automatically run evaluation on the test set.


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



