# kaldi-noise-vectors

This repository contains the baselines and proposed noise vectors from the paper:
[Frustratingly Easy Noise-aware Training of Acoustic Models](https://arxiv.org/abs/2011.02090).

### Setup

**Prerequisite:** The setup below assumes that you have a working installation of
the [Kaldi](https://github.com/kaldi-asr/kaldi) ASR toolkit. 

* Copy the contents of the `src` directory to the corresponding directory in your
Kaldi installation.

* Navigate to `/path/to/kaldi/src` and run the following:

```shell
cd ivectorbin && make compute-noise-vector compute-noise-vector-seltzer && cd ..
```

* If you additionally want to use the proposed online MLE and MAP noise vectors,
run the following:

```shell
cd ivector && make online-noise-vector && cd ..
cd ivectorbin && make compute-noise-prior compute-noise-vector-online && cd ..
```

### Usage

We provide example usage on the Aurora4 dataset. For model details and how to run
the recipes, please check `egs/aurora4/s5/README.md`.

### Citation

If you found this code useful, consider citing:

```shell
@article{Raj2020FrustratinglyEN,
  title={Frustratingly Easy Noise-aware Training of Acoustic Models},
  author={Desh Raj and Jes{\'u}s Villalba and Daniel Povey and Sanjeev Khudanpur},
  journal={ArXiv},
  year={2020},
  volume={abs/2011.02090}
}
```
