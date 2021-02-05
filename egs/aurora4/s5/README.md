## Aurora4 recipes with noise aware training

### Setup

First create symbolic links for `steps` and `utils`. Run the following command with
the path (relative or absolute) to your Kaldi installation. The script also creates
a `path.sh` file and populates it with required paths.

```shell
./setup.sh /path/to/kaldi
```

Setup `cmd.sh` according to your compute cluster. See the Kaldi docs on 
[parallelization in Kaldi](https://kaldi-asr.org/doc/queue.html). If you want to
run everything on a single machine (**not recommended**), you can set

```shell
export train_cmd="run.pl"
export decode_cmd="run.pl"
```

In the `run.sh` script, set the path to `aurora4` and `wsj0` on line 13 and 16.

### Usage

First run the GMM bootstrapping (this is the same as the `run.sh` in the Kaldi recipe).

```shell
./run.sh
```

Once the training is complete, you can run any of the following recipes. Note that
if you have already run one of them, you may want to skip some of the stages while
running others, for e.g., stages for high-resolution feature extraction and lattice
generation.

### Recipes

### Results