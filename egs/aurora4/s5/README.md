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

### Recipes and Results

| **Recipe**               | **Command**                                                 | **test_eval92** | **test_0166** |
|--------------------------|-------------------------------------------------------------|:---------------:|:-------------:|
| Baseline TDNN-F          | `local/chain/tuning/run_tdnn_1a.sh`                         |       8.11      |      8.49     |
| + CMN                    | `local/chain/tuning/run_tdnn_1a.sh --apply-cmvn true`       |       7.77      |      8.08     |
| Multi-condition training | `local/chain/tuning/run_tdnn_1b.sh`                         |       6.83      |      7.00     |
| i-vector                 | `local/chain/tuning/run_tdnn_1c.sh --ivector-type offline`  |       6.76      |      7.08     |
| i-vector (online)        | `local/chain/tuning/run_tdnn_1c.sh --ivector-type online`   |       8.55      |      8.90     |
| NAT vector               | `local/chain/tuning/run_tdnn_1d.sh --noise-type seltzer`    |       8.18      |      8.43     |
| e-vector                 | `local/chain/tuning/run_tdnn_1d.sh --noise-type evec_lda`   |       7.68      |      7.93     |
| Bottleneck NN            | `local/chain/tuning/run_tdnn_1d.sh --noise-type bottleneck` |       8.16      |      8.44     |
| Noise vectors            | `local/chain/tuning/run_tdnn_1e.sh --type offline`          |       7.39      |      7.67     |
| Noise vectors (MLE)      | `local/chain/tuning/run_tdnn_1e.sh --type mle`              |       7.88      |      7.99     |

**Note:** Results in the above table may be different from those in the paper due 
to the use of online CMN in the experiments conducted for the paper. It seems online CMN
degrades offline i-vector performance considerably. The comparison between the noise
vector baselines follow the same trend as in the paper.
