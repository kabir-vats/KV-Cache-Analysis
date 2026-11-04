# KV Cache Analysis

This repository contains Jupyter notebooks for studying the memory and inference behavior of transformer language-model KV caches. The experiments cover:

- Input and generation length versus KV-cache size, peak GPU memory, time to first token (TTFT), and decoding throughput.
- Retrieval accuracy in a synthetic “needle-in-a-haystack” task as context length and the location of the answer change.
- CSV-based plotting of benchmark results collected for multiple models.

The notebooks are research/experiment artifacts, not a packaged Python library. They were written for notebook environments such as Kaggle and include environment-specific paths and setup cells.

## Repository contents

| File | Purpose |
| --- | --- |
| `kv-audit-226.ipynb` | Benchmarks model prefill and generation while recording KV-cache size, peak allocated CUDA memory, TTFT, and decoding throughput. The checked-in configuration loads `google/gemma-2b` and varies input length. |
| `needle-in-hay-stack-226.ipynb` | Creates synthetic contexts containing a random passcode (“needle”), then measures exact/substring retrieval accuracy and latency across context lengths and insertion depths. |
| `226-data-analysis.ipynb` | Reads benchmark CSV files from Kaggle input datasets and plots KV-cache size, throughput changes, and TTFT against input or generated-token length. |

## Metrics and terminology

- **KV cache**: key and value tensors retained during autoregressive generation to avoid recomputing attention states for prior tokens.
- **Prefill KV size**: size of the KV tensors after processing the prompt.
- **Final KV size**: size after the configured generation run, when available from the generation output.
- **Peak allocated memory delta**: maximum CUDA memory allocated during a benchmark condition minus the allocation baseline measured before the run. This is not total device memory usage or reserved memory.
- **TTFT**: time measured around the prompt prefill forward pass that produces the first next-token logits.
- **Tokens/sec**: generated-token count divided by measured decode time. It is decode throughput, not end-to-end throughput including prefill.
- **Needle exact match**: whether the normalized prediction equals the passcode exactly.
- **Needle substring match**: whether the normalized prediction contains the expected passcode.

The KV audit notebook counts tensor bytes using tensor element counts and dtypes. Its KV measurements are therefore estimates of the tensors encountered by its cache-inspection helper, not a complete accounting of model weights, activations, CUDA allocator reservations, or all runtime overhead.

## Running on Kaggle

The notebooks expect a GPU-backed notebook session for the KV audit and model-generation experiments. In Kaggle:

1. Open or upload the notebook.
2. Select an accelerator with a compatible NVIDIA GPU and enable internet/model downloads as permitted by the environment.
3. Install any notebook-specific dependencies in the notebook session.
4. Configure model access and input datasets as described below.
5. Run the notebook from top to bottom so definitions and configuration cells are initialized.
6. Download artifacts from `/kaggle/working` before the session ends, or save them to a Kaggle output dataset.

The repository does not include a `requirements.txt`, pinned package versions, benchmark CSVs, model weights, or a tested local runner. Notebook APIs—especially Hugging Face cache APIs—are version-sensitive. Use mutually compatible versions of PyTorch, Transformers, Datasets, and the optional quantization backend in the notebook environment.

### KV-cache audit notebook

`kv-audit-226.ipynb` imports PyTorch, Matplotlib, Hugging Face Datasets and Transformers. It loads a causal language model and tokenizer with `device_map="auto"` and FP16 weights. Its checked-in settings select `google/gemma-2b`, use a dynamic cache, and sweep input lengths:

```text
100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900
```

The benchmark uses a batch size of 1 and requests 100 new tokens by default. Change the model, cache mode, sweep, and output directory in the final configuration cells to run a different experiment. Gemma model access may require accepting its Hugging Face license and authenticating in the notebook environment.

This notebook's code calls CUDA synchronization and CUDA memory APIs directly. A CUDA GPU is required by the current implementation; the `device_type_for_kv` argument does not make the entire benchmark CPU-compatible. The checked-in run selects `cache_implementation="dynamic"`. Although the benchmark function has a `quantized` branch, that branch refers to `QuantCache`, which is not defined in the notebook as checked in. Do not select quantized mode without supplying a compatible cache implementation and validating it against the installed Transformers version.

### Needle-in-a-haystack notebook

`needle-in-hay-stack-226.ipynb` generates a synthetic context from repeated neutral text, inserts a randomly generated passcode at a specified fractional depth, and asks the model to return only that passcode. Its checked-in configuration uses:

- Model: `microsoft/phi-2`
- Context lengths: 512, 1024, 1536, and 2048 tokens (bounded by the model's configured maximum)
- Needle depths: 0.1, 0.3, 0.5, 0.7, and 0.9
- Trials per length/depth pair: 3
- Batch size: 1
- Generation limit: 20 tokens
- Cache mode: quantized, HQQ backend, 4-bit

The notebook includes an HQQ installation cell (`pip install hqq`). Quantized-cache support depends on the installed Transformers and HQQ versions and their hardware compatibility. If changing to dynamic cache, update the cache configuration cell rather than assuming the current HQQ settings are still active.

The notebook reads an `HF_TOKEN` from Kaggle Secrets. Add that secret only if needed for your chosen model or Hugging Face access; never commit access tokens to the notebook or repository. It also sets `HF_USERNAME`, which should be changed or removed when running under a different account.

## Results and output formats

### KV audit

The KV audit writes plot images and a raw CSV to its configured output directory (the checked-in example uses `/kaggle/working`). Each CSV row represents a sweep condition and includes:

| Column | Meaning |
| --- | --- |
| `feature`, `x` | Swept parameter and its value. |
| `max_new_tokens`, `batch_size_default`, `input_length_default` | Generation and non-swept parameter settings. |
| `peak_mem_delta_mib` | Peak allocated CUDA memory delta in MiB. |
| `kv_prefill_mib`, `kv_final_mib` | Measured cache sizes in MiB after prefill and generation. |
| `tokens_per_sec`, `ttft_sec` | Decode throughput and prefill/first-token latency in seconds. |
| `cache_implementation`, `nbits`, `model_name` | Cache and model configuration. |

The notebook also writes a JSON sidecar containing sweep values and experiment parameters.

### Needle benchmark

The needle notebook writes `needle_results.csv`, `needle_summary.json`, and plots to `/kaggle/working/needle_benchmark` by default. The CSV records context length, needle depth, trial, expected passcode, prediction, exact and substring matches, per-sample batch latency, and model name. The summary JSON includes average match scores and the tested settings. The plotted accuracy aggregation currently uses substring match values; consult the CSV for exact-match results.

### Plotting notebook inputs

`226-data-analysis.ipynb` expects Kaggle datasets mounted under:

```text
/kaggle/input/datasets/kabirvats/inp-len/*.csv
/kaggle/input/datasets/kabirvats/max-new-tok/*.csv
```

The input-length files are expected to contain `model_name`, `x`, `kv_prefill_mib`, `tokens_per_sec`, and `ttft_sec`. The generation-length files are expected to contain `model_name`, `x`, `kv_final_mib`, and `tokens_per_sec`. The plotting notebook assumes each CSV contains results for one model and uses a predefined model-to-color map; an unlisted model name can cause a color lookup error.

It saves `inp_vs_kv.png`, `inp_vs_speed.png`, `gen_vs_kv.png`, and `gen_vs_speed.png` in the current notebook working directory. The TTFT chart is displayed but is not saved by the checked-in cell. The throughput plots calculate change relative to the second row (`y.iloc[1]`); ensure that row is the intended baseline (the chart text currently refers to 200 tokens/input units).

## Interpreting and comparing results

For fair comparisons, hold the model, tokenizer, dtype, device, batch size, prompt contents, generation length, cache implementation, and software versions constant while changing one variable at a time. Warm-up, GPU contention, memory fragmentation, kernel selection, and quantization overhead can affect measurements. Repeat runs and report the hardware/software configuration with results.

The audit notebook's CSV contains configuration metadata, but the plotting notebook may combine independently collected CSVs. Check that the input files use comparable settings before drawing conclusions. A smaller KV cache does not necessarily imply higher throughput or lower end-to-end latency.

Needle retrieval is a synthetic task and its generated contexts are repetitive. Its scores should not be interpreted as general language understanding or as a comprehensive long-context evaluation.

## Reproducibility and limitations

- The notebooks use random values for synthetic passcodes and do not set a global seed in the checked-in configuration.
- The audit and needle notebooks are separate experiments with different model/cache configurations.
- Kaggle paths and Kaggle Secrets are environment-specific; adapt them for Colab or local execution.
- GPU measurements are sensitive to the accelerator model, CUDA/PyTorch versions, and other processes using the GPU.
- Notebooks are the source of truth for the current experiment settings; there is no CLI, package API, automated test suite, or dependency lockfile in this repository.
