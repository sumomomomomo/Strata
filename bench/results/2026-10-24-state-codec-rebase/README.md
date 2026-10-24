# BF16 checkpoint rebase: V100 validation, 2026-10-24

Tested code: `a8d4049d` (18 replayed PR commits plus the integration fix), on upstream `8183c6b6`. Documentation and evidence were added afterward; they do not change the tested engine.

## Environment and method

Tesla V100 PCIe 32 GB, Linux, CUDA 12.8, NVIDIA driver 580.178.04, 93.6 GiB system RAM. The CPU reports `Genuine Intel(R) 0000`. The model was the existing SC117 IQ3_S native Qwen3.8 Flash Next GSQ-RCO abliterated pack, INT8 KV, 32,768 resident KV cells, and the same routing profile in every arm.

The arms were upstream/default FP32, rebased/FP32, and rebased/BF16. Each case started a fresh text-only server on GPU 0, localhost port 8094. The installed production engine/configuration were preserved and restored after testing. Requests used temperature 0, MTP `--spec 4`, `--adapt-swaps 0`, `--pcie-frac 0`, `--suffix-draft 0`, and checkpoint spacing 8,192. Other model settings came from the existing configuration.

The primary matrix ran each arm at both context sizes twice, reversing arm order in round two. The observed prompts contained 32,528 and 67,203 tokens. Each case saved the long conversation, switched to a disjoint conversation, resumed the saved prefix, rewound to a retained checkpoint, restored the disk file, and generated a 512-token continuation.

The primary matrix kept upstream adaptive CPU prompt sharing. Identical-build repeats showed rewind-logit variation. A second, one-round matrix set `STRATA_PREFILL_CPU_SHARE=0` in every arm to isolate codec effects. This control also measured three warm SAVE/RESTORE cycles per arm/context while retaining the full conversation; saved token counts were checked and all 18 warm disk resumptions were verified.

Early primary warm cycles saved short side conversations. Those 36 samples are explicitly excluded from long-context warm timing claims. Their exclusion counts remain in the data. Initial startup/fixture failures were corrected before these completed matrices; the superseded fixture test output is preserved in `checks.txt`.

## Storage and parking

Primary-matrix medians, two fresh runs per cell. Parked bytes are cache-accounted allocation, not process RSS. File sizes include unchanged K/V data.

| Prompt tokens | FP32 file MiB | BF16 file MiB | FP32 parked MiB | BF16 parked MiB | FP32 park ms | BF16 park ms | FP32 resume ms | BF16 resume ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32528 | 699.06 | 586.47 | 1037.03 | 755.55 | 329.55 | 262.80 | 47.95 | 51.20 |
| 67203 | 1204.16 | 1091.56 | 1767.98 | 1373.90 | 564.65 | 462.80 | 81.80 | 82.40 |

BF16 files were 16.1% smaller near 32K and 9.4% smaller near 64K. Parking was about 20% and 18% faster in these two-run comparisons. These are workload-specific measurements, not token-generation speed claims.

## Warm disk operations

CPU-sharing-off control, medians of three warm operations per cell, one process per arm/context. This matrix had one arm order; it does not establish timing improvements on other machines or under memory pressure.

| Prompt tokens | FP32 SAVE ms | BF16 SAVE ms | FP32 RESTORE ms | BF16 RESTORE ms |
| ---: | ---: | ---: | ---: | ---: |
| 32528 | 398.9 | 273.2 | 449.2 | 386.4 |
| 67203 | 578.2 | 434.2 | 783.3 | 717.9 |

## Compatibility and quality

- All 18 long-context cases verified disk restoration. The parked-resume questions recovered all 54 checked code words. These are repeated checks on two deterministic documents, not a broad quality evaluation.
- Upstream versus rebased FP32: all six initial session files were byte-identical, all six 512-token continuation texts were identical, and all six sampled first-token logit vectors in the fixed-path control were exactly equal.
- BF16 versus rebased FP32: every sampled vector contained 248,320 finite values. The top token agreed at 4 of 6 control positions. Full-vocabulary KL values, in nats, are shown below. No acceptance threshold was chosen.
- The logit samples are the first generated token after parked resumption, a retained-checkpoint rewind, and disk restoration/continuation. They are not teacher-forced measurements of the full 512-token continuations. Correct recall and finite logits do not establish equivalent model quality; the maintainer-side quality review remains necessary.

| Prompt tokens | Position | KL(FP32 || BF16) | Same top token |
| ---: | --- | ---: | --- |
| 32528 | resume_A | 0.048051088 | no |
| 32528 | rewind_mid | 0.000805472 | yes |
| 32528 | continuation | 0.046064663 | yes |
| 67203 | resume_A | 0.022407018 | no |
| 67203 | rewind_mid | 0.000649282 | yes |
| 67203 | continuation | 0.053218779 | yes |

## Build and regression coverage

- Release CUDA build for `sm_70`, using CUDA 12.8 and `STRATA_EXPERIMENTAL_SM60=ON`: passed.
- Eleven focused tests with the V100 selected: passed. They cover codec conversion, session files, cache/split metadata, SAVE admission, allocation/transfer failures, snapshots, and the new coded-donor prefix restore.
- Seven standalone conversation/codec tests configured without CUDA: passed. Upstream baseline host and GPU snapshot tests also passed.
- The integration preserves upstream direct/queued FP32 session copies, restores the full fixture before independent codec checks, and checks coded donor immutability and widened spare pooled rows.
- Metal now links the shared codec object and explicitly refuses unsupported BF16 mode/checkpoints. No Metal build or execution was performed. HIP and SYCL compilers were unavailable. Windows and real multi-GPU park/restore were not tested. These limits preclude claiming full cross-backend validation.

Final and superseded test outputs and configure logs are in [checks.txt](checks.txt). Operation samples, ranges, first-file hashes, binary/harness hashes and individual logit comparisons are in [summary.json](summary.json). Large raw logits, session binaries and the full sanitized evidence bundle were retained locally rather than committed.

## Reproduce the control protocol

[run.py](run.py) is a portable copy of the corrected control harness; only machine-specific source/config/Python defaults and the empty-output-directory guard differ from the executed copy. Use the same converted native model and routing profile in both source checkouts. The base model JSON must use absolute model/tokenizer paths and have the dependencies needed by `serve/server.py`. The script creates its own text-only configs and refuses to run while GPU compute processes already exist; stop the serving model before running it.

```bash
export STRATA_BENCH_UPSTREAM=/absolute/path/to/upstream-8183c6b6
export STRATA_BENCH_REBASED=/absolute/path/to/rebased-a8d4049d
export STRATA_BENCH_CONFIG=/absolute/path/to/model.json
export STRATA_BENCH_PYTHON=/absolute/path/to/venv/bin/python
export STRATA_PREFILL_CPU_SHARE=0
"$STRATA_BENCH_PYTHON" run.py \
  --upstream-exe /absolute/path/to/upstream-build/strata \
  --rebased-exe /absolute/path/to/rebased-build/strata \
  --out-dir /absolute/path/to/new-empty-results \
  --sizes 32768,65536 --reps 3 --rounds 1
```

Both engines were configured with `-DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON`, the CUDA 12.8 compiler/toolkit paths, and built with `-j8`. Exact configure output is retained in the checks file. The main two-round cold/parking matrix left CPU sharing at its upstream default; the control intentionally disables it.

Rebase and validation preparation used Codex and a local `qwen3.8-next-sil` worker. Source review, command output and measurement records support the claims; a model review is not validation.
