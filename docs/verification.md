# Verification

Ballast reimplements merge methods that mergekit already defines. A
reimplementation is worth nothing unless it agrees with the original, so
`scripts/verify_real.py` compares the two directly and reports the difference.

It runs by hand rather than in CI: it downloads a model and needs mergekit,
torch and transformers installed. Everything below is its output.

## Environment

"Matches mergekit" is not checkable without saying which mergekit — its methods
change between releases. The script prints the versions it ran under, and these
are the ones the results below were produced with:

```
ballast b66f5e7 · mergekit 0.1.4 · torch 2.14.0 · transformers 5.12.1
peft 0.21.0 · numpy 2.5.3 · safetensors 0.5.3
```

## Merge methods against mergekit's own functions

Each sparsifier and each geometric method is called alongside mergekit's
implementation of it on the same input, and the largest absolute difference
across the outputs is reported.

| method | compared against | max absolute difference |
| --- | --- | --- |
| `magnitude`, the TIES sparsifier | `mergekit.sparsify` | 0.00e+00 |
| `magnitude_outliers`, the breadcrumbs sparsifier | `mergekit.sparsify` | 0.00e+00 |
| `slerp`, over ten interpolation points | `mergekit.merge_methods.slerp` | 2.38e-07 |
| `nuslerp` | `mergekit.merge_methods.nuslerp` | 7.15e-07 |
| `multislerp`, over two to four inputs | `mergekit.merge_methods.multislerp` | 3.17e-07 |
| `sce`, at full and half selection | `mergekit.merge_methods.sce` | 2.38e-07 |
| DELLA keep probabilities | `mergekit.sparsify` | 5.44e-08 |

The deterministic sparsifiers agree exactly. The rest agree to float32
precision, which is the floor for arithmetic carried out in float32.

DELLA and DARE draw a random mask, so their outputs cannot agree with mergekit's
by construction: mergekit draws from the global torch generator, and ballast
draws from a seed stored in the view so that resolving it twice gives the same
answer. What is compared for DELLA is therefore the keep probabilities, which
are the deterministic half of the method.

## Views against mergekit's output

A view is recorded over two deltas and resolved; mergekit is run on the same
recipe; the resolved tensors are compared to the file mergekit wrote.

| method | tensors | result |
| --- | --- | --- |
| `linear` | 60 | max absolute difference 0.00e+00 |
| `ties` | 60 | 4 entries differ, all at a density-boundary tie |

TIES keeps a fixed fraction of entries by magnitude. Where two entries tie
exactly at that boundary, which one survives is decided by a sort that mergekit
leaves unstable and ballast makes stable, so either may keep a different member
of the pair. The script counts the ties per tensor and requires every differing
entry to be accounted for by one; four differ and none are unexplained.

## The model round trip

A delta is extracted from two model directories, composed as a view, and applied
back onto the base as a model directory.

| | |
| --- | --- |
| tensors in the rebuilt model | 272 |
| max absolute difference from mergekit's model | 2.38e-07 |
| untouched tensors byte-identical | 212 of 212 |
| `transformers` loads the result | yes |

## Behaviour on a live model

`PeftRunner` applies an adapter to `HuggingFaceTB/SmolLM2-135M` in process and
generates greedily, so a diff can report what changed in the answers rather than
in the weights.

| | tensors changed | relative change | probes moved |
| --- | --- | --- | --- |
| two layers retrained | 4 of 120 | 8.6125 | 4 of 4, similarity 0.51 |
| one tensor nudged by 3% | 1 of 120 | 0.0020 overall, 0.0300 in that tensor | 0 of 4 |

The second row is the case the diff exists for. The weights moved and no probe
noticed, and that is reported as `UNOBSERVED CHANGE` rather than as no change:
the aggregate is a fifth of a percent, small enough to overlook, while the
tensor that moved changed by 3%. Both numbers are reported, and either crossing
the threshold raises the condition.

## Reproducing

```
uv pip install -e ".[verify]"
python scripts/verify_real.py
```

It downloads about 270 MB, runs on CPU, and takes a few minutes. Every check
prints PASS or FAIL and the script exits non-zero if any fail.
