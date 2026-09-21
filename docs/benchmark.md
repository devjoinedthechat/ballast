# Benchmark

`scripts/benchmark.py` measures the four claims the README makes, at three
sizes. The small size runs in CI; the largest builds a four-gigabyte model and
takes about a minute.

```
python scripts/benchmark.py            # the middle size
python scripts/benchmark.py --size s   # seconds
python scripts/benchmark.py --size l   # a 4 GB model, the real memory test
```

Absolute throughput depends on the machine and is not worth quoting. The ratios
are what the claims are about, and they hold across all three sizes.

## How it is measured

Every scenario runs in its own process, and the fixtures it reads are built by a
different process again.

Both matter. Peak resident memory is a high-water mark that freeing does not
lower, so a process that generated a four-gigabyte model before measuring
reports four gigabytes whatever the measured code then does. The first version
of this script did exactly that, and reported the streaming apply holding the
whole model — a measurement artefact that looked like a refuted claim.

Reading a memory-mapped file also makes its pages resident, and writing one
leaves dirty pages behind, so peak RSS for any code that streams a model through
cannot sit far below the model itself. That is why the memory scenarios measure
a streamed run *against* a whole one: the difference between them is the
allocation, which is the thing actually being claimed.

## A retrain stores what changed

An adapter over q, k, v and o projections, retrained so two of its tensors
differ:

| | small | middle | large |
| --- | --- | --- | --- |
| adapter | 2 MB | 34 MB | 134 MB |
| tensors | 64 | 256 | 256 |
| blocks added by the retrain | 2 | 2 | 2 |
| bytes added | 0.07 MB | 0.26 MB | 1.05 MB |
| versus storing it again | 32x less | 128x less | 128x less |

The bytes added equal the bytes in the two tensors that changed, to within a
rounding error, at every size. Deduplication across the two versions is 1.98x —
the ceiling for two versions that share all but two tensors — and zstd takes
another 1.29x off bf16 weights.

## Applying a delta holds one tensor, not a model

`apply` writes `base + delta` as a model directory. Streamed, as
`apply_commit` does it, against the same result built in memory first:

| | small | middle | large |
| --- | --- | --- | --- |
| model | 67 MB | 537 MB | 4295 MB |
| peak RSS, built whole | 234% of the model | 214% | 159% |
| peak RSS, streamed | 132% | 115% | 108% |
| streaming holds | 68 MB less | 529 MB less | 2228 MB less |
| streamed, relative speed | 1.6x faster | 1.9x | 2.2x |

At every size the difference is one model: that is the copy streaming does not
make. It is also faster, by more as the model grows, because the copy it skips
is the model.

## Resolving a view holds one tensor per input

A TIES view over two adapters, resolved with the view cache off so the merge
actually runs:

| | small | middle | large |
| --- | --- | --- | --- |
| inputs | 4 MB | 67 MB | 268 MB |
| held, whole | +9 MB | +115 MB | +453 MB |
| held, streamed | +3 MB | +14 MB | +50 MB |

Held whole, a merge grows with the inputs. Streamed, it grows with the largest
tensor.

## What it found

Two things, which is what a benchmark is for.

The first version reported the streaming apply holding 134% of the model, which
read as a refuted claim and was a flaw in the measurement: the fixture was built
in the process doing the measuring. Fixtures moved out, and the comparison
against a whole build went in.

The second was real. Writing a tensor went through `tobytes()`, which builds a
copy of it before the write — a few megabytes per tensor, gigabytes over a whole
model. It was enough to make the streaming apply *slower* than the version that
holds everything: on the four-gigabyte model, 23.3 s against 10.7 s.

Tensors are written straight from the array now, with the copy kept only for
destinations that have no file descriptor to write to, such as an HTTP response
body. The same apply takes 4.9 s: 4.8x faster than it was, and 2.2x faster than
building the model in memory rather than 1.8x slower.

That is the whole argument for measuring. The claim was true — streaming did
hold two gigabytes less — and it was also costing twice the time, which nothing
in the test suite could have shown.
