# mojo-pendulum

A focused, standalone port of Pendulum's datetime arithmetic and parsing core to
[Mojo](https://www.modular.com/mojo). It keeps familiar Pendulum names and signatures
while moving ISO-8601 parsing, Gregorian calendar arithmetic, and bulk ISO formatting
through a compiled shared library.

The Python module is named `mojo_pendulum`, so it can be installed beside the real
`pendulum` package for parity testing:

```python
import mojo_pendulum as pendulum

release = pendulum.parse("2024-01-31T18:30:00Z")
next_cycle = release.add(months=1, days=2)

print(next_cycle.to_iso8601_string())
# 2024-03-02T18:30:00Z
```

## Covered subset

The scalar API covers:

- `DateTime`, `Date`, `Time`, `Duration`, and `Interval`
- `datetime`, `date`, `time`, `duration`, `interval`, `instance`, `local`, and `naive`
- `parse` for ISO calendar, ordinal, and week dates, ISO times and durations
- `from_format` for the common Pendulum formatting tokens
- `now`, `today`, `tomorrow`, and `yesterday`
- named IANA zones, UTC, local zones, and fixed numeric offsets
- year/month/week/day arithmetic with month-end clamping
- elapsed time arithmetic across DST changes
- `start_of`, `end_of`, `diff`, `range`, timezone conversion, calendar properties,
  ISO output, and English formatting

Three additive batch functions keep FFI overhead out of compute-heavy workloads:

```python
values = pendulum.parse_many([
    "2024-01-31T12:00:00Z",
    "2024-02-29T12:00:00+05:30",
])

shifted = pendulum.add_many(values, months=2, days=3)
encoded = pendulum.to_iso8601_many(shifted)
```

The batch parser accepts ISO datetimes. Natural-language parsing, non-English locales,
Pendulum's complete formatting-token grammar, human-relative phrases, test-time travel,
custom week boundaries, ISO intervals, and every timezone ambiguity/non-existence policy
are outside this port. Scalar `parse(..., strict=False)` includes a small set of common
English calendar forms; it is not a general natural-language parser.

`DateTime`, `Date`, and `Time` subclass the corresponding standard-library types, so they
remain usable by ordinary Python code. The covered public behavior is checked against
Pendulum 3.2.0 rather than against a hand-written reference.

## Install and run

The repository pins the Mojo nightly used to build it:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-pendulum.so`. The Python bridge also builds it on
first native use if the file is missing or stale. A prebuilt library can be supplied with
`MOJO_PENDULUM_LIB=/absolute/path/libmojo-pendulum.so`.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30GHz, Linux x86-64,
Python 3.13.14. Times are the best of three runs and include construction of the returned
Python objects or strings. Every case checks its result against Pendulum 3.2 before
reporting a timing.

| case | mojo-pendulum | Pendulum 3.2 | ratio |
| --- | ---: | ---: | ---: |
| parse scalar loop (25k) | 132.79 ms | 188.68 ms | 1.42x faster |
| `parse_many` (100k) | 334.76 ms | 749.49 ms | 2.24x faster |
| add scalar loop (25k) | 94.27 ms | 163.55 ms | 1.73x faster |
| `add_many` (100k) | 347.02 ms | 613.20 ms | 1.77x faster |
| `to_iso8601_many` (100k) | 145.63 ms | 367.13 ms | 2.52x faster |

Scalar parsing passes encoded bytes directly and reuses thread-local result storage. Naive
scalar datetime arithmetic avoids native interchange storage, while batch arithmetic stays
in Mojo.

## How it works

All native kernels live in one Mojo compilation unit and export a small C ABI. Python owns
every allocation. Buffer pointers cross the ABI as 64-bit integer addresses and are
reconstructed as mutable `UnsafePointer` values inside non-parametric exported functions.

Parsing concatenates ASCII inputs into one byte buffer and passes an `int64` offset array.
Each result occupies nine `int64` fields: year through microsecond, UTC offset seconds, and
an offset-present flag. Arithmetic uses contiguous seven-field `int64` rows. ISO formatting
writes fixed-stride 40-byte rows plus an `int64` length array. No Mojo allocation crosses
the FFI boundary, so there is no cross-runtime ownership or release protocol.

The parser clears result rows with architecture-width SIMD stores and a scalar tail.
Arithmetic batches stay serial below 4,096 items and split larger independent workloads
across physical CPU cores. Parsing and formatting remain serial because their lighter
native work did not recover thread-runtime overhead.

Calendar conversion uses proleptic-Gregorian ordinals and a constant-time civil-date
transform. Month and year changes are applied before fixed day/time changes, with the day
clamped to the destination month's length. Python's `zoneinfo` handles IANA transition
data; calendar-day changes preserve wall time, while hour/minute/second-only changes use
elapsed UTC time to match Pendulum across DST.

No GPU path is included. Parsing, calendar arithmetic, and ISO formatting all move more
than a byte per simple integer operation, well below the arithmetic intensity needed to
recover device-transfer and launch overhead.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The benchmark task takes a machine-wide file lock. Run it through Pixi so concurrent jobs
do not distort the published numbers.

MIT licensed.
