"""Benchmarks against Pendulum 3.2. Run only through ``pixi run bench``."""

from __future__ import annotations

import os
import platform
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import pendulum as reference  # noqa: E402

import mojo_pendulum as mojo  # noqa: E402


def best_time(fn, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def signature(value):
    offset = value.utcoffset()
    return (
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
        None if offset is None else int(offset.total_seconds()),
    )


def signatures(values):
    return [signature(value) for value in values]


def main():
    iso_source = [
        f"{2000 + index % 24:04d}-{index % 12 + 1:02d}-"
        f"{index % 27 + 1:02d}T{index % 24:02d}:"
        f"{index % 60:02d}:{index * 7 % 60:02d}.{index % 1000000:06d}Z"
        for index in range(100_000)
    ]
    mojo_dates = [
        mojo.naive(1990 + index % 30, index % 12 + 1, index % 27 + 1, index % 24)
        for index in range(100_000)
    ]
    reference_dates = [
        reference.naive(
            value.year, value.month, value.day, value.hour
        )
        for value in mojo_dates
    ]
    mojo_formatted = mojo.parse_many(iso_source)
    reference_formatted = [reference.parse(value) for value in iso_source]

    cases = [
        (
            "parse scalar loop (25k)",
            lambda: [mojo.parse(value) for value in iso_source[:25_000]],
            lambda: [reference.parse(value) for value in iso_source[:25_000]],
            signatures,
        ),
        (
            "parse_many (100k)",
            lambda: mojo.parse_many(iso_source),
            lambda: [reference.parse(value) for value in iso_source],
            signatures,
        ),
        (
            "add scalar loop (25k)",
            lambda: [
                value.add(months=14, days=9, hours=3)
                for value in mojo_dates[:25_000]
            ],
            lambda: [
                value.add(months=14, days=9, hours=3)
                for value in reference_dates[:25_000]
            ],
            signatures,
        ),
        (
            "add_many (100k)",
            lambda: mojo.add_many(mojo_dates, months=14, days=9, hours=3),
            lambda: [
                value.add(months=14, days=9, hours=3)
                for value in reference_dates
            ],
            signatures,
        ),
        (
            "to_iso8601_many (100k)",
            lambda: mojo.to_iso8601_many(mojo_formatted),
            lambda: [value.to_iso8601_string() for value in reference_formatted],
            lambda value: value,
        ),
    ]

    print(
        f"Machine: {cpu_name()} "
        f"({platform.system()} {platform.machine()}, Python {platform.python_version()})"
    )
    print()
    print("| case | mojo-pendulum | Pendulum 3.2 | ratio |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, theirs, normalize in cases:
        ours()
        theirs()
        mojo_seconds, mojo_result = best_time(ours)
        python_seconds, python_result = best_time(theirs)
        assert normalize(mojo_result) == normalize(python_result)
        ratio = python_seconds / mojo_seconds
        label = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{python_seconds * 1000:.2f} ms | {ratio:.2f}x {label} |"
        )


if __name__ == "__main__":
    main()

