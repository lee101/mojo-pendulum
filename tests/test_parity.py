"""Behavioral parity checks against Pendulum 3.x."""

from __future__ import annotations

import datetime as dt
import inspect
from zoneinfo import ZoneInfoNotFoundError

import numpy as np
import pendulum as upstream
import pytest

import mojo_pendulum as mojo


def fields(value):
    result = (
        value.year,
        value.month,
        value.day,
        getattr(value, "hour", None),
        getattr(value, "minute", None),
        getattr(value, "second", None),
        getattr(value, "microsecond", None),
    )
    if isinstance(value, dt.datetime):
        offset = value.utcoffset()
        result += (None if offset is None else int(offset.total_seconds()),)
    return result


@pytest.mark.parametrize(
    "args",
    [
        (2024, 2, 29),
        (2024, 2, 29, 12, 34, 56, 123456),
        (1999, 12, 31, 23, 59, 59, 999999),
    ],
)
def test_datetime_factory_matches_upstream(args):
    assert fields(mojo.datetime(*args)) == fields(upstream.datetime(*args))


@pytest.mark.parametrize("tz", [None, "UTC", "Europe/Paris", 2.5, -4])
def test_datetime_timezone_factory_matches_upstream(tz):
    ours = mojo.datetime(2024, 1, 2, 3, 4, 5, 6, tz=tz)
    theirs = upstream.datetime(2024, 1, 2, 3, 4, 5, 6, tz=tz)
    assert fields(ours) == fields(theirs)
    assert ours.to_iso8601_string() == theirs.to_iso8601_string()


def test_date_and_time_factories_match_upstream():
    assert fields(mojo.date(2024, 2, 29)) == fields(upstream.date(2024, 2, 29))
    ours = mojo.time(23, 4, 5, 6)
    theirs = upstream.time(23, 4, 5, 6)
    assert (ours.hour, ours.minute, ours.second, ours.microsecond) == (
        theirs.hour,
        theirs.minute,
        theirs.second,
        theirs.microsecond,
    )


@pytest.mark.parametrize(
    "text",
    [
        "2024",
        "2024-02",
        "2024-02-29",
        "20240229",
        "2024-060",
        "2024060",
        "2024-W09-4",
        "2024W094",
        "2024-02-29T12:34:56",
        "2024-02-29 12:34:56.123456Z",
        "20240229T123456+0530",
    ],
)
def test_parse_iso_matches_upstream(text):
    ours = mojo.parse(text)
    theirs = upstream.parse(text)
    assert fields(ours) == fields(theirs)
    assert ours.to_iso8601_string() == theirs.to_iso8601_string()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not-a-date",
        "2023-02-29",
        "2024-13-01",
        "2024-01-01T24:00:00Z",
        "2024-01-01T25:00:00",
        "-P2DT3H",
    ],
)
def test_parse_rejects_what_upstream_rejects(text):
    with pytest.raises(Exception):
        upstream.parse(text)
    with pytest.raises(mojo.ParserError):
        mojo.parse(text)


def test_parse_non_strict_common_date_matches_upstream():
    ours = mojo.parse("March 5, 2024", strict=False)
    theirs = upstream.parse("March 5, 2024", strict=False)
    assert fields(ours) == fields(theirs)


def test_parse_exact_date_and_time_types_match_upstream():
    reference_now = dt.datetime(2020, 1, 2, tzinfo=dt.timezone.utc)
    for text in ("2024-02-29", "12:34:56", "2024-02-29T12:34:56Z"):
        ours = mojo.parse(text, exact=True, now=reference_now)
        theirs = upstream.parse(text, exact=True, now=reference_now)
        assert type(ours).__name__ == type(theirs).__name__
        if isinstance(ours, dt.time):
            assert ours == theirs
        else:
            assert fields(ours) == fields(theirs)


def test_parse_time_uses_supplied_now_date():
    reference_now = dt.datetime(2020, 1, 2, tzinfo=dt.timezone.utc)
    ours = mojo.parse("12:34:56", now=reference_now)
    theirs = upstream.parse("12:34:56", now=reference_now)
    assert fields(ours) == fields(theirs)


@pytest.mark.parametrize(
    "text",
    ["P1Y2M3DT4H5M6S", "P3W", "PT1.5S"],
)
def test_parse_duration_matches_upstream(text):
    ours = mojo.parse(text)
    theirs = upstream.parse(text)
    assert ours.total_seconds() == pytest.approx(theirs.total_seconds())
    assert (ours.years, ours.months, ours.weeks, ours.remaining_days) == (
        theirs.years,
        theirs.months,
        theirs.weeks,
        theirs.remaining_days,
    )


def test_from_format_matches_upstream():
    cases = [
        ("2024-02-29 13:05:06", "YYYY-MM-DD HH:mm:ss"),
        ("February 29, 2024 01:05 PM", "MMMM DD, YYYY hh:mm A"),
        ("2024-02-29T13:05:06+0530", "YYYY-MM-DDTHH:mm:ssZZ"),
    ]
    for text, fmt in cases:
        assert fields(mojo.from_format(text, fmt)) == fields(
            upstream.from_format(text, fmt)
        )


@pytest.mark.parametrize(
    "year,month,day,delta",
    [
        (2024, 1, 31, {"months": 1}),
        (2023, 1, 31, {"months": 1}),
        (2024, 3, 31, {"months": -1}),
        (2024, 2, 29, {"years": 1}),
        (2024, 1, 1, {"weeks": 5, "days": 3}),
        (1, 1, 1, {"days": 1}),
    ],
)
def test_date_add_matches_upstream(year, month, day, delta):
    ours = mojo.date(year, month, day).add(**delta)
    theirs = upstream.date(year, month, day).add(**delta)
    assert fields(ours) == fields(theirs)


@pytest.mark.parametrize(
    "delta",
    [
        {"years": 1, "months": 2, "weeks": 3, "days": 4},
        {"hours": 25, "minutes": 2, "seconds": 3.5, "microseconds": 4},
        {"months": -14, "days": -40, "microseconds": -1},
    ],
)
def test_naive_datetime_add_matches_upstream(delta):
    ours = mojo.naive(2024, 1, 31, 23, 59, 58, 900000).add(**delta)
    theirs = upstream.naive(2024, 1, 31, 23, 59, 58, 900000).add(**delta)
    assert fields(ours) == fields(theirs)


def test_dst_calendar_day_and_absolute_hours_match_upstream():
    ours = mojo.datetime(2024, 3, 30, 12, tz="Europe/Paris")
    theirs = upstream.datetime(2024, 3, 30, 12, tz="Europe/Paris")
    assert ours.add(days=1).to_iso8601_string() == theirs.add(
        days=1
    ).to_iso8601_string()
    assert ours.add(hours=24).to_iso8601_string() == theirs.add(
        hours=24
    ).to_iso8601_string()


def test_subtract_matches_upstream():
    ours = mojo.datetime(2024, 3, 31, 12).subtract(months=1, days=2, hours=3)
    theirs = upstream.datetime(2024, 3, 31, 12).subtract(
        months=1, days=2, hours=3
    )
    assert fields(ours) == fields(theirs)


@pytest.mark.parametrize(
    "unit", ["year", "month", "week", "day", "hour", "minute", "second"]
)
def test_datetime_start_and_end_of_match_upstream(unit):
    ours = mojo.datetime(2024, 5, 15, 13, 14, 15, 123456)
    theirs = upstream.datetime(2024, 5, 15, 13, 14, 15, 123456)
    assert fields(ours.start_of(unit)) == fields(theirs.start_of(unit))
    assert fields(ours.end_of(unit)) == fields(theirs.end_of(unit))


@pytest.mark.parametrize("unit", ["year", "month", "week", "day"])
def test_date_start_and_end_of_match_upstream(unit):
    ours = mojo.date(2024, 5, 15)
    theirs = upstream.date(2024, 5, 15)
    assert fields(ours.start_of(unit)) == fields(theirs.start_of(unit))
    assert fields(ours.end_of(unit)) == fields(theirs.end_of(unit))


def test_calendar_properties_match_upstream():
    ours = mojo.datetime(2024, 2, 29, 1, 2, 3)
    theirs = upstream.datetime(2024, 2, 29, 1, 2, 3)
    for name in (
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "days_in_month",
        "quarter",
        "int_timestamp",
        "float_timestamp",
        "offset",
        "offset_hours",
    ):
        assert getattr(ours, name) == getattr(theirs, name)
    assert ours.is_leap_year() == theirs.is_leap_year()


def test_timezone_conversion_matches_upstream():
    ours = mojo.datetime(2024, 7, 1, 12, tz="UTC").in_timezone("America/New_York")
    theirs = upstream.datetime(2024, 7, 1, 12, tz="UTC").in_timezone(
        "America/New_York"
    )
    assert fields(ours) == fields(theirs)
    assert ours.timezone_name == theirs.timezone_name


@pytest.mark.parametrize(
    "fmt",
    [
        "YYYY-MM-DD HH:mm:ss",
        "dddd, MMMM D, YYYY",
        "X",
        "x",
        "ZZ",
        "Z",
        "z",
        "Do",
        "Q",
        "DDD",
        "SSS",
        "SSSSSS",
    ],
)
def test_format_matches_upstream(fmt):
    ours = mojo.datetime(2024, 2, 29, 13, 5, 6, 123456, tz="Europe/Paris")
    theirs = upstream.datetime(2024, 2, 29, 13, 5, 6, 123456, tz="Europe/Paris")
    assert ours.format(fmt) == theirs.format(fmt)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"days": 1, "seconds": 2},
        {"years": 1},
        {"months": 1},
        {
            "years": 1,
            "months": 2,
            "weeks": 3,
            "days": 4,
            "hours": 5,
            "minutes": 6,
            "seconds": 7,
            "microseconds": 8,
        },
        {"seconds": -2},
        {"seconds": 1.5},
    ],
)
def test_duration_components_match_upstream(kwargs):
    ours = mojo.duration(**kwargs)
    theirs = upstream.duration(**kwargs)
    for name in (
        "days",
        "seconds",
        "microseconds",
        "years",
        "months",
        "weeks",
        "remaining_days",
        "hours",
        "minutes",
        "remaining_seconds",
    ):
        assert getattr(ours, name) == getattr(theirs, name)
    assert ours.total_seconds() == pytest.approx(theirs.total_seconds())
    assert ours.in_words() == theirs.in_words()


@pytest.mark.parametrize(
    "left,right",
    [
        ((2024, 1, 31), (2024, 3, 1)),
        ((2023, 1, 31), (2024, 3, 2)),
        ((2025, 3, 2), (2024, 1, 31)),
    ],
)
@pytest.mark.parametrize("absolute", [False, True])
def test_date_interval_matches_upstream(left, right, absolute):
    ours = mojo.interval(mojo.date(*left), mojo.date(*right), absolute=absolute)
    theirs = upstream.interval(
        upstream.date(*left), upstream.date(*right), absolute=absolute
    )
    for name in ("years", "months", "weeks", "remaining_days", "days", "invert"):
        assert getattr(ours, name) == getattr(theirs, name)
    assert ours.total_seconds() == theirs.total_seconds()
    assert ours.in_words() == theirs.in_words()


def test_datetime_interval_across_dst_matches_upstream():
    ours = mojo.interval(
        mojo.datetime(2024, 3, 30, 12, tz="Europe/Paris"),
        mojo.datetime(2024, 4, 1, 12, tz="Europe/Paris"),
    )
    theirs = upstream.interval(
        upstream.datetime(2024, 3, 30, 12, tz="Europe/Paris"),
        upstream.datetime(2024, 4, 1, 12, tz="Europe/Paris"),
    )
    assert ours.total_seconds() == theirs.total_seconds()
    assert (ours.years, ours.months, ours.remaining_days, ours.hours) == (
        theirs.years,
        theirs.months,
        theirs.remaining_days,
        theirs.hours,
    )


def test_interval_range_matches_upstream():
    ours = list(
        mojo.interval(mojo.date(2024, 1, 31), mojo.date(2024, 5, 31)).range(
            "months"
        )
    )
    theirs = list(
        upstream.interval(
            upstream.date(2024, 1, 31), upstream.date(2024, 5, 31)
        ).range("months")
    )
    assert list(map(fields, ours)) == list(map(fields, theirs))


def test_time_arithmetic_and_diff_match_upstream():
    ours = mojo.time(23, 59, 30, 500000)
    theirs = upstream.time(23, 59, 30, 500000)
    assert ours.add(minutes=2, seconds=1).isoformat() == theirs.add(
        minutes=2, seconds=1
    ).isoformat()
    other_ours = mojo.time(1, 2, 3)
    other_theirs = upstream.time(1, 2, 3)
    assert ours.diff(other_ours, abs=False).total_seconds() == theirs.diff(
        other_theirs, abs=False
    ).total_seconds()


def test_instance_matches_upstream_for_stdlib_types():
    values = [
        dt.datetime(2024, 2, 29, 12, 30),
        dt.date(2024, 2, 29),
        dt.time(12, 30),
    ]
    for value in values:
        ours = mojo.instance(value)
        theirs = upstream.instance(value)
        assert type(ours).__name__ == type(theirs).__name__
        if isinstance(value, dt.time):
            assert ours == theirs
        else:
            assert fields(ours) == fields(theirs)


def test_public_factory_signatures_track_upstream():
    for name in ("datetime", "date", "time", "duration", "interval", "from_format"):
        ours = inspect.signature(getattr(mojo, name))
        theirs = inspect.signature(getattr(upstream, name))
        assert list(ours.parameters) == list(theirs.parameters)


def test_parse_many_matches_upstream_on_large_mixed_batch():
    texts = [
        (
            f"{2000 + index % 24:04d}-{index % 12 + 1:02d}-"
            f"{index % 27 + 1:02d}T{index % 24:02d}:"
            f"{index % 60:02d}:{index * 7 % 60:02d}Z"
        )
        for index in range(5000)
    ]
    ours = mojo.parse_many(texts)
    theirs = [upstream.parse(text) for text in texts]
    assert list(map(fields, ours)) == list(map(fields, theirs))


def test_add_many_matches_upstream():
    values = [
        mojo.naive(1990 + index % 30, index % 12 + 1, index % 27 + 1, index % 24)
        for index in range(4000)
    ]
    ours = mojo.add_many(values, months=14, days=9, hours=3)
    theirs = [
        upstream.naive(
            value.year, value.month, value.day, value.hour
        ).add(months=14, days=9, hours=3)
        for value in values
    ]
    assert list(map(fields, ours)) == list(map(fields, theirs))


def test_to_iso8601_many_matches_upstream():
    values = [
        mojo.datetime(2024, 1, 1, 0, 0, tz="UTC"),
        mojo.datetime(2024, 2, 29, 12, 30, 5, 123456, tz=5.5),
        mojo.naive(1999, 12, 31, 23, 59, 59),
    ]
    ours = mojo.to_iso8601_many(values)
    references = [
        upstream.datetime(2024, 1, 1, 0, 0, tz="UTC"),
        upstream.datetime(2024, 2, 29, 12, 30, 5, 123456, tz=5.5),
        upstream.naive(1999, 12, 31, 23, 59, 59),
    ]
    assert ours == [value.to_iso8601_string() for value in references]


def test_timezone_lookup_and_unknown_zone_match_upstream():
    assert str(mojo.timezone("Europe/Paris")) == str(
        upstream.timezone("Europe/Paris")
    )
    with pytest.raises((ZoneInfoNotFoundError, upstream.tz.exceptions.InvalidTimezone)):
        mojo.timezone("Not/A_Real_Zone")


def test_batch_empty_inputs():
    assert mojo.parse_many([]) == []
    assert mojo.add_many([]) == []
    assert mojo.to_iso8601_many([]) == []


def test_parse_many_generator_reports_the_failing_item():
    values = (value for value in ("2024-01-01", "not-a-date"))
    with pytest.raises(
        mojo.ParserError, match=r"index 1: not-a-date"
    ):
        mojo.parse_many(values)


def test_batch_arithmetic_rejects_silent_i64_narrowing():
    values = [mojo.naive(2024, 1, 1)]
    with pytest.raises(OverflowError, match="months"):
        mojo.add_many(values, months=1 << 64)
    with pytest.raises(OverflowError, match=r"years \* 12"):
        mojo.add_many(values, years=1 << 62)
    with pytest.raises(TypeError, match="days"):
        mojo.add_many(values, days=1.5)


def test_scalar_native_arithmetic_rejects_intermediate_overflow():
    with pytest.raises(OverflowError, match=r"weeks \* 7"):
        mojo.date(2024, 1, 1).add(weeks=1 << 62)


def test_relative_date_helpers_and_local_factory():
    local_zone = dt.timezone(dt.timedelta(hours=2))
    previous = mojo.local_timezone()
    mojo.set_local_timezone(local_zone)
    try:
        current = mojo.now()
        assert current.tzinfo == local_zone
        assert mojo.local(2024, 2, 29, 12).tzinfo == local_zone
        assert mojo.today().hour == mojo.today().minute == 0
        assert mojo.tomorrow().date() == mojo.today().add(days=1).date()
        assert mojo.yesterday().date() == mojo.today().subtract(days=1).date()
    finally:
        mojo.set_local_timezone(previous)


@pytest.mark.parametrize("count", [4095, 4096])
def test_batch_parallel_threshold_and_simd_tail(count):
    text = "2024-02-29T23:59:58.123456"
    parsed = mojo.parse_many([text] * count, tz=None)
    assert len(parsed) == count
    assert fields(parsed[0]) == fields(parsed[-1])

    shifted = mojo.add_many(parsed, days=1)
    encoded = mojo.to_iso8601_many(shifted)
    assert len(encoded) == count
    assert encoded[0] == encoded[-1] == "2024-03-01T23:59:58.123456"
