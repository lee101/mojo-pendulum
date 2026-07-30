"""Pendulum-shaped date and time objects backed by Mojo calendar kernels."""

from __future__ import annotations

import calendar
import builtins
import ctypes
import datetime as _dt
import math
import re
import threading
from collections.abc import Iterable, Iterator, Sequence
from enum import IntEnum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

import numpy as np

from ._lib import addr, i64, lib


class ParserError(ValueError):
    pass


class WeekDay(IntEnum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


MONDAY = WeekDay.MONDAY
TUESDAY = WeekDay.TUESDAY
WEDNESDAY = WeekDay.WEDNESDAY
THURSDAY = WeekDay.THURSDAY
FRIDAY = WeekDay.FRIDAY
SATURDAY = WeekDay.SATURDAY
SUNDAY = WeekDay.SUNDAY

UTC = _dt.timezone.utc
Timezone = ZoneInfo
FixedTimezone = _dt.timezone
_LOCAL_TIMEZONE: _dt.tzinfo | None = None
_LOCALE = "en"
_PARSE_FIELDS = None
_ADD_FIELDS = None
_PARSE_MANY = None
_ADD_MANY = None
_FORMAT_MANY = None
_SCALAR_BUFFERS = threading.local()


def _native_deltas(
    years: int,
    months: int,
    weeks: int,
    days: int,
    hours: int,
    minutes: int,
    seconds: int,
    microseconds: int,
) -> tuple[int, ...]:
    values = tuple(
        i64(value, name)
        for value, name in (
            (years, "years"),
            (months, "months"),
            (weeks, "weeks"),
            (days, "days"),
            (hours, "hours"),
            (minutes, "minutes"),
            (seconds, "seconds"),
            (microseconds, "microseconds"),
        )
    )
    years, months, weeks, days, hours, minutes, seconds, microseconds = values
    # Check every expression that the native kernel evaluates in signed Int64.
    year_months = i64(years * 12, "years * 12")
    month_delta = i64(year_months + months, "total month delta")
    i64(month_delta + 120_000, "month index")
    i64(month_delta - 120_000, "month index")
    week_days = i64(weeks * 7, "weeks * 7")
    day_delta = i64(week_days + days, "total day delta")
    i64(day_delta + 3_700_000, "ordinal delta")
    i64(day_delta - 3_700_000, "ordinal delta")
    hour_minutes = i64(hours * 60, "hours * 60")
    total_minutes = i64(hour_minutes + minutes, "total minutes")
    total_seconds = i64(total_minutes * 60, "total minutes * 60")
    total_seconds = i64(total_seconds + seconds, "total seconds")
    total_micros = i64(total_seconds * 1_000_000, "total seconds * 1000000")
    total_micros = i64(total_micros + microseconds, "total microseconds")
    i64(total_micros + 86_400_000_000, "day microseconds")
    i64(total_micros - 86_400_000_000, "day microseconds")
    return values


def fixed_timezone(offset: int) -> _dt.timezone:
    if offset == 0:
        return UTC
    return _dt.timezone(_dt.timedelta(seconds=int(offset)))


def local_timezone() -> _dt.tzinfo:
    if _LOCAL_TIMEZONE is not None:
        return _LOCAL_TIMEZONE
    return _dt.datetime.now().astimezone().tzinfo or UTC


def set_local_timezone(tz: _dt.tzinfo) -> None:
    global _LOCAL_TIMEZONE
    _LOCAL_TIMEZONE = tz


def timezone(name: str | int) -> _dt.tzinfo:
    if isinstance(name, int):
        return fixed_timezone(name)
    if name.lower() == "utc":
        return UTC
    return ZoneInfo(name)


def timezones() -> set[str]:
    return available_timezones()


def _safe_timezone(
    value: str | float | _dt.tzinfo | None,
    dt: _dt.datetime | None = None,
) -> _dt.tzinfo:
    if value is None or value == "local":
        return local_timezone()
    if isinstance(value, (int, float)):
        return fixed_timezone(int(value * 3600))
    if isinstance(value, str):
        return timezone(value)
    if isinstance(value, _dt.tzinfo):
        return value
    raise TypeError(f"invalid timezone: {value!r}")


def set_locale(locale: str) -> None:
    if locale != "en":
        raise ValueError("only the English locale is included")
    global _LOCALE
    _LOCALE = locale


def get_locale() -> str:
    return _LOCALE


def _split_seconds(seconds: float, microseconds: int) -> tuple[int, int]:
    if type(seconds) is int and type(microseconds) is int:
        if 0 <= microseconds < 1_000_000:
            return seconds, microseconds
        carry, micros = divmod(microseconds, 1_000_000)
        return seconds + carry, micros
    whole = math.floor(seconds)
    micros = round((seconds - whole) * 1_000_000) + microseconds
    carry, micros = divmod(micros, 1_000_000)
    return int(whole + carry), int(micros)


def _native_add(
    value: _dt.date | _dt.datetime,
    years: int = 0,
    months: int = 0,
    weeks: int = 0,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: float = 0,
    microseconds: int = 0,
) -> np.ndarray:
    global _ADD_FIELDS
    if _ADD_FIELDS is None:
        _ADD_FIELDS = lib().mp_add_datetime
    second, micros = _split_seconds(seconds, microseconds)
    deltas = _native_deltas(
        years, months, weeks, days, hours, minutes, second, micros
    )
    try:
        source, result = _SCALAR_BUFFERS.add
    except AttributeError:
        source = (ctypes.c_int64 * 7)()
        result = (ctypes.c_int64 * 7)()
        _SCALAR_BUFFERS.add = source, result
    source[0], source[1], source[2] = value.year, value.month, value.day
    if isinstance(value, _dt.datetime):
        source[3], source[4], source[5], source[6] = (
            value.hour,
            value.minute,
            value.second,
            value.microsecond,
        )
    else:
        source[3], source[4], source[5], source[6] = 0, 0, 0, 0
    status = _ADD_FIELDS(
        ctypes.addressof(source),
        ctypes.addressof(result),
        *deltas,
    )
    if status:
        raise OverflowError("date arithmetic moved outside years 1..9999")
    return result


_FORMAT_TOKEN = re.compile(
    "|".join(
        map(
            re.escape,
            sorted(
                (
                    "YYYY",
                    "SSSSSS",
                    "MMMM",
                    "dddd",
                    "SSS",
                    "MMM",
                    "ddd",
                    "DDD",
                    "ZZ",
                    "YY",
                    "MM",
                    "DD",
                    "HH",
                    "hh",
                    "mm",
                    "ss",
                    "Do",
                    "X",
                    "x",
                    "Z",
                    "z",
                    "Q",
                    "M",
                    "D",
                    "H",
                    "h",
                    "m",
                    "s",
                    "A",
                    "a",
                ),
                key=len,
                reverse=True,
            ),
        )
    )
)


def _ordinal_suffix(day: int) -> str:
    if 10 < day % 100 < 14:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def _format(value: _dt.date | _dt.time, fmt: str, locale: str | None) -> str:
    if locale not in (None, "en"):
        raise ValueError("only the English locale is included")

    def token(match: re.Match[str]) -> str:
        text = match.group(0)
        mappings = {
            "YYYY": lambda: f"{value.year:04d}",
            "YY": lambda: f"{value.year % 100:02d}",
            "MMMM": lambda: calendar.month_name[value.month],
            "MMM": lambda: calendar.month_abbr[value.month],
            "MM": lambda: f"{value.month:02d}",
            "M": lambda: str(value.month),
            "DD": lambda: f"{value.day:02d}",
            "D": lambda: str(value.day),
            "Do": lambda: f"{value.day}{_ordinal_suffix(value.day)}",
            "dddd": lambda: calendar.day_name[value.weekday()],
            "ddd": lambda: calendar.day_abbr[value.weekday()],
            "DDD": lambda: str(value.timetuple().tm_yday),
            "Q": lambda: str((value.month - 1) // 3 + 1),
            "HH": lambda: f"{value.hour:02d}",
            "H": lambda: str(value.hour),
            "hh": lambda: f"{value.hour % 12 or 12:02d}",
            "h": lambda: str(value.hour % 12 or 12),
            "mm": lambda: f"{value.minute:02d}",
            "m": lambda: str(value.minute),
            "ss": lambda: f"{value.second:02d}",
            "s": lambda: str(value.second),
            "SSSSSS": lambda: f"{value.microsecond:06d}",
            "SSS": lambda: f"{value.microsecond // 1000:03d}",
            "A": lambda: "AM" if value.hour < 12 else "PM",
            "a": lambda: "am" if value.hour < 12 else "pm",
            "X": lambda: str(int(value.timestamp())),
            "x": lambda: str(int(value.timestamp() * 1_000)),
            "ZZ": lambda: value.strftime("%z"),
            "Z": lambda: (
                value.strftime("%z")[:3] + ":" + value.strftime("%z")[3:]
                if value.strftime("%z")
                else ""
            ),
            "z": lambda: getattr(value.tzinfo, "key", None) or value.tzname() or "",
        }
        return mappings[text]()

    return _FORMAT_TOKEN.sub(token, fmt)


class Date(_dt.date):
    @property
    def day_of_week(self) -> WeekDay:
        return WeekDay(self.weekday())

    @property
    def day_of_year(self) -> int:
        return self.timetuple().tm_yday

    @property
    def week_of_year(self) -> int:
        return self.isocalendar().week

    @property
    def days_in_month(self) -> int:
        return calendar.monthrange(self.year, self.month)[1]

    @property
    def week_of_month(self) -> int:
        return math.ceil((self.day + self.replace(day=1).isoweekday() - 1) / 7)

    @property
    def quarter(self) -> int:
        return (self.month - 1) // 3 + 1

    @property
    def age(self) -> int:
        return self.diff(abs=False).in_years()

    def set(
        self,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> Date:
        return self.replace(year=year, month=month, day=day)

    def add(
        self, years: int = 0, months: int = 0, weeks: int = 0, days: int = 0
    ) -> Date:
        result = _native_add(self, years, months, weeks, days)
        return self.__class__(*map(int, result[:3]))

    def subtract(
        self, years: int = 0, months: int = 0, weeks: int = 0, days: int = 0
    ) -> Date:
        return self.add(-years, -months, -weeks, -days)

    def diff(self, dt: _dt.date | None = None, abs: bool = True) -> Interval:
        if dt is None:
            dt = self.today()
        return Interval(self, Date(dt.year, dt.month, dt.day), absolute=abs)

    def start_of(self, unit: str) -> Date:
        if unit == "day":
            return self
        if unit == "week":
            return self.subtract(days=self.weekday())
        if unit == "month":
            return self.replace(day=1)
        if unit == "year":
            return self.replace(month=1, day=1)
        if unit == "decade":
            return self.replace(year=self.year - self.year % 10, month=1, day=1)
        if unit == "century":
            return self.replace(
                year=(self.year - 1) // 100 * 100 + 1, month=1, day=1
            )
        raise ValueError(f'Invalid unit "{unit}" for start_of()')

    def end_of(self, unit: str) -> Date:
        if unit == "day":
            return self
        starts = {
            "week": lambda: self.start_of("week").add(weeks=1),
            "month": lambda: self.start_of("month").add(months=1),
            "year": lambda: self.start_of("year").add(years=1),
            "decade": lambda: self.start_of("decade").add(years=10),
            "century": lambda: self.start_of("century").add(years=100),
        }
        if unit not in starts:
            raise ValueError(f'Invalid unit "{unit}" for end_of()')
        return starts[unit]().subtract(days=1)

    def format(self, fmt: str, locale: str | None = None) -> str:
        return _format(self, fmt, locale)

    def to_date_string(self) -> str:
        return self.strftime("%Y-%m-%d")

    def to_formatted_date_string(self) -> str:
        return self.strftime("%b %d, %Y")

    def is_leap_year(self) -> bool:
        return calendar.isleap(self.year)

    def is_long_year(self) -> bool:
        return self.replace(month=12, day=28).isocalendar().week == 53

    def is_same_day(self, dt: _dt.date) -> bool:
        return self == dt

    def closest(self, dt1: _dt.date, dt2: _dt.date) -> Date:
        return self.__class__.instance(
            dt1 if abs(self.toordinal() - dt1.toordinal()) <
            abs(self.toordinal() - dt2.toordinal()) else dt2
        )

    def farthest(self, dt1: _dt.date, dt2: _dt.date) -> Date:
        return self.__class__.instance(
            dt1 if abs(self.toordinal() - dt1.toordinal()) >
            abs(self.toordinal() - dt2.toordinal()) else dt2
        )

    @classmethod
    def instance(cls, value: _dt.date) -> Date:
        return cls(value.year, value.month, value.day)

    def __add__(self, other):
        if not isinstance(other, _dt.timedelta):
            return NotImplemented
        if isinstance(other, Duration):
            return self.add(
                years=other.years,
                months=other.months,
                weeks=other.weeks,
                days=other.remaining_days,
            )
        return self.add(days=other.days)

    def __sub__(self, other):
        if isinstance(other, _dt.timedelta):
            return self.add(days=-other.days)
        if isinstance(other, _dt.date):
            return Date(other.year, other.month, other.day).diff(self, abs=False)
        return NotImplemented


class DateTime(_dt.datetime):
    @property
    def day_of_week(self) -> WeekDay:
        return WeekDay(self.weekday())

    @property
    def day_of_year(self) -> int:
        return self.timetuple().tm_yday

    @property
    def week_of_year(self) -> int:
        return self.isocalendar().week

    @property
    def days_in_month(self) -> int:
        return calendar.monthrange(self.year, self.month)[1]

    @property
    def quarter(self) -> int:
        return (self.month - 1) // 3 + 1

    @property
    def int_timestamp(self) -> int:
        return int(self.timestamp())

    @property
    def float_timestamp(self) -> float:
        return self.timestamp()

    @property
    def timezone_name(self) -> str | None:
        return getattr(self.tzinfo, "key", None) or (
            self.tzname() if self.tzinfo else None
        )

    @property
    def timezone(self) -> _dt.tzinfo | None:
        return self.tzinfo

    @property
    def offset(self) -> int:
        delta = self.utcoffset()
        return int(delta.total_seconds()) if delta else 0

    @property
    def offset_hours(self) -> float:
        return self.offset / 3600

    def set(self, **kwargs) -> DateTime:
        return self.replace(**{key: value for key, value in kwargs.items() if value is not None})

    def add(
        self,
        years: int = 0,
        months: int = 0,
        weeks: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: float = 0,
        microseconds: int = 0,
    ) -> DateTime:
        variable = years != 0 or months != 0 or weeks != 0 or days != 0
        if self.tzinfo is not None and not variable:
            delta = _dt.timedelta(
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                microseconds=microseconds,
            )
            base = _dt.datetime(
                self.year,
                self.month,
                self.day,
                self.hour,
                self.minute,
                self.second,
                self.microsecond,
                tzinfo=self.tzinfo,
                fold=self.fold,
            )
            shifted = (base.astimezone(UTC) + delta).astimezone(self.tzinfo)
            return self.__class__.instance(shifted)
        month_index = (
            (self.year - 1) * 12
            + self.month
            - 1
            + years * 12
            + months
        )
        year, month_zero = divmod(month_index, 12)
        year += 1
        month = month_zero + 1
        if year < 1 or year > 9999:
            raise OverflowError("date arithmetic moved outside years 1..9999")
        if month == 2:
            last_day = 29 if calendar.isleap(year) else 28
        elif month in (4, 6, 9, 11):
            last_day = 30
        else:
            last_day = 31
        base = self.replace(year=year, month=month, day=min(self.day, last_day))
        delta = _dt.timedelta(
            weeks=weeks,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            microseconds=microseconds,
        )
        try:
            return _dt.datetime.__add__(base, delta)
        except OverflowError as exc:
            raise OverflowError(
                "date arithmetic moved outside years 1..9999"
            ) from exc

    def subtract(
        self,
        years: int = 0,
        months: int = 0,
        weeks: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: float = 0,
        microseconds: int = 0,
    ) -> DateTime:
        return self.add(
            -years,
            -months,
            -weeks,
            -days,
            -hours,
            -minutes,
            -seconds,
            -microseconds,
        )

    def diff(self, dt: _dt.datetime | None = None, abs: bool = True) -> Interval:
        if dt is None:
            dt = now(self.tzinfo)
        return Interval(self, instance(dt), absolute=abs)

    def start_of(self, unit: str) -> DateTime:
        if unit == "year":
            return self.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        if unit == "month":
            return self.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if unit == "week":
            return self.subtract(days=self.weekday()).start_of("day")
        if unit == "day":
            return self.replace(hour=0, minute=0, second=0, microsecond=0)
        if unit == "hour":
            return self.replace(minute=0, second=0, microsecond=0)
        if unit == "minute":
            return self.replace(second=0, microsecond=0)
        if unit == "second":
            return self.replace(microsecond=0)
        if unit in ("decade", "century"):
            start = Date(self.year, self.month, self.day).start_of(unit)
            return self.replace(
                year=start.year,
                month=start.month,
                day=start.day,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        raise ValueError(f'Invalid unit "{unit}" for start_of()')

    def end_of(self, unit: str) -> DateTime:
        increments = {
            "year": {"years": 1},
            "month": {"months": 1},
            "week": {"weeks": 1},
            "day": {"days": 1},
            "hour": {"hours": 1},
            "minute": {"minutes": 1},
            "second": {"seconds": 1},
            "decade": {"years": 10},
            "century": {"years": 100},
        }
        if unit not in increments:
            raise ValueError(f'Invalid unit "{unit}" for end_of()')
        return self.start_of(unit).add(**increments[unit]).subtract(microseconds=1)

    def in_timezone(self, tz: str | _dt.tzinfo) -> DateTime:
        return self.__class__.instance(self.astimezone(_safe_timezone(tz)))

    def in_tz(self, tz: str | _dt.tzinfo) -> DateTime:
        return self.in_timezone(tz)

    def is_utc(self) -> bool:
        return self.offset == 0 and self.timezone_name in ("UTC", "UTC+00:00")

    def is_dst(self) -> bool:
        value = self.dst()
        return bool(value and value.total_seconds())

    def is_leap_year(self) -> bool:
        return calendar.isleap(self.year)

    def is_same_day(self, dt: _dt.datetime) -> bool:
        return (self.year, self.month, self.day) == (dt.year, dt.month, dt.day)

    def to_iso8601_string(self) -> str:
        text = self.isoformat()
        return text[:-6] + "Z" if text.endswith("+00:00") else text

    def to_atom_string(self) -> str:
        return self.isoformat()

    def to_date_string(self) -> str:
        return self.strftime("%Y-%m-%d")

    def to_time_string(self) -> str:
        return self.strftime("%H:%M:%S")

    def to_datetime_string(self) -> str:
        return self.strftime("%Y-%m-%d %H:%M:%S")

    def format(self, fmt: str, locale: str | None = None) -> str:
        return _format(self, fmt, locale)

    def date(self) -> Date:
        return Date(self.year, self.month, self.day)

    def time(self) -> Time:
        return Time(self.hour, self.minute, self.second, self.microsecond, fold=self.fold)

    def naive(self) -> DateTime:
        return self.replace(tzinfo=None)

    def at(
        self, hour: int, minute: int = 0, second: int = 0, microsecond: int = 0
    ) -> DateTime:
        return self.replace(
            hour=hour, minute=minute, second=second, microsecond=microsecond
        )

    @classmethod
    def instance(
        cls, value: _dt.datetime, tz: str | _dt.tzinfo | None = UTC
    ) -> DateTime:
        tzinfo = value.tzinfo if value.tzinfo is not None else (
            None if tz is None else _safe_timezone(tz)
        )
        return cls(
            value.year,
            value.month,
            value.day,
            value.hour,
            value.minute,
            value.second,
            value.microsecond,
            tzinfo=tzinfo,
            fold=value.fold,
        )

    def __add__(self, other):
        if not isinstance(other, _dt.timedelta):
            return NotImplemented
        if isinstance(other, Duration):
            return self.add(**other._signature)
        return self.add(seconds=other.total_seconds())

    def __sub__(self, other):
        if isinstance(other, _dt.timedelta):
            if isinstance(other, Duration):
                return self.subtract(**other._signature)
            return self.subtract(seconds=other.total_seconds())
        if isinstance(other, _dt.datetime):
            return instance(other).diff(self, abs=False)
        return NotImplemented


class Time(_dt.time):
    def add(
        self, hours: int = 0, minutes: int = 0, seconds: int = 0, microseconds: int = 0
    ) -> Time:
        total = (
            ((self.hour + hours) * 60 + self.minute + minutes) * 60
            + self.second
            + seconds
        ) * 1_000_000 + self.microsecond + microseconds
        total %= 86_400_000_000
        hour, rest = divmod(total, 3_600_000_000)
        minute, rest = divmod(rest, 60_000_000)
        second, micros = divmod(rest, 1_000_000)
        return self.__class__(hour, minute, second, micros, fold=self.fold)

    def subtract(
        self, hours: int = 0, minutes: int = 0, seconds: int = 0, microseconds: int = 0
    ) -> Time:
        return self.add(-hours, -minutes, -seconds, -microseconds)

    def diff(self, dt: _dt.time | None = None, abs: bool = True) -> Duration:
        if dt is None:
            dt = now().time()
        left = ((self.hour * 60 + self.minute) * 60 + self.second) * 1_000_000
        right = ((dt.hour * 60 + dt.minute) * 60 + dt.second) * 1_000_000
        delta = right - left
        if abs:
            delta = builtins.abs(delta)
        return Duration(microseconds=delta)

    def format(self, fmt: str, locale: str | None = None) -> str:
        return _format(self, fmt, locale)

    def __add__(self, other):
        if not isinstance(other, _dt.timedelta):
            return NotImplemented
        if other.days:
            raise TypeError("Cannot add timedelta with days to Time.")
        return self.add(seconds=other.seconds, microseconds=other.microseconds)

    def __sub__(self, other):
        if isinstance(other, _dt.timedelta):
            if other.days:
                raise TypeError("Cannot subtract timedelta with days from Time.")
            return self.subtract(seconds=other.seconds, microseconds=other.microseconds)
        if isinstance(other, _dt.time):
            return Time(other.hour, other.minute, other.second, other.microsecond).diff(
                self, abs=False
            )
        return NotImplemented


class Duration(_dt.timedelta):
    def __new__(
        cls,
        days: float = 0,
        seconds: float = 0,
        microseconds: float = 0,
        milliseconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
        weeks: float = 0,
        years: float = 0,
        months: float = 0,
    ):
        if not isinstance(years, int) or not isinstance(months, int):
            raise ValueError("Float year and months are not supported")
        self = super().__new__(
            cls,
            days + years * 365 + months * 30,
            seconds,
            microseconds,
            milliseconds,
            minutes,
            hours,
            weeks,
        )
        total = self.total_seconds() - (years * 365 + months * 30) * 86400
        sign = -1 if total < 0 else 1
        self._component_microseconds = round((total % sign) * 1_000_000)
        self._component_seconds = abs(int(total)) % 86400 * sign
        component_days = abs(int(total)) // 86400 * sign
        self._remaining_days = abs(component_days) % 7 * sign
        self._weeks = abs(component_days) // 7 * sign
        self._months = months
        self._years = years
        self._signature = {
            "years": years,
            "months": months,
            "weeks": weeks,
            "days": days,
            "hours": hours,
            "minutes": minutes,
            "seconds": seconds,
            "microseconds": microseconds + milliseconds * 1000,
        }
        return self

    @property
    def years(self) -> int:
        return self._years

    @property
    def months(self) -> int:
        return self._months

    @property
    def weeks(self) -> int:
        return self._weeks

    @property
    def remaining_days(self) -> int:
        return self._remaining_days

    @property
    def hours(self) -> int:
        return (abs(self._component_seconds) // 3600 % 24) * (
            -1 if self._component_seconds < 0 else 1
        )

    @property
    def minutes(self) -> int:
        return (abs(self._component_seconds) // 60 % 60) * (
            -1 if self._component_seconds < 0 else 1
        )

    @property
    def remaining_seconds(self) -> int:
        return abs(self._component_seconds) % 60 * (
            -1 if self._component_seconds < 0 else 1
        )

    @property
    def seconds(self) -> int:
        return self._component_seconds

    @property
    def microseconds(self) -> int:
        return self._component_microseconds

    @property
    def invert(self) -> bool:
        return self.total_seconds() < 0

    def total_minutes(self) -> float:
        return self.total_seconds() / 60

    def total_hours(self) -> float:
        return self.total_seconds() / 3600

    def total_days(self) -> float:
        return self.total_seconds() / 86400

    def total_weeks(self) -> float:
        return self.total_days() / 7

    def in_seconds(self) -> int:
        return int(self.total_seconds())

    def in_minutes(self) -> int:
        return int(self.total_minutes())

    def in_hours(self) -> int:
        return int(self.total_hours())

    def in_days(self) -> int:
        return int(self.total_days())

    def in_weeks(self) -> int:
        return int(self.total_weeks())

    def in_words(self, locale: str | None = None, separator: str = " ") -> str:
        if locale not in (None, "en"):
            raise ValueError("only the English locale is included")
        values = (
            ("year", self.years),
            ("month", self.months),
            ("week", self.weeks),
            ("day", self.remaining_days),
            ("hour", self.hours),
            ("minute", self.minutes),
            ("second", self.remaining_seconds),
        )
        parts = [
            f"{value} {unit}{'' if abs(value) == 1 else 's'}"
            for unit, value in values
            if value
        ]
        if not parts:
            if self.microseconds:
                parts.append(f"{abs(self.microseconds) / 1_000_000:.2f} seconds")
            else:
                parts.append("0 microseconds")
        return separator.join(parts)

    def __repr__(self) -> str:
        parts = []
        for name, value in (
            ("years", self.years),
            ("months", self.months),
            ("weeks", self.weeks),
            ("days", self.remaining_days),
            ("hours", self.hours),
            ("minutes", self.minutes),
            ("seconds", self.remaining_seconds),
            ("microseconds", self.microseconds),
        ):
            if value:
                parts.append(f"{name}={value}")
        return f"Duration({', '.join(parts)})"

    def __str__(self) -> str:
        return self.in_words()


def _base_datetime(value: _dt.datetime) -> _dt.datetime:
    return _dt.datetime(
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
        tzinfo=value.tzinfo,
        fold=value.fold,
    )


def _elapsed(end: _dt.date, start: _dt.date) -> _dt.timedelta:
    if isinstance(start, _dt.datetime) and isinstance(end, _dt.datetime):
        a, b = _base_datetime(start), _base_datetime(end)
        if a.tzinfo is not None:
            a = a.astimezone(UTC)
            b = b.astimezone(UTC)
        return _dt.datetime.__sub__(b, a)
    return _dt.date.__sub__(end, start)


def _calendar_components(
    start: Date | DateTime, end: Date | DateTime
) -> tuple[int, int, int, int, int, int]:
    sign = 1
    if start > end:
        start, end = end, start
        sign = -1
    years = end.year - start.year
    anchor = start.add(years=years)
    if anchor > end:
        years -= 1
        anchor = start.add(years=years)
    months = (end.year - anchor.year) * 12 + end.month - anchor.month
    anchor = anchor.add(months=months)
    if anchor > end:
        months -= 1
        anchor = start.add(years=years, months=months)
    if isinstance(start, _dt.datetime):
        remainder = _dt.datetime.__sub__(
            _base_datetime(end).replace(tzinfo=None),
            _base_datetime(anchor).replace(tzinfo=None),
        )
    else:
        remainder = _elapsed(end, anchor)
    days = remainder.days
    seconds = remainder.seconds
    micros = remainder.microseconds
    return (
        years * sign,
        months * sign,
        days * sign,
        (seconds // 3600) * sign,
        ((seconds % 3600) // 60) * sign,
        ((seconds % 60) * 1_000_000 + micros) * sign,
    )


class Interval(Duration):
    def __new__(
        cls, start: Date | DateTime, end: Date | DateTime, absolute: bool = False
    ):
        if isinstance(start, _dt.datetime) != isinstance(end, _dt.datetime):
            raise ValueError("Both start and end of an Interval must have the same type")
        if absolute and start > end:
            start, end = end, start
        delta = _elapsed(end, start)
        return super().__new__(cls, seconds=delta.total_seconds())

    def __init__(
        self, start: Date | DateTime, end: Date | DateTime, absolute: bool = False
    ):
        original_invert = start > end
        if absolute and start > end:
            start, end = end, start
        self._start = start
        self._end = end
        self._absolute = absolute
        self._interval_invert = original_invert
        (
            self._interval_years,
            self._interval_months,
            self._interval_days,
            self._interval_hours,
            self._interval_minutes,
            self._interval_microseconds,
        ) = _calendar_components(start, end)

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def years(self) -> int:
        return self._interval_years

    @property
    def months(self) -> int:
        return self._interval_months

    @property
    def weeks(self) -> int:
        return abs(self._interval_days) // 7 * (-1 if self._interval_days < 0 else 1)

    @property
    def remaining_days(self) -> int:
        return abs(self._interval_days) % 7 * (-1 if self._interval_days < 0 else 1)

    @property
    def hours(self) -> int:
        return self._interval_hours

    @property
    def minutes(self) -> int:
        return self._interval_minutes

    @property
    def remaining_seconds(self) -> int:
        return abs(self._interval_microseconds) // 1_000_000 % 60 * (
            -1 if self._interval_microseconds < 0 else 1
        )

    @property
    def invert(self) -> bool:
        return self._interval_invert

    def in_years(self) -> int:
        return self.years

    def in_months(self) -> int:
        return self.years * 12 + self.months

    def in_days(self) -> int:
        return int(self.total_seconds() / 86400)

    def range(self, unit: str, amount: int = 1) -> Iterator:
        if amount <= 0:
            raise ValueError("amount must be positive")
        current = self.start
        forward = not (not self._absolute and self.invert)
        step = amount
        while current <= self.end if forward else current >= self.end:
            yield current
            method = self.start.add if forward else self.start.subtract
            argument = unit if unit.endswith("s") else f"{unit}s"
            current = method(**{argument: step})
            step += amount

    def __iter__(self) -> Iterator:
        return iter((self.start, self.end))

    def __contains__(self, value) -> bool:
        lo, hi = sorted((self.start, self.end))
        return lo <= value <= hi

    def __repr__(self) -> str:
        return f"<Interval [{self.start} -> {self.end}]>"


def datetime(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
    tz: str | float | _dt.tzinfo | None = UTC,
    fold: int = 1,
    raise_on_unknown_times: bool = False,
) -> DateTime:
    tzinfo = None if tz is None else _safe_timezone(tz)
    value = DateTime(
        year,
        month,
        day,
        hour,
        minute,
        second,
        microsecond,
        tzinfo=tzinfo,
        fold=fold,
    )
    if raise_on_unknown_times and tzinfo is not None:
        roundtrip = value.astimezone(UTC).astimezone(tzinfo)
        if roundtrip.replace(tzinfo=None) != value.replace(tzinfo=None):
            raise ValueError("The datetime does not exist in this timezone")
    return value


def local(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
) -> DateTime:
    return datetime(year, month, day, hour, minute, second, microsecond, tz="local")


def naive(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
    fold: int = 1,
) -> DateTime:
    return datetime(year, month, day, hour, minute, second, microsecond, tz=None, fold=fold)


def date(year: int, month: int, day: int) -> Date:
    return Date(year, month, day)


def time(
    hour: int, minute: int = 0, second: int = 0, microsecond: int = 0
) -> Time:
    return Time(hour, minute, second, microsecond)


def duration(
    days: float = 0,
    seconds: float = 0,
    microseconds: float = 0,
    milliseconds: float = 0,
    minutes: float = 0,
    hours: float = 0,
    weeks: float = 0,
    years: float = 0,
    months: float = 0,
) -> Duration:
    return Duration(
        days,
        seconds,
        microseconds,
        milliseconds,
        minutes,
        hours,
        weeks,
        years,
        months,
    )


def interval(
    start: Date | DateTime, end: Date | DateTime, absolute: bool = False
) -> Interval:
    return Interval(start, end, absolute=absolute)


def instance(
    obj: _dt.datetime | _dt.date | _dt.time,
    tz: str | _dt.tzinfo | None = UTC,
) -> DateTime | Date | Time:
    if isinstance(obj, _dt.datetime):
        return DateTime.instance(obj, tz)
    if isinstance(obj, _dt.date):
        return Date.instance(obj)
    if isinstance(obj, _dt.time):
        tzinfo = obj.tzinfo if obj.tzinfo is not None else (
            None if tz is None else _safe_timezone(tz)
        )
        return Time(
            obj.hour,
            obj.minute,
            obj.second,
            obj.microsecond,
            tzinfo=tzinfo,
            fold=obj.fold,
        )
    raise TypeError("instance() expects a datetime, date, or time")


def now(tz: str | _dt.tzinfo | None = None) -> DateTime:
    tzinfo = local_timezone() if tz is None else _safe_timezone(tz)
    return DateTime.instance(_dt.datetime.now(tzinfo))


def today(tz: str | _dt.tzinfo = "local") -> DateTime:
    return now(tz).start_of("day")


def tomorrow(tz: str | _dt.tzinfo = "local") -> DateTime:
    return today(tz).add(days=1)


def yesterday(tz: str | _dt.tzinfo = "local") -> DateTime:
    return today(tz).subtract(days=1)


_DURATION_RE = re.compile(
    r"^P"
    r"(?:(?P<years>\d+(?:\.\d+)?)Y)?"
    r"(?:(?P<months>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<weeks>\d+(?:\.\d+)?)W)?"
    r"(?:(?P<days>\d+(?:\.\d+)?)D)?"
    r"(?:T"
    r"(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?"
    r")?$"
)


def _parse_duration(text: str) -> Duration | None:
    match = _DURATION_RE.match(text)
    if not match or not any(match.group(name) for name in (
        "years", "months", "weeks", "days", "hours", "minutes", "seconds"
    )):
        return None
    values = {
        name: float(match.group(name) or 0)
        for name in ("years", "months", "weeks", "days", "hours", "minutes", "seconds")
    }
    if not values["years"].is_integer() or not values["months"].is_integer():
        raise ParserError("fractional ISO years and months are not supported")
    values["years"] = int(values["years"])
    values["months"] = int(values["months"])
    return Duration(**values)


def _parse_iso(text: str, default_tz: str | _dt.tzinfo | None = UTC) -> DateTime:
    global _PARSE_FIELDS
    try:
        raw = text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ParserError(f"Unable to parse string [{text}]") from exc
    try:
        fields = _SCALAR_BUFFERS.parse
    except AttributeError:
        fields = (ctypes.c_int64 * 9)()
        _SCALAR_BUFFERS.parse = fields
    if _PARSE_FIELDS is None:
        _PARSE_FIELDS = lib().mp_parse_iso
    status = _PARSE_FIELDS(raw, len(raw), ctypes.addressof(fields))
    if status:
        raise ParserError(f"Unable to parse string [{text}]")
    if fields[8]:
        tzinfo = fixed_timezone(int(fields[7]))
    else:
        tzinfo = None if default_tz is None else _safe_timezone(default_tz)
    return DateTime(*map(int, fields[:7]), tzinfo=tzinfo)


def parse(text: str, **options):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if text == "now":
        return now(options.get("tz", UTC))
    parsed_duration = _parse_duration(text) if text.startswith("P") else None
    if parsed_duration is not None:
        return parsed_duration
    strict = options.pop("strict", True)
    default_tz = options.pop("tz", UTC)
    exact = options.pop("exact", False)
    now_value = options.pop("now", None)
    if options:
        raise TypeError(f"unexpected parse options: {', '.join(options)}")
    time_only = (
        len(text) >= 5
        and text[0].isdigit()
        and text[1].isdigit()
        and text[2] == ":"
        and text[3].isdigit()
        and text[4].isdigit()
    )
    if time_only:
        base = instance(now_value, default_tz) if now_value else now(default_tz)
        parsed = _parse_iso(f"{base:%Y-%m-%d}T{text}", default_tz)
    else:
        try:
            parsed = _parse_iso(text, default_tz)
        except ParserError:
            if strict:
                raise
            value = None
            for fmt in (
                "%B %d, %Y",
                "%b %d, %Y",
                "%d %B %Y",
                "%d %b %Y",
                "%m/%d/%Y",
                "%Y/%m/%d",
                "%Y-%m-%d %I:%M %p",
            ):
                try:
                    value = _dt.datetime.strptime(text, fmt)
                    break
                except ValueError:
                    pass
            if value is None:
                raise ParserError(f"Unable to parse string [{text}]")
            parsed = DateTime.instance(value, default_tz)
    if exact:
        if time_only:
            return parsed.time()
        if "T" not in text and " " not in text and not re.match(r"^\d{2}:", text):
            return parsed.date()
    return parsed


_FROM_FORMAT_MAP = {
    "YYYY": "%Y",
    "YY": "%y",
    "MMMM": "%B",
    "MMM": "%b",
    "MM": "%m",
    "M": "%m",
    "DD": "%d",
    "D": "%d",
    "HH": "%H",
    "H": "%H",
    "hh": "%I",
    "h": "%I",
    "mm": "%M",
    "m": "%M",
    "ss": "%S",
    "s": "%S",
    "SSSSSS": "%f",
    "SSS": "%f",
    "A": "%p",
    "a": "%p",
    "ZZ": "%z",
    "Z": "%z",
}


def from_format(
    string: str,
    fmt: str,
    tz: str | _dt.tzinfo = UTC,
    locale: str | None = None,
) -> DateTime:
    if locale not in (None, "en"):
        raise ValueError("only the English locale is included")
    python_fmt = _FORMAT_TOKEN.sub(
        lambda match: _FROM_FORMAT_MAP.get(match.group(0), match.group(0)), fmt
    )
    value = _dt.datetime.strptime(string, python_fmt)
    tzinfo = value.tzinfo or _safe_timezone(tz)
    return DateTime(
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
        tzinfo=tzinfo,
    )


def parse_many(
    values: Iterable[str], *, tz: str | _dt.tzinfo | None = UTC
) -> list[DateTime]:
    encoded = []
    originals = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError("parse_many() values must be strings")
        try:
            raw = value.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ParserError(f"Unable to parse string [{value}]") from exc
        if not raw:
            raise ParserError("Unable to parse an empty string")
        originals.append(value)
        encoded.append(raw)
    if not encoded:
        return []
    lengths = np.fromiter(map(len, encoded), dtype=np.int64, count=len(encoded))
    offsets = np.empty(len(encoded) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    data = np.frombuffer(b"".join(encoded), dtype=np.uint8)
    fields = np.empty((len(encoded), 9), dtype=np.int64)
    statuses = np.empty(len(encoded), dtype=np.int64)
    global _PARSE_MANY
    if _PARSE_MANY is None:
        _PARSE_MANY = lib().mp_parse_iso_many
    bridge_status = _PARSE_MANY(
        addr(data, np.uint8, ndim=1),
        addr(offsets, np.int64, ndim=1),
        i64(len(data), "encoded byte length"),
        i64(len(encoded), "item count"),
        addr(fields, np.int64, ndim=2),
        addr(statuses, np.int64, ndim=1),
    )
    if bridge_status:
        raise RuntimeError(f"native batch parser failed with status {bridge_status}")
    failed = np.flatnonzero(statuses)
    if failed.size:
        index = int(failed[0])
        raise ParserError(
            f"Unable to parse string at index {index}: {originals[index]}"
        )
    default = None if tz is None else _safe_timezone(tz)
    result = []
    for row in fields.tolist():
        tzinfo = fixed_timezone(int(row[7])) if row[8] else default
        result.append(DateTime(*row[:7], tzinfo=tzinfo))
    return result


def add_many(
    values: Sequence[_dt.date | _dt.datetime],
    *,
    years: int = 0,
    months: int = 0,
    weeks: int = 0,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: float = 0,
    microseconds: int = 0,
) -> list[Date | DateTime]:
    if not values:
        return []
    datetimes = isinstance(values[0], _dt.datetime)
    if any(isinstance(value, _dt.datetime) != datetimes for value in values):
        raise TypeError("add_many() cannot mix dates and datetimes")
    if datetimes and any(value.tzinfo is not None for value in values):
        return [
            instance(value).add(
                years, months, weeks, days, hours, minutes, seconds, microseconds
            )
            for value in values
        ]
    if datetimes:
        source = np.fromiter(
            (
                field
                for value in values
                for field in (
                    value.year,
                    value.month,
                    value.day,
                    value.hour,
                    value.minute,
                    value.second,
                    value.microsecond,
                )
            ),
            dtype=np.int64,
            count=len(values) * 7,
        ).reshape((-1, 7))
    else:
        source = np.fromiter(
            (
                field
                for value in values
                for field in (value.year, value.month, value.day, 0, 0, 0, 0)
            ),
            dtype=np.int64,
            count=len(values) * 7,
        ).reshape((-1, 7))
    result = np.empty_like(source)
    second, micros = _split_seconds(seconds, microseconds)
    deltas = _native_deltas(
        years, months, weeks, days, hours, minutes, second, micros
    )
    global _ADD_MANY
    if _ADD_MANY is None:
        _ADD_MANY = lib().mp_add_datetime_many
    status = _ADD_MANY(
        addr(source, np.int64, ndim=2),
        i64(len(values), "item count"),
        addr(result, np.int64, ndim=2),
        *deltas,
    )
    if status < 0:
        raise RuntimeError(f"native batch arithmetic failed with status {status}")
    if status > 0:
        raise OverflowError(f"date arithmetic failed at batch index {status - 1}")
    if datetimes:
        return [DateTime(*row) for row in result.tolist()]
    return [Date(*row[:3]) for row in result.tolist()]


def to_iso8601_many(values: Sequence[_dt.datetime]) -> list[str]:
    if not values:
        return []
    fields = np.fromiter(
        (
            field
            for value in values
            for offset in (value.utcoffset(),)
            for field in (
                value.year,
                value.month,
                value.day,
                value.hour,
                value.minute,
                value.second,
                value.microsecond,
                int(offset.total_seconds()) if offset else 0,
                value.tzinfo is not None,
            )
        ),
        dtype=np.int64,
        count=len(values) * 9,
    ).reshape((-1, 9))
    output = np.empty((len(values), 40), dtype=np.uint8)
    lengths = np.empty(len(values), dtype=np.int64)
    global _FORMAT_MANY
    if _FORMAT_MANY is None:
        _FORMAT_MANY = lib().mp_format_iso_many
    status = _FORMAT_MANY(
        addr(fields, np.int64, ndim=2),
        i64(len(values), "item count"),
        addr(output, np.uint8, ndim=2),
        i64(output.shape[1], "output stride"),
        addr(lengths, np.int64, ndim=1),
    )
    if status:
        raise RuntimeError(f"native ISO formatter failed with status {status}")
    raw = output.tobytes()
    stride = output.shape[1]
    return [
        raw[index * stride : index * stride + length].decode("ascii")
        for index, length in enumerate(lengths.tolist())
    ]
