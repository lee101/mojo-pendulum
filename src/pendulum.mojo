"""Gregorian calendar arithmetic and ISO-8601 parsing kernels."""

from max.algorithm import parallelize
from std.runtime import initialize_runtime
from std.sys.info import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime PARALLEL_ADD_THRESHOLD = 4096
comptime PARALLEL_FORMAT_THRESHOLD = 16384
comptime PARALLEL_TASKS = 16


def is_leap(year: Int) -> Bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def month_length(year: Int, month: Int) -> Int:
    if month == 2:
        return 29 if is_leap(year) else 28
    if month == 4 or month == 6 or month == 9 or month == 11:
        return 30
    return 31


def ordinal_from_ymd(year: Int, month: Int, day: Int) -> Int:
    var y = year - 1
    var total = 365 * y + y // 4 - y // 100 + y // 400
    var m = 1
    while m < month:
        total += month_length(year, m)
        m += 1
    return total + day


def ymd_from_ordinal(ordinal: Int, result: IPtr):
    # Howard Hinnant's civil calendar transform, shifted to Python ordinals.
    var z = ordinal - 719163 + 719468
    var era = z // 146097
    var doe = z - era * 146097
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var year = yoe + era * 400
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    var mp = (5 * doy + 2) // 153
    var day = doy - (153 * mp + 2) // 5 + 1
    var month = mp + (3 if mp < 10 else -9)
    year += 1 if month <= 2 else 0
    result[0] = Int64(year)
    result[1] = Int64(month)
    result[2] = Int64(day)


def digit(src: BPtr, i: Int) -> Int:
    var value = Int(src[i]) - 48
    return value if value >= 0 and value <= 9 else -1


def digits(src: BPtr, begin: Int, count: Int) -> Int:
    var value = 0
    for i in range(count):
        var d = digit(src, begin + i)
        if d < 0:
            return -1
        value = value * 10 + d
    return value


def iso_weeks_in_year(year: Int) -> Int:
    var jan1 = (ordinal_from_ymd(year, 1, 1) - 1) % 7
    return 53 if jan1 == 3 or (jan1 == 2 and is_leap(year)) else 52


def parse_date(src: BPtr, n: Int, result: IPtr) -> Int:
    if n < 4:
        return 1
    var year = digits(src, 0, 4)
    if year < 1:
        return 1
    var month = 1
    var day = 1
    if n == 4:
        pass
    elif n >= 7 and Int(src[4]) == 87:
        var week = digits(src, 5, 2)
        var weekday = digits(src, 7, 1) if n == 8 else 1
        if (
            (n != 7 and n != 8)
            or week < 1
            or week > iso_weeks_in_year(year)
            or weekday < 1
            or weekday > 7
        ):
            return 1
        var jan4 = ordinal_from_ymd(year, 1, 4)
        ymd_from_ordinal(
            jan4 - ((jan4 - 1) % 7) + (week - 1) * 7 + weekday - 1,
            result,
        )
        return 0
    elif n >= 8 and Int(src[4]) == 45 and Int(src[5]) == 87:
        var week = digits(src, 6, 2)
        var weekday = (
            digits(src, 9, 1)
            if n == 10 and Int(src[8]) == 45
            else 1
        )
        if (
            (n != 8 and n != 10)
            or week < 1
            or week > iso_weeks_in_year(year)
            or weekday < 1
            or weekday > 7
        ):
            return 1
        var jan4 = ordinal_from_ymd(year, 1, 4)
        ymd_from_ordinal(
            jan4 - ((jan4 - 1) % 7) + (week - 1) * 7 + weekday - 1,
            result,
        )
        return 0
    elif n == 7 and Int(src[4]) != 45:
        var doy = digits(src, 4, 3)
        if doy < 1 or doy > (366 if is_leap(year) else 365):
            return 1
        ymd_from_ordinal(ordinal_from_ymd(year, 1, 1) + doy - 1, result)
        return 0
    elif n == 8 and Int(src[4]) == 45:
        var doy = digits(src, 5, 3)
        if doy < 1 or doy > (366 if is_leap(year) else 365):
            return 1
        ymd_from_ordinal(ordinal_from_ymd(year, 1, 1) + doy - 1, result)
        return 0
    elif n == 7 and Int(src[4]) == 45:
        month = digits(src, 5, 2)
    elif n == 8:
        month = digits(src, 4, 2)
        day = digits(src, 6, 2)
    elif n == 10 and Int(src[4]) == 45 and Int(src[7]) == 45:
        month = digits(src, 5, 2)
        day = digits(src, 8, 2)
    else:
        return 1
    if (
        month < 1
        or month > 12
        or day < 1
        or day > month_length(year, month)
    ):
        return 1
    result[0] = Int64(year)
    result[1] = Int64(month)
    result[2] = Int64(day)
    return 0


def parse_time(src: BPtr, begin: Int, end: Int, result: IPtr) -> Int:
    var tz_begin = end
    var i = begin
    while i < end:
        var c = Int(src[i])
        if c == 90 or c == 122 or c == 43 or c == 45:
            tz_begin = i
            break
        i += 1
    var time_n = tz_begin - begin
    if time_n < 2:
        return 2
    var hour = digits(src, begin, 2)
    var minute = 0
    var second = 0
    var micros = 0
    var frac_begin = -1
    var colon = time_n >= 3 and Int(src[begin + 2]) == 58
    if time_n == 2:
        pass
    elif colon:
        if time_n < 5:
            return 2
        minute = digits(src, begin + 3, 2)
        if time_n > 5:
            if time_n < 8 or Int(src[begin + 5]) != 58:
                return 2
            second = digits(src, begin + 6, 2)
            if time_n > 8:
                if Int(src[begin + 8]) != 46 and Int(src[begin + 8]) != 44:
                    return 2
                frac_begin = begin + 9
    else:
        if time_n < 4:
            return 2
        minute = digits(src, begin + 2, 2)
        if time_n > 4:
            if time_n < 6:
                return 2
            second = digits(src, begin + 4, 2)
            if time_n > 6:
                if Int(src[begin + 6]) != 46 and Int(src[begin + 6]) != 44:
                    return 2
                frac_begin = begin + 7
    if frac_begin >= 0:
        if frac_begin >= tz_begin:
            return 2
        var used = 0
        var j = frac_begin
        while j < tz_begin:
            var d = digit(src, j)
            if d < 0:
                return 2
            if used < 6:
                micros = micros * 10 + d
                used += 1
            j += 1
        while used < 6:
            micros *= 10
            used += 1
    if (
        hour < 0
        or hour > 23
        or minute < 0
        or minute > 59
        or second < 0
        or second > 59
    ):
        return 2
    result[3] = Int64(hour)
    result[4] = Int64(minute)
    result[5] = Int64(second)
    result[6] = Int64(micros)
    result[7] = 0
    result[8] = 0
    if tz_begin < end:
        var marker = Int(src[tz_begin])
        result[8] = 1
        if marker == 90 or marker == 122:
            if tz_begin + 1 != end:
                return 3
        else:
            var remain = end - tz_begin - 1
            var tz_hour = -1
            var tz_minute = 0
            if remain == 2:
                tz_hour = digits(src, tz_begin + 1, 2)
            elif remain == 4:
                tz_hour = digits(src, tz_begin + 1, 2)
                tz_minute = digits(src, tz_begin + 3, 2)
            elif remain == 5 and Int(src[tz_begin + 3]) == 58:
                tz_hour = digits(src, tz_begin + 1, 2)
                tz_minute = digits(src, tz_begin + 4, 2)
            else:
                return 3
            if (
                tz_hour < 0
                or tz_hour > 23
                or tz_minute < 0
                or tz_minute > 59
            ):
                return 3
            result[7] = Int64(
                (tz_hour * 3600 + tz_minute * 60)
                * (1 if marker == 43 else -1)
            )
    return 0


def clear_fields(result: IPtr):
    comptime W = simd_width_of[DType.float64]()
    var zeros = SIMD[DType.int64, W](0)
    var i = 0
    while i + W <= 9:
        result.store(i, zeros)
        i += W
    while i < 9:
        result[i] = 0
        i += 1


def parse_iso(src: BPtr, n: Int, result: IPtr) -> Int:
    clear_fields(result)
    var split = n
    if n >= 10 and Int(src[4]) == 45 and Int(src[7]) == 45:
        split = 10
    elif (
        n >= 10
        and Int(src[4]) == 45
        and Int(src[5]) == 87
        and Int(src[8]) == 45
    ):
        split = 10
    elif n > 8 and Int(src[4]) == 45:
        split = 8
    elif (
        n >= 8
        and Int(src[4]) == 87
        and digit(src, 7) >= 1
        and digit(src, 7) <= 7
    ):
        split = 8
    elif n > 7 and Int(src[4]) == 87:
        split = 7
    elif n > 8 and digit(src, 7) >= 0:
        split = 8
    elif n > 7 and digit(src, 7) < 0:
        split = 7
    var status = parse_date(src, split, result)
    if status != 0:
        return status
    if split < n:
        status = parse_time(src, split + 1, n, result)
        if status != 0:
            return status
    return 0


def add_fields(
    src: IPtr,
    dst: IPtr,
    years: Int,
    months: Int,
    weeks: Int,
    days: Int,
    hours: Int,
    minutes: Int,
    seconds: Int,
    micros: Int,
) -> Int:
    var year = Int(src[0])
    var month = Int(src[1])
    var day = Int(src[2])
    if year < 1 or year > 9999 or month < 1 or month > 12:
        return 1
    var month_index = year * 12 + month - 1 + years * 12 + months
    year = month_index // 12
    month = month_index % 12 + 1
    if year < 1 or year > 9999:
        return 1
    day = min(day, month_length(year, month))
    var ordinal = ordinal_from_ymd(year, month, day) + weeks * 7 + days
    var day_micros = (
        ((Int(src[3]) * 60 + Int(src[4])) * 60 + Int(src[5]))
        * 1000000
        + Int(src[6])
        + ((hours * 60 + minutes) * 60 + seconds) * 1000000
        + micros
    )
    ordinal += day_micros // 86400000000
    day_micros %= 86400000000
    if ordinal < 1 or ordinal > 3652059:
        return 1
    ymd_from_ordinal(ordinal, dst)
    dst[3] = Int64(day_micros // 3600000000)
    dst[4] = Int64((day_micros % 3600000000) // 60000000)
    dst[5] = Int64((day_micros % 60000000) // 1000000)
    dst[6] = Int64(day_micros % 1000000)
    return 0


def write_two(dst: BPtr, pos: Int, value: Int):
    dst[pos] = UInt8(48 + value // 10)
    dst[pos + 1] = UInt8(48 + value % 10)


def write_four(dst: BPtr, pos: Int, value: Int):
    dst[pos] = UInt8(48 + (value // 1000) % 10)
    dst[pos + 1] = UInt8(48 + (value // 100) % 10)
    dst[pos + 2] = UInt8(48 + (value // 10) % 10)
    dst[pos + 3] = UInt8(48 + value % 10)


def format_iso(src: IPtr, dst: BPtr) -> Int:
    write_four(dst, 0, Int(src[0]))
    dst[4] = 45
    write_two(dst, 5, Int(src[1]))
    dst[7] = 45
    write_two(dst, 8, Int(src[2]))
    dst[10] = 84
    write_two(dst, 11, Int(src[3]))
    dst[13] = 58
    write_two(dst, 14, Int(src[4]))
    dst[16] = 58
    write_two(dst, 17, Int(src[5]))
    var pos = 19
    var micros = Int(src[6])
    if micros != 0:
        dst[pos] = 46
        pos += 1
        var divisor = 100000
        while divisor > 0:
            dst[pos] = UInt8(48 + (micros // divisor) % 10)
            pos += 1
            divisor //= 10
    if src[8] != 0:
        var offset = Int(src[7])
        if offset == 0:
            dst[pos] = 90
            pos += 1
        else:
            dst[pos] = UInt8(43 if offset > 0 else 45)
            pos += 1
            var absolute = abs(offset)
            write_two(dst, pos, absolute // 3600)
            pos += 2
            dst[pos] = 58
            pos += 1
            write_two(dst, pos, (absolute % 3600) // 60)
            pos += 2
    return pos


def parse_many_item(
    src: BPtr,
    offsets: IPtr,
    data_n: Int,
    results: IPtr,
    statuses: IPtr,
    i: Int,
):
    var begin = Int(offsets[i])
    var end = Int(offsets[i + 1])
    if begin < 0 or end <= begin or end > data_n:
        statuses[i] = -1
    else:
        statuses[i] = Int64(
            parse_iso(src + begin, end - begin, results + i * 9)
        )


def add_many_item(
    src: IPtr,
    dst: IPtr,
    i: Int,
    years: Int,
    months: Int,
    weeks: Int,
    days: Int,
    hours: Int,
    minutes: Int,
    seconds: Int,
    micros: Int,
):
    var status = add_fields(
        src + i * 7,
        dst + i * 7,
        years,
        months,
        weeks,
        days,
        hours,
        minutes,
        seconds,
        micros,
    )
    if status != 0:
        dst[i * 7] = 0


@export("mp_parse_iso")
def mp_parse_iso(data_addr: Int, n: Int, result_addr: Int) abi("C") -> Int:
    if data_addr == 0 or result_addr == 0 or n <= 0:
        return -1
    return parse_iso(
        BPtr(unsafe_from_address=data_addr),
        n,
        IPtr(unsafe_from_address=result_addr),
    )


def parse_iso_many_impl(
    data_addr: Int,
    offsets_addr: Int,
    data_n: Int,
    count: Int,
    results_addr: Int,
    statuses_addr: Int,
) -> Int:
    if (
        data_addr == 0
        or offsets_addr == 0
        or results_addr == 0
        or statuses_addr == 0
        or data_n <= 0
        or count <= 0
    ):
        return -1
    var src = BPtr(unsafe_from_address=data_addr)
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var results = IPtr(unsafe_from_address=results_addr)
    var statuses = IPtr(unsafe_from_address=statuses_addr)

    for i in range(count):
        parse_many_item(src, offsets, data_n, results, statuses, i)
    for i in range(count):
        if statuses[i] < 0:
            return -2
    return 0


@export("mp_parse_iso_many")
def mp_parse_iso_many(
    data_addr: Int,
    offsets_addr: Int,
    data_n: Int,
    count: Int,
    results_addr: Int,
    statuses_addr: Int,
) abi("C") -> Int:
    return parse_iso_many_impl(
        data_addr,
        offsets_addr,
        data_n,
        count,
        results_addr,
        statuses_addr,
    )


@export("mp_add_datetime")
def mp_add_datetime(
    src_addr: Int,
    dst_addr: Int,
    years: Int,
    months: Int,
    weeks: Int,
    days: Int,
    hours: Int,
    minutes: Int,
    seconds: Int,
    micros: Int,
) abi("C") -> Int:
    if src_addr == 0 or dst_addr == 0:
        return -1
    return add_fields(
        IPtr(unsafe_from_address=src_addr),
        IPtr(unsafe_from_address=dst_addr),
        years,
        months,
        weeks,
        days,
        hours,
        minutes,
        seconds,
        micros,
    )


def add_datetime_many_impl(
    src_addr: Int,
    count: Int,
    dst_addr: Int,
    years: Int,
    months: Int,
    weeks: Int,
    days: Int,
    hours: Int,
    minutes: Int,
    seconds: Int,
    micros: Int,
) -> Int:
    if src_addr == 0 or dst_addr == 0 or count <= 0:
        return -1
    var src = IPtr(unsafe_from_address=src_addr)
    var dst = IPtr(unsafe_from_address=dst_addr)

    if count >= PARALLEL_ADD_THRESHOLD:
        initialize_runtime()
        var task_count = min(PARALLEL_TASKS, count)

        @parameter
        def add_chunk(task: Int):
            var first = task * count // task_count
            var last = (task + 1) * count // task_count
            for i in range(first, last):
                add_many_item(
                    src,
                    dst,
                    i,
                    years,
                    months,
                    weeks,
                    days,
                    hours,
                    minutes,
                    seconds,
                    micros,
                )

        parallelize[add_chunk](task_count, task_count)
    else:
        for i in range(count):
            add_many_item(
                src,
                dst,
                i,
                years,
                months,
                weeks,
                days,
                hours,
                minutes,
                seconds,
                micros,
            )
    for i in range(count):
        if dst[i * 7] == 0:
            return i + 1
    return 0


@export("mp_add_datetime_many")
def mp_add_datetime_many(
    src_addr: Int,
    count: Int,
    dst_addr: Int,
    years: Int,
    months: Int,
    weeks: Int,
    days: Int,
    hours: Int,
    minutes: Int,
    seconds: Int,
    micros: Int,
) abi("C") -> Int:
    return add_datetime_many_impl(
        src_addr,
        count,
        dst_addr,
        years,
        months,
        weeks,
        days,
        hours,
        minutes,
        seconds,
        micros,
    )


def format_iso_many_impl(
    fields_addr: Int,
    count: Int,
    output_addr: Int,
    stride: Int,
    lengths_addr: Int,
) -> Int:
    if (
        fields_addr == 0
        or output_addr == 0
        or lengths_addr == 0
        or count <= 0
        or stride < 32
    ):
        return -1
    var fields = IPtr(unsafe_from_address=fields_addr)
    var output = BPtr(unsafe_from_address=output_addr)
    var lengths = IPtr(unsafe_from_address=lengths_addr)

    if count >= PARALLEL_FORMAT_THRESHOLD:
        initialize_runtime()
        var task_count = min(PARALLEL_TASKS, count)

        @parameter
        def format_chunk(task: Int):
            var first = task * count // task_count
            var last = (task + 1) * count // task_count
            for i in range(first, last):
                lengths[i] = Int64(
                    format_iso(fields + i * 9, output + i * stride)
                )

        parallelize[format_chunk](task_count, task_count)
    else:
        for i in range(count):
            lengths[i] = Int64(
                format_iso(fields + i * 9, output + i * stride)
            )
    return 0


@export("mp_format_iso_many")
def mp_format_iso_many(
    fields_addr: Int,
    count: Int,
    output_addr: Int,
    stride: Int,
    lengths_addr: Int,
) abi("C") -> Int:
    return format_iso_many_impl(
        fields_addr,
        count,
        output_addr,
        stride,
        lengths_addr,
    )
