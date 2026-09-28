"""
Marine heatwave detection, to the published definition.

This module implements Hobday et al. (2016), "A hierarchical approach to
defining marine heatwaves", Progress in Oceanography 141:227-238, and the
category scheme of Hobday et al. (2018), Oceanography 31(2):162-173. Both are
arithmetic on a temperature record. There is no model fitted here, nothing is
learned, and nothing is predicted: every number this module returns is a
statement about sea surface temperature that has already been recorded.

The definition, quoted from the papers rather than paraphrased:

  climatology   "using all data within an 11-day window centred on the time of
                year from which the climatological mean and threshold are
                calculated", over a baseline for which "a period of 30 years
                is recommended".
  threshold     the 90th percentile of that same window.
  event         "a MHW if it lasts for five or more days, with temperatures
                warmer than the 90th percentile".
  joining       "gaps between events of two days or less with subsequent five
                day or more events will be" treated as one event.
  categories    multiples of the local difference between the climatological
                mean and the 90th percentile threshold: "moderate (1-2x,
                Category I), strong (2-3x, Category II), severe (3-4x,
                Category III), and extreme (>4x, Category IV)".

Two choices are ours, and are stated wherever results are served rather than
buried here:

  * The 2016 paper specifies the 11-day window and no further smoothing, so
    none is applied. Some published implementations additionally smooth the
    climatology with a 31-day moving average; that would change the threshold
    slightly, so it is offered as an explicit option and defaulted off.
  * Day-of-year 366 is pooled with 365, so a leap day is not compared against
    a baseline of only the seven leap years in the window.
  * An event whose last day is the record's last day is reported as ongoing
    rather than as having ended on that day. The 2016 definition says nothing
    about truncation, but claiming an end date the record cannot support would
    invent the one fact the reader most wants.

The temperature record itself is the Copernicus GLORYS12V1 reanalysis at its
shallowest level (~0.494 m). That is the top layer of an ocean model, not a
satellite skin temperature and not an in-situ measurement, and it is labelled
that way everywhere it reaches the page.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Hobday et al. (2016).
CLIMATOLOGY_WINDOW_DAYS = 11
THRESHOLD_PERCENTILE = 90.0
MIN_DURATION_DAYS = 5
MAX_GAP_DAYS = 2

# Hobday et al. (2018), in ascending severity. Each bound is a multiple of
# (threshold - climatological mean).
CATEGORIES = (
    (1.0, "I", "Moderate"),
    (2.0, "II", "Strong"),
    (3.0, "III", "Severe"),
    (4.0, "IV", "Extreme"),
)


class HeatwaveDataError(RuntimeError):
    """The record cannot support the definition -- never substitute data."""


@dataclass(frozen=True)
class Climatology:
    """Per-day-of-year mean and 90th percentile for one location.

    Both arrays are indexed by day of year minus one, so index 0 is 1 January
    and the arrays are 365 long. Day 366 shares day 365's values.
    """

    mean: np.ndarray
    threshold: np.ndarray
    baseline_first_year: int
    baseline_last_year: int
    window_days: int = CLIMATOLOGY_WINDOW_DAYS
    percentile: float = THRESHOLD_PERCENTILE
    smoothed_days: int | None = None

    def for_day_of_year(self, doy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Look up both curves for an array of days of year (1-366)."""
        index = np.minimum(doy, 365) - 1
        return self.mean[index], self.threshold[index]


@dataclass
class HeatwaveEvent:
    """One detected event, with only what the definition actually defines."""

    start_index: int
    # The last day of the event that the record actually shows. When the record
    # stops here, that is not the same as the event ending here.
    end_index: int
    # Days observed above the threshold. For an ongoing event this is a
    # minimum: the record ran out, the heatwave did not necessarily stop.
    duration_days: int
    # True when the event's last day is the last day of the record, so it was
    # cut off by the data rather than seen to end. A completed event has been
    # observed to finish; an ongoing one has not.
    ongoing: bool
    peak_index: int
    # Intensity is measured from the climatological mean, per the papers.
    peak_intensity_c: float
    mean_intensity_c: float
    cumulative_intensity_c_days: float
    peak_category_number: int
    peak_category_roman: str
    peak_category_name: str
    # How many days sat in each category, so a mostly-moderate event with one
    # severe day cannot be reported as simply "severe".
    days_by_category: dict[str, int] = field(default_factory=dict)


def day_of_year(dates: np.ndarray) -> np.ndarray:
    """Day of year (1-366) for a datetime64 array, without pandas."""
    years = dates.astype("datetime64[Y]")
    return ((dates.astype("datetime64[D]") - years).astype(int) + 1).astype(int)


def build_climatology(
    dates: np.ndarray,
    temperature: np.ndarray,
    baseline_first_year: int,
    baseline_last_year: int,
    window_days: int = CLIMATOLOGY_WINDOW_DAYS,
    percentile: float = THRESHOLD_PERCENTILE,
    smooth_days: int | None = None,
) -> Climatology:
    """Mean and 90th percentile per day of year, from an 11-day window.

    `temperature` is a 1-D record matching `dates`. Only days inside the
    baseline years contribute; later years are what the events are detected in,
    and letting them into their own baseline would flatten the very anomaly
    being looked for.
    """
    if dates.shape[0] != temperature.shape[0]:
        raise HeatwaveDataError("dates and temperature must be the same length")
    if window_days < 1 or window_days % 2 == 0:
        raise HeatwaveDataError("the climatology window must be an odd number of days")

    years = dates.astype("datetime64[Y]").astype(int) + 1970
    in_baseline = (years >= baseline_first_year) & (years <= baseline_last_year)
    if not in_baseline.any():
        raise HeatwaveDataError(
            f"no days between {baseline_first_year} and {baseline_last_year} in this record")

    doy = day_of_year(dates)[in_baseline]
    values = temperature[in_baseline]
    # A leap day joins 31 December's pool rather than standing on its own.
    doy = np.minimum(doy, 365)

    half = window_days // 2
    mean = np.full(365, np.nan)
    threshold = np.full(365, np.nan)
    # Which days of year fall in the window around each target day, wrapping
    # across the new year so 1 January is not built from half a window.
    for target in range(1, 366):
        offsets = (np.arange(target - half, target + half + 1) - 1) % 365 + 1
        pool = values[np.isin(doy, offsets)]
        pool = pool[np.isfinite(pool)]
        if pool.size == 0:
            continue
        mean[target - 1] = float(np.mean(pool))
        threshold[target - 1] = float(np.percentile(pool, percentile))

    if smooth_days:
        mean = _wrap_smooth(mean, smooth_days)
        threshold = _wrap_smooth(threshold, smooth_days)

    return Climatology(
        mean=mean,
        threshold=threshold,
        baseline_first_year=baseline_first_year,
        baseline_last_year=baseline_last_year,
        window_days=window_days,
        percentile=percentile,
        smoothed_days=smooth_days,
    )


def _wrap_smooth(curve: np.ndarray, days: int) -> np.ndarray:
    """Moving average around the year, so 31 December and 1 January meet."""
    if days < 2:
        return curve
    kernel = np.ones(days) / days
    padded = np.concatenate([curve[-days:], curve, curve[:days]])
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[days:-days]


def detect_events(
    dates: np.ndarray,
    temperature: np.ndarray,
    climatology: Climatology,
    min_duration_days: int = MIN_DURATION_DAYS,
    max_gap_days: int = MAX_GAP_DAYS,
) -> list[HeatwaveEvent]:
    """Every marine heatwave in the record, to the 2016 definition.

    The record must be daily and continuous; a gap in the dates would let two
    separate warm spells be counted as one run of days.
    """
    if dates.shape[0] != temperature.shape[0]:
        raise HeatwaveDataError("dates and temperature must be the same length")
    if dates.size == 0:
        return []
    steps = np.diff(dates.astype("datetime64[D]").astype(int))
    if steps.size and not np.all(steps == 1):
        raise HeatwaveDataError("the record must be daily and continuous with no missing days")

    mean, threshold = climatology.for_day_of_year(day_of_year(dates))
    over = np.isfinite(temperature) & np.isfinite(threshold) & (temperature > threshold)

    runs = _runs_of_true(over)
    runs = [r for r in runs if r[1] - r[0] + 1 >= min_duration_days]
    runs = _join_close_runs(runs, max_gap_days)

    # The local difference the categories are multiples of.
    span = threshold - mean

    events: list[HeatwaveEvent] = []
    for start, end in runs:
        piece = slice(start, end + 1)
        intensity = temperature[piece] - mean[piece]
        local_span = span[piece]
        with np.errstate(divide="ignore", invalid="ignore"):
            multiples = np.where(local_span > 0, intensity / local_span, np.nan)

        peak_offset = int(np.nanargmax(intensity))
        counts: dict[str, int] = {}
        for m in multiples:
            name = category_for(m)[2] if np.isfinite(m) else None
            if name:
                counts[name] = counts.get(name, 0) + 1
        number, roman, name = category_for(multiples[peak_offset])

        events.append(HeatwaveEvent(
            start_index=start,
            end_index=end,
            duration_days=end - start + 1,
            # The record's own last day. Nothing beyond it has been observed,
            # so an event reaching it cannot be called finished.
            ongoing=bool(end == dates.size - 1),
            peak_index=start + peak_offset,
            peak_intensity_c=float(intensity[peak_offset]),
            mean_intensity_c=float(np.nanmean(intensity)),
            cumulative_intensity_c_days=float(np.nansum(intensity)),
            peak_category_number=number,
            peak_category_roman=roman,
            peak_category_name=name,
            days_by_category=counts,
        ))
    return events


def category_for(multiple: float) -> tuple[int, str, str]:
    """Hobday et al. (2018): moderate 1-2x, strong 2-3x, severe 3-4x, extreme >4x.

    Below 1x the day is not at any category: Category I starts where the
    temperature crosses the 90th percentile. That case is real rather than
    hypothetical -- the 2016 definition joins two events across a gap of up to
    two days, and those gap days sit inside the event while being below the
    threshold. Calling them Moderate would overstate what the record says.
    """
    if not np.isfinite(multiple):
        return 0, "", "Not categorised"
    if multiple < CATEGORIES[0][0]:
        return 0, "", "Below threshold"
    number, roman, name = 1, "I", "Moderate"
    for index, (bound, this_roman, this_name) in enumerate(CATEGORIES, start=1):
        if multiple >= bound:
            number, roman, name = index, this_roman, this_name
    return number, roman, name


def _runs_of_true(flags: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive [start, end] index pairs for each run of True."""
    if not flags.any():
        return []
    padded = np.concatenate([[False], flags, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _join_close_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Merge runs separated by `max_gap` days or fewer, per the 2016 definition."""
    if not runs:
        return []
    joined = [runs[0]]
    for start, end in runs[1:]:
        last_start, last_end = joined[-1]
        if start - last_end - 1 <= max_gap:
            joined[-1] = (last_start, end)
        else:
            joined.append((start, end))
    return joined
