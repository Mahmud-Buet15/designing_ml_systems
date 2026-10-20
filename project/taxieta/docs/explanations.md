
## Code Explanation 1
```python
def point_in_time_zone_speed(trips: pd.DataFrame, window: str = "15min") -> pd.Series:
    """For each trip, the median speed of trips from the same pickup zone that COMPLETED in
    the `window` before this trip's pickup. Mirrors stream/consumer.py for training.

    TODO(M4): verify this matches the streaming consumer exactly on one day (M2 task C).
    """
    done = trips[["pu_zone", "dropoff_ts", "trip_distance", "duration_s"]].copy()   # taking columns for done/completed trips
    done = done[(done["duration_s"] > 60) & (done["trip_distance"] > 0)]      # filtering out trips with duration less than 60 seconds and distance less than 0. Otherwise it will impact the stats
    done["mph"] = done["trip_distance"] / (done["duration_s"] / 3600)     # calculating the speed in mph

    out = pd.Series(np.nan, index=trips.index)       # initializing the output series
    w = pd.Timedelta(window)    # getting the window in timed
    for zone, grp in trips.groupby("pu_zone"):
        d = done[done["pu_zone"] == zone].sort_values("dropoff_ts")     # sorting the done trips by dropoff time
        if d.empty:
            continue
        ends = d["dropoff_ts"].to_numpy()
        mph = d["mph"].to_numpy()
        starts = grp["pickup_ts"].to_numpy()        # getting the pickup times
        hi = np.searchsorted(ends, starts, side="left")          # completed strictly before pickup
        lo = np.searchsorted(ends, starts - w, side="left")     # completed within the window before pickup
        vals = [np.median(mph[a:b]) if b > a else np.nan for a, b in zip(lo, hi, strict=True)]
        out.loc[grp.index] = vals
    return out
```

This function builds a **point-in-time feature**: for every trip, it computes the typical speed in that trip's pickup zone just before the trip started. It uses only information that existed at pickup time, so the model never "sees the future" during training.

### What the docstring is saying

- **The feature:** for each trip, take all trips from the **same pickup zone** that **finished** (dropped off) in the 15 minutes before this trip's pickup, and return their **median speed**.
- **"Mirrors stream/consumer.py for training":** in production, a streaming consumer computes the same feature live. This function recreates it on historical data so the model trains on features computed the same way as at serving time. If the two differ, you get *training/serving skew*.
- **The TODO** is a reminder to check that both versions produce identical numbers on a sample day.

### Step 1: Prepare the completed trips

```python
done = trips[["pu_zone", "dropoff_ts", "trip_distance", "duration_s"]].copy()
done = done[(done["duration_s"] > 60) & (done["trip_distance"] > 0)]
done["mph"] = done["trip_distance"] / (done["duration_s"] / 3600)
```

- This keeps only the columns needed and makes a copy so the original `trips` isn't changed.
- The filter **keeps** trips longer than 60 seconds with a distance greater than 0. So it removes trips of 60 seconds or less and trips with zero or negative distance. (The comment says "less than 0", but zero-distance trips are dropped too.) Such records are usually data errors or cancelled rides, and they would produce absurd or infinite speeds.
- Speed in mph is miles ÷ hours, and `duration_s / 3600` converts seconds to hours.

The same `trips` table plays two roles here: as **completed trips** (`done`, keyed by drop-off time) and as **trips needing a feature** (keyed by pickup time).

### Step 2: Set up the output

```python
out = pd.Series(np.nan, index=trips.index)
w = pd.Timedelta(window)
```

- `out` has one slot per trip, all starting as `NaN`. Trips with no eligible recent trips stay `NaN`.
- `pd.Timedelta("15min")` turns the string into a time duration, so it can be subtracted from timestamps.

### Step 3: Process each zone

```python
for zone, grp in trips.groupby("pu_zone"):
    d = done[done["pu_zone"] == zone].sort_values("dropoff_ts")
    if d.empty:
        continue
```

For each pickup zone:
- `grp` holds the trips in that zone that need the feature.
- `d` holds that zone's completed trips, **sorted by drop-off time**. The sort is required for the binary search below.
- If the zone has no valid completed trips, it's skipped and its trips keep `NaN`.

### Step 4: Find each trip's window with binary search

```python
ends   = d["dropoff_ts"].to_numpy()
mph    = d["mph"].to_numpy()
starts = grp["pickup_ts"].to_numpy()
hi = np.searchsorted(ends, starts, side="left")
lo = np.searchsorted(ends, starts - w, side="left")
```

`np.searchsorted(sorted_array, values, side="left")` returns, for each value, the position where it would be inserted to keep the array sorted, placed before any equal elements. It does this for all pickups at once using binary search, which is fast.

- `hi`: positions `[0, hi)` hold drop-offs **strictly before** the pickup.
- `lo`: positions `[lo, ...)` hold drop-offs **at or after** `pickup − 15 min`.

So `ends[lo:hi]` are the trips that finished in the window:

```
pickup − 15min  ≤  dropoff  <  pickup
```

The start of the window is included and the end is excluded. A trip that drops off at the exact moment of pickup isn't counted, which is the safe choice for avoiding leakage. A trip's own drop-off always comes after its pickup, so it never counts itself.

### Step 5: Compute the medians

```python
vals = [np.median(mph[a:b]) if b > a else np.nan for a, b in zip(lo, hi, strict=True)]
out.loc[grp.index] = vals
```

- For each trip, take the speeds in its window slice and compute the median. If the slice is empty (`b == a`), use `NaN`.
- `strict=True` in `zip` raises an error if `lo` and `hi` have different lengths. It requires Python 3.10+.
- `out.loc[grp.index] = vals` writes the results back to the correct rows. This works because `vals` is in the same order as `grp`.

The **median** is used rather than the mean because it's robust to outliers, such as one trip with a bad GPS distance.

### A worked example

One zone with completed trips, `window = "15min"`:

| index | dropoff | mph |
|---|---|---|
| 0 | 10:00 | 10 |
| 1 | 10:05 | 14 |
| 2 | 10:12 | 12 |
| 3 | 10:20 | 8 |

A trip picked up at **10:15**:
- `starts − w` = 10:00 → `lo = 0` (10:00 is included)
- `starts` = 10:15 → `hi = 3` (10:20 is excluded because it's in the future)
- `mph[0:3]` = [10, 14, 12] → **median = 12**

A trip picked up at **10:00** gives `lo = hi = 0`, so the result is `NaN` (no earlier trips).

### Things worth knowing

- **Performance:** `done[done["pu_zone"] == zone]` scans all of `done` for every zone, so it slows down when there are many zones. Grouping `done` once (`dict(tuple(done.groupby("pu_zone")))`) avoids repeated scans. The median list comprehension is also a Python loop, which can be slow for millions of trips.
- **Timestamps:** if `dropoff_ts` and `pickup_ts` are timezone-aware, `.to_numpy()` returns an object array. `searchsorted` still works, but much more slowly. Both columns also need the same timezone handling, or the comparisons will be wrong.
- **Missing values:** trips with a `NaN` zone are dropped by `groupby` by default, so they stay `NaN` in the output.

<br><br>

## Code Explanation 2
```
what is the output of "grp". Can you give me a sample
```

`grp` is a regular **DataFrame**: the subset of `trips` rows that share one `pu_zone`. On each loop iteration, `zone` is the zone ID and `grp` holds that zone's rows.

### Sample

Suppose `trips` is this small table:

```
   pu_zone           pickup_ts          dropoff_ts  trip_distance  duration_s
0      161 2019-01-01 10:02:00 2019-01-01 10:12:00            1.8         600
1      237 2019-01-01 10:04:00 2019-01-01 10:18:00            2.5         840
2      161 2019-01-01 10:15:00 2019-01-01 10:27:00            2.1         720
3       48 2019-01-01 10:07:00 2019-01-01 10:25:00            4.0        1080
4      237 2019-01-01 10:20:00 2019-01-01 10:31:00            1.2         660
5      161 2019-01-01 10:30:00 2019-01-01 10:41:00            1.5         660
```

Then `for zone, grp in trips.groupby("pu_zone"):` runs three times:

```
zone = 48
   pu_zone           pickup_ts          dropoff_ts  trip_distance  duration_s
3       48 2019-01-01 10:07:00 2019-01-01 10:25:00            4.0        1080

zone = 161
   pu_zone           pickup_ts          dropoff_ts  trip_distance  duration_s
0      161 2019-01-01 10:02:00 2019-01-01 10:12:00            1.8         600
2      161 2019-01-01 10:15:00 2019-01-01 10:27:00            2.1         720
5      161 2019-01-01 10:30:00 2019-01-01 10:41:00            1.5         660

zone = 237
   pu_zone           pickup_ts          dropoff_ts  trip_distance  duration_s
1      237 2019-01-01 10:04:00 2019-01-01 10:18:00            2.5         840
4      237 2019-01-01 10:20:00 2019-01-01 10:31:00            1.2         660
```

(I ran this with pandas to confirm the output.)

### Things to notice

- **All columns are kept.** `grp` isn't just the pickup times; it has every column of `trips`.
- **The original index is kept.** For zone 161, the index is `[0, 2, 5]`, not `[0, 1, 2]`. This is what makes `out.loc[grp.index] = vals` write each median back to the correct trip.
- **Groups come in sorted order** of the zone key (48, 161, 237), because `groupby` sorts by default.
- **Rows inside a group keep their original order.** They aren't sorted by time. The function sorts `d` (the completed trips), but it doesn't need to sort `grp`, because `searchsorted` handles each pickup independently.

### What the function takes from `grp`

For zone 161:

```python
grp["pickup_ts"].to_numpy()
# array(['2019-01-01T10:02', '2019-01-01T10:15', '2019-01-01T10:30'], dtype='datetime64[ns]')

grp.index
# Index([0, 2, 5])
```

So `starts` holds the three pickup times, and once the medians are computed, `vals[0]`, `vals[1]` and `vals[2]` go to rows 0, 2 and 5 of `out`.

<br><br>

## Code Explanation 3
```python
def event_stream(trips: pd.DataFrame) -> Iterator[tuple[pd.Timestamp, TripStarted | TripCompleted]]:
    """Yield (event_time, event) for all trips, merged in event-time order."""
    starts = trips[["trip_id", "pickup_ts", "pu_zone", "do_zone"]].assign(kind=0, ts=trips["pickup_ts"])
    ends = trips[["trip_id", "pickup_ts", "dropoff_ts", "pu_zone", "do_zone", "duration_s",
                  "trip_distance", "fare_amount"]].assign(kind=1, ts=trips["dropoff_ts"])
    merged = pd.concat([starts, ends], ignore_index=True).sort_values(["ts", "kind"], kind="stable")
    for r in merged.itertuples(index=False):
        if r.kind == 0:
            yield r.ts, TripStarted(r.trip_id, r.pickup_ts.isoformat(), int(r.pu_zone), int(r.do_zone))
        else:
            yield r.ts, TripCompleted(
                r.trip_id, r.pickup_ts.isoformat(), r.dropoff_ts.isoformat(), int(r.pu_zone),
                int(r.do_zone), float(r.duration_s), float(r.trip_distance), float(r.fare_amount),
            )```
```

This function turns a table of trips into a **replay of events in time order**. Each trip produces two events: a `TripStarted` at pickup and a `TripCompleted` at drop-off. The function sorts all of them by time and hands them out one at a time, as if a live system were sending them. It's how historical data gets fed into the streaming consumer (`stream/consumer.py`) that the previous function was written to match.

### The signature

```python
def event_stream(trips: pd.DataFrame) -> Iterator[tuple[pd.Timestamp, TripStarted | TripCompleted]]:
```

- It's a **generator** because it uses `yield`. It doesn't build a list up front. Each loop iteration in the caller pulls out one event.
- Each item is a tuple `(event_time, event)`, where `event` is either a `TripStarted` or a `TripCompleted`. Those are probably dataclasses (or similar message classes) defined elsewhere in the project.

### Step 1: One row per trip start

```python
starts = trips[["trip_id", "pickup_ts", "pu_zone", "do_zone"]].assign(kind=0, ts=trips["pickup_ts"])
```

This keeps only what's known when a trip **begins**: ID, pickup time, pickup zone and destination zone. `.assign()` adds two new columns:
- `kind=0` marks the row as a start event.
- `ts` is the event time, which for a start is the pickup time.

### Step 2: One row per trip end

```python
ends = trips[[..., "dropoff_ts", ..., "duration_s", "trip_distance", "fare_amount"]].assign(kind=1, ts=trips["dropoff_ts"])
```

This keeps the same fields plus what's only known once the trip **finishes**: drop-off time, duration, distance and fare. `kind=1` marks it as an end event, and `ts` is the drop-off time.

The split matters. A start event leaves out duration and fare because nobody knows those at pickup. It's the same "no future information" idea as in `point_in_time_zone_speed`.

### Step 3: Merge and sort by time

```python
merged = pd.concat([starts, ends], ignore_index=True).sort_values(["ts", "kind"], kind="stable")
```

- `pd.concat` stacks the two tables. Columns that exist only in `ends` (like `fare_amount`) are filled with `NaN` in the start rows, but those values are never read.
- `ignore_index=True` gives the combined table a fresh 0, 1, 2, … index, so the copied indexes don't overlap.
- `sort_values(["ts", "kind"])` sorts by event time first. **When two events happen at the same moment, `kind` breaks the tie, so starts (0) come before completions (1).**
- `kind="stable"` here is the sort algorithm (an unrelated meaning of "kind"). A stable sort keeps the original order of rows that are still tied, so the output is the same on every run.

### Step 4: Turn rows into event objects

```python
for r in merged.itertuples(index=False):
    if r.kind == 0:
        yield r.ts, TripStarted(r.trip_id, r.pickup_ts.isoformat(), int(r.pu_zone), int(r.do_zone))
    else:
        yield r.ts, TripCompleted(...)
```

- `itertuples(index=False)` loops over rows as named tuples, so you write `r.trip_id` instead of `r["trip_id"]`. It's much faster than `iterrows()`.
- `kind` decides which event class to build.
- The conversions get the data ready to send as a message:
  - `.isoformat()` turns a Timestamp into a string like `"2019-01-01T10:00:00"`, which can go into JSON or Kafka.
  - `int(...)` and `float(...)` turn NumPy types (`numpy.int64`, `numpy.float64`) into plain Python numbers. Many serializers, including `json.dumps`, reject NumPy types.

### Sample run

Three trips:

| trip_id | pickup | dropoff |
|---|---|---|
| A | 10:00 | 10:12 |
| B | 10:05 | 10:09 |
| C | 10:12 | 10:30 |

After sorting, `merged` looks like this (I ran it to confirm):

```
  trip_id  kind                  ts
0       A     0 2019-01-01 10:00:00
1       B     0 2019-01-01 10:05:00
4       B     1 2019-01-01 10:09:00
2       C     0 2019-01-01 10:12:00   ← C starts...
3       A     1 2019-01-01 10:12:00   ← ...before A completes (same time)
5       C     1 2019-01-01 10:30:00
```

And the generator yields:

```
10:00  TripStarted(trip_id='A', pickup_ts='2019-01-01T10:00:00', pu_zone=161, do_zone=237)
10:05  TripStarted(trip_id='B', ...)
10:09  TripCompleted(trip_id='B', ..., duration_s=240.0, trip_distance=0.7, fare_amount=5.0)
10:12  TripStarted(trip_id='C', ...)
10:12  TripCompleted(trip_id='A', ..., duration_s=720.0, trip_distance=1.8, fare_amount=9.5)
10:30  TripCompleted(trip_id='C', ...)
```

The events come out in the order they happened, not trip by trip. B starts after A but finishes first.

### Why starts come before completions on a tie

This tie-break is what makes the stream match the training feature from your previous snippet. In `point_in_time_zone_speed`, a trip only counts if it was dropped off **strictly before** the pickup (`side="left"`). In this stream, at 10:12 C's start is processed before A's completion. So when the consumer computes C's zone speed, A hasn't finished yet as far as the consumer knows, and A isn't counted. Both code paths make the same choice. If the order were reversed, the streaming feature would include A and the training feature wouldn't, and the TODO's "verify this matches exactly" check would fail.

### Things worth knowing

- **Memory:** the generator yields lazily, but `merged` is built in full first, at twice the number of trips. For very large months you'd process data a day or a file at a time.
- **Timezones:** `.isoformat()` adds an offset like `+00:00` only if the timestamps are timezone-aware. The consumer has to parse whichever format it gets.
- **Missing values:** `int(r.pu_zone)` raises an error if a zone is `NaN`, so trips with a missing zone have to be dropped before this runs.

<br><br>

## Code Explanation 4
```python
def replay_zone_speed(trips: pd.DataFrame, window: pd.Timedelta = DEFAULT_WINDOW) -> pd.Series:
    """Run the STREAMING logic over historical events (no Kafka) and record, at every
    trip_started event, what the online store would hold for its pickup zone."""
    from taxieta.stream.events import TripCompleted, event_stream

    win = ZoneSpeedWindow(window)
    out: dict[str, float | None] = {}
    for ts, ev in event_stream(trips):
        if isinstance(ev, TripCompleted):
            win.add(ev.pu_zone, ts, ev.trip_distance, ev.duration_s)
        else:
            out[ev.trip_id] = win.median_speed(ev.pu_zone, ts)[0]
    return trips["trip_id"].map(out).astype(float)```
```

This function runs the **streaming feature logic over historical data** to see what the live system would have computed. It's the other half of the check in the `TODO(M4)` from `point_in_time_zone_speed`: one function computes the feature the batch way (for training), this one computes it the streaming way, and you compare the two.

### The big picture

| | `point_in_time_zone_speed` | `replay_zone_speed` |
|---|---|---|
| Used for | Training features | Checking the streaming logic |
| Method | Vectorized batch (`groupby` + `searchsorted`) | Event by event, like the live consumer |
| Needs Kafka? | No | No (events come from `event_stream`) |
| Output | `pd.Series`, one value per trip | Same shape, so the two can be compared |

If the streaming logic is correct, both functions should return the same number for every trip.

### Line by line

**1. Local import**
```python
from taxieta.stream.events import TripCompleted, event_stream
```
The import sits inside the function rather than at the top of the file. The usual reason is to **avoid a circular import** (the `stream` package probably imports from this module too), or to load the streaming code only when this function is actually called.

**2. Set up state**
```python
win = ZoneSpeedWindow(window)
out: dict[str, float | None] = {}
```
- `ZoneSpeedWindow` is the **same class the live consumer uses** to hold recent completed trips for each zone. It isn't shown here, but from how it's called it has two methods:
  - `add(zone, ts, distance, duration)` records a completed trip.
  - `median_speed(zone, ts)` returns a tuple whose first element is the median speed over the last `window` before `ts`, or `None` if there's no data. The second element is probably a count.
- `out` maps `trip_id` to the feature value recorded at that trip's pickup.

Reusing the production class is the point of the function. You're testing the real streaming code, not a copy of it.

**3. Replay the events**
```python
for ts, ev in event_stream(trips):
    if isinstance(ev, TripCompleted):
        win.add(ev.pu_zone, ts, ev.trip_distance, ev.duration_s)
    else:
        out[ev.trip_id] = win.median_speed(ev.pu_zone, ts)[0]
```
`event_stream` yields events in time order, starts before completions on a tie. For each event:
- **A trip completed:** add its speed to its pickup zone's window. Now later trips from that zone can see it.
- **A trip started:** ask the window, "what's the median speed in this zone right now?" and save the answer under this trip's ID. This is exactly what the online store would have held at that moment.

`isinstance` picks the event type. Anything that isn't a `TripCompleted` is treated as a `TripStarted`.

**4. Line the results up with `trips`**
```python
return trips["trip_id"].map(out).astype(float)
```
- `.map(out)` looks up each row's `trip_id` in the dict, giving a Series with the **same index as `trips`**, just like the batch function's output.
- `.astype(float)` converts `None` values to `NaN`, so "no data" looks the same in both functions.

### Worked example

`window = 15min`, all trips in zone 161:

| trip | pickup | dropoff | distance | duration | mph |
|---|---|---|---|---|---|
| A | 10:00 | 10:10 | 2.0 mi | 600 s | 12 |
| B | 10:02 | 10:08 | 1.0 mi | 360 s | 10 |
| C | 10:15 | 10:30 | … | … | … |
| D | 10:24 | 10:40 | … | … | … |

What happens at each event:

| time | event | action | window contents | recorded |
|---|---|---|---|---|
| 10:00 | A starts | query | empty | A → NaN |
| 10:02 | B starts | query | empty | B → NaN |
| 10:08 | B completes | add 10 mph | [B] | |
| 10:10 | A completes | add 12 mph | [B, A] | |
| 10:15 | C starts | query 10:00–10:15 | [B, A] | C → **11** |
| 10:24 | D starts | query 10:09–10:24 | [A] (B dropped at 10:08, too old) | D → **12** |
| 10:30, 10:40 | C, D complete | add | | |

Result: `[NaN, NaN, 11.0, 12.0]` for rows A, B, C, D. The batch function should return the same numbers.

### How you'd use it

```python
batch  = point_in_time_zone_speed(day_trips)
replay = replay_zone_speed(day_trips)
assert np.allclose(batch, replay, equal_nan=True)
```

Running this on one day of data is the "M2 task C" check that the TODO mentions.

### Where the two could disagree

All of these live inside `ZoneSpeedWindow`, which isn't shown, so they're worth checking:

- **Filters:** the batch version drops trips with `duration_s <= 60` or `trip_distance <= 0`. `ZoneSpeedWindow.add` has to apply the same filter, or the medians will differ.
- **Window edges:** the batch version includes a drop-off exactly at `pickup − 15min` and excludes one exactly at pickup. The exclusion at pickup is already handled by `event_stream` putting starts first. The window's eviction has to **keep** entries where `dropoff >= now − window` and only drop strictly older ones.
- **Median definition:** with an even number of values, `np.median` averages the two middle ones (10 and 12 → 11). If the window uses a different method, such as picking the lower middle value, results will differ slightly.
- **Duplicate `trip_id`s:** the dict keeps only the last value written for each ID, so a repeated ID would overwrite another trip's result. IDs need to be unique.