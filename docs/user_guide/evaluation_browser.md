# Looking at a data set in the evaluation browser

Audience: you have a data set (from `sensoryforge evaluate run`, or `sensoryforge batch
--dataset`) and want to look at it rather than read a table. By the end you will have opened a
draw, followed one sensor or unit through the pipeline, found the draws that look wrong and seen
how the draws spread over the declared classes.

```bash
sensoryforge evaluate serve runs/first        # then open http://127.0.0.1:8765
```

The server listens on the loopback address only and reads the folder as it is. It reads the stage
files (`audit.json`, `declared.json`, `compare.json`, `checks.json`) from `runs/first/evaluate/`;
name another folder with `--stages`. A page whose stage file is missing says so and, for the
checks, offers to compute it.

## Draws

The front page lists the draws. Filter by split or class, search an entry id, or give a draw seed;
**Random draw** picks one (type a pick seed to get the same one again). Opening a draw plays its
stimulus as a movie, x to the right and y upwards, with a scrubber and a speed control from 0.05
to 20 times real time. The colour range is the audit's peak for the draw when it has one.

Click the movie to **pin** a point. *Pin by* chooses what a click means: a sensor, or a unit of a
population. Under the movie, linked traces follow the pinned point: the stimulus at a sensor; the
drive, the filtered current and the readout of a unit; and the unit's membrane voltage, which the
server re-simulates from the stored filtered current. Every re-run is compared with the stored
spikes, events or state and the page says whether it matches; a bundle written with voltage probes
also has its stored voltage compared sample by sample. A cursor on the traces follows the movie.

**Declared fields** lists what the world declared for this draw. **Compare with…** opens a second
draw beside the first, with its traces laid over the first's, and the declared fields marked where
the two differ.

## Dashboard

The whole data set at once: counts by split, class, kind and status; the class-by-split table;
histograms of entry length, stimulus peak and response rate per population; and a coverage map of
the values each declared axis actually took. Every bar, count and cell is a link to the draws
behind it, so a surprising number is one click from the draws that made it.

The counts and the coverage read the declaration; the histograms of measured quantities read only
the audit, as the audit does. The page says where each number comes from.

## Gallery

Random sampling misses rare failures, so the gallery ranks every draw by what looks wrong and shows
the worst first: bad values, silent or saturated populations, rates far from their class, long
quiet gaps, stimulus that looks like another class, and entries that did not run. Each card lists
its reasons and opens its draw. Tick signals and choose how many to show. Groups of draws that are
silent together, which points at the sensor rather than at one draw, are listed apart.

## Checks

The reference-free checks (the simulator's own consistency tests that need no declaration) with
their pass, fail and error counts. The page can run the checks for the folder on request.

## Next

- [Sharing a snapshot](evaluation_export.md): the same pages as a folder that opens offline.
- [Evaluating a world in one command](evaluate_pipeline.md): produces the folder.
- [The local evaluation server](../developer_guide/evaluation_server.md): endpoints, streaming and timings.
