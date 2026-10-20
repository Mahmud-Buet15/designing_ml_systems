"""M2 solution: the batch and streaming versions of the zone-speed feature must agree exactly."""

import numpy as np

from taxieta.ingest.lake import read_lake
from taxieta.stream.consumer import batch_zone_speed, replay_zone_speed


def test_batch_and_streaming_zone_speed_agree(lake):
    trips = read_lake([(2019, 1)]).head(3000).reset_index(drop=True)
    b, s = batch_zone_speed(trips), replay_zone_speed(trips)
    assert np.allclose(b.fillna(-1), s.fillna(-1))