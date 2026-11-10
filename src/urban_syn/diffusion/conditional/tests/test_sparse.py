"""Regression checks for neutral missing fields, partial inputs, and training-only fits."""
import copy
import unittest

import pandas as pd
import torch

from ..conditioning import SparseMetadataEncoder, encode_requests
from ...models.factory import make_model
from ..sparse_metadata import (POI_FIELD, POI_REASON_AMBIGUOUS, POI_REASON_GAP,
                               POI_REASON_NO_STOP, POI_REASON_UNAVAILABLE,
                               _prepare_poi_assignments, build_sparse_sidecars,
                               fit_fields, load_sparse_split, transform_rows)
from . import test_pilot as fixtures


def fields():
    return [
        {"name": "weather_category", "kind": "categorical",
         "vocabulary": {"<unknown>": 0, "sunny": 1, "rain": 2}},
        {"name": "transportation_mode", "kind": "categorical",
         "vocabulary": {"<unknown>": 0, "bus": 1, "walk": 2}},
        {"name": "temperature", "kind": "continuous",
         "normalization": {"mean": 10., "scale": 5., "observed_training_rows": 2}},
    ]


class SparseSidecarTests(unittest.TestCase):
    setUp = fixtures.AlignmentTests.setUp

    def build_sparse(self):
        return build_sparse_sidecars(self.dataset, self.trip_map, self.fmm, self.test,
                                     self.weather, self.root / "sparse")

    def test_partial_fields_do_not_drop_rows_or_leak_test_statistics(self):
        mapping = pd.read_csv(self.trip_map)
        mapping.loc[mapping.id == 3, "label"] = None
        mapping.to_csv(self.trip_map, index=False)
        weather = pd.read_parquet(self.weather)
        weather["TAVG_C"] = [10., None, 99.]
        weather.to_parquet(self.weather, index=False)
        manifest = self.build_sparse()
        temperature = next(field for field in manifest["fields"] if field["name"] == "temperature")
        self.assertEqual(temperature["normalization"],
                         dict(mean=10., scale=1., observed_training_rows=1, ddof=0))
        ids, _, metadata, rows, _ = load_sparse_split(self.dataset, self.root / "sparse")
        self.assertEqual(sum(map(len, ids.values())), 3)
        self.assertEqual(metadata[3]["masks"]["temperature"].tolist(), [True, False, False])
        self.assertEqual(metadata[3]["masks"]["transportation_mode"].tolist(), [True, True, False])
        self.assertEqual(metadata[3]["masks"]["weather_category"].tolist(), [True, True, False])
        self.assertEqual(rows.fmm_id.tolist(), [1, 2, 3])
        _, _, held_out, _, _ = load_sparse_split(self.dataset, self.root / "sparse", "gen_test")
        self.assertEqual(held_out[2]["values"]["temperature"].item(), 89.)
        self.assertFalse(held_out[2]["masks"]["weather_category"].item())

    def test_entire_feature_absent_is_valid(self):
        # Fixture has no TAVG_C column at all: still retain all trajectories.
        manifest = self.build_sparse()
        descriptor = next(field for field in manifest["fields"] if field["name"] == "temperature")
        self.assertEqual(descriptor["normalization"]["observed_training_rows"], 0)
        _, _, metadata, _, _ = load_sparse_split(self.dataset, self.root / "sparse")
        self.assertFalse(metadata[3]["masks"]["temperature"].any())
        self.assertEqual(metadata[3]["values"]["temperature"].tolist(), [0., 0., 0.])

    def test_normalization_uses_known_training_values_only(self):
        training = pd.DataFrame({"temperature": [10., 20., None]})
        schema = fit_fields(training, ["temperature"])
        self.assertEqual(schema[0]["normalization"]["mean"], 15.)
        self.assertEqual(schema[0]["normalization"]["scale"], 5.)
        result = transform_rows(pd.DataFrame({"temperature": [15., None, 1000.]}), schema)
        self.assertEqual(result.temperature_normalized.tolist(), [0., 0., 197.])
        self.assertEqual(result.temperature_available.tolist(), [True, False, True])


class SparseEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(2)
        self.schema = fields()
        self.encoder = SparseMetadataEncoder(self.schema, all_dropout=0.1, field_dropout=0.1)
        self.encoder.eval()
        self.device = torch.device("cpu")

    def encode(self, metadata):
        return self.encoder(metadata, 2, self.device, torch.float32)

    def test_unknown_is_exact_zero_and_placeholders_are_ignored(self):
        null = encode_requests([{}, {"weather_category": "unseen"}], self.schema)
        with torch.no_grad():
            for field in self.schema:
                if field["kind"] == "categorical":
                    self.encoder.encoders[field["name"]].weight[0].fill_(123)
        self.assertEqual(torch.count_nonzero(self.encode(null)).item(), 0)
        self.assertTrue(torch.equal(self.encode(None), self.encode(null)))
        for name, value in null["values"].items():
            value.fill_(999 if value.dtype == torch.long else float("nan"))
        self.assertEqual(torch.count_nonzero(self.encode(null)).item(), 0)

    def test_known_fields_add_and_partial_missing_is_neutral(self):
        requests = [{"weather_category": "sunny", "transportation_mode": "bus", "temperature": 10.}] * 2
        combined = self.encode(encode_requests(requests, self.schema))
        parts = [self.encode(encode_requests([{name: row[name]} for row in requests], self.schema))
                 for name in requests[0]]
        torch.testing.assert_close(combined, sum(parts))
        # A known mean-valued temperature may affect the model; absent temperature never does.
        self.assertGreater(parts[-1].abs().sum().item(), 0)
        partial = encode_requests([{"weather_category": "sunny"}] * 2, self.schema)
        del partial["values"]["transportation_mode"]
        del partial["masks"]["transportation_mode"]
        self.assertTrue(torch.equal(self.encode(partial), parts[0]))

    def test_dropout_is_training_only_and_can_train_null_path(self):
        known = encode_requests([{"weather_category": "sunny", "temperature": 20.}] * 2, self.schema)
        self.encoder.train()
        self.encoder.all_dropout, self.encoder.field_dropout = 1., 0.
        self.assertEqual(torch.count_nonzero(self.encode(known)).item(), 0)
        self.encoder.all_dropout, self.encoder.field_dropout = 0., 1.
        self.assertEqual(torch.count_nonzero(self.encode(known)).item(), 0)
        self.encoder.eval()
        self.assertGreater(self.encode(known).abs().sum().item(), 0)

    def test_invalid_known_inputs_are_rejected(self):
        batch = encode_requests([{"weather_category": "sunny"}] * 2, self.schema)
        for key, value, exception in [
            ("weather_category", torch.tensor([1.0, 2.0]), TypeError),
            ("weather_category", torch.tensor([99, 1]), ValueError),
            ("weather_category", torch.tensor([1]), ValueError),
        ]:
            invalid = copy.deepcopy(batch)
            invalid["values"][key] = value
            with self.assertRaises(exception):
                self.encode(invalid)
        with self.assertRaises(ValueError):
            encode_requests([{"typo": "sunny"}], self.schema)

    def test_zero_condition_still_trains_denoiser_and_generates(self):
        config = dict(channels=8, steps=2, pre_dim=4, residual_layers=1, condition_dim=8,
                      metadata_fields=self.schema, all_dropout=.1, field_dropout=.1,
                      beta_start=1e-4, beta_end=.02)
        model = make_model(config)
        graph, embeddings = torch.empty(0, dtype=torch.long), torch.randn(8, 4)
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        losses = model(graph, embeddings, ids, torch.zeros(2, 3), None)
        sum(losses).backward()
        self.assertGreater(model.diff_model.input_projection.weight.grad.abs().sum().item(), 0)
        model.eval()
        x, t = torch.randn(2, 3, 5), torch.ones(2, 1, dtype=torch.long)
        null = encode_requests([{}, {}], self.schema)
        self.assertTrue(torch.equal(model.diff_model(x, t, None), model.diff_model(x, t, null)))
        generator = lambda: torch.Generator().manual_seed(123)
        a = model.generate_data(graph, embeddings, 2, 3, 4, None, generator=generator())
        b = model.generate_data(graph, embeddings, 2, 3, 4, null, generator=generator())
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(a, b)))
        self.assertEqual(a[0].shape, (2, 3))


class PoiSelectionTests(unittest.TestCase):
    def _select(self, trip_rows, stop_rows, poi_rows):
        trip_map = pd.DataFrame(trip_rows)
        # Keep compact relative test times while exercising the real Unix-second
        # contract used by the trajectory-ID map.
        epoch = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp())
        trip_map["start_timestamp"] = trip_map.start_timestamp + epoch
        stops = pd.DataFrame(stop_rows)
        poi = pd.DataFrame(poi_rows)
        return _prepare_poi_assignments({}, trip_map, poi, stops).set_index(["uid", "tid", "move_id"])

    @staticmethod
    def _stop(uid, tid, arrival, departure):
        return {"uid": uid, "tid": tid, "datetime": pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=arrival),
                "leaving_datetime": pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=departure)}

    @staticmethod
    def _poi(uid, tid, arrival, departure, category="shop", distance=1, index_poi=1,
             element_type="node", osmid=1):
        return {"uid": uid, "tid": tid,
                "datetime": pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=arrival),
                "leaving_datetime": pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=departure),
                "category": category, "distance": distance, "index_poi": index_poi,
                "element_type": element_type, "osmid": osmid}

    def test_gap_boundaries_future_stop_and_no_backward_fallback(self):
        trips = [
            {"uid": "zero", "tid": "t", "move_id": "m", "start_timestamp": 1000},
            {"uid": "nine", "tid": "t", "move_id": "m", "start_timestamp": 1900},
            {"uid": "over", "tid": "t", "move_id": "m", "start_timestamp": 1901},
            {"uid": "future", "tid": "t", "move_id": "m", "start_timestamp": 1000},
            {"uid": "fallback", "tid": "t", "move_id": "m", "start_timestamp": 1900},
        ]
        stops = [self._stop("zero", "t", 900, 1000),
                 self._stop("nine", "t", 900, 1000),
                 self._stop("over", "t", 900, 1000),
                 self._stop("future", "t", 900, 1100),
                 self._stop("fallback", "t", 400, 500),
                 self._stop("fallback", "t", 900, 1000)]
        pois = [self._poi("zero", "t", 900, 1000),
                self._poi("nine", "t", 900, 1000),
                self._poi("fallback", "t", 400, 500, category="old")]
        result = self._select(trips, stops, pois)
        self.assertEqual(result.loc[("zero", "t", "m"), POI_FIELD], "shop")
        self.assertEqual(result.loc[("nine", "t", "m"), POI_FIELD], "shop")
        self.assertEqual(result.loc[("nine", "t", "m"), f"{POI_FIELD}_reason"], "known")
        self.assertEqual(result.loc[("over", "t", "m"), f"{POI_FIELD}_reason"], POI_REASON_GAP)
        self.assertEqual(result.loc[("future", "t", "m"), f"{POI_FIELD}_reason"], POI_REASON_NO_STOP)
        self.assertEqual(result.loc[("fallback", "t", "m"), f"{POI_FIELD}_reason"], POI_REASON_UNAVAILABLE)

    def test_distance_boundaries_duplicates_shuffle_and_tie_order(self):
        trips = [{"uid": "u", "tid": "t", "move_id": "m", "start_timestamp": 1000},
                 {"uid": "dup", "tid": "t", "move_id": "m", "start_timestamp": 1000},
                 {"uid": "tie", "tid": "t", "move_id": "m", "start_timestamp": 1000}]
        stops = [self._stop("u", "t", 900, 1000), self._stop("dup", "t", 900, 1000),
                 self._stop("tie", "t", 900, 1000)]
        pois = [
            self._poi("u", "t", 900, 1000, category="edge", distance=0, index_poi=1),
            self._poi("u", "t", 900, 1000, category="limit", distance=50, index_poi=2),
            self._poi("dup", "t", 900, 1000, category="a", distance=2, index_poi=3),
            self._poi("dup", "t", 900, 1000, category="b", distance=3, index_poi=3),
            self._poi("tie", "t", 900, 1000, category="z", distance=1, index_poi=5, osmid=20),
            self._poi("tie", "t", 900, 1000, category="a", distance=1, index_poi=4, osmid=10),
        ]
        result = self._select(trips, stops, pois)
        self.assertEqual(result.loc[("u", "t", "m"), POI_FIELD], "edge")
        self.assertEqual(result.loc[("dup", "t", "m"), f"{POI_FIELD}_reason"], POI_REASON_AMBIGUOUS)
        self.assertEqual(result.loc[("tie", "t", "m"), POI_FIELD], "a")
        shuffled = self._select(trips, stops, list(reversed(pois)))
        pd.testing.assert_frame_equal(result, shuffled)
        out_of_range = self._select(
            [{"uid": "far", "tid": "t", "move_id": "m", "start_timestamp": 1000}],
            [self._stop("far", "t", 900, 1000)],
            [self._poi("far", "t", 900, 1000, distance=50.000001)])
        self.assertEqual(out_of_range.iloc[0][f"{POI_FIELD}_reason"], POI_REASON_UNAVAILABLE)

    def test_poi_vocabulary_is_training_only_and_unseen_is_unknown(self):
        schema = fit_fields(pd.DataFrame({POI_FIELD: ["cafe", "cafe", None]}), [POI_FIELD])
        self.assertEqual(schema[0]["vocabulary"], {"<unknown>": 0, "cafe": 1})
        rows = pd.DataFrame({POI_FIELD: ["cafe", "park", "<unknown>"],
                             f"{POI_FIELD}_reason": ["known", "known", POI_REASON_NO_STOP]})
        transformed = transform_rows(rows, schema)
        self.assertEqual(transformed[f"{POI_FIELD}_code"].tolist(), [1, 0, 0])
        self.assertEqual(transformed[f"{POI_FIELD}_available"].tolist(), [True, False, False])
        self.assertEqual(transformed[f"{POI_FIELD}_reason"].tolist(),
                         ["known", "unseen_training_category", POI_REASON_NO_STOP])


if __name__ == "__main__":
    unittest.main()
