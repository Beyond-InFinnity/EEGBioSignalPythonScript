import unittest
import json
import tempfile
import os
from src import ibirsa
import numpy as np

class TestParsing(unittest.TestCase):
    def test_array_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "arr.json")
            data = [{"ts": 1700000000000, "rr": [800, 900]}]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            recs = ibirsa.load_stream(path)
            self.assertEqual(len(recs), 1)

    def test_ndjson(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nd.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"ts": 1700000000000, "rr": [800]}))
                f.write("\n")
                f.write(json.dumps({"ts": 1700000001000, "rr": [850]}))
            recs = ibirsa.load_stream(path)
            self.assertEqual(len(recs), 2)

    def test_ts_unit_detection(self):
        arr_ms = np.array([1700000000000, 1700000005000], dtype=float)
        arr_s = np.array([1700000000, 1700000010], dtype=float)
        self.assertEqual(ibirsa.detect_ts_unit(arr_ms), "ms")
        self.assertEqual(ibirsa.detect_ts_unit(arr_s), "s")

if __name__ == "__main__":
    unittest.main()
