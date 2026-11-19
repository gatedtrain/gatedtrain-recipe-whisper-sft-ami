"""The adapter `hf_seq2seq` conforms to gatedtrain's adapter interface (gatedtrain.ext.conformance)."""

import unittest

from gatedtrain.ext.conformance import AdapterConformance


class TestConformance(AdapterConformance, unittest.TestCase):
    ADAPTER = "hf_seq2seq"


if __name__ == "__main__":
    unittest.main()
