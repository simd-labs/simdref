"""SIMDREF_RELEASE_TAG pins the release tag that `update --from-release` reads."""

import os
import unittest
from unittest import mock

from simdref import cli


class ReleaseTagPinTests(unittest.TestCase):
    def test_default_prefers_the_versioned_tag(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("SIMDREF_RELEASE_TAG", None)
            self.assertEqual(
                cli._release_tag_candidates(), [f"data-v{cli.__version__}", "data-latest"]
            )

    def test_env_pins_one_tag(self):
        with mock.patch.dict(os.environ, {"SIMDREF_RELEASE_TAG": "data-latest"}):
            self.assertEqual(cli._release_tag_candidates(), ["data-latest"])


if __name__ == "__main__":
    unittest.main()
