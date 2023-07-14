import unittest
import argparse

import modal

from .test_common import run_unittest_remote, stub


@stub.local_entrypoint()
def main():
    run_unittest_remote.call()


if __name__ == "__main__":
    unittest.main()
