import unittest

from replay import replay


class ReplayTest(unittest.TestCase):
    def test_window_boundary_user_isolation_and_equal_timestamps(self):
        events = [(0, 1), (0, 1), (0, 2), (600000, 1), (600001, 1), (600001, 2)]
        self.assertEqual(list(replay(events)),
                         [(0, 1, 0), (0, 1, 0), (0, 2, 0),
                          (600000, 1, 2), (600001, 1, 1), (600001, 2, 0)])


if __name__ == "__main__":
    unittest.main()
