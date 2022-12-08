
from src import environments    # The code to test
from src.environments import default_options
from src.display import DisplayVisitor


import unittest   # The test framework
import numpy as np

class MockEnv:
    pass

class Test_Display(unittest.TestCase):

    def setUp(self) -> None:
        self.display = DisplayVisitor(default_options)
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_unpacks_options(self):
        options = {
            'screen_size': np.random.randint(200, 1000),
            'render_mode': None,
            'render_delay': 0 # in seconds
        }
        env = MockEnv(options)
        # test if each option is set as an attrubet of the envioronment wi the same name
        attributes = vars(env)
        for opt in options:
            # self.assertIn(opt, attributes)
            self.assertIn(opt, attributes)

    def test_display(self):
        self.assertEqual(True, False)

    def test_calculates_propper_number_of_rows_for_visible_range(self):
        vis_ranges = np.arange(0,5)
        # rows should be of length, 1, 3, 5, 7, ...
        for i, vis_range in enumerate(vis_ranges):
            row_sizes = self.display.get_row_sizes_for_visible_range(vis_range)
            self.assertEqual(len(row_sizes), (2*i+1))

    def test_calculates_propper_row_sizes_for_visible_range(self):
        vis_range = 2
        row_sizes = self.display.get_row_sizes_for_visible_range(vis_range)
        self.assertEqual(row_sizes[0], 1)
        self.assertEqual(row_sizes[1], 3)
        self.assertEqual(row_sizes[2], 5)
        self.assertEqual(row_sizes[3], 3)
        self.assertEqual(row_sizes[4], 1)

if __name__ == '__main__':
    unittest.main()
