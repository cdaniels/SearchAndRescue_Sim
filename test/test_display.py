
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

if __name__ == '__main__':
    unittest.main()
