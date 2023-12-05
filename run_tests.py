import unittest

if __name__ == '__main__':
    tests = unittest.TestLoader().discover('path_to_test_directory', pattern='test_*.py')
    unittest.TextTestRunner().run(tests)