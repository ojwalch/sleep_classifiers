import os
from builtins import FileNotFoundError
from unittest import TestCase

from numpy.core.multiarray import ndarray


class TestHelper(TestCase):
    @staticmethod
    def remove_file(path):
        try:
            os.remove(str(path.resolve()))
        except OSError or FileNotFoundError:
            pass

    @staticmethod
    def assert_models_equal(test_case, object1, object2):

        for key in object1.__dict__:
            if not type(object1.__dict__[key]) == (type(object2.__dict__[key])):
                test_case.fail('Types do not match')

            if key not in object2.__dict__:
                test_case.fail("Missing Key")
            try:
                value1_as_float = float(object1.__dict__[key])
                value2_as_float = float(object2.__dict__[key])
                if abs(value1_as_float - value2_as_float) > 0.0000000001:
                    test_case.fail("Float values do not match. " + str(value1_as_float) + " does not equal "
                                   + str(value2_as_float))

            except ValueError:
                    test_case.assertEqual(object1.__dict__[key], object2.__dict__[key])
            except TypeError:

                if type(object1.__dict__[key]) == ndarray:
                    test_case.assertListEqual(object1.__dict__[key].tolist(), object2.__dict__[key].tolist())

                if type(object1.__dict__[key]) == list:
                    test_case.assertListEqual(object1.__dict__[key], object2.__dict__[key])
