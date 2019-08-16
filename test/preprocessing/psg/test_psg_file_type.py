from unittest import TestCase

from source.preprocessing.psg.psg_file_type import PSGFileType


class TestPSGFileType(TestCase):
    def test_file_types_exists(self):
        self.assertEqual(PSGFileType.Vitaport.value, 0)
        self.assertEqual(PSGFileType.Compumedics.value, 1)
