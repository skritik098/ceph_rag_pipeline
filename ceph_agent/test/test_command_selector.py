import unittest
from utils.file_ops import build_index, load_index, get_command_output_with_context
import os

class TestCommandSelectionAndContextualization(unittest.TestCase):

    def setUp(self):
        self.data_file = "sample_command_data.json"
        self.index_dir = "test_faiss_index"
        self.query = "How to check OSD usage?"
        self.model = "llama3"

        # Build and load the index for testing
        build_index(self.data_file, self.index_dir)
        self.index, self.commands = load_index(self.index_dir)

    def test_osd_usage_query(self):
        response = get_command_output_with_context(
            self.query,
            self.index,
            self.commands,
            model=self.model,
            top_k=5
        )
        self.assertTrue(isinstance(response, str))
        self.assertIn("OSD", response.upper())

    def test_invalid_query(self):
        invalid_query = "Launch a satellite using ceph"
        response = get_command_output_with_context(
            invalid_query,
            self.index,
            self.commands,
            model=self.model,
            top_k=5
        )
        self.assertTrue(isinstance(response, str))

    def tearDown(self):
        if os.path.exists(self.index_dir):
            for f in os.listdir(self.index_dir):
                os.remove(os.path.join(self.index_dir, f))
            os.rmdir(self.index_dir)

if __name__ == "__main__":
    unittest.main()
