import io
import logging
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path

from alloylm.utils import get_logger


class TestGetLogger(unittest.TestCase):
    def setUp(self):
        self.logger_names = []

    def tearDown(self):
        for name in self.logger_names:
            logger = logging.getLogger(name)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()

    def logger_name(self, suffix):
        name = f"test.{suffix}.{uuid.uuid4()}"
        self.logger_names.append(name)
        return name

    def test_file_and_stdout_have_expected_levels_and_format(self):
        name = self.logger_name("outputs")
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(stdout):
            path = Path(directory) / "test.log"
            logger = get_logger(name, path=path, output_to_stdout=True)
            logger.debug("debug message")
            logger.info("info message")

            file_output = path.read_text()

        self.assertIn(f"[AlloyLM][{name}]", file_output)
        self.assertIn("[DEBUG] debug message", file_output)
        self.assertIn("[INFO] info message", file_output)
        self.assertNotIn("debug message", stdout.getvalue())
        self.assertIn("[INFO] info message", stdout.getvalue())

    def test_named_loggers_write_to_separate_files(self):
        first_name = self.logger_name("first")
        second_name = self.logger_name("second")
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.log"
            second_path = Path(directory) / "second.log"
            first = get_logger(first_name, path=first_path, output_to_stdout=False)
            second = get_logger(second_name, path=second_path, output_to_stdout=False)

            first.info("first only")
            second.info("second only")

            self.assertIn("first only", first_path.read_text())
            self.assertNotIn("second only", first_path.read_text())
            self.assertIn("second only", second_path.read_text())
            self.assertNotIn("first only", second_path.read_text())

    def test_force_recreate_replaces_existing_handlers(self):
        name = self.logger_name("recreate")
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.log"
            second_path = Path(directory) / "second.log"
            logger = get_logger(name, path=first_path, output_to_stdout=False)
            logger.info("before")

            recreated = get_logger(name, path=second_path, output_to_stdout=False, force_recreate=True)
            recreated.info("after")

            self.assertIs(logger, recreated)
            self.assertIn("before", first_path.read_text())
            self.assertNotIn("after", first_path.read_text())
            self.assertIn("after", second_path.read_text())


if __name__ == "__main__":
    unittest.main()
