"""Offline output-contract checks; no Docker, training, or host ownership changes."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lakehouse.container_entrypoint import OUTPUTS, main, output_errors


class ContainerEntrypointTests(unittest.TestCase):
    def test_new_outputs_are_allowed_under_writable_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(output_errors(Path(directory)), [])

    def test_existing_output_access_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / OUTPUTS[0]
            output.parent.mkdir()
            output.write_text("keep")
            with patch("lakehouse.container_entrypoint.os.access", side_effect=lambda p, mode: p != output.resolve()):
                self.assertTrue(any("existing output" in e for e in output_errors(root)))
            self.assertEqual(output.read_text(), "keep")

    def test_unwritable_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("lakehouse.container_entrypoint.os.access", return_value=False):
                self.assertTrue(any("parent" in e for e in output_errors(Path(directory))))

    def test_directory_cannot_replace_output_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / OUTPUTS[0]).mkdir(parents=True)
            self.assertTrue(any("existing output" in e for e in output_errors(root)))

    def test_symlink_escape_is_rejected_without_touching_target(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / "lakehouse").symlink_to(outside, target_is_directory=True)
            self.assertTrue(any("outside project" in e for e in output_errors(root)))
            self.assertEqual(list(Path(outside).iterdir()), [])

    def test_root_never_executes_command(self):
        with patch("os.geteuid", return_value=0), patch("os.execvp") as execute:
            self.assertEqual(main(), 2)
            execute.assert_not_called()

    def test_dangling_internal_output_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lakehouse").mkdir()
            (root / OUTPUTS[0]).symlink_to(root / "missing" / "database")
            self.assertTrue(any("dangling symlink" in e for e in output_errors(root)))
            self.assertFalse((root / "missing").exists())

    def test_dangling_internal_parent_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lakehouse").symlink_to(root / "missing", target_is_directory=True)
            self.assertTrue(any("dangling symlink" in e for e in output_errors(root)))
            self.assertFalse((root / "missing").exists())

    def test_preflight_failure_never_executes_command(self):
        with patch("os.geteuid", return_value=10001), patch(
            "lakehouse.container_entrypoint.output_errors", return_value=["blocked"]
        ), patch("os.execvp") as execute:
            self.assertEqual(main(), 2)
            execute.assert_not_called()
