import unittest
from pathlib import Path
from scripts.download_pilot_assets import Confirmation, target_path


class DownloadTests(unittest.TestCase):
    def test_parse_confirmation_without_using_arbitrary_links(self):
        form = Confirmation()
        form.feed('<form id="download-form" action="https://drive.usercontent.google.com/download">'
                  '<input type="hidden" name="id" value="asset"><input name="confirm" value="t"></form>'
                  '<input name="ignored" value="outside">')
        self.assertEqual(form.fields, {'id': 'asset', 'confirm': 't'})

    def test_asset_cannot_escape_download_root(self):
        root = Path('test-download-root').resolve()
        self.assertEqual(target_path(root, 'data/sample.png'), root / 'data/sample.png')
        for value in ['../outside', str(root.parent / 'outside')]:
            with self.assertRaises(ValueError):
                target_path(root, value)
