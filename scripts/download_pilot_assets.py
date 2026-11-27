"""Download the shared pilot's reload inputs without placing large files in Git.

Uses the supplied Drive file IDs and a normal Drive download confirmation form.
No credentials or session URLs are written to the manifest or logs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urlparse


class Confirmation(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action, self.fields, self.active = None, {}, False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form' and attrs.get('id') == 'download-form':
            self.action, self.active = attrs.get('action'), True
        if tag == 'input' and self.active and 'name' in attrs:
            self.fields[attrs['name']] = attrs.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form':
            self.active = False


def target_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or root not in path.parents:
        raise ValueError(f'Asset path must stay within the download root: {relative}')
    return path


def download_file(spec, root):
    import requests
    path = target_path(root, spec['path'])
    if path.exists():
        if path.stat().st_size != spec['bytes']:
            raise ValueError(f'Existing file has wrong size; inspect before replacing: {path}')
        return {'path': spec['path'], 'bytes': spec['bytes'], 'status': 'already_present_size_matched'}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.download-part')
    if temporary.exists():
        raise FileExistsError(f'Incomplete prior download retained: {temporary}')
    with requests.Session() as session:
        response = session.get('https://drive.google.com/uc', params={'export': 'download', 'id': spec['id']},
                               stream=True, timeout=(30, 120))
        response.raise_for_status()
        if 'text/html' in response.headers.get('content-type', ''):
            form = Confirmation()
            form.feed(response.text)
            response.close()
            parsed = urlparse(form.action or '')
            if parsed.scheme != 'https' or parsed.hostname not in {'drive.google.com', 'drive.usercontent.google.com'}:
                raise RuntimeError(f'Drive download unavailable; check file access for {spec["path"]}')
            if form.fields.get('id') != spec['id']:
                raise RuntimeError('Confirmation refers to a different file')
            response = session.get(form.action, params=form.fields, stream=True, timeout=(30, 120))
            response.raise_for_status()
        with response:
            if 'text/html' in response.headers.get('content-type', ''):
                raise RuntimeError('Drive returned a page instead of file bytes')
            with temporary.open('xb') as handle:
                for block in response.iter_content(1024 * 1024):
                    if block:
                        handle.write(block)
    if temporary.stat().st_size != spec['bytes']:
        raise ValueError(f'Download byte count mismatch; partial retained: {temporary}')
    temporary.rename(path)
    return {'path': spec['path'], 'bytes': spec['bytes'], 'status': 'downloaded'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/pilot/handover-assets.json')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error('workers must be between 1 and 8')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    paths = [item['path'] for item in manifest['files']]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate manifest paths')
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(lambda spec: download_file(spec, args.root), manifest['files']):
            print(json.dumps(result), flush=True)
    print('Input sizes checked. Run verify_bank_reload to check content hashes, index and score.')


if __name__ == '__main__':
    main()
