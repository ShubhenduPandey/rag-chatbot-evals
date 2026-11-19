"""Convenient local-only launch; never stores an API key or access code."""
import os
from pathlib import Path
import secrets
import subprocess
import sys


def main():
    env = dict(os.environ)
    if not env.get('DEMO_ACCESS_CODE'):
        env['DEMO_ACCESS_CODE'] = secrets.token_urlsafe(12)
        print('Temporary local demo access code: ' + env['DEMO_ACCESS_CODE'], flush=True)
        print('Copy it into the browser to unlock chat. It changes on the next launch.', flush=True)
    return subprocess.call([sys.executable, '-m', 'streamlit', 'run', 'app.py',
                            '--server.address', '127.0.0.1', '--server.headless', 'false'],
                           cwd=Path(__file__).resolve().parent, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
