#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

# Put the repo's shared/ dir (the one canonical global_methods.py) on sys.path
# so the Django views' ``from global_methods import *`` resolves there. Walks up
# from this file to find shared/.
_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d) and not os.path.isdir(os.path.join(_d, "shared")):
  _d = os.path.dirname(_d)
_shared = os.path.join(_d, "shared")
if os.path.isdir(_shared) and _shared not in sys.path:
  sys.path.insert(0, _shared)


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'frontend_server.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
