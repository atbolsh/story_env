"""
WSGI config for frontend_server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/
"""

import os
import sys

from django.core.wsgi import get_wsgi_application

# Put the repo's shared/ dir (the one canonical global_methods.py) on sys.path
# so the Django views' ``from global_methods import *`` resolves there. Walks up
# from this file to find shared/.
_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d) and not os.path.isdir(os.path.join(_d, "shared")):
  _d = os.path.dirname(_d)
_shared = os.path.join(_d, "shared")
if os.path.isdir(_shared) and _shared not in sys.path:
  sys.path.insert(0, _shared)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'frontend_server.settings')

application = get_wsgi_application()
