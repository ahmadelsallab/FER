"""
WSGI config for dep project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

import os, sys
sys.path.append('/home/ahmad/anaconda3/lib/python3.7')
#import os
#print(os.system('which python'))

import subprocess
result = subprocess.run(['which', 'python'], stdout=subprocess.PIPE)

print(sys.stderr.write("*******************"))
print(sys.stderr.write(str(result.stdout)))

from django.core.wsgi import get_wsgi_application


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dep.settings')

application = get_wsgi_application()
