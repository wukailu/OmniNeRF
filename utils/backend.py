import os
local_backend = os.getenv('LOCAL_BACKEND')
if local_backend:
    print("using local_backend")
    from .local_backend import *
else:
    if 'use_kubernets.backend' in os.listdir('.'):
        print("using kubernetes_backend")
        from .kubernetes_backend import *
    else:
        print("using atlas_backend")
        from .atlas_backend import *
log('using ', name)
