import os

# Gunicorn configuration
workers = 4
threads = 2
timeout = 60
keepalive = 5
bind = "0.0.0.0:" + os.environ.get("PORT", "10000")
