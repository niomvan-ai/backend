from django.apps import AppConfig
import threading
import os
import time
from django.conf import settings


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

def delete_old_files_periodically():
    while True:
        media_folder = os.path.join(settings.BASE_DIR, 'media')
        now = time.time()

        for filename in os.listdir(media_folder):
            file_path = os.path.join(media_folder, filename)
            if os.path.isfile(file_path):
                if now - os.path.getmtime(file_path) > 1800:  # 30 minutes
                    os.remove(file_path)

        time.sleep(60)  # Run every minute

# Start the thread when Django starts
threading.Thread(target=delete_old_files_periodically, daemon=True).start()