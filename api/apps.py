from django.apps import AppConfig
import threading
import time
import cloudinary
import cloudinary.api
import cloudinary.uploader

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

def delete_old_files_periodically():
    while True:
        now = time.time()
        thirty_minutes_ago = now - 1800  # 30 minutes ago

        try:
            # List all resources in Cloudinary
            resources = cloudinary.api.resources(type="upload", max_results=500)

            for resource in resources.get('resources', []):
                created_at = resource['created_at']
                created_at_timestamp = time.mktime(time.strptime(created_at, '%Y-%m-%dT%H:%M:%SZ'))

                if now - created_at_timestamp > 1800:  # 30 minutes
                    public_id = resource['public_id']
                    cloudinary.uploader.destroy(public_id)
                    print(f"Deleted {public_id} from Cloudinary")

        except Exception as e:
            print(f"Error deleting old files from Cloudinary: {str(e)}")

        time.sleep(60)  # Run every minute

# Start the thread when Django starts
threading.Thread(target=delete_old_files_periodically, daemon=True).start()