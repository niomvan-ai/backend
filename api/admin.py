from django.contrib import admin
from .models import *

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ("username", "is_superuser")
    list_filter = ("is_superuser", "is_active")
    search_fields = ("username",)
    ordering = ("username",)
