"""Admin configuration for detector app."""
from django.contrib import admin
from .models import PredictionLog


@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ("predicted_label", "spam_probability", "risk_level", "short_text", "created_at")
    list_filter = ("predicted_label", "risk_level", "created_at")
    search_fields = ("text",)
    readonly_fields = ("created_at", "inference_time_ms")

    def short_text(self, obj):
        return obj.text[:60] + "..." if len(obj.text) > 60 else obj.text
    short_text.short_description = "Message"
