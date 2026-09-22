"""Database models for the SpamShield detector application."""
from django.db import models


class PredictionLog(models.Model):
    """Audit log tracking SMS/text classification requests."""

    LABEL_CHOICES = [
        ("HAM", "Ham (Legitimate)"),
        ("SPAM", "Spam (Unsolicited)"),
    ]

    RISK_CHOICES = [
        ("SAFE", "Safe"),
        ("SUSPICIOUS", "Suspicious"),
        ("CRITICAL", "Critical Spam"),
    ]

    text = models.TextField(help_text="Classified message text")
    predicted_label = models.CharField(max_length=10, choices=LABEL_CHOICES)
    spam_probability = models.FloatField(help_text="Probability of spam [0.0 - 1.0]")
    risk_level = models.CharField(max_length=20, choices=RISK_CHOICES)
    top_contributing_tokens = models.JSONField(default=list, blank=True)
    inference_time_ms = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Prediction Log"
        verbose_name_plural = "Prediction Logs"

    def __str__(self):
        snippet = self.text[:40] + "..." if len(self.text) > 40 else self.text
        return f"[{self.predicted_label} ({self.spam_probability:.1%})] {snippet}"
