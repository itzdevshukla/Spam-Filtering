"""URL routing configuration for detector app."""

from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("batch/", views.batch_view, name="batch"),
    path("batch/export/", views.export_batch_csv, name="export_batch_csv"),
    path("dashboard/", views.dashboard_view, name="dashboard"),
    path("history/", views.history_view, name="history"),
    path("history/clear/", views.clear_history, name="clear_history"),
    # REST API Endpoints
    path("api/predict/", views.api_predict, name="api_predict"),
    path("api/batch-predict/", views.api_batch_predict, name="api_batch_predict"),
]
