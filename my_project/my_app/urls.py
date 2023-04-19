from django.urls import path
from . import views

urlpatterns = [
    path("predict_cover_type/", views.predict_cover_type, name="predict_cover_type"),
]