from django.urls import path
from . import views

urlpatterns = [
    path('symptoms/', views.SymptomView.as_view(), name='create_user'),
    path('osteoarthritis/', views.OsteoarthritisView.as_view(), name='upload_images'),
    path('summary/', views.SummaryView.as_view(), name='upload_images'),
]