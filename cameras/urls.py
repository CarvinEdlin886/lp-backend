from cameras import views
from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [
    path('', views.CameraList.as_view(), name='camera-list'),
    path('<int:pk>/', views.CameraDetail.as_view(), name='camera-detail'),
]

urlpatterns = format_suffix_patterns(urlpatterns)