from django.urls import path
from live_cameras import views
from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [
    path('', views.LiveCameraList.as_view(), name='live-camera-list'),
    # path('<int:pk>/', views.LiveCameraDetail.as_view(), name='live-camera-detail'),
]

urlpatterns = format_suffix_patterns(urlpatterns)