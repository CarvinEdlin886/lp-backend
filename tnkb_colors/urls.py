from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from tnkb_colors import views

urlpatterns = [
    path('', views.TnkbColorList.as_view(), name='tnkb-color-list'),
    path('<int:pk>/', views.TnkbColorDetail.as_view(), name='tnkb-color-detail'),
]

urlpatterns = format_suffix_patterns(urlpatterns)