from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from vehicles import views

urlpatterns = [
    path('', views.VehicleList.as_view(), name='vehicle-list'),
    path('<int:pk>/', views.VehicleDetail.as_view(), name='vehicle-detail'),
]

urlpatterns = format_suffix_patterns(urlpatterns)