from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from roles import views

urlpatterns = [
    path('', views.RoleList.as_view(), name='role-list'),
    path('<int:pk>/', views.RoleDetail.as_view(), name='role-detail'),
]

urlpatterns = format_suffix_patterns(urlpatterns)