from areas import views
from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [
    path('', views.AreaList.as_view(), name='area-list'),
    path('<int:pk>/', views.AreaDetail.as_view(), name='area-detail'),
]

urlpatterns = format_suffix_patterns(urlpatterns)