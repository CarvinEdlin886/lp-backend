from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from results import views

urlpatterns = [
    path('', views.ResultList.as_view(), name='result-list'),
    path('<int:pk>/', views.ResultDetail.as_view(), name='result-detail'),
]

urlpatterns = format_suffix_patterns(urlpatterns)