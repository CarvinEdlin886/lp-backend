from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from statuses import views

urlpatterns = [
    path('', views.StatusList.as_view(), name='status-list'),
    path('<int:pk>/', views.StatusDetail.as_view(), name='status-detail'),
]

urlpatterns = format_suffix_patterns(urlpatterns)