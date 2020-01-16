from django.urls import path
from messages import views
from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [
    path('', views.MessageList.as_view(), name='message-list'),
    path('<int:pk>/', views.MessageDetail.as_view(), name='message-detail'),
]

urlpatterns = format_suffix_patterns(urlpatterns)