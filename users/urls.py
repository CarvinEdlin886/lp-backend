from users import views
from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [
    path('admins/', views.AdminList.as_view(), name='admin-list'),
    path('admins/<int:pk>/', views.AdminDetail.as_view(), name='admin-detail'),

    path('superusers/', views.SuperuserList.as_view(), name='superuser-list'),
    path('superusers/<int:pk>/', views.SuperuserDetail.as_view(), name='superuser-detail'),

    path('users/', views.UserList.as_view(), name='user-list'),
    path('users/<int:pk>/', views.UserDetail.as_view(), name='user-detail'),
]

urlpatterns = format_suffix_patterns(urlpatterns)