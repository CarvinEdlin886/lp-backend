from django.urls import path, include

urlpatterns = [
    path('areas/', include('areas.urls')),
    path('auth/', include('auth.urls')),
    path('cameras/', include('cameras.urls')),
    path('live_cameras/', include('live_cameras.urls')),
    path('messages/', include('messages.urls')),
    path('results/', include('results.urls')),
    path('roles/', include('roles.urls')),
    path('statuses/', include('statuses.urls')),
    path('tnkb_colors/', include('tnkb_colors.urls')),
    path('users/', include('users.urls')),
    path('vehicles/', include('vehicles.urls')),
]