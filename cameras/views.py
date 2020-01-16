from cameras.models import Camera
from cameras.permissions import IsAdmin
from cameras.serializers import CameraSerializer
from rest_framework import generics, permissions

class CameraList(generics.ListCreateAPIView):
    queryset = Camera.objects.all()
    serializer_class = CameraSerializer
    permission_classes = [permissions.IsAuthenticated, IsAdmin]

class CameraDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Camera.objects.all()
    serializer_class = CameraSerializer
    permission_classes = [permissions.IsAuthenticated, IsAdmin]