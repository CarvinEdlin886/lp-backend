from areas.models import Area
from areas.permissions import IsAdmin
from areas.serializers import AreaSerializer
from rest_framework import generics, permissions

class AreaList(generics.ListCreateAPIView):
    queryset = Area.objects.all()
    serializer_class = AreaSerializer
    permission_classes = [permissions.IsAuthenticated, IsAdmin]

class AreaDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Area.objects.all()
    serializer_class = AreaSerializer
    permission_classes = [permissions.IsAuthenticated, IsAdmin]