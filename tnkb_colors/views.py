from rest_framework import generics
from tnkb_colors.models import TnkbColor
from tnkb_colors.serializers import TnkbColorSerializer

class TnkbColorList(generics.ListCreateAPIView):
    queryset = TnkbColor.objects.all()
    serializer_class = TnkbColorSerializer

class TnkbColorDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = TnkbColor.objects.all()
    serializer_class = TnkbColorSerializer