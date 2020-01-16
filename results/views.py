from rest_framework import generics, permissions
from results.models import Result
from results.permissions import IsAdminOrSuperUser
from results.serializers import ResultSerializer

class ResultList(generics.ListCreateAPIView):
    queryset = Result.objects.all()
    serializer_class = ResultSerializer
    permission_classes = [permissions.IsAuthenticated, IsAdminOrSuperUser]

class ResultDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Result.objects.all()
    serializer_class = ResultSerializer
    permission_classes = [permissions.IsAuthenticated, IsAdminOrSuperUser]