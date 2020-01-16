from django.contrib.auth.hashers import make_password
from django.http import Http404
from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from users.models import CustomUser
from users.permissions import IsAdmin, IsSuperuser
from users.serializers import UserSerializer

adminRoleId = 1
superuserRoleId = 2
userRoleId = 3

def validate_password(value: str) -> str:
    return make_password(value)

class AdminList(APIView):
    def get(self, request, format=None):
        users = CustomUser.objects.filter(role_id=adminRoleId)
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        request.data['password'] = validate_password(request.data['password'])
        request.data['role'] = adminRoleId
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class AdminDetail(APIView):
    def get_object(self, pk):
        try:
            return CustomUser.objects.get(pk=pk, role_id=adminRoleId)
        except CustomUser.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        user = self.get_object(pk)
        serializer = UserSerializer(user)
        return Response(serializer.data)

    def put(self, request, pk, format=None):
        user = self.get_object(pk)
        request.data['password'] = validate_password(request.data['password'])
        request.data['role'] = adminRoleId
        serializer = UserSerializer(user, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        user = self.get_object(pk)
        user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

class SuperuserList(APIView):
    permission_classes = [permissions.IsAuthenticated, IsAdmin]

    def get(self, request, format=None):
        users = CustomUser.objects.filter(role_id=superuserRoleId)
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        request.data['password'] = validate_password(request.data['password'])
        request.data['role'] = superuserRoleId
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SuperuserDetail(APIView):
    permission_classes = [permissions.IsAuthenticated, IsAdmin]

    def get_object(self, pk):
        try:
            return CustomUser.objects.get(pk=pk, role_id=superuserRoleId)
        except CustomUser.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        user = self.get_object(pk)
        serializer = UserSerializer(user)
        return Response(serializer.data)

    def put(self, request, pk, format=None):
        user = self.get_object(pk)
        request.data['password'] = validate_password(request.data['password'])
        request.data['role'] = superuserRoleId
        serializer = UserSerializer(user, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        user = self.get_object(pk)
        user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

class UserList(APIView):
    permission_classes = [permissions.IsAuthenticated, IsSuperuser]

    def get(self, request, format=None):
        users = CustomUser.objects.filter(role_id=userRoleId)
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        request.data['password'] = validate_password(request.data['password'])
        request.data['role'] = userRoleId
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class UserDetail(APIView):
    permission_classes = [permissions.IsAuthenticated, IsSuperuser]

    def get_object(self, pk):
        try:
            return CustomUser.objects.get(pk=pk, role_id=userRoleId)
        except CustomUser.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        user = self.get_object(pk)
        serializer = UserSerializer(user)
        return Response(serializer.data)

    def put(self, request, pk, format=None):
        user = self.get_object(pk)
        request.data['password'] = validate_password(request.data['password'])
        request.data['role'] = userRoleId
        serializer = UserSerializer(user, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        user = self.get_object(pk)
        user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)