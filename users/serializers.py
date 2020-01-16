from users.models import CustomUser
from rest_framework import serializers

class UserSerializer(serializers.HyperlinkedModelSerializer):
    role = serializers.IntegerField(source='role_id')

    class Meta:
        model = CustomUser
        fields = ['id', 'email', 'password', 'role']