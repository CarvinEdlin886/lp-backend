from live_cameras.models import LiveCamera
from rest_framework import serializers

class LiveCameraSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = LiveCamera
        fields = ['id', 'base64']