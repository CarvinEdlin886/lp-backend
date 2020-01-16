from areas.models import Area
from areas.serializers import AreaSerializer
from cameras.models import Camera
from rest_framework import serializers

class CameraSerializer(serializers.HyperlinkedModelSerializer):
    area_id = serializers.IntegerField()
    areaObject = Area.objects.all()
    area = AreaSerializer(areaObject, read_only=True)

    class Meta:
        model = Camera
        fields = ['id', 'name', 'area_id', 'area']