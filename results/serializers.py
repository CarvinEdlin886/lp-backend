from areas.models import Area
from areas.serializers import AreaSerializer
from results.models import Result
from rest_framework import serializers
from statuses.models import Status
from statuses.serializers import StatusSerializer

class ResultSerializer(serializers.HyperlinkedModelSerializer):
    area_id = serializers.IntegerField()
    areaObject = Area.objects.all()
    area = AreaSerializer(areaObject, read_only=True)

    status_id = serializers.IntegerField()
    statusObject = Status.objects.all()
    status = StatusSerializer(statusObject, read_only=True)

    class Meta:
        model = Result
        fields = ['id', 'vehicle', 'license_number', 'area_id', 'area', 'status_id', 'status', 'created']