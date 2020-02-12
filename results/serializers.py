from areas.models import Area
from areas.serializers import AreaSerializer
from results.models import Result
from rest_framework import serializers
from statuses.models import Status
from statuses.serializers import StatusSerializer

class ResultSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Result
        fields = ['id', 'vehicle', 'license_number', 'created']