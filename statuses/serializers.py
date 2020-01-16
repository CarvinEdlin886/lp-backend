from rest_framework import serializers
from statuses.models import Status

class StatusSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Status
        fields = ['id', 'name']