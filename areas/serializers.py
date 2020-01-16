from areas.models import Area
from rest_framework import serializers

class AreaSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Area
        fields = ['id', 'name']