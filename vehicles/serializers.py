from rest_framework import serializers
from tnkb_colors.models import TnkbColor
from tnkb_colors.serializers import TnkbColorSerializer
from vehicles.models import Vehicle

class VehicleSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Vehicle
        fields = ['id', 'owner_name', 'license_number', 'phone_number', 'created']