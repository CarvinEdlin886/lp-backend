from rest_framework import serializers
from tnkb_colors.models import TnkbColor
from tnkb_colors.serializers import TnkbColorSerializer
from vehicles.models import Vehicle

class VehicleSerializer(serializers.HyperlinkedModelSerializer):
    tnkb_color_id = serializers.IntegerField()
    tnkbColorObject = TnkbColor.objects.all()
    tnkb_color = TnkbColorSerializer(tnkbColorObject, read_only=True)

    class Meta:
        model = Vehicle
        fields = ['id', 'owner_name', 'license_number', 'brand', 'type', 'variety', 'model', 'tnkb_color_id', 'tnkb_color']