from rest_framework import serializers
from tnkb_colors.models import TnkbColor

class TnkbColorSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = TnkbColor
        fields = ['id', 'name', 'description']