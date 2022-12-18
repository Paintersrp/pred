from rest_framework import serializers
from .models import HoldNums


class HoldSerializer(serializers.ModelSerializer):
    class Meta:
        model = HoldNums
        fields = ("favNum", "underNum")
