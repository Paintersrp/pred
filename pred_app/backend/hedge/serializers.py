from rest_framework import serializers
from .models import HedgeNums


class HedgeSerializer(serializers.ModelSerializer):
    class Meta:
        model = HedgeNums
        fields = ("wagerNum", "oddsNum", "hedgeNum")
