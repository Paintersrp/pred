from rest_framework import serializers
from .models import PayoutNums


class PayoutSerializer(serializers.ModelSerializer):
    class Meta:
        model = PayoutNums
        fields = ("wagerNum", "oddsNum")
