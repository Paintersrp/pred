from django.db import models


class HoldNums(models.Model):
    favNum = models.FloatField()
    underNum = models.FloatField()

    def __str__(self):
        return self.favNum, self.underNum
