from django.db import migrations


def create_data(apps, schema_editor):
    HedgeNums = apps.get_model("hedge", "HedgeNums")
    HedgeNums(wagerNum="100", oddsNum="900", hedgeNum="-200").save()


class Migration(migrations.Migration):

    dependencies = [
        ("hedge", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(create_data),
    ]
