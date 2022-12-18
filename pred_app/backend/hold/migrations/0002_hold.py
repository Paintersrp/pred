from django.db import migrations


def create_data(apps, schema_editor):
    HoldNums = apps.get_model("hold", "HoldNums")
    HoldNums(favNum="100", underNum="900").save()


class Migration(migrations.Migration):

    dependencies = [
        ("hold", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(create_data),
    ]
