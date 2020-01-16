# Generated by Django 2.2.6 on 2019-11-03 15:50

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('statuses', '0001_initial'),
        ('areas', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Result',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('vehicle', models.CharField(blank=True, default='', max_length=100)),
                ('license_number', models.CharField(blank=True, default='', max_length=100)),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('area', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='results', to='areas.Area')),
                ('status', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='results', to='statuses.Status')),
            ],
            options={
                'ordering': ['created'],
            },
        ),
    ]
