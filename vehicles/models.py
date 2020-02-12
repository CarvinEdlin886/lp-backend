from django.db import models

class Vehicle(models.Model):
    owner_name = models.CharField(max_length=100, blank=True, default='')
    license_number = models.CharField(max_length=100, blank=True, default='')
    phone_number = models.CharField(max_length=20, blank=True, default='')
    # brand = models.CharField(max_length=100, blank=True, default='')
    # type = models.CharField(max_length=100, blank=True, default='')
    # variety = models.CharField(max_length=100, blank=True, default='')
    # model = models.CharField(max_length=100, blank=True, default='')
    # tnkb_color = models.ForeignKey('tnkb_colors.TnkbColor', related_name='vehicles', on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created']