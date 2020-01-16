from django.db import models

class Message(models.Model):
    message = models.TextField(max_length=200, blank=True, default='')
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created']