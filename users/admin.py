from .forms import CustomUserChangeForm, CustomUserCreationForm
from .models import CustomUser
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

class CustomUserAdmin(UserAdmin):
    add_form = CustomUserCreationForm
    form = CustomUserChangeForm
    model = CustomUser
    list_display = ['email', 'password', 'role',]

    ordering = None

admin.site.register(CustomUser, CustomUserAdmin)