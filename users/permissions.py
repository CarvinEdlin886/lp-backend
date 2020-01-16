from rest_framework import permissions

class IsAdmin(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user.role_id == 1

class IsSuperuser(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user.role_id == 2