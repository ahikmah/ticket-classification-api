from django.urls import path
from .views import get_ticket_class

urlpatterns = [
    path('ticket/', get_ticket_class.as_view(), name = 'get_ticket_class'),
]