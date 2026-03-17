from django.contrib import admin
from django.urls import path
from monitoring.views import dashboard_overview, drift_monitoring, prediction_ui, landing_page
from experiments.views import experiment_list, refresh_training_history

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', landing_page, name='home'),
    path('dashboard/', dashboard_overview, name='dashboard'),
    path('experiments/', experiment_list, name='experiments'),
    path('experiments/refresh/', refresh_training_history, name='refresh_runs'),
    path('monitoring/drift/', drift_monitoring, name='drift'),
    path('prediction/', prediction_ui, name='prediction'),
]
