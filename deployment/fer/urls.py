from django.urls import path

from . import views

urlpatterns = [
    path('', views.upload_model, name='upload_model'),
    path('upload_model', views.upload_model, name='upload_model'),
	path('upload_video', views.upload_video, name='upload_video'),
    path('predict', views.predict, name='predict')
]