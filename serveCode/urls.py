"""serveCode URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from mysite import views
urlpatterns = [
    path('admin/', admin.site.urls),

    # 01登录
    path('login',views.login),
    # 02注册
    path('sign',views.sign),

    # 03匀色
    path('rioHist', views.rioHist),
    # 04拼接
    path('rasterMosaic', views.rasterMosaic),
    # 05掩膜裁剪
    path('maskCrop', views.maskCrop),
    # 06重投影
    path('projection', views.projection),



]
