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

    # 用户功能
    # 01登录
    path('login',views.login),
    # 02注册
    path('sign',views.sign),
    # 03获取所有影像数据
    path('getAllDataset', views.getAllDataset),
    # 04修改用户个人信息
    path('updateUserById', views.updateUserById),

    # 遥感处理功能
    # 01匀色
    path('rioHist', views.rioHist),
    # 02拼接
    path('rasterMosaic', views.rasterMosaic),
    # 03掩膜裁剪
    path('maskCrop', views.maskCrop),
    # 04重投影
    path('projection', views.projection),
    # 05展示unet语义分割结果
    path('showUnetImage', views.showUnetImage),

    #管理员功能
    #01获取所有用户
    path('getAllUsers', views.getAllUsers),
    #02删除用户
    path('deleteUserById', views.deleteUserById),

]
