from django.db import models


# Create your models here.
class User(models.Model):
    account = models.CharField(max_length=64, unique=True)
    pwd = models.CharField(max_length=255)
    email = models.CharField(max_length=255)
    type = models.SmallIntegerField()  # 用户类型,0为管理员,1为普通用户

    class Meta:
        db_table = "user"
