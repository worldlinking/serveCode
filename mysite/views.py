from django.shortcuts import render
from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from mysite.models import User
import json


# Create your views here.
def login(req):
    result = {
        "code": 200,
        "info": "success",
        "data": []
    }
    try:
        json_str = req.body
        req_data = json.loads(json_str)
        account = req_data['account']
        pwd = req_data['pwd']

        # 从数据库中获取用户信息
        user = User.objects.filter(account=account).first()
        if user:
            if user.pwd!=pwd:
                result["code"] = 500
                result["info"] = '密码错误'
                return JsonResponse(result, safe=False, content_type='application/json')
            else:
                return JsonResponse(result, safe=False, content_type='application/json')
        else:
            result["code"] = 500
            result["info"] = '用户不存在'
            return JsonResponse(result, safe=False, content_type='application/json')
    except Exception as e:
        print(e)
        result["code"] = 500
        result["info"] = 'failed'
        return JsonResponse(result, safe=False, content_type='application/json')

def sign(req):
    result = {
        "code": 200,
        "info": "success",
        "data": []
    }
    try:
        json_str = req.body
        req_data = json.loads(json_str)
        account = req_data['account']
        pwd = req_data['pwd']
        email = req_data['email']

        # 判断是否账户重复
        if len(User.objects.filter(account=account))!=0:
            result["code"]=500
            result["info"]='账号重复，请更换账号'

        User.objects.create(account=account, pwd=pwd,email=email, type=1)
        return JsonResponse(result, safe=False, content_type='application/json')
    except Exception as e:
        print(e)
        result["code"] = 500
        result["info"] = 'failed'
        return JsonResponse(result, safe=False, content_type='application/json')
