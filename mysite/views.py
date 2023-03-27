from django.shortcuts import render
from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from mysite.models import User
import json
import os
from osgeo import osr, gdal
from mysite.utils.RasterMosaic import RasterMosaic
from mysite.utils.mask_crop import mask_crop
from mysite.utils.rio_hist_test import test_hist_cli


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
            if user.pwd != pwd:
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
        if len(User.objects.filter(account=account)) != 0:
            result["code"] = 500
            result["info"] = '账号重复，请更换账号'

        User.objects.create(account=account, pwd=pwd, email=email, type=1)
        return JsonResponse(result, safe=False, content_type='application/json')
    except Exception as e:
        print(e)
        result["code"] = 500
        result["info"] = 'failed'
        return JsonResponse(result, safe=False, content_type='application/json')


def rioHist(req):
    result = {
        "code": 200,
        "info": "success",
        "data": []
    }
    try:
        post = req.POST
        input_dataset_id = post.get("input_dataset_id")
        reference_dataset_id = post.get("reference_dataset_id")
        user_id = post.get("user_id")
        input_file = 'mysite/dataset/source1.tif'
        reference_file = 'mysite/dataset/reference1.tif'
        output_filename = os.path.basename(input_file).split('.')[0] + '_' + os.path.basename(reference_file)

        user_path = "mysite/processed/user" + user_id
        if not os.path.exists(user_path):
            os.makedirs(user_path)

        output_path = user_path + '/matched-' + output_filename
        test_hist_cli(input_file, reference_file, output_path)
        result['data'] = output_path
        return JsonResponse(result, safe=False, content_type='application/json')
    except Exception as e:
        print(e)
        result["code"] = 500
        result["info"] = 'failed'
        return JsonResponse(result, safe=False, content_type='application/json')


def maskCrop(req):
    result = {
        "code": 200,
        "info": "success",
        "data": []
    }
    try:
        post = req.POST
        input_dataset_id = post.get("input_dataset_id")
        reference_dataset_id = post.get("reference_dataset_id")
        user_id = post.get("user_id")
        input_file = 'mysite/dataset/8youyi-dark.tif'
        shpFile = 'mysite/dataset/yy2022_shp/yy2022.shp'
        output_filename = os.path.basename(input_file).split('.')[0] + '_' + os.path.basename(shpFile).split('.')[
            0] + '.tif '

        user_path = "mysite/processed/user" + user_id
        if not os.path.exists(user_path):
            os.makedirs(user_path)

        output_path = user_path + '/maskCrop-' + output_filename
        mask_crop(input_file, shpFile, output_path)
        result['data'] = output_path
        return JsonResponse(result, safe=False, content_type='application/json')
    except Exception as e:
        print(e)
        result["code"] = 500
        result["info"] = 'failed'
        return JsonResponse(result, safe=False, content_type='application/json')


def rasterMosaic(req):
    result = {
        "code": 200,
        "info": "success",
        "data": []
    }
    try:
        post = req.POST
        input_dataset_id = post.get("input_dataset_id")
        reference_dataset_id = post.get("reference_dataset_id")
        user_id = post.get("user_id")
        input_file = 'mysite/dataset/2.tif'
        reference_file = 'mysite/dataset/3.tif'
        output_filename = os.path.basename(input_file).split('.')[0] + '_' + os.path.basename(reference_file)

        user_path = "mysite/processed/user" + user_id
        if not os.path.exists(user_path):
            os.makedirs(user_path)

        output_path = user_path + '/Mosaic-' + output_filename
        RasterMosaic(input_file, reference_file, output_path)
        result['data'] = output_path
        return JsonResponse(result, safe=False, content_type='application/json')
    except Exception as e:
        print(e)
        result["code"] = 500
        result["info"] = 'failed'
        return JsonResponse(result, safe=False, content_type='application/json')


def projection(req):
    result = {
        "code": 200,
        "info": "success",
        "data": []
    }
    try:
        post = req.POST
        dataset_id = post.get("dataset_id")
        user_id = post.get("user_id")
        file = 'mysite/dataset/1.tif'

        user_path = "mysite/processed/user" + user_id
        if not os.path.exists(user_path):
            os.makedirs(user_path)

        if file.endswith("tif"):
            filename = os.path.basename(file)
            srs = osr.SpatialReference()
            srs.SetWellKnownGeogCS('WGS84')
            old_ds = gdal.Open(file)
            vrt_ds = gdal.AutoCreateWarpedVRT(old_ds, None, srs.ExportToWkt(), gdal.GRA_Bilinear)
            gdal.GetDriverByName('gtiff').CreateCopy(user_path + '/projection-' + filename, vrt_ds)
            result['data'] = "user" + user_id + '/projection-' + filename
            return JsonResponse(result, safe=False, content_type='application/json')
    except Exception as e:
        print(e)
        result["code"] = 500
        result["info"] = 'failed'
        return JsonResponse(result, safe=False, content_type='application/json')
