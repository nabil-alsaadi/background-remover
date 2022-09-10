from flask import Flask
from flask_restful import Resource, Api, reqparse
import werkzeug
import torch
from carvekit.api.high import HiInterface

# import pandas as pd
# import ast
app = Flask(__name__)
api = Api(app)
data = []

class Users(Resource):
    def get(self):

        return {'data': "test"}, 200  # return data and 200 OK

    def post(self):
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('name', required=True)  # add args
        args = parser.parse_args()
        temp = {'Data': args}
        data.append(temp)
        return temp
class Upload(Resource):
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        image_file = args['file']
        image_file.save("uploads/before.jpg")
        interface = HiInterface(batch_size_seg=5, batch_size_matting=1,
                                device='cuda' if torch.cuda.is_available() else 'cpu',
                                seg_mask_size=320, matting_mask_size=2048)
        images_without_background = interface(["uploads/before.jpg"])
        after = images_without_background[0]
        after.save("uploads/after.png")
        return {'success': True}, 200
api.add_resource(Users, '/users')  # '/users' is our entry point
api.add_resource(Upload, '/upload')  # '/users' is our entry point
if __name__ == '__main__':
    app.run(debug=True)  # run our Flask app