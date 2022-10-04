import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *

from PIL import Image
import torchvision.transforms as transforms
import glob
from bottle import route, run, template, request, static_file, url, get, post, response, error, abort, redirect, os

# get options
opt = BaseOptions().parse()

class Evaluator:
    def __init__(self, opt, projection_mode='orthogonal'):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        # create net
        netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

        if opt.load_netC_checkpoint_path is not None:
            print('loading for net C ...', opt.load_netC_checkpoint_path)
            netC = ResBlkPIFuNet(opt).to(device=cuda)
            netC.load_state_dict(torch.load(opt.load_netC_checkpoint_path, map_location=cuda))
        else:
            netC = None

        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

        opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
        with open(opt_log, 'w') as outfile:
            outfile.write(json.dumps(vars(opt), indent=2))

        self.cuda = cuda
        self.netG = netG
        self.netC = netC

    def load_image(self, image_path):
        # Name
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float()
        # image & mask
        rawimage = Image.open(image_path)
        image = rawimage.convert('RGB')
        r, g, b, mask = rawimage.split()
        mask = mask.convert('L')
        mask = transforms.Resize(self.load_size)(mask)
        mask = transforms.ToTensor()(mask).float()
        
        image = self.to_tensor(image)
        image = mask.expand_as(image) * image

        return {
            'name': img_name,
            'img': image.unsqueeze(0),
            'calib': calib.unsqueeze(0),
            'mask': mask.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX,
        }

    def eval(self, data, use_octree=False):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'] and ['b_max'] tensors.
        :return:
        '''
        opt = self.opt
        with torch.no_grad():
            self.netG.eval()
            if self.netC:
                self.netC.eval()
            save_path = '%s/%s/result_%s.obj' % (opt.results_path, opt.name, data['name'])
            if self.netC:
                gen_mesh_color(opt, self.netG, self.netC, self.cuda, data, save_path, use_octree=use_octree)
            else:
                gen_mesh(opt, self.netG, self.cuda, data, save_path, use_octree=use_octree)

@get('/upload')
def upload():
    return '''
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="submit" value="Upload"></br>
            <input type="file" name="upload"></br>
        </form>
    '''

@route('/upload', method='POST')
def do_upload():
    global evaluator

    upload = request.files.get('upload', '')
    if not upload.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return 'File extension not allowed!'

    filename = upload.filename.lower()
    root, ext = os.path.splitext(filename)
    save_path = os.path.join('C:/Users/poly/Downloads', filename)

    print("trying to save")
    upload.save(save_path, overwrite=True)
    print("saved input file")

    try:
        print(save_path)
        data = evaluator.load_image(save_path)
        evaluator.eval(data, True)
    except Exception as e:
       print("error:", e.args)
    
    return "success"

def main():
    global evaluator
    evaluator = Evaluator(opt)
    print("run")
    run(host="0.0.0.0", port=8000, debug=True)

if __name__ == '__main__':
    main()
