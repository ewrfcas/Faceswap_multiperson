from scripts import train_multi
from scripts import convert_multi
import argparse
import os
from plugins.PluginLoader_multi import PluginLoader
from scripts import train

parser = argparse.ArgumentParser(description='parameters for hacker')
# extract & convert
parser.add_argument('-i','--input-dir', default='data/A/')
parser.add_argument('-il','--input-list', default=None)
parser.add_argument('-a','--input-aligned-dir', default=None)
parser.add_argument('-pn','--person-num', default=5)
parser.add_argument('-ct','--convert-target', default=3)
parser.add_argument('-o','--output-dir', default='data/B/')
parser.add_argument('-A','--alignments-path', default=None)
parser.add_argument('-se','--serializer', default='json')
parser.add_argument('-D','--detector', default='mtcnn')
parser.add_argument('-mtms','--mtcnn-minsize', default=20)
parser.add_argument('-mtth','--mtcnn-threshold', default=["0.6", "0.7", "0.7"])
parser.add_argument('-mtsc','--mtcnn-scalefactor', default=0.709)
parser.add_argument('-l','--ref_threshold', default=0.6)
parser.add_argument('-n','--nfilter', default=None)
parser.add_argument('-f','--filter', default=None)
parser.add_argument('-c','--converter', choices=("Masked", "Adjust"), default='Masked')
parser.add_argument('-M','--mask-type', choices=["rect", "facehull", "facehullandrect"], default='facehullandrect')
parser.add_argument('-an','--anti', default=True)
parser.add_argument('-anp','--anti-path', default='anti_model/anti_model.h5')

# train
parser.add_argument('-in','--inputs', default=['data/trainset/A/','data/trainset/B/','data/trainset/C/','data/trainset/D/','data/trainset/E/'])
parser.add_argument('-si','--save-interval', default=30)
parser.add_argument('-bs','--batch-size', default=32)
parser.add_argument('-it','--iterations', default=1000)
parser.add_argument('-p','--preview', default=False)
parser.add_argument('-w','--write-image', default=True)
parser.add_argument('-pl','--perceptual-loss', default=False)
parser.add_argument('-ag','--allow-growth', default=True)
parser.add_argument('-gui','--redirect_gui', default=False)

# common
parser.add_argument('-m','--model-dir', default='models_128')
parser.add_argument('-t','--trainer', choices=PluginLoader.get_default_model(), default='OriginalHighRes')
parser.add_argument('-b','--blur-size', default=2)
parser.add_argument('-e','--erosion-kernel-size', default=None)
parser.add_argument('-ek','--erosion-kernel', default=None)
parser.add_argument('-sh','--sharpen_image', choices=["bsharpen", "gsharpen"], default=None)
parser.add_argument('-g','--gpus', default=1)
parser.add_argument('-fr','--frame-ranges', default=None)
parser.add_argument('-d','--discard-frames', default=False)
# parser.add_argument('-s','--swap-model', default=False, help='Swap the model. Instead of A -> B, B -> A')
parser.add_argument('-S','--seamless-clone', default=True)
parser.add_argument('-mh','--match-histogram', default=False)
parser.add_argument('-sm','--smooth-mask', default=True)
parser.add_argument('-aca','--avg-color-adjust', default=True)
parser.add_argument('-mp','--multiprocess', default=False)
parser.add_argument('-v','--verbose', default=True)

args = parser.parse_args()

convert_model = convert_multi.Convert(args)
# convert_model.process()

while True:
    files = os.listdir(args.input_dir)
    if 'alignments.json' in files:
        files.remove('alignments.json')
        files = list(map(lambda x:args.input_dir+x,files))
        files = list(filter(lambda x:os.access(x,os.R_OK),files))
        print('length:',len(files))
        if len(files) > 0:
            convert_model.args.input_list = files
            convert_model.process()
            [os.remove(f) for f in files]
            if os.path.exists(os.path.join(args.input_dir, 'alignments.json')):
                os.remove(os.path.join(args.input_dir,'alignments.json'))
    else:
        print('alignments.json is not found')

