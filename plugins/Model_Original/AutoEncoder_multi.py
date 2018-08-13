# AutoEncoder base classes

from lib.utils import backup_file

hdf = {'encoderH5': 'encoder.h5'}

class AutoEncoder_multi:
    def __init__(self, nums, model_dir, gpus):
        # nums: num of people
        self.model_dir = model_dir
        self.gpus = gpus

        self.encoder = self.Encoder()
        self.decoder = []
        for i in range(nums):
            self.decoder.append(self.Decoder())

        self.initModel()

    def load(self):
        try:
            self.encoder.load_weights(str(self.model_dir / hdf['encoderH5']))
            for i in range(len(self.decoder)):
                self.decoder[i].load_weights(str(self.model_dir)+'/'+'decoder_'+str(i)+'.h5')
            print('loaded model weights')
            return True
        except Exception as e:
            print('Failed loading existing training data.')
            print(e)
            return False

    def save_weights(self, diff = None):
        model_dir = str(self.model_dir)
        if diff is not None:
            self.encoder.save_weights(str(self.model_dir) +'/'+ str(diff) + '_' + hdf['encoderH5'])
            for i in range(len(self.decoder)):
                self.decoder[i].save_weights(str(self.model_dir) + '/' + str(diff) + '_' + 'decoder_' + str(i) + '.h5')
            print('saved model weights')
        else:
            for model in hdf.values():
                backup_file(model_dir, model)
            self.encoder.save_weights(str(self.model_dir / hdf['encoderH5']))
            for i in range(len(self.decoder)):
                self.decoder[i].save_weights(str(self.model_dir)+'/'+'decoder_'+str(i)+'.h5')
            print('saved model weights')
