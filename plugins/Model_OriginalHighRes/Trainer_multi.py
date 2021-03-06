import time
import numpy

from lib.training_data import TrainingDataGenerator, stack_images


TRANSFORM_PRC = 115.


class Trainer_multi():
    
    _random_transform_args = {
        'rotation_range': 10 * (TRANSFORM_PRC * .01),
        'zoom_range': 0.05 * (TRANSFORM_PRC * .01),
        'shift_range': 0.05 * (TRANSFORM_PRC * .01),
        'random_flip': 0.4 * (TRANSFORM_PRC * .01),
    }
    
    def __init__(self, model, fns, batch_size, *args):
        self.batch_size = batch_size
        self.model = model
        from timeit import default_timer as clock
        self._clock = clock
        
        generator = TrainingDataGenerator(self.random_transform_args, 160, 5, zoom=2)
        self.images = [generator.minibatchAB(f, self.batch_size) for f in fns]
                
        self.generator = generator        
        

    def train_one_step(self, iter, viewer):
        warpeds = []
        targets = []
        for i in self.images:
            _, warped, target = next(i)
            warpeds.append(warped)
            targets.append(target)

        losses = [0] * len(warpeds)
        loss_str = ""
        for i in range(len(warpeds)):
            losses[i] = self.model.autoencoder[i].train_on_batch(warpeds[i], targets[i])
            loss_str += ("loss_{:d}: {:.5f} ".format(i, losses[i]))
        time_str = "[{0}] [#{1:05d}] ".format(time.strftime("%H:%M:%S"), iter)
        print(time_str + loss_str)

        if viewer is not None:
            sub_targets = [t[0:4] for t in targets]
            viewer(self.show_sample(sub_targets), "training")
            

    def show_sample(self, tests):
        figures = []
        for i in range(len(tests)):
            figure_temp = [tests[i]]
            for j in range(len(tests)):
                figure_temp.append(self.model.autoencoder[j].predict(tests[i]))
            figures.append(numpy.stack(figure_temp, axis=1))

        if tests[0].shape[0] % 2 == 1:
            for i in range(len(figures)):
                figures[i] = numpy.concatenate([figures[i], numpy.expand_dims(figures[i][0], 0)])

        figure = numpy.concatenate(figures, axis=0)
        w = len(tests)
        h = int( figure.shape[0] / w)
        figure = figure.reshape((w, h) + figure.shape[1:])
        figure = stack_images(figure)

        return numpy.clip(figure * 255, 0, 255).astype('uint8')
    
    
    @property
    def random_transform_args(self):
        return self._random_transform_args
