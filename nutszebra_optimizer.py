import six
import chainer
from chainer import optimizers
import nutszebra_basic_print


class Optimizer(object):

    def __init__(self, model=None):
        self.model = model
        self.optimizer = None

    def __call__(self, i):
        pass

    def update(self):
        self.optimizer.update()


class OptimizerResnet(Optimizer):

    def __init__(self, model=None, schedule=(int(32000. / (50000. / 128)), int(48000. / (50000. / 128))), lr=0.1, momentum=0.9, weight_decay=1.0e-4, warm_up_lr=0.01):
        super(OptimizerResnet, self).__init__(model)
        optimizer = optimizers.MomentumSGD(warm_up_lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.schedule = schedule
        self.lr = lr
        self.warmup_lr = warm_up_lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        if i == 1:
            lr = self.lr
            print('finishded warming up')
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerDense(Optimizer):

    def __init__(self, model=None, schedule=(150, 225), lr=0.1, momentum=0.9, weight_decay=1.0e-4):
        super(OptimizerDense, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.schedule = schedule
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerWideRes(Optimizer):

    def __init__(self, model=None, schedule=(60, 120, 160), lr=0.1, momentum=0.9, weight_decay=5.0e-4):
        super(OptimizerWideRes, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.schedule = schedule
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr * 0.2
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerSwapout(Optimizer):

    def __init__(self, model=None, schedule=(196, 224), lr=0.1, momentum=0.9, weight_decay=1.0e-4):
        super(OptimizerSwapout, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.schedule = schedule
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerXception(Optimizer):

    def __init__(self, model=None, lr=0.045, momentum=0.9, weight_decay=1.0e-5, period=2):
        super(OptimizerXception, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.period = int(period)

    def __call__(self, i):
        if i % self.period == 0:
            lr = self.optimizer.lr * 0.94
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerVGG(Optimizer):

    def __init__(self, model=None, lr=0.01, momentum=0.9, weight_decay=5.0e-4):
        super(OptimizerVGG, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        # 150 epoch means (0.94 ** 75) * lr
        # if lr is 0.01, then (0.94 ** 75) * 0.01 is 0.0001 at the end
        if i % 2 == 0:
            lr = self.optimizer.lr * 0.94
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerGooglenet(Optimizer):

    def __init__(self, model=None, lr=0.0015, momentum=0.9, weight_decay=2.0e-4):
        super(OptimizerGooglenet, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        if i % 8 == 0:
            lr = self.optimizer.lr * 0.96
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerNetworkInNetwork(Optimizer):

    def __init__(self, model=None, lr=0.1, momentum=0.9, weight_decay=1.0e-4, schedule=(int(1.0e5 / (50000. / 128)), )):
        super(OptimizerNetworkInNetwork, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.schedule = schedule

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerGooglenetV2(Optimizer):

    def __init__(self, model=None, lr=0.045, momentum=0.9, weight_decay=4.0e-5):
        super(OptimizerGooglenetV2, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        if i % 2 == 0:
            lr = self.optimizer.lr * 0.94
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerGooglenetV3(Optimizer):

    def __init__(self, model=None, lr=0.045, decay=0.9, eps=1.0, weight_decay=4.0e-5, clip=2.0):
        super(OptimizerGooglenetV3, self).__init__(model)
        optimizer = optimizers.RMSprop(lr, decay, eps)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        clip = chainer.optimizer.GradientClipping(clip)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        optimizer.add_hook(clip)
        self.optimizer = optimizer

    def __call__(self, i):
        if i % 2 == 0:
            lr = self.optimizer.lr * 0.94
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerResNext(Optimizer):

    def __init__(self, model=None, lr=0.1, momentum=0.9, weight_decay=5.0e-4, schedule=(150, 225)):
        super(OptimizerResNext, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.schedule = schedule
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerFractalNet(Optimizer):

    def __init__(self, model=None, lr=0.02, momentum=0.9, schedule=(150, 225, 300, 375)):
        super(OptimizerFractalNet, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        optimizer.setup(self.model)
        self.optimizer = optimizer
        self.schedule = schedule
        self.lr = lr
        self.momentum = momentum

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerPyramidalResNet(Optimizer):

    def __init__(self, model=None, lr=0.5, momentum=0.9, schedule=(150, 225), weight_decay=1.0e-4):
        super(OptimizerPyramidalResNet, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.schedule = schedule
        self.lr = lr
        self.momentum = momentum

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerStochasticDepth(Optimizer):

    def __init__(self, model=None, lr=0.1, momentum=0.9, schedule=(250, 375), weight_decay=1.0e-4):
        super(OptimizerStochasticDepth, self).__init__(model)
        self.lr = lr
        self.momentum = momentum
        self.schedule = schedule
        self.weight_decay = weight_decay
        all_links = OptimizerStochasticDepth._find(model)
        optimizer_set = []
        for link in all_links:
            optimizer = optimizers.MomentumSGD(lr, momentum)
            weight_decay = chainer.optimizer.WeightDecay(self.weight_decay)
            optimizer.setup(link[0])
            optimizer.add_hook(weight_decay)
            optimizer_set.append(optimizer)
        self.optimizer_set = optimizer_set
        self.all_links = all_links

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            for optimizer in self.optimizer_set:
                optimizer.lr = lr

    def update(self):
        for i in six.moves.range(len(self.all_links)):
            if self.all_links[i][1].grad is not None:
                self.optimizer_set[i].update()

    @staticmethod
    def _grad(ele):
        if hasattr(ele, 'W') and hasattr(ele.W, 'grad'):
            return (ele, ele.W)
        if hasattr(ele, 'beta') and hasattr(ele.beta, 'grad'):
            return (ele, ele.beta)
        return None

    @staticmethod
    def _children(ele):
        return hasattr(ele, '_children')

    @staticmethod
    def _find(model):
        links = []

        def dfs(ele):

            grad = OptimizerStochasticDepth._grad(ele)
            if grad is not None:
                links.append(grad)
            else:
                if OptimizerStochasticDepth._children(ele):
                    for link in ele._children:
                        dfs(ele[link])
        dfs(model)
        return links


class OptimizerResnetOfResnet(Optimizer):

    def __init__(self, model=None, lr=0.1, momentum=0.9, schedule=(250, 375), weight_decay=1.0e-4):
        super(OptimizerResnetOfResnet, self).__init__(model)
        self.lr = lr
        self.momentum = momentum
        self.schedule = schedule
        self.weight_decay = weight_decay
        all_links = OptimizerStochasticDepth._find(model)
        optimizer_set = []
        for link in all_links:
            optimizer = optimizers.MomentumSGD(lr, momentum)
            weight_decay = chainer.optimizer.WeightDecay(self.weight_decay)
            optimizer.setup(link[0])
            optimizer.add_hook(weight_decay)
            optimizer_set.append(optimizer)
        self.optimizer_set = optimizer_set
        self.all_links = all_links

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            for optimizer in self.optimizer_set:
                optimizer.lr = lr

    def update(self):
        for i in six.moves.range(len(self.all_links)):
            if self.all_links[i][1].grad is not None:
                self.optimizer_set[i].update()

    @staticmethod
    def _grad(ele):
        if hasattr(ele, 'W') and hasattr(ele.W, 'grad'):
            return (ele, ele.W)
        if hasattr(ele, 'beta') and hasattr(ele.beta, 'grad'):
            return (ele, ele.beta)
        return None

    @staticmethod
    def _children(ele):
        return hasattr(ele, '_children')

    @staticmethod
    def _find(model):
        links = []

        def dfs(ele):

            grad = OptimizerStochasticDepth._grad(ele)
            if grad is not None:
                links.append(grad)
            else:
                if OptimizerStochasticDepth._children(ele):
                    for link in ele._children:
                        dfs(ele[link])
        dfs(model)
        return links


class OptimizerPReLUNet(Optimizer):

    def __init__(self, model=None, lr=0.01, momentum=0.9, schedule=(150, 225), weight_decay=5.0e-4):
        super(OptimizerPReLUNet, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.schedule = schedule
        self.lr = lr
        self.momentum = momentum

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerResnetInResnet(Optimizer):

    def __init__(self, model=None, schedule=(42, 62), lr=0.1, momentum=0.9, weight_decay=1.0e-4):
        super(OptimizerResnetInResnet, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.schedule = schedule
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerAppendixA(Optimizer):

    def __init__(self, model=None, schedule=(150, 175), lr=0.1, momentum=0.9, weight_decay=1.0e-4):
        super(OptimizerAppendixA, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.schedule = schedule
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr
