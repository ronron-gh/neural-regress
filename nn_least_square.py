# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions
import matplotlib.pyplot as plt

class NN1Layer(chainer.Chain):
    def __init__(self):
        w = I.Normal(scale=1.0)
        super(NN1Layer, self).__init__(
                l1 = L.Linear(3, 1, initialW=w),
        )
        
    def __call__(self, x):
        y = self.l1(x)
        return y
   
    
def main():
    
    epoch = 1500
    batchsize = 10
    
    #データの作成
    x = np.arange(-5,5,0.1)
    n = x.size
    np.random.seed(seed=32)
    err = np.random.randn(n)
    y = 0.3*x*x*x + 0.5*x*x - 4*x + 5 + err

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y, marker="o", linewidth=0, label="train data")
    plt.legend()


    #学習データとテストデータを作成
    trainx = np.empty((n,3), dtype=np.float32)
    trainy = np.empty((n,1), dtype=np.float32)
    for i in range(n):
        trainx[i][2] = x[i] * x[i] * x[i]
        trainx[i][1] = x[i] * x[i]
        trainx[i][0] = x[i]
        trainy[i][0] = y[i]

    train = chainer.datasets.TupleDataset(trainx, trainy)
    test = chainer.datasets.TupleDataset(trainx, trainy)
       

    #ニューラルネットワークの登録
    model = L.Classifier(NN1Layer(), lossfun=F.mean_squared_error)  #平均2乗誤差
    model.compute_accuracy = False
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    
    #イテレータの登録
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)
    
    #トレーナーの登録
    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = training.Trainer(updater, (epoch, 'epoch'))
    
    #学習状況の表示や保存
    trainer.extend(extensions.Evaluator(test_iter, model))
    #誤差のグラフ
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                         'epoch', file_name='loss.png'))
    #ログ
    trainer.extend(extensions.LogReport())
    #計算状態の表示
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))
    #エポック毎にトレーナーの状態を保存する
    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))    
    
    #スナップショットから学習を再開する
#    chainer.serializers.load_npz("result/snapshot_iter_12000", trainer)
    
    #学習開始
    trainer.run()
    
    #途中状態の保存
    chainer.serializers.save_npz("result/nn.model", model)
    
    #学習結果の評価
    y_pred = np.empty((n), dtype=np.float32)

    for i in range(n):
        evalx = chainer.Variable(trainx[i].reshape(1,3))
        result = model.predictor(evalx)
        y_pred[i] = result.data
            
    plt.plot(x, y_pred, label="regress")
    plt.legend()
    
    plt.grid(b=True, which='major', axis='both')
    plt.show()

    
if __name__ == '__main__':
    main()
    
