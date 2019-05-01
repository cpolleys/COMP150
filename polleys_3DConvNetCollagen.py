import os
from skimage import img_as_ubyte
import skimage.io as swag
import matplotlib.pyplot as plt
import os.path as osp
import skimage.color as skc
import numpy as np
import tensorflow as tf
from math import ceil,sqrt

class Config(object):

    """
            :param input_size: size of our input which will be 512 x 512 x 4 to start
            :param output_size: 1? - # of classes
            :param filter_size: size H x W x # of filter which will be 5 x 5 x 100
            :param pooling_schedule: ?
            :param weight_scale: initialization scale of weights
            :param centering_data: centering?
            :param use_dropout: dropout?

    """
    tf.reset_default_graph()
    input_size = [512,512,4]
    output_size = 1
    filter_size = [[5,5,10], [5,5,10],[5,5,10],[5,5,10],[5,5,10],[5,5,10],[5,5,10],[5,5,10],[5,5,10],[5,5,1]]
    pooling_schedule = np.arange(0,10,2)
    weight_scale = None
    centering_data = False
    use_dropout = False

class Sheesh(object):

    def get_data(self):

        """
        Imports all images from directory and converts them to grayscale. Stores all images in a dictionary with
        folder name as the dict key
        :return:
        dictionary of grayscale image stacks with keys:
                                29032019_Gel1
                                29032019_Gel2
                                29032019_Gel3
        """
        stacks = os.listdir('Images')
        images = []
        self.dataset = {}

        for folder in stacks:
            list = os.listdir('Images/' + folder)
            for file in list:
                if file.endswith('.tif'):
                    filepath = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'Images', folder, file)
                    im = swag.imread(filepath)
                    gray = skc.rgb2grey(im)
                    double = img_as_ubyte(gray)
                    images.append(double)
            self.dataset[folder] = images
            dataset = self.dataset
            images = []

        return dataset

    def train_data(self):

        self.data = {}

        skip = 3
        batch_size = 8
        X_train = []
        X_val = []
        y_train = []
        y_val = []
        for i in range(len(self.dataset['29032019_Gel1']) - 2 * skip):
            X_train_batch = []
            X_val_batch = []
            batch = np.arange(i, i + batch_size, 2)
            for j in batch:
                X_train_batch.append(self.dataset['29032019_Gel1'][j])
                X_val_batch.append(self.dataset['29032019_Gel2'][j])
            X_train.append(np.stack(X_train_batch, axis=2))
            X_val.append(np.stack(X_val_batch, axis=2))
            y_train.append(self.dataset['29032019_Gel1'][i + 3])
            y_val.append(self.dataset['29032019_Gel2'][i + 3])

        self.data['y_train'] = np.expand_dims(np.stack(y_train, axis=0), axis=3)
        self.data['y_val'] = np.expand_dims(np.stack(y_val, axis=0), axis=3)
        self.data['X_train'] = np.stack(X_train, axis=0)
        self.data['X_val'] = np.stack(X_val, axis=0)

        """for i in range(10):
            ground = self.data['y_train'][i+3]
            swag.imsave('Series_10_ground_truth' + str(i+3) + '.tif', ground)"""

        return self.data

    def get_params(self):

        self.options = {'centering_data': self.config.centering_data,
                        'use_dropout': self.config.use_dropout}

        self.num_conv_layers = len(self.config.filter_size)
        self.conv_params = {'W':[],
                            'b':[]}
        self.filter = []

        for i in range(self.num_conv_layers):
            if i == 0:
                self.config.filter_size[i].append(self.config.input_size[2])
            else:
                self.config.filter_size[i].append(self.config.filter_size[i-1][2])
            self.filter.append(tuple(self.config.filter_size[i]))
            if self.config.weight_scale is None:
                self.config.weight_scale = np.sqrt(2/np.prod(self.config.filter_size[i]))

            w = self.config.weight_scale*np.random.rand(np.prod(self.config.filter_size[i])).reshape(self.filter[i])
            W = tf.Variable(np.transpose(w,[0,1,3,2]), dtype = tf.float32) #HWDF

            b = tf.Variable(0.01*np.ones(self.filter[i][2]), dtype = tf.float32)

            self.conv_params['W'].append(W)
            self.conv_params['b'].append(b)

        conv_params = self.conv_params

        return conv_params

    def add_placeholders(self):

        self.placeholders = {}

        self.placeholders['x_batch'] = tf.placeholder(dtype = tf.float32,
                                                      shape = [None, self.config.input_size[0],self.config.input_size[1],self.config.input_size[2]])

        self.placeholders['y_batch'] = tf.placeholder(dtype = tf.float32,
                                                      shape = None)

        self.placeholders['x_center'] = tf.placeholder(dtype = tf.float32,
                                                       shape = [self.config.input_size[0],self.config.input_size[1],self.config.input_size[2]])

        self.placeholders['keep_prob'] = tf.placeholder(dtype = tf.float32,
                                                        shape = [])

        self.placeholders['reg_weight'] = tf.placeholder(dtype = tf.float32,
                                                         shape = [])

        self.placeholders['learning_rate'] = tf.placeholder(dtype = tf.float32,
                                                           shape = [])

        self.pool_params = {'pool_width':2,
                            'pool_height':2,
                            'stride':[2,2]}

    def compute_scores(self, X):

        if self.options['centering_data']:
            X = X - self.placeholders['x_center']

        num_conv_layers = len(self.conv_params['W'])
        hidden = X

        for i in range(num_conv_layers):
            W = self.conv_params['W'][i]
            b = self.conv_params['b'][i]

            hidden = tf.nn.conv2d(hidden, W, strides = [1,1,1,1],
                              padding = 'SAME',
                              use_cudnn_on_gpu = False,
                              data_format = 'NHWC')

            hidden = tf.nn.bias_add(hidden,b)

            if self.options['use_dropout']:
                hidden = tf.layers.nn.dropout(hidden, keep_prob = self.placeholders['keep_prob'])
                hidden = tf.nn.relu(hidden)
            else:
                hidden = tf.nn.relu(hidden)

        self.scores = hidden
        scores = self.scores

        return scores

    def regularizer(self,scores):

        reg = np.float(0.0)

        score = tf.squeeze(scores, axis=3)
        norm_term = tf.map_fn(lambda x: 0.001 * tf.reduce_sum(tf.matmul(x, x))/ (tf.cast(x.shape[0], dtype=tf.float32) * tf.cast(x.shape[1], dtype=tf.float32)),
                              score,dtype=tf.float32)
        reg = tf.reduce_sum(norm_term)


        """for score in scores:
            score = tf.squeeze(score,axis = 2)
            reg = reg + self.placeholders['reg_weight'] * tf.reduce_sum(tf.matmul(score,score))/(score[1]*score[2])"""

        return reg

    def compute_objective(self, scores, y):

        self.loss = tf.losses.mean_squared_error(y,scores, reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        reg = self.regularizer(scores)
        tf.losses.add_loss(reg)

        self.objective = self.loss + reg
        objective = self.objective

        return objective

    def __init__(self, config):

        self.config = Config()
        self.get_data()
        self.train_data()
        self.get_params()
        self.add_placeholders()
        self.scores = self.compute_scores(self.placeholders['x_batch'])

        self.operations = {}
        self.operations['y_pred'] = self.compute_scores(self.placeholders['x_batch'])
        self.objective = self.compute_objective(self.scores, self.placeholders['y_batch'])
        self.operations['objective'] = self.objective

        self.minimizer = tf.train.AdamOptimizer(learning_rate = self.placeholders['learning_rate'])
        self.training_step = self.minimizer.minimize(self.objective)
        self.operations['training_step'] = self.training_step

        self.session = tf.Session()
        self.x_center = None

    def get_learned_params(self):

        session = self.session
        session.run(tf.global_variables_initializer())
        conv_params = self.conv_params
        self.learned_params = {}
        keys = len(conv_params['W'])

        for i in range(keys):
            self.learned_params['W' + str(i)] = session.run(conv_params['W'][i])
            self.learned_params['b' + str(i)] = session.run(conv_params['b'][i])

        learned_params = self.learned_params

        return learned_params

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=1.0, keep_prob=1.0,
              reg=np.float32(5e-6), num_iters = 100,
              batch_size= 200, verbose=False):

        num_train = X.shape[0] #591
        iterations_per_epoch = max(num_train / batch_size, 1)

        self.x_center = np.mean(X, axis = 0)

        session = self.session
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        self.objective_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        for i in range(num_iters):

            bo = i*batch_size
            batch = range(bo, min(bo+batch_size,num_train))

            X_batch = X[batch]
            y_batch = y[batch]


            feed_dict = {self.placeholders['x_batch']: X_batch,
                         self.placeholders['y_batch']: y_batch,
                         self.placeholders['learning_rate']:learning_rate,
                         self.placeholders['reg_weight']:reg}

            learning_rate *= learning_rate_decay

            if self.options['centering_data']:
                feed_dict[self.placeholders['x_center']] = self.x_center

            if self.options['use_dropout']:
                feed_dict[self.placeholders['keep_prob']] = np.float32(keep_prob)

            np_objective, _ = session.run([self.operations['objective'],
                                              self.operations['training_step']],
                                             feed_dict=feed_dict)

            self.objective_history.append(np_objective)


            if verbose: #and i % 100 == 0:
                print('iteration %d / %d: objective %f' % (i, num_iters, np_objective))

            # Every epoch, check train and val accuracy and decay learning rate.
            """if i % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = np.float32(self.predict(X_batch) == y_batch).mean()
                val_acc = np.float32(self.predict(X_val) == y_val).mean()
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)"""

        return {
            'objective_history': self.objective_history,
            'train_acc_history': self.train_acc_history,
            'val_acc_history': self.val_acc_history,
        }

    def predict(self,X):

        np_y_pred = self.session.run(self.operations['y_pred'], feed_dict={self.placeholders['x_batch']: X,
                                                                           self.placeholders['x_center']: self.x_center,
                                                                           self.placeholders['keep_prob']: 1.0}
                                     )

        return np_y_pred

    def visualize_filters(self, Xs, ubound = 255, padding = 1):

        (N, H, W, C) = Xs.shape  # 3,1,10,10

        grid_size = int(ceil(sqrt(N)))
        grid_height = H * grid_size + padding * (grid_size - 1)  # 10
        grid_width = W * grid_size + padding * (grid_size - 1)  # 10
        grid = np.zeros((grid_height, grid_width, C))  # preallocating grid space with 0s
        next_idx = 0
        y0, y1 = 0, H  # yo is 0 and y1 is 10
        for y in range(grid_size):  # [0,1)
            x0, x1 = 0, W  # x0 is 0 and x1 is 10
            for x in range(grid_size):  # [0,1)
                if next_idx < N:  # yes when 0: 1 batch
                    img = Xs[next_idx]  # 1st grid in input
                    low, high = np.min(img), np.max(img)  # min and max values of image
                    grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)  # normalize filters to 0,255
                    # grid[y0:y1, x0:x1] = Xs[next_idx]
                    next_idx += 1  # next batch
                x0 += W + padding  # make plot larger by the size of next image+padding
                x1 += W + padding
            y0 += H + padding
            y1 += H + padding
        #grid = np.squeeze(grid, axis = 2)

        return grid

def main():

    config = Config()
    sheesh = Sheesh(config)
    train_data = sheesh.train_data()

    num_train = train_data['X_train'].shape[0] #591
    batch_size = 10
    num_iters = int((num_train -1) / batch_size) #59

    send_it = sheesh.train(train_data['X_train'], train_data['y_train'],
                           train_data['X_val'], train_data['y_val'],
                           num_iters = 59, batch_size = batch_size,
                           verbose = True)

    y_pred = sheesh.predict(train_data['X_train'][0:10])
    for i in range(y_pred.shape[0]):
        camel = np.squeeze(y_pred[i])
        plt.imshow(camel, cmap = 'gray')
        plt.axis('off')
        plt.gcf().set_size_inches(5, 5)
        plt.show()
        swag.imsave('Series_10_prediction_60_iters_norm'+str(i)+'.tif', camel)

    plt.subplot(2,1,1)
    plt.plot(send_it['objective_history'],'o')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(2,1,2)
    plt.plot(send_it['train_acc_history'],'o')
    plt.plot(send_it['val_acc_history'],'o')
    plt.legend(['train','val'], loc = 'upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

    """learned_params = sheesh.get_learned_params()
    #for i in range(len(learned_params)):
    grid = sheesh.visualize_filters(learned_params['W0'].transpose(3,0,1,2))# +str(i)].transpose(3,0,1,2))
    plt.imshow(grid.astype('uint8'))
    plt.axis('off')
    plt.gcf().set_size_inches(5, 5)
    plt.show()"""

main()