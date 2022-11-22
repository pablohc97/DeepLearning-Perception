import tensorflow as tf
from tensorflow.keras import layers


class KernelIdentity(tf.keras.initializers.Initializer):
    def __init__(self, gain):
        self.gain = gain

    def __call__(self, shape, dtype = None):
        """
        shape has the form [Kx, Ky, Cin, Cout] disregarding data_format.
        """
        identity_matrix = tf.eye(shape[0])*self.gain
        identity_matrix = tf.expand_dims(identity_matrix, axis = -1)
        identity_matrix = tf.expand_dims(identity_matrix, axis = -1)
        identity_matrix = tf.repeat(identity_matrix, shape[2], axis = -2)
        identity_matrix = tf.repeat(identity_matrix, shape[3], axis = -1)
        return identity_matrix
    
    def get_config(self):
        return {'gain':self.gain}

class PearsonCorrelation(tf.keras.losses.Loss):
    """
    Loss used to train PerceptNet. Is calculated as the 
    Pearson Correlation Coefficient for a sample.
    """
    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        y_true_mean = tf.reduce_mean(y_true)
        y_pred_mean = tf.reduce_mean(y_pred)
        num = y_true-y_true_mean
        num *= y_pred-y_pred_mean
        num = tf.reduce_sum(num)
        denom = tf.sqrt(tf.reduce_sum((y_true-y_true_mean)**2))
        denom *= tf.sqrt(tf.reduce_sum((y_pred-y_pred_mean)**2))
        return num/denom

class GDN(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size = 3,
                 gamma_init = .1,
                 alpha_init = 2,
                 epsilon_init = 1/2,
                 alpha_trainable = False,
                 epsilon_trainable = False,
                 reparam_offset = 2**(-18),
                 beta_min = 1e-6,
                 apply_independently = False,
                 kernel_initializer = "identity",
                 data_format = "channels_last",
                 **kwargs):

        super(GDN, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset
        self.beta_min = beta_min
        self.beta_reparam = (self.beta_min+self.reparam_offset**2)**(1/2)
        self.apply_independently = apply_independently
        self.kernel_initializer = KernelIdentity(gain = gamma_init) if kernel_initializer == "identity" else kernel_initializer
        self.data_format = data_format
        self.alpha_init = alpha_init
        self.epsilon_init = epsilon_init
        self.alpha_trainable = alpha_trainable
        self.epsilon_trainable = epsilon_trainable        

    def build(self, input_shape):
        n_channels = input_shape[-1] if self.data_format == "channels_last" else input_shape[0]

        if self.data_format == "channels_last":
            n_channels = input_shape[-1]
        elif self.data_format == "channels_first":
            n_channels = input_shape[0]
        else:
            raise ValueError("data_format not supported")

        if self.apply_independently:
            self.groups = n_channels
        else:
            self.groups = 1

        self.conv = layers.Conv2D(filters = n_channels,
                                  kernel_size = self.kernel_size,
                                  padding = "valid",
                                  strides = 1,
                                  groups = self.groups,
                                  data_format = self.data_format,
                                  trainable = True,
                                  kernel_initializer = self.kernel_initializer,
                                  kernel_constraint = lambda x: tf.clip_by_value(x, 
                                                                                 clip_value_min = self.reparam_offset,
                                                                                 clip_value_max = tf.float32.max),
                                  bias_initializer = "ones",
                                  bias_constraint = lambda x: tf.clip_by_value(x, 
                                                                               clip_value_min = self.beta_reparam,
                                                                               clip_value_max = tf.float32.max))
        self.conv.build(input_shape)

        self.alpha = self.add_weight(shape = (1),
                                     initializer = tf.keras.initializers.Constant(self.alpha_init),
                                     trainable = self.alpha_trainable,
                                     name = 'alpha')
        self.epsilon = self.add_weight(shape = (1),
                                       initializer = tf.keras.initializers.Constant(self.epsilon_init),
                                       trainable = self.epsilon_trainable,
                                       name = 'epsilon')

    def call(self, X):
        X_pad = tf.pad(X, 
                       mode = 'REFLECT',
                       paddings = tf.constant([[0, 0], # Batch dim
                                               [int((self.kernel_size-1)/2),
                                                int((self.kernel_size-1)/2)], 
                                               [int((self.kernel_size-1)/2), 
                                                int((self.kernel_size-1)/2)], 
                                               [0, 0]]))
        norm_pool = self.conv(tf.pow(X_pad, self.alpha))
        norm_pool = tf.pow(norm_pool, self.epsilon)

        return X / norm_pool

    def get_config(self):
        """
        Returns a dictionary used to initialize this layer. Is used when
        saving the layer or a model that contains it.
        """
        base_config = super(GDN, self).get_config()
        config = {'alpha':self.alpha,
                  'epsilon':self.epsilon}
        return dict(list(base_config.items()) + list(config.items()))


class Inverse_GDN(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size = 3,
                 gamma_init = .1,
                 alpha_init = 2,
                 epsilon_init = 1/2,
                 alpha_trainable = False,
                 epsilon_trainable = False,
                 reparam_offset = 2**(-18),
                 beta_min = 1e-6,
                 apply_independently = False,
                 kernel_initializer = "identity",
                 data_format = "channels_last",
                 **kwargs):

        super(Inverse_GDN, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset
        self.beta_min = beta_min
        self.beta_reparam = (self.beta_min+self.reparam_offset**2)**(1/2)
        self.apply_independently = apply_independently
        self.kernel_initializer = KernelIdentity(gain = gamma_init) if kernel_initializer == "identity" else kernel_initializer
        self.data_format = data_format
        self.alpha_init = alpha_init
        self.epsilon_init = epsilon_init
        self.alpha_trainable = alpha_trainable
        self.epsilon_trainable = epsilon_trainable        

    def build(self, input_shape):
        n_channels = input_shape[-1] if self.data_format == "channels_last" else input_shape[0]

        if self.data_format == "channels_last":
            n_channels = input_shape[-1]
        elif self.data_format == "channels_first":
            n_channels = input_shape[0]
        else:
            raise ValueError("data_format not supported")

        if self.apply_independently:
            self.groups = n_channels
        else:
            self.groups = 1

        self.conv = layers.Conv2D(filters = n_channels,
                                  kernel_size = self.kernel_size,
                                  padding = "valid",
                                  strides = 1,
                                  groups = self.groups,
                                  data_format = self.data_format,
                                  trainable = True,
                                  kernel_initializer = self.kernel_initializer,
                                  kernel_constraint = lambda x: tf.clip_by_value(x, 
                                                                                 clip_value_min = self.reparam_offset,
                                                                                 clip_value_max = tf.float32.max),
                                  bias_initializer = "ones",
                                  bias_constraint = lambda x: tf.clip_by_value(x, 
                                                                               clip_value_min = self.beta_reparam,
                                                                               clip_value_max = tf.float32.max))
        self.conv.build(input_shape)

        self.alpha = self.add_weight(shape = (1),
                                     initializer = tf.keras.initializers.Constant(self.alpha_init),
                                     trainable = self.alpha_trainable,
                                     name = 'alpha')
        self.epsilon = self.add_weight(shape = (1),
                                       initializer = tf.keras.initializers.Constant(self.epsilon_init),
                                       trainable = self.epsilon_trainable,
                                       name = 'epsilon')

    def call(self, X):
        X_pad = tf.pad(X, 
                       mode = 'REFLECT',
                       paddings = tf.constant([[0, 0], # Batch dim
                                               [int((self.kernel_size-1)/2),
                                                int((self.kernel_size-1)/2)], 
                                               [int((self.kernel_size-1)/2), 
                                                int((self.kernel_size-1)/2)], 
                                               [0, 0]]))
        norm_pool = self.conv(tf.pow(X_pad, self.alpha))
        norm_pool = tf.pow(norm_pool, self.epsilon)

        return X * norm_pool

    def get_config(self):
        """
        Returns a dictionary used to initialize this layer. Is used when
        saving the layer or a model that contains it.
        """
        base_config = super(GDN, self).get_config()
        config = {'alpha':self.alpha,
                  'epsilon':self.epsilon}
        return dict(list(base_config.items()) + list(config.items()))


class PerceptNetAutoEncoder(tf.keras.Model):
    def __init__(self, kernel_initializer = 'identity', gdn_kernel_size = 1):
        super(PerceptNetAutoEncoder, self).__init__()
        self.gdn1 = GDN(kernel_size = gdn_kernel_size, apply_independently = True, kernel_initializer = kernel_initializer)
        self.conv1 = layers.Conv2D(filters = 3, kernel_size = 1, strides = 1, padding = 'same')
        self.undersampling1 = layers.MaxPool2D(2)
        self.gdn2 = GDN(kernel_size = gdn_kernel_size, kernel_initializer = kernel_initializer)
        self.conv2 = layers.Conv2D(filters = 6, kernel_size = 5, strides = 1, padding = 'same')
        self.undersampling2 =  layers.MaxPool2D(2)
        self.gdn3 = GDN(kernel_size = gdn_kernel_size, kernel_initializer = kernel_initializer)
        self.conv3 = layers.Conv2D(filters = 128, kernel_size = 5, strides = 1, padding = 'same')

        self.gdn4 = GDN(kernel_size = gdn_kernel_size, kernel_initializer = kernel_initializer)
        self.inversegdn1 = Inverse_GDN(kernel_size = gdn_kernel_size, kernel_initializer = kernel_initializer)

        self.conv4 = layers.Conv2D(filters = 6, kernel_size = 5, strides = 1, padding = 'same')
        self.inversegdn2 = Inverse_GDN(kernel_size = gdn_kernel_size, kernel_initializer = kernel_initializer)
        self.upsampling1 =  layers.UpSampling2D(2)
        self.conv5 = layers.Conv2D(filters = 3, kernel_size = 5, strides = 1, padding = 'same')
        self.inversegdn3 = Inverse_GDN(kernel_size = gdn_kernel_size, kernel_initializer = kernel_initializer)
        self.upsampling2 =  layers.UpSampling2D(2)
        self.conv6 = layers.Conv2D(filters = 3, kernel_size = 1, strides = 1, padding = 'same')
        self.inversegdn4 = Inverse_GDN(kernel_size = gdn_kernel_size, apply_independently = True, kernel_initializer = kernel_initializer)


    def call(self, X):
        #Encoder
        output = self.gdn1(X)
        output = self.conv1(output)
        output = self.undersampling1(output)
        output = self.gdn2(output)
        output = self.conv2(output)
        output = self.undersampling2(output)
        output = self.gdn3(output)
        output = self.conv3(output)
        #Bottleneck
        output = self.gdn4(output)
        output = self.inversegdn1(output)
        #Decoder
        output = self.conv4(output)
        output = self.inversegdn2(output)
        output = self.upsampling1(output)
        output = self.conv5(output)
        output = self.inversegdn3(output)
        output = self.upsampling2(output)
        output = self.conv6(output)
        output = self.inversegdn4(output)
        return output

    def train_step(self, img):
        with tf.GradientTape() as tape:
            reconstructed_img = self(img)
            l2 = (reconstructed_img-img)**2
            l2 = tf.reduce_sum(l2, axis = [1,2,3])/(img.shape[1]*img.shape[2]*img.shape[3])
            loss = tf.sqrt(l2)
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {'reconstruction_loss':loss}
    
    def test_step(self, img):
        reconstructed_img = self(img)
        l2 = (reconstructed_img-img)**2
        l2 = tf.reduce_sum(l2, axis = [1,2,3])/(img.shape[1]*img.shape[2]*img.shape[3])
        loss = tf.sqrt(l2)
        loss = tf.reduce_mean(loss)

        return {'reconstruction_loss':loss}


class PerceptNetAutoEncoderOverComplete(tf.keras.Model):
    def __init__(self, kernel_initializer = 'identity', gdn_kernel_size = 1):
        super(PerceptNetAutoEncoderOverComplete, self).__init__()
        self.gdn1 = GDN(kernel_size = gdn_kernel_size, apply_independently = True, kernel_initializer = kernel_initializer)
        self.conv1 = layers.Conv2D(filters = 12, kernel_size = 1, strides = 1, padding = 'same')
        self.undersampling1 = layers.MaxPool2D(2)
        self.gdn2 = GDN(kernel_size = gdn_kernel_size, kernel_initializer = kernel_initializer)
        self.conv2 = layers.Conv2D(filters = 48, kernel_size = 5, strides = 1, padding = 'same')
        self.undersampling2 =  layers.MaxPool2D(2)
        self.gdn3 = GDN(kernel_size = gdn_kernel_size, kernel_initializer = kernel_initializer)
        self.conv3 = layers.Conv2D(filters = 128, kernel_size = 5, strides = 1, padding = 'same')

        self.gdn4 = GDN(kernel_size = gdn_kernel_size, kernel_initializer = kernel_initializer)
        self.inversegdn1 = Inverse_GDN(kernel_size = gdn_kernel_size, kernel_initializer = kernel_initializer)

        self.conv4 = layers.Conv2D(filters = 48, kernel_size = 5, strides = 1, padding = 'same')
        self.inversegdn2 = Inverse_GDN(kernel_size = gdn_kernel_size, kernel_initializer = kernel_initializer)
        self.upsampling1 =  layers.UpSampling2D(2)
        self.conv5 = layers.Conv2D(filters = 12, kernel_size = 5, strides = 1, padding = 'same')
        self.inversegdn3 = Inverse_GDN(kernel_size = gdn_kernel_size, kernel_initializer = kernel_initializer)
        self.upsampling2 =  layers.UpSampling2D(2)
        self.conv6 = layers.Conv2D(filters = 3, kernel_size = 1, strides = 1, padding = 'same')
        self.inversegdn4 = Inverse_GDN(kernel_size = gdn_kernel_size, apply_independently = True, kernel_initializer = kernel_initializer)


    def call(self, X):
        #Encoder
        output = self.gdn1(X)
        output = self.conv1(output)
        output = self.undersampling1(output)
        output = self.gdn2(output)
        output = self.conv2(output)
        output = self.undersampling2(output)
        output = self.gdn3(output)
        output = self.conv3(output)
        #Bottleneck
        output = self.gdn4(output)
        output = self.inversegdn1(output)
        #Decoder
        output = self.conv4(output)
        output = self.inversegdn2(output)
        output = self.upsampling1(output)
        output = self.conv5(output)
        output = self.inversegdn3(output)
        output = self.upsampling2(output)
        output = self.conv6(output)
        output = self.inversegdn4(output)
        return output

    def train_step(self, img):
        with tf.GradientTape() as tape:
            reconstructed_img = self(img)
            l2 = (reconstructed_img-img)**2
            l2 = tf.reduce_sum(l2, axis = [1,2,3])/(img.shape[1]*img.shape[2]*img.shape[3])
            loss = tf.sqrt(l2)
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {'reconstruction_loss':loss}
    
    def test_step(self, img):
        reconstructed_img = self(img)
        l2 = (reconstructed_img-img)**2
        l2 = tf.reduce_sum(l2, axis = [1,2,3])/(img.shape[1]*img.shape[2]*img.shape[3])
        loss = tf.sqrt(l2)
        loss = tf.reduce_mean(loss)

        return {'reconstruction_loss':loss}
