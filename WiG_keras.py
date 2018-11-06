#!/usr/bin/env python
# -*- coding: utf-8 -*-


from keras import backend as K
from keras.layers.merge import Multiply
from keras.layers import Dense, Conv2D
from keras.initializers import Initializer
from keras import regularizers

import numpy as np

#####
def to_list(x, allow_tuple=False):
    """Normalizes a list/tensor into a list.
    If a tensor is passed, we return
    a list of size 1 containing the tensor.
    # Arguments
        x: target object to be normalized.
        allow_tuple: If False and x is a tuple,
            it will be converted into a list
            with a single element (the tuple).
            Else converts the tuple to a list.
    # Returns
        A list.
    """
    if isinstance(x, list):
        return x
    if allow_tuple and isinstance(x, tuple):
        return list(x)
    return [x]

##### Dense #####
class WiG_Dense(Dense):
	def __init__(self, units, gate_activity_regularizer=None, **kwargs):
		super(WiG_Dense, self).__init__(units, **kwargs)
		self.gate_activity_regularizer=regularizers.get(gate_activity_regularizer)
	
	def call(self, x):
		output = super(WiG_Dense, self).call(x)
		output = K.sigmoid( output )
		
		# Apply activity regularizer if any:
		if (hasattr(self, 'gate_activity_regularizer') and self.gate_activity_regularizer is not None):
			with K.name_scope('gate_activity_regularizer'):
				regularization_losses = [self.gate_activity_regularizer(z) for z in to_list(output)]
				self.add_loss(regularization_losses,inputs=to_list(x))
				
		return x * output

	def get_config(self):
		config = {
			'gate_activity_regularizer':regularizers.serialize(self.gate_activity_regularizer),
		}
		base_config = super(WiG_Dense, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class Identical_Dense(Initializer):
	def __init__(self, gain=1., stddev=0.):
		self.gain = gain
		self.stddev = stddev

	def __call__(self, shape, dtype=None):
		if( len(shape) != 2 ):
			raise ValueError( 'Identical_Dense can only be used for Dense.' )
		else:
			return np.eye(shape[0]) * self.gain + np.random.randn( shape[0], shape[1] ) * self.stddev

	def get_config(self):
		return {
			'gain': self.gain,
			'stddev': self.stddev,
		}

##### Conv2D #####
class WiG_Conv2D(Conv2D):
	def __init__(self, filters, kernel_size, gate_activity_regularizer=None, **kwargs):
		super(WiG_Conv2D, self).__init__(filters, kernel_size, **kwargs)
		self.gate_activity_regularizer=regularizers.get(gate_activity_regularizer)
	
	def call(self, x):
		output = super(WiG_Conv2D, self).call(x)
		output = K.sigmoid( output )
		
		# Apply activity regularizer if any:
		if (hasattr(self, 'gate_activity_regularizer') and self.gate_activity_regularizer is not None):
			with K.name_scope('gate_activity_regularizer'):
				regularization_losses = [self.gate_activity_regularizer(z) for z in to_list(output)]
				self.add_loss(regularization_losses,inputs=to_list(x))
				
		return x * output

	def get_config(self):
		config = {
			'gate_activity_regularizer':regularizers.serialize(self.gate_activity_regularizer),
		}
		base_config = super(WiG_Conv2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class Identical_Conv2D(Initializer):
	def __init__(self, gain=1., stddev=0.):
		self.gain = gain
		self.stddev = stddev

	def __call__(self, shape, dtype=None):
		if( len(shape) != 4 ):
			raise ValueError( 'Identical_Conv2D can only be used for Conv2D.' )
		else:
			if( shape[2] != shape[3] ):
				raise ValueError( 'Dimensions of input and output features should be same.' )
			else:
				if( self.stddev > 0 ):
					w = np.random.randn( shape[0], shape[1], shape[2], shape[3] ) * self.stddev
				else:
					w = np.zeros( shape )
				c0 = shape[0]//2
				c1 = shape[1]//2
				for channel in range(shape[3]):
					w[c0,c1,channel,channel] += self.gain
			return w

	def get_config(self):
		return {
			'gain': self.gain,
			'stddev': self.stddev,
		}

custom_objects={'WiG_Conv2D':WiG_Conv2D, 'WiG_Dense':WiG_Dense, 'Identical_Dense':Identical_Dense, 'Identical_Conv2D':Identical_Conv2D}

##### SAMPLE #####
if( __name__ == '__main__' ):
	from keras.datasets import cifar10
	from keras.utils import np_utils
	from keras.layers import Conv2D, Input, Flatten, Dropout, Dense, MaxPooling2D, Activation
	from keras.regularizers import l1, l2
	from keras.models import Model, load_model
	from keras.optimizers import Adam
	from keras.metrics import categorical_accuracy
	
	import os

	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	Y_train = np_utils.to_categorical(y_train, 10)
	Y_train = np.single( Y_train )
	Y_test  = np_utils.to_categorical(y_test, 10)
	Y_test = np.single( Y_test )
	
	X_train = X_train.astype('float32')/255.0
	X_test = X_test.astype('float32')/255.0

	Wl2 = 1E-8
	Al1 = 1E-8
	g = 10

	acts = {'ReLU', 'WiG'}

	eva = {}
	for act in acts:
		model_name = 'cifar_sample_{}.hdf5'.format(act)
		if( os.path.exists( model_name ) ):
			model = load_model( model_name, custom_objects=custom_objects )
		else:
			inp = Input(shape=X_train.shape[1:])
			x = inp
			
			x = Conv2D( 32, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Wl2) ) (x)	
			if( act == 'WiG' ):
				x = WiG_Conv2D( 32, (3,3), padding='same', kernel_initializer=Identical_Conv2D(g), gate_activity_regularizer=l1(Al1) ) (x)
			else:
				x = Activation('relu') (x)
			
			x = Conv2D( 32, (3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(Wl2) ) (x)	
			if( act == 'WiG' ):
				x = WiG_Conv2D( 32, (3,3), padding='same', kernel_initializer=Identical_Conv2D(g), gate_activity_regularizer=l1(Al1) ) (x)
			else:
				x = Activation('relu') (x)

			x = MaxPooling2D((2,2)) (x)


			x = Conv2D( 64, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Wl2) ) (x)	
			if( act == 'WiG' ):
				x = WiG_Conv2D( 64, (3,3), padding='same', kernel_initializer=Identical_Conv2D(g), gate_activity_regularizer=l1(Al1) ) (x)
			else:
				x = Activation('relu') (x)
			
			x = Conv2D( 64, (3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(Wl2) ) (x)	
			if( act == 'WiG' ):
				x = WiG_Conv2D( 64, (3,3), padding='same', kernel_initializer=Identical_Conv2D(g), gate_activity_regularizer=l1(Al1) ) (x)
			else:
				x = Activation('relu') (x)

			x = MaxPooling2D((2,2)) (x)
			
			x = Flatten() (x)
			
			x = Dropout(0.5) (x)
			x = Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(Wl2)) (x)
			if( act == 'WiG' ):
				x = WiG_Dense(128, kernel_initializer=Identical_Dense(g), gate_activity_regularizer=l1(Al1) ) (x)
			else:
				x = Activation('relu') (x)

			x = Dropout(0.5) (x)
			x = Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(Wl2)) (x)
			if( act == 'WiG' ):
				x = WiG_Dense(128, kernel_initializer=Identical_Dense(g), gate_activity_regularizer=l1(Al1) ) (x)
			else:
				x = Activation('relu') (x)

			x = Dropout(0.5) (x)
			x = Dense(10, kernel_initializer='he_normal', kernel_regularizer=l2(Wl2)) (x)

			x = Activation('softmax') (x)
			
			model = Model(inputs=inp, outputs=x)
			model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy', 'categorical_crossentropy'])	
		
			model.fit( x=X_train, y=Y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_test,Y_test ) )
			model.save( model_name )
		
		eva[act] = model.evaluate( x=X_test, y=Y_test, verbose=0 )
	
	for i in range(1,3):
		print( 'val_' + model.metrics_names[i] + ':' )
		for act in acts:
			print( ' {act:5s}: {eva}'.format(act=act, eva=eva[act][i]) )
		print()
	print( ' NOTE: This is just sample. The network structure is different from the paper version.' )
