#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
K.set_image_data_format('channels_last')

from keras.layers import Layer, Activation, LeakyReLU, PReLU, ELU, ThresholdedReLU
from WiG import WiG_Dense, WiG_Conv2D
import os

class Swish(Layer):
	def __init__(self, **kwargs):
		super(Swish, self).__init__(**kwargs)
		
	def build(self, input_shape):
		self.beta = self.add_weight(name='beta', shape=(1,1,1,input_shape[-1]), initializer='ones', trainable=True)
		super(Swish, self).build(input_shape)
	
	def call(self, x):
		return x * K.sigmoid( self.beta * x )

	def compute_output_shape(self, input_shape):
		return input_shape

class SiL(Layer):
	def __init__(self, **kwargs):
		super(SiL, self).__init__(**kwargs)
		
	def build(self, input_shape):
		super(SiL, self).build(input_shape)
	
	def call(self, x):
		return x * K.sigmoid( x )

	def compute_output_shape(self, input_shape):
		return input_shape

def get( activation, nb_features = 0, **kwargs ):
	if( activation == 'WiG_Dense' ):
		if( not 'kernel_initializer' in kwargs ):
			kwargs['kernel_initializer'] = 'zeros'

		return WiG_Dense(nb_features, **kwargs)

	elif( activation == 'WiG_Conv2D' ):
		if( not 'kernel_size' in kwargs ):
			kwargs['kernel_size'] = (3,3)
			
		if( not 'padding' in kwargs ):
			kwargs['padding'] = 'same'
			
		if( not 'kernel_initializer' in kwargs ):
			kwargs['kernel_initializer'] = 'zeros'
			
		return WiG_Conv2D(nb_features, **kwargs)
		
	elif( activation == 'LeakyReLU' ):
		return LeakyReLU(**kwargs)

	elif( activation == 'PReLU' ):
		return PReLU(**kwargs)
	
	elif( activation == 'ELU' ):
		return ELU(**kwargs)
	
	elif( activation == 'ThresholdedReLU' ):
		return ThresholdedReLU(**kwargs)

	elif( activation == 'Swish' ):
		return Swish(**kwargs)

	elif( activation == 'SiL' ):
		return SiL(**kwargs)

	return Activation(activation,**kwargs)

def afterDense( activation, nb_features = 0, **kwargs ):
	if( activation == 'WiG' ):
		return get( 'WiG_Dense', nb_features=nb_features, **kwargs )
	return get( activation, nb_features=nb_features, **kwargs )

def afterConv2D( activation, nb_features = 0, **kwargs ):
	if( activation == 'WiG' ):
		return get( 'WiG_Conv2D', nb_features=nb_features, **kwargs )
	return get( activation, nb_features=nb_features, **kwargs )


custom_objects = {'WiG_Conv2D':WiG_Conv2D, 'WiG_Dense':WiG_Dense, 'SiL':SiL, 'Swish': Swish}

if( __name__ == '__main__' ):
	import os
	from keras.models import Sequential, load_model
	from keras.layers import Dense
	
	act = get('WiG_Conv2D', 10)
	
	nb_features = 256
	temp = '__activation__.hdf5'
	activations = [ 'softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear', 'LeakyReLU', 'PReLU', 'ThresholdedReLU', 'SiL', 'Swish', 'WiG_Dense' ]
	for act in activations:
		print( act )
		name = act
		model = Sequential()
		model.add( Dense( nb_features, input_shape=(128,) ) )
		model.add( get( act, name=name, nb_features=nb_features ) )
		print( model.get_layer(name).name )
		model.save( temp )
		model = load_model( temp, custom_objects = custom_objects )
	os.remove(temp)
