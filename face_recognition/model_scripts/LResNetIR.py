# modified from mxnet to tensorflow from https://github.com/deepinsight/insightface/blob/master/recognition/symbol/fresnet.py

import tensorflow as tf


def Conv(data, **kwargs):
	# name = kwargs.get('name')
	# _weight = mx.symbol.Variable(name+'_weight')
	# _bias = mx.symbol.Variable(name+'_bias', lr_mult=2.0, wd_mult=0.0)
	# body = mx.sym.Convolution(weight = _weight, bias = _bias, **kwargs)
	kwargs["kernel_size"] = kwargs["kernel"]
	kwargs["filters"] = kwargs["num_filter"]
	kwargs["strides"] = kwargs["stride"]
	padding="valid"
	try:
		# data = tf.keras.layers.ZeroPadding2D(kwargs["pad"])(data)
		padding="same"
		del kwargs["pad"]
	except KeyError:
		pass

	del kwargs["kernel"]
	del kwargs["num_filter"]
	del kwargs["stride"]
	body = tf.keras.layers.Conv2D(padding=padding, **kwargs)(data)
	return body


def Act(data, act_type, name):
	if act_type == 'prelu':
		body = tf.keras.layers.PReLU(name=name)(data)
	else:
		body = tf.keras.layers.Activation(act_type, name=name)(data)
	return body


def residual_unit_v1(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
	"""Return ResNet Unit symbol for building ResNet
	Parameters
	----------
	data : str
		Input data
	num_filter : int
		Number of output channels
	bnf : int
		Bottle neck channels factor with regard to num_filter
	stride : tuple
		Stride used in convolution
	dim_match : Boolean
		True means channel number between input and output is the same, otherwise means differ
	name : str
		Base name of the operators
	workspace : int
		Workspace used in convolution operator
	"""
	use_se = kwargs.get('version_se', 1)
	bn_mom = kwargs.get('bn_mom', 0.9)
	workspace = kwargs.get('workspace', 256)
	memonger = kwargs.get('memonger', False)
	act_type = kwargs.get('version_act', 'prelu')
	# print('in unit1')
	if bottle_neck:
		conv1 = Conv(data=data, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=stride, pad=(0, 0),
					 use_bias=False, name=name + '_conv1')
		bn1 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn1')(conv1)
		act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
		conv2 = Conv(data=act1, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=(1, 1), pad=(1, 1),
					 use_bias=False, name=name + '_conv2')
		bn2 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn2')(conv2)
		act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
		conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), use_bias=False, name=name + '_conv3')
		bn3 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn3')(conv3)

		if use_se:
			# se begin
			body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool1')(bn3)
			body = Conv(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv1")
			body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
			body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv2")
			body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
			bn3 = tf.keras.layers.Multiply()([bn3, body])
		# se end

		if dim_match:
			shortcut = data
			x = tf.keras.layers.Add()([bn3, shortcut])
		else:
			conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False, name=name + '_conv1sc')
			shortcut = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_sc')(conv1sc)
			x = tf.keras.layers.Add()([bn3, shortcut])
		return Act(data=x, act_type=act_type, name=name + '_relu3')
	else:
		conv1 = Conv(data=data, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
					 use_bias=False, name=name + '_conv1')
		bn1 = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=2e-5, name=name + '_bn1')(conv1)
		act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
		conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
					 use_bias=False, name=name + '_conv2')
		bn2 = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=2e-5, name=name + '_bn2')(conv2)
		if use_se:
			# se begin
			body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool2')(bn2)
			body = Conv(data=body, num_filter=int(num_filter // 16), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv1")
			body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
			body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv2")
			body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
			bn2 = tf.keras.layers.Multiply()([bn2, body])
		# se end

		if dim_match:
			shortcut = data
			x = tf.keras.layers.Add()([bn2 + shortcut])
		else:
			conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False,
						   name=name + '_conv1sc')
			shortcut = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_sc')(conv1sc)
			x = tf.keras.layers.Add()([bn2 + shortcut])
		return Act(data=x, act_type=act_type, name=name + '_relu3')


def residual_unit_v1_L(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
	"""Return ResNet Unit symbol for building ResNet
	Parameters
	----------
	data : str
		Input data
	num_filter : int
		Number of output channels
	bnf : int
		Bottle neck channels factor with regard to num_filter
	stride : tuple
		Stride used in convolution
	dim_match : Boolean
		True means channel number between input and output is the same, otherwise means differ
	name : str
		Base name of the operators
	workspace : int
		Workspace used in convolution operator
	"""
	use_se = kwargs.get('version_se', 1)
	bn_mom = kwargs.get('bn_mom', 0.9)
	workspace = kwargs.get('workspace', 256)
	memonger = kwargs.get('memonger', False)
	act_type = kwargs.get('version_act', 'prelu')
	# print('in unit1')
	if bottle_neck:
		conv1 = Conv(data=data, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
					 use_bias=False, name=name + '_conv1')
		bn1 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn1')(conv1)
		act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
		conv2 = Conv(data=act1, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=(1, 1), pad=(1, 1),
					 use_bias=False, name=name + '_conv2')
		bn2 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn2')(conv2)
		act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
		conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), use_bias=False, name=name + '_conv3')
		bn3 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn3')(conv3)

		if use_se:
			# se begin
			body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool1')(bn3)
			body = Conv(data=body, num_filter=int(num_filter // 16), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv1")
			body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
			body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv2")
			body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
			bn3 = tf.keras.layers.Multiply()([bn3, body])
		# se end

		if dim_match:
			shortcut = data
			x = tf.keras.layers.Add()([bn3, shortcut])
		else:
			conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False, name=name + '_conv1sc')
			shortcut = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_sc')(conv1sc)
			x = tf.keras.layers.Add()([bn3, shortcut])
		return Act(data=x, act_type=act_type, name=name + '_relu3')
	else:
		conv1 = Conv(data=data, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
					 use_bias=False, name=name + '_conv1')
		bn1 = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=2e-5, name=name + '_bn1')(conv1)
		act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
		conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
					 use_bias=False, name=name + '_conv2')
		bn2 = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=2e-5, name=name + '_bn2')(conv2)
		if use_se:
			# se begin
			body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool1')(bn2)
			body = Conv(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv1")
			body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
			body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv2")
			body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
			bn2 = tf.keras.layers.Multiply()([bn2, body])
		# se end

		if dim_match:
			shortcut = data
			x = tf.keras.layers.Add()([bn2 + shortcut])
		else:
			conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False,
							name=name + '_conv1sc')
			shortcut = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=2e-5, name=name + '_sc')(conv1sc)
			x = tf.keras.layers.Add()([bn2 + shortcut])
		return Act(data=x, act_type=act_type, name=name + '_relu3')


def residual_unit_v2(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
	"""Return ResNet Unit symbol for building ResNet
	Parameters
	----------
	data : str
		Input data
	num_filter : int
		Number of output channels
	bnf : int
		Bottle neck channels factor with regard to num_filter
	stride : tuple
		Stride used in convolution
	dim_match : Boolean
		True means channel number between input and output is the same, otherwise means differ
	name : str
		Base name of the operators
	workspace : int
		Workspace used in convolution operator
	"""
	use_se = kwargs.get('version_se', 1)
	bn_mom = kwargs.get('bn_mom', 0.9)
	workspace = kwargs.get('workspace', 256)
	memonger = kwargs.get('memonger', False)
	act_type = kwargs.get('version_act', 'prelu')
	# print('in unit2')
	if bottle_neck:
		# the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
		bn1 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn1')(data)
		act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
		conv1 = Conv(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
					 use_bias=False, name=name + '_conv1')
		bn2 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn2')(conv1)
		act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
		conv2 = Conv(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1),
					 use_bias=False, name=name + '_conv2')
		bn3 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn3')(conv2)
		act3 = Act(data=bn3, act_type=act_type, name=name + '_relu3')
		conv3 = Conv(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), use_bias=False, name=name + '_conv3')
		if use_se:
			# se begin
			body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool1')(conv3)
			body = Conv(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv1")
			body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
			body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv2")
			body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
			conv3 = tf.keras.layers.Multiply()([conv3, body])
		if dim_match:
			shortcut = data
			x = tf.keras.layers.Add()([conv3 + shortcut])
		else:
			shortcut = Conv(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False,
							name=name + '_sc')
			x = tf.keras.layers.Add()([conv3 + shortcut])
		return x
	else:
		bn1 = tf.keras.layers.BatchNormalization( momentum=bn_mom, epsilon=2e-5, name=name + '_bn1')(data)
		act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
		conv1 = Conv(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
					 use_bias=False, name=name + '_conv1')
		bn2 = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=2e-5, name=name + '_bn2')(conv1)
		act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
		conv2 = Conv(data=act2, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
					 use_bias=False, name=name + '_conv2')
		if use_se:
			# se begin
			body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool1')(conv2)
			body = Conv(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv1")
			body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
			body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv2")
			body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
			conv2 = tf.keras.layers.Multiply()([conv2, body])
		if dim_match:
			shortcut = data
		else:
			shortcut = Conv(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False,
							name=name + '_sc')
		x = tf.keras.layers.Add()([conv2 + shortcut])
		return x


def residual_unit_v3(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
	"""Return ResNet Unit symbol for building ResNet
	Parameters
	----------
	data : str
		Input data
	num_filter : int
		Number of output channels
	bnf : int
		Bottle neck channels factor with regard to num_filter
	stride : tuple
		Stride used in convolution
	dim_match : Boolean
		True means channel number between input and output is the same, otherwise means differ
	name : str
		Base name of the operators
	workspace : int
		Workspace used in convolution operator
	"""
	use_se = kwargs.get('version_se', 1)
	bn_mom = kwargs.get('bn_mom', 0.9)
	workspace = kwargs.get('workspace', 256)
	memonger = kwargs.get('memonger', False)
	act_type = kwargs.get('version_act', 'prelu')
	# print('in unit3')
	if bottle_neck:
		bn1 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn1')(data)
		conv1 = Conv(data=bn1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
					 use_bias=False, name=name + '_conv1')
		bn2 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn2')(conv1)
		act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1')
		conv2 = Conv(data=act1, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=(1, 1), pad=(1, 1),
					 use_bias=False, name=name + '_conv2')
		bn3 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn3')(conv2)
		act2 = Act(data=bn3, act_type=act_type, name=name + '_relu2')
		conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), use_bias=False, name=name + '_conv3')
		bn4 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn4')(conv3)

		if use_se:
			# se begin
			body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool1')(bn4)
			body = Conv(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv1")
			body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
			body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv2")

			body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
			bn4 = tf.keras.layers.Multiply()([bn4, body])
		# se end

		if dim_match:
			shortcut = data
		else:
			conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False,
						   name=name + '_conv1sc')
			shortcut = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_sc')(conv1sc)

		x = tf.keras.layers.Add()([bn4 + shortcut])
		return x
	else:
		bn1 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn1')(data)
		conv1 = Conv(data=bn1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
					 use_bias=False, name=name + '_conv1')
		bn2 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn2')(conv1)
		act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1')
		conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
					 use_bias=False, name=name + '_conv2')
		bn3 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn3')(conv2)
		if use_se:
			# se begin
			body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool1')(bn3)
			body = Conv(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv1")
			body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
			body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv2")
			body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
			bn3 = tf.keras.layers.Multiply()([bn3, body])
		# se end

		if dim_match:
			shortcut = data
		else:
			conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False,
						   name=name + '_conv1sc')
			shortcut = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=2e-5, name=name + '_sc')(conv1sc)

		x = tf.keras.layers.Add()([bn3, shortcut])
		return x


def residual_unit_v3_x(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
	"""Return ResNeXt Unit symbol for building ResNeXt
	Parameters
	----------
	data : str
		Input data
	num_filter : int
		Number of output channels
	bnf : int
		Bottle neck channels factor with regard to num_filter
	stride : tuple
		Stride used in convolution
	dim_match : Boolean
		True means channel number between input and output is the same, otherwise means differ
	name : str
		Base name of the operators
	workspace : int
		Workspace used in convolution operator
	"""
	assert (bottle_neck)
	use_se = kwargs.get('version_se', 1)
	bn_mom = kwargs.get('bn_mom', 0.9)
	workspace = kwargs.get('workspace', 256)
	memonger = kwargs.get('memonger', False)
	act_type = kwargs.get('version_act', 'prelu')
	num_group = 32
	# print('in unit3')
	bn1 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn1')(data)
	conv1 = Conv(data=bn1, num_group=num_group, num_filter=int(num_filter * 0.5), kernel=(1, 1), stride=(1, 1),
				 pad=(0, 0),
				 use_bias=False, name=name + '_conv1')
	bn2 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn2')(conv1)
	act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1')
	conv2 = Conv(data=act1, num_group=num_group, num_filter=int(num_filter * 0.5), kernel=(3, 3), stride=(1, 1),
				 pad=(1, 1),
				 use_bias=False, name=name + '_conv2')
	bn3 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn3')(conv2)
	act2 = Act(data=bn3, act_type=act_type, name=name + '_relu2')
	conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), use_bias=False, name=name + '_conv3')
	bn4 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn4')(conv3)

	if use_se:
		# se begin
		body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool1')(bn4)
		body = Conv(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
					name=name + "_se_conv1")
		body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
		body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
					name=name + "_se_conv2")
		body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
		bn4 = tf.keras.layers.Multiply()([bn4, body])
	# se end

	if dim_match:
		shortcut = data
	else:
		conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False, name=name + '_conv1sc')
		shortcut = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_sc')(conv1sc)

	x = tf.keras.layers.Add()([bn4 + shortcut])
	return x


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
	uv = kwargs.get('version_unit', 3)
	version_input = kwargs.get('version_input', 1)
	if uv == 1:
		if version_input == 0:
			return residual_unit_v1(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
		else:
			return residual_unit_v1_L(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
	elif uv == 2:
		return residual_unit_v2(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
	elif uv == 4:
		return residual_unit_v4(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
	else:
		return residual_unit_v3(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)


def get_fc1(last_conv, num_classes, fc_type, input_channel=512):
	bn_mom = 0.9
	body = last_conv

	return body


def resnet(units, num_stages, filter_list, num_classes, bottle_neck):
	bn_mom = 0.9
	kwargs = {'version_se': 0,
			  'version_input': 1,
			  'version_output': "E",
			  'version_unit': 3,
			  'version_act': "prelu",
			  'bn_mom': bn_mom,
			  }
	"""Return ResNet symbol of
	Parameters
	----------
	units : list
		Number of units in each stage
	num_stages : int
		Number of stage
	filter_list : list
		Channel size of each stage
	num_classes : int
		Ouput size of symbol
	dataset : str
		Dataset type, only cifar10 and imagenet supports
	workspace : int
		Workspace used in convolution operator
	"""
	version_se = kwargs.get('version_se', 1)
	version_input = kwargs.get('version_input', 1)
	assert version_input >= 0
	version_output = kwargs.get('version_output', 'E')
	fc_type = version_output
	version_unit = kwargs.get('version_unit', 3)
	act_type = kwargs.get('version_act', 'prelu')
	memonger = kwargs.get('memonger', False)
	print(version_se, version_input, version_output, version_unit, act_type, memonger)
	num_unit = len(units)
	assert (num_unit == num_stages)
	data = tf.keras.layers.Input((112, 112, 3))
	body = Conv(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
				use_bias=False, name="conv0")
	body = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name='bn0')(body)
	body = Act(data=body, act_type=act_type, name='relu0')

	for i in range(num_stages):
		# if version_input==0:
		#  body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
		#                       name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
		# else:
		#  body = residual_unit(body, filter_list[i+1], (2, 2), False,
		#    name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
		body = residual_unit(body, filter_list[i + 1], (2, 2), False,
							 name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
		for j in range(units[i] - 1):
			body = residual_unit(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
								 bottle_neck=bottle_neck, **kwargs)

	if bottle_neck:
		body = Conv(data=body, num_filter=512, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
					use_bias=False, name="convd")
		body = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name='bnd')(body)
		body = Act(data=body, act_type=act_type, name='relu')  # relud?

	body = get_fc1(body, num_classes, fc_type)
	return data, body


def get_symbol(num_layers: int = 100):
	num_classes = 85000
	if num_layers >= 500:
		filter_list = [64, 256, 512, 1024, 2048]
		bottle_neck = True
	else:
		filter_list = [64, 64, 128, 256, 512]
		bottle_neck = False
	num_stages = 4
	if num_layers == 18:
		units = [2, 2, 2, 2]
	elif num_layers == 34:
		units = [3, 4, 6, 3]
	elif num_layers == 49:
		units = [3, 4, 14, 3]
	elif num_layers == 50:
		units = [3, 4, 14, 3]
	elif num_layers == 74:
		units = [3, 6, 24, 3]
	elif num_layers == 90:
		units = [3, 8, 30, 3]
	elif num_layers == 98:
		units = [3, 4, 38, 3]
	elif num_layers == 99:
		units = [3, 8, 35, 3]
	elif num_layers == 100:
		units = [3, 13, 30, 3]
	elif num_layers == 134:
		units = [3, 10, 50, 3]
	elif num_layers == 136:
		units = [3, 13, 48, 3]
	elif num_layers == 140:
		units = [3, 15, 48, 3]
	elif num_layers == 124:
		units = [3, 13, 40, 5]
	elif num_layers == 160:
		units = [3, 24, 49, 3]
	elif num_layers == 101:
		units = [3, 4, 23, 3]
	elif num_layers == 152:
		units = [3, 8, 36, 3]
	elif num_layers == 200:
		units = [3, 24, 36, 3]
	elif num_layers == 269:
		units = [3, 30, 48, 8]
	else:
		raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

	input_layer, body = resnet(units=units,
				 num_stages=num_stages,
				 filter_list=filter_list,
				 num_classes=num_classes,
				 bottle_neck=bottle_neck)

	model = tf.keras.models.Model(input_layer, body)
	model.summary()

	return model


if __name__ == '__main__':
	get_symbol()
