# modified from mxnet to tensorflow from https://github.com/deepinsight/insightface/blob/master/recognition/symbol/vargfacenet.py


import tensorflow as tf


def Act(data, act_type, name):
	if act_type == 'prelu':
		body = tf.keras.layers.PReLU(name=name)(data)
	else:
		body = tf.keras.layers.Activation(act_type, name=name)(data)
	return body


def get_setting_params(**kwargs):
	# bn_params
	bn_mom = kwargs.get('bn_mom', 0.9)
	bn_eps = kwargs.get('bn_eps', 2e-5)
	fix_gamma = kwargs.get('fix_gamma', False)
	use_global_stats = kwargs.get('use_global_stats', False)
	# net_setting param
	workspace = kwargs.get('workspace', 512)
	act_type = kwargs.get('act_type', 'prelu')
	use_se = kwargs.get('use_se', True)
	se_ratio = kwargs.get('se_ratio', 4)
	group_base = kwargs.get('group_base', 8)

	setting_params = {}
	setting_params['bn_mom'] = bn_mom
	setting_params['bn_eps'] = bn_eps
	setting_params['fix_gamma'] = fix_gamma
	setting_params['use_global_stats'] = use_global_stats
	setting_params['workspace'] = workspace
	setting_params['act_type'] = act_type
	setting_params['use_se'] = use_se
	setting_params['se_ratio'] = se_ratio
	setting_params['group_base'] = group_base

	return setting_params


def se_block(data, num_filter, setting_params, name):
	se_ratio = setting_params['se_ratio']
	act_type = setting_params['act_type']

	pool1 = tf.keras.layers.AveragePooling2D(name=name + '_se_pool1')(data)
	conv1 = tf.keras.layers.Conv2D(filters=int(num_filter // se_ratio),
	                               kernel_size=(1, 1),
	                               strides=1,
	                               name=name + "_se_conv1")(pool1)
	act1 = Act(data=conv1, act_type=act_type, name=name + '_se_act1')

	conv2 = tf.keras.layers.Conv2D(filters=num_filter,
	                               kernel_size=(1, 1),
	                               strides=1,
	                               name=name + "_se_conv2")(act1)
	act2 = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(conv2)
	out_data = tf.keras.layers.multiply()([data, act2])
	return out_data


def separable_conv2d(data,
                     in_channels,
                     out_channels,
                     kernel,
                     pad,
                     setting_params,
                     stride=(1, 1),
                     factor=1,
                     bias=False,
                     bn_dw_out=True,
                     act_dw_out=True,
                     bn_pw_out=True,
                     act_pw_out=True,
                     dilate=1,
                     name=None):
	bn_mom = setting_params['bn_mom']
	bn_eps = setting_params['bn_eps']
	fix_gamma = setting_params['fix_gamma']
	use_global_stats = setting_params['use_global_stats']
	workspace = setting_params['workspace']
	group_base = setting_params['group_base']
	act_type = setting_params['act_type']
	assert in_channels % group_base == 0

	# depthwise
	dw_out = tf.keras.layers.Conv2D(filters=int(in_channels * factor),
	                                kernel_size=kernel,
	                                strides=stride[0],
	                                use_bias=True if bias else False,
	                                # num_group=int(in_channels / group_base),
	                                dilation_rate=(dilate, dilate),
	                                name=name + '_conv2d_depthwise')(data)
	dw_out = tf.keras.layers.ZeroPadding2D((pad))(dw_out)
	if bn_dw_out:
		dw_out = tf.keras.layers.BatchNormalization(epsilon=bn_eps,
		                                            momentum=bn_mom,
		                                            # use_global_stats=use_global_stats,
		                                            name=name + '_conv2d_depthwise_bn')(dw_out)
	if act_dw_out:
		dw_out = Act(data=dw_out,
		             act_type=act_type,
		             name=name + '_conv2d_depthwise_act')
	# pointwise
	pw_out = tf.keras.layers.Conv2D(filters=out_channels,
	                                kernel_size=(1, 1),
	                                strides=1,
	                                # num_group=1,
	                                use_bias=True if bias else False,
	                                name=name + '_conv2d_pointwise')(dw_out)
	if bn_pw_out:
		pw_out = tf.keras.layers.BatchNormalization(epsilon=bn_eps,
		                                            momentum=bn_mom,
		                                            # use_global_stats=use_global_stats,
		                                            name=name + '_conv2d_pointwise_bn')(pw_out)
	if act_pw_out:
		pw_out = Act(data=pw_out,
		             act_type=act_type,
		             name=name + '_conv2d_pointwise_act')
	return pw_out


def vargnet_block(data,
                  n_out_ch1,
                  n_out_ch2,
                  n_out_ch3,
                  setting_params,
                  factor=2,
                  dim_match=True,
                  multiplier=1,
                  kernel=(3, 3),
                  stride=(1, 1),
                  dilate=1,
                  with_dilate=False,
                  name=None):
	use_se = setting_params['use_se']
	act_type = setting_params['act_type']

	out_channels_1 = int(n_out_ch1 * multiplier)
	out_channels_2 = int(n_out_ch2 * multiplier)
	out_channels_3 = int(n_out_ch3 * multiplier)

	pad = (((kernel[0] - 1) * dilate + 1) // 2,
	       ((kernel[1] - 1) * dilate + 1) // 2)

	if with_dilate:
		stride = (1, 1)
	if dim_match:
		short_cut = data
	else:
		short_cut = separable_conv2d(data=data,
		                             in_channels=out_channels_1,
		                             out_channels=out_channels_3,
		                             kernel=kernel,
		                             pad=pad,
		                             setting_params=setting_params,
		                             stride=stride,
		                             factor=factor,
		                             bias=False,
		                             act_pw_out=False,
		                             dilate=dilate,
		                             name=name + '_shortcut')
	sep1_data = separable_conv2d(data=data,
	                             in_channels=out_channels_1,
	                             out_channels=out_channels_2,
	                             kernel=kernel,
	                             pad=pad,
	                             setting_params=setting_params,
	                             stride=stride,
	                             factor=factor,
	                             bias=False,
	                             dilate=dilate,
	                             name=name + '_sep1_data')
	sep2_data = separable_conv2d(data=sep1_data,
	                             in_channels=out_channels_2,
	                             out_channels=out_channels_3,
	                             kernel=kernel,
	                             pad=pad,
	                             setting_params=setting_params,
	                             stride=(1, 1),
	                             factor=factor,
	                             bias=False,
	                             dilate=dilate,
	                             act_pw_out=False,
	                             name=name + '_sep2_data')

	if use_se:
		sep2_data = se_block(data=sep2_data,
		                     num_filter=out_channels_3,
		                     setting_params=setting_params,
		                     name=name)

	out_data = sep2_data + short_cut
	out_data = Act(data=out_data, act_type=act_type, name=name + '_out_data_act')
	return out_data


def vargnet_branch_merge_block(data,
                               n_out_ch1,
                               n_out_ch2,
                               n_out_ch3,
                               setting_params,
                               factor=2,
                               dim_match=False,
                               multiplier=1,
                               kernel=(3, 3),
                               stride=(2, 2),
                               dilate=1,
                               with_dilate=False,
                               name=None):
	act_type = setting_params['act_type']

	out_channels_1 = int(n_out_ch1 * multiplier)
	out_channels_2 = int(n_out_ch2 * multiplier)
	out_channels_3 = int(n_out_ch3 * multiplier)

	pad = (((kernel[0] - 1) * dilate + 1) // 2,
	       ((kernel[1] - 1) * dilate + 1) // 2)

	if with_dilate:
		stride = (1, 1)
	if dim_match:
		short_cut = data
	else:
		short_cut = separable_conv2d(data=data,
		                             in_channels=out_channels_1,
		                             out_channels=out_channels_3,
		                             kernel=kernel,
		                             pad=pad,
		                             setting_params=setting_params,
		                             stride=stride,
		                             factor=factor,
		                             bias=False,
		                             act_pw_out=False,
		                             dilate=dilate,
		                             name=name + '_shortcut')
	sep1_data_brach1 = separable_conv2d(data=data,
	                                    in_channels=out_channels_1,
	                                    out_channels=out_channels_2,
	                                    kernel=kernel,
	                                    pad=pad,
	                                    setting_params=setting_params,
	                                    stride=stride,
	                                    factor=factor,
	                                    bias=False,
	                                    dilate=dilate,
	                                    act_pw_out=False,
	                                    name=name + '_sep1_data_branch')
	sep1_data_brach2 = separable_conv2d(data=data,
	                                    in_channels=out_channels_1,
	                                    out_channels=out_channels_2,
	                                    kernel=kernel,
	                                    pad=pad,
	                                    setting_params=setting_params,
	                                    stride=stride,
	                                    factor=factor,
	                                    bias=False,
	                                    dilate=dilate,
	                                    act_pw_out=False,
	                                    name=name + '_sep2_data_branch')
	sep1_data = sep1_data_brach1 + sep1_data_brach2
	sep1_data = Act(data=sep1_data, act_type=act_type, name=name + '_sep1_data_act')
	sep2_data = separable_conv2d(data=sep1_data,
	                             in_channels=out_channels_2,
	                             out_channels=out_channels_3,
	                             kernel=kernel,
	                             pad=pad,
	                             setting_params=setting_params,
	                             stride=(1, 1),
	                             factor=factor,
	                             bias=False,
	                             dilate=dilate,
	                             act_pw_out=False,
	                             name=name + '_sep2_data')
	out_data = sep2_data + short_cut
	out_data = Act(data=out_data, act_type=act_type, name=name + '_out_data_act')
	return out_data


def add_vargnet_conv_block(data,
                           stage,
                           units,
                           in_channels,
                           out_channels,
                           setting_params,
                           kernel=(3, 3),
                           stride=(2, 2),
                           multiplier=1,
                           factor=2,
                           dilate=1,
                           with_dilate=False,
                           name=None):
	assert stage >= 2, 'stage is {}, stage must be set >=2'.format(stage)
	data = vargnet_branch_merge_block(data=data,
	                                  n_out_ch1=in_channels,
	                                  n_out_ch2=out_channels,
	                                  n_out_ch3=out_channels,
	                                  setting_params=setting_params,
	                                  factor=factor,
	                                  dim_match=False,
	                                  multiplier=multiplier,
	                                  kernel=kernel,
	                                  stride=stride,
	                                  dilate=dilate,
	                                  with_dilate=with_dilate,
	                                  name=name + '_stage_{}_unit_1'.format(stage))
	for i in range(units - 1):
		data = vargnet_block(data=data,
		                     n_out_ch1=out_channels,
		                     n_out_ch2=out_channels,
		                     n_out_ch3=out_channels,
		                     setting_params=setting_params,
		                     factor=factor,
		                     dim_match=True,
		                     multiplier=multiplier,
		                     kernel=kernel,
		                     stride=(1, 1),
		                     dilate=dilate,
		                     with_dilate=with_dilate,
		                     name=name + '_stage_{}_unit_{}'.format(stage, i + 2))
	return data


def add_head_block(data,
                   num_filter,
                   setting_params,
                   multiplier,
                   head_pooling=False,
                   kernel=(3, 3),
                   stride=(2, 2),
                   pad=(1, 1),
                   name=None):
	bn_mom = setting_params['bn_mom']
	bn_eps = setting_params['bn_eps']
	fix_gamma = setting_params['fix_gamma']
	use_global_stats = setting_params['use_global_stats']
	workspace = setting_params['workspace']
	act_type = setting_params['act_type']
	channels = int(num_filter * multiplier)

	conv1 = tf.keras.layers.Conv2D(filters=channels,
	                               kernel_size=kernel,
	                               strides=stride[0],
	                               use_bias=False,
	                               # num_group=1,
	                               name=name + '_conv1')(data)
	conv1 = tf.keras.layers.ZeroPadding2D(pad)(conv1)
	bn1 = tf.keras.layers.BatchNormalization(epsilon=bn_eps,
	                                         momentum=bn_mom,
	                                         # use_global_stats=use_global_stats,
	                                         name=name + '_conv1_bn')(conv1)

	act1 = Act(data=bn1, act_type=act_type, name=name + '_conv1_act')

	if head_pooling:
		head_data = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, name=name + '_max_pooling')(
			act1)
		head_data = tf.keras.layers.ZeroPadding2D((1, 1))(head_data)
	else:
		head_data = vargnet_block(data=act1,
		                          n_out_ch1=num_filter,
		                          n_out_ch2=num_filter,
		                          n_out_ch3=num_filter,
		                          setting_params=setting_params,
		                          factor=1,
		                          dim_match=False,
		                          multiplier=multiplier,
		                          kernel=kernel,
		                          stride=(2, 2),
		                          dilate=1,
		                          with_dilate=False,
		                          name=name + '_head_pooling')
	return head_data


def add_emb_block(data,
                  input_channels,
                  last_channels,
                  emb_size,
                  fc_type,
                  setting_params,
                  bias=False,
                  name=None):
	bn_mom = setting_params['bn_mom']
	bn_eps = setting_params['bn_eps']
	fix_gamma = setting_params['fix_gamma']
	use_global_stats = setting_params['use_global_stats']
	workspace = setting_params['workspace']
	act_type = setting_params['act_type']
	group_base = setting_params['group_base']
	# last channels
	if input_channels != last_channels:
		data = tf.keras.layers.Conv2D(filters=last_channels,
		                              kernel_size=(1, 1),
		                              strides=1,
		                              use_bias=True if bias else False,
		                              name=name + '_convx')(data)
		data = tf.keras.layers.BatchNormalization(epsilon=bn_eps,
		                                          momentum=bn_mom,
		                                          # use_global_stats=use_global_stats,
		                                          name=name + '_convx_bn')(data)
		data = Act(data=data, act_type=act_type, name=name + '_convx_act')
	# depthwise
	convx_depthwise = tf.keras.layers.Conv2D(filters=last_channels,
	                                         kernel_size=(7, 7),
	                                         strides=1,
	                                         # num_group=int(last_channels / group_base),
	                                         use_bias=True if bias else False,
	                                         name=name + '_convx_depthwise')(data)
	convx_depthwise = tf.keras.layers.BatchNormalization(epsilon=bn_eps,
	                                                     momentum=bn_mom,
	                                                     # use_global_stats=use_global_stats,
	                                                     name=name + '_convx_depthwise_bn')(convx_depthwise)
	# pointwise
	convx_pointwise = tf.keras.layers.Conv2D(filters=int(last_channels // 2),
	                                         kernel_size=(1, 1),
	                                         strides=1,
	                                         use_bias=True if bias else False,
	                                         name=name + '_convx_pointwise')(convx_depthwise)
	convx_pointwise = tf.keras.layers.BatchNormalization(epsilon=bn_eps,
	                                                     momentum=bn_mom,
	                                                     # use_global_stats=use_global_stats,
	                                                     name=name + '_convx_pointwise_bn')(convx_pointwise)
	convx_pointwise = Act(data=convx_pointwise,
	                      act_type=act_type,
	                      name=name + '_convx_pointwise_act')

	return convx_pointwise


def get_symbol():
	multiplier = 1.25
	emb_size = 512
	fc_type = "j"

	kwargs = {'use_se': 0,
	          'act_type': "prelu",
	          'bn_mom': 0.9,
	          'workspace': None,
	          }

	setting_params = get_setting_params(**kwargs)

	factor = 2
	head_pooling = False
	num_stage = 3
	stage_list = [2, 3, 4]
	units = [3, 7, 4]
	filter_list = [32, 64, 128, 256]
	last_channels = 1024
	dilate_list = [1, 1, 1]
	with_dilate_list = [False, False, False]

	data = tf.keras.layers.Input((112, 112, 3))

	body = add_head_block(data=data,
	                      num_filter=filter_list[0],
	                      setting_params=setting_params,
	                      multiplier=multiplier,
	                      head_pooling=head_pooling,
	                      kernel=(3, 3),
	                      stride=(1, 1),
	                      pad=(1, 1),
	                      name="vargface_head")

	for i in range(num_stage):
		body = add_vargnet_conv_block(data=body,
		                              stage=stage_list[i],
		                              units=units[i],
		                              in_channels=filter_list[i],
		                              out_channels=filter_list[i + 1],
		                              setting_params=setting_params,
		                              kernel=(3, 3),
		                              stride=(2, 2),
		                              multiplier=multiplier,
		                              factor=factor,
		                              dilate=dilate_list[i],
		                              with_dilate=with_dilate_list[i],
		                              name="vargface")
	emb_feat = add_emb_block(data=body,
	                         input_channels=filter_list[3],
	                         last_channels=last_channels,
	                         emb_size=emb_size,
	                         fc_type=fc_type,
	                         setting_params=setting_params,
	                         bias=False,
	                         name='embed')

	model = tf.keras.models.Model(data, emb_feat)
	model.summary()

	return emb_feat


if __name__ == '__main__':
	get_symbol()
