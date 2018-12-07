
# coding: utf-8
import tensorflow as tf
import os
import input_preprocess
import cv2
import numpy as np
slim=tf.contrib.slim
import time

Config={
    'GPUS':'1',
    'file_dir':"/home/jlong.yuan/data/humanparsing/humanparsing.txt",
    'root_dir':'/home/jlong.yuan/data/',
    'batch_size':5,
    'crop_height':513,
    'crop_width' :513,
    'min_scale_factor':0.5,
    'max_scale_factor':2.0,
    'scale_factor_step_size':0.25,
    'ignore_label':-1,
    'classes':19,
    'weight_decay':0.0004,
    'learning_rate':0.1,
    'training_number_of_steps':30000,
    'power':0.9,
    'momentum':0.9,
    'grad_mult':1,
    'grad_mult_bias':2,
    'val_file_dir':"/home/jlong.yuan/data/humanparsing/humanparsing.txt",
    'val_batch_size':1,
    'val_crop_height':513,
    'val_crop_width':513,
    'save_path':'./output',
    'init_path':"/home/jlong.yuan/models/init_models/xception_65/model.ckpt",
    'save_summary_step':1000,
    'save_model_step':1000,
    'val_data_number':500,
}
os.environ['CUDA_VISIBLE_DEVICES']=Config['GPUS']


def loadData(File_dir, Root_dir, Dataset, Batch_size, 
            Crop_height, Crop_width, Min_scale_factor, Max_scale_factor, Scale_factor_step_size, 
            Ignore_label, Mean_pixel):
    
    f=open(File_dir,'r')
    rgb_dirs=[]
    lab_dirs=[]
    idx=1
    if 'train' in Dataset or 'val' in Dataset:
        for iline in f:
            rgb_dir,lab_dir=iline.strip('\n').split(' ')
            if os.path.exists(Root_dir+rgb_dir) and os.path.exists(Root_dir+lab_dir):
                rgb_dirs.append(Root_dir+rgb_dir)
                lab_dirs.append(Root_dir+lab_dir)
                idx=idx+1
    elif 'test' in Dataset:
        for iline in f:
            rgb_dir=iline.strip('\n')
            if os.path.exists(Root_dir+rgb_dir):
                rgb_dirs.append(Root_dir+rgb_dir)
        raise RuntimeError('have not finished error')
    else:
        raise RuntimeError('path error')
    

    
    rgb_dirs = tf.convert_to_tensor(rgb_dirs, dtype=tf.string)
    lab_dirs = tf.convert_to_tensor(lab_dirs, dtype=tf.string)
    rgb_dir_q, lab_dir_q = tf.train.slice_input_producer([rgb_dirs, lab_dirs],shuffle='train' in Dataset, capacity=16 * Batch_size)
    rgb = tf.read_file(rgb_dir_q)
    lab = tf.read_file(lab_dir_q)
    
    rgb = tf.image.decode_jpeg(rgb,channels=3)
    lab = tf.image.decode_png(lab,dtype=tf.uint8,channels=1)
    
    original_image, image, label = input_preprocess.preprocess_image_and_label(
        rgb,
        lab,
        crop_height=Crop_height,
        crop_width=Crop_width,
        min_resize_value=None,
        max_resize_value=None,
        resize_factor=None,
        min_scale_factor=Min_scale_factor,
        max_scale_factor=Max_scale_factor,
        scale_factor_step_size=Scale_factor_step_size,
        ignore_label=Ignore_label,
        is_training='train' in Dataset,
        mean_pixel=Mean_pixel)

    image = (2.0 / 255.0) * tf.to_float(image) - 1.0
    
    img_rgb,img_dep=tf.train.batch([image, label],batch_size=Batch_size, dynamic_pad=True)
    
    return img_rgb, img_dep



slim=tf.contrib.slim
epsilon=0.001
decay=0.9997
conv3x3=lambda inputs, num_outputs, stride, padding, scope:slim.myconv2d(inputs, num_outputs=num_outputs, kernel_size=[3,3], stride=stride, padding=padding, 
                                                                rate=1, scope=scope, biases_initializer=None, activation_fn=None)

conv1x1=lambda inputs, num_outputs, stride, scope:slim.myconv2d(inputs, num_outputs=num_outputs, kernel_size=1, stride=stride,
                                                                rate=1, scope=scope, biases_initializer=None, activation_fn=None)
batch_norm=lambda inputs, is_training, scope:slim.batch_norm(inputs, is_training=is_training, decay=decay, scale=True, scope=scope)

depthwise_conv2d=lambda inputs, kernel_size, stride, rate, scope:slim.myseparable_conv2d(inputs, None, kernel_size, depth_multiplier=1, stride=stride, rate=rate,
                                                                                padding='VALID', scope=scope+'_depthwise', biases_initializer=None, activation_fn=None)
pointwise_conv2d=lambda inputs, num_outputs, scope:slim.myconv2d(inputs, num_outputs, 1, scope=scope+'_pointwise', biases_initializer=None, activation_fn=None)

depthwise_conv2d_same=lambda inputs, kernel_size, stride, rate, scope:slim.myseparable_conv2d(inputs, None, kernel_size, depth_multiplier=1, stride=stride, rate=rate,
                                                                                padding='SAME', scope=scope+'_depthwise', biases_initializer=None, activation_fn=None)

def fixed_padding(inputs, kernel_size, rate=1):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
    return padded_inputs
def conv2d_same1(inputs, num_outputs, kernel_size, is_training, stride=2, padding='VALID', rate=1, scope=None):
    net = fixed_padding(inputs, kernel_size=kernel_size, rate=rate)
    net = conv3x3(net, num_outputs, stride, padding, scope)
    net = batch_norm(net, is_training, scope=scope+'/BatchNorm')
    net = tf.nn.relu(net)
    return net
def conv2d_same2(inputs, num_outputs, kernel_size, is_training, stride=1, padding='SAME', rate=1, scope=None):
    net = conv3x3(inputs, num_outputs, stride, padding, scope)
    net = batch_norm(net, is_training, scope=scope+'/BatchNorm')
    net = tf.nn.relu(net)
    return net

bags=[]


def xception_module(inputs, num_outputs_list, kernel_size, is_training, stride, rate, bf_relu, skip_connection_type, outputs_collections=None, scope=None):

    def _separable_conv(inputs, num_outputs, kernel_size, is_training, stride, rate, bf_relu=True, scope=None):
        net = inputs
        if bf_relu:
            net=tf.nn.relu(net)
        net = fixed_padding(net, kernel_size=kernel_size, rate=rate)
        net = depthwise_conv2d(net, kernel_size, stride, rate, scope)
        net = batch_norm(net, is_training, scope=scope+'_depthwise/BatchNorm')
        if not bf_relu:
            net=tf.nn.relu(net)
        net = pointwise_conv2d(net, num_outputs, scope)
        net = batch_norm(net, is_training, scope=scope+'_pointwise/BatchNorm')
        if not bf_relu:
            net=tf.nn.relu(net)

        return net

    with tf.variable_scope('xception_module') as sc:
        net = inputs
        for i in range(3):
            net=_separable_conv(net, num_outputs_list[i], kernel_size=kernel_size, is_training=is_training, stride=(stride if i==2 else 1), rate=rate, bf_relu=bf_relu, scope='separable_conv'+str(i+1))
            bags.append(net)

        if skip_connection_type == 'conv':
            shortcut = conv1x1(inputs, num_outputs_list[-1], stride, scope='shortcut')
            shortcut = batch_norm(shortcut, is_training=is_training, scope='shortcut/BatchNorm')
            outputs = net + shortcut
        elif skip_connection_type == 'sum':
            outputs = net + inputs
        elif skip_connection_type == 'none':
            outputs = net
        else:
            raise ValueError('Unsupported skip connection type.')
    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            outputs)
def scale_dimension(dim, scale):
    if isinstance(dim, tf.Tensor):
        return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0, dtype=tf.int32)
    else:
        return int((float(dim) - 1.0) * scale + 1.0)

def inference(inputs, is_training, reuse, num_classes,
             crop_height, crop_width):

    with tf.variable_scope('xception_65', reuse=reuse):

        ## entry flow
        net = conv2d_same1(inputs , 32, 3, is_training=is_training, stride=2, scope='entry_flow/conv1_1')
        net = conv2d_same2(net , 64, 3, is_training=is_training, stride=1, scope='entry_flow/conv1_2')
        # 1/2
        for i in range(1):
            with tf.variable_scope('entry_flow/block1/unit_%d'%(i+1)):
                net=xception_module(net, num_outputs_list=[128,128,128], is_training=is_training, kernel_size=3, stride=2, rate=1, bf_relu=True, skip_connection_type='conv')
        # 1/4
        for i in range(1):
            with tf.variable_scope('entry_flow/block2/unit_%d'%(i+1)):
                net=xception_module(net, num_outputs_list=[256,256,256], is_training=is_training, kernel_size=3, stride=2, rate=1, bf_relu=True, skip_connection_type='conv')

        # "xception_65/entry_flow/block2/unit_1/xception_module/separable_conv2_pointwise/BatchNorm/FusedBatchNorm
        entry_b2 = bags[-2]
        print(entry_b2)

        # 1/8
        for i in range(1):
            with tf.variable_scope('entry_flow/block3/unit_%d'%(i+1)):
                net=xception_module(net, num_outputs_list=[728,728,728], is_training=is_training, kernel_size=3, stride=2, rate=1, bf_relu=True, skip_connection_type='conv')
        
        ## middle flow
        # 1/16
        for i in range(16):
            with tf.variable_scope('middle_flow/block1/unit_%d'%(i+1)):
                net=xception_module(net, num_outputs_list=[728,728,728], is_training=is_training, kernel_size=3, stride=1, rate=1, bf_relu=True, skip_connection_type='sum')

        ## exit flow
        for i in range(1):
            with tf.variable_scope('exit_flow/block1/unit_%d'%(i+1)):
                net=xception_module(net, num_outputs_list=[728,1024,1024], is_training=is_training, kernel_size=3, stride=1, rate=1, bf_relu=True, skip_connection_type='conv')
        for i in range(1):
            with tf.variable_scope('exit_flow/block2/unit_%d'%(i+1)):
                net=xception_module(net, num_outputs_list=[1536,1536,2048], is_training=is_training, kernel_size=3, stride=1, rate=2, bf_relu=False, skip_connection_type='none')


    with tf.variable_scope('layer_extra', reuse=reuse):
        
        net1=conv1x1(net, 256, 1, scope='ASPP_0')
        net1=batch_norm(net1, is_training, scope='ASPP_0_BN')
        net1=tf.nn.relu(net1)
        
        net2=tf.reduce_mean(net, axis=[1,2], keepdims=True)
        
        print(net2)
        
        net2=conv1x1(net2, 256, 1, scope='ASPP_1')
        net2=batch_norm(net2, is_training, scope='ASPP_1_BN')
        net2=tf.nn.relu(net2)
        net2=tf.image.resize_bilinear(net2, [scale_dimension(crop_height, 1/16), scale_dimension(crop_width, 1/16)], align_corners=True)
        
        net3=depthwise_conv2d_same(net, 3, 1, 6, scope='ASPP_2')
        net3=batch_norm(net3, is_training, scope='ASPP_2_BN')
        net3=tf.nn.relu(net3)
        net3=pointwise_conv2d(net3, 256, scope='ASPP_2_p')
        net3=batch_norm(net3, is_training, scope='ASPP_2_p_BN')
        net3=tf.nn.relu(net3)
        
        net4=depthwise_conv2d_same(net, 3, 1, 9, scope='ASPP_3')
        net4=batch_norm(net4, is_training, scope='ASPP_3_BN')
        net4=tf.nn.relu(net4)
        net4=pointwise_conv2d(net4, 256, scope='ASPP_3_p')
        net4=batch_norm(net4, is_training, scope='ASPP_3_p_BN')
        net4=tf.nn.relu(net4)
        
        net5=depthwise_conv2d_same(net, 3, 1, 12, scope='ASPP_4')
        net5=batch_norm(net5, is_training, scope='ASPP_4_BN')
        net5=tf.nn.relu(net5)
        net5=pointwise_conv2d(net5, 256, scope='ASPP_4_p')
        net5=batch_norm(net5, is_training, scope='ASPP_2_4_BN')
        net5=tf.nn.relu(net5)
        
        net = tf.concat([net1, net2, net3, net4, net5], axis=3)
        
        net = conv1x1(net, 256, 1, scope='layer_extra')
        net = batch_norm(net, is_training, scope='layer_extra_bn')
        net = tf.nn.relu(net)
        if is_training:
            net = tf.nn.dropout(net ,0.9, name='layer_dropout')
        
        entry_b2 = tf.image.resize_bilinear(entry_b2, [int(crop_height/2), int(crop_width/2)], align_corners=True)
        net = tf.image.resize_bilinear(net, [int(crop_height/2), int(crop_width/2)], align_corners=True)

        c_net = tf.concat([net, entry_b2], axis=3)
        c_net = tf.reduce_mean(c_net, [1, 2], name='global_pool', keepdims=True)
        c_net = conv1x1(c_net, 256, 1, scope='layer_attention1')
        c_net = batch_norm(c_net, is_training, scope='layer_attention1_bn')
        c_net = tf.nn.relu(c_net)
        c_net = conv1x1(c_net, 256, 1, scope='layer_attention2')
        c_net = batch_norm(c_net, is_training, scope='layer_attention2_bn')
        c_net = tf.nn.relu(c_net)
        
        entry_b2 = c_net * entry_b2
        net = entry_b2 + net

        net = depthwise_conv2d(net, 3, 1, 1, scope='layer_out1')
        net = batch_norm(net, is_training, scope='layer_out1_bn')
        net = tf.nn.relu(net)
        net = pointwise_conv2d(net, 256, scope='layer_out1_p')
        net = batch_norm(net, is_training, scope='layer_out1_p_bn')
        net = tf.nn.relu(net)
        net = depthwise_conv2d(net, 3, 1, 1, scope='layer_out2')
        net = batch_norm(net, is_training, scope='layer_out2_bn')
        net = tf.nn.relu(net)
        net = pointwise_conv2d(net, 256, scope='layer_out2_p')
        net = batch_norm(net, is_training, scope='layer_out2_p_bn')
        net = tf.nn.relu(net)
        net = slim.conv2d(net, num_classes, kernel_size=1, rate=1, activation_fn=None, normalizer_fn=None,scope='logist_1x1')
        print(net)
    return net

def train(Config):

    image, label=loadData(File_dir=Config['file_dir'],
                        Root_dir=Config['root_dir'],
                        Dataset='train',
                        Batch_size=Config['batch_size'],
                        Crop_height=Config['crop_height'],
                        Crop_width=Config['crop_width'],
                        Min_scale_factor=Config['min_scale_factor'],
                        Max_scale_factor=Config['max_scale_factor'],
                        Scale_factor_step_size=Config['scale_factor_step_size'],
                        Ignore_label=Config['ignore_label'],
                        Mean_pixel=[127.5, 127.5, 127.5],
                        )



    prediction=inference(image,True,False, Config['classes'],Config['crop_height'], Config['crop_width'])
    prediction=tf.image.resize_bilinear(prediction, size=(Config['crop_height'], Config['crop_width']), align_corners=True)
    tf.summary.image('prediction',tf.to_float(tf.expand_dims(tf.argmax(tf.nn.softmax(prediction),axis=3), axis=3)))
    
    print(prediction)
    print(label)

    label     =tf.reshape(label, shape=[-1])
    not_ignore_mask=tf.to_float(tf.not_equal(label, Config['ignore_label']))
    label_one_hot=tf.one_hot(label, Config['classes'], on_value=1.0, off_value=0.0)
    print(label_one_hot)
    print(tf.reshape(prediction,[-1, Config['classes']]))
    loss_ce=tf.losses.softmax_cross_entropy(label_one_hot, 
                                        tf.reshape(prediction,[-1, Config['classes']]),
                                        weights=not_ignore_mask)

    loss_l2=Config['weight_decay'] * tf.reduce_mean([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'depthwise' not in v.name])

    loss = loss_ce+loss_l2
    tf.summary.scalar('loss_ce', loss_ce)
    tf.summary.scalar('loss_l2', loss_l2)
    tf.summary.scalar('loss',    loss)


    global_step = tf.train.get_or_create_global_step()
    learning_rate=tf.train.polynomial_decay(Config['learning_rate'],
                                            global_step,
                                            Config['training_number_of_steps'],
                                            end_learning_rate=0,
                                            power=Config['power'])
    tf.summary.scalar('learning_rate', learning_rate)

    for v in tf.trainable_variables():
        tf.summary.histogram(v.name.replace(':0',''), v)

    
    opt_encoder=tf.train.MomentumOptimizer(learning_rate, momentum=Config['momentum'])
    opt_decoder=tf.train.MomentumOptimizer(learning_rate*Config['grad_mult'], momentum=Config['momentum'])
    opt_decoder_bias=tf.train.MomentumOptimizer(learning_rate*Config['grad_mult']*Config['grad_mult_bias'], momentum=Config['momentum'])
    
    var_encoder=[v for v in tf.trainable_variables() if 'xception_65' in v.name]
    var_decoder=[v for v in tf.trainable_variables() if 'xception_65' not in v.name and 'bias' not in v.name]
    var_decoder_bias=[v for v in tf.trainable_variables() if 'xception_65' not in v.name and 'bias' in v.name]
    
    grads = tf.gradients(loss, var_encoder+var_decoder+var_decoder_bias)
    grad_encoder=grads[:len(var_encoder)]
    grad_decoder=grads[len(var_encoder):len(var_encoder)+len(var_decoder)]
    grad_decoder_bias=grads[len(var_encoder)+len(var_decoder):]
    
    ops = tf.group(
    opt_encoder.apply_gradients(zip(grad_encoder, var_encoder),  global_step=global_step),
    opt_decoder.apply_gradients(zip(grad_decoder, var_decoder)),
    opt_decoder_bias.apply_gradients(zip(grad_decoder_bias, var_decoder_bias)),
    tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    )
    
    
    with tf.control_dependencies([ops]):
        train_op = tf.identity(loss)
    
    train_summary = tf.summary.merge_all()
    
    return train_op, train_summary, global_step


def evaluation(Config, reuse):
    

    with tf.name_scope('val'):
        image, label=loadData(File_dir=Config['val_file_dir'],
                            Root_dir=Config['root_dir'],
                            Dataset='train',
                            Batch_size=Config['val_batch_size'],
                            Crop_height=Config['val_crop_height'],
                            Crop_width=Config['val_crop_width'],
                            Min_scale_factor=Config['min_scale_factor'],
                            Max_scale_factor=Config['max_scale_factor'],
                            Scale_factor_step_size=Config['scale_factor_step_size'],
                            Ignore_label=Config['ignore_label'],
                            Mean_pixel=[127.5, 127.5, 127.5],
                            )

        print(image)

        prediction=inference(image,False,reuse, Config['classes'],Config['val_crop_height'], Config['val_crop_width'])
        prediction=tf.image.resize_bilinear(prediction, size=(Config['val_crop_height'], Config['val_crop_width']), align_corners=True)
        prediction=tf.to_float(tf.argmax(tf.nn.softmax(prediction), axis=3))
        summary_pred=tf.summary.image('val_prediction',tf.to_float(tf.expand_dims(prediction, axis=3)))
        
        print(label)
        label     =tf.reshape(label, shape=[-1])
        print(label)
        print(prediction)
        prediction=tf.reshape(prediction, shape=[-1])
        print(prediction)
        not_ignore_mask=tf.to_float(tf.not_equal(label, Config['ignore_label']))
        label = tf.where(tf.equal(label, Config['ignore_label']), tf.zeros_like(label), label)
        metrics_to_values, metrics_to_updates = tf.metrics.mean_iou(prediction, label, Config['classes'], weights=not_ignore_mask)
        
        summary_mIoU=tf.summary.scalar('mIoU', metrics_to_values)
        
        summary_op = tf.summary.merge([summary_pred, summary_mIoU])
        
        return metrics_to_values, metrics_to_updates,summary_op
        

def main(Config):
    train_op, train_summary, global_step= train(Config)
    metrics_to_values, metrics_to_updates, eval_summary = evaluation(Config, True)
    
    summary_writer=tf.summary.FileWriter(Config['save_path']+'/summary', graph=tf.get_default_graph())

    session_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    sess=tf.Session(config=session_config)
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    saver=tf.train.Saver(max_to_keep=10)
    
    if tf.train.latest_checkpoint(Config['save_path']):
        print('init path: %s'%(tf.train.latest_checkpoint(Config['save_path'])))
        slim.assign_from_checkpoint_fn(tf.train.latest_checkpoint(Config['save_path']),
                                    tf.global_variables(),
                                    ignore_missing_vars=False)
    
    else:
        print('init path: %s'%(Config['init_path']))
        slim.assign_from_checkpoint_fn(Config['init_path'],
                                    tf.global_variables(),
                                    ignore_missing_vars=True)

    saver.save(sess, Config['save_path']+'/model.ckpt',global_step=0)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for i in range(Config['training_number_of_steps']):
        start_time = time.time()
        if i % Config['save_summary_step'] == 0:
            o_train_op, o_train_summary, step = sess.run([train_op, train_summary, global_step])
            summary_writer.add_summary(o_train_summary, global_step=step)
        else:
            o_train_op, step = sess.run([train_op, global_step])
        time_during = time.time() - start_time
        print('%4d =====>> loss: %.3f  ======>>train time: %.2f'%(step, o_train_op, time_during))

        start_time = time.time()
        if i % Config['save_model_step'] == 0 and i !=0:
            saver.save(sess, Config['save_path']+'/model.ckpt', global_step=step)
            sess.run(tf.initialize_local_variables())
            print('>>>>>>>>>>>>>>>>>>>begin eval<<<<<<<<<<<<<<<<<<<<<<')
            for j in range(Config['val_data_number']):
                _,_,i_eval_summary = sess.run([metrics_to_values, metrics_to_updates, eval_summary])
                if j%50 == 0:
                    print('eval num: %d'%j)

            o_metrics_to_values = sess.run(metrics_to_values)
            o_i_eval_summary = sess.run(eval_summary)

            summary_writer.add_summary(o_i_eval_summary, global_step=step)
            eval_time_during = time.time() - start_time

            print('eval =====>> eval_time: %.2f'%eval_time_during)
            print('eval =====>> eval mIoU: %.3f'%o_metrics_to_values)

    coord.request_stop()
    coord.join(threads)

if __name__=='__main__':
    main(Config)

