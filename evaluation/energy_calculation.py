def model_energy(model):
    e_mac = 3.1
    e_ac = 0.1
    e_cnn, e_snn = 0, 0
    e_cnn_layer, e_snn_layer = 0, 0
    for layer in model:
        if not layer.startswith('linear') :
            (k, h, w, c_i, c_o, avg_spike) = model[layer]
            e_cnn_layer = c_i*c_o*k*k*h*w * e_mac
            e_cnn += e_cnn_layer
            e_snn_layer = (avg_spike) * (e_ac/e_mac) * e_cnn_layer  
            e_snn += e_snn_layer
            print(layer, 'neurons', c_o*h*w)
            print(layer, 'energy savings:', e_cnn_layer/e_snn_layer)

    print ('total energy savings: ', e_cnn/e_snn)
    return

vgg = {}
vgg['conv1-1'] = (3, 32, 32, 3, 64, 2)
vgg['conv1-2'] = (3, 32, 32, 64, 64, 0.9)
vgg['conv2-1'] = (3, 16, 16, 64, 128, 1.5)
vgg['conv2-2'] = (3, 16, 16, 128, 128, 0.3)
vgg['conv3-1'] = (3, 8, 8, 128, 256, 1.5)
vgg['conv3-2'] = (3, 8, 8, 256, 256, 0.3)
vgg['conv3-3'] = (3, 8, 8, 256, 256, 0.3)
vgg['conv4-1'] = (3, 4, 4, 256, 512, 5)
vgg['conv4-2'] = (3, 4, 4, 512, 512, 9)
vgg['conv4-3'] = (3, 4, 4, 512, 512, 6)
vgg['conv5-1'] = (3, 2, 2, 512, 512, 21)
vgg['conv5-2'] = (3, 2, 2, 512, 512, 16)
vgg['conv5-3'] = (3, 2, 2, 512, 512, 15)

escape_net = {}
escape_net['conv1'] = (8, 56, 100, 1, 64, 0.314)
escape_net['conv2'] = (4, 28, 50, 64, 64, 0.685)
escape_net['conv3'] = (2, 14, 25, 64, 64, 0.542)
escape_net['linear1'] = (22400, 256, 6.05)
escape_net['linear2'] = (256, 3, 11.03)

model_energy(escape_net)