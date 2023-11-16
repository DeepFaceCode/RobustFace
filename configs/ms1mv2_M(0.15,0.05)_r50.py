from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (0.5, -0.15) # m & m1
config.error_rate_close = 0.15  # close noise
config.error_rate_open = 0.05   # open noise
config.t = 0.2

config.loss = 'robustface'
# config.loss = 'arcface'
# config.loss = 'curricularface'
# config.loss = 'boundarymargin'
# config.loss = 'adaface'
# config.loss = 'robustface'

config.network = "r34"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1
config.verbose = 2000
config.dali = False

#config.rec = "/train_tmp/faces_emore"
config.rec = "E:/CV_Data/faces_emore"
config.rec_noise = "E:/CV_Data/vggface_48w_facesdata"
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]



