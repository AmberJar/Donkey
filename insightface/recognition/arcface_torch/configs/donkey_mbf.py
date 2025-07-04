from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "mbf"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.interclass_filtering_threshold = 0
config.fp16 = True
config.weight_decay = 1e-3
config.batch_size = 128
config.optimizer = "adamw"
config.lr = 1e-4
config.verbose = 2000
config.dali = False

config.rec = "/scratch/pf2m24/projects/donkey_place/insightface/recognition/arcface_torch"
config.num_classes = 131
config.num_image = 1950
config.num_epoch = 220
config.warmup_epoch = 0
config.val_targets = []
