from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "vit_b_dp005_mask_005"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 0.3
config.fp16 = True
config.weight_decay = 0.05
config.batch_size = 384
config.optimizer = "adamw"
config.lr = 5e-4
config.verbose = 2000
config.dali = False

config.rec = "/scratch/pf2m24/projects/donkey_place/insightface/recognition/arcface_torch"
config.num_classes = 131
config.num_image = 1950
config.num_epoch = 1500
config.warmup_epoch = config.num_epoch // 10
config.val_targets = []
