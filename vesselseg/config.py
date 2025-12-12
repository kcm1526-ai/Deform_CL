from detectron2.config import CfgNode as CN


def add_seg3d_config(cfg):
    # We use n_fold cross validation for evaluation.
    # Number of folds
    cfg.DATASETS.NUM_FOLDS = 5
    # IDs of test folds, every entry in [0, num_folds - 1]
    cfg.DATASETS.TEST_FOLDS = (0,)
    # CSV file for train/val/test split (format: pid,subset)
    # If specified, this overrides NUM_FOLDS and TEST_FOLDS
    cfg.DATASETS.SPLIT_CSV = ""

    # categories
    cfg.MODEL.PRED_CLASS = 0
    cfg.MODEL.N_CONTROL_POINTS = 4
    cfg.MODEL.TASK = ['cline']

    cfg.MODEL.OUT_CHANNELS = (1, 1, 1, 1)  # background included in 11
    cfg.MODEL.OUT_TASKS = ["cline"]

    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.BACKBONE.PRETRAINED = ""  # Path to pretrained weights (e.g., model_best.model)
    cfg.MODEL.BACKBONE.FREEZE_BACKBONE = False  # Freeze entire encoder for transfer learning
    cfg.MODEL.LOSS = 'dice'

    # MedNeXt backbone configuration
    cfg.MODEL.MEDNEXT_SIZE = "S"  # Model size: 'S', 'B', 'M', 'L'
    cfg.MODEL.MEDNEXT_KERNEL_SIZE = 3  # Kernel size: 3 or 5

    cfg.MODEL.UNETENCODER = CN()
    cfg.MODEL.UNETENCODER.BASE_CHANNELS = 16
    cfg.MODEL.UNETENCODER.NUM_LAYERS = 4
    cfg.MODEL.UNETENCODER.NORM = 'SyncBN'
    # New options for improved backbone
    cfg.MODEL.UNETENCODER.USE_ATTENTION_GATES = True
    cfg.MODEL.UNETENCODER.USE_SE = True
    cfg.MODEL.UNETENCODER.DEEP_SUPERVISION = True

    # config for segmentation
    cfg.MODEL.SEGMENTOR = CN()
    cfg.MODEL.SEGMENTOR.FOCAL_LOSS_GAMMA = 2.0
    cfg.MODEL.SEGMENTOR.FOCAL_LOSS_ALPHA = 0.25
    cfg.MODEL.SEGMENTOR.HEAD = False
    cfg.MODEL.SEGMENTOR.AORTA = False
    cfg.MODEL.SEGMENTOR.DIST_INPUT = False
    cfg.MODEL.SEGMENTOR.LOSS = "Diceloss"

    # Thin vessel loss configuration
    cfg.MODEL.THIN_VESSEL_LOSS = CN()
    cfg.MODEL.THIN_VESSEL_LOSS.ENABLED = False
    cfg.MODEL.THIN_VESSEL_LOSS.DICE_WEIGHT = 0.3
    cfg.MODEL.THIN_VESSEL_LOSS.CLDICE_WEIGHT = 0.3
    cfg.MODEL.THIN_VESSEL_LOSS.FOCAL_WEIGHT = 0.2
    cfg.MODEL.THIN_VESSEL_LOSS.BOUNDARY_WEIGHT = 0.1
    cfg.MODEL.THIN_VESSEL_LOSS.MULTISCALE_WEIGHT = 0.1
    cfg.MODEL.THIN_VESSEL_LOSS.CLDICE_ALPHA = 0.5
    cfg.MODEL.THIN_VESSEL_LOSS.FOCAL_GAMMA = 2.0
    cfg.MODEL.THIN_VESSEL_LOSS.DEEP_SUPERVISION_WEIGHT = 0.3

    # deform
    cfg.MODEL.DEFORM = CN()
    cfg.MODEL.DEFORM.NUM_STEPS = 4
    cfg.MODEL.DEFORM.NORM = 'SyncBN'
    cfg.MODEL.DEFORM.LOSS_EDGE_WEIGHT = 1.0
    cfg.MODEL.DEFORM.SDF_LOSS_WEIGHT = 1.0
    cfg.MODEL.DEFORM.CHAMFER_LOSS_WEIGHT = 1.0
    cfg.MODEL.DEFORM.ADPTPL = True
    cfg.MODEL.DEFORM.LOWER_THRES = 30
    cfg.MODEL.DEFORM.PTS_NUM = 350
    cfg.MODEL.DEFORM.USE_LOCAL_CHAMFER = True

    # config for input
    cfg.INPUT.CROP_SIZE_TRAIN = (128, 128, 128)

    # Data augmentation for thin structures
    cfg.INPUT.AUGMENTATION = CN()
    cfg.INPUT.AUGMENTATION.ELASTIC_DEFORM = False
    cfg.INPUT.AUGMENTATION.ELASTIC_ALPHA = 100.0
    cfg.INPUT.AUGMENTATION.ELASTIC_SIGMA = 10.0
    cfg.INPUT.AUGMENTATION.INTENSITY_SHIFT = 0.1
    cfg.INPUT.AUGMENTATION.INTENSITY_SCALE = 0.1
    cfg.INPUT.AUGMENTATION.GAMMA_RANGE = (0.8, 1.2)

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.SOLVER.CHECKPOINT_PERIOD = 2500
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.TEST.EVAL_PERIOD = 2500

    # non benchmark version of 3D convolution is very slow, so
    # we use CUDNN_BENCHMARK = True during training. Remember
    # to use fix input size to accelerate training speed.
    cfg.CUDNN_BENCHMARK = True
