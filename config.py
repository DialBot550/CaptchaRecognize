
class ModelConfig:
    '''MODEL'''
    model_name:str

    '''OUTPUT SHAPE'''
    input_lengths:int

    def __init__(self) -> None:
        self.model_name = 'CRNN_res'
        self.input_lengths = 50

class TrainConfig:
    '''HYPER PARAMETERS'''
    batch_size:int
    max_epoch:int
    initial_learning_rate:int
    gamma:int
    weight_decay:int

    '''PRETRAIN'''
    pre_train:bool

    '''DEVICE STATE'''
    num_gpu:int
    num_workers:int

    def __init__(self) -> None:
        self.batch_size = 32
        self.max_epoch = 50
        self.initial_learning_rate = 1e-3
        self.gamma = 0.1
        self.weight_decay = 5e-4
        self.pre_train = False
        self.num_gpu = 0
        self.num_workers = 0

class ProjectConfig:
    '''PATH'''
    absolute_root_path:str
    weights_save_folder:str
    image_folder:str
    train_annotation_path:str
    val_annotation_path:str

    def __init__(self) -> None:
        self.weights_save_folder = './weights'
        self.image_folder = './data/dataAll/test'
        self.train_annotation_path = './data/dataAll/test_v1.csv'
        self.val_annotation_path = './data/dataAll/test.csv'
        
class DataConfig:
    '''CHARACTERS'''
    character_set:str
    
    '''IMAGE SIZE'''
    width:int
    height:int

    def __init__(self) -> None:
        self.character_set = "-1234567890qwertyuiopasdfghjklzxcvbnm"
        self.height = 64
        self.width = 200