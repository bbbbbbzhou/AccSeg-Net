import importlib
from models.base_model import BaseModel


def create_model(opt):
    model = None
    print(opt.model)

    if opt.model == 'cut2seg_model_train':
        assert(opt.dataset_mode == 'cut2seg_train')
        from .cut2seg_model import CUT2SEGModel_TRAIN
        model = CUT2SEGModel_TRAIN(opt)
    elif opt.model == 'cut2seg_model_test':
        assert(opt.dataset_mode == 'cut2seg_test')
        from .cut2seg_model import CUT2SEGModel_TEST
        model = CUT2SEGModel_TEST(opt)

    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    model.__init__(opt)
    return model
