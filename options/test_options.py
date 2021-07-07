from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options

        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--which_epoch_S', type=str, default='latest',
                            help='which epoch to load for S set to latest to use latest cached model')

        # Dropout and Batchnorm has different behavior during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=float("inf"), help='how many test images to run')

        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))

        self.isTrain = False
        return parser
