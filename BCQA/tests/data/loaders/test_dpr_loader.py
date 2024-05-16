import unittest
from constants import Split
from localdata.datastructures.dataset import DprDataset
from localdata.datastructures.sample import Sample
from localdata.loaders.DprDataLoader import DprDataLoader
from localdata.loaders.Tokenizer import Tokenizer
from localdata.datastructures.hyperparameters.dpr import DenseHyperParams


class MyTestCase(unittest.TestCase):
    def test_loader(self):
        config_instance = DenseHyperParams(query_encoder_path="facebook-dpr-question_encoder-single-nq-base",
                                     document_encoder_path="facebook-dpr-ctx_encoder-single-nq-base",
                                     ann_search="annoy_search",
                                     num_negative_samples=5)
        loader = DprDataLoader("nq-data", config_path="tests/data/loaders/test_dpr_config.ini",config=config_instance, split=Split.DEV, batch_size=10)
        assert len(loader.raw_data) == len(loader.dataset)
        self.assertTrue(isinstance(loader.dataset, DprDataset))
        self.assertTrue(isinstance(loader.tokenizer, Tokenizer))
        self.assertTrue(isinstance(loader.raw_data[0], Sample))



if __name__ == '__main__':
    unittest.main()
