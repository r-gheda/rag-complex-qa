from constants import Dataset, Split
from localdata.loaders.AmbigQADataLoader import AmbigQADataLoader
from localdata.loaders.FinQADataLoader import FinQADataLoader
from localdata.loaders.MusiqueQaDataLoader import MusiqueQADataLoader

from localdata.loaders.OTTQADataLoader import OTTQADataLoader
from localdata.loaders.TATQADataLoader import TATQADataLoader
from localdata.loaders.WikiMultihopQADataLoader import WikiMultihopQADataLoader
from localdata.loaders.StrategyQADataLoader import StrategyQADataLoader



class DataLoaderFactory:
    def __int__(self):
        pass

    def create_dataloader(
        self,
        dataloader_name: str,
        tokenizer="bert-base-uncased",
        config_path="test_config.ini",
        split=Split.TRAIN,
        batch_size=None,
        corpus=None
    ):
        if Dataset.AMBIGQA in dataloader_name:
            loader = AmbigQADataLoader
        elif Dataset.FINQA in dataloader_name:
            loader = FinQADataLoader
        elif Dataset.WIKIMULTIHOPQA in dataloader_name:
            loader = WikiMultihopQADataLoader
        elif Dataset.MUSIQUEQA in dataloader_name:
            loader = MusiqueQADataLoader
        elif Dataset.TATQA in dataloader_name:
            loader = TATQADataLoader  
        elif Dataset.OTTQA in dataloader_name:
            loader = OTTQADataLoader
        elif Dataset.StrategyQA in dataloader_name:
            loader = StrategyQADataLoader
        else:
            raise NotImplemented(f"{dataloader_name} not implemented yet.")
        return loader(dataset=dataloader_name, config_path=config_path,
                      split=split,batch_size=batch_size,
                      tokenizer=tokenizer,
                      corpus=corpus)