import json
from retriever.Contriever import Contriever
from localdata.loaders.WikiMultihopQADataLoader import WikiMultihopQADataLoader
from constants import Split
from localdata.loaders.RetrieverDataset import RetrieverDataset
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from metrics.SimilarityMatch import CosineSimilarity as CosScore
from localdata.datastructures.hyperparameters.dpr import DenseHyperParams


if __name__ == "__main__":

    config_instance = DenseHyperParams(query_encoder_path="facebook/contriever",
                                     document_encoder_path="facebook/contriever"
                                     ,batch_size=32)
   # config = config_instance.get_all_params()
    corpus_path = "wiki_musique_corpus.json"

    with open(corpus_path) as f:
        corpus = json.load(f)

    loader = RetrieverDataset("wikimultihopqa","wiki-musiqueqa-corpus","evaluation/config.ini",Split.DEV)
    queries, qrels, corpus = loader.qrels()
    #print("queries",len(queries),len(qrels),len(corpus),queries[0],qrels["0"])
    tasb_search = Contriever(config_instance)

    ## wikimultihop

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    similarity_measure = CosScore()
    response = tasb_search.retrieve(corpus[:10],queries,2,similarity_measure)
    print("indices",len(response))
    metrics = RetrievalMetrics(k_values=[1,10,100])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))