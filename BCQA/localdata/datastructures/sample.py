from typing import List

from typing import Optional

from localdata.datastructures.answer import Answer, AmbigNQAnswer
from localdata.datastructures.evidence import Evidence
from localdata.datastructures.question import Question


class Sample:
    def __init__(self, idx, question: Question, answer: Answer, evidences: Optional[Evidence] = None):
        self.question = question
        self.evidences = evidences
        self.answer = answer
        self.idx = idx


class AmbigNQSample:
    def __init__(self, idx, question: Question, answers: AmbigNQAnswer, evidences: Optional[Evidence] = None):
        self.question = question
        self.evidences = evidences
        self.answer = answers
        self.idx = idx
