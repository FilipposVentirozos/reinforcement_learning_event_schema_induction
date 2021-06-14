import pprint

import numpy as np
import json
import spacy
# import nltk
from nltk.corpus import wordnet as wn
import re
from collections import defaultdict
# nltk.download('wordnet')
import tabulate
import logging

logger = logging.getLogger('embeddings')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)


class LostAndFound(Exception):
    pass


class DataSet:

    def __init__(self,
                 path: str):
        self.nlp = spacy.load("en_core_web_trf")
        self.y = list()
        self.ingr_class_dict = dict()
        with open(path, "r") as js:
            d_ = json.load(js)
        for recipe in d_:
            doc = self.nlp(recipe["text"])
            ingrs = dict()
            for idx, attr in zip(recipe["spans"], recipe["meta"]["annos"]):
                ingrs[idx["start"]] = {"end": idx["end"], "span": attr["span"], "attr": attr["anno"]}
            ingr_counter_match = 0
            target_var = list()

            mem = list()
            last_target_var, last_token_idx = None, None
            ongoing = False
            for counter, token in enumerate(doc):
                # logger.debug(token.text)
                # logger.debug(token.pos_)
                key = ingrs.get(token.idx, None)
                if key or ongoing:
                    try:
                        assert token.text == ingrs[token.idx]["span"]
                    except (AssertionError, KeyError):
                        if not mem:
                            last_target_var = self.extract_ingr(ingrs[token.idx]["attr"])
                            last_token_idx = token.idx
                        mem.append(counter)
                        # try:
                        if doc[mem[0]:(mem[-1]+1)].text == ingrs[last_token_idx]["span"]:
                            # for idx in mem:
                            #     target_var[idx] = last_target_var
                            target_var.append(last_target_var)
                            last_token_idx = None
                            logger.debug("Added: " + doc[mem[0]:(mem[-1]+1)].text)
                            mem = list()
                            ingr_counter_match += 1
                            ongoing = False
                            continue
                        else:
                            # mem.append(counter)
                            logger.debug("___Assertion Error___")
                            logger.debug(token.text)
                            # logger.debug(ingrs[token.idx]["span"])
                            target_var.append(last_target_var)
                            ongoing = True
                            continue
                        # except IndexError:
                        #     # mem.append(counter)
                        #     logger.debug("___Assertion Error___")
                        #     logger.debug(token.text)
                        #     logger.debug(ingrs[token.idx]["span"])
                        #     target_var.append(last_target_var)
                        #     ongoing = True
                        #     continue
                    # last_target_var = self.extract_ingr(ingrs[token.idx]["attr"])
                    target_var.append(self.extract_ingr(ingrs[token.idx]["attr"]))
                    ingr_counter_match += 1
                elif token.pos_ == "VERB":
                    target_var.append("verb")
                elif token.pos_ == "NOUN":
                    if self.find_hypernym(token, tag="kitchen appliance"):
                        target_var.append("device")
                    else:
                        target_var.append("O")
                else:
                    target_var.append("O")
            # Check if all the ingredients got matched
            assert len(ingrs) == ingr_counter_match
            # Check if the target variables is the same with the tokens
            assert len(target_var) == len(doc)
            doc_pos = [t.pos_ for t in doc]
            print(tabulate.tabulate([list(doc), doc_pos, target_var], tablefmt="github"))
            # logger.debug(doc)
            # logger.debug(target_var)
            pprint.pprint(self.ingr_class_dict)
            logger.info("___ New recipe ___")
        self.y_train = np.random.randint(4, size=(50, 30))
        # We hypothesize a vector of 100 length
        self.X_train = np.random.uniform(low=0, high=1, size=(50, 30, 100))
        # print(self.y_train[0, ])
        # print(self.y_train[0, ].shape)
        # print(self.X_train[0, ])
        print(self.X_train[0, ].shape)
        # for i in np.nditer(self.X_train):
        # for i in self.X_train:
        #     print(i)
        #     print(i.shape)

    def recipe_iter(self):
        for X, y in zip(self.X_train, self.y_train):
            yield X, y

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y

    def recursive_find_device(self):
        pass

    def extract_ingr(self, attr: str):
        pattern = re.compile(r'(?<=\[).*?(?=\])', re.UNICODE)
        # Create orderless ingr class key
        key = frozenset(re.findall(pattern, attr))
        if key not in self.ingr_class_dict:
            # Assign new name to it
            self.ingr_class_dict[key] = {"id": "i_" + str(len(self.ingr_class_dict) + 1), "attr": attr}
        return self.ingr_class_dict[key]["id"]

    @staticmethod
    def recursive_tag(ss, tag="kitchen appliance"):
        if ss.hypernyms():
            for hyp in ss.hypernyms():
                for lemma in hyp.lemmas():
                    if lemma.name() == tag:
                        raise LostAndFound
            DataSet.recursive_tag(hyp)
        else:
            return False

    def find_hypernym(self, token, tag="kitchen appliance"):
        # nltk.download('wordnet')
        exclusions_food = {"rack", "c", "center", "centre", "broiler", "roast", "medium"}
        if token.lemma_ in exclusions_food:
            return False
        for ss in wn.synsets(token.lemma_):
            try:
                DataSet.recursive_tag(ss, tag=tag)
            except LostAndFound:
                return True


if __name__ == '__main__':
    d = DataSet("/home/chroner/PhD_remote/RL_Event_Schema_Induction/data/raw/recipes_0.0.json")
    for i in d.recipe_iter():
        print(i)
