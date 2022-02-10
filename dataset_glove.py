import pprint
import sys

import numpy as np
import json
import spacy
# import nltk
import tensorflow as tf
from nltk.corpus import wordnet as wn
import re
from collections import defaultdict
# nltk.download('wordnet')
import tabulate
# import tensorflow as tf
# from transformers import BertTokenizer, TFBertModel
import numpy as np
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL

import torch
# from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger('embeddings')
# logger.addHandler(logging.StreamHandler())
# logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


class LostAndFound(Exception):
    pass


class DataSet:

    def __init__(self):
        self.embeddings_dict = dict()
        pass

    def create_dataset(self, path_in: str, path_out: str):
        self.get_glove()
        self.nlp = spacy.load("en_core_web_trf")
        # self.nlp = spacy.load("en_core_web_lg")
        # config = {"model": DEFAULT_TOK2VEC_MODEL}
        # tok2vec = self.nlp.add_pipe("tok2vec")
        self.y = list()
        self.ingr_class_dict = dict()
        # Included import since having up and invoking the file externally causes segmentation error
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        with open(path_in, "r") as js:
            d_ = json.load(js)
        all_ingrs = 0
        recipes_x, recipes_label, recipes_ingr_label = list(), list(), list()
        deubgger_counter = 0
        max_recipe_length = 0
        for recipe in d_:
            deubgger_counter += 1
            doc = self.nlp(recipe["text"])
            ingrs = dict()
            for idx, attr in zip(recipe["spans"], recipe["meta"]["annos"]):
                ingrs[idx["start"]] = {"end": idx["end"], "span": attr["span"], "attr": attr["anno"]}
            ingr_counter_match = 0
            x, target_var, mem = list(), list(), list()
            last_target_var, last_token_idx = None, None
            ongoing = False
            for counter, token in enumerate(doc):
                # logger.debug(token.text)
                # logger.debug(token.pos_)
                # Get each word embedding
                try:
                    vector = self.embeddings_dict[token.lemma_]
                except KeyError:
                    print("token OOV:")
                    print(token)
                    vector = np.zeros((300,), dtype=np.float32)
                x.append(vector)
                key = ingrs.get(token.idx, None)
                if key or ongoing:
                    try:
                        assert token.text == ingrs[token.idx]["span"]
                        if mem:  # In the case that there is a false match later on
                            ingr_counter_match += 1
                            raise AssertionError
                    except (AssertionError, KeyError):
                        if not mem:
                            last_target_var = self.extract_ingr(ingrs[token.idx]["attr"])
                            last_token_idx = token.idx
                        mem.append(counter)
                        if doc[mem[0]:(mem[-1]+1)].text == ingrs[last_token_idx]["span"]:
                            target_var.append(last_target_var)
                            last_token_idx = None
                            logger.debug("Added: " + doc[mem[0]:(mem[-1]+1)].text)
                            mem = list()
                            ingr_counter_match += 1
                            all_ingrs += 1
                            ongoing = False
                            continue
                        else:
                            logger.debug("___Assertion Error___")
                            logger.debug(token.text)
                            # logger.debug(ingrs[token.idx]["span"])
                            target_var.append(last_target_var)
                            ongoing = True
                            continue

                    target_var.append(self.extract_ingr(ingrs[token.idx]["attr"]))
                    ingr_counter_match += 1
                    all_ingrs += 1
                elif token.pos_ == "VERB":
                    target_var.append("verb")
                elif token.pos_ == "NOUN":
                    # print(token.text)
                    if DataSet.find_hypernym(token, tags=["kitchen_appliance", "utensil", "device", "equipment",
                                                          "paper", "cutlery", "eating_utensil", "vessel",
                                                          "instrumentality", "dish", "pot", "hand_tool",
                                                          "white_goods"]):
                        target_var.append("device")
                        # print("is device\n")
                    else:
                        target_var.append("O")
                        # print("-\n")
                else:
                    target_var.append("O")
            if counter + 1 > max_recipe_length:
                max_recipe_length = counter + 1
            # Check if all the ingredients got matched
            assert len(ingrs) == ingr_counter_match
            # Check if the target variables is the same with the tokens
            assert len(target_var) == len(doc)
            assert len(target_var) == len(x)
            doc_pos = [t.pos_ for t in doc]
            print(tabulate.tabulate([list(doc), doc_pos, target_var], tablefmt="github"))
            # logger.debug(doc)
            # logger.debug(target_var)
            # pprint.pprint(self.ingr_class_dict)
            logger.info("___ New recipe ___")
            # Assign the embeddings and make the variables to two integer columns for array
            # recipes.append((x, DataSet.get_target_int(target_var)))
            recipes_x.append(x)
            _label, _ingr_label = DataSet.get_target_int(target_var)
            recipes_label.append(_label)
            recipes_ingr_label.append(_ingr_label)
        # Save the X
        pad = len(max(recipes_x, key=len))
        for recipe_counter, _ in enumerate(recipes_x):
            padding = np.zeros((pad - len(recipes_x[recipe_counter]), 300), dtype=np.float32)
            origin = np.array([tok for tok in recipes_x[recipe_counter]])
            recipes_x[recipe_counter] = np.concatenate((origin, padding), axis=0)
        np.save(path_out + "_x", np.array([recipe for recipe in recipes_x]))  # Keeps precision
        # To load
        # np.load(path_out + "_x.npy")

        # Save the labels, as numpy
        labs = list()
        for lab in recipes_label:
            padding = np.full(max_recipe_length - len(lab), -1, dtype=np.int)
            labs.append(np.append(np.array(lab), padding))
        np.save(path_out + "_labels", np.array(labs))
        # To load
        # np.load(path_out + "_labels.npy")

        # Save the Ingredient labels
        labs = list()
        for lab in recipes_ingr_label:
            padding = np.full(max_recipe_length - len(lab), 0, dtype=np.int)
            labs.append(np.append(np.array(lab), padding))
        np.save(path_out + "_labels_ingrs", np.array(labs))
        # To load
        np.load(path_out + "_labels_ingrs.npy")
        # sys.exit()
        pprint.pprint(self.ingr_class_dict)
        # Transform to numpy array
        print()
        # Write to file

    def recipe_iter(self):
        for X, y in zip(self.X_train, self.y_train):
            yield X, y

    def fetch_from_path(self, path: str):
        """ Provide path and generic filename

        :param path:
        :return:
        """
        self.X = np.load(path + "_x.npy")
        self.label = np.load(path + "_labels.npy")
        self.label_ingr = np.load(path + "_labels_ingrs.npy")

    def get_X(self):
        return self.X

    def get_label(self):
        return self.label

    def get_glove(self):
        self.embeddings_dict = dict()
        with open("/home/chroner/PhD_remote/RL_Event_Schema_Induction/data/external/glove.840B.300d.txt", 'r',
                  encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    vector = np.asarray(values[1:], "float32")
                    self.embeddings_dict[word] = vector
                except ValueError:
                    vector = np.asarray(values[-300:], "float32")
                    self.embeddings_dict[values[0] + " " + values[1]] = vector

    def get_label_ingr(self):
        return self.label_ingr

    def extract_ingr(self, attr: str):
        pattern = re.compile(r'(?<=\[).*?(?=\])', re.UNICODE)
        # Create orderless ingr class key
        key = frozenset(re.findall(pattern, attr))
        if key not in self.ingr_class_dict:
            # Assign new name to it
            self.ingr_class_dict[key] = {"id": "i_" + str(len(self.ingr_class_dict) + 1), "attr": attr}
        return self.ingr_class_dict[key]["id"]

    @staticmethod
    def recursive_tag(ss, tags: list):
        if ss.hypernyms():
            for hyp in ss.hypernyms():
                for lemma in hyp.lemmas():
                    if lemma.name() in tags:
                        raise LostAndFound
            DataSet.recursive_tag(hyp, tags)
        else:
            return False

    @staticmethod
    def find_hypernym(token, tags: list):
        # nltk.download('wordnet')
        exclusions = {"contents", "C", "c", "mixture", "heat", "sprinkle", "pocket", "top", "mouth", "holes",
                      "bite", "pieces", "edges", "tablespoons", "seconds", "side", "layers", "ounces", "sides",
                      "broiler", "stem", "cup", "balls", "bit", "base", "rest", "ball", "square", "triangle",
                      "liquor", "bottom", "souffle"}
        if token.lemma_ in exclusions:
            return False
        for ss in wn.synsets(token.lemma_):
            try:
                DataSet.recursive_tag(ss, tags=tags)
            except LostAndFound:
                return True
        return False

    @staticmethod
    def get_BIO(target_var):
        mem = None
        for counter, token in enumerate(target_var):
            if token == mem:
                pass

    @staticmethod
    def get_target_int(target: str):
        """ The mapping is 0 = O, 1 = verb, 2 = device, 3 = ingredient.
        The count of ingrs groups start from 1, 0 are the non ingrs for the ingr_label.
        :param target:
        :return:
        """
        label, ingr_label = list(), list()
        for lab in target:
            if lab == "O":
                label.append(0)
                ingr_label.append(0)
            elif lab == "verb":
                label.append(1)
                ingr_label.append(0)
            elif lab == "device":
                label.append(2)
                ingr_label.append(0)
            else:  # Ingredient
                label.append(3)
                ingr_label.append(int(lab.split("_")[1]))
        return label, ingr_label


if __name__ == '__main__':
    d = DataSet()
    d.create_dataset("/home/chroner/PhD_remote/RL_Event_Schema_Induction/data/raw/recipes_0_0.json",
                     "/home/chroner/PhD_remote/RL_Event_Schema_Induction/data/processed/GloVe/recipes_0_0_proc")
    # for i in d.recipe_iter():
    #     print(i)
