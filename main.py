import os
import pickle
import sys

from zoe_utils import DataReader
from zoe_utils import BertProcessor
from zoe_utils import EsaProcessor
from zoe_utils import Evaluator
from zoe_utils import InferenceProcessor
from generate_candidates import generate_main
from zoe_utils import Sentence
from Evaluate import evaluate_main

class ZoeRunner:

    """
    @allow_tensorflow sets whether the system will do run-time ELMo processing.
                      It's set to False in experiments as ELMo results are cached,
                      but please set it to default True when running on new sentences.
    """
    def __init__(self):
        self.bert_processor = BertProcessor()
        self.esa_processor = EsaProcessor()
        self.inference_processor = InferenceProcessor("ontonotes")
        self.evaluator = Evaluator()
        self.evaluated = []

    """
    Process a single sentence
    @sentence: a sentence in zoe_utils.Sentence structure
    @return: a sentence in zoe_utils that has predicted types set
    """
    def process_sentence(self, sentence, inference_processor=None):
        esa_candidates = self.esa_processor.get_candidates(sentence)
        bert_candidates = self.bert_processor.rank_candidates(sentence, esa_candidates)
        if inference_processor is None:
            inference_processor = self.inference_processor
        print("Ranking finished")
        inference_processor.inference(sentence, bert_candidates, esa_candidates)
        return sentence

    """
    Helper function to evaluate on a dataset that has multiple sentences
    @file_name: A string indicating the data file. 
                Note the format needs to be the common json format, see examples
    @mode: A string indicating the mode. This adjusts the inference mode, and set caches etc.
    @return: None
    """
    def evaluate_dataset(self, file_name, mode, do_inference=True, use_prior=True, use_context=True, size=-1):
        if not os.path.isfile(file_name):
            print("[ERROR] Invalid input data file.")
            return
        self.inference_processor = InferenceProcessor(mode, do_inference, use_prior, use_context)
        dataset = DataReader(file_name, size)
        for sentence in dataset.sentences:
            processed = self.process_sentence(sentence)
            if processed == -1:
                continue
            self.evaluated.append(processed)
            processed.print_self()
            evaluator = Evaluator()
            evaluator.print_performance(self.evaluated)

    """
    Helper function that saves the predicted sentences list to a file.
    @file_name: A string indicating the target file path. 
                Note it will override the content
    @return: None
    """
    def save(self, file_name):
        with open(file_name, "wb") as handle:
            pickle.dump(self.evaluated, handle, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def evaluate_saved_runlog(log_name):
        with open(log_name, "rb") as handle:
            sentences = pickle.load(handle)
        evaluator = Evaluator()
        evaluator.print_performance(sentences)

    def find_best_prediction_seq(self, i, choice):
        if i == -1:
            return []
        if i not in choice:
            result = self.find_best_prediction_seq(i-1, choice)
            return result
        else:
            result = self.find_best_prediction_seq(choice[i][0]-1, choice)
            result.append(choice[i])
            return result

    def dp(self, sen, cands, scores):
        f = [0] * len(sen.tokens)
        choice = {}
        f[0] = 0
        for i in range(0, len(sen.tokens)):
            # if i > 0:
            #     f[i] = f[i-1]
            for mention in cands:
                if mention[1]-1 > i:
                    break
                # if f[i] == 0:
                #     f[i] = f[mention[0]-1] + scores[mention]
                # else:
                if f[mention[0]-1] + scores[mention] > f[i]:
                    f[i] = f[mention[0]-1] + scores[mention]
                    choice[i] = mention
        final_prediction = self.find_best_prediction_seq(len(sen.tokens) -1 , choice)
        # print(choice)
        return final_prediction

    def print_prediction(self, outfile, sen, prediction, types):
        p_labels = ['O'] * len(sen.tokens)
        for mention in prediction:
            if types[mention] == 'O':
                p_labels[mention[0]] = 'O'
            else:
                p_labels[mention[0]] = 'B-' + types[mention]
            for i in range(mention[0]+1, mention[1]):
                if types[mention] == 'O':
                    p_labels[i] = 'O'
                else:
                    p_labels[i] = 'I-' + types[mention]

        gold_labels = ['O'] * len(sen.tokens)
        for mention in sen.gold:
            gold_labels[mention[0]] = 'B-'+mention[2]
            for i in range(mention[0]+1, mention[1]):
                gold_labels[i] = 'I-'+mention[2]

        for i in range(0, len(sen.tokens)):
            outfile.write(sen.tokens[i] + '\t' + p_labels[i] + '\t' + gold_labels[i] + '\n')
        outfile.write('\n')

    def post_processing(self, sen):
        surface = ' '.join(sen.tokens[sen.mention_start:sen.mention_end]).lower()

        if surface in self.inference_processor.prior_prob_map:
            prior_prob = self.inference_processor.prior_prob_map[surface]
            if float(prior_prob[1]) > 0.7:
                type = self.inference_processor.get_coarse_types_of_title(prior_prob[0])
                if len(type) == 0:
                    return sen
                # print(type)
                type = list(type)[0]
                if type == '':
                    return sen
                print(surface, prior_prob[0], type)
                sen.set_predictions(type, 1)
        return sen


def load_caching_pickles(name, isExtended):
    sens = {}
    if isExtended:
        path = 'processed_sen/' + name + '_extend'
    else:
        path = 'processed_sen/' + name
    for file_name in os.listdir(path):
        with open(path + '/' + file_name, "rb") as data:
            print(file_name)
            m = pickle.load(data)
            sens.update(m)
    return sens


def using_pickle_main():

    # name = 'CoNLL_dev'
    name = 'On'

    sentences = generate_main("CLM_output/CLM_" + name + ".out")
    processed_sentences = load_caching_pickles(name, True)
    sum_sen = max(processed_sentences.keys())

    runner = ZoeRunner()
    #
    outfile = open('result_' + name + '.out', 'w')

    sen_dict = {}

    for i in range(0, sum_sen+1):
    # for i in range(0, 1):

        sen = processed_sentences[i]
        sentences[i].generate_phrase_predictions()
        # print(sentences[i].phrase_predictions)

        cands = sen.keys()
        cands = sorted(cands, key=lambda kv: kv[1])

        types = {}
        scores = {}

        # print(sentences[i].tokens)
        sen_dict[i] = {}

        for mention in cands:
            sen_dict[i][mention] = sen[mention]

        # for mention in cands:
        for mention in sentences[i].phrase_predictions:
            # print(sen[mention].get_mention_surface(), sen[mention].predicted_types, sen[mention].confidence_score)
            if mention in sen:
                types[mention] = sen[mention].predicted_types.replace('/', '')
                scores[mention] = sen[mention].confidence_score
            else:
                cur_sen = Sentence(sentences[i].lower_tokens, mention[0], mention[1])

                processed = runner.process_sentence(cur_sen)
                print("Process Finished", processed.confidence_score)
                # processed = runner.post_processing(cur_sen)
                types[mention] = processed.predicted_types.replace('/', '')
                scores[mention] = processed.confidence_score
                sen_dict[i][(mention[0], mention[1])] = processed


        prediction = runner.dp(sentences[i], sentences[i].phrase_predictions, scores)
        # prediction = runner.dp(sen, sen.gold, scores)
        runner.print_prediction(outfile, sentences[i], prediction, types)

    evaluate_main(name)

    outfile2 = open("processed_sen/" + name + '_extend/sentences.pickle', "wb")
    pickle.dump(sen_dict, outfile2)



def whole_system_main():

    runner = ZoeRunner()

    name = 'On'

    sentences = generate_main("CLM_output/CLM_" + name + ".out")
    # sentences = generate_main("result_CoNLL.out")

    outfile = open('result_' + name + '.out', 'w')

    sen_dict = {}
    counter = 0
    cur_counter = 0
    idx = 0

    for sen in sentences:
        cands = sen.union_cands()
        cands = list(cands)
        cands = sorted(cands, key=lambda kv: kv[1])
        # print(cands)
        types = {}
        scores = {}

        if cur_counter == 1000:
            outfile2 = open("processed_sen/" + name + '/sentences' + str(idx) + ".pickle", "wb")
            pickle.dump(sen_dict, outfile2)
            sen_dict = {}
            idx += 1
            cur_counter = 0

        sen_dict[counter] = {}

        for mention in cands:
            cur_sen = Sentence(sen.lower_tokens, mention[0], mention[1])
            # for mention in sen.gold:
            #     tokens = sen.tokens[mention[0]:mention[1]]
            #     cur_sen = Sentence(sen.lower_tokens, mention[0], mention[1])

            # cur_sen = Sentence(tokens, 0, 1)

            processed = runner.process_sentence(cur_sen)
            print("Process Finished", processed.confidence_score)
            # processed = runner.post_processing(cur_sen)
            types[mention] = processed.predicted_types.replace('/', '')
            scores[mention] = processed.confidence_score
            sen_dict[counter][(mention[0], mention[1])] = processed

        counter += 1
        cur_counter += 1

        prediction = runner.dp(sen, cands, scores)
        # prediction = runner.dp(sen, sen.gold, scores)
        runner.print_prediction(outfile, sen, prediction, types)

    evaluate_main(name)

    if cur_counter != 0:
        outfile = open("processed_sen/" + name + '/sentences' + str(idx) + ".pickle", "wb")
        pickle.dump(sen_dict, outfile)
        sen_dict = {}
        idx += 1
        cur_counter = 0


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("[ERROR] choose from 'figer', 'bbn', 'ontonotes' or 'eval'")
    #     exit(0)
    # if sys.argv[1] == "figer":
    #     runner = ZoeRunner(allow_tensorflow=False)
    #     runner.elmo_processor.load_cached_embeddings("data/FIGER/target.min.embedding.pickle",
    #     "data/FIGER/wikilinks.min.embedding.pickle")
    #     runner.evaluate_dataset("data/FIGER/test_sampled.json", "figer")
    #     runner.save("data/log/runlog_figer.pickle")
    # if sys.argv[1] == "bbn":
    #     runner = ZoeRunner(allow_tensorflow=False)
    #     runner.elmo_processor.load_cached_embeddings("data/BBN/target.min.embedding.pickle",
    #     "data/BBN/wikilinks.min.embedding.pickle")
    #     runner.evaluate_dataset("data/BBN/test.json", "bbn")
    #     runner.save("data/log/runlog_bbn.pickle")
    # if sys.argv[1] == "ontonotes":
    #     runner = ZoeRunner(allow_tensorflow=False)
    #     runner.elmo_processor.load_cached_embeddings("data/ONTONOTES/target.min.embedding.pickle", "
    #     data/ONTONOTES/wikilinks.min.embedding.pickle")
    #     runner.evaluate_dataset("data/ONTONOTES/test.json", "ontonotes", size=1000)
    #     runner.save("data/log/runlog_ontonotes.pickle")
    # if sys.argv[1] == "eval" and len(sys.argv) > 2:
    #     ZoeRunner.evaluate_saved_runlog(sys.argv[2])

    using_pickle_main()
    # whole_system_main()