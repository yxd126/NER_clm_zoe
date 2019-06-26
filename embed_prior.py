import pickle
from zoe_utils import InferenceProcessor
from Evaluate import evaluate_main


class Sentence:

    def __init__(self):
        self.tokens = []
        self.predictions = []
        self.gold = []


def ReadData(file):
    sentences = []
    infile = open(file, "r")
    sen = Sentence()

    lStart = 0
    l_label = 'O'

    pStart = 0
    p_label = 'O'

    token_id = -1
    flag = False

    for line in infile:
        line = line.replace('\n', '')
        if len(line) == 0:
            if flag == False:
                if l_label != 'O':
                    sen.gold.append((lStart, token_id + 1, l_label))
                    lStart = 0
                    l_label = 'O'

                if p_label != 'O':
                    sen.predictions.append((pStart, token_id + 1, p_label))
                    pStart = 0
                    p_label = 'O'

                sentences.append(sen)
                sen = Sentence()
                token_id = -1
                flag = True
            continue
        else:
            flag = False
            split_sen = line.split('\t')
            token = split_sen[0]
            if token == "'s":
                split_sen[2] = 'O'
            sen.tokens.append(token)
            token_id += 1

            if split_sen[1] == 'O' or split_sen[1].startswith('B-'):
                if p_label != 'O':
                    sen.predictions.append((pStart, token_id, p_label))
                    pStart = 0
                    p_label = 'O'

                if split_sen[1].startswith('B-'):
                    pStart = token_id
                    p_label = split_sen[1][2:]

            if split_sen[2] == 'O' or split_sen[2].startswith('B-'):
                if l_label != 'O':
                    sen.gold.append((lStart, token_id, l_label))
                    lStart = 0
                    l_label = 'O'

                if split_sen[2].startswith('B-'):
                    lStart = token_id
                    l_label = split_sen[2][2:]

    return sentences


def print_prediction(file, sen):

    p_labels = ['O'] * len(sen.tokens)
    for mention in sen.predictions:
        if mention[2] != 'O':
            p_labels[mention[0]] = 'B-'+mention[2]
            for i in range(mention[0]+1, mention[1]):
                p_labels[i] = 'I-'+mention[2]

    gold_labels = ['O'] * len(sen.tokens)
    for mention in sen.gold:
        gold_labels[mention[0]] = 'B-'+mention[2]
        for i in range(mention[0]+1, mention[1]):
            gold_labels[i] = 'I-'+mention[2]

    for i in range(0, len(sen.tokens)):
        file.write(sen.tokens[i] + '\t' + p_labels[i] + '\t' + gold_labels[i] + '\n')
    file.write('\n')

# file_name = 'CoNLL_dev'
file_name = 'On'

freebase_file = open('data/title2freebase.pickle', 'rb')

freebase = pickle.load(freebase_file)
prediction_data = ReadData('result_' + file_name + '.out')

outfile = open('fixed_result_' + file_name + '.out', 'w')

inference_processor = InferenceProcessor("ontonotes")

prior_threshold = 0.5

for sen in prediction_data:
    for idx, prediction in enumerate(sen.predictions):
        surface = '_'.join(sen.tokens[prediction[0]:prediction[1]])
        if surface not in freebase:
            if surface[0] + surface[1:].lower() in freebase:
                surface = surface[0] + surface[1:].lower()
            else:
                if surface.upper() in freebase:
                    surface = surface.upper()
                else:
                    if surface.lower() in freebase:
                        surface = surface.lower()

        if surface.lower().replace('_', ' ') in inference_processor.prior_prob_map:

            title = inference_processor.get_prob_title(surface.replace('_', ' '))
            score = inference_processor.prior_prob_map[surface.lower().replace('_', ' ')][1]
            mapped_type = list(inference_processor.get_mapped_types_of_title(title))

            if len(mapped_type) != 0:
                if score > prior_threshold:
                    sen.predictions[idx] = (prediction[0], prediction[1], mapped_type[0].replace('/',''))
                    surface = title

        if surface in freebase:
            type = freebase[surface]

            if surface[0] + surface[1:].lower() in freebase:
                type += freebase[surface[0] + surface[1:].lower()]

            if surface.upper() in freebase:
                type += freebase[surface.upper()]

            if surface.lower() in freebase:
                type += freebase[surface.lower()]

            types = type.split(',')
            for idx2, type in enumerate(types):
                types[idx2] = type.split('.')[-1]
            types = set(types)

            coarse_types = inference_processor.get_coarse_types_of_title(surface)
            if len(coarse_types) == 1 and list(coarse_types)[0] != '\MISC':
                sen.predictions[idx] = (prediction[0], prediction[1], list(coarse_types)[0].replace('/',''))

            if 'countries' in types or 'country' in types:
                sen.predictions[idx] = (prediction[0], prediction[1], 'LOC')
                continue

            if 'citytown' in types or 'city' in types or 'state' in types:
                sen.predictions[idx] = (prediction[0], prediction[1], 'LOC')
                continue

            if 'sports_team' in types:
                sen.predictions[idx] = (prediction[0], prediction[1], 'ORG')
                continue

            if 'person' in types:
                sen.predictions[idx] = (prediction[0], prediction[1], 'PER')
                continue
    print_prediction(outfile, sen)

evaluate_main(file_name)
