
import pickle

sentences = []

class Sentence_cands:

	def __init__(self):
		self.tokens = []
		self.lower_tokens = []
		self.predictions = []
		self.phrase_predictions = []
		# self.candidates = []
		self.candidates_by_token = {}
		self.candidates_form_by_token = {}

		self.gold = []


	def generate_phrase_predictions(self):
		if len(self.predictions) == 0:
			return
		start_idx = self.predictions[0][1]
		last_idx = self.predictions[0][1]
		# print(self.predictions)
		for i in range(1, len(self.predictions)):
			# print(pred)
			# print(i)
			pred = self.predictions[i]
			idx = pred[1]
			if idx != last_idx + 1:
				self.phrase_predictions.append((start_idx, last_idx+1))
				start_idx = idx
				last_idx = idx
			else:
				last_idx = idx
		self.phrase_predictions.append((start_idx, last_idx+1))


	def generate_surface_form_by_token(self, idx):
		forms = []
		for cand in self.candidates_by_token[idx]:
			l = cand[0]
			r = cand[1]
			forms.append(' '.join(self.tokens[l:r]))
		self.candidates_form_by_token[idx] = forms


	def union_cands(self):
		union = set()
		for pred in self.candidates_by_token:
			for cand in self.candidates_by_token[pred]:
				union.add(cand)
		return union

def LoadWikiTitle():
	file = open("wiki_data/wiki_title_type.pickle", "rb")
	# file = open("wiki_title.pickle", "rb")

	data = pickle.load(file)
	return data


def ReadData(file):
	infile = open(file, "r")
	sen = Sentence_cands()
	lStart = 0
	label = 'O'
	token_id = -1
	flag = False
	for line in infile:
		line = line.replace('\n','')
		if len(line) == 0:
			if flag == False:
				if label != 'O':
					sen.gold.append((lStart, token_id+1, label))
					lStart = 0
					label = 'O'
				sentences.append(sen)
				sen = Sentence_cands()
				token_id = -1
				flag = True
			continue
		else:
			flag = False
			split_sen = line.split('\t')
			token = split_sen[0]
			pred = split_sen[1]
			if token == "'s":
				split_sen[2] = 'O'
			sen.tokens.append(token)
			sen.lower_tokens.append(token.lower())
			token_id += 1
			if split_sen[1] != "O":
				sen.predictions.append((token, token_id))
			
			if split_sen[2] == 'O' or split_sen[2].startswith('B-'):
				if label != 'O':
					sen.gold.append((lStart, token_id, label))
					lStart = 0
					label = 'O'
				if split_sen[2].startswith('B-'):
					lStart = token_id
					label = split_sen[2][2:]

					# # TODO: Just for post-processing!
					# if split_sen[1][2:] != 'O':
					# 	label = split_sen[1][2:]
					# else:
					# 	label = 'O'

# for sen in sentences:
	# 	for gold in sen.gold:
	# 		print(' '.join(sen.tokens[gold[0] : gold[1]]), gold[2])


def GenerateCandidate():

	left = [-2,-1,0]
	right = [1,2,3]

	total_cand = 0

	for sen in sentences:
		for pred in sen.predictions:
			token = pred[0]
			t_idx = pred[1]
			# print(pred)
			sen.candidates_by_token[t_idx] = []
			for l in left:
				for r in right:
					left_idx = t_idx + l
					right_idx = t_idx + r
					if left_idx < 0 or right_idx > len(sen.tokens):
						continue
					# print(t_idx)
					sen.candidates_by_token[t_idx].append((left_idx, right_idx))
					total_cand += 1
		#ToDO: merge candidates	
		for pred in sen.predictions:
			token = pred[0]
			t_idx = pred[1]
			sen.generate_surface_form_by_token(t_idx)

	print(total_cand)


def FilterCandByWiki(wiki):
	removed_counter = 0
	for sen in sentences:
		for pred in sen.predictions:
			token = pred[0]
			t_idx = pred[1]
			safe_cand = []
			filter_cand = []
			for idx, cand in enumerate(sen.candidates_form_by_token[t_idx]):
				form = cand.lower().replace(' ', '_')
				# print(form)
				if form in wiki:
					# print(form)
					safe_cand.append(sen.candidates_by_token[t_idx][idx])
				else:
					filter_cand.append(sen.candidates_by_token[t_idx][idx])

			#remove nested candidates
			# remove_cand = []
			# for idx1, cand in enumerate(safe_cand):
			# 	for idx2, cand2 in enumerate(safe_cand):
			# 		if idx1 != idx2 and cand[0] <= cand2[0] and cand[1] >= cand2[1]:
			# 			if cand[0]+1 == cand2[0] and sen.tokens[cand[0]].lower() == 'the':
			# 				continue
			# 			remove_cand.append(cand2)
			# safe_cand = [x for x in safe_cand if x not in remove_cand]
			
			second_safe_cand = []
			if safe_cand != []:
				for cand in filter_cand:
					flag = True
					for safe in safe_cand:
						if (cand[0] >= safe[0] and cand[1] <= safe[1]) or (cand[0] > safe[0] and cand[0] < safe[1]) or (cand[1] > safe[0] and cand[1] < safe[1]):
							flag = False
							removed_counter += 1
							break
					if flag:
						second_safe_cand.append(cand)
				sen.candidates_by_token[t_idx] = safe_cand + second_safe_cand
				# print(sen.candidates_by_token[t_idx])
				sen.generate_surface_form_by_token(t_idx)
				# print(sen.candidates_form_by_token[t_idx])
	print(removed_counter)


def PrintCandidates():
	outfile = open('candidates.out', 'w')
	for sen in sentences:
		for token in sen.predictions:
			# print(token[1])
			outfile.write(token[0] + " : " + str(sen.candidates_form_by_token[token[1]]) + '\n')
		# print('\n')

def CalculateCoverage():
	total = 0.0
	cover = 0.0

	miss = []
	for sen in sentences:
		for gold in sen.gold:
			total += 1
			flag = False
			for pred in sen.predictions:
				t_idx = pred[1]
				for cand in sen.candidates_by_token[t_idx]:
					if gold[0] == cand[0] and gold[1] == cand[1]:
						cover += 1
						flag = True
						break
				if flag:
					break
			if flag == False:
				miss.append(' '.join(sen.tokens[gold[0] : gold[1]]))

	print("Coverage of gold mentions: ", float(cover)/float(total))
	return miss

def generate_main(filename):
	wiki = LoadWikiTitle()
	ReadData(filename)
	GenerateCandidate()
	FilterCandByWiki(wiki)
	return sentences

def main():
	wiki = LoadWikiTitle()
	# prisnt(wiki)
	ReadData("CLM_output/CLM_CoNLL_dev.out")
	# ReadData("CLM_On.out")
	GenerateCandidate()
	before_filter = CalculateCoverage()
	FilterCandByWiki(wiki)
	after_filter = CalculateCoverage()
	missed = [x for x in after_filter if x not in before_filter]
	print(missed)
	PrintCandidates()


if __name__ == "__main__":
    main()