import glob
import sys

presicion = {"PER" : 0.0, "LOC":0.0, "ORG":0.0, "MISC":0.0, "Entity":0.0}
recall = {"PER" : 0.0, "LOC":0.0, "ORG":0.0, "MISC":0.0, "Entity":0.0}
F1 = {"PER" : 0.0, "LOC":0.0, "ORG":0.0, "MISC":0.0, "Entity":0.0}
totalP = 0.0
totalR = 0.0
totalC = 0.0
totalF1 = 0.0
labeledPhrase = {"PER" : 0.0, "LOC":0.0, "ORG":0.0, "MISC":0.0, "Entity":0.0}
predictedPhrase = {"PER" : 0.0, "LOC":0.0, "ORG":0.0, "MISC":0.0, "Entity":0.0}
correct = {"PER" : 0.0, "LOC":0.0, "ORG":0.0, "MISC":0.0, "Entity":0.0}
token = []
predictions = []
lables = []

EntityInTrain = {"PER" : [], "ORG":[], "LOC":[], "MISC":[]}
EntityInTest = {"PER" : [], "ORG":[], "LOC":[], "MISC":[]}
EntityInGaz = []
EntityInTrainTest = {"PER" : [], "ORG":[], "LOC":[], "MISC":[]}
EntityInGazTest = {"PER" : [], "ORG":[], "LOC":[], "MISC":[]}


def RecordSeenEntities():
	# for name in glob.glob('gaz/*'):
	# 	infile=open(name, 'r')
	# 	for line in infile:
	# 		line = line.replace('\n','')
	# 		EntityInGaz.append(line)

	# for name in glob.glob('train/*'):
	for name in glob.glob(address+'tgl/column/tgl.train'):
		infile=open(name, 'r')
		newNE = ""
		currentL = "O"
		for line in infile:
			line = line.replace('\n','')
			if len(line) < 2:
				if newNE != "" and currentL != 'O':
					EntityInTrain[currentL].append(newNE)
					newNE = ""
					currentL = "O"
				continue
			splited = line.split(" ")
			if (splited[1].startswith("B-") or splited[1] == "O" ) and newNE != "" and currentL != 'O':
				EntityInTrain[currentL].append(newNE)
				newNE = ""
				currentL = "O"
			if splited[1][:2] == "B-":
				newNE = splited[0]
				currentL = splited[1][2:]
			if splited[1][:2] == "I-":
				newNE = newNE + " " + splited[0]
	# print EntityInTrain
	# for name in glob.glob('test/*'):
	for name in glob.glob(address+'tgl/column/tgl.test'):
		infile=open(name, 'r')
		newNE = ""
		currentL = "O"
		for line in infile:
			line = line.replace('\n','')
			if len(line) < 2:
				if newNE != "" and currentL != 'O':
					EntityInTest[currentL].append(newNE)
					if newNE in EntityInTrain[currentL]:
						EntityInTrainTest[currentL].append(newNE)
					if newNE in EntityInGaz:
						EntityInGazTest[currentL].append(newNE)
					newNE = ""
					currentL = "O"
				continue
			splited = line.split(" ")
			if (splited[1].startswith("B-") or splited[1] == "O" ) and newNE != "" and currentL != 'O':
				EntityInTest[currentL].append(newNE)
				if newNE in EntityInTrain[currentL]:
					EntityInTrainTest[currentL].append(newNE)
				if newNE in EntityInGaz:
					EntityInGazTest[currentL].append(newNE)
				newNE = ""
				currentL = "O"
			if splited[1][:2] == "B-":
				newNE = splited[0]
				currentL = splited[1][2:]
			if splited[1][:2] == "I-":
				newNE = newNE + " " + splited[0]



def PrintTotalF1():
	totalC = correct["PER"] + correct["LOC"] + correct["ORG"] + correct["MISC"]
	totalP = predictedPhrase["PER"] + predictedPhrase["LOC"] + predictedPhrase["ORG"] + predictedPhrase["MISC"]
	totalR = labeledPhrase["PER"] + labeledPhrase["LOC"] + labeledPhrase["ORG"] + labeledPhrase["MISC"]

	presicion = totalC / totalP
	recall = totalC / totalR
	F1 = 2*presicion*recall / (presicion + recall)
	print("Overall")
	print(presicion, recall, F1)


def PrintLabelF1 (label):
	presicion = correct[label] / predictedPhrase[label]
	recall = correct[label] / labeledPhrase[label]
	if presicion + recall != 0:
		F1 = 2*presicion*recall / (presicion + recall)
	else:
		F1 = 0
	print(label)
	print(presicion, recall, F1)

def ReadFile(lines):
	fileLen = len(lines)
	for i in range(0, fileLen):
		if len(lines[i]) < 2:
			token.append(" ")
			predictions.append(" ")
			lables.append(" ")
			continue
		splited = lines[i].replace("\n","").split("\t")
		token.append(splited[0])
		predictions.append(splited[1])
		lables.append(splited[2])

def CalculateF1(lines):
	fileLen = len(lines)
	for i in range(0, fileLen):
		p = "O"
		l = "O"
		pEnd = -1
		lEnd = -1
		if predictions[i][:2] == "B-" or (predictions[i][:2] == "I-" and (i == 0 or predictions[i-1] == " " or predictions[i-1][2:] != predictions[i][2:])):
			p = predictions[i][2:]
			pEnd = i;
			while (pEnd + 1 < fileLen and predictions[pEnd + 1] == "I-" + p):
				pEnd += 1;
		if lables[i][:2] == "B-":
			l = lables[i][2:]
			# if l == 'MISC':
			# 	l = 'LOC'
			lEnd = i;
			while (lEnd + 1 < fileLen and lables[lEnd + 1] == "I-" + l):
				lEnd += 1;
		if (p != "O" or l != "O"):
			if pEnd == lEnd and p == l:
				correct[p] += 1.0
				predictedPhrase[p] += 1.0
				labeledPhrase[l] += 1.0
			else:
				if l != "O":
					labeledPhrase[l] += 1.0
					# print token[i]
				if p != "O":
					predictedPhrase[p] += 1.0
	# print correct
	PrintLabelF1("PER")
	PrintLabelF1("ORG")
	PrintLabelF1("LOC")
	# PrintLabelF1("MISC")
	# PrintLabelF1("Entity")
	PrintTotalF1()

def CalculateUnseenF1(lines):
	fileLen = len(lines)
	numOfseen = {"PER":0, "LOC":0, "ORG":0, "MISC":0}
	numOfunseen = {"PER":0, "LOC":0, "ORG":0, "MISC":0}
	i = 0
	while i < fileLen:
		p = "O"
		l = "O"
		NE = ""
		pEnd = -1
		lEnd = -1

		if lables[i][:2] == "B-":
			l = lables[i][2:]
			NE = token[i]
			lEnd = i;
			while (lEnd + 1 < fileLen and lables[lEnd + 1] == "I-" + l):
				NE += " " + token[lEnd + 1]
				lEnd += 1
			if NE in EntityInTrainTest[l]:
			# if NE in EntityInTrainTest[l] or NE in EntityInGazTest[l]:
			# if NE not in EntityInTrainTest[l] and NE not in EntityInGazTest[l]:
				if predictions[i][:2] == "I-":
					predictedPhrase[predictions[i][2:]] -= 1.0
				i = lEnd+1
				numOfseen[l] += 1
				continue
			else:
				numOfunseen[l] += 1

		if predictions[i][:2] == "B-" or (predictions[i][:2] == "I-" and (i == 0 or predictions[i-1] == " " or predictions[i-1][2:] != predictions[i][2:])):
			p = predictions[i][2:]
			pEnd = i;
			while (pEnd + 1 < fileLen and predictions[pEnd + 1] == "I-" + p):
				pEnd += 1;
		if lables[i][:2] == "B-":
			l = lables[i][2:]
			lEnd = i;
			while (lEnd + 1 < fileLen and lables[lEnd + 1] == "I-" + l):
				lEnd += 1;
		if (p != "O" or l != "O"):
			if pEnd == lEnd and p == l:
				correct[p] += 1.0
				predictedPhrase[p] += 1.0
				labeledPhrase[l] += 1.0
			else:
				if l != "O":
					labeledPhrase[l] += 1.0
				if p != "O":
					predictedPhrase[p] += 1.0
		i += 1
	PrintLabelF1("PER")
	PrintLabelF1("ORG")
	PrintLabelF1("LOC")
	PrintLabelF1("MISC")
	PrintTotalF1()
	# print correct
	# print numOfseen["PER"]+numOfseen["ORG"]+numOfseen["LOC"]+numOfseen["MISC"]
	# print numOfunseen["PER"]+numOfunseen["ORG"]+numOfunseen["LOC"]+numOfunseen["MISC"]
	# print len(EntityInGazTest["PER"])+len(EntityInGazTest["ORG"])+len(EntityInGazTest["LOC"])+len(EntityInGazTest["MISC"])
	# print len(EntityInTrainTest["PER"])+len(EntityInTrainTest["ORG"])+len(EntityInTrainTest["LOC"])+len(EntityInTrainTest["MISC"])

def CalculateCoverage(lines):
	fileLen = len(lines)
	labeled = 0.0
	predicted = 0.0
	covered = 0.0
	for i in range(0, fileLen):
		p = "O"
		l = "O"
		pEnd = -1
		lEnd = -1

		if lables[i][:2] == "B-":
			labeled += 1
			l = lables[i][2:]
			# if l == 'MISC':
			# 	l = 'LOC'
			lEnd = i;
			flag = 0
			while (lEnd + 1 < fileLen):
				# if predictions[lEnd] != 'O':
				# 	predicted += 1
				# 	break
				if predictions[lEnd] == 'O':
					flag = 1
					break
				if lables[lEnd + 1] == "I-" + l:
					lEnd += 1;
				else:
					# tmp_s = ''
					# for j in range(i, lEnd+1):
					# 	tmp_s += token[j] + ' '
					# print tmp_s
					break
			if flag == 0:
				covered += 1
	# print labeled, predicted, predicted/labeled
	print(labeled, covered, covered/labeled)
	# PrintLabelF1("PER")
	# PrintLabelF1("ORG")
	# PrintLabelF1("LOC")
	# PrintLabelF1("MISC")
	# PrintLabelF1("Entity")
	# PrintTotalF1()


def evaluate_main(name):
	infile = open('result_'+ name +'.out', 'r')

	lines = []
	for line in infile:
		# line = line.decode('utf-8')
		if line.count("\t") == 0:
			continue
		if 'MISC' in line:
			line = line.replace('PER', 'MISC').replace('ORG', 'MISC').replace('LOC', 'MISC')
			line = line.replace('B-MISC', 'O').replace('I-MISC', 'O')
		lines.append(line)

	ReadFile(lines)
	# RecordSeenEntities()
	CalculateF1(lines)
	# CalculateUnseenF1(lines)
	# CalculateCoverage(lines)


if __name__ == "__main__":

	file = sys.argv[1]

	infile = open(file, 'r')

	lines = []
	for line in infile:
		# line = line.decode('utf-8')
		if line.count("\t") == 0:
			continue
		# line = line.replace('PER','Entity').replace('ORG','Entity').replace('LOC','Entity')
		# replace('MISC','Entity').replace('I-','B-')
		if 'MISC' in line:
			line = line.replace('PER', 'MISC').replace('ORG', 'MISC').replace('LOC', 'MISC')
			line = line.replace('B-MISC', 'O').replace('I-MISC', 'O')
		lines.append(line)

	ReadFile(lines)
	# RecordSeenEntities()
	CalculateF1(lines)


# print len(EntityInTrain["PER"]), len(EntityInTrain["ORG"]), len(EntityInTrain["LOC"]), len(EntityInTrain["MISC"])
# print len(EntityInTest["PER"]), len(EntityInTest["ORG"]), len(EntityInTest["LOC"]), len(EntityInTest["MISC"])


# print len(EntityInTrainTest["PER"]), len(EntityInTrainTest["ORG"]), len(EntityInTrainTest["LOC"]), len(EntityInTrainTest["MISC"])
# print len(EntityInGazTest["PER"]), len(EntityInGazTest["ORG"]), len(EntityInGazTest["LOC"]), len(EntityInGazTest["MISC"])

# print predictedPhrase
# print labeledPhrase



