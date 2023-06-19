
import fasttext.util

ft = fasttext.load_model('cc.en.300.bin')


FOLDER_NAME = 'DataSetLocaltion/'

import pickle
with open(FOLDER_NAME+'all__video_vosk_audioMap.p','rb') as fp:
    transCript = pickle.load(fp)


fastTextEmbedding ={}
for i in transCript:
    fastTextEmbedding[t] = ft.get_sentence_vector(transCript[i])



with open(FOLDER_NAME + 'all_fastTextEmbedding.p', 'wb') as fp:
    pickle.dump(fastTextEmbedding,fp)


from laserembeddings import Laser

laser = Laser()




allName =[]
allTrans = []



for i in transCript:
    allName.append(i)
    allTrans.append(transCript[i])



embeddings = laser.embed_sentences(allTrans,
    lang='en')  # lang is only used for tokenization

:


laserEmbedding ={}
for i in zip(allName, embeddings):
    laserEmbedding[i[0]] = i[1]




with open(FOLDER_NAME+'all_LaserEmbedding.p', 'wb') as fp:
    pickle.dump(laserEmbedding,fp)

