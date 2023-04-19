#-*- encoding:utf-8 -*-
import sys   #reload()之前必须要引入模块
reload(sys)
sys.setdefaultencoding('utf-8')

import pickle
import re

from flask import Flask, request, jsonify

import tensorflow as tf
from Batch import BatchGenerator
from bilstm_crf import Model

import codecs
import re
import numpy as np
from utils import padding
# from utils import *

app = Flask(__name__)


max_len = 60

def get_entity(x,y,id2tag):
    entity=""
    res=[]
    for i in range(len(x)): #for every sen
        for j in range(len(x[0])): #for every word
            if y[i][j]==0:
                continue
            if id2tag[y[i][j]][0]=='B':
                entity=id2tag[y[i][j]][1:]+':'+x[i][j]
            elif id2tag[y[i][j]][0]=='M' and len(entity)!=0 :
                entity+=x[i][j]
            elif id2tag[y[i][j]][0]=='E' and len(entity)!=0 :
                entity+=x[i][j]
                res.append(entity)
                entity=[]
            else:
                entity=[]
    return res
def test_input(model, sess, word2id, id2tag, batch_size, text):
    # while True:
        # text = raw_input("Enter your input: ").decode('gbk')
    # 将text以全角标点符号分割
    print 'before split-------------' + text
    text = re.split(u'[，。！？、‘’“”（）]', text)
    # 打印一下list对象 text
    print 'after split-------------'
    for item in text:
        print item


    text_id = []
    for sen in text:
        word_id = []
        for word in sen:
            if word in word2id:
                word_id.append(word2id[word])
            else:
                word_id.append(word2id["unknow"])
        text_id.append(padding(word_id))
    zero_padding = []
    zero_padding.extend([0] * max_len)
    text_id.extend([zero_padding] * (batch_size - len(text_id)))
    feed_dict = {model.input_data: text_id}
    pre = sess.run([model.viterbi_sequence], feed_dict)
    entity = get_entity(text, pre[0], id2tag)
    print 'result:'
    for i in entity:
        print i
    return entity

@app.route("/inference", methods=['POST'])
def inference():
    data = request.get_json()
    text = data['text'].encode('utf-8')
    print text
    print '--------------------------初始化参数--------------------------'
    result = '未知错误'
    # 调用模型进行推理
    with open('../data/renmindata.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
        x_valid = pickle.load(inp)
        y_valid = pickle.load(inp)
    print "train len:", len(x_train)
    print "test len:", len(x_test)
    print "word2id len", len(word2id)
    print 'Creating the data generator ...'
    data_train = BatchGenerator(x_train, y_train, shuffle=True)
    data_valid = BatchGenerator(x_valid, y_valid, shuffle=False)
    data_test = BatchGenerator(x_test, y_test, shuffle=False)
    print 'Finished creating the data generator.'

    epochs = 51
    batch_size = 32

    config = {}
    config["lr"] = 0.001
    config["embedding_dim"] = 100
    config["sen_len"] = len(x_train[0])
    config["batch_size"] = batch_size
    config["embedding_size"] = len(word2id) + 1
    config["tag_size"] = len(tag2id)
    config["pretrained"] = False

    embedding_pre = []

    print '-----------------------test-----------------------'
    print "begin to test..."
    model = Model(config,embedding_pre,dropout_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt is None:
            print 'Model not found, please train your model first'
        else:
            path = ckpt.model_checkpoint_path
            print 'loading pre-trained model from %s.....' % path
            saver.restore(sess, path)
            print "begin to test..."
            result = test_input(model,sess,word2id,id2tag,batch_size, text)

    print text
    return result

if __name__ == '__main__':
    app.run()
