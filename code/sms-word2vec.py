from __future__ import print_function
import collections
import math
import numpy as np
#import os
import random
import tensorflow as tf
#import zipfile
#from matplotlib import pylab
#from six.moves import range
#from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

from scipy import spatial
import operator


vocabulary_size = 0

#read file and return all words in file
def read_data(filename) :
	f = open(filename, 'r')
	t = f.read().splitlines()
	f.close()

	data = []
	for line in t :
		words = line.split(" ")
		for i in range(1,len(words)) :
			data.append(words[i])

	global vocabulary_size 
	vocabulary_size = len(set(data))
	
	return data

words = read_data('train.txt')
print(set(words))

#build dictionary and map word to index in dictonary
def build_dataset(words) :
	#order word by appearance
	count = []
	count.extend(collections.Counter(words).most_common(vocabulary_size))
	
	#{word : index of word in dict}
	dictionary = dict()
	for word, _ in count :
		dictionary[word] = len(dictionary)

	#dictionary's index of per word in words 
	data = list()
	for word in words :
		index = dictionary[word]
		data.append(index)

	#reverse dictionary : {index : word}
	reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))

	return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words
print("len(data) = ",len(data))
data_index = 0
def generate_batch(batch_size, num_skips, skip_window) :
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=(batch_size),dtype = np.int32)
	labels = np.ndarray(shape=(batch_size,1),dtype = np.int32)
	span = 2 * skip_window + 1 #[skip_window target skip_window]
	buffer = collections.deque(maxlen = span)

	for _ in range(span) :
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)

	for i in range(batch_size // num_skips) :
		target = skip_window
		targets_to_avoid = [skip_window]
		for j in range(num_skips) :
			while target in targets_to_avoid :
				target = random.randint(0,span -1)

			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j] = buffer[target]

		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)

	return batch, labels


batch_size = 128
embedding_size = 100 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. 
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0') :
	#tf.placholder : data
	train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
	train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

	#tf.Variable : variable like embedding vectors
	#generate matrix embedding with value in [-1,1]
	embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
	softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],stddev=1.0/math.sqrt(embedding_size)))
	softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

	embed = tf.nn.embedding_lookup(embeddings, train_dataset)
	loss = tf.reduce_mean(
		tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                               	labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

	optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  	normalized_embeddings = embeddings / norm
  	valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
  	similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


num_steps = 100001

final_embeddings = np.array([])

with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print('Initialized')
	average_loss = 0
	for step in range(num_steps):
		batch_data, batch_labels = generate_batch(batch_size, num_skips, skip_window)
		feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    	_, l = session.run([optimizer, loss], feed_dict=feed_dict)
    	average_loss += l
    	if step % 2000 == 0:
      		if step > 0:
        		average_loss = average_loss / 2000
      	print('Average loss at step %d: %f' % (step, average_loss))
      	average_loss = 0
    	
    	if step % 10000 == 0:
      		sim = similarity.eval()
      		for i in range(valid_size):
        		valid_word = reverse_dictionary[valid_examples[i]]
        		top_k = 8 # number of nearest neighbors
        		nearest = (-sim[i, :]).argsort()[1:top_k+1]
        		log = 'Nearest to %s:' % valid_word
        		for k in range(top_k):
          			close_word = reverse_dictionary[nearest[k]]
          			log = '%s %s,' % (log, close_word)
        		print(log)
	
	global final_embeddings
  	final_embeddings = normalized_embeddings.eval()
 
#print("final_embedding size",len(final_embeddings))
#print("final_embeddings[0] = ",final_embeddings[0])

#return list sms that each sms contains list word embedded vector
def convert_sms_to_vector(filename,type) :
	f = open(filename,'r')
	t = f.read().splitlines()
	f.close()
	
	list_sms_vector = []
	for line in t :
		sms_vector = []
		words = line.split(" ")
		
		if type == 'trainning' :
			sms_vector.append([words[0]])

		for i in range(1,len(words)) :
			if words[i] in dictionary :
				index_in_dict = dictionary[words[i]]
				#index_in_data = data.index(index_in_dict)
				word_vector = final_embeddings[index_in_dict].tolist()
			else :
				word_vector = [0] * embedding_size
				
			sms_vector.append(word_vector) 

		list_sms_vector.append(sms_vector)

	return list_sms_vector

#list_sms_vector = convert_sms_to_vector('train.txt','trainning')
#print("sms[0] = ",list_sms_vector[0])

def convert_all_sms_same_length(list_sms_vector) :
	maxlen = -1
	for i in range(len(list_sms_vector)) :
		if maxlen < len(list_sms_vector[i]) :
			maxlen = len(list_sms_vector[i])

	for i in range(len(list_sms_vector)) :
		if len(list_sms_vector[i]) < maxlen :
			for _ in range(len(list_sms_vector[i]),maxlen) :
				list_sms_vector[i].append([0] * embedding_size)

	return list_sms_vector

def split_set(input_set,split) :
	trainning_set = []
	test_set = []
	for i in range(len(input_set)) :
		if random.random() < split :
			trainning_set.append(input_set[i])
		else :
			test_set.append(input_set[i])

	return trainning_set, test_set

def cosine_distance(instance1, instance2) :

	distance = 0
	for i in range(len(instance1)) :
		diff = 0
		for j in range(len(instance1[i])) :
			diff += math.pow(instance1[i][j] - instance2[i][j],2)
		distance += math.sqrt(diff)/float(len(instance1[i]))

	return distance/float(len(instance1))

#get k neighbors neariest test_instance
def getNeighbors(training_set,test_instance,k) :
	distances = []
	#print("training_set[0][0]",training_set[0][0])
	for train_instance in training_set :
		dist = cosine_distance(train_instance[1:], test_instance)
		distances.append((train_instance,dist))

	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for i in range(k) :
		neighbors.append(distances[i][0])

	return neighbors

#get max label in neighbors
def getResponse(neighbors) :
	classVotes = {}
	for neighbor in neighbors :
		response = neighbor[0][0]
		if response in classVotes :
			classVotes[response] += 1
		else:
			classVotes[response] = 1

	sortedVotes = sorted(classVotes.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortedVotes[0][0]

def main() :
	split = 0.8
	k = 10
	list_sms_vector_trainning = convert_sms_to_vector('train.txt','trainning')
	#print("list_sms_vector_trainning[0] = ",list_sms_vector_trainning[0])
	list_sms_vector_trainning = convert_all_sms_same_length(list_sms_vector_trainning)

	training_set, test_set = split_set(list_sms_vector_trainning,split)
	instance_correct = 0
	for test_item in test_set :
		neighbors = getNeighbors(training_set,test_item[1:],k)
		label = getResponse(neighbors)
		if label == test_item[0][0] :
			instance_correct += 1

	accurancy = (instance_correct/float(len(test_set)))*100

	print("accurancy = ",accurancy)

main()
