import numpy as np
import time, sys, os, csv
import argparse
import functions, read_data
from sklearn.metrics.pairwise import cosine_similarity
import write_clusters as wc

def FLAGS(args=sys.argv[1:]):
	parser = argparse.ArgumentParser()
	parser.add_argument("-alpha", default = 0, type = float, help = "Fractional Cluster Presence.")
	parser.add_argument("-delta", default = 0, type = float, help = "Count-Min Sketch.")
	parser.add_argument("-epsilon", default = 0, type = float, help = "Count-Min Sketch.")
	parser.add_argument("-num_clusters", default = 20, type = int, help = "Number of Clusters.")
	parser.add_argument("-Lambda", default = 0.2, type = float, help = "Combining both sims")
	parser.add_argument("-path_words", default = "/home/srinivas/Downloads/PROJECT_EMAIL/DATA/test_email_data.csv", type = str, help = "--")
	parser.add_argument("-path_nodes", default = "/home/srinivas/Downloads/PROJECT_EMAIL/DATA/test_email_user_data.csv", type = str, help = "--")
	parser.add_argument("-cluster_tweet_ids", default = "/home/sai/Documents/DATA MINING/PROJECT_EMAIL/DATA/CLUSTERS/cluster", type = str, help = "--")
	parser.add_argument("-cluster_summary", default = "/home/sai/Documents/DATA MINING/PROJECT_EMAIL/DATA/CLUSTERS_SUMMARY/cluster", type = str, help = "--")
	flags = parser.parse_args(args)
	return flags

class Event_Detection:

	def __init__(self, flags):
		'''
		Initialise all parameters
		'''
		self.flags = flags
		self.reader_words = read_data.read_words(self.flags.path_words)
		self.reader_nodes = read_data.read_nodes(self.flags.path_nodes)
		self.curr_num_clusters = 0
		self.tweet_counter = 1

		self.mean = 0.0
		self.standard_deviation = 0.0
		self.moments = np.zeros(3)
		# self.moments[0] = total count, self.moments[1] = total sum, self.moments[2] = square sum
		self.lru_list = []
		self.zero_idf_words = []

		self.initialize_clusters()
		return

	def initialize_clusters(self):
		'''
		Initialise clusters
		'''
		self.clusters = [functions.cluster() for _ in range(self.flags.num_clusters)]
		return

	def update_moments(self, similarity):
		self.moments[0] = self.moments[0] + 1
		self.moments[1] = self.moments[1] + similarity
		self.moments[2] = self.moments[2] + similarity*similarity
		self.mean = self.moments[1]/self.moments[0]
		variance = (self.moments[2]/self.moments[0]) - self.mean**2
		self.standard_deviation = np.sqrt(variance)
		return

	def idf(self, word = None) :
		count = 0
		tf_count = np.zeros(self.curr_num_clusters)
		flag = np.zeros(self.curr_num_clusters)
		for index,cluster in enumerate(self.clusters):
			if word in cluster.words:
				## Assuming cluster word frequency is a dict
				tf_count[index] = cluster.word_frequencies[word] #check type of word
				count = count + 1
				flag[index] = 1

		idf = np.log((self.curr_num_clusters + 1.0)/(count + 1))
		return idf, tf_count, flag

	def content_similarity(self, stream):
		self.zero_idf_words = []
		if self.curr_num_clusters == 0 :
			return
		else:
			similarities = []
			tf_idf_clusters = []
			idf_of_words = []
			for i in range(len(stream['words'])) :				
				idf, vector, x = self.idf(stream['words'][i])
				if idf == 0 :
					self.zero_idf_words.append(stream['words'][i])
				stream['tf_idf'].append(int(stream['word_frequencies'][i])*idf)
				tf_idf_clusters.append(vector)
				idf_of_words.append(idf)
			# print(self.zero_idf_words)

			for i in range(len(tf_idf_clusters)):
				tf_idf_clusters[i] = np.multiply(tf_idf_clusters[i], idf_of_words[i])

			tf_idf_clusters = np.array(tf_idf_clusters).T

			for i in range(self.curr_num_clusters) :
				den = np.linalg.norm(np.array(stream['tf_idf']))*np.linalg.norm(tf_idf_clusters[i])
				if den != 0:
					similarities.append( np.matmul( np.array(stream['tf_idf']), tf_idf_clusters[i].T)/den)
				else:
					similarities.append(0)
			return similarities

	def structural_similarity(self, stream):
		if self.curr_num_clusters == 0 :
			return
		else:
			similarities = []
			for i in range(self.curr_num_clusters):
				cluster = self.clusters[i]
				# B = np.zeros(len(cluster.nodes)) # Optimizable :)
				# cluster_node_frequencies = []
				cluster_cms = cluster.count_min_sketch()
				# sigma_cluster_node_frequencies = cluster_cms.get_total_frequency()
				# for i,node in enumerate(cluster.nodes):
				# 	cluster_node_frequencies.append(cluster_cms.estimate(node))
				# 	if node in stream['nodes']:
				# 		B[i] = 1
				bit_vector_frequencies =  0
				for node in stream['nodes'] :
					bit_vector_frequencies += cluster_cms.estimate(node)
				# normal fre and count min fre
				# print(np.sum(cluster.node_frequencies), cluster_cms.get_total_frequency())
				# print(cluster.node_frequencies, cluster_node_frequencies)
				sim = (bit_vector_frequencies)/(np.sqrt(len(stream['nodes'])+1)*(cluster_cms.get_total_frequency()))
				# print(cluster_node_frequencies,cluster.node_frequencies)
				# sim = np.sum(np.multiply(B,cluster_node_frequencies))/(np.sqrt(len(stream['nodes'])+1)*np.sum(np.array(cluster_node_frequencies)))
				similarities.append(sim)
			return similarities

	def lru(self, index) :
		if index in self.lru_list :
			self.lru_list.remove(index)
			self.lru_list.append(index)
		else :
			self.lru_list.append(index)

		return

	def assign_to_cluster(self, stream, structural_similarity, content_similarity):
		if self.curr_num_clusters == 0:
			self.clusters[0].add_stream(stream)
			self.lru(0)
			wc.write_cluster(self.flags.cluster_tweet_ids, 0, [self.tweet_counter], True)
			self.curr_num_clusters = self.curr_num_clusters + 1
			print("\t New: 0")
			return
		else:
			SIM = list( self.flags.Lambda*np.array(structural_similarity) + (1-self.flags.Lambda)*np.array(content_similarity) )
			similarity = max(SIM)

			index = SIM.index(similarity)
			threshold = self.mean - 2*self.standard_deviation
			if threshold <= 0 :
				threshold = self.mean - self.standard_deviation
				if threshold <= 0 :
					threshold = self.mean
			# if threshold <= 0:
			# 	threshold = self.mean
			if similarity > threshold :

				# for i in self.curr_num_clusters:
				# 	for word in zero_idf_words :
				# 		del self.clusters[i].word_frequencies[word]
				# 		self.clusters[i].words.remove(word)

				# for word in zero_idf_words :
				# 	stream['words']

				if self.curr_num_clusters == self.flags.num_clusters:
					for word in self.zero_idf_words :
						ind = stream['words'].index(word)
						del stream['words'][ind]
						del stream['word_frequencies'][ind]

						self.clusters[index].words.remove(word)
						del self.clusters[index].word_frequencies[word]

						print("stop word")
						print(word)

				self.clusters[index].add_stream(stream)
				self.lru(index)
				wc.write_cluster(self.flags.cluster_tweet_ids, index, [self.tweet_counter])
				print('\tAssigned to-' + str(index))
			else:
				if self.curr_num_clusters < self.flags.num_clusters:
					self.clusters[self.curr_num_clusters].add_stream(stream)
					self.lru(index)
					wc.write_cluster(self.flags.cluster_tweet_ids, self.curr_num_clusters, [self.tweet_counter], True)
					print('\tNew: ' + str(self.curr_num_clusters))
					self.curr_num_clusters = self.curr_num_clusters + 1
				else:
					index = self.lru_list[0]
					summary = []
					wc.write_cluster(self.flags.cluster_summary, index, [], True)
					for freq in self.clusters[index].word_frequencies :
						wc.write_cluster(self.flags.cluster_summary, index,  [str(freq) + ':' + str(self.clusters[index].word_frequencies[freq])])
					# re-create the cluster
					self.clusters[index].create_new_cluster()
					self.clusters[index].add_stream(stream)
					self.lru(index)
					wc.write_cluster(self.flags.cluster_tweet_ids, index, [self.tweet_counter], True)
					print('\tNovel Event '+ str(index))


			self.update_moments(similarity)
			return

	def monitoring(self):
		while True:
			print("Tweet: " + str(self.tweet_counter) + "  clusters: " + str(self.curr_num_clusters) + "  Mean: " + str(self.mean) + "  standard_deviation: " + str(self.standard_deviation))
			# print("Moments[0]: ", self.moments[0], "Moments[1]: ", self.moments[1], "Moments[2]: ", self.moments[2])
			tweet = read_data.read_tweets(self.reader_words, self.reader_nodes)
			if tweet is None:
				break
			elif len(tweet['words']) is 0:
				continue
			else :
				c_s = self.content_similarity(tweet)
				s_s = self.structural_similarity(tweet)

				self.assign_to_cluster(tweet, s_s, c_s)
			# print("\n=====================")
			# for i_ in range(self.curr_num_clusters):
			# 	cluster = self.clusters[i_]
			# 	print("\t",cluster.word_frequencies)
			# print("=====================\n")
			# if self.tweet_counter >12:
			# 	break


			if self.tweet_counter == 153002 :
				for i in range(self.flags.num_clusters) :
					wc.write_cluster(self.flags.cluster_summary, i, [], True)
					for freq in self.clusters[i].word_frequencies :
						wc.write_cluster(self.flags.cluster_summary, i,  [str(freq) + ':' + str(self.clusters[i].word_frequencies[freq])])
				break

			self.tweet_counter = self.tweet_counter + 1

		return

if __name__ == "__main__":

	flags = FLAGS()
	obj = Event_Detection(flags)
	obj.monitoring()
