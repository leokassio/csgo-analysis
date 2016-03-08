# -*- coding: utf-8 -*- 
# ============================================================================================
# GNU Public License
# Dev: Kassio Machado - Brazil
# Created on 2015-12-11 - Ottawa/Canada
# Simple script to collect the ammount of live viewers on a list o Twitch channels
# ============================================================================================

import sys
import math
import time
import numpy
import string
import datetime
import colorama
import scipy.stats
from scipy.misc import imread
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm 

# import plotly.graph_objs as go
# import plotly.plotly as py
# from plotly.graph_objs import *

plt.style.use('ggplot')

def getTeams():
	teams = list()
	teams.append('fnatic')
	teams.append('luminosity')
	teams.append('clgaming')
	teams.append('natusvincere')
	teams.append('teamenvyus')
	teams.append('teamliquidpro')
	teams.append('teamvirtuspro')
	teams.append('tempostorm')
	teams.append('astralisgg')
	teams.append('flipsid3tactics')
	teams.append('nipgaming')
	teams.append('cloud9gg')
	teams.append('opticgaming')
	teams.append('zengamingx')
	teams.append('fazeclan')
	teams.append('mousesports')
	teams.append('teamdignitas')
	teams.append('clgred')
	teams.append('teamsolomid')
	teams.append('g2esports')
	return [t.lower() for t in teams]

def getPlayers():
	players = list()
	players.append('lgfallen')
	players.append('jwcsgo')
	players.append('olofmcs')
	players.append('fazerain')
	players.append('lgfnx')
	players.append('krimzcsgo')
	players.append('lgtaco')
	players.append('fnaticfebiven')
	players.append('lgfergod')
	players.append('envykennys')
	players.append('fnaticflusha')
	players.append('fnaticdennis')
	players.append('lgcoldzera')
	players.append('flamiecs')
	players.append('f0restCS')
	players.append('alluCSGO')
	players.append('OfficialXizt')
	players.append('fribergCS')
	players.append('GeT_RiGhTcs')
	players.append('deviceCS')
	players.append('dupreehCSGO')
	players.append('Xyp9x')
	players.append('Liquidnitr0')
	players.append('Liquid_Hiko')
	players.append('LiquidEliGE')
	players.append('C9n0thing')
	players.append('C9_Skadoodle')
	players.append('C9freakazoid')
	players.append('nV_HappyV')
	players.append('EnVy_kennyS')
	players.append('ENVYUS_apEX')
	players.append('kiocsgoo')
	players.append('nV_NBK')
	players.append('g5taz')
	players.append('neo_fkubski')
	players.append('paszaBiceps')
	players.append('byalics')
	players.append('seizedwf')
	players.append('NaViGuardiaN')
	players.append('IaonnSukhariev')
	players.append('navizeus4444')
	return [p.lower() for p in players]

def plotlyTwitchViewersAlternative(filename):
	houroffset = 2
	dictTimeline = dict() # channel, timestamp, viewers
	fileinput = open(filename, 'r')
	settimeline = set()
	for line in fileinput:
		line = line.replace('\n', '')
		dictRegister = eval(line)
		timestampkey = dictRegister['timestamp']
		timestampkey = timestampkey + datetime.timedelta(hours=houroffset)
		timestampkey = timestampkey.strftime('%Y-%m-%d %H%:%M:00')
		settimeline.add(timestampkey)
		channel = dictRegister['stream']['channel']['name']
		viewers = dictRegister['stream']['viewers']
		if channel in ['pashabiceps', 'stormstudio_csgo_ru', 'esl_csgo', ]:
			continue
		try:
			dictTimeline[channel][timestampkey].append(viewers)
		except KeyError:
			if channel not in dictTimeline:
				dictTimeline[channel] = dict()
			dictTimeline[channel][timestampkey] = list()
			dictTimeline[channel][timestampkey].append(viewers)

	data = list()
	labels = sorted(settimeline)
	for c in sorted(dictTimeline):
		for t in labels:
			try:
				dictTimeline[c][t] = numpy.average(dictTimeline[c][t])
			except KeyError:
				dictTimeline[c][t] = 0
		timeserie = [dictTimeline[c][t] for t in labels]
		trace = go.Scatter(x=labels,y=timeserie,mode='lines+markers',name=c[0].upper()+c[1:])
		data.append(trace)
	layout = Layout(
		autosize=True,
		height=545,
		width=1043,
		margin=Margin(r=10, b=100,l=60),
		plot_bgcolor='rgb(255, 255, 255)',
		title='Twitch - ESL ESEA Finals',
		xaxis=XAxis(autorange=True, title='Timeline', type='date'),
		yaxis=YAxis(autorange=True, title='Twitch Viewers', type='linear')
	)
	figure = Figure(data=data, layout=layout)
	py.plot(figure, filename='ESL ESEA Finals - Twitch Viewers')

def plotlyTwitchViewers(filename):
	houroffset = 2
	dictTimeline = dict() # channel, timestamp, viewers
	fileinput = open(filename, 'r')
	settimeline = set()
	for line in fileinput:
		line = line.replace('\n', '')
		dictRegister = eval(line)
		timestampkey = dictRegister['timestamp']
		timestampkey = timestampkey + datetime.timedelta(hours=houroffset)
		timestampkey = timestampkey.strftime('%Y-%m-%d %H%:%M:00')
		settimeline.add(timestampkey)
		channel = dictRegister['stream']['channel']['name']
		viewers = dictRegister['stream']['viewers']
		if channel in ['pashabiceps']:
			continue
		try:
			dictTimeline[channel][timestampkey].append(viewers)
		except KeyError:
			if channel not in dictTimeline:
				dictTimeline[channel] = dict()
			dictTimeline[channel][timestampkey] = list()
			dictTimeline[channel][timestampkey].append(viewers)

	data = list()
	labels = sorted(settimeline)
	for c in sorted(dictTimeline):
		for t in labels:
			try:
				dictTimeline[c][t] = numpy.average(dictTimeline[c][t])
			except KeyError:
				dictTimeline[c][t] = 0
	
	dictData = dict()
	for t in labels:
		s = sum([dictTimeline[c][t] for c in dictTimeline])
		dictData[t] = s
	timeserie = [dictData[t] for t in labels]
	trace = go.Scatter(x=labels,y=timeserie,mode='lines+markers',name='ESL ESEA Finals')
	data.append(trace)
	layout = Layout(
		autosize=True,
		height=545,
		width=1043,
		margin=Margin(r=10, b=100,l=60),
		plot_bgcolor='rgb(255, 255, 255)',
		title='Twitch - ESL ESEA Finals',
		xaxis=XAxis(autorange=True, title='Timeline', type='date'),
		yaxis=YAxis(autorange=True, title='Twitch Viewers', type='linear')
	)
	figure = Figure(data=data, layout=layout)
	py.plot(figure, filename='Twitch - ESL ESEA Finals')







def plotHltvViewers():
	args = sys.argv
	tracefile = open(args[2], 'r')
	nLines = sum(1 for line in tracefile)
	tracefile.seek(0)

	dictStreamAlias = dict()
	dictStreamTimeline = dict()

	for line in tqdm(tracefile, total=nLines):
		linesplited = line.replace('\n', '').split(',')
		timestamp = datetime.datetime.strptime(linesplited[0], '%Y-%m-%d %H:%M:%S') # 2016-02-24 14:15:00
		streamName = linesplited[1]
		streamViewers = int(linesplited[2])
		streamUrl = linesplited[3]
		
		dictStreamAlias[streamUrl] = streamName
		try:
			dictStreamTimeline[streamUrl]['viewers'].append(streamViewers)
			dictStreamTimeline[streamUrl]['timestamp'].append(timestamp)
		except KeyError:
			dictStreamTimeline[streamUrl] = dict()
			dictStreamTimeline[streamUrl]['viewers'] = list()
			dictStreamTimeline[streamUrl]['timestamp'] = list()
			dictStreamTimeline[streamUrl]['viewers'].append(streamViewers)
			dictStreamTimeline[streamUrl]['timestamp'].append(timestamp)

	data = list()
	dataMean = list()
	dataStd = list()
	labels = list()
	most_popular = sorted(dictStreamTimeline.keys(), key=lambda w:max(dictStreamTimeline[w]['viewers']), reverse=True)
	for url in most_popular:
		alias = dictStreamAlias[url].lower()
		if 'esl' in alias:
			d = dictStreamTimeline[url]['viewers']
			m, h, md, mu = mean_confidence_interval(d)
			print alias, numpy.mean(d), url
			dataMean.append(m)
			dataStd.append(h)
			data.append(d)
			labels.append(alias)

	data = data[:10]
	dataMean = dataMean[:10]
	dataStd = dataStd[:10]
	labels = labels[:10]
	# plt.boxplot(data, range(len(data)))
	# plt.bar(range(len(dataMean)), dataMean, width=1)
	plt.bar(range(len(dataMean)), dataMean, color='green', yerr=dataStd, error_kw=dict(ecolor='k', elinewidth=2, capsize=4))
	plt.xticks(range(len(labels)), labels, fontsize=15, rotation=45)
	plt.ylabel('# of Viewers', fontsize=15)
	plt.tick_params(axis='x', labelsize=15)
	plt.tick_params(axis='y', labelsize=15)
	plt.tight_layout()
	plt.savefig('stream-viewers-hltv.png', dpi=100)
	plt.clf()
	plt.close()

	timeline = dictStreamTimeline[most_popular[0]]['timestamp']
	viewers = dictStreamTimeline[most_popular[0]]['viewers']
	dateBegin = datetime.datetime.strptime('2016-03-02 06:00:00', '%Y-%m-%d %H:%M:%S')
	dateEnd = datetime.datetime.strptime('2016-03-05 23:59:00', '%Y-%m-%d %H:%M:%S')
	
	dictData = dict()
	data = list()
	labels = list()
	for t, v in zip(timeline, viewers):
		t = (t + datetime.timedelta(hours=6)) # katowice timezone
		if t >= dateBegin and t <= dateEnd:
			minutes = roundMetric(t.minute, offset=30)
			if minutes == 0:
				minutes = str(minutes) + '0'
			else:
				minutes = str(minutes)
			timestampkey = t.strftime('%Y-%m-%d %H:' + minutes + ':%S')
			data.append(v)
			labels.append(t)
			try:
				dictData[timestampkey].append(v)
			except KeyError:
				print timestampkey
				dictData[timestampkey] = list()
				dictData[timestampkey].append(v)

	labels = sorted(dictData)
	dataplot = [numpy.mean(dictData[l]) for l in labels]
	plt.plot(range(len(dataplot)), dataplot, '-x', linewidth=2.5)
	plt.xlim((0, len(dataplot)))
	plt.tick_params(axis='x', labelsize=15)
	plt.tick_params(axis='y', labelsize=15)
	plt.xticks(range(0, len(labels), 8), [i.replace(':00:00', 'h').replace('2016-03', 'Mar') for i in labels[::8]], fontsize=15, rotation='vertical')
	plt.ylabel('ESL TV # of Viewers (HLTV)', fontsize=15)
	plt.tight_layout()
	plt.savefig('stream-viewers-hltv-esl.png', dpi=100)


	# plt.plot(range(len(data)), data, '-x', linewidth=2.5)
	# plt.xlim((0, len(data)+30))
	# plt.show()

def plotlyWordCloud():
	args = sys.argv
	tracefile = open(args[2], 'r')
	nLines = sum(1 for line in tracefile)
	tracefile.seek(0)

	dictTerms = dict()
	blacklist = STOPWORDS.copy()
	blacklist.add('rt')
	# blacklist.add('000')
	punctuation = set(string.punctuation)
	punctuation.remove('@')
	punctuation.remove('&')
	# punctuation.remove('#')
	for line in tqdm(tracefile, total=nLines):
		try:
			linesplited = line.split(', ')
			tweet = linesplited[6].lower()
			for p in punctuation:
				tweet = tweet.replace(p, '')
			terms = tweet.split(' ')
			for t in terms:
				if (len(t) > 1) and ('http' not in t) and (t not in blacklist):
					try:
						dictTerms[t] += 1
					except KeyError:
						dictTerms[t] = 1
		except IndexError:
			print 'IndexError'
	for t in blacklist:
		try:
			del dictTerms[t]
		except KeyError:
			continue
	popularTerms = sorted(dictTerms.keys(), key=lambda w:dictTerms[w], reverse=True)
	popularTerms = [p for p in popularTerms if (dictTerms[p]) > 1]
	print len(popularTerms)
	text = list()
	terms = ''
	for p in popularTerms:
		text.append((p, dictTerms[p]))
		for i in range(dictTerms[p]):
			terms += ' ' + p
	# print terms
	maskfile = 'csgo-icon'
	mask = imread(maskfile + '.jpg') # mask=mask
	wc = WordCloud(mask=mask, background_color='black', width=1280, height=720).generate(terms) # max_words=10000
	default_colors = wc.to_array()
	plt.figure()
	plt.imshow(default_colors)
	plt.axis('off')
	plt.savefig(maskfile + '-wordcloud.png', dpi=500, bbox_inches='tight', pad_inches=0) # bbox_inches='tight'
	plt.show()

def plotlyPopularity():
	args = sys.argv
	filename = args[2]

	dictUsers = dict()
	
	punctuation = set(string.punctuation)
	punctuation.remove('@')
	tracefile = open(filename, 'r')
	nLines = sum(1 for line in tracefile)
	tracefile.seek(0)

	for line in tqdm(tracefile, total=nLines):
		try:
			linesplited = line.split(', ')
			tweet = linesplited[6].lower().split(' ')
			for t in tweet:
				if len(t) > 1 and t[0] == '@':
					for p in punctuation:
						t = t.replace(p, '')
					try:
						dictUsers[t] += 1
					except KeyError:
						dictUsers[t] = 1
		except IndexError:
			print 'IndexError', line

	most_popular = sorted(dictUsers.keys(), key=lambda w:dictUsers[w], reverse=True)
	# for u in most_popular[:100]:
	# 	print u, dictUsers[u]
	
	players = set(getPlayers())
	labelsplayers = list()
	dataplayers = list()
	print '# Most Popular Places (already defined)'
	for mp in most_popular:
		mp_alias = mp.replace('@', '')
		if mp_alias in players:
			dataplayers.append(dictUsers[mp])
			labelsplayers.append(mp)
			print mp, dictUsers[mp]
	plt.bar(range(len(dataplayers)), dataplayers, width=1, color='red')
	plt.xticks(range(len(labelsplayers)), labelsplayers, fontsize=13, rotation='vertical')
	plt.ylabel('Twitter Mentions', fontsize=15)
	plt.xlim((0, len(labelsplayers)+1))
	plt.tick_params(axis='x', labelsize=15)
	plt.tick_params(axis='y', labelsize=15)
	plt.tight_layout()
	plt.savefig('csgo-player-mentions.png', dpi=100)
	plt.clf()
	plt.close()

	teams = set(getTeams())
	labelsteams = list()
	datateams = list()
	print '# Most Popular Teams (already defined)'
	for mp in most_popular:
		mp_alias = mp.replace('@', '')
		if mp_alias in teams:
			n = dictUsers[mp]
			if n > 1:
				datateams.append(n)
				labelsteams.append(mp)
				print mp, dictUsers[mp]
	plt.bar(range(len(datateams)), datateams, width=1, color='red')
	plt.xticks(range(len(labelsteams)), labelsteams, fontsize=12, rotation='vertical')
	plt.ylabel('Twitter Mentions', fontsize=15)
	plt.xlim((0, len(labelsteams)+1))
	plt.tick_params(axis='x', labelsize=15)
	plt.tick_params(axis='y', labelsize=15)
	plt.tight_layout()
	plt.savefig('csgo-teams-mentions.png', dpi=100)
	plt.clf()
	plt.close()

	exit() # TODO: PUBLISH ALSO ON PLOTLY
	dataPop = list()
	labels = list()
	for t in sorted(dictCounter.keys(), key=lambda w:dictCounter[w], reverse=True):
		if dictCounter[t] > 2:
			# print t, dictCounter[t]
			labels.append(t[0].upper()+t[1:])
			dataPop.append(dictCounter[t])
	dataColors = list()
	# datacolor showing 1 color for teams and other for players
	data = go.Bar(
		y = labels,
		x = dataPop,
		text = zip(labels, dataPop),
		orientation = 'h',
		opacity=0.6,
		marker = dict(color='rgba(50, 171, 96, 0.6)', line=dict(color='rgba(50, 171, 96, 1.0)', width=1))
		)
	layout = go.Layout(
		autosize=True,
		height=545,
		width=1043,
		# margin=Margin(r=10, b=100,l=60),
		# plot_bgcolor='rgb(255, 255, 255)',
		title='Twitter - Popular Players',
		xaxis=XAxis(autorange=True, title='Twitter Mentions'),
		yaxis=YAxis(autorange=True)
		)
	figure = Figure(data=[data], layout=layout)
	py.plot(figure, filename='Twitter Mentions - ESL ESEA Finals')

def plotlyTweetScatterMap():
	args = sys.argv
	filename = args[2]
	tracefile = open(filename, 'r')
	nLines = sum(1 for line in tracefile)
	tracefile.seek(0)

	dictLocations = dict()
	dictCoords = dict(lats=list(), lngs=list())

	for line in tqdm(tracefile, total=nLines):
		try:
			linesplited = line.split(', ')
			if linesplited[3] != 'None':
				dictCoords['lats'].append(float(linesplited[3]))
				dictCoords['lngs'].append(float(linesplited[2]))				
		except IndexError:
			print 'IndexError'
	print 'Precise Data Points:', len(dictCoords['lats'])
	
	tracefile.seek(0)
	for line in tqdm(tracefile, total=nLines):
		try:
			linesplited = line.split(', ')
			loc = linesplited[5]
			if loc != 'None':
				try:
					dictLocations[loc] += 1
				except KeyError:
					dictLocations[loc] = 1
		except IndexError:
			print 'IndexError'
	print 'Aproximated Data Points:', len(dictLocations)
	most_popular = sorted(dictLocations.keys(), key=lambda w:dictLocations[w], reverse=True)
	print 'Most Popular Locations'
	fileout = open(tracefile.name.replace('.csv', 'places-world-cloud.txt'), 'w')
	for i, mp in enumerate(most_popular):
		n = dictLocations[mp]
		print '#' + str(i), mp, n
		terms = mp.split(' ')
		for t in terms:
			if len(t) > 1:
				for x in range(n):
					fileout.write(t+'\n')
	fileout.close()

def exportValidTweets():
	blacklist_terms = set()
	blacklist_terms.add('giveaway')
	blacklist_terms.add('sorteo')

	args = sys.argv
	fileinput = open(args[2], 'r')
	nLines = sum(1 for line in fileinput)
	fileinput.seek(0)
	fileout = open(fileinput.name.replace('.', '_cleaned.'), 'w')

	for line in tqdm(fileinput, total=nLines, desc='cleaning dataset'):
		linesplited = line.split(', ')
		if 'IFTTT' in linesplited[-1]:
			continue
		text = linesplited[6].lower()
		flagWrite = True
		for t in blacklist_terms:
			if t in text:
				flagWrite = False
				break
		if flagWrite:
			fileout.write(line)

	fileout.close()
	fileinput.close()

# Calculate the confidence interval of array of data
def mean_confidence_interval(data, confidence=0.95):
	a = 1.0 * numpy.array(data)
	n = len(a)
	m, se = numpy.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
	return m, h, m-h, m+h

# Simple method to round an value to their near integer value
# according to offset parameter
def roundMetric(metric, offset):
	metric = int(math.ceil(metric/offset)*offset)
	return metric






# args = sys.argv

# filename = 'esl-esea-finals-twitch.data'
# plotlyTwitchViewers(filename)
# plotlyTwitchViewersAlternative(filename)

filename = '2015-12-10-csgo-esl-esea-finals.csv'
# plotlyPopularity(filename)
# plotlyWordCloud(filename)





# plotHltvViewers()
# plotlyWordCloud()
plotlyPopularity()
# plotlyTweetScatterMap()
# exportValidTweets()












