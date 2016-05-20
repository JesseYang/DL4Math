require 'torch'
require 'nn'
require 'nnx'
require 'rnn'


-- model_0 is a toy model
function model_0()
	label_set = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "x", ".", "=", "(", ")", "f", "c", ":" }
	klass = 5

	m = nn.Sequential()
	m:add(nn.SpatialConvolution(1, 5, 3, 3))
	m:add(nn.Reshape(5))

	s = use_cuda == true and nn.Sequencer(m):cuda() or nn.Sequencer(m)
end

function model_1()
	label_set = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "x", ".", "=", "(", ")", "f", "c", ":" }
	klass = table.getn(label_set) + 1
	ksize = 3
	length = 27
	window = 27
	padding_height = 81

	horizon_pad = 20

	m = nn.Sequential()
	-- first stage
	m:add(nn.SpatialConvolution(1, 16, ksize, ksize, 1, 1, 1, 1))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(1, 3, 1, 3))
	-- second stage
	m:add(nn.SpatialConvolution(16, 32, ksize, ksize, 1, 1, 1, 1))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(3, 3, 3, 3))
	-- third stage
	m:add(nn.SpatialConvolution(32, 64, ksize, ksize, 1, 1, 1, 1))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(3, 3, 3, 3))
	-- last stage: standard 1-layer mlp
	m:add(nn.Reshape(64 * 3 * 3))
	m:add(nn.Linear(64 * 3 * 3, klass))

	s = use_cuda == true and nn.Sequencer(m):cuda() or nn.Sequencer(m)
end

function model_2()
	label_set = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "x", ".", "=", "(", ")", "f", "c", ":" }
	klass = table.getn(label_set) + 1
	ksize = 5
	window = 40
	padding_height = 80

	horizon_pad = 20

	m = nn.Sequential()
	-- first stage
	m:add(nn.SpatialConvolution(1, 16, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- second stage
	m:add(nn.SpatialConvolution(16, 32, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- third stage
	m:add(nn.SpatialConvolution(32, 64, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- last stage: standard 1-layer mlp
	m:add(nn.Reshape(64 * 10 * 5))
	m:add(nn.Linear(64 * 10 * 5, klass))

	s = use_cuda == true and nn.Sequencer(m):cuda() or nn.Sequencer(m)
end

function model_3()
	label_set = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "x", ".", "=", "(", ")", "f", "c", ":", "k" }
	klass = table.getn(label_set) + 1
	ksize = 5
	window = 40
	padding_height = 80

	horizon_pad = 20

	m = nn.Sequential()
	-- first stage
	m:add(nn.SpatialConvolution(1, 16, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- second stage
	m:add(nn.SpatialConvolution(16, 32, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- third stage
	m:add(nn.SpatialConvolution(32, 64, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- last stage: standard 1-layer mlp
	m:add(nn.Reshape(64 * 10 * 5))
	m:add(nn.Linear(64 * 10 * 5, klass))

	s = use_cuda == true and nn.Sequencer(m):cuda() or nn.Sequencer(m)
end

function model_4()
	-- the rnn model
	use_rnn = true
	label_set = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "x", ".", "=", "(", ")", "f", "c", ":" }
	klass = table.getn(label_set) + 1
	padding_height = 80
	horizon_pad = 0
	feature_len = padding_height
	hidden_size = 200

	l1 = nn.LSTM(feature_len, hidden_size)
	l2 = nn.LSTM(feature_len, hidden_size)
	o = nn.Linear(hidden_size * 2, klass)

	fwd = l1
	fwdSeq = nn.Sequencer(fwd)
	bwd = l2
	bwdSeq = nn.Sequencer(bwd)
	merge = nn.JoinTable(1, 1)
	mergeSeq = nn.Sequencer(merge)

	parallel = nn.ParallelTable()
	parallel:add(fwdSeq):add(bwdSeq)
	brnn = nn.Sequential()
		:add(parallel)
		:add(nn.ZipTable())
		:add(mergeSeq)

	rnn = nn.Sequential()
		:add(brnn) 
		:add(nn.Sequencer(o, 1)) -- times two due to JoinTable

	s = use_cuda == true and rnn:cuda() or rnn
end

function model_5()
	-- the rnn model
	use_rnn = true
	use_mixed = true
	label_set = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "x", ".", "=", "(", ")", "f", "c", ":" }
	klass = table.getn(label_set) + 1
	padding_height = 80
	horizon_pad = 10
	feature_len = 640
	hidden_size = 100
	window = 27

	ksize = 3
	m = nn.Sequential()
	-- first stage
	m:add(nn.SpatialConvolution(1, 16, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(3, 2, 3, 2))
	-- second stage
	m:add(nn.SpatialConvolution(16, 32, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(3, 2, 3, 2))
	-- third stage
	m:add(nn.SpatialConvolution(32, 64, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(3, 2, 3, 2))
	-- reshape
	m:add(nn.Reshape(1, feature_len))

	l1 = nn.LSTM(feature_len, hidden_size)
	l2 = nn.LSTM(feature_len, hidden_size)
	o = nn.Linear(hidden_size * 2, klass)

	fwd = nn.Sequential()
		:add(m)
		:add(l1)
	fwdSeq = nn.Sequencer(fwd)
	bwd = nn.Sequential()
		:add(m)
		:add(l2)
	bwdSeq = nn.Sequencer(bwd)
	merge = nn.JoinTable(1, 1)
	mergeSeq = nn.Sequencer(merge)

	parallel = nn.ParallelTable()
	parallel:add(fwdSeq):add(bwdSeq)
	brnn = nn.Sequential()
		:add(parallel)
		:add(nn.ZipTable())
		:add(mergeSeq)

	rnn = nn.Sequential()
		:add(brnn) 
		:add(nn.Sequencer(o, 1)) -- times two due to JoinTable

	s = use_cuda == true and rnn:cuda() or rnn
end

function model_6()
	label_set = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "x", ".", "=", "(", ")", "f", "c", ":" }
	klass = table.getn(label_set) + 1
	k1 = 7
	k2 = 3
	window = 64
	padding_height = 80

	horizon_pad = 30

	m = nn.Sequential()
	-- first stage
	m:add(nn.SpatialConvolution(1, 32, k1, k1, 1, 1, (k1 - 1) / 2, (k1 - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- second stage
	m:add(nn.SpatialConvolution(32, 32, k2, k2, 1, 1, (k2 - 1) / 2, (k2 - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- third stage
	m:add(nn.SpatialConvolution(32, 64, k2, k2, 1, 1, (k2 - 1) / 2, (k2 - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- fourth stage
	m:add(nn.SpatialConvolution(64, 64, k2, k2, 1, 1, (k2 - 1) / 2, (k2 - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- last stage: standard 1-layer mlp
	m:add(nn.Reshape(64 * 4 * 5))
	m:add(nn.Linear(64 * 4 * 5, klass))

	s = use_cuda == true and nn.Sequencer(m):cuda() or nn.Sequencer(m)
end
