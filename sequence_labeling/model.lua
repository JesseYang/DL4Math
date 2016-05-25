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
	use_pca = true
	pca_dim = 80
	window = 1
	-- label_set = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "x", ".", "=", "(", ")", "f", "c", ":" }
	label_set = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "=" }
	klass = table.getn(label_set) + 1
	padding_height = 80
	horizon_pad = 0
	feature_len = padding_height
	if (use_pca == true) then
		feature_len = padding_height * window
	end
	hidden_size = 200

	l1 = nn.LSTM(pca_dim, hidden_size)
	l2 = nn.LSTM(pca_dim, hidden_size)
	o = nn.Linear(hidden_size * 2, klass)

	fwd = l1
	fwdSeq = nn.Sequencer(fwd)
	bwd = l2
	bwdSeq = nn.Sequencer(bwd)
	merge = nn.JoinTable(1, 1)
	mergeSeq = nn.Sequencer(merge)

	concat = nn.ConcatTable()
	concat:add(fwdSeq):add(nn.Sequential():add(nn.ReverseTable()):add(bwdSeq):add(nn.ReverseTable()))
	brnn = nn.Sequential()
		:add(concat)
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
	label_set = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "x", ".", "=", "(", ")", "f", "c", ":" }
	klass = table.getn(label_set) + 1
	padding_height = 80
	horizon_pad = 0
	feature_len = padding_height
	hidden_size = 50

	l1_1 = nn.LSTM(feature_len, hidden_size)
	l1_2 = nn.LSTM(feature_len, hidden_size)

	fwdSeq_1 = nn.Sequencer(l1_1)
	bwdSeq_1 = nn.Sequencer(l1_2)
	merge_1 = nn.JoinTable(1, 1)
	mergeSeq_1 = nn.Sequencer(merge_1)

	concat_1 = nn.ConcatTable()
	concat_1:add(fwdSeq_1):add(nn.Sequential():add(nn.ReverseTable()):add(bwdSeq_1))
	brnn_1 = nn.Sequential()
		:add(concat_1)
		:add(nn.ZipTable())
		:add(mergeSeq_1)


	l2_1 = nn.LSTM(2 * hidden_size, hidden_size)
	l2_2 = nn.LSTM(2 * hidden_size, hidden_size)

	fwdSeq_2 = nn.Sequencer(l2_1)
	bwdSeq_2 = nn.Sequencer(l2_2)
	merge_2 = nn.JoinTable(1, 1)
	mergeSeq_2 = nn.Sequencer(merge_2)

	concat_2 = nn.ConcatTable()
	concat_2:add(fwdSeq_2):add(nn.Sequential():add(nn.ReverseTable()):add(bwdSeq_2))
	brnn_2 = nn.Sequential()
		:add(concat_2)
		:add(nn.ZipTable())
		:add(mergeSeq_2)

	o = nn.Linear(hidden_size * 2, klass)
	rnn = nn.Sequential()
		:add(brnn_1)
		:add(brnn_2)
		:add(nn.Sequencer(o, 1)) -- times two due to JoinTable

	s = use_cuda == true and rnn:cuda() or rnn
end

function model_6()
	label_set = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "x", ".", "=", "(", ")", "f", "c", ":" }
	klass = table.getn(label_set) + 1
	ksize = 5
	window = 40
	padding_height = 80

	horizon_pad = 20

	m = nn.Sequential()
	-- first stage
	m:add(nn.SpatialConvolution(1, 32, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- second stage
	m:add(nn.SpatialConvolution(32, 64, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	-- third stage
	m:add(nn.SpatialConvolution(64, 64, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- fourth stage
	m:add(nn.SpatialConvolution(64, 64, ksize, ksize, 1, 1, (ksize - 1) / 2, (ksize - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- last stage: standard 2-layer mlp
	m:add(nn.Reshape(64 * 10 * 5))
	m:add(nn.Linear(64 * 10 * 5, 200))
	m:add(nn.Linear(200, klass))

	s = use_cuda == true and nn.Sequencer(m):cuda() or nn.Sequencer(m)
end

function model_7()
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

function model_8()
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
	m:add(nn.SpatialConvolution(32, 64, k2, k2, 1, 1, (k2 - 1) / 2, (k2 - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- third stage
	m:add(nn.SpatialConvolution(64, 64, k2, k2, 1, 1, (k2 - 1) / 2, (k2 - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- fourth stage
	m:add(nn.SpatialConvolution(64, 64, k2, k2, 1, 1, (k2 - 1) / 2, (k2 - 1) / 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	-- last stage: standard 1-layer mlp
	m:add(nn.Reshape(64 * 4 * 5))
	m:add(nn.Linear(64 * 4 * 5, klass))
	-- m:add(nn.Linear(200, klass))

	s = use_cuda == true and nn.Sequencer(m):cuda() or nn.Sequencer(m)
end
