require 'torch'
require 'nn'
require 'nnx'


-- model_0 is a toy model
function model_0()
	label_set = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "x", ".", "=", "(", ")", "f", "c", ":" }
	klass = 5

	m = nn.Sequential()
	m:add(nn.SpatialConvolution(1, 5, 3, 3))
	m:add(nn.Reshape(5))
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
end
