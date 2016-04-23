require 'torch'
require 'nn'
require 'nnx'

label_set = { "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "+", "-", "*", "/", "=", ".", "x", "(", ")", "c", ":" }

-- model_0 is a toy model
function model_0()
	klass = 5

	m = nn.Sequential()
	m:add(nn.SpatialConvolution(1, 5, 3, 3))
	m:add(nn.Reshape(5))
end

function model_1()
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
