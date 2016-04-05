require 'torch'
require 'nn'
require 'nnx'


function model_1()
	klass = 8
	ksize = 3
	length = 27
	pad = (length - 1) / 2

	m = nn.Sequential()
	-- first stage
	m:add(nn.SpatialConvolution(1, 16, ksize, ksize, 1, 1, 1, 1))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(3, 3, 3, 3))
	-- second stage
	m:add(nn.SpatialConvolution(16, 64, ksize, ksize, 1, 1, 1, 1))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(3, 3, 3, 3))
	-- last stage: standard 1-layer mlp
	m:add(nn.Reshape(64 * 3 * 3))
	m:add(nn.Linear(64 * 3 * 3, klass))

	-- LogSoftMax
	m:add(nn.LogSoftMax())
end

function model_2()
	klass = 8
	length = 45
	pad = (length - 1) / 2

	m = nn.Sequential()
	-- first stage
	m:add(nn.SpatialConvolution(1, 16, 5, 5, 1, 1, 2, 2))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(5, 5, 5, 5))
	-- second stage
	m:add(nn.SpatialConvolution(16, 64, 3, 3, 1, 1, 1, 1))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(3, 3, 3, 3))
	-- last stage: standard 1-layer mlp
	m:add(nn.Reshape(64 * 3 * 3))
	m:add(nn.Linear(64 * 3 * 3, klass))

	-- LogSoftMax
	m:add(nn.LogSoftMax())
end

function model_3()
	klass = 8
	ksize = 3
	length = 81
	pad = (length - 1) / 2

	-- 0th stage
	m = nn.Sequential()
	c = nn.Concat(1)
	n = nn.Sequential()
	n:add(nn.Narrow(2, 28, 27)):add(nn.Narrow(3, 28, 27))
	c:add(n)
	c:add(nn.SpatialDownSampling{rW=3, rH=3})
	m:add(c)
	-- first stage
	m:add(nn.SpatialConvolution(2, 16, ksize, ksize, 1, 1, 1, 1))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(3, 3, 3, 3))
	-- second stage
	m:add(nn.SpatialConvolution(16, 64, ksize, ksize, 1, 1, 1, 1))
	m:add(nn.ReLU())
	m:add(nn.SpatialMaxPooling(3, 3, 3, 3))
	-- last stage: standard 1-layer mlp
	m:add(nn.Reshape(64 * 3 * 3))
	m:add(nn.Linear(64 * 3 * 3, klass))

	-- LogSoftMax
	m:add(nn.LogSoftMax())
end
