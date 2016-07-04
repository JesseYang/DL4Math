local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'torch'
require 'util'
require 'image'
require 'gnuplot'
require 'lfs'


mean_file = assert(io.open("models/mean", "r"))
train_mean = tonumber(mean_file:read())

function output_pred_on_test_set()
	if (colors == nil) then
		load_color_config()
	end
	-- use the model to predict the output of the test images, and save for the usage of the line extraction app
	for img_idx = 1, table.getn(imgs_test) do
		local img = imgs_test[img_idx]
		local ori_img = ori_imgs_test[img_idx]
		local s = img:size()
		local height = s[2]
		local width = s[3]
		local result = torch.zeros(3, height, width)
		print(height - 2 * pad)
		print(width - 2 * pad)
		local label_file = assert(io.open("results/" .. test_imgs_prefix[img_idx] .. "_label.dat", "w"))
		for x = pad + 1, width - pad do
			for y = pad + 1, height - pad do
				if (ori_img[1][y][x] ~= 255) then
					local i = use_cuda and
						img:sub(1, 1, y - (length-1)/2, y + (length-1)/2, x - (length-1)/2, x + (length-1)/2):cuda() or
						img:sub(1, 1, y - (length-1)/2, y + (length-1)/2, x - (length-1)/2, x + (length-1)/2)
					score = m:forward(i)
					local m_t, m_i = torch.max(score, 1)

					for c = 1,3 do
						result[c][y][x] = colors[m_i[1]][c]
					end

					label_file:write(string.char(m_i[1]))
				else
					label_file:write(string.char(0))
				end
			end
		end
		label_file:flush()
		label_file:close()
		-- image.display(result, 3)
	end
end

function load_test_data_for_predict()
	local imgs_test_idx = 1

	local imgs_count = 0
	for img_filename in lfs.dir("for_predict") do
		if (img_filename ~= "." and img_filename ~= "..") then
			imgs_count = imgs_count + 1
		end
	end
	print("Loading data set for predict (" .. imgs_count .. " images)")

	imgs_test = { }
	ori_imgs_test = { }
	test_imgs_prefix = { }

	for img_filename in lfs.dir("for_predict") do
		if (img_filename ~= "." and img_filename ~= "..") then
			temp_idx = string.find(img_filename, ".jpg")
			filename_prefix = img_filename:sub(1, temp_idx - 1)
			image_file = cv.imread { "for_predict/" .. img_filename, cv.IMREAD_GRAYSCALE }
			cv.threshold{image_file, image_file, 10, 255, cv.THRESH_BINARY}
			height = image_file:size()[1] + pad * 2
			width = image_file:size()[2] + pad * 2
			ori_img = torch.ByteTensor(1, height, width):fill(255)

			for x = 1 + pad, width - pad do
				for y = 1 + pad, height - pad do
					ori_img[1][y][x] = image_file[y - pad][x - pad]
				end
			end

			local img = ori_img:clone():float() - train_mean

			imgs_test[imgs_test_idx] = img
			ori_imgs_test[imgs_test_idx] = ori_img
			test_imgs_prefix[imgs_test_idx] = filename_prefix

			if (imgs_test_idx % 10 == 0) then
				print(imgs_test_idx .. " images loaded")
			end

			imgs_test_idx = imgs_test_idx + 1

		end
	end

	print("Finish loading data set for predict.")
end

function load_data()
	local imgs_type_idx = 1
	local type_idx = 1

	local imgs_count = 0
	for label_filename in lfs.dir(type_str .. "_set/labels") do
		if (label_filename ~= "." and label_filename ~= "..") then
			imgs_count = imgs_count + 1
		end
	end
	print("Loading " .. type_str .. " data set (" .. imgs_count .. " images)")

	for label_filename in lfs.dir(type_str .. "_set/labels") do
		if (label_filename ~= "." and label_filename ~= "..") then
			local temp_idx = string.find(label_filename, "_label") - 1
			local filename_prefix = label_filename:sub(1, temp_idx)
			local image_filename = filename_prefix .. "_gray.dat"
			local spec_filename = filename_prefix .. ".txt"

			local label_file = assert(io.open(type_str .. "_set/labels/" .. label_filename, "r"))
			local image_file = assert(io.open(type_str .. "_set/images/" .. image_filename, "r"))
			local spec_file = assert(io.open(type_str .. "_set/specs/" .. spec_filename, "r"))

			local spec = spec_file:read()
			local spec_ary = mysplit(spec, ",")
			local width = spec_ary[1] + pad * 2
			local height = spec_ary[2] + pad * 2
			local img = torch.FloatTensor(1, height, width):fill(255)
			-- local ori_img = torch.zeros(1, height, width):fill(255)
			local ori_img = torch.ByteTensor(1, height, width):fill(255)
			-- local label_img = torch.zeros(height, width):fill(0)
			local label_img = torch.ByteTensor(height, width):fill(0)
			for x = 1 + pad, width - pad do
				for y = 1 + pad, height - pad do
					img[1][y][x] = tonumber(string.byte(image_file:read(1)))
					ori_img[1][y][x] = img[1][y][x]
					local label = tonumber(string.byte(label_file:read(1)))
					label_img[y][x] = label
					if (label > 0) then
						type_data[type_idx] = { imgs_type_idx, y, x, label }
						type_idx = type_idx + 1
					elseif (img[1][y][x] < 10) then
						-- ATTENTION: there seems to be a bug when exporting data in the C# app.
						-- It seems that the color value of some black point is not 0, but some small value.
						-- so here we make label of all such values as "1", which is the normal pixel label,
						-- Pay attention that both "label" appears in both type_data and the label_img  
						type_data[type_idx] = { imgs_type_idx, y, x, 1 }
						label_img[y][x] = 1
						type_idx = type_idx + 1
					end
				end
			end
			local mean
			if (type_str == "training") then
				mean = img:sum() / (width * height)
				train_mean = mean
			else
				mean = train_mean
			end
			local img = (img - mean)
			imgs_type[imgs_type_idx] = img
			label_imgs_type[imgs_type_idx] = label_img
			ori_imgs_type[imgs_type_idx] = ori_img
			type_imgs_prefix[imgs_type_idx] = filename_prefix

			if (imgs_type_idx % 10 == 0) then
				print(imgs_type_idx .. " images loaded")
			end

			imgs_type_idx = imgs_type_idx + 1
		end
	end

	for i = 1,table.getn(type_data) do
		type_idx_ary[i] = i
	end

	print("Finish loading " .. type_str .. " data set.")
end

function load_training_data()
	ori_imgs_train = { }
	imgs_train = { }
	label_imgs_train = { }
	train_data = { }
	train_idx_ary = { }
	train_imgs_prefix = { }

	ori_imgs_type = ori_imgs_train
	imgs_type = imgs_train
	label_imgs_type = label_imgs_train
	type_data = train_data
	type_idx_ary = train_idx_ary
	type_imgs_prefix = train_imgs_prefix
	type_str = "training"

	load_data()

	train_idx = 1
end

function load_test_data()
	ori_imgs_test = { }
	imgs_test = { }
	label_imgs_test = { }
	test_data = { }
	test_idx_ary = { }
	test_imgs_prefix = { }

	ori_imgs_type = ori_imgs_test
	imgs_type = imgs_test
	label_imgs_type = label_imgs_test
	type_data = test_data
	type_idx_ary = test_idx_ary
	type_imgs_prefix = test_imgs_prefix
	type_str = "test"

	load_data()

	test_idx = 1
end

function load_validate_data()
	ori_imgs_validate = { }
	imgs_validate = { }
	label_imgs_validate = { }
	validate_data = { }
	validate_idx_ary = { }

	ori_imgs_type = ori_imgs_validate
	imgs_type = imgs_validate
	label_imgs_type = label_imgs_validate
	type_data = validate_data
	type_idx_ary = validate_idx_ary
	type_str = "validate"

	load_data()

	validate_idx = 1
end

function nextSample()
	epoch = epoch or 0

	-- whether goto next epoch
	if (train_idx == 1) then
		epoch = epoch + 1
		shuffle(train_idx_ary)
	end

	local train_ele = train_data[train_idx_ary[train_idx]]
	local target = train_ele[4]
	local img_idx = train_ele[1]
	local stepX = train_ele[3]
	local stepY = train_ele[2]

	train_idx = (train_idx == table.getn(train_data)) and 1 or (train_idx + 1)

	local input = use_cuda and
		imgs_train[img_idx]:sub(1, 1, stepY - (length-1)/2, stepY + (length-1)/2, stepX - (length-1)/2, stepX + (length-1)/2):cuda() or
		imgs_train[img_idx]:sub(1, 1, stepY - (length-1)/2, stepY + (length-1)/2, stepX - (length-1)/2, stepX + (length-1)/2)
	return input, target
end

function load_color_config()
	colors = torch.Tensor(klass, 3)
	label_name_ary = { }
	local color_config = assert(io.open("color_config.txt", "r"))
	for i = 1,klass do
		local str = color_config:read()
		local ary = mysplit(str, ':')
		label_name_ary[i] = ary[1]
		ary = mysplit(ary[2], ',')
		colors[i][1] = ary[1]
		colors[i][2] = ary[2]
		colors[i][3] = ary[3]
	end
end
