require 'torch'
require 'util'
require 'image'
require 'gnuplot'
require 'lfs'


train_mean = 0

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
			local uuid = mysplit(label_filename, "_")[1]
			local image_filename = uuid .. "_gray.dat"
			local spec_filename = uuid .. ".txt"

			local label_file = assert(io.open(type_str .. "_set/labels/" .. label_filename, "r"))
			local image_file = assert(io.open(type_str .. "_set/images/" .. image_filename, "r"))
			local spec_file = assert(io.open(type_str .. "_set/specs/" .. spec_filename, "r"))

			local spec = spec_file:read()
			local spec_ary = mysplit(spec, ",")
			local width = spec_ary[1] + pad * 2
			local height = spec_ary[2] + pad * 2
			local img = torch.zeros(1, height, width):fill(255)
			local ori_img = torch.zeros(1, height, width):fill(255)
			local label_img = torch.zeros(height, width):fill(0)
			for x = 1 + pad, width - pad do
				for y = 1 + pad, height - pad do
					img[1][y][x] = tonumber(string.byte(image_file:read(1)))
					ori_img[1][y][x] = img[1][y][x]
					local label = tonumber(string.byte(label_file:read(1)))
					label_img[y][x] = label
					if (label > 0) then
						type_data[type_idx] = { imgs_type_idx, y, x, label }
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
			imgs_type_idx = imgs_type_idx + 1

			if (imgs_type_idx % 10 == 0) then
				print(imgs_type_idx .. " images loaded")
			end
		end
	end

	for i = 1,table.getn(type_data) do
		type_idx_ary[i] = i
	end

	print("Finish loading training data set.")
end

function load_training_data()
	ori_imgs_train = { }
	imgs_train = { }
	label_imgs_train = { }
	train_data = { }
	train_idx_ary = { }

	ori_imgs_type = ori_imgs_train
	imgs_type = imgs_train
	label_imgs_type = label_imgs_train
	type_data = train_data
	type_idx_ary = train_idx_ary
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

	ori_imgs_type = ori_imgs_test
	imgs_type = imgs_test
	label_imgs_type = label_imgs_test
	type_data = test_data
	type_idx_ary = test_idx_ary
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

	local input = imgs_train[img_idx]:sub(1, 1, stepY - (length-1)/2, stepY + (length-1)/2, stepX - (length-1)/2, stepX + (length-1)/2)
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
