local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'lfs'
require 'util'

lfs.mkdir("300dpi_binary")

for img_file in lfs.dir("300dpi") do
	if (img_file ~= "." and img_file ~= "..") then
		print(img_file)

		-- binarization
		local im_gray = cv.imread { "300dpi/" .. img_file, cv.IMREAD_GRAYSCALE }
		local im_bw = cv.adaptiveThreshold{im_gray, im_bw, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 21}
		cv.imwrite { "300dpi_binary/" .. img_file, im_bw }

		-- line extraction
		local name_ary = mysplit(img_file, ".")
		local line_dir_name = name_ary[1] .. "_lines"
		lfs.mkdir("300dpi_binary/" .. line_dir_name)
		for line_file in lfs.dir("300dpi_binary/" .. line_dir_name) do
			os.remove("300dpi_binary/" .. line_dir_name .. "/" .. line_file)
		end
		local is_last_row_space = true
		local start_row = 0
		local size =  im_bw:size()
		local height = size[1]
		local width = size[2]
		local line_idx = 1
		for cur_row = 1,height do
			if (torch.sum(im_bw[cur_row]) == 255 * width) then
				-- this should be a separation row, until last row should be a line
				if (is_last_row_space ~= true) then
					-- get new line, from start_row to last_row
					cv.imwrite { "300dpi_binary/" .. line_dir_name .. "/line_" .. line_idx .. ".jpg", im_bw:sub(start_row, cur_row - 1, 1, width) }
					line_idx = line_idx + 1
				end
				is_last_row_space = true
			else
				if (is_last_row_space) then
					start_row = cur_row
				end
				is_last_row_space = false
			end
		end
		if (is_last_row_space ~= true) then
			cv.imwrite { "300dpi_binary/" .. line_dir_name .. "/line_" .. line_idx .. ".jpg", im_bw:sub(start_row, height, 1, width) }
		end

		-- special symbol extraction (fraction, radical)


		-- char or sequence of chars recgnition


		-- assemble as a whole


		-- judge the result
	end
end
