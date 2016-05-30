local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'lfs'
require './util'
require 'image'

dpi = "150dpi"

function binarize_all()
	lfs.mkdir(dpi .. "_binary")

	for img_file in lfs.dir(dpi) do
		if (img_file ~= "." and img_file ~= "..") then
			print(img_file)

			-- binarization
			local im_gray = cv.imread { dpi .. "/" .. img_file, cv.IMREAD_GRAYSCALE }
			if (img_file:len() > 13) then
				img_file = img_file:sub(7)
			end
			-- local im_bw = cv.adaptiveThreshold{im_gray, im_bw, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 21}
			local im_bw = binarize(im_gray)
			cv.imwrite { dpi .. "_binary/" .. img_file, im_bw }
		end
	end
end


function binarize(im_gray)
	local im_bw = cv.adaptiveThreshold{im_gray, im_bw, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 21}
	return im_bw
end
