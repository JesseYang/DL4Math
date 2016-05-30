local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'binarize/main'

local app = require('waffle')

app.post('/upload', function(req, res)
	-- 1. convert to ByteTensor
	local img = req.form.file:toImage()
	img = (img * 255):byte()
	img = img:transpose(1, 2):transpose(2, 3)
	-- 2. convert to binary image
	cv.cvtColor{img, img, cv.COLOR_BGR2GRAY}
	img = img:transpose(2,3):transpose(1,2):sub(1,1,1,-1,1,-1)[1]
	img = binarize(img)
	print(img)
	-- 3. extract lines
	-- 4. scale and padding to 80 pixels high
	-- 5 recognize
	res.send("hello")
end)

app.get('/upload', function(req, res)
	res.render('upload.html', { })
end)

app.listen({host="10.50.101.163", port="8080"})

