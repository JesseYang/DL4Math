-- require 'main'
local app = require('waffle')
local http = require('socket.http')
local ltn12 = require("ltn12")

app.post('/upload', function(req, res)
	-- local img = req.form.file:toImage()
	local img_file = req.form.file
	img_file:save()
	print(img_file)
	-- print(torch.type(img_file.data))
	-- img = (img * 255):byte()
	-- img = img:transpose(1, 2):transpose(2, 3)
	-- local ret = recognize(img)
	-- res.send(ret)
	local body,code,headers,status = http.request{
		url = "http://10.8.0.4:8080",
		method = "POST",
		headers = {
			["Content-Type"] = "application/x-www-form-urlencoded"-- ,
			-- ["Content-Length"] = file_size
		},
		file = img_file,
		file = ltn12.source.file(io.open(img_file.filename),"rb"),
		sink = ltn12.sink.table(resp)
	}
	res.send("hello")
end)

app.post('/do', function(req, res)
	local img = req.form.file:toImage()
	img = (img * 255):byte()
	img = img:transpose(1, 2):transpose(2, 3)
	res.send("hello")
end)

app.get('/upload', function(req, res)
	res.render('upload.html', { })
end)

app.listen({host="10.50.101.163", port="8080"})

