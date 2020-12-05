
const express = require("express");
const app = express();
const spawnSync = require("child_process").spawnSync;

app.use(function(req, res, next) {
	res.header("Access-Control-Allow-Origin", "*");
	res.header('Access-Control-Allow-Methods', 'DELETE, PUT');
	res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
	if ('OPTIONS' == req.method) {
	   res.sendStatus(200);
	}
	else {
	   next();
	}
});	

app.get("/", (req, res, next) => {
	var input = req.query.input;
	var clusternumber = req.query.clusternumber;
	var clusterData, posData;
	var reply = {};
	console.log("Input   :  " + input + "\n\n" + "Number of clusters : " + clusternumber + "\n\n");
	
	const proPOS = spawnSync('python', ["./postagger.py", input], { encoding : 'utf8' });
	posData = proPOS.output;

	const proCtr = spawnSync('python', ["./cluster.py", input, clusternumber], { encoding : 'utf8' });
	clusterData = proCtr.output;
	
	reply["pos"] = posData;
	reply["cluster"] = clusterData;

	console.log("\npos : " + posData + "\n\n");

	res.json(reply); 
});
 

app.listen(3000, () => {
 	console.log("Server running on port 3000");
});
