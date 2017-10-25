var num_inputs = 160; //16 cells * 10 possible values
var num_actions = 4;
var temporal_window = 0;
var network_size = num_inputs*temporal_window + num_actions*temporal_window + num_inputs;

var layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:network_size});

//Normal NN
layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});

//ConvNet
//8 3x3 filters
//layer_defs.push({type:'conv', sx:3, filters:8, stride:1, activation:'relu'});
//layer_defs.push({type:'pool', sx:2, stride: 2});
//layer_defs.push({type:'conv', sx:2, filters: 16, stride: 1, activation:'relu'});
//layer_defs.push({type:'pool', sx:3, stride: 2});

layer_defs.push({type:'regression', num_neurons:num_actions});

var tdtrainer_options = {learning_rate:0.005, momentum:0.2, batch_size:64, l2_decay:0.02};

var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 60000;
opt.start_learn_threshold = 1000;
opt.gamma = 0.75
opt.learning_steps_total = 100000;
opt.learning_steps_burnin = 1000;
opt.epsilon_max = 1.00;
opt.epsilon_min = 0.01;
opt.epsilon_test_time =0.001;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;

var brain = new deepqlearn.Brain(num_inputs, num_actions, opt);

var Agent = function() {
	this.brain = brain;
	this.reward_bonus = 0.0;
	this.observing = false
}
function rescale(x) {
	if(x==0) return 0;
	return Math.log2(x);
	//return invert(x);
	// /return invert(Math.log2(x)+1);
}
function invert(x) {
	//converts arbitrary value to (0-1)
	//skews large numbers together, but separates out small numbers)
	//consider log2 to squash large numbers
	if(x==0) return 0;
	return 1+ (-1/x);
}
Agent.prototype = {
	buildInput: function() {
		var input_array = new Array(num_inputs);
		input_array.fill(0);
		manager.grid.eachCell(function(x, y, tile) {
			if(tile) {
				var value = rescale(tile.value);
				if(value != 0) {
					input_array[(value-1)*16 + x*4 + y] = 1;
				}
			}
		});
		return input_array;
	},
	forward: function() {
		var input_array = this.buildInput();
		var action = this.brain.forward(input_array);
		this.input = input_array.toString();
		this.action = action;
	},
	backwards: function() {
		var terminal = false;
		//var reward = rescale(this.new_score - this.old_score);
		var reward = this.new_score - this.old_score;
		if(reward != 0) reward = Math.log2(reward);
		// var reward = Math.max(Math.log2(this.new_score), 0);
		//var reward = 1;
		// 100,000 is approx the final score when you win
		//var reward = this.new_score / 10000;
		//manager.grid.eachCell(function(x, y, tile) {
		//	if(tile && tile.value > reward) {
		//		reward = tile.value	;
		//	}
		//});
		if(manager.over || manager.won) {
			terminal = true;
			draw_scores(manager.score);
			if(manager.score > high_score) {
				high_score = manager.score;
				reward = reward + 100;
				console.log("High Score! "+high_score);
			}
		}
		if(manager.over) {
			manager.restart();
			console.log("game over");
			reward = -50;
			//reward = reward - 500;
		}
		if(manager.won) {
			manager.restart();
			console.log("WON!!!!!");
			reward = reward + 1000;
		}
		//penalize repeat move
		if(this.input == this.prev_input) reward = -1;
		//if(this.input === this.prev_input) reward = 0;
		this.prev_input = this.input;
		//penalize making too many moves
		//reward = reward - 0.25;
		this.brain.backward(reward, terminal);
	},
	tick: function() {
		this.old_score = manager.score;
		this.forward();
		manager.inputManager.emit("move", this.action);
		this.new_score = manager.score;
		this.backwards();
	},

	addObserver: function() {
		if(this.observing === true) return; 
		var self = this
		manager.inputManager.on("move", function(direction) {
			if(observing) {
				self.action = direction
				self.new_score = manager.score;
				self.brain.observation(self.input_array, self.action);
				self.backwards();
				brain.visSelf(document.getElementById("brain_stats"));
				requestAnimationFrame(runAI);
			}
		});
		this.observing = true;
	},

	readyObservation: function(direction) {
		this.addObserver();
		this.old_score = manager.score;
		this.input_array = this.buildInput()
		this.input = this.input_array.toString();
	},

	savebrain: function() {
		var j = this.brain.value_net.toJSON();
		var t = JSON.stringify(j);
		return t
	},

	loadbrain: function(t) {
		var j = JSON.parse(t);
		this.brain.value_net.fromJSON(j);
		this.brain.learning=false; 
	}
}
var reward_graph = new cnnvis.Graph();
var loss_graph = new cnnvis.Graph();
var score_graph = new cnnvis.Graph();
function draw_stats() { 
	if(clock % 20 === 0) {
		reward_graph.add(clock/20, brain.average_reward_window.get_average());
		loss_graph.add(clock/20, brain.average_loss_window.get_average());
		var gcanvas = document.getElementById("graph_canvas");
		reward_graph.drawSelf(gcanvas);
		var lcanvas = document.getElementById("loss_canvas");
		loss_graph.drawSelf(lcanvas);
		var avg = 0;
		score_graph.pts.forEach(function(x) { avg += x.y/score_graph.pts.length; });
		document.getElementById("avg_score").textContent = "average score: "+avg;
	}
	brain.visSelf(document.getElementById("brain_stats"));

}
function draw_scores(score) {
	games++;
	score_graph.add(games, score);
	var scanvas = document.getElementById("score_canvas");
	score_graph.drawSelf(scanvas);
}
function draw_net() {
    if(!slow && clock % 10 !== 0) return;  // do this sparingly
	var canvas = document.getElementById("net_canvas");
	var ctx = canvas.getContext("2d");
	var W = canvas.width;
	var H = canvas.height;
	ctx.clearRect(0, 0, W, H);
	var L = brain.value_net.layers;
	var dx = (W - 50)/L.length;
	var x = 10;
	var y = 40;
	ctx.font="12px Verdana";
	ctx.fillStyle = "rgb(0,0,0)";
	ctx.fillText("Value Function Approximating Neural Network:", 10, 14);
	for(var k=0;k<L.length;k++) {
	    if(typeof(L[k].out_act)==='undefined') continue; // maybe not yet ready
	    var kw = L[k].out_act.w;
	    var n = kw.length;
	    var dy = (H-50)/n;
	    ctx.fillStyle = "rgb(0,0,0)";
	    ctx.fillText(L[k].layer_type + "(" + n + ")", x, 35);
	    var min_max = max_min_element(kw);
	    for(var q=0;q<n;q++) {
			var v = kw[q]
			if(v >= 0) {
				if(v == min_max[1]) {
					ctx.fillStyle = "rgb(0,256,0)";
				} else {
					v = Math.floor((v/min_max[1]) * 256);
					ctx.fillStyle = "rgb(0,0," + v + ")";
				}
			} 
			if(v < 0) {
				v = Math.floor((v/min_max[0]) * 256);
				ctx.fillStyle = "rgb(" + (v) + ",0,0)";
			}
			ctx.fillRect(x,y,10,10);
			y += 12;
			if(y>H-25) { y = 40; x += 12};
		}
		x += 50;
		y = 40;
	}
}
function max_min_element(array) {
	var max = -Infinity;
	var min = Infinity;
	var n=array.length;
	for(var i=0; i<n;i++) {
		if(array[i] < min) {
			min = array[i];
		}
		if(array[i] > max) {
			max = array[i];
		}
	}
	return [min, max];
}
function resetGraph() {
	clock = 0;
	games = 0;
	reward_graph = new cnnvis.Graph();
	loss_graph = new cnnvis.Graph();
	score_graph = new cnnvis.Graph();
}
function toggleTraining() {
	agent.brain.learning = !agent.brain.learning
}
function toggleRunning() {
	running = !running;
	if(running) {
		runAI();
	}
}
function toggleSpeed() {
	slow = !slow
}
function exportNet() {
	var output = document.getElementById("pretrained");
	var text = agent.savebrain();
	output.value = text;
}
function importNet() {
	var output = document.getElementById("pretrained");
	var text = output.value;
	agent.loadbrain(text);
}

function storeNet() {
	try {
		localStorage.setItem("brain", agent.savebrain())
		localStorage.setItem("clock", clock)
		localStorage.setItem("age", agent.brain.age)
		localStorage.setItem("learning", agent.brain.learning)
	} catch (e) {

	}
}
function restoreNet() {
	try {
		if(localStorage.getItem("brain")) { 
			console.log("restoring brain")
			agent.loadbrain(localStorage.getItem("brain"));
		}
		if(localStorage.getItem("age")) { agent.brain.age = parseInt(localStorage.getItem("age")); }
		if(localStorage.getItem("learning")) { agent.brain.learning = JSON.parse(localStorage.getItem("learning")); }
	} catch (e) {

	}
}
function resetLocalStore() {
	try {
		localStorage.removeItem("brain")
		localStorage.removeItem("clock")
		localStorage.removeItem("age")
		localStorage.removeItem("learning")
	} catch (e) {

	}
}
function loadOpts() {
	try {
		if(localStorage.getItem("clock")) { clock = parseInt(localStorage.getItem("clock")); }
	} catch (e) {
		
	}
}

var slow = false;
//var timeout = 10;
var running = false;
var clock = 0;
var games = 0;
var high_score = 0;
var observing = true;
var observing_steps = 0;
function runAI() {
	clock++;
	if(clock < observing_steps) {
		observing = true;
		agent.readyObservation();
	} else {
		if(observing) {
			observing = false;
			running = true;
		}
		agent.tick();
	}
	if(clock % 1000 === 0) {
		storeNet();
	}
	draw_stats();
	//draw_net();
	if(running) {
		if(slow) {
			setTimeout(runAI, 250);
		} else {
			requestAnimationFrame(runAI);
		}
	}
}
document.addEventListener("DOMContentLoaded", function () {
	loadOpts();
	agent = new Agent();
	restoreNet();
	setTimeout(runAI, 1000);
});
