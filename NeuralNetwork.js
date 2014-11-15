/**
 * Copyright 2014 Martin Vyšňovský (martinvysnovsky@gmail.com)
 */

'use strict';

/**
 * Neural network object
 */
function NeuralNetwork()
{
	// layers in network
	var layers = [];

	/**
	 * Method to add layer to network
	 *
	 * @param  {Layer}  layer  Layer to add
	 */
	this.addLayer = function(layer)
	{
		if(!(layer instanceof Layer))
			throw new Error('Argument must be instance of Layer object.');

		layer.network = this;

		// set layer position in network
		var last_layer = (layers.length > 0) ? layers[layers.length - 1] : null;
		layer.prevLayer = last_layer;
		layer.nextLayer = null;

		// fix last layer nextLayer variable
		if(last_layer)
			last_layer.nextLayer = layer;

		layers.push(layer);
	};

	/**
	 * Method to get all layers in network
	 *
	 * @return  {array}
	 */
	this.getLayers = function()
	{
		return layers;
	};
}

NeuralNetwork.prototype = {
	constructor: NeuralNetwork,

	/**
	 * Method to initialize all neurons in network
	 *
	 * @param   {function}  initFunction  Function to initialize value of neuron
	 *
	 * @return  {void}
	 */
	initialize: function(initFunction)
	{
		if(typeof(initFunction) != 'function')
			throw new Error('Argument must be valid function.');

		var layers = this.getLayers();

		// for all layers except input layer
		for(var i=1, len=layers.length; i<len; i++)
		{
			var neurons = layers[i].getNeurons();

			// for all neuron in layer
			for(var j=0, len2=neurons.length; j<len2; j++)
			{
				var inputConnections = neurons[j].getInputConnections();

				// for all input connections to neuron
				for(var k=0, len3=inputConnections.length; k<len3; k++)
				{
					inputConnections[k].weight = initFunction();
				}
			}
		}
	},

	/**
	 * Method to get output from NN
	 *
	 * @param   {array}  input  Input data
	 *
	 * @return  {array}         Output data from NN
	 */
	getOutput: function(input)
	{
		var layers = this.getLayers();

		// set input layer
		var input_neurons = layers[0].getNeurons();

		if(input_neurons.length != input.length)
			throw new Error('Input items does not match input neurons count.');

		for(var i=0, len=input_neurons.length; i<len; i++)
		{
			var input_value = input[i];

			if(input_value < -1 || input_value > 1)
				throw new Error('Input value must be in range from -1 to 1.');

			input_neurons[i].output = input_value;
		}

		var output = [];

		// compute output on another layers
		for(i=1, len=layers.length; i<len; i++)
			output = layers[i].computeOutput();

		return output;
	},

	/**
	 * Error backpropagation learning
	 *
	 * @param   {array}  input   Array of input data
	 * @param   {array}  target  Array of target output data
	 * @param   {float}  gamma   Learning parameter
	 *
	 * @return  {void}
	 */
	backprop: function(input, target, gamma)
	{
		var layers        = this.getLayers();
		var layers_length = layers.length;

		var outputLayer   = layers[layers_length - 1];
		var outputNeurons = outputLayer.getNeurons();

		var output_length = outputNeurons.length;

		if(target.length != output_length)
			throw new Error('Target count must be same as output neurons count.');

		// get output from NN
		this.getOutput(input);

		// compute output errors
		for(var i=0; i<output_length; i++)
			outputNeurons[i].computeOutputError(target[i]);

		var adaptWeightsFunction = function(connection)
		{
			return connection.weight + (gamma * connection.toNeuron.error * connection.fromNeuron.output);
		};

		// adapt weights for output layer
		outputLayer.adaptWeights(adaptWeightsFunction);

		for(i=layers_length-2; i>=0; i--)
		{
			var layer   = layers[i];
			var neurons = layer.getNeurons();

			for(var j=0, len=neurons.length; j<len; j++)
			{
				neurons[j].computeError();
			}

			// adapt weights for layer
			layer.adaptWeights(adaptWeightsFunction);
		}
	}
};

/**
 * Layer object for neural network
 */
function Layer()
{
	// neurons in layer
	var neurons = [];

	// back refrence to network
	this.network = null;

	// position in network
	this.prevLayer = null;
	this.nextLayer = null;

	/**
	 * Method to add neuron to layer
	 *
	 * @param  {neuron}  layer  Neuron to add
	 */
	this.addNeuron = function(neuron)
	{
		if(!(neuron instanceof Neuron))
			throw new Error('Argument must be instance of Neuron object.');

		neurons.push(neuron);
	};

	/**
	 * Method to get all neurons in layer
	 *
	 * @return  {array}
	 */
	this.getNeurons = function()
	{
		return neurons;
	};
}

Layer.prototype = {
	constructor: Layer,

	/**
	 * Method to add multiple neurons to layer
	 *
	 * @param  {object}  type        Neuron constructor object name
	 * @param  {int}     n           number of neurons to add
	 * @param  {string}  connection  Method to connect neurons
	 */
	addNeurons: function(Type, n, connection)
	{
		if(typeof Type !== 'function')
			throw new Error('Type must be valid Neuron object.');

		connection = connection || 'full';

		var connections = [];

		if(connection == 'full')
		{
			if(this.prevLayer)
				connections = this.prevLayer.getNeurons();
		}

		for(var i=0; i<n; i++)
		{
			var neuron = new Type();

			// connect neuron
			neuron.addInputConnections(connections);

			this.addNeuron(neuron);
		}
	},

	/**
	 * Method to compute output on every neuron in layer
	 *
	 * @return  {array}
	 */
	computeOutput: function()
	{
		var neurons = this.getNeurons();

		var output = [];
		for(var i=0, len=neurons.length; i<len; i++)
		{
			var neuron = neurons[i];

			var neuron_output = neuron.computeOutput(neuron.getInput());

			output.push(neuron_output);
		}

		return output;
	},

	/**
	 * Method to adapt weights for all neurons in layer
	 *
	 * @param   {function}  adaptFunction  Function to adapt weights
	 *
	 * @return  {void}
	 */
	adaptWeights: function(adaptFunction)
	{
		var neurons = this.getNeurons();

		// for all neurons in layer
		for(var i=0, len=neurons.length; i<len; i++)
		{
			var neuron            = neurons[i];
			var input_connections = neuron.getInputConnections();

			// for all input connections to neuron
			for(var j=0, len2=input_connections.length; j<len2; j++)
			{
				var connection = input_connections[j];

				// adapt weight on connection
				connection.weight = adaptFunction(connection);
			}
		}
	}
};

/**
 * Neuron object for neural network
 */
function Neuron()
{
	var inputConnections  = []; // array of input connections
	var outputConnections = []; // array of output connection

	// input for neuron
	this.input = null;

	// output for neuron
	this.output = null;

	// error on neuron
	this.error = 0;

	/**
	 * Method to insert input connections
	 *
	 * @param  {array}  connections  Array of neurons to connect
	 */
	this.addInputConnections = function(connections)
	{
		if(connections instanceof Connection)
		{
			inputConnections.push(connections);
			return;
		}

		if(Object.prototype.toString.call(connections) !== '[object Array]')
			connections = [connections];

		for(var i=0, len=connections.length; i<len; i++)
		{
			var connected_neuron = connections[i];

			// check if connected neuron is Neuron object
			if(!(connected_neuron instanceof Neuron))
				throw new Error('Connection must be to Neuron object.');

			// add connection
			var connection = new Connection(connected_neuron, this);
			inputConnections.push(connection);

			// set output connection for connected neuron
			connected_neuron.addOutputConnections(connection);
		}
	};

	/**
	 * Method to get input connections
	 *
	 * @return  {array}
	 */
	this.getInputConnections = function()
	{
		return inputConnections;
	};

	/**
	 * Method to insert output connections
	 *
	 * @param  {array}  connections  Array of neurons to connect
	 */
	this.addOutputConnections = function(connections)
	{
		if(connections instanceof Connection)
		{
			outputConnections.push(connections);
			return;
		}

		if(Object.prototype.toString.call(connections) !== '[object Array]')
			connections = [connections];

		for(var i=0, len=connections.length; i<len; i++)
		{
			var connected_neuron = connections[i];

			// check if connected neuron is Neuron object
			if(!(connected_neuron instanceof Neuron))
				throw new Error('Connection must be to Neuron object.');

			// add connection
			var connection = new Connection(this, connected_neuron);
			outputConnections.push(connection);

			// set output connection for connected neuron
			connected_neuron.addInputConnections(connection);
		}
	};

	/**
	 * Method to get output connections
	 *
	 * @return  {array}
	 */
	this.getOutputConnections = function()
	{
		return outputConnections;
	};
}

Neuron.prototype = {
	constructor: Neuron,

	/**
	 * Method to check if exists input connection from given neuron
	 *
	 * @param   {object}   connected_neuron  Input neuron to check
	 *
	 * @return  {Boolean}
	 */
	hasInputConnection: function(connected_neuron)
	{
		var input_connections = this.getInputConnections();

		for(var i=0, len=input_connections.length; i<len; i++)
		{
			if(input_connections[i].fromNeuron == connected_neuron)
				return true;
		}

		return false;
	},

	/**
	 * Method to check if exists output connection from given neuron
	 *
	 * @param   {object}   connected_neuron  Output neuron to check
	 *
	 * @return  {Boolean}
	 */
	hasOutputConnection: function(connected_neuron)
	{
		var output_connections = this.getOutputConnections();

		for(var i=0, len=output_connections.length; i<len; i++)
		{
			if(output_connections[i].toNeuron == connected_neuron)
				return true;
		}

		return false;
	},

	/**
	 * Method to calculate input to neuron
	 *
	 * @return  {float}
	 */
	getInput: function()
	{
		this.input = this.getInputConnections().reduce(function(sum, connection)
		{
			return sum + (connection.weight * connection.fromNeuron.output);
		}, 0);

		return this.input;
	},

	/**
	 * Method to compute and set output from neuron
	 *
	 * @param   {float}  input  Input to neuron
	 *
	 * @return  {float}         Output prom neuron
	 */
	computeOutput: function(input)
	{
		this.output = input;

		return this.output;
	},

	/**
	 * Method to compute output error for given target value
	 *
	 * @param   {float}  target  Target value
	 *
	 * @return  {float}          Error
	 */
	computeOutputError: function(target)
	{
		this.error = (target - this.output);

		return this.error;
	},

	/**
	 * Method to compute error on neuron
	 *
	 * @return  {void}
	 */
	computeError: function()
	{
		this.error = this.getOutputConnections().reduce(function(sum, connection)
		{
			return sum + (connection.weight * connection.toNeuron.error);
		}, 0);

		return this.error;
	}
};

/**
 * Neuron with threshold activation funcion
 */
function ThresholdNeuron(options)
{
	Neuron.call(this);

	options = options || {};

	this.threshold = options.threshold || 0;
}

ThresholdNeuron.prototype = Object.create(Neuron.prototype);

/**
 * Method to compute and set output from neuron
 *
 * @param   {float}  input  Input to neuron
 *
 * @return  {float}  Output from neuron
 */
ThresholdNeuron.prototype.computeOutput = function(input)
{
	this.output = (input > this.threshold) ? 1 : 0;

	return this.output;
};

/**
 * Neuron with sigmoid activation funcion
 */
function SigmoidNeuron(options)
{
	Neuron.call(this);

	options = options || {};
}

SigmoidNeuron.prototype = Object.create(Neuron.prototype);

/**
 * Method to compute and set output from neuron
 *
 * @param   {float}  input  Input to neuron
 *
 * @return  {float}  Output from neuron
 */
SigmoidNeuron.prototype.computeOutput = function(input)
{
	this.output = (1 / (1 + Math.exp(- input)));

	return this.output;
};

/**
 * Method to compute output error for given target value
 *
 * @param   {float}  target  Target value
 *
 * @return  {float}          Error
 */
SigmoidNeuron.prototype.computeOutputError = function(target)
{
	// call parent method
	Neuron.prototype.computeOutputError.call(this, target);

	return this.output * (1 - this.output) * this.error;
};

/**
 * Method to compute error on neuron
 *
 * @return  {void}
 */
SigmoidNeuron.prototype.computeError = function()
{
	// call parent method
	Neuron.prototype.computeError.call(this);

	this.error = this.output * (1 - this.output) * this.error;

	return this.error;
};

/**
 * Connection object between neurons
 *
 * @param  {Neuron}  fromNeuron  Neuron where connection starts
 * @param  {Neuron}  toNeuron    Neuron where connection ends
 */
function Connection(fromNeuron, toNeuron)
{
	if(!(fromNeuron instanceof Neuron) || !(toNeuron instanceof Neuron))
		throw new Error('Arguments must be instance of Neuron object.');

	this.fromNeuron = fromNeuron;
	this.toNeuron   = toNeuron;

	this.weight = 0;
}