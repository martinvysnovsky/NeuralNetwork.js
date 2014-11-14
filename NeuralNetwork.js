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
	}
};

/**
 * Neuron object for neural network
 */
function Neuron()
{
	var inputConnections  = []; // array of input connections
	var outputConnections = []; // array of output connection

	// output for neuron
	this.output = null;

	/**
	 * Method to insert input connections
	 *
	 * @param  {array}  connections  Array of neurons to connect
	 */
	this.addInputConnections = function(connections)
	{
		if(Object.prototype.toString.call(connections) !== '[object Array]')
			connections = [connections];

		for(var i=0, len=connections.length; i<len; i++)
		{
			var connected_neuron = connections[i];

			// check if connected neuron is Neuron object
			if(!(connected_neuron instanceof Neuron))
				throw new Error('Connection must be to Neuron object.');

			// add connection
			inputConnections.push({
				neuron: connected_neuron,
				weight: 0
			});

			// set output connection for connected neuron
			if(!connected_neuron.hasOutputConnection(this))
				connected_neuron.addOutputConnections([this]);
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
		if(Object.prototype.toString.call(connections) !== '[object Array]')
			connections = [connections];

		for(var i=0, len=connections.length; i<len; i++)
		{
			var connected_neuron = connections[i];

			// check if connected neuron is Neuron object
			if(!(connected_neuron instanceof Neuron))
				throw new Error('Connection must be to Neuron object.');

			// add connection
			outputConnections.push({
				neuron: connected_neuron,
				weight: 0
			});

			// set output connection for connected neuron
			if(!connected_neuron.hasInputConnection(this))
				connected_neuron.addInputConnections([this]);
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
			if(input_connections[i].neuron == connected_neuron)
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
			if(output_connections[i].neuron == connected_neuron)
				return true;
		}

		return false;
	}
};

/**
 * Neuron with treshold activation funcion
 */
function TresholdNeuron()
{
	Neuron.call(this);
}

TresholdNeuron.prototype = Object.create(Neuron.prototype);