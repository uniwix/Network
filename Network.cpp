#include "pch.h"
#include "Network.hpp"

#include <stdexcept>

// Initialize the seed
unsigned int seed(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
// Initialize the random distribution
std::uniform_real_distribution<double> distribution(-1., 1.);
// Initialize the random generator
std::default_random_engine generator(seed);

NetNeurons::Network::Network() = default;

NetNeurons::Network::Network(const std::vector<int>& layers)
{
	// Initialize the network
	const std::vector<double>::size_type n_layers = layers.size();
	m_layers_ = std::vector<std::vector<std::vector<double>>>(n_layers - 1);

	// Initialize the layers
	for (size_t i = 0; i < n_layers - 1; i++)
	{
		const std::vector<std::vector<double>>::size_type first_layer_size = layers[i];
		const std::vector<double>::size_type second_layer_size = layers[i + 1];
		m_layers_[i] = std::vector<std::vector<double>>(first_layer_size, std::vector<double>(second_layer_size));
	}

	// Initialize the weights
	for (auto& layer : m_layers_)
	{
		for (auto& weights : layer)
		{
			set_weights(weights);
		}
	}
}

void NetNeurons::Network::set_weights(std::vector<double>& weights)
{

	for (auto& weight : weights)
	{
		weight = distribution(generator);
	}
}

NetNeurons::Network::Network(const std::vector<std::vector<std::vector<double>>>& data) : m_layers_(data) {}

std::vector<std::vector<std::vector<double>>> NetNeurons::Network::getNetwork()
{
	return m_layers_;
}

std::vector<double> NetNeurons::Network::compute_one_layer(const std::vector<double>& inputs, const std::vector<std::vector<double>>& layer)
{
	// Check if the inputs have a valid size for the layer
	if (inputs.size() != layer.size())
	{
		throw std::invalid_argument("The inputs has a invalid size for the layer");
	}

	// Initialize the outputs
	std::vector<double> outputs(layer[0].size(), 0.);

	// Compute the outputs
	for (size_t i = 0; i < inputs.size(); i++)
	{
		for (size_t j = 0; j < outputs.size(); j++)
		{
			outputs[j] += inputs[i] * layer[i][j];
		}
	}

	for (auto& output : outputs)
	{
		output = tanh(output);
	}

	return outputs;
}

std::vector<double> NetNeurons::Network::compute(std::vector<double> input) const
{
	for (auto& layer : m_layers_)
	{
		input = compute_one_layer(input, layer);
	}
	return input;
}
