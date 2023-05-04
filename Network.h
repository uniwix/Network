// Network.h - Contains declarations of math functions
#pragma once

#include <chrono>
#include <random>
#include <vector>

#ifdef NETWORK_EXPORTS
#define NETWORK_API __declspec(dllexport)
#else
#define NETWORK_API __declspec(dllimport)
#endif

class NETWORK_API Network
{
public:
	/**
	 * \brief Create an empty neuron network
	 */
	Network();

	/**
	 * \brief Create a neuron network with the given layers
	 * \param layers The number of neurons in each layer
	 */
	Network(const std::vector<int> &layers);

	/**
	 * \brief Compute the result of a single layer of the network for the given inputs
	 * \param input Vector containing the inputs values
	 * \param layer Layer to compute
	 * \return The processed inputs by the layer
	 */
	static std::vector<double> compute_one_layer(const std::vector<double>& input, const std::vector<std::vector<double>>& layer);

	/**
	 * \brief Compute the result of the network for the given inputs
	 * \param input Vector containing the inputs values
	 * \return The processed inputs by the network
	 */
	std::vector<double> compute(std::vector<double> input) const;

	/**
	 * \brief Set the weights of a layer
	 * \param weights Vector containing the weights
	 */
	static void set_weights(std::vector<double> &weights);

private:
	/**
	 * \brief Vector containing the layers of the network
	 */
	std::vector<std::vector<std::vector<double>>> m_layers_;
};



// Initialize the random distribution

unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
std::default_random_engine generator(seed); // Random numbers generator
std::uniform_real_distribution<double> distribution(0.0, 1.0);