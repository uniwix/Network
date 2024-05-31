// Network.hpp - Contains declarations of Network class
#pragma once


#include <random>
#include <vector>

#ifdef NETWORK_EXPORTS
#define NETWORK_API __declspec(dllexport)
#else
#define NETWORK_API __declspec(dllimport)
#endif


namespace NetNeurons
{
	template <typename T>
	struct vector_wrap
	{
		std::vector<T> data;
	};

	class Network
	{
	public:
		/**
		 * \brief Create an empty neuron network
		 */
		NETWORK_API Network();

		/**
		 * \brief Create a neuron network with the given layers
		 * \param layers The number of neurons in each layer
		 */
		NETWORK_API Network(const std::vector<int>& layers);
		
		/**
		 * \brief Constructor that creates a Network from an existing linking
		 * \param data the input network
		 */
		NETWORK_API Network(std::vector<std::vector<std::vector<float>>> data);

		NETWORK_API Network(std::istream& is);

		//NETWORK_API Network(const std::vector<int>* layers);
		//NETWORK_API Network(const std::vector<int>::const_iterator begin, const std::vector<int>::const_iterator end);
		//NETWORK_API Network(int params...);

		/**
		 * \brief Create a neuron network with the given layers and recurrent connections
		 * \param layers The number of neurons in each layer
		 * \param rec The recurrent connections
		 */
		NETWORK_API Network(const std::vector<int>& layers, const std::vector<bool>& rec);

		/**
		 * \brief Constructor that creates a Network from an existing linking
		 * \param data the input network
		 */
		NETWORK_API Network(std::vector<std::vector<std::vector<float>>> data, const std::vector<bool>& rec);

		NETWORK_API Network(std::istream& is, const std::vector<bool>& rec);

		NETWORK_API Network(int n_inputs, int n_hidden, int n_wires, int n_outputs);

		/**
		 * \brief Compute the result of the network for the given inputs
		 * \param input Vector containing the inputs values
		 * \return The processed inputs by the network
		 */
		NETWORK_API std::vector<float> compute(const std::vector<float>& input);



		/**
		 * \brief creates a copy of the network
		 * \return the current network
		 */
		NETWORK_API std::vector<std::vector<std::vector<float>>> get_layers() const;

		NETWORK_API std::vector<bool> get_rec() const;


		/**
		 * \brief Mutate the network
		 * \param mutation_rate The mutation rate between 0 and 1 (default: 1%)
		 */
		NETWORK_API void mutate(float mutation_rate = .01f);

		/**
		* \brief Cross the network with another one
		* \param other The other network
		* \return The crossed network
		*/
		NETWORK_API Network operator*(const Network& other) const;

		/**
		 * \brief Create a child network from two parents
		 * \param parent1 The first parent
		 * \param parent2 The second parent
		 * \param mutation_rate The mutation rate between 0 and 1 (default: 1%)
		 * \return The child network
		 */
		NETWORK_API static Network reproduce(const Network& parent1, const Network& parent2, float mutation_rate = .01f);

		NETWORK_API std::ostream& print_to(std::ostream& os) const;

		NETWORK_API void serialize(std::ostream& os) const;

		NETWORK_API std::vector<int8_t> serialize() const;

		NETWORK_API static std::vector<int8_t> serialize(const std::vector<std::vector<std::vector<float>>>& data);

		NETWORK_API static std::vector<std::vector<std::vector<float>>> deserialize(std::istream& is);

		NETWORK_API static std::vector<std::vector<std::vector<float>>> deserialize(const std::vector<int8_t>& serialized);

		NETWORK_API float train(const std::vector<float>& inputs, const std::vector<float>& expected_outputs, float epsilon = 0.1f);

		NETWORK_API static std::vector<bool> decode(int rec, int n_layers);

		NETWORK_API static int encode(const std::vector<bool>& rec);

		NETWORK_API void wire_fit(const std::vector<float>& xt, const std::vector<float>& xt1, float R, const std::vector<std::vector<float>>& wires, const std::vector<float>& q_values, int imax, float alpha, float gamma, float epsilon, float c);

		NETWORK_API int getWireCount() const { return m_n_wires_; };
		NETWORK_API int getSortiesCount() const { return m_n_sorties_; };
		NETWORK_API int getControlsCount() const { return m_n_controls_; };
		
	private:
		/**
		 * \brief Set the weights of a layer
		 * \param weights Vector containing the weights
		 */
		static void set_weights(std::vector<float>& weights);

		/**
		 * \brief Compute the result of a single layer of the network for the given inputs
		 * \param inputs Vector containing the inputs values
		 * \param layer Layer to compute
		 * \return The processed inputs by the layer
		 */
		// static std::vector<float> compute_one_layer(const std::vector<float>& inputs, const std::vector<std::vector<float>>& layer);

		/**
		 * \brief Vector containing the layers of the network
		 * \note The first layer is the input layer, the last one is the output layer
		 * \note The first dimension is the layer, the second is the neuron, the third is the weight of the connection from the previous layer
		 */
		std::vector<std::vector<std::vector<float>>> m_layers_;
		std::vector<std::vector<float>> m_neurons_previous_state_;
		std::vector<bool> m_rec_;

		int m_n_wires_ = 10;
		int m_n_sorties_ = 40;
		int m_n_controls_ = 3;
	};
}


float distance(const std::vector<float>& u, const std::vector<float>& ui, float qi, float qmax, float c, float epsilon);

float wsum(const std::vector<float>& distances, const std::vector<float>& q);

float norm(const std::vector<float>& distances);

float diffQ(float qk, float dist, float wsum_, float norm_, float c);

float diffQ(float ukj, float uj, float qk, float dist, float wsum_, float norm_);

