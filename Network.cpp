#include "pch.h"
#include "Network.hpp"

#include <assert.h>
#include <chrono>
#include <ostream>
#include <sstream>
#include <stdexcept>

// Initialize the seed
unsigned int seed(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
// Initialize the random distribution
std::uniform_real_distribution<float> distribution(-1.f, 1.f);
// Initialize the random generator
std::default_random_engine generator(seed);

float produit_scalaire(const std::vector<float>& a, const std::vector<float>& b)
{
	assert(a.size() == b.size());
	float out(0.f);
	for (size_t i = 0; i < a.size(); ++i)
	{
		out += a[i] * b[i];
	}
	return out;
}

float produit_scalaire_transpose(const std::vector<float>& a, const std::vector<std::vector<float>>& b, int i)
{
	assert(a.size() == b.size());
	float out(0.f);
	for (size_t j = 0; j < a.size(); ++j)
	{
		out += a[j] * b[j][i];
	}
	return out;
}

std::vector<float> mat_mul(const std::vector<std::vector<float>>& mat, const std::vector<float>& vec)
{
	std::vector<float> out(mat.size());
	for (size_t i = 0; i < mat.size(); ++i)
	{
		out[i] = produit_scalaire(mat[i], vec);
	}
	return out;
}

std::vector<float> mat_mul_tanh(const std::vector<std::vector<float>>& mat, const std::vector<float>& vec)
{
	std::vector<float> out(mat.size());
	for (size_t i = 0; i < mat.size(); ++i)
	{
		out[i] = tanh(produit_scalaire(mat[i], vec));
	}
	return out;
}

float cost(std::vector<float> outputs, std::vector<float> expected_outputs)
{
	assert(outputs.size() == expected_outputs.size());
	float out(0.f);
	for (size_t i = 0; i < outputs.size(); ++i)
	{
		float d = outputs[i] - expected_outputs[i];
		out += d * d;
	}
	return out;
}

NetNeurons::Network::Network() = default;

NetNeurons::Network::Network(std::istream& is)
{
	m_layers_ = deserialize(is);
	size_t n_layers = m_layers_.size();
	m_neurons_previous_state_ = std::vector<std::vector<float>>(n_layers+ 1);
	for (size_t i = 0; i < n_layers + 1; ++i)
	{
		m_neurons_previous_state_[i] = std::vector<float>();
	}
	m_rec_ = std::vector<bool>(n_layers + 1, false);
}

NetNeurons::Network::Network(std::istream& is, const std::vector<bool>& rec)
{
	m_layers_ = deserialize(is);
	m_rec_ = rec;

	size_t n_layers = m_layers_.size();
	m_neurons_previous_state_ = std::vector<std::vector<float>>(n_layers+1);
	m_neurons_previous_state_[0] = std::vector<float>();
	for (size_t i = 1; i < n_layers+1; ++i)
	{
		if (rec[i])
		{
			m_neurons_previous_state_[i] = std::vector<float>(m_layers_[i-1].size());
		}
		else
		{
			m_neurons_previous_state_[i] = std::vector<float>();
		}
	}
}

NetNeurons::Network::Network(int n_inputs, int n_hidden, int n_wires, int n_outputs)
	: Network({ n_inputs, n_hidden, n_wires * (n_outputs + 1) })
{
	m_n_controls_ = n_outputs;
	m_n_wires_ = n_wires;
	m_n_sorties_ = n_wires * (n_outputs + 1);
}

NetNeurons::Network::Network(const std::vector<int>& layers)
	: Network(layers, std::vector<bool>(layers.size(), false))
{}

NetNeurons::Network::Network(const std::vector<int>& layers, const std::vector<bool>& rec)
{
	assert(layers.size() == rec.size());
	m_rec_ = rec;
	std::vector<int> actual_layers(layers.size());
	actual_layers[layers.size()-1] = layers[layers.size()-1];
	for (size_t i = layers.size() - 1; i > 0; --i)
	{
		actual_layers[i - 1] = layers[i - 1] + (rec[i] ? actual_layers[i] : 0);
	}
	// Initialize the network
	const size_t n_layers = actual_layers.size();
	assert(n_layers > 0);
	m_layers_ = std::vector<std::vector<std::vector<float>>>(n_layers - 1);

	// Initialize the layers
	for (size_t i = 0; i < n_layers - 1; i++)
	{
		const size_t nombre_de_neurones_couche_precedente = actual_layers[i];
		const size_t nombre_de_neurone = actual_layers[i + 1];
		m_layers_[i] = std::vector<std::vector<float>>(nombre_de_neurone, std::vector<float>(nombre_de_neurones_couche_precedente));
	}

	m_neurons_previous_state_ = std::vector<std::vector<float>>(n_layers);
	for (size_t i = 0; i < n_layers; ++i)
	{
		if (rec[i])
		{
			m_neurons_previous_state_[i] = std::vector<float>(actual_layers[i]);
		}
		else
		{
			m_neurons_previous_state_[i] = std::vector<float>();
		}
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
/*
NetNeurons::Network::Network(const std::vector<int>* layers)
	: Network(*layers)
{}

NetNeurons::Network::Network(const std::vector<int>::const_iterator begin, const std::vector<int>::const_iterator end)
	: Network(std::vector<int>{begin, end})
{}

NetNeurons::Network::Network(int params...)
	: Network(std::vector<int>{params})
{}*/

void NetNeurons::Network::set_weights(std::vector<float>& weights)
{
	for (auto& weight : weights)
	{
		weight = distribution(generator);
	}
}

NetNeurons::Network::Network(std::vector<std::vector<std::vector<float>>> data, const std::vector<bool>& rec)
{
	m_layers_ = std::move(data);
	m_rec_ = rec;

	size_t n_layers = m_layers_.size();
	m_neurons_previous_state_ = std::vector<std::vector<float>>(n_layers+1);
	m_neurons_previous_state_[0] = std::vector<float>();
	for (size_t i = 1; i < n_layers + 1; ++i)
	{
		if (rec[i])
		{
			m_neurons_previous_state_[i] = std::vector<float>(m_layers_[i-1].size());
		}
		else
		{
			m_neurons_previous_state_[i] = std::vector<float>();
		}
	}
}

NetNeurons::Network::Network(std::vector<std::vector<std::vector<float>>> data)
	: Network(data, std::vector<bool>(data.size() + 1, false))
{}

std::vector<std::vector<std::vector<float>>> NetNeurons::Network::get_layers() const
{
	return m_layers_;
}

NETWORK_API std::vector<bool> NetNeurons::Network::get_rec() const
{
	return m_rec_;
}

void NetNeurons::Network::mutate(float mutation_rate)
{
	for (auto& layer : m_layers_)
	{
		for (auto& weights : layer)
		{
			for (auto& weight : weights)
			{
				if (abs(distribution(generator)) < mutation_rate)
				{
					weight += distribution(generator);
					if (weight > 1.)
					{
						weight -= 2.;
					}
					else if (weight < -1.)
					{
						weight += 2.;
					}
				}
			}
		}
	}
}

NetNeurons::Network NetNeurons::Network::operator*(const Network& other) const
{
	std::vector<std::vector<std::vector<float>>> new_network(m_layers_.size());
	for (size_t i = 0; i < m_layers_.size(); ++i)
	{
		new_network[i] = std::vector<std::vector<float>>(m_layers_[i].size());
		for (size_t j = 0; j < m_layers_[i].size(); ++j)
		{
			new_network[i][j] = std::vector<float>(m_layers_[i][j].size());
			for (size_t k = 0; k < m_layers_[i][j].size(); ++k)
			{
				new_network[i][j][k] = distribution(generator) < 0 ? m_layers_[i][j][k] : other.m_layers_[i][j][k];
			}
		}
	}
	Network network(new_network, m_rec_);
	return network;
}

NetNeurons::Network NetNeurons::Network::reproduce(const Network& parent1, const Network& parent2, float mutation_rate)
{
	Network child = parent1 * parent2;
	child.mutate(mutation_rate);
	return child;
}

std::ostream& NetNeurons::Network::print_to(std::ostream& os) const
{
	const auto old_precision = os.precision();
	os.precision(3);
	for (size_t i = 0; i < m_layers_.size(); ++i)
	{
		os << "layer " << i << ":\n";
		for (auto& layer : m_layers_[i])
		{
			os << "\t";
			for (const float value : layer)
			{
				os << value << "\t";
			}
			os << "\n";
		}
	}
	os.precision(old_precision);
	return os;
}

void NetNeurons::Network::serialize(std::ostream& os) const
{
	for (size_t i = 0; i < m_layers_.size(); ++i)
	{
		for (size_t j = 0; j < m_layers_[i].size(); ++j)
		{
			for (size_t k = 0; k < m_layers_[i][j].size(); ++k)
			{
				os << m_layers_[i][j][k];
				if (k != m_layers_[i][j].size() - 1)
				{
					os << " ";
				}
			}
			if (j != m_layers_[i].size() - 1)
			{
				os << "/";
			}
		}
		if (i != m_layers_.size() - 1)
		{
			os << "*";
		}
	}
	os << "\n";
}

std::vector<std::vector<std::vector<float>>> NetNeurons::Network::deserialize(std::istream& is)
{
	std::vector<std::vector<std::vector<float>>> layers;
	std::string line;
	std::getline(is, line, '\n');
	std::istringstream iss(line);
	std::string str_layer;
	while (std::getline(iss, str_layer, '*'))
	{
		std::istringstream iss_layer(str_layer);
		std::string str_neuron;
		std::vector<std::vector<float>> layer;
		while (std::getline(iss_layer, str_neuron, '/'))
		{
			std::istringstream iss_neuron(str_neuron);
			std::string weight;
			std::vector<float> neuron;
			while (std::getline(iss_neuron, weight, ' '))
			{
				neuron.push_back(std::stof(weight));
			}
			layer.push_back(neuron);
		}
		layers.push_back(layer);
	}
	return layers;
}

float NetNeurons::Network::train(const std::vector<float>& inputs, const std::vector<float>& expected_outputs, float epsilon)
{
	// Check if the inputs have a valid size for the layer
	if (inputs.size() != m_layers_[0][0].size())
	{
		throw std::invalid_argument("The inputs has a invalid size for the layer");
	}
	if (expected_outputs.size() != m_layers_[m_layers_.size() - 1].size())
	{
		throw std::invalid_argument("The expected outputs has a invalid size for the layer");
	}

	std::vector<std::vector<float>> activation_neurones(m_layers_.size()+1);
	std::vector<std::vector<float>> erreur_neurones(m_layers_.size()+1);

	activation_neurones[0] = inputs;  // si i est une cellule d'entr�e: Khi_i=X_i
	
	for (size_t n = 1; n <= m_layers_.size(); ++n)
	{
		activation_neurones[n] = mat_mul_tanh(m_layers_[n - 1], activation_neurones[n-1]);  // sinon X_i = f[A_i] avec A_i = somme(j; W_i,j * X_j)
	}
	for (size_t i = 0; i < m_layers_[m_layers_.size() - 1].size(); ++i)  // si s est une cellule de sortie: Y_s = 2 (Sh_s - Yh_s) * f'(As)
	{
		float tanh_A = tanh(produit_scalaire(activation_neurones[m_layers_.size()-1], m_layers_[m_layers_.size() - 1][i]));
		erreur_neurones[m_layers_.size()].push_back(2 * (activation_neurones[m_layers_.size()][i] - expected_outputs[i]) * (1 - tanh_A * tanh_A));
	}
	for (int n = m_layers_.size(); n > 1; --n)
	{
		std::vector<float> tanh_A_vector = mat_mul_tanh(m_layers_[n-1 - 1], activation_neurones[n-1-1]);
		for (size_t j = 0; j < m_layers_[n - 1][0].size(); ++j)  // sinon: Y_i = f'(A_i) * somme(j; W_j,i * Y_j)
		{
			// float tanh_A = tanh(produit_scalaire(activation_neurones[i-1], m_layers_[i-1 - 1][j]));
			float B = produit_scalaire_transpose(erreur_neurones[n], m_layers_[n - 1], j);
			erreur_neurones[n - 1].push_back((1 - tanh_A_vector[j] * tanh_A_vector[j]) * B);
		}
	}

	for (size_t n = 0; n < m_layers_.size(); ++n)
	{
		for (size_t i = 0; i < m_layers_[n].size(); ++i)
		{
			for (size_t j = 0; j < m_layers_[n][i].size(); ++j)
			{
				m_layers_[n][i][j] -= epsilon * activation_neurones[n][j] * erreur_neurones[n + 1][i];
			}
		}
	}
	return cost(activation_neurones[m_layers_.size()], expected_outputs);
}


/*
std::vector<float> NetNeurons::Network::compute_one_layer(const std::vector<float>& inputs, const std::vector<std::vector<float>>& layer)
{
	return mat_mul_tanh(layer, inputs);
}*/

std::vector<float> NetNeurons::Network::compute(const std::vector<float>& inputs)
{
	std::vector<float> computed = inputs;
	for (size_t i = 0 ; i < m_layers_.size(); ++i)
	{
		if (m_rec_[i+1])
		{
			for (size_t j = 0; j < m_neurons_previous_state_[i+1].size(); ++j)
			{
				computed.push_back(m_neurons_previous_state_[i+1][j]);
			}
		}
		computed = mat_mul_tanh(m_layers_[i], computed);
		// TODO: support multi-layer rec: only computed values are stored, not the previous state
		if (m_rec_[i])
		{
			for (size_t j = 0; j < computed.size(); ++j)
			{
				m_neurons_previous_state_[i][j] = computed[j];
			}
		}
		
		
	}
	return computed;
}


int8_t to_int8(float f)
{
	f = f > 1.f ? 1.f : f;
	f = f < -1.f ? -1.f : f;
	return static_cast<int8_t>(f * 126);
}

int8_t to_int8(char f)
{
	switch (f)
	{
	case ' ':
		return 0b10000000;
	case '/':
		return 0b10000001;
	case '*':
		return 0b01111111;
	default:  // should not happen
		return 0;
	};
}

float to_float(int8_t i)
{
	return static_cast<float>(i) / 126;
}

std::vector<int8_t> NetNeurons::Network::serialize(const std::vector<std::vector<std::vector<float>>>& data)
{
	std::vector<int8_t> serialized;
	for (size_t i = 0; i < data.size(); ++i)
	{
		for (size_t j = 0; j < data[i].size(); ++j)
		{
			for (size_t k = 0; k < data[i][j].size(); ++k)
			{
				serialized.push_back(to_int8(data[i][j][k]));
			}
			if (j != data[i].size() - 1)
			{
				serialized.push_back(to_int8('/'));
			}
		}
		if (i != data.size() - 1)
		{
			serialized.push_back(to_int8('*'));
		}
	}
	return serialized;
}

std::vector<int8_t> NetNeurons::Network::serialize() const
{
	std::vector<int8_t> serialized;
	for (size_t i = 0; i < m_layers_.size(); ++i)
	{
		for (size_t j = 0; j < m_layers_[i].size(); ++j)
		{
			for (size_t k = 0; k < m_layers_[i][j].size(); ++k)
			{
				serialized.push_back(to_int8(m_layers_[i][j][k]));
			}
			if (j != m_layers_[i].size() - 1)
			{
				serialized.push_back(to_int8('/'));
			}
		}
		if (i != m_layers_.size() - 1)
		{
			serialized.push_back(to_int8('*'));
		}
	}
	return serialized;
}

std::vector<std::vector<std::vector<float>>> NetNeurons::Network::deserialize(const std::vector<int8_t>& serialized)
{
	std::vector<std::vector<std::vector<float>>> layers(1, std::vector<std::vector<float>>(1, std::vector<float>()));
	int layer = 0;
	int neuron = 0;
	for (size_t i = 0; i < serialized.size(); ++i)
	{
		if (serialized[i] == to_int8('*'))
		{
			++layer;
			neuron = 0;
			layers.push_back(std::vector<std::vector<float>>(1, std::vector<float>()));
		}
		else if (serialized[i] == to_int8('/'))
		{
			++neuron;
			layers[layer].push_back(std::vector<float>());
		}
		else
		{
			layers[layer][neuron].push_back(to_float(serialized[i]));
		}
	}
	return layers;
}

std::vector<bool> NetNeurons::Network::decode(int rec, int n_neurons)
{
	std::vector<bool> recs(n_neurons);
	for (int i = 0; i < n_neurons; ++i)
	{
		recs[i] = rec & 1;
		rec >>= 1;
	}
	return recs;
}

int NetNeurons::Network::encode(const std::vector<bool>& rec)
{
	int recs = 0;
	for (int i = rec.size() - 1; i >= 0; --i)
	{
		recs <<= 1;
		recs += rec[i];
	}
	return recs;
}

void NetNeurons::Network::wire_fit(const std::vector<float>& xt, const std::vector<float>& xt1, float R, const std::vector<std::vector<float>>& wires, const std::vector<float>& q_values, int imax, float alpha, float gamma, float epsilon, float c)
{
	std::vector<float> sorties1 = compute(xt1);
	float Q1 = 0.f;
	for (int i = 0; i < m_n_sorties_; i += m_n_wires_)
	{
		if (sorties1[i] > Q1)
		{
			Q1 = sorties1[i];
		}
	}

	float new_Q = (1 - alpha) * q_values[imax] + alpha * (R + gamma * Q1);

	std::vector<float> distances(m_n_wires_, 0.f);

	for (int i = 0; i < m_n_wires_; ++i)
	{
		distances[i] = distance(wires[i], wires[imax], q_values[i], q_values[imax], c, epsilon);
	}

	float wsum_ = wsum(distances, q_values);
	float norm_ = norm(distances);

	std::vector<float> sorties(m_n_sorties_);

	for (size_t i = 0; i < m_n_wires_; ++i)
	{
		if (i == imax)
		{
			sorties[i*(m_n_controls_ + 1)] = new_Q;
		}
		else 
		{
			sorties[i * (m_n_controls_ + 1)] = q_values[i] + alpha * diffQ(q_values[i], distances[i], wsum_, norm_, c);
		}
		for (size_t j = 0; j < m_n_controls_; ++j)
		{
			sorties[i * (m_n_controls_ + 1) + j + 1] += wires[i][j] + alpha * diffQ(wires[i][j], wires[imax][j], q_values[i], distances[i], wsum_, norm_);
		}
	}

	train(xt, sorties, alpha);
}

float distance(const std::vector<float>& u, const std::vector<float>& ui, float qi, float qmax, float c, float epsilon)
{
	float d = c * (qmax - qi) + epsilon;
	for (size_t j = 0; j < u.size(); ++j)
	{
		d += (u[j] - ui[j]) * (u[j] - ui[j]);
	}
	return d;
}

float wsum(const std::vector<float>& distances, const std::vector<float>& q)
{
	float s = 0.f;
	for (size_t i = 0; i < distances.size(); ++i)
	{
		s += q[i] / distances[i];
	}
	return s;
}

float norm(const std::vector<float>& distances)
{
	float s = 0.f;
	for (size_t i = 0; i < distances.size(); ++i)
	{
		s += 1.f / distances[i];
	}
	return s;
}

float diffQ(float qk, float dist, float wsum_, float norm_, float c)
{
	return (norm_ * (dist + c * qk) - wsum_ * c) / (norm_ * norm_ * dist * dist);
}

float diffQ(float ukj, float uj, float qk, float dist, float wsum_, float norm_)
{
	return  2.f * (wsum_ - norm_ * qk) * (ukj - uj) / (norm_ * norm_ * dist * dist);
}
