// Network.hpp - Contains declarations of Network class
#pragma once

#include <random>
#include <vector>
#include <istream>

#ifdef NETWORK_EXPORTS
#define NETWORK_API __declspec(dllexport)
#else
#define NETWORK_API __declspec(dllimport)
#endif

namespace NeuralNetwork {
	using namespace std;

	class Network
	{
	public:
		NETWORK_API Network(); // Construit un r�seau vide

		NETWORK_API Network(const vector<int>& layers);  // Construit un r�seau avec les couches donn�es
		NETWORK_API Network(vector<vector<vector<float>>> data);  // Construit un r�seau � partir de donn�es existantes
		NETWORK_API Network(istream& is);  // Construit un r�seau � partir des donn�es du flux
		// Similaire aux constructeurs pr�c�dents, mais avec des neurones r�currents
		NETWORK_API Network(const vector<int>& layers, const vector<bool>& rec);
		NETWORK_API Network(vector<vector<vector<float>>> data, const vector<bool>& rec);
		NETWORK_API Network(istream& is, const vector<bool>& rec);

		NETWORK_API Network(int n_inputs, int n_hidden, int n_wires, int n_outputs);  // Constructeur sp�cifique au Q-learning

		NETWORK_API vector<float> compute(const vector<float>& inputs);  // Calcule la valeur de sortie du r�seau appliqu� aux entr�es

		NETWORK_API vector<vector<vector<float>>> get_layers() const { return m_layers_; };  // Renvoie les poids du r�seau
		NETWORK_API vector<bool> get_rec() const { return m_rec_; };  // Renvoie les informations sur les neurones r�currents

		NETWORK_API static Network reproduce(const Network& parent1, const Network& parent2, float mutation_rate = .01f);  // Op�rateur g�n�tique de croisement et mutation

		// Donne une repr�sentation du r�seau
		NETWORK_API ostream& print_to(ostream& os) const;


		// Applique l'algorithme de r�tropropagation du gradient
		NETWORK_API float train(const vector<float>& inputs, const vector<float>& expected_outputs, float epsilon = 0.1f);

		// Applique l'apprentissage Q-learning
		NETWORK_API void wire_fit(const vector<float>& xt, const vector<float>& xt1, float R, const vector<vector<float>>& wires, const vector<float>& q_values, int imax, float alpha, float gamma, float epsilon, float c);

		NETWORK_API int getWireCount() const { return m_n_wires_; };
		NETWORK_API int getOutputsCount() const { return m_n_outputs_; };
		NETWORK_API int getControlsCount() const { return m_n_controls_; };

	private:
		NETWORK_API void mutation(float mutation_rate = .01f);  // Applique l'op�rateur g�n�tique de mutation
		NETWORK_API Network operator*(const Network& other) const;  // Op�rateur g�n�tique de croisement

		static void set_weights(vector<float>& weights);  // Assigne des valeurs al�atoires � une liste

		vector<vector<vector<float>>> m_layers_;
		vector<vector<float>> m_previous_state_;
		vector<bool> m_rec_;

		int m_n_wires_ = 5;
		int m_n_outputs_ = 20;
		int m_n_controls_ = 3;
	};

	// Encode ou d�code les informations de neurones r�currents
	NETWORK_API vector<bool> decode(int rec, int n_layers);
	NETWORK_API int encode(const vector<bool>& rec);

	// Encode ou d�code les donn�es sous forme binaire
	NETWORK_API vector<int8_t> serialize(const vector<vector<vector<float>>>& data);
	NETWORK_API vector<vector<vector<float>>> deserialize(const vector<int8_t>& serialized);
	NETWORK_API vector<vector<vector<float>>> deserialize(istream& is);
}
