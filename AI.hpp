#pragma once

#include <functional>

#include "Network.hpp"

template class NETWORK_API std::function<std::vector<float>()>;
template class NETWORK_API std::function<void(std::vector<float>)>;

namespace NeuralNetwork
{
	class NETWORK_API AI : public Network
	{
	public:
		AI(std::function<std::vector<float>()> getter,
			std::function<void(std::vector<float>)> setter,
			float mutation_rate = 0.01f);

		AI(Network network,
			std::function<std::vector<float>()> getter,
			std::function<void(std::vector<float>)> setter,
			float mutation_rate = 0.01f);

		AI(std::istream& is,
			std::function<std::vector<float>()> getter,
			std::function<void(std::vector<float>)> setter,
			float mutation_rate = 0.01f);

		AI(const std::vector<int>& layers,
			std::function<std::vector<float>()> getter,
			std::function<void(std::vector<float>)> setter,
			float mutation_rate = 0.01f);

		AI(std::vector<std::vector<std::vector<float>>> data,
			std::function<std::vector<float>()> getter,
			std::function<void(std::vector<float>)> setter,
			float mutation_rate = 0.01f);

		void reward(float bonus);
		void punish(float penalty);
		float get_score() const;
		void set_score(float score);
		void update() const;

		bool operator<(const AI& other) const;

		static AI cross(const AI& parent1, const AI& parent2);
	private:
		float m_score_ = 0;
		float m_mutation_rate_;
		std::function<std::vector<float>()> m_getter_;
		std::function<void(std::vector<float>)> m_setter_;
	};
}
