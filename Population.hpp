#pragma once
#include "AI.hpp"

namespace NeuralNetwork
{
	class NETWORK_API Population
	{
	public:
		Population(int size, int select, std::function<std::vector<float>()> getter,
			std::function<void(std::vector<float>)> setter,
			float mutation_rate = 0.01f);
		Population(int size, std::function<bool(AI)> select, std::function<std::vector<float>()> getter,
			std::function<void(std::vector<float>)> setter,
			float mutation_rate = 0.01f);
		void select();
	private:
		std::vector<AI> individuals;

		enum select_method
		{
			SELECT_AMOUNT,
			SELECT_FUNC,
		};
		select_method select_;
		int select_amount = 0;
	};
}
