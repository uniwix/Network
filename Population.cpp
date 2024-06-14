#include "pch.h"
#include "Population.hpp"

using namespace NeuralNetwork;

Population::Population(int size, int select, std::function<std::vector<float>()> getter,
	std::function<void(std::vector<float>)> setter,
	float mutation_rate) : select_(SELECT_AMOUNT), select_amount(select)
{
	for (int i = 0; i < size; ++i)
	{
		individuals.emplace_back(getter, setter, mutation_rate);
	}
}
