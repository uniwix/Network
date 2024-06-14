#include "pch.h"
#include "AI.hpp"

#include <utility>

using namespace NeuralNetwork;

AI::AI(std::function<std::vector<float>()> getter, std::function<void(std::vector<float>)> setter, const float mutation_rate)
    : m_mutation_rate_(mutation_rate), m_getter_(std::move(getter)), m_setter_(std::move(setter))
{}

AI::AI(Network network, std::function<std::vector<float>()> getter, std::function<void(std::vector<float>)> setter, const float mutation_rate)
    : Network(std::move(network)), m_mutation_rate_(mutation_rate), m_getter_(std::move(getter)),
    m_setter_(std::move(setter))
{}

AI::AI(std::istream& is, std::function<std::vector<float>()> getter, std::function<void(std::vector<float>)> setter, const float mutation_rate)
    : Network(is), m_mutation_rate_(mutation_rate), m_getter_(std::move(getter)), m_setter_(std::move(setter))
{}

AI::AI(const std::vector<int>& layers, std::function<std::vector<float>()> getter, std::function<void(std::vector<float>)> setter, const float mutation_rate)
    : Network(layers), m_mutation_rate_(mutation_rate), m_getter_(std::move(getter)), m_setter_(std::move(setter))
{}

AI::AI(std::vector<std::vector<std::vector<float>>> data, std::function<std::vector<float>()> getter, std::function<void(std::vector<float>)> setter, const float mutation_rate)
    : Network(std::move(data)), m_mutation_rate_(mutation_rate), m_getter_(std::move(getter)),
    m_setter_(std::move(setter))
{}

void AI::reward(const float bonus)
{
    m_score_ += bonus;
}

void AI::punish(const float penalty)
{
    m_score_ -= penalty;
}

float AI::get_score() const
{
    return m_score_;
}

void AI::set_score(const float score)
{
		m_score_ = score;
}

void AI::update() const
{
	std::vector<float> inputs(m_getter_());
	//std::vector<float> outputs = compute(inputs);
	//m_setter_(outputs);
}


bool AI::operator<(const AI& other) const
{
    return m_score_ < other.m_score_;
}

AI AI::cross(const AI& parent1, const AI& parent2)
{
    const float mutation_rate = (parent1.m_mutation_rate_ + parent2.m_mutation_rate_) / 2.f;
    const Network child_net = Network::reproduce(parent1, parent2, mutation_rate);
    return { child_net, parent1.m_getter_, parent1.m_setter_ , mutation_rate };
}