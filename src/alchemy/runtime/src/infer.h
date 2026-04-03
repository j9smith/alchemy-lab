#pragma once
#include <vector>
#include "batch_queue.h"
#include "pipeline.h"

std::vector<std::vector<float>> infer(std::vector<BatchItem>& batch, AlchemyPipeline& pipeline);