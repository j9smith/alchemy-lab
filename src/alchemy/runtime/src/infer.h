#pragma once
#include <vector>
#include "batch_queue.h"

std::vector<std::vector<float>> infer(std::vector<BatchItem>& batch);