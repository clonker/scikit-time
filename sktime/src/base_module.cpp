//
// Created by mho on 7/6/20.
//

#include "common.h"

template<typename dtype>
void gatherFrames(const py::list &listOfTrajs, const np_array<int> &trajs,
                  const np_array<int> &frames, np_array<dtype> &out) {
    const auto *trajsPtr = trajs.data();
    const auto *framesPtr = frames.data();
    auto *outPtr = out.mutable_data();
    auto dim = out.shape(1);
    std::vector<np_array<dtype>> vecOfTrajs;
    vecOfTrajs.reserve(listOfTrajs.size());
    for (auto traj : listOfTrajs) {
        vecOfTrajs.emplace_back(traj.cast<np_array<dtype>>());
    }
    for (auto i = 0U; i < trajs.shape(0); ++i) {
        auto traj = trajsPtr[i];
        auto frame = framesPtr[i];
        if (frame > vecOfTrajs[traj].shape(1)) {
            throw std::invalid_argument(
                    "Tried accessing frame " + std::to_string(frame) + " of trajectory " + std::to_string(traj) +
                    " with length " + std::to_string(vecOfTrajs[traj].shape(1)));
        }
        auto begin = vecOfTrajs[traj].data() + frame * dim;
        auto end = begin + dim;
        std::copy(begin, end, outPtr + i * dim);
    }
}

PYBIND11_MODULE(_base_bindings, m) {
    m.def("gather_frames_f", &gatherFrames<float>);
    m.def("gather_frames_d", &gatherFrames<double>);
}
