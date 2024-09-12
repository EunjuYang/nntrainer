// SPDX-License-Identifier: Apache-2.0
/**
 * @file	tensor_uint32.cpp
 * @date	12 September 2024
 * @brief	This is Uint32Tensor class for 32-bit unsigned integer calculation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <iomanip>
#include <iostream>

#include <blas_interface.h>
#include <tensor_uint32.h>
#include <tensor.h>

namespace nntrainer {

    Uint32Tensor::Uint32Tensor(std::string name_, Tformat fm) : 
        TensorBase(name_, fm, Tdatatype::UINT32){}

    Uint32Tensor::Uint32Tensor(const TensorDim &d, bool alloc_now, Initializer init, std::string name) :
        TensorBase(d, alloc_now, init, name){
            if (alloc_now)
                allocate();
    }

    Uint32Tensor::Uint32Tensor(const TensorDim &d, const void *buf) :
        Uint32Tensor(d, true){
            if (d.getDataLen() != 0){
                if (buf != nullptr)
                    copy(buf);
            }
    }

    Uint32Tensor::Uint32Tensor(std::vector<std::vector<std::vector<std::vector<uint32_t>>>> const &d, Tformat fm){
        if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty()){
            throw std::out_of_range(
            "[Tensor] trying to initialize Uint32Tensor from empty vector");
        }

        dim.setTensorDim(0, d.size());
        if (fm == Tformat::NCHW){
            dim.setTensorDim(1, d[0].size());
            dim.setTensorDim(2, d[0][0].size());
            dim.setTensorDim(3, d[0][0][0].size());
        } else {
            dim.setTensorDim(2, d[0].size());
            dim.setTensorDim(3, d[0][0].size());
            dim.setTensorDim(1, d[0][0][0].size());
        }

        dim.setTensorType({fm, Tdatatype::UINT16});

        strides = dim.computeStrides();
        contiguous = true;
        initializer = Initializer::NONE;

        MemoryData *mem_data = new MemoryData((void *)(new uint32_t[dim.getDataLen()]()));
        data = std::shared_ptr<MemoryData>(mem_data, [](MemoryData *mem_data) {
        delete[] mem_data->getAddr<uint32_t>();});

        offset = 0;


        // if fm == Tformat::NCHW, then dim[0] == batch , dim[1] == channel, dim[2]
        // == height, dim[3] == width. and if fm == Tformat::NHWC, dim[0] == batch,
        // dim[1] == height, dim[2] == width, dim[3] == channel
        if (fm == Tformat::NCHW) {
            for (unsigned int i = 0; i < batch(); ++i)
            for (unsigned int j = 0; j < channel(); ++j)
                for (unsigned int k = 0; k < height(); ++k)
                for (unsigned int l = 0; l < width(); ++l)
                    this->setValue(i, j, k, l, d[i][j][k][l]);
        } else {
            for (unsigned int i = 0; i < batch(); ++i)
            for (unsigned int j = 0; j < height(); ++j)
                for (unsigned int k = 0; k < width(); ++k)
                for (unsigned int l = 0; l < channel(); ++l)
                    this->setValue(i, l, j, k, d[i][j][k][l]);
        }
    }

}
}
