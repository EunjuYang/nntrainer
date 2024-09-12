// SPDX-License-Identifier: Apache-2.0
/**
 * @file	tensor_uint32.h
 * @date	12 September 2024
 * @brief	This is Uint32Tensor class for 32-bit unsigned integer calculation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __TENSOR_UINT32_H__
#define __TENSOR_UINT32_H__
#ifdef __cplusplus

#include <tensor_base.h>

namespace nntrainer{

    /**
     * @class Uint32Tensor class
     * @brief Uint32Tensor class for 32-bit unsigned integer calculation
     */
    class Uint32Tensor : public TensorBase {
        public:
        /**
         * @brief Basic Custroctor of Uint32Tensor class
         */
        Uint32Tensor(std::string name_="", Tformat fm = Tformat::NCHW);

        /**
         * @brief Construct a new Uint32Tensor object
         *
         * @param d Tensor dim for this uint_32 tensor
         * @param alloc_now Allocate memory to this tensor or not
         * @param init Initializer for the tensor
         * @param name Name of the tensor
         */
        Uint32Tensor(const TensorDim &d, bool alloc_now, Initializer init,
                     std::string name);

        /**
         * @brief Construct a new Uint32Tensor object
         *
         * @param d Tensor dim for this tensor
         * @param buf buffer
         */
        Uint32Tensor(const TensorDim &d, const void *buf = nullptr);

        
        /**
         * @brief Construct a new Uint32Tensor object
         *
         * @param d data for the Tensor
         * @param fm format for the Tensor
         */
        Uint32Tensor(std::vector<std::vector<std::vector<std::vector<uint32_t>>>> const &d, Tformat fm);

        /**
         * @brief Construct a new Uint32Tensor object
         * @param rhs TensorBase object to copy
         */
        Uint32Tensor(TensorBase &rhs) : TensorBase(rhs) {}

        /**
         * @brief Basic Destructor
         */
        ~Uint32Tensor() {}

        /**
         * @brief     Comparison operator overload
         * @param[in] rhs Tensor to be compared with
         * @note      Only compares Tensor data
         */
        bool operator==(const Uint32Tensor &rhs) const;

}

#endif // end of __cplusplus
#endif // end of __TENSOR_UINT32_H__
