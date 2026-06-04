/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <utility>
#include <new>
#include <type_traits>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

namespace cuda {
namespace std {

struct nullopt_t {
    struct init {};
    __host__ __device__ constexpr explicit nullopt_t(init) {}
};

static constexpr nullopt_t nullopt{nullopt_t::init{}};

template <typename T>
class optional {
private:
    union {
        T value_;
        char dummy_;
    };
    bool has_value_;

    // Helper to check if U is a "foreign" type (not T, not optional<T>, not nullopt_t)
    template <typename U>
    struct is_foreign_type {
        using Decay = typename ::std::decay<U>::type;
        static constexpr bool value =
            !::std::is_same<Decay, T>::value &&
            !::std::is_same<Decay, optional<T>>::value &&
            !::std::is_same<Decay, nullopt_t>::value;
    };

public:
    __host__ __device__ constexpr optional() : dummy_(0), has_value_(false) {}
    __host__ __device__ constexpr optional(nullopt_t) : dummy_(0), has_value_(false) {}
    
    __host__ __device__ optional(const T& val) : value_(val), has_value_(true) {}
    __host__ __device__ optional(T&& val) : value_(::std::move(val)), has_value_(true) {}

    // Converting constructor: allow optional<T> to be constructed from U where T(U) is valid
    template <typename U, typename ::std::enable_if<is_foreign_type<U>::value, int>::type = 0>
    __host__ __device__ optional(U&& val) : value_(T(::std::forward<U>(val))), has_value_(true) {}

    // Converting assignment: allow optional<T> = u where T(u) is valid
    template <typename U>
    __host__ __device__ typename ::std::enable_if<is_foreign_type<U>::value, optional&>::type
    operator=(U&& val) {
        if (has_value_) {
            value_.~T();
        }
        new (&value_) T(::std::forward<U>(val));
        has_value_ = true;
        return *this;
    }

    __host__ __device__ ~optional() {
        if (has_value_) {
            value_.~T();
        }
    }

    __host__ __device__ optional(const optional& other) : has_value_(other.has_value_) {
        if (has_value_) {
            new (&value_) T(other.value_);
        } else {
            dummy_ = 0;
        }
    }

    __host__ __device__ optional(optional&& other) : has_value_(other.has_value_) {
        if (has_value_) {
            new (&value_) T(::std::move(other.value_));
        } else {
            dummy_ = 0;
        }
    }

    __host__ __device__ optional& operator=(const optional& other) {
        if (this == &other) return *this;
        if (has_value_) {
            if (other.has_value_) {
                value_ = other.value_;
            } else {
                value_.~T();
                has_value_ = false;
            }
        } else {
            if (other.has_value_) {
                new (&value_) T(other.value_);
                has_value_ = true;
            }
        }
        return *this;
    }

    __host__ __device__ optional& operator=(optional&& other) {
        if (this == &other) return *this;
        if (has_value_) {
            if (other.has_value_) {
                value_ = ::std::move(other.value_);
            } else {
                value_.~T();
                has_value_ = false;
            }
        } else {
            if (other.has_value_) {
                new (&value_) T(::std::move(other.value_));
                has_value_ = true;
            }
        }
        return *this;
    }

    __host__ __device__ constexpr bool has_value() const noexcept { return has_value_; }
    __host__ __device__ constexpr const T& value() const { return value_; }
    __host__ __device__ constexpr T& value() { return value_; }
    __host__ __device__ constexpr const T& operator*() const noexcept { return value_; }
    __host__ __device__ constexpr T& operator*() noexcept { return value_; }
    __host__ __device__ constexpr const T* operator->() const noexcept { return &value_; }
    __host__ __device__ constexpr T* operator->() noexcept { return &value_; }
    __host__ __device__ constexpr explicit operator bool() const noexcept { return has_value_; }
};

} // namespace std
} // namespace cuda
