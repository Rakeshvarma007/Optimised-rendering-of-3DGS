/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/*
 * GSPLAT_FOR_EACH(macro, a, b, c, ...):
 *   Expands to: macro(a) macro(b) macro(c) ...
 *   - Works with zero or more arguments.
 *   - When called with no arguments, it expands to nothing.
 *   - Example:
 *       #define F(x) FEATURE_ITEM(x)
 *       GSPLAT_FOR_EACH(F, 1, 2, 3)
 *     expands to:
 *       FEATURE_ITEM(1) FEATURE_ITEM(2) FEATURE_ITEM(3)
 */

#define GSPLAT_EXPAND(...) __VA_ARGS__

#define GSPLAT_FOR_EACH_1(m, x) m(x)
#define GSPLAT_FOR_EACH_2(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_1(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_3(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_2(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_4(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_3(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_5(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_4(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_6(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_5(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_7(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_6(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_8(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_7(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_9(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_8(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_10(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_9(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_11(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_10(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_12(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_11(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_13(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_12(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_14(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_13(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_15(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_14(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_16(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_15(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_17(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_16(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_18(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_17(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_19(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_18(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_20(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_19(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_21(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_20(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_22(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_21(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_23(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_22(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_24(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_23(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_25(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_24(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_26(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_25(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_27(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_26(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_28(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_27(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_29(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_28(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_30(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_29(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_31(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_30(m, __VA_ARGS__))
#define GSPLAT_FOR_EACH_32(m, x, ...) m(x) GSPLAT_EXPAND(GSPLAT_FOR_EACH_31(m, __VA_ARGS__))

#define GSPLAT_FOR_EACH_NARG(...) GSPLAT_EXPAND(GSPLAT_FOR_EACH_NARG_HELPER(__VA_ARGS__, GSPLAT_FOR_EACH_RSEQ_N()))
#define GSPLAT_FOR_EACH_NARG_HELPER(...) GSPLAT_EXPAND(GSPLAT_FOR_EACH_ARG_N(__VA_ARGS__))
#define GSPLAT_FOR_EACH_ARG_N( \
     _1, _2, _3, _4, _5, _6, _7, _8, _9,_10, \
    _11,_12,_13,_14,_15,_16,_17,_18,_19,_20, \
    _21,_22,_23,_24,_25,_26,_27,_28,_29,_30, \
    _31,_32,N,...) N
#define GSPLAT_FOR_EACH_RSEQ_N() \
    32,31,30,29,28,27,26,25, \
    24,23,22,21,20,19,18,17, \
    16,15,14,13,12,11,10, 9, \
     8, 7, 6, 5, 4, 3, 2, 1, 0

#define GSPLAT_PASTE(a, b) GSPLAT_PASTE_HELPER(a, b)
#define GSPLAT_PASTE_HELPER(a, b) a ## b

#define GSPLAT_FOR_EACH(m, ...) GSPLAT_EXPAND(GSPLAT_PASTE(GSPLAT_FOR_EACH_, GSPLAT_EXPAND(GSPLAT_FOR_EACH_NARG(__VA_ARGS__))) (m, __VA_ARGS__))
