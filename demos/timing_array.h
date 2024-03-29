/*
 * Copyright 2020 Google LLC
 *
 * Licensed under both the 3-Clause BSD License and the GPLv2, found in the
 * LICENSE and LICENSE.GPL-2.0 files, respectively, in the root directory.
 *
 * SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0
 */

#ifndef DEMOS_TIMING_ARRAY_H_
#define DEMOS_TIMING_ARRAY_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hardware_constants.h"

// TimingArray is an indexable container that makes it easy to induce and
// measure cache timing side-channels to leak the value of a single byte.
//
// TimingArray goes to some trouble to make sure that element accesses do not
// interfere with the cache presence of other elements.
//
// Specifically:
//   - Each element is on its own physical page of memory, which usually
//     prevents interference from hardware prefetchers.[1]
//   - Elements' order in memory is different than their index order, which
//     further frustrates hardware prefetchers by avoiding memory accesses at
//     any repeated stride.
//   - Elements are distributed across cache sets -- as opposed to e.g. always
//     being at offset 0 within a page -- which reduces contention within cache
//     sets, optimizing the use of L1 and L2 caches and therefore improving
//     side-channel signal by increasing the timing difference between cached
//     and uncached accesses.
//
// TimingArray also includes convenience functions for cache manipulation and
// timing measurement.
//
// Example use:
//
//     TimingArray ta;
//     int i = -1;
//
//     // Loop until we're sure we saw an element come from cache
//     while (i == -1) {
//       ta.FlushFromCache();
//       ForceRead(&ta[4]);
//       i = ta.FindFirstCachedElementIndex();
//     }
//     std::cout << "item " << i << " was in cache" << std::endl;
//
// [1] See e.g. Intel's documentation at https://cpu.fyi/d/83c#G3.1121453,
//   which says data is only prefetched if it is on the "same 4K byte page".

extern uint64_t saved_threadhold;
extern uint64_t saved_char_latencies[256];

class TimingArray {
 public:
  // ValueType is an alias for the element type of the array. Someday we might
  // want to make TimingArray a template class and take this as a parameter,
  // but for now the flexibility isn't worth the extra hassle.
  using ValueType = int;

  // This could likewise be a parameter and, likewise, isn't. Making the array
  // smaller doesn't improve the bandwidth of the side-channel, and trying to
  // leak more than a byte at a time significantly increases noise due to
  // greater cache contention.
  //
  // "Real" elements because we add buffer elements before and after. See
  // comment on `elements_` below.
  static const size_t kRealElements = 256;

  TimingArray();

  TimingArray(TimingArray&) = delete;
  TimingArray& operator=(TimingArray&) = delete;

  ValueType& operator[](size_t i) {
    // Map index to element.
    //
    // As mentioned in the class comment, we try to frustrate hardware
    // prefetchers by applying a permutation so elements don't appear in
    // memory in index order. We do this by using a Linear Congruential
    // Generator (LCG) where the size of the array is the modulus.
    // To guarantee that this is a permutation, we just need to follow the
    // requirements here:
    //
    // https://en.wikipedia.org/wiki/Linear_congruential_generator#c_%E2%89%A0_0
    //
    // In this case, 113 makes a good choice: 113 is prime, 113-1 = 112 is
    // divisible by 256's only prime factor (2), both are divisible by 4.
    // If the array were dynamically sized, we would not be able to hardcode
    // 113, and it would be difficult to produce a good multiplier.
    // It may be desirable, instead, to switch to a different PRNG
    // that also supports small periods and efficient seeking.
    // (For example, a Permuted Congruential Generator.)
    static_assert(kRealElements == 256, "consider changing 113");
    size_t el = (100 + i*113) % kRealElements;

    // Add 1 to skip leading buffer element.
    return elements_[1 + el].cache_lines[0].value;
  }

  // We intentionally omit the "const" accessor:
  //    const ValueType& operator[](size_t i) const { ... }
  //
  // At the level of abstraction we care about, accessing an element at all
  // (even to read it) is not "morally const" since it mutates cache state.

  size_t size() const { return kRealElements; }

  // Flushes all elements of the array from the cache.
  void FlushFromCache();

  // Reads elements of the array in index order, starting with index 0, and
  // looks for the first read that was fast enough to have come from the cache.
  //
  // Returns the index of the first "fast" element, or -1 if no element was
  // obviously read from the cache.
  //
  // This function uses a heuristic that errs on the side of false *negatives*,
  // so it is common to use it in a loop. Of course, measuring the time it
  // takes to read an element has the side effect of bringing that element into
  // the cache, so the loop must include a cache flush and re-attempt the
  // side-channel leak.
  int FindFirstCachedElementIndex();

  // Just like `FindFirstCachedElementIndex`, except it begins right *after*
  // the index `start_after`, wrapping around to try all array elements. That
  // is, the first element read is `(start_after+1) % size` and the last
  // element read before returning -1 is `start_after`.
  int FindFirstCachedElementIndexAfter(int start_after);

  // Returns the threshold value used by FindFirstCachedElementIndex to
  // identify reads that came from cache.
  uint64_t cached_read_latency_threshold() const {
    return cached_read_latency_threshold_;
  }

 private:
  // Convenience so we don't have (*this)[i] everywhere.
  ValueType& ElementAt(size_t i) { return (*this)[i]; }

  uint64_t cached_read_latency_threshold_;
  uint64_t FindCachedReadLatencyThreshold();

  // Define a struct that occupies one full cache line. Some compilers may not
  // support aligning at `kCacheLineBytes`, which is almost always greater than
  // `sizeof(std::max_align_t)`. In those cases we'll get a compile error.
  // See: https://timsong-cpp.github.io/cppwp/n3337/basic.align#9
  struct alignas(kCacheLineBytes) CacheLine {
    ValueType value;
  };
  static_assert(sizeof(CacheLine) == kCacheLineBytes, "");

  // Define our "Element" struct, which takes up one page plus one cache line.
  // When we allocate an array of these, we know that adjacent elements will
  // start on different pages and in different cache sets.
  static const int kCacheLinesPerPage = kPageBytes / kCacheLineBytes;
  struct Element {
    std::array<CacheLine, kCacheLinesPerPage + 1> cache_lines;
  };
  static_assert(sizeof(Element) == kPageBytes + kCacheLineBytes, "");

  // The actual backing store for the timing array, with buffer elements before
  // and after to avoid interference from adjacent heap allocations.
  //
  // We use `vector` here instead of `array` to avoid problems where
  // `TimingArray` is put on the stack and the class is so large it skips past
  // the stack guard page. This is more likely on PowerPC where the page size
  // (and therefore our element stride) is 64K.
  std::vector<Element> elements_{1 + kRealElements + 1};
};

#endif  // DEMOS_TIMING_ARRAY_H_
