/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/memory/detail/buddy_allocator.h"

#include <algorithm>
#include <utility>

#include "glog/logging.h"

#ifdef PADDLE_WITH_CUDA
DECLARE_uint64(reallocate_gpu_memory_in_mb);
#endif

namespace paddle {
namespace memory {
namespace detail {

BuddyAllocator::BuddyAllocator(
    std::unique_ptr<SystemAllocator> system_allocator, size_t min_chunk_size,
    size_t max_chunk_size)
    : min_chunk_size_(min_chunk_size),
      max_chunk_size_(max_chunk_size),
      cache_(system_allocator->UseGpu()),
      system_allocator_(std::move(system_allocator)) {}

BuddyAllocator::~BuddyAllocator() {
  VLOG(10) << "BuddyAllocator Disconstructor makes sure that all of these "
              "have actually been freed";
  while (!pool_.empty()) {
    auto block = static_cast<MemoryBlock*>(std::get<2>(*pool_.begin()));
    auto desc = cache_.load_desc(block);
    VLOG(10) << "Free from block (" << block << ", " << desc->size << ")";

    system_allocator_->Free(block, desc->size, desc->index);
    cache_.invalidate(block);
    pool_.erase(pool_.begin());
  }
}

inline size_t align(size_t size, size_t alignment) {
  size_t remaining = size % alignment;
  return remaining == 0 ? size : size + (alignment - remaining);
}

void* BuddyAllocator::Alloc(size_t unaligned_size) {
  // adjust allocation alignment
  size_t size =
      align(unaligned_size + sizeof(MemoryBlock::Desc), min_chunk_size_);

  // acquire the allocator lock
  std::lock_guard<std::mutex> lock(mutex_);

  VLOG(10) << "Allocate " << unaligned_size << " bytes from chunk size "
           << size;

  // if the allocation is huge, send directly to the system allocator
  if (size > max_chunk_size_) {
    VLOG(10) << "Allocate from system allocator.";
    return SystemAlloc(size);
  }

  // query and allocate from the existing chunk
  auto it = FindExistChunk(size);

  // refill the pool if failure
  if (it == pool_.end()) {
    it = RefillPool(size);
    // if still failure, fail fatally
    if (it == pool_.end()) {
      return nullptr;
    }
  } else {
    VLOG(10) << "Allocation from existing memory block " << std::get<2>(*it)
             << " at address "
             << reinterpret_cast<MemoryBlock*>(std::get<2>(*it))->data();
  }

  total_used_ += size;
  total_free_ -= size;

  // split the allocation and return data for use
  return reinterpret_cast<MemoryBlock*>(SplitToAlloc(it, size))->data();
}

void BuddyAllocator::Free(void* p) {
  // Point back to metadata
  auto block = static_cast<MemoryBlock*>(p)->metadata();

  // Acquire the allocator lock
  std::lock_guard<std::mutex> lock(mutex_);

  VLOG(10) << "Free from address " << block;

  auto* desc = cache_.load_desc(block);
  if (desc->type == MemoryBlock::HUGE_CHUNK) {
    VLOG(10) << "Free directly from system allocator";
    system_allocator_->Free(block, desc->total_size, desc->index);

    // Invalidate GPU allocation from cache
    cache_.invalidate(block);

    return;
  }

  block->mark_as_free(&cache_);

  total_used_ -= desc->total_size;
  total_free_ += desc->total_size;

  // Trying to merge the right buddy
  MemoryBlock* right_buddy = nullptr;
  if (block->get_right_buddy(cache_, right_buddy)) {
    VLOG(10) << "Merging this block " << block << " with its right buddy "
             << right_buddy;

    auto rb_desc = cache_.load_desc(right_buddy);
    if (rb_desc->type == MemoryBlock::FREE_CHUNK) {
      // Take away right buddy from pool
      pool_.erase(
          IndexSizeAddress(rb_desc->index, rb_desc->total_size, right_buddy));

      // merge its right buddy to the block
      block->merge(&cache_, right_buddy);
    }
  }

  // Trying to merge the left buddy
  MemoryBlock* left_buddy = nullptr;
  if (block->get_left_buddy(cache_, left_buddy)) {
    VLOG(10) << "Merging this block " << block << " with its left buddy "
             << left_buddy;

    // auto left_buddy = block->left_buddy(cache_);
    auto* lb_desc = cache_.load_desc(left_buddy);
    if (lb_desc->type == MemoryBlock::FREE_CHUNK) {
      // Take away right buddy from pool
      pool_.erase(
          IndexSizeAddress(lb_desc->index, lb_desc->total_size, left_buddy));

      // merge the block to its left buddy
      left_buddy->merge(&cache_, block);
      block = left_buddy;
      desc = lb_desc;
    }
  }

  // Dumping this block into pool
  VLOG(10) << "Inserting free block (" << block << ", " << desc->total_size
           << ")";
  pool_.insert(IndexSizeAddress(desc->index, desc->total_size, block));
}

size_t BuddyAllocator::Used() { return total_used_; }
size_t BuddyAllocator::GetMinChunkSize() { return min_chunk_size_; }
size_t BuddyAllocator::GetMaxChunkSize() { return max_chunk_size_; }

void* BuddyAllocator::SystemAlloc(size_t size) {
  size_t index = 0;
  void* p = system_allocator_->Alloc(&index, size);

  VLOG(10) << "Allocated " << p << " from system allocator.";

  if (p == nullptr) return nullptr;

  static_cast<MemoryBlock*>(p)->init(&cache_, MemoryBlock::HUGE_CHUNK, index,
                                     size, nullptr, nullptr);

  return static_cast<MemoryBlock*>(p)->data();
}

BuddyAllocator::PoolSet::iterator BuddyAllocator::RefillPool(
    size_t request_bytes) {
  size_t allocate_bytes = max_chunk_size_;
  size_t index = 0;

#ifdef PADDLE_WITH_CUDA
  if (system_allocator_->UseGpu()) {
    if ((total_used_ + total_free_) == 0) {
      // Compute the allocation size for gpu for the first allocation.
      allocate_bytes = std::max(platform::GpuInitAllocSize(), request_bytes);
    } else {
      // Compute the re-allocation size, we store the re-allocation size when
      // user set FLAGS_reallocate_gpu_memory_in_mb to fix value.
      if (realloc_size_ == 0 || FLAGS_reallocate_gpu_memory_in_mb == 0ul) {
        realloc_size_ = platform::GpuReallocSize();
      }
      allocate_bytes = std::max(realloc_size_, request_bytes);
    }
  }
#endif

  // Allocate a new block
  void* p = system_allocator_->Alloc(&index, allocate_bytes);

  if (p == nullptr) return pool_.end();

  VLOG(10) << "Creating and inserting new block " << p
           << " from system allocator";

  static_cast<MemoryBlock*>(p)->init(&cache_, MemoryBlock::FREE_CHUNK, index,
                                     allocate_bytes, nullptr, nullptr);

  total_free_ += allocate_bytes;

  // dump the block into pool
  return pool_.insert(IndexSizeAddress(index, allocate_bytes, p)).first;
}

BuddyAllocator::PoolSet::iterator BuddyAllocator::FindExistChunk(size_t size) {
  size_t index = 0;

  while (1) {
    auto it = pool_.lower_bound(IndexSizeAddress(index, size, nullptr));

    // no match chunk memory
    if (it == pool_.end()) return it;

    if (std::get<0>(*it) > index) {
      // find suitable one
      if (std::get<1>(*it) >= size) {
        return it;
      }
      // update and continue
      index = std::get<0>(*it);
      continue;
    }
    return it;
  }
}

void* BuddyAllocator::SplitToAlloc(BuddyAllocator::PoolSet::iterator it,
                                   size_t size) {
  auto block = static_cast<MemoryBlock*>(std::get<2>(*it));
  auto desc = cache_.load_desc(block);
  pool_.erase(it);

  VLOG(10) << "Split block (" << block << ", " << desc->total_size << ") into";
  block->split(&cache_, size);

  VLOG(10) << "Left block (" << block << ", " << desc->total_size << ")";
  desc->type = MemoryBlock::ARENA_CHUNK;
  desc->update_guards();

  // the rest of memory if exist
  MemoryBlock* right_buddy = nullptr;
  if (block->get_right_buddy(cache_, right_buddy)) {
    auto* rb_desc = cache_.load_desc(right_buddy);
    if (rb_desc->type == MemoryBlock::FREE_CHUNK) {
      VLOG(10) << "Insert right block (" << right_buddy << ", "
               << rb_desc->total_size << ")";

      pool_.insert(
          IndexSizeAddress(rb_desc->index, rb_desc->total_size, right_buddy));
    }
  }

  return block;
}

}  // namespace detail
}  // namespace memory
}  // namespace paddle
