#pragma once

#include <vector>
#include <array>
#include <span>
#include <memory>
#include <optional>
#include <variant>
#include <tuple>

template<typename T, typename Allocator = std::allocator<T>>
using Array = std::vector<T, Allocator>;

template<typename T, size_t SIZE>
using StaticArray = std::array<T, SIZE>;

template<typename T, size_t EXTENT = std::dynamic_extent>
using Span = std::span<T, EXTENT>;

template<typename T, typename Deleter = std::default_delete<T>>
using UniquePtr = std::unique_ptr<T, Deleter>;

template<typename T>
using SharedPtr = std::shared_ptr<T>;

template<typename T>
using Optional = std::optional<T>;

template<typename... Ts>
using Variant = std::variant<Ts...>;

template<typename... Ts>
using Tuple = std::tuple<Ts...>;