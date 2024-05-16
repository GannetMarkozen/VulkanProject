#include "core/CoreTypes.hpp"
#include "core/Defines.hpp"

#include <vulkan/vulkan_core.h>

#define GLFW_INCLUDE_VULKAN// Enable GLFW to use Vulkan.
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>

#define FMT_EXCEPTIONS 0// Disable exceptions from fmt.
#include <fmt/printf.h>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

[[nodiscard]]
static fn read_file(const char* file_name) -> Array<char> {
  std::ifstream stream{file_name, std::ios::ate | std::ios::binary};
  assert(stream.is_open());

  const auto file_size = static_cast<usize>(stream.tellg());

  Array<char> out_data;
  out_data.resize(file_size);

  stream.seekg(0);
  stream.read(out_data.data(), file_size);

  return out_data;
}

struct Vertex {
  glm::vec2 position;
  glm::vec3 color;

  [[nodiscard, clang::always_inline]]
  static constexpr fn get_binding_description() -> VkVertexInputBindingDescription {
    return {
      .binding = 0,
      .stride = sizeof(Vertex),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
  }

  [[nodiscard, clang::always_inline]]
  static constexpr fn get_attribute_descriptions() -> StaticArray<VkVertexInputAttributeDescription, 2> {
    return {{
      VkVertexInputAttributeDescription{
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32_SFLOAT,
        .offset = offsetof(Vertex, position),
      },
      VkVertexInputAttributeDescription {
        .location = 1,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(Vertex, color),
      },
    }};
  }
};

struct App {
  fn run() -> void {
    init_window();
    init_vulkan();
    main_loop();
    cleanup();
  }

private:
  static constexpr u32 WIDTH = 800;
  static constexpr u32 HEIGHT = 600;
  static constexpr u32 MAX_FRAMES_IN_FLIGHT = 2;
  static constexpr const char* APPLICATION_NAME = "VulkanProject";

  struct QueueFamilies {
    [[nodiscard]]
    bool is_valid() const {
      return graphics_family.has_value() && present_family.has_value();
    }

    Optional<u32> graphics_family;
    Optional<u32> present_family;
  };

  static constexpr const char* VALIDATION_LAYERS[]{
    "VK_LAYER_KHRONOS_validation",
  };

  static constexpr const char* REQUIRED_DEVICE_EXTENSIONS[]{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
  };

  static constexpr StaticArray<Vertex, 4> VERTICES{{
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}},
  }};

  static constexpr StaticArray<u16, 6> INDICES{{
    0, 1, 2, 2, 3, 0,
  }};

  fn init_window() -> void {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WIDTH, HEIGHT, APPLICATION_NAME, nullptr, nullptr);
  }

  fn init_vulkan() -> void {
    const VkApplicationInfo app_info{
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = APPLICATION_NAME,
      .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
      .pEngineName = "GanEngine",
      .engineVersion = VK_MAKE_VERSION(0, 1, 0),
      .apiVersion = VK_API_VERSION_1_3,
    };

    u32 glfw_extensions_count = 0;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extensions_count);

    // Enable applicable validation layers for debug builds.
    u32 enabled_layer_count;
    const char** enabled_layers;
#if DEBUG_BUILD
    {
      u32 applicable_layers_count;
      vkEnumerateInstanceLayerProperties(&applicable_layers_count, nullptr);

      auto* applicable_layers = STACK_ALLOCATE_ZEROED(VkLayerProperties, applicable_layers_count);
      vkEnumerateInstanceLayerProperties(&applicable_layers_count, applicable_layers);

      enabled_layer_count = 0;
      enabled_layers = STACK_ALLOCATE_UNINIT(const char*, applicable_layers_count);

      const Span<const VkLayerProperties> applicable_layers_span{applicable_layers, applicable_layers_count};
      for (const char* validation_layer : VALIDATION_LAYERS) {
        const auto it = std::find_if(applicable_layers_span.begin(), applicable_layers_span.end(), [&](const VkLayerProperties &layer) {
          return strcmp(layer.layerName, validation_layer) == 0;
        });

        if (it != applicable_layers_span.end()) {
          enabled_layers[enabled_layer_count++] = validation_layer;
          fmt::println("Enabling validation layer {}!", validation_layer);
        }
      }
    }
#else
    enabled_layer_count = 0;
    enabled_layers = nullptr;
#endif

    const VkInstanceCreateInfo create_info{
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
      .enabledLayerCount = enabled_layer_count,
      .ppEnabledLayerNames = enabled_layers,
      .enabledExtensionCount = glfw_extensions_count,
      .ppEnabledExtensionNames = glfw_extensions,
    };

    const VkResult result = vkCreateInstance(&create_info, nullptr, &vulkan_instance);
    assert(result == VK_SUCCESS);

    create_surface();
    pick_physical_device();
    create_logical_device();
    create_swapchain();
    create_image_views();
    create_render_pass();
    create_graphics_pipeline();
    create_framebuffers();
    create_command_pool();
    create_vertex_buffer();
    create_index_buffer();
    create_command_buffers();
    create_sync_objects();
  }

  fn create_surface() -> void {
    const VkResult result = glfwCreateWindowSurface(vulkan_instance, window, nullptr, &surface);
    assert(result == VK_SUCCESS);
  }

  fn pick_physical_device() -> void {
    // Pick a physical device.
    u32 physical_device_count = 0;
    vkEnumeratePhysicalDevices(vulkan_instance, &physical_device_count, nullptr);
    assert(physical_device_count > 0);

    auto* physical_devices = STACK_ALLOCATE_UNINIT(VkPhysicalDevice, physical_device_count);
    vkEnumeratePhysicalDevices(vulkan_instance, &physical_device_count, physical_devices);

    for (u32 i = 0; i < physical_device_count; ++i) {
      VkPhysicalDeviceProperties physical_device_props;
      vkGetPhysicalDeviceProperties(physical_devices[i], &physical_device_props);

      VkPhysicalDeviceFeatures physical_device_features;
      vkGetPhysicalDeviceFeatures(physical_devices[i], &physical_device_features);

      VkBool32 present_supported;
      const VkResult result = vkGetPhysicalDeviceSurfaceSupportKHR(physical_devices[i], i, surface, &present_supported);
      assert(result == VK_SUCCESS);

      u32 extension_count;
      vkEnumerateDeviceExtensionProperties(physical_devices[i], nullptr,&extension_count, nullptr);

      auto* available_extensions = STACK_ALLOCATE_UNINIT(VkExtensionProperties, extension_count);
      vkEnumerateDeviceExtensionProperties(physical_devices[i], nullptr, &extension_count, available_extensions);

      bool has_all_required_extensions = true;
      for (u32 j = 0; j < std::size(REQUIRED_DEVICE_EXTENSIONS); ++j) {
        const bool has_required_extension = std::find_if(available_extensions, available_extensions + extension_count, [&](const VkExtensionProperties &available_extension) {
          return strcmp(REQUIRED_DEVICE_EXTENSIONS[j], available_extension.extensionName) == 0;
        }) != available_extensions + extension_count;

        if (!has_required_extension) {
          has_all_required_extensions = false;
          break;
        }
      }

      // Set physical device if found an applicable one.
      if (has_all_required_extensions && physical_device_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && physical_device_features.multiViewport && present_supported) {
        // Only attempt to check if the swap-chain is suitable after determining
        // that this physical device has all the required extensions.
        u32 present_mode_count;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physical_devices[i], surface, &present_mode_count, nullptr);

        u32 format_count;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical_devices[i], surface, &format_count, nullptr);

        const bool is_swap_chain_suitable =  present_mode_count != 0 && format_count != 0;

        if (is_swap_chain_suitable) {
          physical_device = physical_devices[i];
          fmt::println("Selected physical device {}!", physical_device_props.deviceName);
          break;
        }
      }
    }

    assert(physical_device != nullptr);
  }

  fn create_logical_device() -> void {
    // Create a logical device.
    const QueueFamilies queue_families = find_queue_families();

    const f32 queue_priorities[1]{1.f};
    const VkDeviceQueueCreateInfo queue_create_info{
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = *queue_families.graphics_family,
      .queueCount = 1,
      .pQueuePriorities = queue_priorities,
    };

    const VkPhysicalDeviceFeatures device_features{};

    const VkDeviceCreateInfo create_info{
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_create_info,
      .enabledExtensionCount = static_cast<u32>(std::size(REQUIRED_DEVICE_EXTENSIONS)),
      .ppEnabledExtensionNames = REQUIRED_DEVICE_EXTENSIONS,
      .pEnabledFeatures = &device_features,
    };

    const VkResult result = vkCreateDevice(physical_device, &create_info, nullptr, &logical_device);
    assert(result == VK_SUCCESS);

    vkGetDeviceQueue(logical_device, *queue_families.graphics_family, 0, &graphics_queue);
    vkGetDeviceQueue(logical_device, *queue_families.present_family, 0, &present_queue);
    assert(graphics_queue != nullptr);
    assert(present_queue != nullptr);
  }

  fn create_swapchain() -> void {
    VkSurfaceCapabilitiesKHR surface_capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface,&surface_capabilities);

    u32 format_count, present_mode_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, nullptr);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, nullptr);

    assert(format_count > 0);
    assert(present_mode_count > 0);

    auto* formats = STACK_ALLOCATE_UNINIT(VkSurfaceFormatKHR, format_count);
    auto* present_modes = STACK_ALLOCATE_UNINIT(VkPresentModeKHR, present_mode_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, formats);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, present_modes);

    const auto &format = [&]() -> const VkSurfaceFormatKHR& {
      for (u32 i = 0; i < format_count; ++i) {
        if (formats[i].format == VK_FORMAT_B8G8R8_SRGB &&
            formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
          return formats[i];
        }
      }
      return formats[0];
    }();

    const auto present_mode = std::find(present_modes, present_modes + present_mode_count,VK_PRESENT_MODE_MAILBOX_KHR) != present_modes + present_mode_count ?
      VK_PRESENT_MODE_MAILBOX_KHR : VK_PRESENT_MODE_FIFO_KHR;

    const VkExtent2D swap_extent = [&] {
      if (surface_capabilities.currentExtent.width !=std::numeric_limits<u32>::max()) {
        return surface_capabilities.currentExtent;
      }

      i32 width, height;
      glfwGetFramebufferSize(window, &width, &height);

      VkExtent2D out_extent{
        .width = static_cast<u32>(width),
        .height = static_cast<u32>(height),
      };

      out_extent.width = std::clamp(out_extent.width, surface_capabilities.minImageExtent.width, surface_capabilities.maxImageExtent.width);
      out_extent.height = std::clamp(out_extent.height, surface_capabilities.minImageExtent.height, surface_capabilities.maxImageExtent.height);

      return out_extent;
    }();

    const u32 min_image_count = surface_capabilities.minImageCount + 1;

    VkSwapchainCreateInfoKHR create_info{
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = surface,
      .minImageCount = min_image_count,
      .imageFormat = format.format,
      .imageColorSpace = format.colorSpace,
      .imageExtent = swap_extent,
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .preTransform = surface_capabilities.currentTransform,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = present_mode,
      .clipped = VK_TRUE,
      .oldSwapchain = VK_NULL_HANDLE,
    };

    const QueueFamilies queue_families = find_queue_families();
    if (*queue_families.graphics_family != *queue_families.present_family) {
      const u32 queue_family_indices[] { *queue_families.graphics_family,*queue_families.present_family };
      create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      create_info.queueFamilyIndexCount = 2;
      create_info.pQueueFamilyIndices = queue_family_indices;
    } else {
      create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
      create_info.queueFamilyIndexCount = 0;
      create_info.pQueueFamilyIndices = nullptr;
    }

    const VkResult result = vkCreateSwapchainKHR(logical_device, &create_info, nullptr, &swapchain);
    assert(result == VK_SUCCESS);

    u32 image_count;
    vkGetSwapchainImagesKHR(logical_device, swapchain, &image_count, nullptr);

    swapchain_images.resize(image_count);
    vkGetSwapchainImagesKHR(logical_device, swapchain, &image_count, swapchain_images.data());

    // Store for use later.
    swapchain_image_format = format.format;
    swapchain_extent = swap_extent;
  }

  fn main_loop() -> void {
    assert(!!window);
    while (!glfwWindowShouldClose(window)) {
      const auto start = std::chrono::high_resolution_clock::now();

      glfwPollEvents();
      draw_frame();

      const auto end = std::chrono::high_resolution_clock::now();
      const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

      fmt::println("FPS[{}] ms[{}]", 1000000.0 / duration.count(), static_cast<double>(duration.count()) / 1000);
    }
  }

  fn cleanup_swapchain() -> void {
    for (VkFramebuffer& framebuffer : swapchain_framebuffers) {
      vkDestroyFramebuffer(logical_device, framebuffer, nullptr);
      framebuffer = VK_NULL_HANDLE;
    }
    swapchain_framebuffers.clear();

    for (VkImageView& image_view : swapchain_image_views) {
      vkDestroyImageView(logical_device, image_view, nullptr);
      image_view = VK_NULL_HANDLE;
    }
    swapchain_image_views.clear();

    vkDestroySwapchainKHR(logical_device, swapchain, nullptr);
    swapchain = VK_NULL_HANDLE;
  }

  fn recreate_swapchain() -> void {
    i32 width, height;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(window, &width, &height);
      glfwWaitEvents();
    }

    vkDeviceWaitIdle(logical_device);

    // First destroy the current swapchains.
    cleanup_swapchain();

    // Create new swapchains.
    create_swapchain();
    create_image_views();
    create_framebuffers();
  }

  fn cleanup() -> void {
    vkDeviceWaitIdle(logical_device);

    for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
      vkDestroySemaphore(logical_device, image_available_semaphores[i], nullptr);
      vkDestroySemaphore(logical_device, render_finished_semaphores[i], nullptr);
      vkDestroyFence(logical_device, in_flight_fences[i], nullptr);
    }

    vkDestroyCommandPool(logical_device, command_pool, nullptr);

    cleanup_swapchain();

    vkDestroyBuffer(logical_device, vertex_buffer, nullptr);
    vkFreeMemory(logical_device, vertex_buffer_memory, nullptr);

    vkDestroyPipeline(logical_device, graphics_pipeline, nullptr);
    vkDestroyPipelineLayout(logical_device, pipeline_layout, nullptr);
    vkDestroyRenderPass(logical_device, render_pass, nullptr);

    vkDestroyDevice(logical_device, nullptr);
    vkDestroySurfaceKHR(vulkan_instance, surface, nullptr);
    vkDestroyInstance(vulkan_instance, nullptr);

    glfwDestroyWindow(window);
    glfwTerminate();
  }

  [[nodiscard]]
  fn find_queue_families() const -> QueueFamilies {
    assert(physical_device != nullptr);

    QueueFamilies out_family;

    u32 queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);

    assert(queue_family_count > 0);

    auto* queue_family_props = STACK_ALLOCATE_UNINIT(VkQueueFamilyProperties, queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_family_props);

    for (u32 i = 0; i < queue_family_count; ++i) {
      const auto &queue_family = queue_family_props[i];
      if (!out_family.graphics_family.has_value() && queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        out_family.graphics_family = i;
      }

      if (!out_family.present_family.has_value()) {
        VkBool32 present_supported;
        vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, i, surface, &present_supported);
        if (present_supported) {
          out_family.present_family = i;
        }
      }

      if (out_family.is_valid()) {
        break;
      }
    }

    assert(out_family.is_valid());
    return out_family;
  }

  fn create_image_views() -> void {
    swapchain_image_views.resize(swapchain_images.size());
    for (i32 i = 0; i < swapchain_image_views.size(); ++i) {
      const VkImageViewCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = swapchain_images[i],
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = swapchain_image_format,
        .components{
          .r = VK_COMPONENT_SWIZZLE_IDENTITY,
          .g = VK_COMPONENT_SWIZZLE_IDENTITY,
          .b = VK_COMPONENT_SWIZZLE_IDENTITY,
          .a = VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .subresourceRange{
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .baseMipLevel = 0,
          .levelCount = 1,
          .baseArrayLayer = 0,
          .layerCount = 1,
        },
      };

      const VkResult result = vkCreateImageView(logical_device, &create_info, nullptr, &swapchain_image_views[i]);
      assert(result == VK_SUCCESS);
    }
  }

  fn create_graphics_pipeline() -> void {
    const Array<char> vert_data = read_file("shaders/bin/vert.spv");
    const Array<char> frag_data = read_file("shaders/bin/frag.spv");

    const VkShaderModule vert_shader_module = create_shader_module(vert_data);
    const VkShaderModule frag_shader_module = create_shader_module(frag_data);

    const VkPipelineShaderStageCreateInfo vert_shader_stage_create_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = vert_shader_module,
      .pName = "main",
      .pSpecializationInfo = nullptr,
    };

    const VkPipelineShaderStageCreateInfo frag_shader_stage_create_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = frag_shader_module,
      .pName = "main",
      .pSpecializationInfo = nullptr,
    };

    const VkPipelineShaderStageCreateInfo shader_stages[]{ vert_shader_stage_create_info, frag_shader_stage_create_info };

    const VkDynamicState dynamic_states[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

    const VkPipelineDynamicStateCreateInfo dynamic_state_create_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = static_cast<u32>(std::size(dynamic_states)),
      .pDynamicStates = dynamic_states,
    };

    constexpr auto vertex_binding_description = Vertex::get_binding_description();
    constexpr auto vertex_attribute_descriptions = Vertex::get_attribute_descriptions();

    const VkPipelineVertexInputStateCreateInfo vertex_input_create_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertex_binding_description,
      .vertexAttributeDescriptionCount = static_cast<u32>(vertex_attribute_descriptions.size()),
      .pVertexAttributeDescriptions = vertex_attribute_descriptions.data(),
    };

    const VkPipelineInputAssemblyStateCreateInfo input_assembly_create_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .primitiveRestartEnable = VK_FALSE,
    };

    const VkViewport viewport{
      .x = 0.f,
      .y = 0.f,
      .width = static_cast<f32>(swapchain_extent.width),
      .height = static_cast<f32>(swapchain_extent.height),
      .minDepth = 0.f,
      .maxDepth = 1.f,
    };

    const VkRect2D scissor{
      .offset{0, 0},
      .extent = swapchain_extent,
    };

    const VkPipelineViewportStateCreateInfo viewport_state{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
    };

    const VkPipelineRasterizationStateCreateInfo rasterizer_create_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .depthClampEnable = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_BACK_BIT,
      .frontFace = VK_FRONT_FACE_CLOCKWISE,
      .depthBiasEnable = VK_FALSE,
      .depthBiasConstantFactor = 0.f,
      .depthBiasClamp = 0.f,
      .depthBiasSlopeFactor = 0.f,
      .lineWidth = 1.f,
    };

    const VkPipelineMultisampleStateCreateInfo multisample_create_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .pNext = nullptr,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
      .sampleShadingEnable = VK_FALSE,
      .minSampleShading = 1.f,
      .pSampleMask = nullptr,
      .alphaToCoverageEnable = VK_FALSE,
      .alphaToOneEnable = VK_FALSE,
    };

    const VkPipelineColorBlendAttachmentState color_blend_attachment{
      .blendEnable = VK_FALSE,
      .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .alphaBlendOp = VK_BLEND_OP_ADD,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo color_blend_create_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .logicOpEnable = VK_FALSE,
      .logicOp = VK_LOGIC_OP_COPY,
      .attachmentCount = 1,
      .pAttachments = &color_blend_attachment,
      .blendConstants{
        0.f,
        0.f,
        0.f,
        0.f,
      },
    };

    const VkPipelineLayoutCreateInfo pipeline_layout_create_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 0,
      .pSetLayouts = nullptr,
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = nullptr,
    };

    const VkResult pipeline_layout_create_result =
        vkCreatePipelineLayout(logical_device, &pipeline_layout_create_info,nullptr, &pipeline_layout);
    assert(pipeline_layout_create_result == VK_SUCCESS);

    const VkGraphicsPipelineCreateInfo graphics_pipeline_create_info{
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = shader_stages,
      .pVertexInputState = &vertex_input_create_info,
      .pInputAssemblyState = &input_assembly_create_info,
      .pViewportState = &viewport_state,
      .pRasterizationState = &rasterizer_create_info,
      .pMultisampleState = &multisample_create_info,
      .pDepthStencilState = nullptr,
      .pColorBlendState = &color_blend_create_info,
      .pDynamicState = &dynamic_state_create_info,
      .layout = pipeline_layout,
      .renderPass = render_pass,
      .subpass = 0,
      .basePipelineHandle = VK_NULL_HANDLE,
      .basePipelineIndex = -1,
    };

    const VkResult graphics_pipeline_create_result = vkCreateGraphicsPipelines(logical_device, VK_NULL_HANDLE, 1, &graphics_pipeline_create_info,nullptr, &graphics_pipeline);
    assert(graphics_pipeline_create_result == VK_SUCCESS);

    vkDestroyShaderModule(logical_device, vert_shader_module, nullptr);
    vkDestroyShaderModule(logical_device, frag_shader_module, nullptr);
  }

  fn create_shader_module(const Span<const char> data) const -> VkShaderModule {
    const VkShaderModuleCreateInfo create_info{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = data.size(),
      .pCode = reinterpret_cast<const u32*>(data.data()),
    };

    VkShaderModule shader_module;
    const VkResult result = vkCreateShaderModule(logical_device, &create_info, nullptr, &shader_module);
    assert(result == VK_SUCCESS);

    return shader_module;
  }

  fn create_render_pass() -> void {
    const VkAttachmentDescription color_attachment{
      .format = swapchain_image_format,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    const VkAttachmentReference color_attachment_reference{
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    const VkSubpassDescription subpass{
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_attachment_reference,
    };

    const VkSubpassDependency dependency{
      .srcSubpass = VK_SUBPASS_EXTERNAL,
      .dstSubpass = 0,
      .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .srcAccessMask = 0,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    const VkRenderPassCreateInfo render_pass_create_info{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &color_attachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 1,
      .pDependencies = &dependency,
    };

    const VkResult render_pass_create_result = vkCreateRenderPass(logical_device, &render_pass_create_info, nullptr, &render_pass);
    assert(render_pass_create_result == VK_SUCCESS);
  }

  fn create_framebuffers() -> void {
    swapchain_framebuffers.resize(swapchain_image_views.size());
    for (usize i = 0; i < swapchain_framebuffers.size(); ++i) {
      const VkFramebufferCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = render_pass,
        .attachmentCount = 1,
        .pAttachments = &swapchain_image_views[i],
        .width = swapchain_extent.width,
        .height = swapchain_extent.height,
        .layers = 1,
      };

      const VkResult result = vkCreateFramebuffer(logical_device, &create_info, nullptr, &swapchain_framebuffers[i]);
      assert(result == VK_SUCCESS);
    }
  }

  fn create_command_pool() -> void {
    const QueueFamilies queue_families = find_queue_families();

    const VkCommandPoolCreateInfo create_info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = *queue_families.graphics_family,
    };

    const VkResult result = vkCreateCommandPool(logical_device, &create_info, nullptr, &command_pool);
    assert(result == VK_SUCCESS);
  }

  fn create_command_buffers() -> void {
    const VkCommandBufferAllocateInfo create_info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
    };

    const VkResult result = vkAllocateCommandBuffers(logical_device, &create_info, command_buffers.data());
    assert(result == VK_SUCCESS);
  }

  fn record_command_buffer(const VkCommandBuffer out_command_buffer, const u32 image_index) const -> void {
    assert(image_index < swapchain_framebuffers.size());

    const VkCommandBufferBeginInfo command_buffer_begin_info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = 0,
      .pInheritanceInfo = nullptr,
    };

    const VkResult begin_result = vkBeginCommandBuffer(out_command_buffer, &command_buffer_begin_info);
    assert(begin_result == VK_SUCCESS);

    const VkClearValue clear_color{
      .color{
        0.f,
        0.f,
        0.f,
        0.f,
      },
    };

    const VkRenderPassBeginInfo render_pass_begin_info{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass,
      .framebuffer = swapchain_framebuffers[image_index],
      .renderArea{
        .offset{0, 0},
        .extent = swapchain_extent,
      },
      .clearValueCount = 1,
      .pClearValues = &clear_color,
    };

    //~
    // Render pass.
    vkCmdBeginRenderPass(out_command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(out_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

    const VkViewport viewport{
      .x = 0.f,
      .y = 0.f,
      .width = static_cast<f32>(swapchain_extent.width),
      .height = static_cast<f32>(swapchain_extent.height),
      .minDepth = 0.f,
      .maxDepth = 0.f,
    };
    vkCmdSetViewport(out_command_buffer, 0, 1, &viewport);

    const VkRect2D scissor{
      .offset{0, 0},
      .extent = swapchain_extent,
    };
    vkCmdSetScissor(out_command_buffer, 0, 1, &scissor);

    const VkDeviceSize offsets[]{ 0 };
    vkCmdBindVertexBuffers(out_command_buffer, 0, 1, &vertex_buffer, offsets);

    vkCmdBindIndexBuffer(out_command_buffer, index_buffer, offsets[0], VK_INDEX_TYPE_UINT16);

    //vkCmdDraw(out_command_buffer, static_cast<u32>(VERTICES.size()), 1, 0, 0);
    vkCmdDrawIndexed(out_command_buffer, static_cast<u32>(INDICES.size()), 1, 0, 0, 0);

    vkCmdDraw(out_command_buffer, 3, 1, 0, 0);

    vkCmdEndRenderPass(out_command_buffer);
    //~

    const VkResult end_command_buffer_result = vkEndCommandBuffer(out_command_buffer);
    assert(end_command_buffer_result == VK_SUCCESS);
  }

  fn create_vertex_buffer() -> void {
    constexpr VkDeviceSize BUFFER_SIZE = sizeof(Vertex) * VERTICES.size();
    constexpr VkMemoryPropertyFlags MEMORY_REQUIREMENTS = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    create_buffer_from_data(vertex_buffer, vertex_buffer_memory, Span<const char>{reinterpret_cast<const char*>(VERTICES.data()), BUFFER_SIZE}, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);



#if 0
#if 01
    // Staging buffer allocation.
    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;
    create_buffer(BUFFER_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);

    void* data;
    vkMapMemory(logical_device, staging_buffer_memory, 0, BUFFER_SIZE, 0, &data);

    memcpy(data, VERTICES.data(), BUFFER_SIZE);

    vkUnmapMemory(logical_device, staging_buffer_memory);

    create_buffer(BUFFER_SIZE, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertex_buffer, vertex_buffer_memory);

    copy_buffer(vertex_buffer, staging_buffer, BUFFER_SIZE);

    vkDestroyBuffer(logical_device, staging_buffer, nullptr);
    vkFreeMemory(logical_device, staging_buffer_memory, nullptr);

#else
    create_buffer(BUFFER_SIZE, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, MEMORY_REQUIREMENTS, vertex_buffer, vertex_buffer_memory);

    void* data;
    vkMapMemory(logical_device, vertex_buffer_memory, 0, BUFFER_SIZE, 0, &data);

    memcpy(data, vertices.data(), static_cast<usize>(BUFFER_SIZE));

    vkUnmapMemory(logical_device, vertex_buffer_memory);
#endif
#endif
  }

  fn create_index_buffer() -> void {
    constexpr VkDeviceSize BUFFER_SIZE = INDICES.size() * sizeof(u16);

    create_buffer_from_data(index_buffer, index_buffer_memory, Span<const char>{reinterpret_cast<const char*>(INDICES.data()), BUFFER_SIZE}, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  }

  fn create_buffer_from_data(VkBuffer& out_buffer, VkDeviceMemory& out_buffer_memory, const Span<const char> src_data, const VkBufferUsageFlags usage, const VkMemoryPropertyFlags properties) const -> void {
    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;
    create_buffer(static_cast<VkDeviceSize>(src_data.size()), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);

    void* data;
    vkMapMemory(logical_device, staging_buffer_memory, 0, static_cast<VkDeviceSize>(src_data.size()), 0, &data);

    memcpy(data, src_data.data(), src_data.size());

    vkUnmapMemory(logical_device, staging_buffer_memory);

    create_buffer(src_data.size(), usage, properties, out_buffer, out_buffer_memory);

    copy_buffer(out_buffer, staging_buffer, static_cast<VkDeviceSize>(src_data.size()));

    vkDestroyBuffer(logical_device, staging_buffer, nullptr);
    vkFreeMemory(logical_device, staging_buffer_memory, nullptr);
  }

  fn find_memory_type(const u32 filter_type, const VkMemoryPropertyFlags properties) const -> u32 {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);

    for (u32 i = 0; i < mem_props.memoryTypeCount; ++i) {
      if ((filter_type & 1 << i) && (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
        return i;
      }
    }

    assert(false);
    return UINT32_MAX;
  }

  fn copy_buffer(const VkBuffer dst, const VkBuffer src, const VkDeviceSize size) const -> void {
    assert(command_pool != VK_NULL_HANDLE);

    const VkCommandBufferAllocateInfo command_buffer_allocate_info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
    };

    VkCommandBuffer command_buffer;
    vkAllocateCommandBuffers(logical_device, &command_buffer_allocate_info, &command_buffer);

    const VkCommandBufferBeginInfo begin_info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(command_buffer, &begin_info);// Begin recording.

    const VkBufferCopy copy_region{
      .srcOffset = 0,
      .dstOffset = 0,
      .size = size,
    };
    vkCmdCopyBuffer(command_buffer, src, dst, 1, &copy_region);

    vkEndCommandBuffer(command_buffer);

    const VkSubmitInfo submit_info{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
    };
    vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphics_queue);

    vkFreeCommandBuffers(logical_device, command_pool, 1, &command_buffer);
  }

  fn create_sync_objects() -> void {
    const VkSemaphoreCreateInfo semaphore_create_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };

    const VkFenceCreateInfo fence_create_info{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
      VkResult result = vkCreateSemaphore(logical_device, &semaphore_create_info, nullptr, &image_available_semaphores[i]);
      assert(result == VK_SUCCESS);

      result = vkCreateSemaphore(logical_device, &semaphore_create_info, nullptr, &render_finished_semaphores[i]);
      assert(result == VK_SUCCESS);

      result = vkCreateFence(logical_device, &fence_create_info, nullptr, &in_flight_fences[i]);
      assert(result == VK_SUCCESS);
    }
  }

  fn create_buffer(const VkDeviceSize size, const VkBufferUsageFlags usage, const VkMemoryPropertyFlags properties, VkBuffer& out_buffer, VkDeviceMemory& out_buffer_memory) const -> void {
    const VkBufferCreateInfo buffer_create_info{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    VkResult result = vkCreateBuffer(logical_device, &buffer_create_info, nullptr, &out_buffer);
    assert(result == VK_SUCCESS);

    VkMemoryRequirements memory_requirements;
    vkGetBufferMemoryRequirements(logical_device, out_buffer, &memory_requirements);

    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    u32 memory_type = UINT32_MAX;
    for (u32 i = 0; i < memory_properties.memoryTypeCount; ++i) {
      if ((memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
        memory_type = i;
        break;
      }
    }
    assert(memory_type != UINT32_MAX);

    const VkMemoryAllocateInfo alloc_create_info{
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = memory_requirements.size,
      .memoryTypeIndex = memory_type,
    };

    result = vkAllocateMemory(logical_device, &alloc_create_info, nullptr, &out_buffer_memory);
    assert(result == VK_SUCCESS);

    vkBindBufferMemory(logical_device, out_buffer, out_buffer_memory, 0);
  }

  fn draw_frame() -> void {
    const VkResult wait_for_in_flight_fence_result = vkWaitForFences(logical_device, 1, &in_flight_fences[current_frame], VK_TRUE, UINT64_MAX);
    assert(wait_for_in_flight_fence_result == VK_SUCCESS);

    const VkResult reset_in_flight_fence_result = vkResetFences(logical_device, 1, &in_flight_fences[current_frame]);
    assert(reset_in_flight_fence_result == VK_SUCCESS);

    u32 image_index;
    const VkResult acquire_next_image_index_result = vkAcquireNextImageKHR(logical_device, swapchain, UINT64_MAX, image_available_semaphores[current_frame], VK_NULL_HANDLE, &image_index);
    if (acquire_next_image_index_result == VK_ERROR_OUT_OF_DATE_KHR || acquire_next_image_index_result == VK_SUBOPTIMAL_KHR) {
      recreate_swapchain();
      return;
    } else {
      assert(acquire_next_image_index_result == VK_SUCCESS);
    }

    // Only reset the fence if we are submitting work
    vkResetFences(logical_device, 1, &in_flight_fences[current_frame]);

    const VkResult reset_command_buffer_result = vkResetCommandBuffer(command_buffers[current_frame], 0);
    assert(reset_command_buffer_result == VK_SUCCESS);

    record_command_buffer(command_buffers[current_frame], image_index);

    // Submit command buffer.
    const VkPipelineStageFlags wait_stages[]{ VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    const VkSubmitInfo submit_info{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,

      // Only run after an image is available to display.
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &image_available_semaphores[current_frame],

      .pWaitDstStageMask = wait_stages,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffers[current_frame],

      // Signal rendering has finished on completion.
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &render_finished_semaphores[current_frame],
    };

    const VkResult queue_submit_result = vkQueueSubmit(graphics_queue, 1, &submit_info, in_flight_fences[current_frame]);
    assert(queue_submit_result == VK_SUCCESS);

    const VkPresentInfoKHR present_info{
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &render_finished_semaphores[current_frame],
      .swapchainCount = 1,
      .pSwapchains = &swapchain,
      .pImageIndices = &image_index,
      .pResults = nullptr,
    };

    const VkResult present_result = vkQueuePresentKHR(present_queue, &present_info);
    if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR) {
      recreate_swapchain();
      return;
    } else {
      assert(present_result == VK_SUCCESS);
    }

    // Advance the current frame.
    current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  GLFWwindow* window = nullptr;
  VkInstance vulkan_instance = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice logical_device = VK_NULL_HANDLE;

  // @NOTE: Both graphics_queue and present_queue can point to the same queue.
  VkQueue graphics_queue = VK_NULL_HANDLE;
  VkQueue present_queue = VK_NULL_HANDLE;

  VkSwapchainKHR swapchain = VK_NULL_HANDLE;

  Array<VkImage> swapchain_images;
  VkFormat swapchain_image_format;
  VkExtent2D swapchain_extent;

  Array<VkImageView> swapchain_image_views;

  VkRenderPass render_pass = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline graphics_pipeline = VK_NULL_HANDLE;

  Array<VkFramebuffer> swapchain_framebuffers;

  VkCommandPool command_pool = VK_NULL_HANDLE;

  VkBuffer vertex_buffer = VK_NULL_HANDLE;
  VkDeviceMemory vertex_buffer_memory = VK_NULL_HANDLE;

  VkBuffer index_buffer = VK_NULL_HANDLE;
  VkDeviceMemory index_buffer_memory = VK_NULL_HANDLE;

  StaticArray<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> command_buffers;
  StaticArray<VkSemaphore, MAX_FRAMES_IN_FLIGHT> image_available_semaphores;
  StaticArray<VkSemaphore, MAX_FRAMES_IN_FLIGHT> render_finished_semaphores;
  StaticArray<VkFence, MAX_FRAMES_IN_FLIGHT> in_flight_fences;
  u32 current_frame = 0;
};

fn main() -> i32 {
  App app;
  app.run();

  return EXIT_SUCCESS;
}