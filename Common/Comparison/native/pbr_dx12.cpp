// Headless D3D12 renderer. C ABI keeps Python independent of C++ binding packages.
#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>

using Microsoft::WRL::ComPtr;
using namespace DirectX;
namespace {
constexpr UINT shadowSize = 1024;
struct Mip { const float* data; UINT width, height; };
struct Texture { const Mip* levels; UINT count, channels; };
struct Geometry { const float* data; UINT count; };
struct PointLight { XMFLOAT3 position; float power; XMFLOAT3 color; float padding; };
struct View {
    UINT width, height; XMFLOAT2 center; float zoom, mipBias; UINT normalMode;
    XMFLOAT3 eye, target; float fov; PointLight sun;
};
static_assert(sizeof(PointLight) == 32 && sizeof(View) == 88, "Python/native ABI layout mismatch");
struct Light { XMFLOAT4X4 vp; XMFLOAT4 positionPower, color; };
struct Camera {
    XMFLOAT4X4 vp; XMFLOAT4 eye;
    float mipBias; UINT normalMode, pointCount; float padding;
};

void validateTextures(const Texture* textures) {
    if (!textures) return;  // Later viewports reuse the already uploaded material.
    for (UINT slot = 0; slot < 6; ++slot) {
        const auto& texture = textures[slot];
        if (!texture.count || !texture.levels) throw std::runtime_error("Material requires mip levels");
        if (texture.channels != 1 && texture.channels != 2 && texture.channels != 4)
            throw std::runtime_error("Invalid material channel count");
        UINT width = texture.levels[0].width, height = texture.levels[0].height;
        if (!width || !height || width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION || height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION)
            throw std::runtime_error("Invalid material texture dimensions");
        UINT maxMips = 1;
        for (UINT size = std::max(width, height); size > 1; size /= 2) ++maxMips;
        if (texture.count > maxMips) throw std::runtime_error("Too many material mip levels");
        for (UINT mip = 0; mip < texture.count; ++mip) {
            const auto& level = texture.levels[mip];
            if (!level.data || level.width != width || level.height != height)
                throw std::runtime_error("Material mip dimensions must follow D3D12 floor-halving");
            width = std::max(1u, width / 2); height = std::max(1u, height / 2);
        }
    }
}

void check(HRESULT hr, const char* operation) {
    if (FAILED(hr)) {
        char message[256];
        snprintf(message, sizeof(message), "%s failed (HRESULT 0x%08lX)", operation, static_cast<unsigned long>(hr));
        throw std::runtime_error(message);
    }
}
void report(char* error, size_t capacity, const std::exception& exception) {
    if (capacity) { strncpy_s(error, capacity, exception.what(), _TRUNCATE); }
}
std::string utf8(const wchar_t* input) {
    int size = WideCharToMultiByte(CP_UTF8, 0, input, -1, nullptr, 0, nullptr, nullptr);
    std::string value(size, 0);
    WideCharToMultiByte(CP_UTF8, 0, input, -1, value.data(), size, nullptr, nullptr);
    value.pop_back();
    return value;
}
D3D12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES value{}; value.Type = type; value.CreationNodeMask = value.VisibleNodeMask = 1;
    return value;
}
D3D12_RESOURCE_DESC textureDesc(UINT width, UINT height, DXGI_FORMAT format, UINT levels = 1, UINT slices = 1) {
    D3D12_RESOURCE_DESC value{};
    value.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; value.Width = width; value.Height = height;
    value.DepthOrArraySize = static_cast<UINT16>(slices); value.MipLevels = static_cast<UINT16>(levels);
    value.Format = format; value.SampleDesc.Count = 1;
    return value;
}
void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    list->ResourceBarrier(1, &barrier);
}
ComPtr<ID3DBlob> compile(const char* source, size_t size, const char* entry, const char* profile) {
    ComPtr<ID3DBlob> bytecode, errors;
    HRESULT hr = D3DCompile(source, size, "pbr.hlsl", nullptr, nullptr, entry, profile,
                           D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &bytecode, &errors);
    if (FAILED(hr)) throw std::runtime_error(errors ? static_cast<const char*>(errors->GetBufferPointer()) : "HLSL compilation failed");
    return bytecode;
}

class Renderer {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12InfoQueue> debugQueue;
    HANDLE event = nullptr;
    UINT64 serial = 0;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> shadingPipeline, shadowPipeline;
    ComPtr<ID3D12DescriptorHeap> resources, targets, depths;
    ComPtr<ID3D12Resource> vertices, displacedVertices, constants, lights, shadows;
    struct Target {
        ComPtr<ID3D12Resource> output, depth, readback;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    };
    std::map<std::pair<UINT, UINT>, Target> viewTargets;
    struct Material {
        ComPtr<ID3D12Resource> texture;
        std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprints;
        UINT64 uploadBytes;
    };
    // At most three layouts per channel: floating inference, compressed
    // inference (BC5 may use two normal channels), and GT resolution.
    std::array<std::vector<Material>, 6> materials;
    std::array<Material*, 6> selectedMaterials{};
    std::array<ComPtr<ID3D12Resource>, 6> materialUploads;
    D3D12_VERTEX_BUFFER_VIEW vertexView{};
    UINT vertexCount = 0, baseVertexCount = 0, shadowCount = 0;
    UINT srvStride = 0, dsvStride = 0;
    std::vector<unsigned char> lightingKey;
    std::vector<Light> lightRecords;

    ComPtr<ID3D12Resource> buffer(UINT64 bytes, D3D12_HEAP_TYPE type) {
        D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = std::max<UINT64>(bytes, 256); desc.Height = 1; desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        auto properties = heap(type);
        auto state = type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST;
        ComPtr<ID3D12Resource> value;
        check(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                              IID_PPV_ARGS(&value)), "Create buffer");
        return value;
    }
    ComPtr<ID3D12Resource> image(D3D12_RESOURCE_DESC desc, D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE* clear = nullptr) {
        auto properties = heap(D3D12_HEAP_TYPE_DEFAULT);
        ComPtr<ID3D12Resource> value;
        check(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, state, clear,
                                              IID_PPV_ARGS(&value)), "Create texture");
        return value;
    }
    ComPtr<ID3D12DescriptorHeap> descriptors(D3D12_DESCRIPTOR_HEAP_TYPE type, UINT count, bool visible = false) {
        D3D12_DESCRIPTOR_HEAP_DESC desc{}; desc.Type = type; desc.NumDescriptors = count;
        desc.Flags = visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        ComPtr<ID3D12DescriptorHeap> value;
        check(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&value)), "Create descriptor heap");
        return value;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE srv(UINT index) {
        auto handle = resources->GetCPUDescriptorHandleForHeapStart(); handle.ptr += SIZE_T(index) * srvStride; return handle;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE dsv(UINT index) {
        auto handle = depths->GetCPUDescriptorHandleForHeapStart(); handle.ptr += SIZE_T(index) * dsvStride; return handle;
    }
    void begin() {
        check(allocator->Reset(), "Reset allocator");
        check(commands->Reset(allocator.Get(), nullptr), "Reset command list");
        commands->SetGraphicsRootSignature(root.Get());
        ID3D12DescriptorHeap* heaps[] = {resources.Get()}; commands->SetDescriptorHeaps(1, heaps);
        commands->SetGraphicsRootConstantBufferView(0, constants->GetGPUVirtualAddress());
        commands->SetGraphicsRootDescriptorTable(2, resources->GetGPUDescriptorHandleForHeapStart());
        commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commands->IASetVertexBuffers(0, 1, &vertexView);
    }
    void submit() {
        check(commands->Close(), "Close command list");
        ID3D12CommandList* lists[] = {commands.Get()}; queue->ExecuteCommandLists(1, lists);
        check(queue->Signal(fence.Get(), ++serial), "Signal queue fence");
        if (fence->GetCompletedValue() < serial) {
            check(fence->SetEventOnCompletion(serial, event), "Set fence completion");
            if (WaitForSingleObject(event, 60000) != WAIT_OBJECT_0) throw std::runtime_error("D3D12 GPU fence timed out");
        }
        check(device->GetDeviceRemovedReason(), "D3D12 device");
        if (debugQueue) {
            std::string errors;
            for (UINT64 index = 0; index < debugQueue->GetNumStoredMessages(); ++index) {
                SIZE_T bytes = 0; debugQueue->GetMessage(index, nullptr, &bytes);
                std::vector<unsigned char> storage(bytes);
                auto message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
                check(debugQueue->GetMessage(index, message, &bytes), "Read D3D12 validation");
                if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) errors += std::string(message->pDescription) + '\n';
            }
            debugQueue->ClearStoredMessages();
            if (!errors.empty()) throw std::runtime_error("D3D12 validation errors:\n" + errors);
        }
    }
    void viewport(UINT width, UINT height) {
        D3D12_VIEWPORT viewport{0, 0, float(width), float(height), 0, 1};
        D3D12_RECT scissor{0, 0, LONG(width), LONG(height)};
        commands->RSSetViewports(1, &viewport); commands->RSSetScissorRects(1, &scissor);
    }
    void pipelines(const char* shader, size_t size) {
        D3D12_DESCRIPTOR_RANGE range{};
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; range.NumDescriptors = 8;
        D3D12_ROOT_PARAMETER parameters[3]{};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameters[0].Descriptor.ShaderRegister = 0;
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[1].Constants = {1, 0, 1};
        parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[2].DescriptorTable = {1, &range};
        D3D12_STATIC_SAMPLER_DESC samplers[2]{};
        samplers[0].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        samplers[0].AddressU = samplers[0].AddressV = samplers[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        samplers[0].MaxLOD = D3D12_FLOAT32_MAX; samplers[0].ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        samplers[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        samplers[1] = samplers[0]; samplers[1].ShaderRegister = 1;
        samplers[1].Filter = D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
        samplers[1].AddressU = samplers[1].AddressV = samplers[1].AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
        samplers[1].BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
        samplers[1].ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
        D3D12_ROOT_SIGNATURE_DESC desc{3, parameters, 2, samplers, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
        ComPtr<ID3DBlob> blob, errors;
        check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors), "Serialize root signature");
        check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "Create root signature");
        auto vs = compile(shader, size, "VS", "vs_5_1"), ps = compile(shader, size, "PS", "ps_5_1");
        auto shadowVS = compile(shader, size, "ShadowVS", "vs_5_1");
        D3D12_INPUT_ELEMENT_DESC layout[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TANGENT", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 40, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root.Get(); pso.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
        pso.PS = {ps->GetBufferPointer(), ps->GetBufferSize()}; pso.InputLayout = {layout, 4};
        pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID; pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso.RasterizerState.DepthClipEnable = TRUE;
        pso.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso.SampleMask = UINT_MAX; pso.SampleDesc.Count = 1;
        pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso.NumRenderTargets = 1; pso.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT; pso.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        pso.DepthStencilState.DepthEnable = TRUE;
        pso.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        pso.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
        check(device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&shadingPipeline)), "Create shading pipeline");
        pso.VS = {shadowVS->GetBufferPointer(), shadowVS->GetBufferSize()}; pso.PS = {};
        pso.NumRenderTargets = 0; pso.RTVFormats[0] = DXGI_FORMAT_UNKNOWN;
        pso.RasterizerState.DepthBias = 10; pso.RasterizerState.SlopeScaledDepthBias = 1.5f;
        check(device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&shadowPipeline)), "Create shadow pipeline");
    }
    void prepareMaterial(UINT slot, const Texture& texture) {
        DXGI_FORMAT format = texture.channels == 1 ? DXGI_FORMAT_R32_FLOAT :
                             texture.channels == 2 ? DXGI_FORMAT_R32G32_FLOAT : DXGI_FORMAT_R32G32B32A32_FLOAT;
        auto desc = textureDesc(texture.levels[0].width, texture.levels[0].height, format, texture.count);
        auto& entries = materials[slot];
        auto found = std::find_if(entries.begin(), entries.end(), [&](const Material& entry) {
            auto previous = entry.texture->GetDesc();
            return previous.Width == desc.Width && previous.Height == desc.Height &&
                   previous.MipLevels == desc.MipLevels && previous.Format == format;
        });
        if (found == entries.end()) {
            Material entry;
            entry.texture = image(desc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            entry.footprints.resize(texture.count);
            device->GetCopyableFootprints(&desc, 0, texture.count, 0, entry.footprints.data(), nullptr, nullptr, &entry.uploadBytes);
            if (entries.size() == 3) entries.erase(entries.begin());
            entries.push_back(std::move(entry));
        } else std::rotate(found, found + 1, entries.end());
        auto& material = entries.back();
        selectedMaterials[slot] = &material;
        auto& upload = materialUploads[slot];
        if (!upload || upload->GetDesc().Width < material.uploadBytes) upload = buffer(material.uploadBytes, D3D12_HEAP_TYPE_UPLOAD);
        unsigned char* mapped; D3D12_RANGE noRead{0, 0};
        check(upload->Map(0, &noRead, reinterpret_cast<void**>(&mapped)), "Map material upload");
        for (UINT mip = 0; mip < texture.count; ++mip) {
            const auto& level = texture.levels[mip]; const auto& footprint = material.footprints[mip];
            for (UINT row = 0; row < level.height; ++row)
                memcpy(mapped + footprint.Offset + SIZE_T(row) * footprint.Footprint.RowPitch,
                       level.data + SIZE_T(row) * level.width * texture.channels, SIZE_T(level.width) * texture.channels * sizeof(float));
        }
        upload->Unmap(0, nullptr);
        D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format = format; view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels = texture.count;
        device->CreateShaderResourceView(material.texture.Get(), &view, srv(slot));
    }
    void uploadMaterial(UINT slot) {
        auto& material = *selectedMaterials[slot];
        auto* resource = material.texture.Get();
        transition(commands.Get(), resource, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        for (UINT mip = 0; mip < material.footprints.size(); ++mip) {
            D3D12_TEXTURE_COPY_LOCATION source{}; source.pResource = materialUploads[slot].Get();
            source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; source.PlacedFootprint = material.footprints[mip];
            D3D12_TEXTURE_COPY_LOCATION target{}; target.pResource = resource;
            target.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; target.SubresourceIndex = mip;
            commands->CopyTextureRegion(&target, 0, 0, 0, &source, nullptr);
        }
        transition(commands.Get(), resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }
    bool prepareLights(const View& settings, const PointLight* points, UINT count) {
        if (count > 341) throw std::runtime_error("D3D12 shadow texture arrays support at most 341 point lights");
        std::vector<unsigned char> key(sizeof(PointLight) * (count + 1));
        memcpy(key.data(), &settings.sun, sizeof(PointLight));
        if (count) memcpy(key.data() + sizeof(PointLight), points, count * sizeof(PointLight));
        if (key == lightingKey && !lightRecords.empty()) return false;
        lightingKey = key;
        lightRecords.resize(1 + count * 6);
        XMVECTOR center = XMVectorSet(0, 0, 1.1f, 1), up = XMVectorSet(0, 1, 0, 0);
        XMVECTOR direction = XMVectorSet(settings.sun.position.x, settings.sun.position.y, settings.sun.position.z, 0);
        if (XMVectorGetX(XMVector3LengthSq(direction)) < 1e-8f) direction = XMVectorSet(0, 1, -1, 0);
        direction = XMVector3Normalize(direction);
        if (std::abs(XMVectorGetX(XMVector3Dot(direction, up))) > .95f) up = XMVectorSet(0, 0, 1, 0);
        XMStoreFloat4x4(&lightRecords[0].vp, XMMatrixLookAtLH(center + direction * 7, center, up) * XMMatrixOrthographicLH(8, 8, .1f, 16));
        XMStoreFloat4(&lightRecords[0].positionPower, XMVectorSetW(direction, settings.sun.power));
        lightRecords[0].color = {settings.sun.color.x, settings.sun.color.y, settings.sun.color.z, 1};
        const float axes[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
        const float ups[6][3] = {{0,1,0},{0,1,0},{0,0,-1},{0,0,1},{0,1,0},{0,1,0}};
        // Overlap cube faces by four texels per edge. The 3x3 PCF kernel plus
        // bilinear comparison taps must see real neighboring depth, not the
        // white sampler border that otherwise leaks light along face seams.
        const float pointFov = 2 * std::atan(1.f / (1.f - 8.f / shadowSize));
        for (UINT index = 0; index < count; ++index) {
            const auto& point = points[index]; auto position = XMLoadFloat3(&point.position);
            for (UINT face = 0; face < 6; ++face) {
                auto& record = lightRecords[1 + index * 6 + face];
                auto axis = XMVectorSet(axes[face][0], axes[face][1], axes[face][2], 0);
                auto faceUp = XMVectorSet(ups[face][0], ups[face][1], ups[face][2], 0);
                XMStoreFloat4x4(&record.vp, XMMatrixLookAtLH(position, position + axis, faceUp) * XMMatrixPerspectiveFovLH(pointFov, 1, .025f, 12));
                record.positionPower = {point.position.x, point.position.y, point.position.z, point.power}; record.color = {point.color.x, point.color.y, point.color.z, 1};
            }
        }
        lights = buffer(lightRecords.size() * sizeof(Light), D3D12_HEAP_TYPE_UPLOAD);
        void* mapped; D3D12_RANGE noRead{0,0}; check(lights->Map(0, &noRead, &mapped), "Map lights");
        memcpy(mapped, lightRecords.data(), lightRecords.size() * sizeof(Light)); lights->Unmap(0, nullptr);
        D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.Buffer.NumElements = static_cast<UINT>(lightRecords.size()); view.Buffer.StructureByteStride = sizeof(Light);
        device->CreateShaderResourceView(lights.Get(), &view, srv(7));
        return true;
    }
    Target& target(UINT width, UINT height, UINT maps) {
        if (maps != shadowCount) {
            shadowCount = maps; depths = descriptors(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, maps + 1);
            auto desc = textureDesc(shadowSize, shadowSize, DXGI_FORMAT_R32_TYPELESS, 1, maps);
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
            D3D12_CLEAR_VALUE clear{}; clear.Format = DXGI_FORMAT_D32_FLOAT; clear.DepthStencil.Depth = 1;
            shadows = image(desc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &clear);
            for (UINT index = 0; index < maps; ++index) {
                D3D12_DEPTH_STENCIL_VIEW_DESC view{}; view.Format = DXGI_FORMAT_D32_FLOAT;
                view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
                view.Texture2DArray.FirstArraySlice = index; view.Texture2DArray.ArraySize = 1;
                device->CreateDepthStencilView(shadows.Get(), &view, dsv(index));
            }
            D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format = DXGI_FORMAT_R32_FLOAT;
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY; view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            view.Texture2DArray.MipLevels = 1; view.Texture2DArray.ArraySize = maps;
            device->CreateShaderResourceView(shadows.Get(), &view, srv(6));
        }
        const auto dimensions = std::make_pair(width, height);
        // Retain the Scene/Detail pair, releasing old resolutions after changes.
        if (viewTargets.size() == 2 && !viewTargets.count(dimensions)) viewTargets.clear();
        auto [iterator, inserted] = viewTargets.try_emplace(dimensions);
        auto& target = iterator->second;
        if (inserted) {
            auto desc = textureDesc(width, height, DXGI_FORMAT_R32G32B32A32_FLOAT);
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            D3D12_CLEAR_VALUE clear{}; clear.Format = desc.Format;
            clear.Color[0] = clear.Color[1] = clear.Color[2] = .11f; clear.Color[3] = 1;
            target.output = image(desc, D3D12_RESOURCE_STATE_COPY_SOURCE, &clear);
            UINT64 bytes;
            device->GetCopyableFootprints(&desc, 0, 1, 0, &target.footprint, nullptr, nullptr, &bytes);
            target.readback = buffer(bytes, D3D12_HEAP_TYPE_READBACK);
            desc = textureDesc(width, height, DXGI_FORMAT_D32_FLOAT);
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
            clear = {}; clear.Format = desc.Format; clear.DepthStencil.Depth = 1;
            target.depth = image(desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear);
        }
        device->CreateRenderTargetView(target.output.Get(), nullptr, targets->GetCPUDescriptorHandleForHeapStart());
        device->CreateDepthStencilView(target.depth.Get(), nullptr, dsv(maps));
        return target;
    }
public:
    std::string adapterName;
    Renderer(const float* data, UINT count, const char* shader, size_t shaderSize) {
        if (GetEnvironmentVariableA("FNTC_DX12_DEBUG", nullptr, 0)) {
            ComPtr<ID3D12Debug> debug;
            check(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)), "Enable D3D12 debug layer (Windows Graphics Tools required)");
            debug->EnableDebugLayer();
        }
        ComPtr<IDXGIFactory6> factory;
        check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "Create DXGI factory");
        for (UINT index = 0; ; ++index) {
            ComPtr<IDXGIAdapter1> adapter;
            if (factory->EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 desc; check(adapter->GetDesc1(&desc), "Read adapter");
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
                adapterName = utf8(desc.Description); break;
            }
        }
        if (!device) throw std::runtime_error("No hardware DirectX 12 adapter is available");
        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        check(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)), "Create command queue");
        check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "Create command allocator");
        check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&commands)), "Create command list");
        check(commands->Close(), "Close initial command list");
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "Create fence");
        resources = descriptors(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8, true);
        targets = descriptors(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1);
        srvStride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        dsvStride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        pipelines(shader, shaderSize);
        device.As(&debugQueue);
        constants = buffer(256, D3D12_HEAP_TYPE_UPLOAD);
        vertices = buffer(SIZE_T(count) * 48, D3D12_HEAP_TYPE_UPLOAD);
        void* mapped; D3D12_RANGE noRead{0,0}; check(vertices->Map(0, &noRead, &mapped), "Map vertices");
        if (count) memcpy(mapped, data, SIZE_T(count) * 48); vertices->Unmap(0, nullptr);
        vertexCount = baseVertexCount = count; vertexView = {vertices->GetGPUVirtualAddress(), count * 48, 48};
        event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (!event) throw std::runtime_error("Create GPU fence event failed");
    }
    ~Renderer() { if (event) CloseHandle(event); }

    void render(const Texture* textures, const View& settings, const PointLight* points, UINT count,
                const Geometry* geometry, float* result) {
        if (settings.width < 2 || settings.height < 2 || settings.width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION || settings.height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION) throw std::runtime_error("Invalid D3D12 render resolution");
        bool updateShadows = prepareLights(settings, points, count);
        if (geometry) {
            vertexCount = geometry->count ? geometry->count : baseVertexCount;
            if (geometry->count) {
                SIZE_T bytes = SIZE_T(vertexCount) * 48;
                if (!displacedVertices || displacedVertices->GetDesc().Width < bytes)
                    displacedVertices = buffer(bytes, D3D12_HEAP_TYPE_UPLOAD);
                void* mapped; D3D12_RANGE noRead{0, 0};
                check(displacedVertices->Map(0, &noRead, &mapped), "Map displaced vertices");
                memcpy(mapped, geometry->data, bytes); displacedVertices->Unmap(0, nullptr);
            }
            auto* resource = geometry->count ? displacedVertices.Get() : vertices.Get();
            vertexView = {resource->GetGPUVirtualAddress(), vertexCount * 48, 48};
            updateShadows = true;
        }
        auto& viewportTarget = target(settings.width, settings.height, 1 + count * 6);
        Camera camera{};
        XMVECTOR eye = XMLoadFloat3(&settings.eye), target = XMLoadFloat3(&settings.target);
        auto vp = XMMatrixLookAtLH(eye, target, XMVectorSet(0, 1, 0, 0)) * XMMatrixPerspectiveFovLH(XMConvertToRadians(settings.fov), 1, .05f, 20);
        XMMATRIX crop = XMMatrixIdentity(); crop.r[0] *= settings.zoom * settings.height / settings.width; crop.r[1] *= settings.zoom;
        crop.r[3] = XMVectorSet(settings.zoom * settings.height / settings.width * (1 - 2 * settings.center.x), settings.zoom * (2 * settings.center.y - 1), 0, 1);
        XMStoreFloat4x4(&camera.vp, vp * crop); XMStoreFloat4(&camera.eye, eye);
        camera.mipBias = settings.mipBias;
        camera.normalMode = settings.normalMode; camera.pointCount = count;
        void* mapped; D3D12_RANGE noRead{0,0}; check(constants->Map(0, &noRead, &mapped), "Map camera");
        memcpy(mapped, &camera, sizeof(camera)); constants->Unmap(0, nullptr);
        if (textures) for (UINT slot = 0; slot < 6; ++slot) prepareMaterial(slot, textures[slot]);
        begin();
        if (textures) for (UINT slot = 0; slot < 6; ++slot) uploadMaterial(slot);
        if (updateShadows) {
            transition(commands.Get(), shadows.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE);
            commands->SetPipelineState(shadowPipeline.Get()); viewport(shadowSize, shadowSize);
            for (UINT index = 0; index < shadowCount; ++index) {
                auto handle = dsv(index);
                commands->OMSetRenderTargets(0, nullptr, FALSE, &handle);
                commands->ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH, 1, 0, 0, nullptr);
                commands->SetGraphicsRoot32BitConstant(1, index, 0);
                if (index || settings.sun.power > 0) commands->DrawInstanced(vertexCount, 1, 0, 0);
            }
            transition(commands.Get(), shadows.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }
        transition(commands.Get(), viewportTarget.output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        commands->SetPipelineState(shadingPipeline.Get()); viewport(settings.width, settings.height);
        auto rtv = targets->GetCPUDescriptorHandleForHeapStart(), mainDepth = dsv(shadowCount);
        commands->OMSetRenderTargets(1, &rtv, FALSE, &mainDepth);
        const float clear[] = {.11f, .11f, .11f, 1};
        commands->ClearRenderTargetView(rtv, clear, 0, nullptr);
        commands->ClearDepthStencilView(mainDepth, D3D12_CLEAR_FLAG_DEPTH, 1, 0, 0, nullptr);
        commands->DrawInstanced(vertexCount, 1, 0, 0);
        transition(commands.Get(), viewportTarget.output.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION source{}; source.pResource = viewportTarget.output.Get(); source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION destination{}; destination.pResource = viewportTarget.readback.Get();
        destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; destination.PlacedFootprint = viewportTarget.footprint;
        commands->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        submit();
        D3D12_RANGE range{0, SIZE_T(viewportTarget.readback->GetDesc().Width)};
        check(viewportTarget.readback->Map(0, &range, &mapped), "Map rendered pixels");
        for (UINT row = 0; row < settings.height; ++row)
            memcpy(result + SIZE_T(row) * settings.width * 4, static_cast<const char*>(mapped) + SIZE_T(row) * viewportTarget.footprint.Footprint.RowPitch, SIZE_T(settings.width) * 16);
        D3D12_RANGE noWrite{0,0}; viewportTarget.readback->Unmap(0, &noWrite);
    }
};
}

extern "C" {
__declspec(dllexport) void* pbr_create(const float* vertices, UINT count, const char* shader, size_t shaderSize, char* error, size_t capacity) {
    try { return new Renderer(vertices, count, shader, shaderSize); }
    catch (const std::exception& e) { report(error, capacity, e); return nullptr; }
}
__declspec(dllexport) void pbr_destroy(void* renderer) { delete static_cast<Renderer*>(renderer); }
__declspec(dllexport) const char* pbr_adapter(void* renderer) { return static_cast<Renderer*>(renderer)->adapterName.c_str(); }
__declspec(dllexport) UINT pbr_shadow_size() { return shadowSize; }
__declspec(dllexport) int pbr_render(void* renderer, const Texture* textures, const View* settings,
                                  const PointLight* points, UINT count, const Geometry* geometry,
                                  float* output, char* error, size_t capacity) {
    // Reject bad input before mapping resources, invalidating caches or opening
    // a command list. A recording/submission failure requires a fresh renderer.
    try {
        validateTextures(textures);
        if (geometry && ((geometry->count && !geometry->data) || geometry->count % 3 || geometry->count > UINT_MAX / 48))
            throw std::runtime_error("Invalid displaced geometry");
    }
    catch (const std::exception& e) { report(error, capacity, e); return 0; }
    try { static_cast<Renderer*>(renderer)->render(textures, *settings, points, count, geometry, output); return 1; }
    catch (const std::exception& e) { report(error, capacity, e); return -1; }
}
}
