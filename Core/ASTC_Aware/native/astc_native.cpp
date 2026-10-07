#include <torch/extension.h>
#include "astcenc_internal.h"
#include <memory>
#include <cmath>

namespace {
const block_size_descriptor& descriptor() {
    static std::unique_ptr<block_size_descriptor> bsd;
    if (!bsd) {
        bsd = std::make_unique<block_size_descriptor>();
        init_block_size_descriptor(6, 6, 1, false, 4, 1.f, *bsd);
    }
    return *bsd;
}
int color_quant(int q, int value) {
    const int code=color_uquant_to_scrambled_pquant_tables[q-QUANT_6][astc::clamp(value,0,255)];
    return color_scrambled_pquant_to_uquant_tables[q-QUANT_6][code];
}
int weight_quant(int q, int value) {
    const int levels=get_quant_level(static_cast<quant_method>(q));
    const int index=astc::clamp((int)std::floor(value*(levels-1)/64.f+0.5f),0,levels-1);
    return quant_and_xfer_tables[q].quant_to_unquant[index];
}
void cpu_tensor(torch::Tensor t, at::ScalarType dtype) {
    TORCH_CHECK(!t.is_cuda() && t.scalar_type()==dtype && t.is_contiguous(),"Expected contiguous CPU tensor");
}
symbolic_compressed_block from_parameters(const int* meta,const float* ep,const float* weight) {
    symbolic_compressed_block scb{};
    scb.block_type=meta[0];
    if(scb.block_type==SYM_BTYPE_CONST_U16) {
        for(int c=0;c<4;++c) scb.constant_color[c]=astc::clamp((int)std::nearbyint(ep[c]*65535.f),0,65535);
        return scb;
    }
    TORCH_CHECK(scb.block_type==SYM_BTYPE_NONCONST,"Only LDR ASTC blocks are supported");
    scb.partition_count=meta[1]; scb.plane2_component=meta[2];
    scb.quant_mode=static_cast<quant_method>(meta[3]);
    scb.block_mode=meta[10];scb.partition_index=meta[11];scb.color_formats_matched=meta[12];
    for(int p=0;p<scb.partition_count;++p) {
        scb.color_formats[p]=meta[6+p];
        const int count=2*(scb.color_formats[p]>>2)+2;
        for(int j=0;j<count;++j) scb.color_values[p][j]=color_quant(meta[3],(int)std::nearbyint(ep[p*8+j]*255.f));
    }
    for(int j=0;j<meta[5];++j) {
        scb.weights[j]=weight_quant(meta[4],astc::clamp((int)std::nearbyint(weight[j]*64.f),0,64));
        if(meta[2]>=0) scb.weights[32+j]=weight_quant(meta[4],astc::clamp((int)std::nearbyint(weight[32+j]*64.f),0,64));
    }
    return scb;
}
}

std::vector<torch::Tensor> import_astc_blocks(torch::Tensor physical) {
    cpu_tensor(physical,torch::kUInt8);TORCH_CHECK(physical.dim()==2 && physical.size(1)==16,"Expected [N,16] physical blocks");
    const int n=physical.size(0);auto opts=torch::TensorOptions().device(torch::kCPU);
    auto endpoints=torch::zeros({n,32},opts.dtype(torch::kFloat32));
    auto weights=torch::zeros({n,64},opts.dtype(torch::kFloat32));
    auto metadata=torch::zeros({n,16},opts.dtype(torch::kInt32));
    auto partitions=torch::zeros({n,36},opts.dtype(torch::kInt32));
    auto indices=torch::zeros({n,36,4},opts.dtype(torch::kInt32));
    auto coefficients=torch::zeros({n,36,4},opts.dtype(torch::kInt32));
    const auto& bsd=descriptor();
    for(int i=0;i<n;++i) {
        symbolic_compressed_block scb{};physical_to_symbolic(bsd,physical.data_ptr<uint8_t>()+i*16,scb);
        int* m=metadata.data_ptr<int>()+i*16;m[0]=scb.block_type;
        float* e=endpoints.data_ptr<float>()+i*32;float* w=weights.data_ptr<float>()+i*64;
        if(scb.block_type==SYM_BTYPE_CONST_U16) {
            for(int c=0;c<4;++c)e[c]=scb.constant_color[c]/65535.f;
            continue;
        }
        TORCH_CHECK(scb.block_type==SYM_BTYPE_NONCONST,"Error or HDR constant ASTC block");
        const auto& bm=bsd.get_block_mode(scb.block_mode);const auto& di=bsd.get_decimation_info(bm.decimation_mode);
        const auto& pi=bsd.get_partition_info(scb.partition_count,scb.partition_index);
        m[1]=scb.partition_count;m[2]=scb.plane2_component;m[3]=scb.quant_mode;m[4]=bm.quant_mode;
        m[5]=di.weight_count;m[10]=scb.block_mode;m[11]=scb.partition_index;m[12]=scb.color_formats_matched;
        m[13]=di.weight_x;m[14]=di.weight_y;
        for(int p=0;p<scb.partition_count;++p) {
            const int fmt=scb.color_formats[p];
            TORCH_CHECK(fmt==0||fmt==1||fmt==4||fmt==5||fmt==6||fmt==8||fmt==9||fmt==10||fmt==12||fmt==13,"HDR endpoint mode unsupported");
            m[6+p]=fmt;
            for(int j=0;j<8;++j)e[p*8+j]=scb.color_values[p][j]/255.f;
        }
        for(int j=0;j<64;++j)w[j]=scb.weights[j]/64.f;
        for(int p=0;p<36;++p) {
            partitions.data_ptr<int>()[i*36+p]=pi.partition_of_texel[p];
            for(int k=0;k<4;++k) {
                indices.data_ptr<int>()[(i*36+p)*4+k]=di.texel_weights_tr[k][p];
                coefficients.data_ptr<int>()[(i*36+p)*4+k]=di.texel_weight_contribs_int_tr[k][p];
            }
        }
    }
    auto colors=torch::empty({17,256},opts.dtype(torch::kInt32));
    auto weight_lut=torch::empty({12,65},opts.dtype(torch::kInt32));
    for(int q=0;q<17;++q)for(int value=0;value<256;++value)colors.data_ptr<int>()[q*256+value]=color_quant(q+QUANT_6,value);
    for(int q=0;q<12;++q)for(int value=0;value<65;++value)weight_lut.data_ptr<int>()[q*65+value]=weight_quant(q,value);
    return {endpoints,weights,metadata,partitions,indices,coefficients,colors,weight_lut};
}

torch::Tensor export_astc_blocks(torch::Tensor endpoints,torch::Tensor weights,torch::Tensor metadata) {
    cpu_tensor(endpoints,torch::kFloat32);cpu_tensor(weights,torch::kFloat32);cpu_tensor(metadata,torch::kInt32);
    const int n=metadata.size(0);
    TORCH_CHECK(metadata.sizes()==at::IntArrayRef({n,16}) && endpoints.sizes()==at::IntArrayRef({n,32}) && weights.sizes()==at::IntArrayRef({n,64}),"Invalid ASTC parameter shapes");
    auto result=torch::empty({n,16},torch::TensorOptions().dtype(torch::kUInt8));const auto& bsd=descriptor();
    for(int i=0;i<n;++i) {
        auto scb=from_parameters(metadata.data_ptr<int>()+i*16,endpoints.data_ptr<float>()+i*32,weights.data_ptr<float>()+i*64);
        symbolic_to_physical(bsd,scb,result.data_ptr<uint8_t>()+i*16);
    }
    return result;
}

torch::Tensor decode_astc_blocks_cpu(torch::Tensor physical) {
    cpu_tensor(physical,torch::kUInt8);TORCH_CHECK(physical.dim()==2 && physical.size(1)==16,"Expected [N,16]");
    const int n=physical.size(0);auto result=torch::empty({n,6,6,4},torch::TensorOptions().dtype(torch::kFloat32));
    const auto& bsd=descriptor();
    for(int i=0;i<n;++i) {
        symbolic_compressed_block scb{};physical_to_symbolic(bsd,physical.data_ptr<uint8_t>()+i*16,scb);
        TORCH_CHECK(scb.block_type!=SYM_BTYPE_ERROR,"Invalid ASTC block");
        image_block blk{};blk.decode_unorm8=true;
        decompress_symbolic_block(ASTCENC_PRF_LDR,bsd,0,0,0,scb,blk);
        for(int p=0;p<36;++p) {
            const float values[4]={blk.data_r[p],blk.data_g[p],blk.data_b[p],blk.data_a[p]};
            for(int c=0;c<4;++c)result.data_ptr<float>()[(i*36+p)*4+c]=std::nearbyint(values[c]*255.f)/255.f;
        }
    }
    return result;
}

torch::Tensor astc_proxy_forward(torch::Tensor endpoints,torch::Tensor weights,torch::Tensor metadata,
    torch::Tensor partitions,torch::Tensor indices,torch::Tensor coefficients,torch::Tensor colors,torch::Tensor weight_lut);
std::vector<torch::Tensor> astc_proxy_backward(torch::Tensor gradient,torch::Tensor endpoints,torch::Tensor weights,
    torch::Tensor metadata,torch::Tensor partitions,torch::Tensor indices,torch::Tensor coefficients,
    torch::Tensor colors,torch::Tensor weight_lut);
void astc_adam_full_grid(torch::Tensor endpoints,torch::Tensor weights,torch::Tensor endpoint_gradient,
    torch::Tensor weight_gradient,torch::Tensor endpoint_moment,torch::Tensor endpoint_variance,
    torch::Tensor endpoint_count,torch::Tensor weight_moment,torch::Tensor weight_variance,
    torch::Tensor weight_count,torch::Tensor metadata,torch::Tensor colors,torch::Tensor branch_bounds,
    double endpoint_lr,double weight_lr);
torch::Tensor astc_packed_pair_proposals(torch::Tensor endpoints,torch::Tensor gradient,torch::Tensor metadata,
    torch::Tensor colors,torch::Tensor branch_bounds,torch::Tensor ids);

PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
    module.def("import_blocks",&import_astc_blocks);
    module.def("export_blocks",&export_astc_blocks);
    module.def("decode_cpu",&decode_astc_blocks_cpu);
    module.def("decode_cuda",&astc_proxy_forward);
    module.def("backward_cuda",&astc_proxy_backward);
    module.def("adam_full_grid_cuda",&astc_adam_full_grid);
    module.def("packed_pair_proposals_cuda",&astc_packed_pair_proposals);
}
