// SPDX-License-Identifier: Apache-2.0
// LDR endpoint transforms derived from Arm astcenc 5.3.0.
// Copyright 2011-2023 Arm Limited (upstream endpoint transforms).
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath>

namespace {
__device__ int bounded(int v,int hi=255) {return min(hi,max(0,v));}
__device__ int raw(float v,int hi) {return bounded(__float2int_rn(v*hi),hi);}
__device__ void uncontract(int* v) {v[0]=(v[0]+v[2])>>1;v[1]=(v[1]+v[2])>>1;}
__device__ void unpack(int fmt,const int* v,int* a,int* b) {
    for(int c=0;c<4;++c)a[c]=b[c]=255;
    if(fmt==0||fmt==1||fmt==4||fmt==5) {
        int l0=v[0],l1=v[1],a0=255,a1=255;
        if(fmt==1) {l0=(v[0]>>2)|(v[1]&0xC0);l1=min(255,l0+(v[1]&63));}
        if(fmt==4) {a0=v[2];a1=v[3];}
        if(fmt==5) {
            l0=(v[0]|((v[1]&128)<<1))>>1;
            int d=v[1]&127;if(d&64)d-=128;l1=bounded(l0+(d>>1));
            a0=(v[2]|((v[3]&128)<<1))>>1;
            d=v[3]&127;if(d&64)d-=128;a1=bounded(a0+(d>>1));
        }
        for(int c=0;c<3;++c){a[c]=l0;b[c]=l1;}a[3]=a0;b[3]=a1;return;
    }
    if(fmt==6||fmt==10) {
        for(int c=0;c<3;++c){b[c]=v[c];a[c]=(v[c]*v[3])>>8;}
        if(fmt==10){a[3]=v[4];b[3]=v[5];}return;
    }
    const bool rgba=fmt==12||fmt==13;
    for(int c=0;c<3;++c){a[c]=v[2*c];b[c]=v[2*c+1];}
    if(rgba){a[3]=v[6];b[3]=v[7];}
    bool swap=false;
    if(fmt==9||fmt==13) {
        int sum=0;
        for(int c=0;c<(rgba?4:3);++c) {
            const int base=(a[c]>>1)|(b[c]&128);
            int delta=(b[c]>>1)&63;if(delta&32)delta-=64;
            a[c]=base;b[c]=base+delta;if(c<3)sum+=delta;
        }
        swap=sum<0;
    } else swap=(a[0]+a[1]+a[2])>(b[0]+b[1]+b[2]);
    if(swap){uncontract(a);uncontract(b);for(int c=0;c<4;++c){int tmp=a[c];a[c]=b[c];b[c]=tmp;}}
    for(int c=0;c<4;++c){a[c]=bounded(a[c]);b[c]=bounded(b[c]);}
}
__device__ void values(const float* ep,const int* m,const int* colors,int part,int* v) {
    const int q=m[3]-4;
    for(int k=0;k<8;++k)v[k]=colors[q*256+raw(ep[part*8+k],255)];
}
__device__ void contract_adjoint(const float* gradient,float* result) {
    result[0]=gradient[0]*0.5f;result[1]=gradient[1]*0.5f;
    result[2]=gradient[2]+(gradient[0]+gradient[1])*0.5f;result[3]=gradient[3];
}
__device__ void endpoint_adjoint(int fmt,const int* v,const float* ga,const float* gb,float* out) {
    for(int k=0;k<8;++k)out[k]=0;
    if(fmt==0||fmt==4) {
        for(int c=0;c<3;++c){out[0]+=ga[c];out[1]+=gb[c];}
        if(fmt==4){out[2]=ga[3];out[3]=gb[3];}return;
    }
    if(fmt==1) {
        const int l0=(v[0]>>2)|(v[1]&192);
        const bool clipped=l0+(v[1]&63)>255;
        for(int c=0;c<3;++c){out[0]+=(ga[c]+(clipped?0:gb[c]))*0.25f;out[1]+=clipped?0:gb[c];}return;
    }
    if(fmt==5) {
        const int l0=(v[0]|((v[1]&128)<<1))>>1;
        int ld=v[1]&127;if(ld&64)ld-=128;const int l1=l0+(ld>>1);
        const int a0=(v[2]|((v[3]&128)<<1))>>1;
        int ad=v[3]&127;if(ad&64)ad-=128;const int a1=a0+(ad>>1);
        for(int c=0;c<3;++c){const float g=(l1>=0&&l1<=255)?gb[c]:0;out[0]+=(ga[c]+g)*0.5f;out[1]+=g*0.5f;}
        const float g=(a1>=0&&a1<=255)?gb[3]:0;out[2]=(ga[3]+g)*0.5f;out[3]=g*0.5f;return;
    }
    if(fmt==6||fmt==10) {
        for(int c=0;c<3;++c){out[c]=gb[c]+ga[c]*v[3]/256.f;out[3]+=ga[c]*v[c]/256.f;}
        if(fmt==10){out[4]=ga[3];out[5]=gb[3];}return;
    }
    const bool delta=fmt==9||fmt==13,rgba=fmt==12||fmt==13;
    const int channels=rgba?4:3;int base[4]={},difference[4]={};bool swap;
    if(delta) {
        int sum=0;
        for(int c=0;c<channels;++c){base[c]=(v[2*c]>>1)|(v[2*c+1]&128);difference[c]=(v[2*c+1]>>1)&63;if(difference[c]&32)difference[c]-=64;if(c<3)sum+=difference[c];}
        swap=sum<0;
    } else swap=(v[0]+v[2]+v[4])>(v[1]+v[3]+v[5]);
    float g0[4],g1[4];
    float clipped_ga[4],clipped_gb[4];
    for(int c=0;c<4;++c){clipped_ga[c]=ga[c];clipped_gb[c]=gb[c];}
    if(delta) {
        int pre0[4]={0,0,0,255},pre1[4]={0,0,0,255};
        for(int c=0;c<channels;++c){pre0[c]=base[c];pre1[c]=base[c]+difference[c];}
        if(swap){uncontract(pre0);uncontract(pre1);for(int c=0;c<4;++c){int temp=pre0[c];pre0[c]=pre1[c];pre1[c]=temp;}}
        for(int c=0;c<channels;++c){if(pre0[c]<0||pre0[c]>255)clipped_ga[c]=0;if(pre1[c]<0||pre1[c]>255)clipped_gb[c]=0;}
    }
    if(swap){contract_adjoint(clipped_gb,g0);contract_adjoint(clipped_ga,g1);}
    else for(int c=0;c<4;++c){g0[c]=clipped_ga[c];g1[c]=clipped_gb[c];}
    for(int c=0;c<channels;++c) {
        if(delta) {
            // High-bit transfers and sign selection are held fixed in the surrogate.
            out[2*c]=(g0[c]+g1[c])*0.5f;out[2*c+1]=g1[c]*0.5f;
        } else {out[2*c]=g0[c];out[2*c+1]=g1[c];}
    }
}
__device__ int interpolated_weight(const float* weights,const int* m,const int* ix,const int* cf,
                                   const int* lut,int plane) {
    int sum=8;
    for(int k=0;k<4;++k)sum+=cf[k]*lut[m[4]*65+raw(weights[plane*32+ix[k]],64)];
    return sum>>4;
}
__global__ void decode_kernel(const float* endpoints,const float* weights,const int* metadata,
    const int* parts,const int* indices,const int* coefficients,const int* colors,const int* lut,float* output,int n) {
    const int pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel>=n*36)return;
    const int block=pixel/36;const int* m=metadata+block*16;const float* ep=endpoints+block*32;
    if(m[0]==2) {
        for(int c=0;c<4;++c)output[pixel*4+c]=(raw(ep[c],65535)>>8)/255.f;
        return;
    }
    int v[8],a[4],b[4];const int part=parts[pixel];values(ep,m,colors,part,v);unpack(m[6+part],v,a,b);
    const int* ix=indices+pixel*4;const int* cf=coefficients+pixel*4;
    const int w0=interpolated_weight(weights+block*64,m,ix,cf,lut,0);
    const int w1=m[2]>=0?interpolated_weight(weights+block*64,m,ix,cf,lut,1):w0;
    for(int c=0;c<4;++c) {
        const int w=c==m[2]?w1:w0;
        const int color=(a[c]*257*(64-w)+b[c]*257*w+32)>>6;
        output[pixel*4+c]=(color>>8)/255.f;
    }
}
__global__ void backward_kernel(const float* gradient,const float* endpoints,const float* weights,const int* metadata,
    const int* parts,const int* indices,const int* coefficients,const int* colors,const int* lut,
    float* endpoint_gradient,float* weight_gradient,int n) {
    const int block=blockIdx.x*blockDim.x+threadIdx.x;if(block>=n)return;
    const int* m=metadata+block*16;const float* ep=endpoints+block*32;const float* wg=weights+block*64;
    float* ge=endpoint_gradient+block*32;float* gw=weight_gradient+block*64;
    for(int j=0;j<32;++j)ge[j]=0;for(int j=0;j<64;++j)gw[j]=0;
    if(m[0]==2) {
        for(int p=0;p<36;++p)for(int c=0;c<4;++c)ge[c]+=gradient[(block*36+p)*4+c];
        for(int c=0;c<4;++c)if(ep[c]<0||ep[c]>1)ge[c]=0;
        return;
    }
    int e0[16],e1[16],encoded[32];float ga[16]={},gb[16]={};
    for(int part=0;part<m[1];++part){values(ep,m,colors,part,encoded+part*8);unpack(m[6+part],encoded+part*8,e0+part*4,e1+part*4);}
    for(int p=0;p<36;++p) {
        const int pixel=block*36+p,part=parts[pixel];
        const float* gp=gradient+pixel*4;
        // Bilinear training touches only a few texels in each block.
        if(gp[0]==0&&gp[1]==0&&gp[2]==0&&gp[3]==0)continue;
        const int* ix=indices+pixel*4;const int* cf=coefficients+pixel*4;
        const int w0=interpolated_weight(wg,m,ix,cf,lut,0);
        const int w1=m[2]>=0?interpolated_weight(wg,m,ix,cf,lut,1):w0;
        for(int c=0;c<4;++c) {
            const float g=gradient[pixel*4+c];const int plane=c==m[2]?1:0;
            const float t=(plane?w1:w0)/64.f;
            ga[part*4+c]+=(1-t)*g;gb[part*4+c]+=t*g;
            for(int k=0;k<4;++k)gw[plane*32+ix[k]]+=g*(e1[part*4+c]-e0[part*4+c])/255.f*cf[k]/16.f;
        }
    }
    // Piecewise endpoint-transform Jacobian; bit transfers/signs are fixed, quantization uses STE.
    for(int part=0;part<m[1];++part) {
        const int count=2*(m[6+part]>>2)+2;
        float result[8];endpoint_adjoint(m[6+part],encoded+part*8,ga+part*4,gb+part*4,result);
        for(int k=0;k<count;++k) {
            ge[part*8+k]=(ep[part*8+k]>=0&&ep[part*8+k]<=1)?result[k]:0;
        }
    }
    for(int j=0;j<64;++j)if(wg[j]<0||wg[j]>1)gw[j]=0;
}

__device__ int blue_branches(const float* ep,const int* m,const int* colors) {
    if(m[0]!=3)return 0;
    int result=0;
    for(int part=0;part<m[1];++part) {
        const int fmt=m[6+part];if(fmt!=8&&fmt!=9&&fmt!=12&&fmt!=13)continue;
        int v[8];values(ep,m,colors,part,v);
        bool blue=false;
        if(fmt==9||fmt==13) {
            int sum=0;
            for(int c=0;c<3;++c){int delta=(v[c*2+1]>>1)&63;if(delta&32)delta-=64;sum+=delta;}
            blue=sum<0;
        } else blue=(v[0]+v[2]+v[4])>(v[1]+v[3]+v[5]);
        if(blue)result|=1<<part;
    }
    return result;
}

__global__ void adam_full_kernel(float* e,float* w,const float* ge,const float* gw,
    float* em,float* ev,int64_t* ec,float* wm,float* wv,int64_t* wc,const int* metadata,
    const int* colors,const int* bounds,float elr,float wlr) {
    const int row=blockIdx.x,col=threadIdx.x;
    __shared__ int active,invalid;
    __shared__ float eb1,eb2,wb1,wb2;
    __shared__ int blue_before,blue_after;
    __shared__ float endpoint_steps[32];
    if(col==0){active=0;invalid=0;blue_before=blue_branches(e+row*32,metadata+row*16,colors);}
    __syncthreads();
    float g=col<32?ge[row*32+col]:(col<96?gw[row*64+col-32]:0.f);
    if(g!=0.f)atomicOr(&active,1);
    if(!isfinite(g))atomicOr(&invalid,1);
    __syncthreads();
    if(!active||invalid)return;
    if(col==0) {
        const float es=static_cast<float>(++ec[row]),ws=static_cast<float>(++wc[row]);
        eb1=1.f-powf(0.9f,es);eb2=1.f-powf(0.999f,es);
        wb1=1.f-powf(0.9f,ws);wb2=1.f-powf(0.999f,ws);
    }
    __syncthreads();
    const float original=col<32?e[row*32+col]:0;
    const int* meta=metadata+row*16;
    const int fmt=col<32?meta[6+col/8]:0;
    if(col<96) {
    const bool endpoint=col<32;
    const int i=endpoint?row*32+col:row*64+col-32;
    float* params=endpoint?e:w;float* moment=endpoint?em:wm;float* variance=endpoint?ev:wv;
    // Separate multiply/add matches the existing sequence of ATen operations.
    const float mv=__fadd_rn(__fmul_rn(moment[i],0.9f),__fmul_rn(g,0.1f));
    const float vv=__fadd_rn(__fmul_rn(variance[i],0.999f),__fmul_rn(__fmul_rn(g,g),0.001f));
    moment[i]=mv;variance[i]=vv;
    const float mh=mv/(endpoint?eb1:wb1),vh=vv/(endpoint?eb2:wb2);
    const float delta=__fmul_rn(endpoint?elr:wlr,mh)/__fadd_rn(sqrtf(vh),1e-8f);
    if(endpoint)endpoint_steps[col]=-delta;
    else params[i]=fminf(1.f,fmaxf(0.f,__fsub_rn(params[i],delta)));
    }
    __syncthreads();
    if(col<32) {
        float change=endpoint_steps[col];
        float lower=0.f,upper=1.f;
        const int position=col%8;
        const bool packed=meta[0]==3&&((fmt==1&&position==1)||
                          ((fmt==5||fmt==9||fmt==13)&&(position%2==1)));
        if(packed) {
            const int q=bounded(meta[3]-4,16),flag=colors[q*256+raw(original,255)]>>6;
            lower=fmaxf(0.f,(bounds[(q*4+flag)*2]-0.49f)/255.f);
            upper=fminf(1.f,(bounds[(q*4+flag)*2+1]+0.49f)/255.f);
        }
        e[row*32+col]=fminf(upper,fmaxf(lower,__fadd_rn(original,change)));
    }
    __syncthreads();
    if(col==0)blue_after=blue_branches(e+row*32,metadata+row*16,colors);
    __syncthreads();
    if(col<32&&((blue_before^blue_after)&(1<<(col/8))))e[row*32+col]=original;
}

__global__ void packed_pair_kernel(const float* endpoints,const float* gradient,const int* metadata,
    const int* colors,const int* bounds,const int64_t* ids,float* proposals,int n) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
    const int64_t id=ids[row];const int* m=metadata+id*16;
    const float* ep=endpoints+id*32;const float* ge=gradient+id*32;
    float* out=proposals+row*4;out[0]=-1;out[1]=out[2]=out[3]=0;
    if(m[0]!=3)return;
    const int q=bounded(m[3]-4,16);const int* table=colors+q*256;
    for(int part=0;part<m[1];++part) {
        const int fmt=m[6+part];if(fmt!=1&&fmt!=5&&fmt!=9&&fmt!=13)continue;
        const int pairs=fmt==1?1:(fmt==5?2:(fmt==9?3:4));
        const int base_scale=fmt==1?4:2,high_mask=fmt==1?192:128;
        int v[8];for(int k=0;k<2*pairs;++k)v[k]=table[raw(ep[part*8+k],255)];
        int rgb_sum=0;
        if(fmt==9||fmt==13)for(int c=0;c<3;++c){int d=(v[2*c+1]>>1)&63;if(d&32)d-=64;rgb_sum+=d;}
        for(int c=0;c<pairs;++c) {
            const int slot=part*8+2*c,a=v[2*c],b=v[2*c+1];
            const int base=(a/base_scale)|(b&high_mask);
            int delta=fmt==1?(b&63):((b>>1)&63);if(fmt!=1&&(delta&32))delta-=64;
            const float g0=ge[slot],g1=ge[slot+1];
            const int flag=b>>6;
            const float lo=fmaxf(0.f,(bounds[(q*4+flag)*2]-0.49f)/255.f);
            const float hi=fminf(1.f,(bounds[(q*4+flag)*2+1]+0.49f)/255.f);
            const bool blocked=(a==255&&g0<0)||(a==0&&g0>0)||
                               (ep[slot+1]<=lo+1e-5f&&g1>0)||(ep[slot+1]>=hi-1e-5f&&g1<0);
            if(!blocked)continue;
            int previous=-1;
            for(int code=0;code<256;++code) {
                const int nb=table[code];if(nb==previous)continue;previous=nb;
                if((nb&192)==(b&192))continue;
                // Carry the base's high bit together with its lower payload.
                const int na=(nb&high_mask)==(b&high_mask)?a:table[bounded(base_scale*(base-(nb&high_mask)))];
                const int new_base=(na/base_scale)|(nb&high_mask);
                int new_delta=fmt==1?(nb&63):((nb>>1)&63);if(fmt!=1&&(new_delta&32))new_delta-=64;
                const int db=new_base-base,dd=new_delta-delta;
                if(abs(db)>8||abs(dd)>8||abs(db+dd)>8)continue;
                if((fmt==9||fmt==13)&&c<3&&((rgb_sum+dd<0)!=(rgb_sum<0)))continue;
                // The surrogate is wrt encoded symbols: g_base=2*g_even,
                // g_delta=2*g_odd. Raw-byte changes would rank a carry backwards.
                const float gain=-(base_scale*g0*db+(fmt==1?1:2)*g1*dd)/255.f;
                if(isfinite(gain)&&gain>out[3]) {
                    out[0]=slot;out[1]=na/255.f;out[2]=nb/255.f;out[3]=gain;
                }
            }
        }
    }
}

void validate(at::Tensor e,at::Tensor w,at::Tensor m,at::Tensor p,at::Tensor ix,at::Tensor cf,at::Tensor colors,at::Tensor lut) {
    TORCH_CHECK(e.is_cuda()&&e.scalar_type()==at::kFloat&&e.dim()==2&&e.size(1)==32,"Expected float32 CUDA endpoints [N,32]");
    const auto n=e.size(0);TORCH_CHECK(w.sizes()==at::IntArrayRef({n,64})&&w.scalar_type()==at::kFloat,"Invalid weights");
    TORCH_CHECK(m.sizes()==at::IntArrayRef({n,16})&&p.sizes()==at::IntArrayRef({n,36})&&ix.sizes()==at::IntArrayRef({n,36,4})&&cf.sizes()==ix.sizes(),"Invalid metadata shapes");
    TORCH_CHECK(colors.sizes()==at::IntArrayRef({17,256})&&lut.sizes()==at::IntArrayRef({12,65}),"Invalid quantization tables");
    for(auto t:{w,m,p,ix,cf,colors,lut})TORCH_CHECK(t.device()==e.device()&&t.is_contiguous(),"Device/layout mismatch");
    for(auto t:{m,p,ix,cf,colors,lut})TORCH_CHECK(t.scalar_type()==at::kInt,"Expected int32 metadata");
    TORCH_CHECK(e.is_contiguous(),"Endpoints must be contiguous");
}
}

at::Tensor astc_proxy_forward(at::Tensor e,at::Tensor w,at::Tensor m,at::Tensor p,at::Tensor ix,at::Tensor cf,at::Tensor colors,at::Tensor lut) {
    validate(e,w,m,p,ix,cf,colors,lut);const c10::cuda::CUDAGuard guard(e.device());
    auto result=at::empty({e.size(0),6,6,4},e.options());const int n=e.size(0);
    if(n){decode_kernel<<<(n*36+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(e.data_ptr<float>(),w.data_ptr<float>(),m.data_ptr<int>(),p.data_ptr<int>(),ix.data_ptr<int>(),cf.data_ptr<int>(),colors.data_ptr<int>(),lut.data_ptr<int>(),result.data_ptr<float>(),n);C10_CUDA_KERNEL_LAUNCH_CHECK();}
    return result;
}
std::vector<at::Tensor> astc_proxy_backward(at::Tensor g,at::Tensor e,at::Tensor w,at::Tensor m,at::Tensor p,at::Tensor ix,at::Tensor cf,at::Tensor colors,at::Tensor lut) {
    validate(e,w,m,p,ix,cf,colors,lut);TORCH_CHECK(g.sizes()==at::IntArrayRef({e.size(0),6,6,4})&&g.device()==e.device()&&g.scalar_type()==at::kFloat&&g.is_contiguous(),"Invalid gradient");
    const c10::cuda::CUDAGuard guard(e.device());auto ge=at::empty_like(e),gw=at::empty_like(w);const int n=e.size(0);
    if(n){backward_kernel<<<(n+63)/64,64,0,at::cuda::getCurrentCUDAStream()>>>(g.data_ptr<float>(),e.data_ptr<float>(),w.data_ptr<float>(),m.data_ptr<int>(),p.data_ptr<int>(),ix.data_ptr<int>(),cf.data_ptr<int>(),colors.data_ptr<int>(),lut.data_ptr<int>(),ge.data_ptr<float>(),gw.data_ptr<float>(),n);C10_CUDA_KERNEL_LAUNCH_CHECK();}
    return {ge,gw};
}

void astc_adam_full_grid(at::Tensor e,at::Tensor w,at::Tensor ge,at::Tensor gw,at::Tensor em,at::Tensor ev,
    at::Tensor ec,at::Tensor wm,at::Tensor wv,at::Tensor wc,at::Tensor m,at::Tensor colors,at::Tensor bounds,
    double elr,double wlr) {
    TORCH_CHECK(e.is_cuda()&&e.dim()==2&&e.size(1)==32&&e.scalar_type()==at::kFloat,"Expected CUDA endpoints [N,32]");
    const auto n=e.size(0);
    for(auto t:{e,ge,em,ev})TORCH_CHECK(t.sizes()==at::IntArrayRef({n,32})&&t.scalar_type()==at::kFloat&&t.device()==e.device()&&t.is_contiguous(),"Invalid endpoint Adam tensor");
    for(auto t:{w,gw,wm,wv})TORCH_CHECK(t.sizes()==at::IntArrayRef({n,64})&&t.scalar_type()==at::kFloat&&t.device()==e.device()&&t.is_contiguous(),"Invalid weight Adam tensor");
    for(auto t:{ec,wc})TORCH_CHECK(t.sizes()==at::IntArrayRef({n})&&t.scalar_type()==at::kLong&&t.device()==e.device()&&t.is_contiguous(),"Invalid Adam counters");
    TORCH_CHECK(m.sizes()==at::IntArrayRef({n,16})&&colors.sizes()==at::IntArrayRef({17,256})&&bounds.sizes()==at::IntArrayRef({17,4,2}),"Invalid ASTC tables");
    for(auto t:{m,colors,bounds})TORCH_CHECK(t.scalar_type()==at::kInt&&t.device()==e.device()&&t.is_contiguous(),"Invalid ASTC table device/layout");
    TORCH_CHECK(std::isfinite(elr)&&std::isfinite(wlr)&&elr>0&&wlr>0,"Invalid Adam learning rates");
    const c10::cuda::CUDAGuard guard(e.device());
    if(n){adam_full_kernel<<<n,128,0,at::cuda::getCurrentCUDAStream()>>>(e.data_ptr<float>(),w.data_ptr<float>(),ge.data_ptr<float>(),gw.data_ptr<float>(),em.data_ptr<float>(),ev.data_ptr<float>(),ec.data_ptr<int64_t>(),wm.data_ptr<float>(),wv.data_ptr<float>(),wc.data_ptr<int64_t>(),m.data_ptr<int>(),colors.data_ptr<int>(),bounds.data_ptr<int>(),static_cast<float>(elr),static_cast<float>(wlr));C10_CUDA_KERNEL_LAUNCH_CHECK();}
}

at::Tensor astc_packed_pair_proposals(at::Tensor e,at::Tensor g,at::Tensor m,at::Tensor colors,at::Tensor bounds,at::Tensor ids) {
    TORCH_CHECK(e.is_cuda()&&e.dim()==2&&e.size(1)==32&&e.scalar_type()==at::kFloat,"Expected CUDA endpoints [N,32]");
    const auto n=e.size(0);
    TORCH_CHECK(g.sizes()==e.sizes()&&g.scalar_type()==at::kFloat&&g.device()==e.device()&&g.is_contiguous(),"Invalid endpoint gradient");
    TORCH_CHECK(m.sizes()==at::IntArrayRef({n,16})&&colors.sizes()==at::IntArrayRef({17,256})&&bounds.sizes()==at::IntArrayRef({17,4,2}),"Invalid ASTC tables");
    for(auto t:{m,colors,bounds})TORCH_CHECK(t.scalar_type()==at::kInt&&t.device()==e.device()&&t.is_contiguous(),"Invalid ASTC table device/layout");
    TORCH_CHECK(e.is_contiguous()&&ids.dim()==1&&ids.scalar_type()==at::kLong&&ids.device()==e.device()&&ids.is_contiguous(),"Invalid selected rows");
    const c10::cuda::CUDAGuard guard(e.device());auto out=at::empty({ids.size(0),4},e.options());const int count=ids.size(0);
    if(count){packed_pair_kernel<<<(count+63)/64,64,0,at::cuda::getCurrentCUDAStream()>>>(e.data_ptr<float>(),g.data_ptr<float>(),m.data_ptr<int>(),colors.data_ptr<int>(),bounds.data_ptr<int>(),ids.data_ptr<int64_t>(),out.data_ptr<float>(),count);C10_CUDA_KERNEL_LAUNCH_CHECK();}
    return out;
}
