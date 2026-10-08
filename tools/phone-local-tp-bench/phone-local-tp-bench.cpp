#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms(clk::time_point t) { return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }
static void check(bool x,const std::string & msg) { if(!x) throw std::runtime_error(msg); }
static double med(std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];}
struct Opt {std::string model;int layer=0,layers=2,tokens=400,topk=8,threads=4,runs=5,warmup=2;double cpu_ratio=0.25;bool probe_opencl=false;bool full_layer=false;};
static Opt options(int argc,char **argv) {
 Opt o;for(int i=1;i<argc;i++){std::string a=argv[i];if(a=="--probe-opencl"){o.probe_opencl=true;continue;}if(a=="--full-layer"){o.full_layer=true;continue;}check(i+1<argc,"missing argument for "+a);std::string v=argv[++i];
 if(a=="-m")o.model=v;else if(a=="--layer")o.layer=std::stoi(v);else if(a=="--layers")o.layers=std::stoi(v);
 else if(a=="--tokens")o.tokens=std::stoi(v);else if(a=="--topk")o.topk=std::stoi(v);
 else if(a=="--threads")o.threads=std::stoi(v);else if(a=="--runs")o.runs=std::stoi(v);
 else if(a=="--warmup")o.warmup=std::stoi(v);else if(a=="--cpu-ratio")o.cpu_ratio=std::stod(v);
 else throw std::runtime_error("unknown option "+a);}
 check((!o.model.empty()||o.probe_opencl)&&o.layer>=0&&o.layers>0&&o.tokens>0&&o.topk>0&&o.threads>0&&o.runs>0&&o.warmup>=0&&o.cpu_ratio>=0&&o.cpu_ratio<1,"invalid benchmark options");return o;
}
struct File {
 gguf_context *uf=nullptr;ggml_context *ctx=nullptr;std::ifstream f;size_t data=0;
 File(std::string name):f(name,std::ios::binary){check(f.good(),"GGUF open failed");gguf_init_params p={true,&ctx};uf=gguf_init_from_file(name.c_str(),p);check(uf&&ctx,"GGUF metadata failed");data=gguf_get_data_offset(uf);}
 ~File(){if(uf)gguf_free(uf);if(ctx)ggml_free(ctx);}
 struct Weight{int64_t id; ggml_tensor *t; std::string name;};
 Weight get(int layer,std::string key){std::string name="blk."+std::to_string(layer)+"."+key+".weight";auto id=gguf_find_tensor(uf,name.c_str());check(id>=0,"missing tensor "+name);auto t=ggml_get_tensor(ctx,name.c_str());check(t!=nullptr,"missing GGUF tensor metadata");return{id,t,name};}
 void read(size_t off,void *dst,size_t n){f.clear();f.seekg(off);check(f.good(),"seek failed");f.read((char*)dst,n);check((size_t)f.gcount()==n,"short GGUF read");}
 void load(Weight w,ggml_tensor *dst,int64_t start,int64_t width,bool down){
  auto type=w.t->type;int64_t full=down?w.t->ne[0]:w.t->ne[1],rows=down?w.t->ne[1]:w.t->ne[0],experts=w.t->ne[2];
  check(start>=0&&start+width<=full,"bad slice");
  check(!down||(start%ggml_blck_size(type)==0&&width%ggml_blck_size(type)==0),"unaligned down slice");
  size_t rf=ggml_row_size(type,down?full:rows),rp=ggml_row_size(type,down?width:rows);
  size_t ef=rf*(down?rows:full),ep=rp*(down?rows:width),base=data+gguf_get_tensor_offset(uf,w.id);
  check(gguf_get_tensor_size(uf,w.id)==ef*(size_t)experts&&ggml_nbytes(dst)==ep*(size_t)experts,"weight bytes mismatch");
  // OpenCL SOA_Q converts the ENTIRE quantized tensor on the first set_tensor
  // call, regardless of the supplied partial size/offset. Never submit a
  // single-expert buffer to a multi-expert quantized OpenCL tensor: the device
  // implementation reads ggml_nbytes(dst) from the supplied host pointer.
  // Assemble all expert shards in canonical GGUF AoS order and upload once.
  const size_t dst_bytes = ggml_nbytes(dst);
  check(ep != 0 && static_cast<size_t>(experts) <= dst_bytes / ep &&
        static_cast<size_t>(experts) * ep == dst_bytes,
        "destination tensor size mismatch");
  std::vector<uint8_t> packed(dst_bytes);
  std::vector<uint8_t> scratch(down ? ef : 0);
  std::fprintf(stderr,
       "[PHONE_LOCAL_WEIGHT] phase=assemble_begin tensor=%s type=%s src_expert_bytes=%zu dst_expert_bytes=%zu experts=%lld total_bytes=%zu\n",
       w.name.c_str(), ggml_type_name(type), ef, ep, (long long)experts, dst_bytes);
  for(int64_t e=0;e<experts;e++){
    const size_t pos=base+static_cast<size_t>(e)*ef;
    uint8_t *destination=packed.data()+static_cast<size_t>(e)*ep;
    if(!down){
      // Gate/Up shard is a contiguous range of rows within each expert.
      read(pos+static_cast<size_t>(start)*rf,destination,ep);
    }else{
      // Down shard consists of a quant-block-aligned slice of EVERY row.
      read(pos,scratch.data(),ef);
      const size_t inside_row=ggml_row_size(type,start);
      for(int64_t row=0;row<rows;row++){
        std::memcpy(destination+static_cast<size_t>(row)*rp,
                    scratch.data()+static_cast<size_t>(row)*rf+inside_row,rp);
      }
    }
  }
  std::fprintf(stderr,
       "[PHONE_LOCAL_WEIGHT] phase=upload_begin tensor=%s bytes=%zu\n",
       w.name.c_str(), dst_bytes);
  ggml_backend_tensor_set(dst,packed.data(),0,dst_bytes);
  std::fprintf(stderr,
       "[PHONE_LOCAL_WEIGHT] phase=upload_ok tensor=%s\n",w.name.c_str());
 }
};
struct Graph {
 ggml_context *ctx=nullptr;ggml_cgraph *gf=nullptr;ggml_backend_buffer_t buf=nullptr;ggml_backend_t backend=nullptr;
 ggml_tensor *x=nullptr,*ids=nullptr,*mix=nullptr,*out=nullptr,*gate=nullptr,*up=nullptr,*down=nullptr;
 ~Graph(){if(buf)ggml_backend_buffer_free(buf);if(ctx)ggml_free(ctx);}
};
static std::unique_ptr<Graph> build(File&f,int layer,ggml_backend_t backend,int64_t from,int64_t width,int tokens,int topk){
 auto gw=f.get(layer,"ffn_gate_exps"),uw=f.get(layer,"ffn_up_exps"),dw=f.get(layer,"ffn_down_exps");
 int64_t embd=gw.t->ne[0],experts=gw.t->ne[2];
 check(gw.t->ne[1]==uw.t->ne[1]&&gw.t->ne[1]==dw.t->ne[0]&&uw.t->ne[0]==embd&&dw.t->ne[1]==embd&&topk<=experts,"incompatible MoE shapes");
 auto b=std::make_unique<Graph>();b->backend=backend;
 ggml_init_params p={256*ggml_tensor_overhead()+ggml_graph_overhead_custom(128,false),nullptr,true};
 b->ctx=ggml_init(p);check(b->ctx!=nullptr,"GGML context allocation");
 b->x=ggml_new_tensor_2d(b->ctx,GGML_TYPE_F32,embd,tokens);
 b->ids=ggml_new_tensor_2d(b->ctx,GGML_TYPE_I32,topk,tokens);
 b->mix=ggml_new_tensor_3d(b->ctx,GGML_TYPE_F32,1,topk,tokens);
 b->gate=ggml_new_tensor_3d(b->ctx,gw.t->type,embd,width,experts);
 b->up=ggml_new_tensor_3d(b->ctx,uw.t->type,embd,width,experts);
 b->down=ggml_new_tensor_3d(b->ctx,dw.t->type,width,embd,experts);
 ggml_set_name(b->gate,"ffn_gate_exps_probe");ggml_set_name(b->up,"ffn_up_exps_probe");ggml_set_name(b->down,"ffn_down_exps_probe");
 auto x3=ggml_reshape_3d(b->ctx,b->x,embd,1,tokens);
 auto g=ggml_silu(b->ctx,ggml_mul_mat_id(b->ctx,b->gate,x3,b->ids));
 auto u=ggml_mul_mat_id(b->ctx,b->up,x3,b->ids);
 auto h=ggml_mul(b->ctx,g,u);
 auto e=ggml_mul(b->ctx,ggml_mul_mat_id(b->ctx,b->down,h,b->ids),b->mix);
 b->out=ggml_view_2d(b->ctx,e,embd,tokens,e->nb[2],0);
 for(int k=1;k<topk;k++)b->out=ggml_add(b->ctx,b->out,ggml_view_2d(b->ctx,e,embd,tokens,e->nb[2],(size_t)k*e->nb[1]));
 if(topk==1)b->out=ggml_cont(b->ctx,b->out);
 b->gf=ggml_new_graph_custom(b->ctx,128,false);ggml_build_forward_expand(b->gf,b->out);
 for(int i=0;i<ggml_graph_n_nodes(b->gf);i++){
  ggml_tensor *node=ggml_graph_node(b->gf,i);
  check(ggml_backend_supports_op(backend,node),std::string("unsupported op: ")+ggml_op_name(node->op));
 }
 std::fprintf(stderr,"[PHONE_LOCAL_BOOT] phase=buffer_alloc_begin layer=%d backend=%s width=%lld\n",layer,ggml_backend_name(backend),(long long)width);
 b->buf=ggml_backend_alloc_ctx_tensors(b->ctx,backend);check(b->buf!=nullptr,"backend allocation failed");
 std::fprintf(stderr,"[PHONE_LOCAL_BOOT] phase=buffer_alloc_ok layer=%d backend=%s\n",layer,ggml_backend_name(backend));
 std::fprintf(stderr,"[PHONE_LOCAL_BOOT] phase=weights_load_begin layer=%d backend=%s\n",layer,ggml_backend_name(backend));
 f.load(gw,b->gate,from,width,false);f.load(uw,b->up,from,width,false);f.load(dw,b->down,from,width,true);
 std::fprintf(stderr,"[PHONE_LOCAL_BOOT] phase=weights_load_ok layer=%d backend=%s\n",layer,ggml_backend_name(backend));
 std::vector<float> inp((size_t)embd*tokens);
 for(size_t i=0;i<inp.size();i++)inp[i]=0.02f*std::sin(float((i+(size_t)layer*9973)%8191)*0.037f);
 std::vector<int32_t> ids((size_t)topk*tokens);
 for(int t=0;t<tokens;t++)for(int k=0;k<topk;k++)ids[(size_t)t*topk+k]=(t*17+k*31+layer*43)%experts;
 std::vector<float> mix(ids.size(),1.0f/topk);
 ggml_backend_tensor_set(b->x,inp.data(),0,inp.size()*sizeof(float));
 ggml_backend_tensor_set(b->ids,ids.data(),0,ids.size()*sizeof(int32_t));
 ggml_backend_tensor_set(b->mix,mix.data(),0,mix.size()*sizeof(float));ggml_backend_synchronize(backend);
 return b;
}
static double compute(Graph &b){auto t=clk::now();check(ggml_backend_graph_compute(b.backend,b.gf)==GGML_STATUS_SUCCESS,"graph_compute failed");ggml_backend_synchronize(b.backend);return ms(t);}
static std::vector<float> output(Graph &b){std::vector<float> x((size_t)ggml_nelements(b.out));ggml_backend_tensor_get(b.out,x.data(),0,x.size()*sizeof(float));return x;}
static ggml_backend_dev_t device(bool gpu){
 for(size_t i=0;i<ggml_backend_dev_count();i++){auto d=ggml_backend_dev_get(i);std::string name=ggml_backend_dev_name(d);
 if(gpu&&name.find("OpenCL")!=std::string::npos)return d;
 if(!gpu&&ggml_backend_dev_type(d)==GGML_BACKEND_DEVICE_TYPE_CPU)return d;
 }throw std::runtime_error(gpu?"OpenCL GPU device unavailable":"CPU device unavailable");
}
static void threads(ggml_backend_t b,int count){auto d=ggml_backend_get_device(b);auto r=ggml_backend_dev_backend_reg(d);
 if(r){auto fn=(ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(r,"ggml_backend_set_n_threads");if(fn)fn(b,count);}}
static void compare(const std::vector<float>&a,const std::vector<float>&b,int layer){
 check(a.size()==b.size(),"output mismatch");double err=0,norm=0;
 for(size_t i=0;i<a.size();i++){check(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite output");err+=double(a[i]-b[i])*(a[i]-b[i]);norm+=double(a[i])*a[i];}
 double rel=std::sqrt(err/std::max(norm,1e-24));std::cout<<"[PHONE_LOCAL_CHECK] layer="<<layer<<" rel_l2="<<rel<<" status="<<(rel<0.03?"OK":"CHECK")<<"\n";
}
static int run(const Opt&o){
 std::fprintf(stderr,"[PHONE_LOCAL_BOOT] phase=backend_load_begin\n");
 ggml_backend_load_all();
 std::fprintf(stderr,"[PHONE_LOCAL_BOOT] phase=backend_load_ok\n");
 auto gpu_device=device(true);
 std::fprintf(stderr,"[PHONE_LOCAL_BOOT] phase=gpu_init_begin device=%s\n",ggml_backend_dev_name(gpu_device));
 auto gpu=ggml_backend_dev_init(gpu_device,nullptr);
 check(gpu!=nullptr,"GPU backend init failed");
 std::fprintf(stderr,"[PHONE_LOCAL_BOOT] phase=gpu_init_ok\n");
 auto cpu_device=device(false);
 std::fprintf(stderr,"[PHONE_LOCAL_BOOT] phase=cpu_init_begin device=%s\n",ggml_backend_dev_name(cpu_device));
 auto cpu=ggml_backend_dev_init(cpu_device,nullptr);
 check(cpu!=nullptr,"CPU backend init failed");threads(cpu,o.threads);
 std::fprintf(stderr,"[PHONE_LOCAL_BOOT] phase=cpu_init_ok\n");
 if(o.probe_opencl){
  std::fprintf(stderr,"[PHONE_LOCAL_BOOT] phase=opencl_probe_alloc_begin size=4096\n");
  ggml_backend_buffer_t probe=ggml_backend_alloc_buffer(gpu,4096);
  check(probe!=nullptr,"OpenCL probe allocation failed");
  std::fprintf(stderr,"[PHONE_LOCAL_BOOT] phase=opencl_probe_alloc_ok\n");
  ggml_backend_buffer_free(probe);
  ggml_backend_free(cpu);ggml_backend_free(gpu);return 0;
 }
 std::cout<<std::fixed<<std::setprecision(3);
 std::cout<<"[PHONE_LOCAL_CONFIG] tokens="<<o.tokens<<" layers="<<o.layers<<" cpu_ratio="<<o.cpu_ratio<<" gpu="<<ggml_backend_name(gpu)<<" cpu="<<ggml_backend_name(cpu)<<"\n";
 {
 std::fprintf(stderr,"[PHONE_LOCAL_BOOT] phase=gguf_open_begin\n");
 File f(o.model);
 std::fprintf(stderr,"[PHONE_LOCAL_BOOT] phase=gguf_open_ok\n");
 for(int il=o.layer;il<o.layer+o.layers;il++){
 std::fprintf(stderr,"[PHONE_LOCAL_BOOT] phase=layer_begin layer=%d\n",il);
 auto w=f.get(il,"ffn_down_exps");int64_t ff=w.t->ne[0],alignment=std::lcm<int64_t>(128,ggml_blck_size(w.t->type));
 check(ff%alignment==0,"FFN width not aligned");int64_t blocks=ff/alignment;
 int64_t cb=std::llround(blocks*o.cpu_ratio);cb=std::min<int64_t>(blocks-1,std::max<int64_t>(o.cpu_ratio>0?1:0,cb));
 int64_t cw=cb*alignment,gw=ff-cw;std::vector<double> base,times;std::vector<float> reference;
 std::cout<<"[PHONE_LOCAL_LAYER] layer="<<il<<" ffn="<<ff<<" gpu_width="<<gw<<" cpu_width="<<cw<<" actual_cpu_ratio="<<double(cw)/ff<<"\n";
 {
 std::fprintf(stderr,"[PHONE_LOCAL_BOOT] phase=gpu_reference_build_begin layer=%d\n",il);
 auto full=build(f,il,gpu,0,ff,o.tokens,o.topk);
 std::fprintf(stderr,"[PHONE_LOCAL_BOOT] phase=gpu_reference_build_ok layer=%d\n",il);
 for(int rep=-o.warmup;rep<o.runs;rep++){double t=compute(*full);if(rep>=0){base.push_back(t);std::cout<<"[PHONE_LOCAL_TP] layer="<<il<<" mode=GPU_ONLY rep="<<rep<<" total_ms="<<t<<"\n";}}
 reference=output(*full);
 }
 if(cw>0){
 auto gp=build(f,il,gpu,0,gw,o.tokens,o.topk);auto cp=build(f,il,cpu,gw,cw,o.tokens,o.topk);
 for(int rep=-o.warmup;rep<o.runs;rep++){
 auto start=clk::now();
 ggml_backend_tensor_copy(gp->x,cp->x);ggml_backend_tensor_copy(gp->ids,cp->ids);ggml_backend_tensor_copy(gp->mix,cp->mix);ggml_backend_synchronize(cpu);
 double stage=ms(start),gt=0,ct=0;std::exception_ptr cpu_error;
 std::thread worker([&]{try{ct=compute(*cp);}catch(...){cpu_error=std::current_exception();}});
 try{gt=compute(*gp);}catch(...){worker.join();throw;}worker.join();if(cpu_error)std::rethrow_exception(cpu_error);
 auto jt=clk::now();auto a=output(*gp),b=output(*cp);
 for(size_t i=0;i<a.size();i++)a[i]+=b[i];
 ggml_backend_tensor_set(gp->out,a.data(),0,a.size()*sizeof(float));ggml_backend_synchronize(gpu);
 double join=ms(jt),total=ms(start);
 if(rep>=0){times.push_back(total);std::cout<<"[PHONE_LOCAL_TP] layer="<<il<<" mode=CPU_GPU rep="<<rep<<" stage_ms="<<stage<<" gpu_ms="<<gt<<" cpu_ms="<<ct<<" join_ms="<<join<<" total_ms="<<total<<"\n";}
 }
 compare(reference,output(*gp),il);
 }
 std::cout<<"[PHONE_LOCAL_SUM] layer="<<il<<" gpu_median_ms="<<med(base);
 if(!times.empty())std::cout<<" mixed_median_ms="<<med(times)<<" speedup="<<med(base)/med(times);
 std::cout<<"\n";
 }
 }
 ggml_backend_free(cpu);ggml_backend_free(gpu);return 0;
}


// Full Qwen3-MoE layer experiment: GPU Attention + Router; GPU/CPU MoE FFN.
// This mode is independent of the legacy FFN-only microbenchmark above.
// It uses real model weights, deterministic hidden states, causal attention
// over the entire prefix, and propagates the layer output into the next layer.
// No tokenizer, layer pipeline, KV cache reuse, flash attention, or PC RPC.

static float metadata_f32(File & f, const std::string & name, float fallback) {
 const int64_t key=gguf_find_key(f.uf,name.c_str());
 if(key<0) return fallback;
 const gguf_type t=gguf_get_kv_type(f.uf,key);
 if(t==GGUF_TYPE_FLOAT32) return gguf_get_val_f32(f.uf,key);
 if(t==GGUF_TYPE_FLOAT64) return static_cast<float>(gguf_get_val_f64(f.uf,key));
 throw std::runtime_error("unexpected GGUF numeric type: "+name);
}
static int64_t metadata_i64(File & f, const std::string & name, int64_t fallback) {
 const int64_t key=gguf_find_key(f.uf,name.c_str());
 if(key<0) return fallback;
 const gguf_type t=gguf_get_kv_type(f.uf,key);
 if(t==GGUF_TYPE_UINT32) return gguf_get_val_u32(f.uf,key);
 if(t==GGUF_TYPE_UINT64) return static_cast<int64_t>(gguf_get_val_u64(f.uf,key));
 if(t==GGUF_TYPE_INT32) return gguf_get_val_i32(f.uf,key);
 if(t==GGUF_TYPE_INT64) return gguf_get_val_i64(f.uf,key);
 throw std::runtime_error("unexpected GGUF integer type: "+name);
}

struct LayerGeometry {
 int64_t embd=0,head_dim=0,heads=0,kv_heads=0,experts=0,ff=0;
 float rms_eps=1.0e-6f,rope_base=1000000.0f,rope_scale=1.0f;
 int context=40960;
};
static LayerGeometry layer_geometry(File & f,int layer,int topk) {
 const int64_t arch_key=gguf_find_key(f.uf,"general.architecture");
 check(arch_key>=0 && std::string(gguf_get_val_str(f.uf,arch_key))=="qwen3moe",
       "--full-layer currently supports GGUF architecture qwen3moe only");
 LayerGeometry g;
 auto norm=f.get(layer,"attn_norm"),qn=f.get(layer,"attn_q_norm");
 auto q=f.get(layer,"attn_q"),k=f.get(layer,"attn_k"),v=f.get(layer,"attn_v");
 auto wo=f.get(layer,"attn_output"),gate=f.get(layer,"ffn_gate_exps");
 g.embd=norm.t->ne[0];
 g.head_dim=qn.t->ne[0];
 check(g.head_dim>0 && q.t->ne[1]%g.head_dim==0 &&
       k.t->ne[1]%g.head_dim==0 && v.t->ne[1]==k.t->ne[1],
       "Qwen3 Q/K/V attention geometry mismatch");
 g.heads=q.t->ne[1]/g.head_dim;
 g.kv_heads=k.t->ne[1]/g.head_dim;
 g.experts=gate.t->ne[2];
 g.ff=gate.t->ne[1];
 check(g.embd>0 && g.heads>0 && g.kv_heads>0 && g.heads%g.kv_heads==0 &&
       g.ff>0 && g.experts>=topk && g.head_dim%2==0,
       "invalid Qwen3-MoE layer parameters");
 check(q.t->ne[0]==g.embd && k.t->ne[0]==g.embd &&
       v.t->ne[0]==g.embd && wo.t->ne[0]==g.heads*g.head_dim &&
       wo.t->ne[1]==g.embd, "attention weight sizes mismatch");
 g.rms_eps=metadata_f32(f,"qwen3moe.attention.layer_norm_rms_epsilon",1.0e-6f);
 g.rope_base=metadata_f32(f,"qwen3moe.rope.freq_base",1000000.0f);
 float scaling=metadata_f32(f,"qwen3moe.rope.scaling.factor",1.0f);
 check(scaling>0,"invalid rope scaling");
 g.rope_scale=1.0f/scaling;
 g.context=static_cast<int>(metadata_i64(f,"qwen3moe.context_length",40960));
 const int64_t used=metadata_i64(f,"qwen3moe.expert_used_count",topk);
 check(used==topk,"--topk must equal GGUF expert_used_count for full-layer correctness");
 return g;
}

struct Prefix {
 ggml_context * ctx=nullptr;
 ggml_backend_buffer_t buffer=nullptr;
 ggml_cgraph * graph=nullptr;
 ggml_backend_t backend=nullptr;
 ggml_tensor * input=nullptr,*positions=nullptr,*mask=nullptr;
 ggml_tensor * residual=nullptr,*ffn_norm=nullptr,*ids=nullptr,*mix=nullptr;
 std::vector<std::pair<File::Weight,ggml_tensor*>> weights;
 ~Prefix(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);}
 Prefix(const Prefix&)=delete;
 Prefix& operator=(const Prefix&)=delete;
 Prefix()=default;

 ggml_tensor* weight(File& f,int layer,const char *suffix) {
  auto weight=f.get(layer,suffix);
  const int dims=weight.t->ne[3]>1?4:(weight.t->ne[2]>1?3:(weight.t->ne[1]>1?2:1));
  ggml_tensor *t=ggml_new_tensor(ctx,weight.t->type,dims,weight.t->ne);
  ggml_set_name(t,weight.name.c_str());
  weights.emplace_back(weight,t);
  return t;
 }
};
static void prefix_upload_weight(File & f,const File::Weight &w,ggml_tensor*dst) {
 const size_t bytes=ggml_nbytes(dst);
 check(bytes==gguf_get_tensor_size(f.uf,w.id),"prefix weight size mismatch: "+w.name);
 std::vector<uint8_t> raw(bytes);
 f.read(f.data+gguf_get_tensor_offset(f.uf,w.id),raw.data(),bytes);
 // Quantized OpenCL SoA conversion must receive a complete contiguous tensor.
 ggml_backend_tensor_set(dst,raw.data(),0,bytes);
}
static std::unique_ptr<Prefix> make_prefix(
 File& f,int layer,ggml_backend_t gpu,int tokens,int topk,const LayerGeometry &g) {
 auto p=std::make_unique<Prefix>();p->backend=gpu;
 constexpr size_t graph_capacity=512;
 ggml_init_params params={1536*ggml_tensor_overhead()+
                          ggml_graph_overhead_custom(graph_capacity,false),
                          nullptr,true};
 p->ctx=ggml_init(params);
 check(p->ctx!=nullptr,"full-layer GGML graph context failed");
 ggml_context *c=p->ctx;
 p->input=ggml_new_tensor_2d(c,GGML_TYPE_F32,g.embd,tokens);
 p->positions=ggml_new_tensor_1d(c,GGML_TYPE_I32,tokens);
 p->mask=ggml_new_tensor_2d(c,GGML_TYPE_F32,tokens,tokens);
 ggml_set_name(p->input,"phone_local_layer_input");
 ggml_set_name(p->positions,"phone_local_positions");
 ggml_set_name(p->mask,"phone_local_causal_mask");
 auto norm_a=p->weight(f,layer,"attn_norm");
 auto wq=p->weight(f,layer,"attn_q");
 auto wk=p->weight(f,layer,"attn_k");
 auto wv=p->weight(f,layer,"attn_v");
 auto wo=p->weight(f,layer,"attn_output");
 auto nq=p->weight(f,layer,"attn_q_norm");
 auto nk=p->weight(f,layer,"attn_k_norm");
 auto norm_f=p->weight(f,layer,"ffn_norm");
 auto router=p->weight(f,layer,"ffn_gate_inp");

 // Original Qwen3-MoE order: RMS Norm -> QKV -> headwise Q/K RMS Norm
 // -> NeoX RoPE -> causal GQA Attention -> attention output + residual.
 ggml_tensor *n=ggml_mul(c,ggml_rms_norm(c,p->input,g.rms_eps),norm_a);
 ggml_tensor *q=ggml_mul_mat(c,wq,n);
 ggml_tensor *k=ggml_mul_mat(c,wk,n);
 ggml_tensor *v=ggml_mul_mat(c,wv,n);
 q=ggml_reshape_3d(c,q,g.head_dim,g.heads,tokens);
 k=ggml_reshape_3d(c,k,g.head_dim,g.kv_heads,tokens);
 v=ggml_reshape_3d(c,v,g.head_dim,g.kv_heads,tokens);
 q=ggml_mul(c,ggml_rms_norm(c,q,g.rms_eps),nq);
 k=ggml_mul(c,ggml_rms_norm(c,k,g.rms_eps),nk);
 q=ggml_rope_ext(c,q,p->positions,nullptr,g.head_dim,
                 GGML_ROPE_TYPE_NEOX,g.context,g.rope_base,g.rope_scale,
                 0.0f,1.0f,32.0f,1.0f);
 k=ggml_rope_ext(c,k,p->positions,nullptr,g.head_dim,
                 GGML_ROPE_TYPE_NEOX,g.context,g.rope_base,g.rope_scale,
                 0.0f,1.0f,32.0f,1.0f);
 q=ggml_permute(c,q,0,2,1,3); // [head_dim,tokens,heads]
 k=ggml_permute(c,k,0,2,1,3); // [head_dim,tokens,kv_heads]
 v=ggml_permute(c,v,0,2,1,3);
 ggml_tensor *kq=ggml_mul_mat(c,k,q); // [keys,queries,heads]
 ggml_mul_mat_set_prec(kq,GGML_PREC_F32);
 kq=ggml_soft_max_ext(c,kq,p->mask,1.0f/std::sqrt(float(g.head_dim)),0.0f);
 v=ggml_cont(c,ggml_transpose(c,v));
 ggml_tensor *av=ggml_mul_mat(c,v,kq);
 av=ggml_permute(c,av,0,2,1,3);
 av=ggml_cont_2d(c,av,g.head_dim*g.heads,tokens);
 ggml_tensor *attn_out=ggml_mul_mat(c,wo,av);
 p->residual=ggml_add(c,p->input,attn_out);

 // GPU executes the full real Router: Softmax -> TopK -> normalized
 // expert weights. Both FFN shards subsequently consume exactly these IDs.
 p->ffn_norm=ggml_mul(c,ggml_rms_norm(c,p->residual,g.rms_eps),norm_f);
 ggml_tensor *router_logits=ggml_mul_mat(c,router,p->ffn_norm);
 ggml_tensor *probs=ggml_soft_max(c,router_logits);
 // ggml_argsort_top_k returns a STRIDED view of all expert IDs.
 // Pack the [topk, tokens] view before handing it to separately allocated
 // FFN graphs, because ggml_backend_tensor_copy requires identical layouts.
 p->ids=ggml_cont(c,ggml_argsort_top_k(c,probs,topk));
 ggml_tensor *probs_3d=ggml_reshape_3d(c,probs,1,g.experts,tokens);
 ggml_tensor *chosen=ggml_get_rows(c,probs_3d,p->ids);
 chosen=ggml_reshape_2d(c,chosen,topk,tokens);
 ggml_tensor *weight_sum=ggml_clamp(c,ggml_sum_rows(c,chosen),6.103515625e-5f,INFINITY);
 p->mix=ggml_reshape_3d(c,ggml_div(c,chosen,weight_sum),1,topk,tokens);

 p->graph=ggml_new_graph_custom(c,graph_capacity,false);
 ggml_build_forward_expand(p->graph,p->residual);
 ggml_build_forward_expand(p->graph,p->ffn_norm);
 ggml_build_forward_expand(p->graph,p->ids);
 ggml_build_forward_expand(p->graph,p->mix);
 for(int i=0;i<ggml_graph_n_nodes(p->graph);++i){
  auto *node=ggml_graph_node(p->graph,i);
  check(ggml_backend_supports_op(gpu,node),
        std::string("OpenCL cannot execute full-layer op: ")+ggml_op_name(node->op));
 }
 std::fprintf(stderr,"[PHONE_LAYER_BOOT] layer=%d phase=gpu_attn_router_alloc_begin\n",layer);
 p->buffer=ggml_backend_alloc_ctx_tensors(c,gpu);
 check(p->buffer!=nullptr,"GPU attention/router buffer allocation failed");
 std::fprintf(stderr,"[PHONE_LAYER_BOOT] layer=%d phase=gpu_attn_router_weights_begin\n",layer);
 for(const auto &w:p->weights)prefix_upload_weight(f,w.first,w.second);
 std::vector<int32_t> pos(tokens);
 for(int t=0;t<tokens;t++)pos[t]=t;
 ggml_backend_tensor_set(p->positions,pos.data(),0,pos.size()*sizeof(int32_t));
 std::vector<float> mask(static_cast<size_t>(tokens)*tokens);
 for(int qpos=0;qpos<tokens;qpos++)
  for(int kpos=0;kpos<tokens;kpos++)
   mask[static_cast<size_t>(qpos)*tokens+kpos]=kpos<=qpos?0.0f:-INFINITY;
 ggml_backend_tensor_set(p->mask,mask.data(),0,mask.size()*sizeof(float));
 ggml_backend_synchronize(gpu);
 std::fprintf(stderr,"[PHONE_LAYER_BOOT] layer=%d phase=gpu_attn_router_ready\n",layer);
 return p;
}

struct FullLayer {
 int il=-1;
 std::unique_ptr<Prefix> pre;
 std::unique_ptr<Graph> gpu_ffn;
 std::unique_ptr<Graph> cpu_ffn;
};
struct StageBreakdown {
 double gpu_norm=0,gpu_ids=0,gpu_mix=0;
 double cpu_norm=0,cpu_ids=0,cpu_mix=0;
 double cpu_sync=0,gpu_sync=0,other=0;
 double accounted() const {
  return gpu_norm+gpu_ids+gpu_mix+cpu_norm+cpu_ids+cpu_mix+cpu_sync+gpu_sync;
 }
};
struct FullTimes {
 double attention_router=0,stage=0,gpu_ffn=0,cpu_ffn=0,join=0,total=0;
 StageBreakdown stage_parts;
};
struct FullRound {
 double wall=0;
 std::vector<FullTimes> layers;
 std::vector<float> output;
};

static std::vector<float> full_initial(int64_t embd,int tokens) {
 std::vector<float> data(static_cast<size_t>(embd)*tokens);
 for(size_t i=0;i<data.size();++i)
  data[i]=0.02f*std::sin(float(i%8191)*0.037f);
 return data;
}

// Do not let GGML's hard assertion abort the phone process with no context.
// A view can have the correct shape but a different stride layout.
static void full_copy_exact(const ggml_tensor *source, ggml_tensor *dest,
                            int layer, const char *label) {
 bool compatible=source && dest && source->type==dest->type;
 for(int i=0;i<GGML_MAX_DIMS && compatible;i++){
  compatible=source->ne[i]==dest->ne[i] && source->nb[i]==dest->nb[i];
 }
 if(!compatible){
  std::fprintf(stderr,"[PHONE_FULL_LAYOUT_ERROR] layer=%d tensor=%s\n",layer,label);
  if(source) std::fprintf(stderr,
   "  src type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
   ggml_type_name(source->type),
   (long long)source->ne[0],(long long)source->ne[1],
   (long long)source->ne[2],(long long)source->ne[3],
   source->nb[0],source->nb[1],source->nb[2],source->nb[3]);
  if(dest) std::fprintf(stderr,
   "  dst type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
   ggml_type_name(dest->type),
   (long long)dest->ne[0],(long long)dest->ne[1],
   (long long)dest->ne[2],(long long)dest->ne[3],
   dest->nb[0],dest->nb[1],dest->nb[2],dest->nb[3]);
  throw std::runtime_error(std::string("FFN input layout mismatch: ")+label);
 }
 ggml_backend_tensor_copy(source,dest);
}

static FullTimes run_full_layer(FullLayer &layer,ggml_backend_t gpu,
                               std::vector<float> &hidden) {
 FullTimes t;
 const bool step_trace=std::getenv("PHONE_FULL_TRACE")!=nullptr;
 const auto trace=[&](const char *phase){
  if(step_trace)std::fprintf(stderr,"[PHONE_FULL_STEP] layer=%d phase=%s\n",layer.il,phase);
 };
 const auto begin=clk::now();
 auto &pre=*layer.pre;
 auto &gb=*layer.gpu_ffn;
 check(hidden.size()*sizeof(float)==ggml_nbytes(pre.input),"hidden state shape mismatch");

 auto phase=clk::now();
 trace("attention_router_begin");
 ggml_backend_tensor_set(pre.input,hidden.data(),0,hidden.size()*sizeof(float));
 check(ggml_backend_graph_compute(gpu,pre.graph)==GGML_STATUS_SUCCESS,
       "GPU attention/router graph failed");
 ggml_backend_synchronize(gpu);
 t.attention_router=ms(phase);
 trace("attention_router_ok");

 phase=clk::now();
 trace("ffn_stage_begin");
 const auto stage_copy=[&](const ggml_tensor *src,ggml_tensor *dst,
                           const char *name,double &elapsed){
  const auto start=clk::now();
  full_copy_exact(src,dst,layer.il,name);
  elapsed=ms(start);
 };
 const auto stage_sync=[&](ggml_backend_t backend,double &elapsed){
  const auto start=clk::now();
  ggml_backend_synchronize(backend);
  elapsed=ms(start);
 };
 auto &part=t.stage_parts;
 stage_copy(pre.ffn_norm,gb.x,"GPU/ffn_norm",part.gpu_norm);
 stage_copy(pre.ids,gb.ids,"GPU/expert_ids",part.gpu_ids);
 stage_copy(pre.mix,gb.mix,"GPU/expert_weights",part.gpu_mix);
 if(layer.cpu_ffn){
  auto &cb=*layer.cpu_ffn;
  stage_copy(pre.ffn_norm,cb.x,"CPU/ffn_norm",part.cpu_norm);
  stage_copy(pre.ids,cb.ids,"CPU/expert_ids",part.cpu_ids);
  stage_copy(pre.mix,cb.mix,"CPU/expert_weights",part.cpu_mix);
  stage_sync(cb.backend,part.cpu_sync);
 }
 stage_sync(gpu,part.gpu_sync);
 t.stage=ms(phase);
 part.other=std::max(0.0,t.stage-part.accounted());
 trace("ffn_stage_ok");
 trace("ffn_parallel_begin");

 if(layer.cpu_ffn){
  auto &cb=*layer.cpu_ffn;
  std::exception_ptr error;
  std::thread worker([&]{try{t.cpu_ffn=compute(cb);}catch(...){error=std::current_exception();}});
  try{t.gpu_ffn=compute(gb);}catch(...){worker.join();throw;}
  worker.join();
  if(error)std::rethrow_exception(error);
 }else{
  t.gpu_ffn=compute(gb);
 }
 trace("ffn_parallel_ok");
 trace("join_begin");
 phase=clk::now();
 auto gpu_partial=output(gb);
 if(layer.cpu_ffn){
  auto cpu_partial=output(*layer.cpu_ffn);
  check(cpu_partial.size()==gpu_partial.size(),"FFN partial shapes differ");
  for(size_t i=0;i<gpu_partial.size();i++)gpu_partial[i]+=cpu_partial[i];
 }
 std::vector<float> residual(hidden.size());
 ggml_backend_tensor_get(pre.residual,residual.data(),0,residual.size()*sizeof(float));
 check(residual.size()==gpu_partial.size(),"FFN/residual shape mismatch");
 for(size_t i=0;i<residual.size();i++)residual[i]+=gpu_partial[i];
 hidden.swap(residual);
 t.join=ms(phase);
 t.total=ms(begin);
 trace("join_ok");
 return t;
}
static std::vector<std::unique_ptr<FullLayer>> full_build_layers(
 File&f,const Opt&o,ggml_backend_t gpu,ggml_backend_t cpu,bool mixed) {
 std::vector<std::unique_ptr<FullLayer>> layers;
 for(int il=o.layer;il<o.layer+o.layers;il++){
  const LayerGeometry g=layer_geometry(f,il,o.topk);
  const int64_t alignment=std::lcm<int64_t>(128,ggml_blck_size(f.get(il,"ffn_down_exps").t->type));
  check(g.ff%alignment==0,"FFN width not aligned for quantized tensor split");
  const int64_t blocks=g.ff/alignment;
  int64_t cb=0;
  if(mixed){
   cb=std::llround(blocks*o.cpu_ratio);
   cb=std::min<int64_t>(blocks-1,std::max<int64_t>(1,cb));
  }
  const int64_t cpu_width=cb*alignment;
  const int64_t gpu_width=g.ff-cpu_width;
  std::fprintf(stderr,
   "[PHONE_LAYER_LAYOUT] layer=%d mode=%s n_embd=%lld heads=%lld kv_heads=%lld head_dim=%lld ffn=%lld cpu_ffn=%lld gpu_ffn=%lld actual_cpu_ratio=%.3f\n",
   il,mixed?"CPU_GPU":"GPU_ONLY",(long long)g.embd,(long long)g.heads,
   (long long)g.kv_heads,(long long)g.head_dim,(long long)g.ff,
   (long long)cpu_width,(long long)gpu_width,double(cpu_width)/g.ff);
  auto lr=std::make_unique<FullLayer>();lr->il=il;
  lr->pre=make_prefix(f,il,gpu,o.tokens,o.topk,g);
  lr->gpu_ffn=build(f,il,gpu,0,gpu_width,o.tokens,o.topk);
  if(cpu_width>0){
   lr->cpu_ffn=build(f,il,cpu,gpu_width,cpu_width,o.tokens,o.topk);
  }
  std::fprintf(stderr,
   "[PHONE_FULL_STAGE_BYTES] mode=%s layer=%d norm_bytes=%zu ids_bytes=%zu mix_bytes=%zu copies_gpu=3 copies_cpu=%d\n",
   mixed?"CPU_GPU":"GPU_ONLY",il,
   ggml_nbytes(lr->pre->ffn_norm),ggml_nbytes(lr->pre->ids),
   ggml_nbytes(lr->pre->mix),mixed?3:0);
  layers.push_back(std::move(lr));
 }
 return layers;
}
static std::vector<FullRound> full_measure(
 const Opt&o,ggml_backend_t gpu,const std::vector<std::unique_ptr<FullLayer>>&layers,
 const std::vector<float> &initial,const char *mode) {
 std::vector<FullRound> measured;
 for(int rep=-o.warmup;rep<o.runs;rep++){
  FullRound round;
  round.layers.reserve(layers.size());
  std::vector<float> hidden=initial;
  const auto begin=clk::now();
  for(const auto &layer:layers){
   round.layers.push_back(run_full_layer(*layer,gpu,hidden));
  }
  // Record wall time BEFORE formatting/printing per-layer records. Console I/O
  // can otherwise create apparent multi-ms latency spikes in total_ms.
  round.wall=ms(begin);
  round.output.swap(hidden);
  if(rep>=0){
   for(size_t i=0;i<layers.size();i++){
    const auto &t=round.layers[i];
    const auto &p=t.stage_parts;
    std::cout<<"[PHONE_FULL_LAYER] mode="<<mode<<" layer="<<layers[i]->il
      <<" rep="<<rep<<" attn_router_ms="<<t.attention_router
      <<" stage_ms="<<t.stage<<" gpu_ffn_ms="<<t.gpu_ffn
      <<" cpu_ffn_ms="<<t.cpu_ffn<<" join_ms="<<t.join
      <<" total_ms="<<t.total<<"\n";
    std::cout<<"[PHONE_FULL_STAGE] mode="<<mode<<" layer="<<layers[i]->il
      <<" rep="<<rep<<" gpu_norm_ms="<<p.gpu_norm
      <<" gpu_ids_ms="<<p.gpu_ids<<" gpu_mix_ms="<<p.gpu_mix
      <<" cpu_norm_ms="<<p.cpu_norm<<" cpu_ids_ms="<<p.cpu_ids
      <<" cpu_mix_ms="<<p.cpu_mix<<" cpu_sync_ms="<<p.cpu_sync
      <<" gpu_sync_ms="<<p.gpu_sync<<" other_ms="<<p.other
      <<" accounted_ms="<<p.accounted()<<" total_ms="<<t.stage<<"\n";
   }
   std::cout<<"[PHONE_FULL_RUN] mode="<<mode<<" rep="<<rep
            <<" layers="<<layers.size()<<" total_ms="<<round.wall<<"\n";
   measured.push_back(std::move(round));
  }
 }
 // Median cost per copy and per synchronization point, grouped by layer.
 for(size_t li=0;li<layers.size();li++){
  std::vector<double> stage,gpu_norm,gpu_ids,gpu_mix;
  std::vector<double> cpu_norm,cpu_ids,cpu_mix,cpu_sync,gpu_sync,other;
  for(const auto &round:measured){
   const FullTimes &t=round.layers[li];
   const auto &p=t.stage_parts;
   stage.push_back(t.stage);
   gpu_norm.push_back(p.gpu_norm);
   gpu_ids.push_back(p.gpu_ids);
   gpu_mix.push_back(p.gpu_mix);
   cpu_norm.push_back(p.cpu_norm);
   cpu_ids.push_back(p.cpu_ids);
   cpu_mix.push_back(p.cpu_mix);
   cpu_sync.push_back(p.cpu_sync);
   gpu_sync.push_back(p.gpu_sync);
   other.push_back(p.other);
  }
  std::cout<<"[PHONE_FULL_STAGE_SUM] mode="<<mode<<" layer="<<layers[li]->il
   <<" total_median_ms="<<med(stage)
   <<" gpu_norm_median_ms="<<med(gpu_norm)
   <<" gpu_ids_median_ms="<<med(gpu_ids)
   <<" gpu_mix_median_ms="<<med(gpu_mix)
   <<" cpu_norm_median_ms="<<med(cpu_norm)
   <<" cpu_ids_median_ms="<<med(cpu_ids)
   <<" cpu_mix_median_ms="<<med(cpu_mix)
   <<" cpu_sync_median_ms="<<med(cpu_sync)
   <<" gpu_sync_median_ms="<<med(gpu_sync)
   <<" other_median_ms="<<med(other)<<"\n";
 }
 return measured;
}

// One diagnostic pass per mode, fully OUTSIDE the measured trials.
// Capture hidden states after EVERY layer and each GPU Router Top-K choice.
struct FullDiagnostics {
 std::vector<std::vector<float>> layer_output;
 std::vector<std::vector<int32_t>> router_ids;
};
static FullDiagnostics full_capture_diagnostics(
 ggml_backend_t gpu,
 const std::vector<std::unique_ptr<FullLayer>> &layers,
 const std::vector<float> &initial, const char *mode) {
 FullDiagnostics d;
 d.layer_output.reserve(layers.size());
 d.router_ids.reserve(layers.size());
 std::vector<float> hidden=initial;
 std::fprintf(stderr,"[PHONE_FULL_DIAG] mode=%s phase=begin layers=%zu\n",mode,layers.size());
 for(const auto &layer:layers){
  (void)run_full_layer(*layer,gpu,hidden);
  d.layer_output.push_back(hidden);
  const ggml_tensor *ids=layer->pre->ids;
  check(ids->type==GGML_TYPE_I32,"Router IDs type mismatch during diagnostics");
  const size_t count=static_cast<size_t>(ggml_nelements(ids));
  std::vector<int32_t> values(count);
  ggml_backend_tensor_get(ids,values.data(),0,count*sizeof(int32_t));
  d.router_ids.push_back(std::move(values));
 }
 std::fprintf(stderr,"[PHONE_FULL_DIAG] mode=%s phase=end layers=%zu\n",mode,layers.size());
 return d;
}

static bool full_report_layer_checks(
 const Opt &o,const FullDiagnostics &base,const FullDiagnostics &mixed) {
 check(base.layer_output.size()==static_cast<size_t>(o.layers) &&
       mixed.layer_output.size()==base.layer_output.size() &&
       base.router_ids.size()==base.layer_output.size() &&
       mixed.router_ids.size()==base.layer_output.size(),
       "per-layer diagnostic count mismatch");
 bool all_ok=true;
 for(size_t li=0;li<base.layer_output.size();li++){
  const auto &a=base.layer_output[li];
  const auto &b=mixed.layer_output[li];
  check(a.size()==b.size(),"per-layer hidden-state lengths differ");
  double error2=0,reference2=0,max_abs=0;
  for(size_t j=0;j<a.size();j++){
   check(std::isfinite(a[j])&&std::isfinite(b[j]),
         "non-finite value in per-layer hidden-state check");
   const double diff=static_cast<double>(a[j])-b[j];
   error2+=diff*diff;
   reference2+=static_cast<double>(a[j])*a[j];
   max_abs=std::max(max_abs,std::abs(diff));
  }
  const double relative_l2=std::sqrt(error2/std::max(reference2,1.0e-24));
  const auto &ref_ids=base.router_ids[li];
  const auto &mix_ids=mixed.router_ids[li];
  const size_t expected=static_cast<size_t>(o.tokens)*static_cast<size_t>(o.topk);
  check(ref_ids.size()==expected&&mix_ids.size()==expected,
        "Router Top-K IDs diagnostic shape mismatch");
  size_t changed_choices=0,changed_tokens=0;
  for(int token=0;token<o.tokens;token++){
   bool token_changed=false;
   for(int k=0;k<o.topk;k++){
    const size_t j=static_cast<size_t>(token)*o.topk+k;
    if(ref_ids[j]!=mix_ids[j]){
     token_changed=true;
     changed_choices++;
    }
   }
   if(token_changed)changed_tokens++;
  }
  const bool ok=relative_l2<0.03;
  all_ok=all_ok&&ok;
  std::cout<<"[PHONE_FULL_LAYER_CHECK] layer="<<(o.layer+static_cast<int>(li))
           <<" rel_l2="<<relative_l2<<" max_abs="<<max_abs
           <<" changed_router_tokens="<<changed_tokens
           <<" changed_router_slots="<<changed_choices
           <<" status="<<(ok?"OK":"CHECK")<<"\n";
 }
 return all_ok;
}

static int run_full(const Opt&o) {
 check(o.cpu_ratio>0,"--full-layer requires --cpu-ratio > 0 for a comparison");
 ggml_backend_load_all();
 auto gpu=ggml_backend_dev_init(device(true),nullptr);
 auto cpu=ggml_backend_dev_init(device(false),nullptr);
 check(gpu&&cpu,"GPU/CPU backend init failed");
 threads(cpu,o.threads);
 std::cout<<std::fixed<<std::setprecision(3);
 std::fprintf(stderr,
  "[PHONE_FULL_CONFIG] first_layer=%d layers=%d tokens=%d topk=%d cpu_ratio_request=%.4f threads=%d warmup=%d runs=%d attn=causal_nonflash router=real_gguf_weights input=synthetic\n",
  o.layer,o.layers,o.tokens,o.topk,o.cpu_ratio,o.threads,o.warmup,o.runs);
 int status=0;
 {
  File f(o.model);
  const auto geom=layer_geometry(f,o.layer,o.topk);
  auto initial=full_initial(geom.embd,o.tokens);
  std::vector<FullRound> baseline,split;
  FullDiagnostics base_diag,mixed_diag;
  std::fprintf(stderr,"[PHONE_FULL_BOOT] phase=gpu_only_build_begin\n");
  {
   auto all=full_build_layers(f,o,gpu,cpu,false);
   std::fprintf(stderr,"[PHONE_FULL_BOOT] phase=gpu_only_build_ok\n");
   baseline=full_measure(o,gpu,all,initial,"GPU_ONLY");
   base_diag=full_capture_diagnostics(gpu,all,initial,"GPU_ONLY");
  }
  std::fprintf(stderr,"[PHONE_FULL_BOOT] phase=mixed_build_begin\n");
  {
   auto all=full_build_layers(f,o,gpu,cpu,true);
   std::fprintf(stderr,"[PHONE_FULL_BOOT] phase=mixed_build_ok\n");
   split=full_measure(o,gpu,all,initial,"CPU_GPU");
   mixed_diag=full_capture_diagnostics(gpu,all,initial,"CPU_GPU");
  }
  const bool layer_checks_ok=full_report_layer_checks(o,base_diag,mixed_diag);
  std::vector<double> a,b;
  for(const auto &v:baseline)a.push_back(v.wall);
  for(const auto &v:split)b.push_back(v.wall);
  const double base_ms=med(a),mixed_ms=med(b);
  const auto &reference=baseline.back().output;
  const auto &got=split.back().output;
  check(reference.size()==got.size(),"full-layer comparison output length mismatch");
  double error=0,norm=0,max_abs=0;
  for(size_t i=0;i<reference.size();i++){
   check(std::isfinite(reference[i])&&std::isfinite(got[i]),
         "non-finite full-layer hidden state");
   const double diff=double(reference[i])-got[i];
   error+=diff*diff;
   norm+=double(reference[i])*reference[i];
   max_abs=std::max(max_abs,std::abs(diff));
  }
  const double rel=std::sqrt(error/std::max(norm,1.0e-24));
  const bool good=rel<0.03;
  std::cout<<"[PHONE_FULL_CHECK] rel_l2="<<rel<<" max_abs="<<max_abs
           <<" status="<<(good?"OK":"CHECK")<<"\n";
  std::cout<<"[PHONE_FULL_SUM] gpu_only_median_ms="<<base_ms
           <<" mixed_median_ms="<<mixed_ms
           <<" speedup="<<base_ms/mixed_ms
           <<" layers="<<o.layers<<" tokens="<<o.tokens<<"\n";
  // A failed numerical check invalidates the timing comparison.
  if(!good || !layer_checks_ok)status=2;
 }
 ggml_backend_free(cpu);
 ggml_backend_free(gpu);
 return status;
}

int main(int argc,char **argv){try{const Opt o=options(argc,argv);return o.full_layer?run_full(o):run(o);}catch(const std::exception&e){std::cerr<<"[PHONE_LOCAL_ERROR] "<<e.what()<<"\n";return 1;}}

