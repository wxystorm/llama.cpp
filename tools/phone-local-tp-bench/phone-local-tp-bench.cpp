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
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms(clk::time_point t) { return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }
static void check(bool x,const std::string & msg) { if(!x) throw std::runtime_error(msg); }
static double med(std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];}
struct Opt {std::string model;int layer=0,layers=2,tokens=400,topk=8,threads=4,runs=5,warmup=2;double cpu_ratio=0.25;bool probe_opencl=false;};
static Opt options(int argc,char **argv) {
 Opt o;for(int i=1;i<argc;i++){std::string a=argv[i];if(a=="--probe-opencl"){o.probe_opencl=true;continue;}check(i+1<argc,"missing argument for "+a);std::string v=argv[++i];
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
int main(int argc,char **argv){try{return run(options(argc,argv));}catch(const std::exception&e){std::cerr<<"[PHONE_LOCAL_ERROR] "<<e.what()<<"\n";return 1;}}
