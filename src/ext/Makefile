top_builddir = ../..
include $(top_builddir)/src/Makefile.global

ifneq ($(with_llvm), yes)
    $(error "not building with LLVM support")
endif

CFLAGS += $(LLVM_CFLAGS)
CXXFLAGS += $(LLVM_CXXFLAGS)
override CPPFLAGS := $(LLVM_CPPFLAGS) $(CPPFLAGS)
SHLIB_LINK += $(LLVM_LIBS)

override COMPILER = $(CXX) $(CFLAGS)

all: lambda_ext lambda_ext.bc kmeans_ext kmeans_ext.bc pagerank_ext pagerank_ext.bc autodiff_ext autodiff_ext.bc autodiff_timing autodiff_timing.bc gradient_desc_ext gradient_desc_ext.bc gradient_desc_m_ext gradient_desc_m_ext.bc

autodiff_ext: autodiff_ext.c
	$(CC) $(LLVM_CPPFLAGS) $(CPPFLAGS) -O2 -I../include -fPIC -c autodiff_ext.c
	$(CC) $(LLVM_CPPFLAGS) $(CPPFLAGS) -O2 -I../include -shared -o autodiff_ext.so autodiff_ext.o

autodiff_timing: autodiff_timing.c
	$(CC) $(LLVM_CPPFLAGS) $(CPPFLAGS) -O2 -I../include -fPIC -c autodiff_timing.c
	$(CC) $(LLVM_CPPFLAGS) $(CPPFLAGS) -O2 -I../include -shared -o autodiff_timing.so autodiff_timing.o

gradient_desc_ext: gradient_desc_ext.c
	$(CC) $(LLVM_CPPFLAGS) $(CPPFLAGS) -O2 -I../include -fPIC -c gradient_desc_ext.c
	$(CC) $(LLVM_CPPFLAGS) $(CPPFLAGS) -O2 -I../include -shared -o gradient_desc_ext.so gradient_desc_ext.o

gradient_desc_m_ext: gradient_desc_m_ext.c
	$(CC) $(LLVM_CPPFLAGS) $(CPPFLAGS) -O2 -I../include -fPIC -c gradient_desc_m_ext.c
	$(CC) $(LLVM_CPPFLAGS) $(CPPFLAGS) -O2 -I../include -shared -o gradient_desc_m_ext.so gradient_desc_m_ext.o

lambda_ext: lambda_ext.c
	$(CC) $(LLVM_CPPFLAGS) $(CPPFLAGS) -O2 -I../include -fPIC -c lambda_ext.c
	$(CC) $(LLVM_CPPFLAGS) $(CPPFLAGS) -O2 -I../include -shared -o lambda_ext.so lambda_ext.o

kmeans_ext: kmeans_ext.c
	$(CC) $(LLVM_CPPFLAGS) $(CPPFLAGS) -O2 -I../include -fPIC -c kmeans_ext.c
	$(CC) $(LLVM_CPPFLAGS) $(CPPFLAGS) -O2 -I../include -shared -o kmeans_ext.so kmeans_ext.o

pagerank_ext: pagerank_ext.c
	$(CC) $(LLVM_CPPFLAGS) $(CPPFLAGS) -O2 -I../include -fPIC -c pagerank_ext.c
	$(CC) $(LLVM_CPPFLAGS) $(CPPFLAGS) -O2 -I../include -shared -o pagerank_ext.so pagerank_ext.o

install-postgres-bitcode: $(OBJS) all
	$(call install_llvm_module,ext,lambda_ext.o)
	$(call install_llvm_module,ext,kmeans_ext.o)
	$(call install_llvm_module,ext,pagerank_ext.o)
	$(call install_llvm_module,ext,autodiff_ext.o)
	$(call install_llvm_module,ext,autodiff_timing.o)
	$(call install_llvm_module,ext,gradient_desc_ext.o)
	$(call install_llvm_module,ext,gradient_desc_m_ext.o)
	
	
