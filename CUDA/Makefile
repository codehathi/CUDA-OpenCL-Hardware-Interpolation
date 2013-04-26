NAME := cuda_interp

CC   := gcc
NVCC := nvcc --ptxas-options=-v
LINK := $(CC)

CUDA_DIR   := /nfs/sw/cuda/current/
CUDA_ARCH  := -arch=sm_13

# You shouldn't have to go below here
CUDA_INC := -I$(CUDA_DIR)/include/
CUDA_LIB := -L$(CUDA_DIR)/lib64/

MODULES := src include 

CFLAGS := -O3
LFLAGS := -O3
NFLAGS := -O3 -I$(CUDA_INC) $(CUDA_ARCH) 

DEPFLAGS := -O3 -std=c99 

LIBS := -lm

CFLAGS += -D__CUDA__ $(CUDA_INC)
LFLAGS += -D__CUDA__ $(CUDA_LIB)
NFLAGS += -D__CUDA__
LIBS   += -lcudart

DIRNAME = `dirname $1`
MAKEDEPS = gcc -MM -MG $2 $3 | sed -e "s@^\(.*\)\.o:@.dep/$1/\1.d obj/$1/\1.o:@"

.PHONY : all

all : $(NAME)

# look for include files in each of the modules
INCLUDEFLAGS := $(patsubst %, -I%, $(MODULES)) 

CFLAGS += $(INCLUDEFLAGS)
CPPFLAGS += $(INCLUDEFLAGS)
DEPFLAGS += $(INCLUDEFLAGS)

# each module will add to this
SRC :=  $(wildcard $(patsubst %, %/*.c, $(MODULES))) $(wildcard $(patsubst %, %/*.cu, $(MODULES)))

# determine the object files
OBJ :=   $(patsubst %.c, obj/%.o, $(filter %.c, $(SRC)))  $(patsubst %.cu, obj/%.o, $(filter %.cu, $(SRC)))

# link the program
$(NAME) : $(OBJ)
	$(LINK) $(LFLAGS) -o $@ $(OBJ) $(LIBS)

# calculate C include dependencies
.dep/%.d : %.c
	@mkdir -p `echo '$@' | sed -e 's|/[^/]*.d$$||'`
	$(call MAKEDEPS,$(call DIRNAME, $<), $(DEPFLAGS), $<) > $@

.dep/%.d : %.cu
	@mkdir -p `echo '$@' | sed -e 's|/[^/]*.d$$||'`
	$(call MAKEDEPS,$(call DIRNAME, $<), $(NFLAGS), $<) > $@

obj/%.o : %.c
	@mkdir -p `echo '$@' | sed -e 's|/[^/]*.o$$||'`
	$(CC) $(CFLAGS) -c -o $@ $<

obj/%.o : %.cu
	@mkdir -p `echo '$@' | sed -e 's|/[^/]*.o$$||'`
	$(NVCC) $(NFLAGS) $(INCLUDEFLAGS) -c -o $@ $< 

# include the C include dependencies
DEP := $(patsubst obj/%.o, .dep/%.d, $(OBJ))

ifneq ($(MAKECMDGOALS),clean)
-include $(DEP)
endif

clean :
	@rm -f $(NAME) $(OBJ) $(DEP)


