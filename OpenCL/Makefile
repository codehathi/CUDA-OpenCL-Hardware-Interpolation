NAME := opencl_interp

CC   := gcc
NVCC := nvcc --ptxas-options=-v
LINK := $(CC)

OPENCL_DIR := /nfs/sw/cuda/current/

# You shouldn't have to go below here
OPENCL_INC := -I$(OPENCL_DIR)/include/
OPENCL_LIB := -L$(OPENCL_DIR)/lib64/

MODULES := src include 

CFLAGS := -O3 -std=c99
LFLAGS := -O3 -std=c99

DEPFLAGS := -O3 -std=c99 

LIBS := -lm

CFLAGS += -D__OPENCL__ $(OPENCL_INC)
LFLAGS += -D__OPENCL__ $(OPENCL_LIB)
NFLAGS += -D__OPENCL__
LIBS   += -lOpenCL

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
SRC :=  $(wildcard $(patsubst %, %/*.c, $(MODULES)))

# determine the object files
OBJ :=   $(patsubst %.c, obj/%.o, $(filter %.c, $(SRC)))

# link the program
$(NAME) : $(OBJ)
	$(LINK) $(LFLAGS) -o $@ $(OBJ) $(LIBS)

# calculate C include dependencies
.dep/%.d : %.c
	@mkdir -p `echo '$@' | sed -e 's|/[^/]*.d$$||'`
	$(call MAKEDEPS,$(call DIRNAME, $<), $(DEPFLAGS), $<) > $@

obj/%.o : %.c
	@mkdir -p `echo '$@' | sed -e 's|/[^/]*.o$$||'`
	$(CC) $(CFLAGS) -c -o $@ $<


# include the C include dependencies
DEP := $(patsubst obj/%.o, .dep/%.d, $(OBJ))

ifneq ($(MAKECMDGOALS),clean)
-include $(DEP)
endif

clean :
	@rm -f $(NAME) $(OBJ) $(DEP)


