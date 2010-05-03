CC          = gcc
CCFLAGS     = -Wall -Werror -std=gnu99
CCOPTS      = -O3 -march=core2 -mtune=core2
CCDEBUG     = -g -D=_DEBUG_
CCPROFILE   = -pg
LDFLAGS     = -lm
NAME        = test

OBJS        = $(NAME).o                 \
              neural_network.o          \
              neuron.o                  \
              synapse.o                 \
              train_data.o              \

all: $(NAME)

debug: CCFLAGS += $(CCDEBUG)
debug: $(NAME)

$(NAME): $(OBJS) Makefile
	$(CC) $(CCFLAGS) $(OBJS) $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(NAME) $(OBJS)