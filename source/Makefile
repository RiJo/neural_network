CC          = gcc
CCFLAGS     = -Wall -Werror -std=gnu99
CCOPTS      = -O3 -march=core2 -mtune=core2
CCDEBUG     = -g -D=_DEBUG_
CCPROFILE   = -pg
LDFLAGS     = -lm
NAME        = test

OBJS        = \
              neural_network.o          \
              neuron.o                  \
              synapse.o                 \
              supervised.o              \

all: $(NAME)

debug: CCFLAGS += $(CCDEBUG)
debug: $(NAME)

$(NAME): $(OBJS) $(NAME).o Makefile
	$(CC) $(CCFLAGS) $(OBJS) $(NAME).c $(LDFLAGS) -o $@

unit: CCFLAGS += $(CCDEBUG)
unit: $(OBJS) unit_tests.o Makefile
	$(CC) $(CCFLAGS) $(OBJS) unit_tests.c $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(NAME) $(OBJS)