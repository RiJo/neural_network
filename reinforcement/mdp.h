/*
 * Description...
 */

#ifndef _MDP_H_
#define _MDP_H_

#ifdef DEBUG
#include <assert.h>
#endif

struct mdp_t {
    State *states;
    Action *actions;
    Probability *probability;
    Reward *reward;

    unsigned int state_count;
    unsigned int action_count;
};

typedef struct mdp_t mdp_t;

void mdp_init(mdp_t *);
void mdp_destroy(mdp_t *);

#endif
