/***********************************************************************/
/*                                                                     */
/*   svm_struct_api.c                                                  */
/*                                                                     */
/*   Definition of API for attaching implementing SVM learning of      */
/*   structures (e.g. parsing, multi-label classification, HMM)        */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#include <stdio.h>
#include <string.h>
#include "svm_struct/svm_struct_common.h"
#include "svm_struct_api.h"

int FEATURE_DIM = 48
int LABEL_NUM = 48
long EXAMPLE_NUM = 100;  // the amount of sentences

void svm_struct_learn_api_init(int argc, char* argv[])
{
  /* Called in learning part before anything else is done to allow
     any initializations that might be necessary. */
}

void svm_struct_learn_api_exit()
{
  /* Called in learning part at the very end to allow any clean-up
     that might be necessary. */
}

void svm_struct_classify_api_init(int argc, char* argv[])
{
  /* Called in prediction part before anything else is done to allow
     any initializations that might be necessary. */
}

void svm_struct_classify_api_exit()
{
  /* Called in prediction part at the very end to allow any clean-up
     that might be necessary. */
}

SAMPLE read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads struct examples and returns them in sample. The number of
     examples must be written into sample.n */
  SAMPLE   sample;  /* sample */
  EXAMPLE  *examples;
  long     n;       /* number of examples */

  n = EXAMPLE_NUM;
  examples=(EXAMPLE *)my_malloc(sizeof(EXAMPLE) * n);

  /* fill in your code here */
  FILE* stream = fopen(file, "r");
  char line[1024];
  for (int sentIdx = 0; sentIdx < n; sentIdx++) {
    fgets(line, 1024, stream)
    char* tmp = strdup(line);
    printf("Field 3 would be %s\n", getfield(tmp, 3));
    // NOTE strtok clobbers tmp
    examples[i].x =
    examples[i].y =

    free(tmp);
  }

  sample.n=n;
  sample.examples=examples;
  return(sample);
}

void init_struct_model(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, KERNEL_PARM *kparm)
{
  sm->sizePsi = FEATURE_DIM * LABEL_NUM + LABEL_NUM * LABEL_NUM
}

CONSTSET init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm,
				    STRUCT_LEARN_PARM *sparm)
{
  /* Initializes the optimization problem. Typically, you do not need
     to change this function, since you want to start with an empty
     set of constraints. However, if for example you have constraints
     that certain weights need to be positive, you might put that in
     here. The constraints are represented as lhs[i]*w >= rhs[i]. lhs
     is an array of feature vectors, rhs is an array of doubles. m is
     the number of constraints. The function returns the initial
     set of constraints. */
  CONSTSET c;
  long     sizePsi=sm->sizePsi;
  long     i;
  WORD     words[2];

  if(1) { /* normal case: start with empty set of constraints */
    c.lhs=NULL;
    c.rhs=NULL;
    c.m=0;
  }
  else { /* add constraints so that all learned weights are
            positive. WARNING: Currently, they are positive only up to
            precision epsilon set by -e. */
    c.lhs=my_malloc(sizeof(DOC *)*sizePsi);
    c.rhs=my_malloc(sizeof(double)*sizePsi);
    for(i=0; i<sizePsi; i++) {
      words[0].wnum=i+1;
      words[0].weight=1.0;
      words[1].wnum=0;
      /* the following slackid is a hack. we will run into problems,
         if we have move than 1000000 slack sets (ie examples) */
      c.lhs[i]=create_example(i,0,1000000+i,1,create_svector(words,"",1.0));
      c.rhs[i]=0.0;
    }
  }
  return(c);
}

LABEL classify_struct_example(PATTERN x, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
  LABEL y;

  y.sentenceLen = x.sentenceLen;
  y.labels = (int *)my_malloc(sizeof(int) * y.sentenceLen);

  double score[y.sentenceLen][LABEL_NUM] = {};
  double backTrack[y.sentenceLen][LABEL_NUM] = {};
  int w_o_offset = LABEL_NUM * FEATURE_DIM;

  for (int frameIdx = 0; frameIdx < y.sentenceLen; frameIdx++) {
    for (int labelIdx = 0; labelIdx < LABEL_NUM; labelIdx++ ) {
      if (frameIdx == 0) {
        score[frameIdx][labelIdx] = 0.0;
        // Dot(sm.w[labelIdx * FEATURE_DIM:(labelIdx + 1) * FEATURE_DIM], x[frameIdx])
        int tmp_offset = labelIdx * FEATURE_DIM + 1;
        for (int i = 0; i < FEATURE_DIM; i++) {
          score[frameIdx][labelIdx] += sm->w[tmp_offset + i] * x[frameIdx][i];
        }
        // end of Dot
        backTrack[frameIdx][labelIdx] = labelIdx;
      } else {
        double bestLabel = 0
        double bestScore = 0
        for (int i = 0; i < LABEL_NUM; i++) {
          if (sm->w[w_o_offset + i * LABEL_NUM + labelIdx +1] + score[frameIdx-1][i]) > bestScore) {
            bestScore = sm->w[w_o_offset + i * LABEL_NUM + labelIdx +1] + score[frameIdx-1][i];
            bestLabel = i;
          }
        }
        score[frameIdx][labelIdx] = bestScore;
        // Dot(sm.w[labelIdx * FEATURE_DIM:(labelIdx + 1) * FEATURE_DIM], x[frameIdx])
        int tmp_offset = labelIdx * FEATURE_DIM + 1
        for (int i = 0; i < FEATURE_DIM; i++) {
          score[frameIdx][labelIdx] += sm->w[tmp_offset + i] * x[frameIdx][i];
        }
        // end of Dot
        backTrack[frameIdx][labelIdx] = bestLabel;
      }
    }
  }

  double score_max = score[y.sentenceLen - 1][0];
  int label_max = 0;
  for (int i = 0; i < LABEL_NUM; i++) {
    if (score[yLen - 1][i] > score_max) {
      score_max = score[yLen - 1][i];
      label_max = i;
    }
  }

  y.labels[y.sentenceLen - 1] = label_max;
  int prev = backTrack[y.sentenceLen - 1][label_max];
  for (int i = y.sentenceLen - 2; i >= 0; i--) {
    y.labels[i] = prev;
    if (i > 0) {
      prev = backTrack[i][prev];
    }
  }
  return(y);
}

LABEL find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label ybar for pattern x that that is responsible for
     the most violated constraint for the slack rescaling
     formulation. For linear slack variables, this is that label ybar
     that maximizes

            argmax_{ybar} loss(y,ybar)*(1-psi(x,y)+psi(x,ybar))

     Note that ybar may be equal to y (i.e. the max is 0), which is
     different from the algorithms described in
     [Tschantaridis/05]. Note that this argmax has to take into
     account the scoring function in sm, especially the weights sm.w,
     as well as the loss function, and whether linear or quadratic
     slacks are used. The weights in sm.w correspond to the features
     defined by psi() and range from index 1 to index
     sm->sizePsi. Most simple is the case of the zero/one loss
     function. For the zero/one loss, this function should return the
     highest scoring label ybar (which may be equal to the correct
     label y), or the second highest scoring label ybar, if
     Psi(x,ybar)>Psi(x,y)-1. If the function cannot find a label, it
     shall return an empty label as recognized by the function
     empty_label(y). */
  LABEL ybar;

  /* insert your code for computing the label ybar here */

  return(ybar);
}

LABEL find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
  LABEL y;

  y.sentenceLen = x.sentenceLen;
  y.labels = (int *)my_malloc(sizeof(int) * y.sentenceLen);

  double score[y.sentenceLen][LABEL_NUM] = {};
  double backTrack[y.sentenceLen][LABEL_NUM] = {};
  int w_o_offset = LABEL_NUM * FEATURE_DIM;

  for (int frameIdx = 0; frameIdx < y.sentenceLen; frameIdx++) {
    for (int labelIdx = 0; labelIdx < LABEL_NUM; labelIdx++ ) {
      if (frameIdx == 0) {
        score[frameIdx][labelIdx] = 0.0;
        // Dot(sm.w[labelIdx * FEATURE_DIM:(labelIdx + 1) * FEATURE_DIM], x[frameIdx])
        int tmp_offset = labelIdx * FEATURE_DIM + 1;
        for (int i = 0; i < FEATURE_DIM; i++) {
          score[frameIdx][labelIdx] += sm->w[tmp_offset + i] * x[frameIdx][i];
        }
        // end of Dot
        if(labelIdx != y.labels[frameIdx]) score[frameIdx][labelIdx] += 1;
        backTrack[frameIdx][labelIdx] = labelIdx;
      } else {
        double bestLabel = 0
        double bestScore = 0
        for (int i = 0; i < LABEL_NUM; i++) {
          if (sm->w[w_o_offset + i * LABEL_NUM + labelIdx +1] + score[frameIdx-1][i]) > bestScore) {
            bestScore = sm->w[w_o_offset + i * LABEL_NUM + labelIdx +1] + score[frameIdx-1][i];
            bestLabel = i;
          }
        }
        score[frameIdx][labelIdx] = bestScore;
        // Dot(sm.w[labelIdx * FEATURE_DIM:(labelIdx + 1) * FEATURE_DIM], x[frameIdx])
        int tmp_offset = labelIdx * FEATURE_DIM + 1
        for (int i = 0; i < FEATURE_DIM; i++) {
          score[frameIdx][labelIdx] += sm->w[tmp_offset + i] * x[frameIdx][i];
        }
        // end of Dot
        if(labelIdx != y.labels[frameIdx]) score[frameIdx][labelIdx] += 1;
        backTrack[frameIdx][labelIdx] = bestLabel;
      }
    }
  }

  double score_max = score[y.sentenceLen - 1][0];
  int label_max = 0;
  for (int i = 0; i < LABEL_NUM; i++) {
    if (score[yLen - 1][i] > score_max) {
      score_max = score[yLen - 1][i];
      label_max = i;
    }
  }

  y.labels[y.sentenceLen - 1] = label_max;
  int prev = backTrack[y.sentenceLen - 1][label_max];
  for (int i = y.sentenceLen - 2; i >= 0; i--) {
    y.labels[i] = prev;
    if (i > 0) {
      prev = backTrack[i][prev];
    }
  }
  return(y);
  return(ybar);
}

int empty_label(LABEL y)
{
  /* Returns true, if y is an empty label. An empty label might be
     returned by find_most_violated_constraint_???(x, y, sm) if there
     is no incorrect label that can be found for x, or if it is unable
     to label x at all */
  return(0);
}

SVECTOR *psi(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
  SVECTOR *fvec = (SVECTOR *)my_malloc(sizeof(SVECTOR));
  fvec->words = (WORD *)my_malloc(sizeof(WORD) * sm->sizePsi);

  // set to zeros
  for (int i = 0; i < sm->sizePsi; i++) {
    fvec->words[i].wnum = i;
    fvec->words[i].weight = 0;
  }

  int observationLen = FEATURE_DIM * LABEL_NUM
  int prevY;
  for (int frameIdx = 0; frameIdx < x.sentenceLen; frameIdx++) {
    int offset = FEATURE_DIM * y.labels[frameIdx];
    for (int i = 0; i < FEATURE_DIM; i++) {
      fvec->words[i + offset].weight += x.frames[sentIdx][i];
    }
    if (frameIdx != 0) {
      pvec[observationLen + prevY * LABEL_NUM + y.labels[frameIdx]] += 1
    }
    prevY = y.labels[i]
  }

  return(fvec);
}

double loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm)
{
  int myLoss = 0;
  for (int i = 0; i < y.sentenceLen; i++) {
      if (y.labels[i] != ybar.labels[i]) myLoss++;
  }
  return myLoss
}

int finalize_iteration(double ceps, int cached_constraint, SAMPLE sample, STRUCTMODEL *sm, CONSTSET cset, double *alpha, STRUCT_LEARN_PARM *sparm)
{
  return(0);
}

void print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm, CONSTSET cset, double *alpha, STRUCT_LEARN_PARM *sparm)
{
}

void print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, STRUCT_TEST_STATS *teststats)
{
}

void eval_prediction(long exnum, EXAMPLE ex, LABEL ypred, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, STRUCT_TEST_STATS *teststats)
{
  if(exnum == 0) { }
}

void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm)
{
}

void write_label(FILE *fp, LABEL y)
{
  /* Writes label y to file handle fp. */
}

void free_pattern(PATTERN x) {
  /* Frees the memory of x. */
  for (int i = 0; i < x.sentenceLen; i++)
    free(x.frames[i]);
}

void free_label(LABEL y) {
  /* Frees the memory of y. */
  free(y.labels);
}

void free_struct_model(STRUCTMODEL sm)
{
  /* Frees the memory of model. */
  /* if(sm.w) free(sm.w); */ /* this is free'd in free_model */
  if(sm.svm_model) free_model(sm.svm_model,1);
  /* add free calls for user defined data here */
}

void free_struct_sample(SAMPLE s)
{
  /* Frees the memory of sample s. */
  int i;
  for(i=0;i<s.n;i++) {
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
  }
  free(s.examples);
}

void print_struct_help()
{
  printf("         --* string  -> custom parameters that can be adapted for struct\n");
  printf("                        learning. The * can be replaced by any character\n");
  printf("                        and there can be multiple options starting with --.\n");
}

void parse_struct_parameters(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- */
  int i;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2])
      {
      case 'a': i++; /* strcpy(learn_parm->alphafile,argv[i]); */ break;
      case 'e': i++; /* sparm->epsilon=atof(sparm->custom_argv[i]); */ break;
      case 'k': i++; /* sparm->newconstretrain=atol(sparm->custom_argv[i]); */ break;
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}

void print_struct_help_classify()
{
  printf("         --* string -> custom parameters that can be adapted for struct\n");
  printf("                       learning. The * can be replaced by any character\n");
  printf("                       and there can be multiple options starting with --.\n");
}

void parse_struct_parameters_classify(STRUCT_LEARN_PARM *sparm)
{
  int i;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2])
      {
      /* case 'x': i++; strcpy(xvalue,sparm->custom_argv[i]); break; */
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}

