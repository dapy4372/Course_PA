# include <iostream>
# include <stdio.h>
# include <string.h>
# include <stdlib.h>
# include <vector>
# include <fstream>
# include <string>
# include <sstream>
# include <algorithm>
# include <deque>
# include <math.h>
# define BUF_SIZE 128
# define LABEL_NUM 48 
# define NEG_INF -1000.0
using namespace std;

typedef struct
{
  double phone[LABEL_NUM]; 
  double trans[LABEL_NUM][LABEL_NUM];
}CountProb;

typedef struct
{
  double feat[LABEL_NUM];
  string sentid;
  int order;
}Frame;

typedef vector<Frame> Sentence;
typedef vector<Sentence> VSent;
typedef deque<int> Sequence;  //decoded by viterbi

void split(char **, char *, char *);
void load_count_prob(char *, CountProb &);
void load_dnn_prob(char *, VSent &);
Sequence viterbi(const Sentence &, const CountProb &);
void write_result(const Sequence &, const Sentence &, ofstream &);

int main(int argc, char** argv)
{
  if( argc != 4){
    cout << "Error! Usage: ./hmm [count prob file] [dnn prob file] [output file]\n"; 
    return 0;
  }

  char *count_prob_filename = argv[1];
  char *dnn_prob_filename = argv[2];
  char *output_filename = argv[3];

  CountProb prob;
  VSent vsent;

  load_count_prob(count_prob_filename, prob);
  load_dnn_prob(dnn_prob_filename, vsent);

  //deque<Sequence> result;
  char buf[BUF_SIZE];
  ofstream fout(output_filename);
  for(int i = 0; i < vsent.size(); ++i){
    Sequence tmp_seq;
    tmp_seq = viterbi(vsent[i], prob);
    write_result(tmp_seq, vsent[i], fout);
    //result.push_back(tmp_seq);
  }
  fout.close();
  return 0;
}

void err_sys(char *x)
{
  perror(x);
  exit(1);
}

Sequence viterbi(const Sentence &sent, const CountProb &cp)
{
  int sent_len = sent.size();
  double score[sent_len][LABEL_NUM];
  int backtrack[sent_len][LABEL_NUM];
  
  // initial
  for(int y = 0; y < LABEL_NUM; ++y){
    score[0][y] = cp.phone[y] + sent[0].feat[y];
    backtrack[0][y] = y;
  }
  // forward pass
  for(int x = 1; x < sent_len; ++x){
    for(int y = 0; y < LABEL_NUM; ++y){
      int best_label = 0;
      double best_score = -10000000000000000;
      for(int i = 0; i < LABEL_NUM; ++i){
        double tmp = score[x-1][i] + cp.trans[i][y];
        if(best_score < tmp){
          best_score = tmp;
          //backtrack[x][y] = backtrack[x-1][i];
          backtrack[x][y] = i;
        }
      }
      score[x][y] = best_score + sent[x].feat[y];
    }
  }
  // backward track
  // find last column max
  double last_col_score_max = score[sent_len-1][0];
  int last_col_label_max = 0;
  Sequence ret;  
  for(int i = 1; i < LABEL_NUM; ++i){
    if(score[sent_len-1][i] > last_col_score_max){
      last_col_score_max = score[sent_len-1][i];
      last_col_label_max = i;
    }
  }
  ret.push_front(last_col_label_max);
  int prev_label = backtrack[sent_len-1][last_col_label_max];
  
  // find path
  for(int i = sent_len-2; i > -1; --i){
    ret.push_front(backtrack[i][prev_label]);
    prev_label = backtrack[i][prev_label];
  }
  return ret;
}

void load_count_prob(char *filename, CountProb &prob)
{
  ifstream fin;
  fin.open(filename);

  stringstream ss;
  string buf;
  int i;
  for(i = 0; i < LABEL_NUM; ++i){
    double tmp;
    getline(fin, buf);
    ss << buf;
    ss >> tmp;
    if(tmp == 0)
      prob.phone[i] = NEG_INF;
    else
      prob.phone[i] = log(tmp);
    ss.str("");
    ss.clear();
  }
  int prev_state = 0;
  bool first = true;
  for(; i < LABEL_NUM * LABEL_NUM + LABEL_NUM; ++i){
    getline(fin, buf);
    ss << buf;
    int next_state = ((i - LABEL_NUM) % LABEL_NUM);
    if( !first && next_state == 0)
      ++prev_state;
    double tmp;
    ss >> tmp;
    if(tmp == 0)
      prob.trans[prev_state][next_state] = NEG_INF;
    else
      prob.trans[prev_state][next_state] = log(tmp);
    ss.str("");
    ss.clear();
    if(first)
      first = false;
  }
  fin.close();
}

void load_dnn_prob(char *prob_filename, VSent &vsent)
{
  ifstream fin1;
  fin1.open(prob_filename);

  stringstream ss;
  string buf1;
  Sentence tmp_sent;

  string prev_sentid = "";
  bool first = true;
  while( getline(fin1, buf1)){
    Frame tmp_frame;
    string tmp_sentid;
    int tmp_order;
    double tmp_feat;
    // get prob
    ss << buf1;
    ss >> tmp_sentid >> tmp_order;
    for(int i = 0; i < LABEL_NUM; ++i){
      ss >> tmp_feat;
      if(tmp_feat == 0)
        tmp_frame.feat[i] = NEG_INF;
      else
        tmp_frame.feat[i] = log(tmp_feat);
    }

    tmp_frame.sentid = tmp_sentid;
    tmp_frame.order = tmp_order;
    // next sentence
    if(!first && prev_sentid != tmp_sentid){
      vsent.push_back(tmp_sent);
      tmp_sent.clear();
    }
    tmp_sent.push_back(tmp_frame);
    prev_sentid = tmp_sentid;
    ss.str("");
    ss.clear();
    if(first)
      first = false;
  }
  vsent.push_back(tmp_sent);

  fin1.close(); 
}

void write_result(const Sequence &seq, const Sentence &sent, ofstream &fout)
{
  // TODO filename
  string buf;
  for(int i = 0; i < seq.size(); ++i){
    string str_label, str_order;
    stringstream ss;
    ss << sent[i].order;
    ss >> str_order;
    ss.clear(); ss.str("");
    ss << seq[i];
    ss >> str_label;
    ss.clear(); ss.str("");
    buf = sent[i].sentid + '_' + str_order + ',' + str_label + '\n';
 //   snprintf(buf, BUF_SIZE, "%s_%s,%d\n", sent[i].sentid, sent[i].order, seq[i])
    fout << buf;
  }
}
