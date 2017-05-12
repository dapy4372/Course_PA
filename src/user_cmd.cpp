// **************************************************************************
// File       [ test_cmd.cpp ]
// Author     [ littleshamoo ]
// Synopsis   [ ]
// Date       [ 2012/04/10 created ]
// **************************************************************************

#include "user_cmd.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <queue>
#include <sstream>
#include "../lib/tm_usage.h"
using namespace std;
using namespace CommonNs;
char G[30];
extern Graph *graph;

TestCmd::TestCmd(const char * const name) : Cmd(name) {
	optMgr_.setShortDes("test");
	optMgr_.setDes("test");

	Opt *opt = new Opt(Opt::BOOL, "print usage", "");
	opt->addFlag("h");
	opt->addFlag("help");
	optMgr_.regOpt(opt);

	opt = new Opt(Opt::STR_REQ, "print the string of -s", "[string]");
	opt->addFlag("s");
	optMgr_.regOpt(opt);
}

TestCmd::~TestCmd() {}

bool TestCmd::exec(int argc, char **argv) {
	optMgr_.parse(argc, argv);

	if (optMgr_.getParsedOpt("h")) {
		optMgr_.usage();
		return true;
	}

	if (optMgr_.getParsedOpt("s")) {
		printf("%s\n", optMgr_.getParsedValue("s"));
	}
	else
		printf("hello world !!\n");


	return true;
}

Read_graphCmd::Read_graphCmd(const char * const name) : Cmd(name) {
	optMgr_.setShortDes("Read the graph");
	optMgr_.setDes("Read the graph in dot format");

	Opt *opt = new Opt(Opt::BOOL, "print usage", "");
	opt->addFlag("h");
	opt->addFlag("help");
	optMgr_.regOpt(opt);

}

Read_graphCmd::~Read_graphCmd() {}

bool Read_graphCmd::exec(int argc, char **argv) {
	optMgr_.parse(argc, argv);

	if (optMgr_.getParsedOpt("h")) {
		optMgr_.usage();
		return true;
	}

	ifstream fin( argv[1] );
	//static Graph *graph;
	string buf;
	string garbage;
	string graph_name;
	char ch;
	int id[2];
    stringstream ss("");
	while ( getline( fin, buf ) ){

		switch( buf[0] ){
			case '/':
				break;
			case '}':
			    break;
			case 'g':
			    ss << buf;
			    ss >> garbage >> graph_name >> garbage;
			    if( graph != NULL )
			    	delete graph;
			    graph = new Graph( graph_name );
				break;
			case 'v':
			    ss << buf;
			    ss >> ch >> id[0] >> garbage >> ch >> id[1] >> ch;
				graph -> addEdge( id[0], id[1] );
				break;
		}
		ss.clear();
	}
	return true;
}

Write_tree_bfsCmd::Write_tree_bfsCmd(const char * const name) : Cmd(name) {
	optMgr_.setShortDes("Perform BFS");
	optMgr_.setDes("Perform breadth first search starting from source node.  Then write to a dot file.");

	Opt *opt = new Opt(Opt::BOOL, "print usage", "");
	opt->addFlag("h");
	opt->addFlag("help");
	optMgr_.regOpt(opt);

	opt = new Opt(Opt::STR_REQ, "source node", "e.g. v0,v1,v3 ...");
	opt->addFlag("s");
	optMgr_.regOpt(opt);

	opt = new Opt(Opt::STR_REQ, "output file. Default is <input>.<format>", "OUTPUT");
	opt->addFlag("o");
	optMgr_.regOpt(opt);

}

Write_tree_bfsCmd::~Write_tree_bfsCmd() {}

bool Write_tree_bfsCmd::exec(int argc, char **argv) {
	optMgr_.parse(argc, argv);

 	if (optMgr_.getParsedOpt("h")) {
		optMgr_.usage();
		return true;
	}

	CommonNs::TmUsage tmusg;
	CommonNs::TmStat stat;

	tmusg.periodStart();
	string tmp = optMgr_.getParsedValue("s");
	tmp.erase(0, 1);
	stoi(tmp, _srcNode);
	
	graph->init();
	graph->sortEdgesOfNode();

	fstream fout( argv[4], ios::out );

	fout<<"// BFS tree produced by graphlab"<<endl;
	fout<<"graph "<<graph->name<<"_bfs {"<<endl;

	Node *srcNode = graph->getNodeById( _srcNode );
	//initial source node
	srcNode->traveled = true;
	srcNode->d = 0;
	srcNode->prev = 0;
	queue<int> nodeQueue;
	nodeQueue.push(srcNode->id);	
	int vNum = 1;
	int eNum = 0;
	while( !nodeQueue.empty() ){
		int u = nodeQueue.front();
		nodeQueue.pop();
		Node *uNode = graph->getNodeById( u );

		for( int i = 0; i < uNode->edge.size(); ++i ){
			//find v
			Node *vNode;
			vNode = uNode->edge[i]->getNeighbor( uNode );

			if( !vNode->traveled ){
				++vNum;
				++eNum;
				fout<<"v"<<u<<" -- "<<"v"<<vNode->id<<endl;
				vNode->traveled = true;
				vNode->d = uNode->d + 1;
				vNode->prev = uNode;
				nodeQueue.push(vNode->id);
			}
		}
	}
	tmusg.getPeriodUsage(stat);
	fout << "}" << endl;
	fout << "// vertices = " << vNum << endl;
	fout << "// edges = " << eNum << endl;
	fout << "// runtime = " <<(stat.uTime + stat.sTime) / 1000000. << " sec" << endl;
	fout << "// memory = " << stat.vmPeak / 1000.0 << " MB";
	fout.close();

	return true;

}

Write_tree_dfsCmd::Write_tree_dfsCmd(const char * const name) : Cmd(name) {
	optMgr_.setShortDes("Perform DFS");
	optMgr_.setDes("Perform depth first search starting from source node.  Then write to a dot file.");

	Opt *opt = new Opt(Opt::BOOL, "print usage", "");
	opt->addFlag("h");
	opt->addFlag("help");
	optMgr_.regOpt(opt);

	opt = new Opt(Opt::STR_REQ, "source node", "e.g. v0, v1 ...");
	opt->addFlag("s");
	optMgr_.regOpt(opt);

	opt = new Opt(Opt::STR_REQ, "output file. Default is <input>.dot", "OUTPUT");
	opt->addFlag("o");
	optMgr_.regOpt(opt);
}

Write_tree_dfsCmd::~Write_tree_dfsCmd() {}

bool Write_tree_dfsCmd::exec(int argc, char **argv) { 
	optMgr_.parse(argc, argv);

	if (optMgr_.getParsedOpt("h")) { 
		optMgr_.usage();
		return true;
	}

	CommonNs::TmUsage tmusg;
	CommonNs::TmStat stat;

	tmusg.periodStart();

	string tmp = optMgr_.getParsedValue("s");
	tmp.erase( 0, 1 );
	stoi( tmp, _srcNode );

    string filename = optMgr_.getParsedValue("o");

	graph->init();
	graph->sortEdgesOfNode();
	fstream fout( filename.c_str(), ios::out );
	
	fout<<"// DFS tree produced by graphlab"<<endl;
	fout<<"graph "<<graph->name<<"_dfs {"<<endl;

	Node *u = graph->getNodeById( _srcNode );
	u->traveled = true;
	u->prev = 0;

	time = 0;
	vNum = 1;
	eNum = 0;
	dfs_visit( u, fout );
  /*
  for( int i = 0; i < graph->nodes.size(); ++i ){
		u = graph->nodes[i];
		if( !u->traveled )
			dfs_visit( u, fout );
	}
*/
	tmusg.getPeriodUsage(stat);
	fout << "}" << endl;
	fout << "// vertices = " << vNum << endl;
	fout << "// edges = " << eNum << endl;
	fout << "// runtime = " <<(stat.uTime + stat.sTime) / 1000000. << " sec" << endl;
	fout << "// memory = " << stat.vmPeak / 1000.0 << " MB";
	fout.close();

	return true;

}

void Write_tree_dfsCmd::dfs_visit( Node *uNode, fstream& fout ){
	//++time;
	//uNode->d = time;
	uNode->traveled = true;

	for( int i = 0; i < uNode->edge.size(); ++i ){
		//find v
		Node *vNode;
		vNode = uNode->edge[i]->getNeighbor(uNode);

		if( !vNode->traveled ){
			++vNum;
			++eNum;
			vNode->traveled = true;
			vNode->prev = uNode;
			fout<<"v"<<uNode->id<<" -- "
				<<"v"<<vNode->id<<endl;
			dfs_visit(vNode, fout);
		}
//		++time;
//		uNode->f=time;
	}
}

Color_graphCmd::Color_graphCmd(const char * const name) : Cmd(name) {
	optMgr_.setShortDes("color the graph by greedy");
	optMgr_.setDes("Perform greedy algorithm to color the graph. Then write to a dot file.");

	Opt *opt = new Opt(Opt::BOOL, "print usage", "");
	opt->addFlag("h");
	opt->addFlag("help");
	optMgr_.regOpt(opt);

	opt = new Opt(Opt::STR_REQ, "-m greedy", "greedy");
	opt->addFlag("m");
	optMgr_.regOpt(opt);

	opt = new Opt(Opt::STR_REQ, "output file. Default is <input>.<format>", "OUTPUT");
	opt->addFlag("o");
	optMgr_.regOpt(opt);

}

Color_graphCmd::~Color_graphCmd() {}

bool Color_graphCmd::exec(int argc, char **argv) {
	optMgr_.parse(argc, argv);

	if (optMgr_.getParsedOpt("h")) {
		optMgr_.usage();
		return true;
	}

	CommonNs::TmUsage tmusg;
	CommonNs::TmStat stat;

	tmusg.periodStart();
	//string tmp = optMgr_.getParsedValue("m");
	string filename = optMgr_.getParsedValue("o");

	graph->init();
	graph->sortEdgesOfNode();
	graph->sortNodesByDegree();

	fstream fout;
	fout.open( filename.c_str(), ios::out );
	fout<<"// coloring produced by graphlab"<<endl;
	fout<<"graph "<<graph->name<<"_color {"<<endl;

	vNum = graph->nodes.size();
	eNum = graph->edges.size();
	for( int i = 0; i<eNum; ++i ){
		fout << "v" << graph->edges[i]->node[0]->id << " -- "
			<< "v" << graph->edges[i]->node[1]->id <<";"<<endl;
	}

	int numClr = 1;
	graph->nodes[0]->color = 1;

	for( int i = 1; i < graph->nodes.size(); ++i ){
		Node *tmpnode = graph->nodes[i];
		bool *tagClr = new bool[numClr];
		memset(tagClr, 0, numClr);
		//find proper color to the node
		for( int j = 0; j < tmpnode->edge.size(); ++j ){
			int &adjclr =  tmpnode->edge[j]->getNeighbor(tmpnode)->color; 
			if(adjclr != -1){
				tagClr[adjclr-1] = 1;
			}
		}

		for( int i = 0; i<numClr; ++i){
			if( tagClr[i] == 0 ){
				tmpnode->color = i+1;
				break;
			}
		}

		if( tmpnode->color == -1 ){
			++numClr;
			tmpnode->color = numClr;
		}
		delete [] tagClr;
	}


	graph->sortNodesByID();
	for(int i=0; i<graph->nodes.size();++i)
		fout << "v" << i << " [label = \"v" << i << "_color"
			<< graph->nodes[i]->color << "\"];" << endl;

	tmusg.getPeriodUsage(stat);

	fout << "}" << endl;
	fout << "// vertices = " << vNum << endl;
	fout << "// edges = " << eNum << endl;
	fout << "// number_of_colors = " << numClr << endl;
	fout << "// runtime = " <<(stat.uTime + stat.sTime) / 1000000. << " sec" << endl;
	fout << "// memory = " << stat.vmPeak / 1000.0 << " MB";
	fout.close();

	return true;

}

bool stoi(const string& str, int& num)
{
	num = 0;
	size_t i = 0;
	int sign = 1;
	if (str[0] == '-') { sign = -1; i = 1; }
	bool valid = false;
	for (; i < str.size(); ++i) {
		if (isdigit(str[i])) {
			num *= 10;
			num += int(str[i] - '0');
			valid = true;
		}
		else return false;
	}
	num *= sign;
	return valid;
}
