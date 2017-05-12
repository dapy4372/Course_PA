// **************************************************************************
// File       [ test_cmd.h ]
// Author     [ littleshamoo ]
// Synopsis   [ ]
// Date       [ 2012/04/10 created ]
// **************************************************************************

#ifndef _TEST_CMD_H_
#define _TEST_CMD_H_

#include "graph.h"
#include "../lib/cmd.h"

bool stoi(const string&, int&);

class TestCmd : public CommonNs::Cmd {
public:
         TestCmd(const char * const name);
         ~TestCmd();

    bool exec(int argc, char **argv);
};

class Read_graphCmd : public CommonNs::Cmd{
	public:
		//friend class Write_tree_bfsCmd;

	    Read_graphCmd(const char * const name);
		~Read_graphCmd();
	    bool exec(int argc, char **argv);
			
};

class Write_tree_bfsCmd : public CommonNs::Cmd{
	public:
	    Write_tree_bfsCmd(const char * const name);
		~Write_tree_bfsCmd();
	    bool exec(int argc, char **argv);
	private:
		int _srcNode;
};

class Write_tree_dfsCmd : public CommonNs::Cmd{
	public:
	    Write_tree_dfsCmd(const char * const name);
		~Write_tree_dfsCmd();
	    bool exec(int argc, char **argv);
		void dfs_visit( Node*, fstream& );
	private:
		int _srcNode;
		int time;
		int vNum;
		int eNum;
};

class Color_graphCmd : public CommonNs::Cmd{
    public:
		Color_graphCmd(const char * const name);
        ~Color_graphCmd();
		bool exec(int argc, char **argv);
	private:
		int _srcNode;
		int vNum;
		int eNum;
};

#endif
