/*

   Copyright (c) 2015, Song Gao <song@gao.io>
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name of flags.cc nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

/* GitHub: https://github.com/songgao/flags.hh */

#ifndef FLAGS_CUH
#define FLAGS_CUH

#include <getopt.h>

#include <string>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <sstream>


class Flags {
public:
Flags() : autoId(256) {
}

template <typename T>
void Var(T & var, char shortFlag, std::string longFlag, T defaultValue, std::string description, std::string descriptionGroup = "");

void Bool(bool & var, char shortFlag, std::string longFlag, std::string description, std::string descriptionGroup = "");

bool Parse(int argc, char ** argv);
void PrintHelp(std::ostream & to = std::cout);

private:
int autoId;
std::map<int, std::function< void(std::string optarg) > > setters;     // flag id -> setters
std::set<std::string> longFlags;
std::map<std::string, std::vector<std::string> > help;     // group -> help itmes
std::vector<struct option> options;
std::string optionStr;
std::string argv0;

template <typename T>
void set(T & var, std::string optarg);

template <typename T>
void entry(struct option & op, char shortFlag, std::string longFlag, T & defaultValue, std::string description, std::string descriptionGroup);
};




template <typename T>
inline void Flags::entry(struct option & op, char shortFlag, std::string longFlag, T & defaultValue, std::string description, std::string descriptionGroup) {
        if (!shortFlag && !longFlag.size()) {
                throw std::string("no flag specified");
        }

        if (shortFlag) {
                if (this->setters.find((int)(shortFlag)) != this->setters.end()) {
                        throw std::string("short flag exists: ") + shortFlag;
                }
                this->optionStr += shortFlag;
                op.val = shortFlag;
        } else {
                op.val = this->autoId++;
        }
        if (longFlag.size()) {
                if (this->longFlags.find(longFlag) != this->longFlags.end()) {
                        throw std::string("long flag exists: ") + longFlag;
                }
                op.name = this->longFlags.insert(longFlag).first->c_str();
        }

        op.flag = NULL;

        // generate help item
        std::stringstream ss;
        ss << "  ";
        if (shortFlag) {
                ss << "-" << shortFlag << " ";
        }
        if (longFlag.size()) {
                ss << "--" << longFlag << " ";
        }
        ss << "[default: " << defaultValue << "]";
        ss << std::endl;
        constexpr size_t step = 80 - 6;
        for (size_t i = 0; i < description.size(); i += step) {
                ss << "      ";
                if (i + step < description.size()) {
                        ss << description.substr(i, step) << std::endl;
                } else {
                        ss << description.substr(i);
                }
        }
        this->help[descriptionGroup].push_back(ss.str());
}

template <typename T>
inline void Flags::Var(T & var, char shortFlag, std::string longFlag, T defaultValue, std::string description, std::string descriptionGroup) {
        struct option op;

        this->entry(op, shortFlag, longFlag, defaultValue, description, descriptionGroup);

        this->optionStr += ":";
        op.has_arg = required_argument;
        var = defaultValue;

        this->setters[op.val] = std::bind(&Flags::set<T>, this, std::ref(var), std::placeholders::_1);

        this->options.push_back(op);
}

inline void Flags::Bool(bool & var, char shortFlag, std::string longFlag, std::string description, std::string descriptionGroup) {
        struct option op;
        this->entry(op, shortFlag, longFlag, "(unset)", description, descriptionGroup);

        op.has_arg = no_argument;
        var = false;

        this->setters[op.val] = [&var](std::string) {
                                        var = true;
                                };

        this->options.push_back(op);
}

inline bool Flags::Parse(int argc, char ** argv) {
        this->argv0 = std::string(argv[0]);
        this->options.push_back({NULL, 0, NULL, 0});
        int ch;
        while ((ch = getopt_long(argc, argv, this->optionStr.c_str(), &this->options[0], NULL)) != -1) {
                auto it = this->setters.find(ch);
                if (it != this->setters.end()) {
                        if (optarg) {
                                it->second(optarg);
                        } else {
                                it->second("");
                        }
                } else {
                        return false;
                }
        }
        return true;
}

inline void Flags::PrintHelp(std::ostream & to) {
        to << "Usage: " << this->argv0 << " [options]" << std::endl <<std::endl;
        for (auto& it : this->help) {
                if (it.first.size()) {
                        to << it.first << ":" << std::endl;
                }
                for (auto& h : it.second) {
                        to << h << std::endl;
                }
                to << std::endl;
        }
}

template <typename T>
inline void Flags::set(T & var, std::string optarg) {
        std::stringstream ss(optarg);
        ss >> var;
}
template <>
inline void Flags::set<std::string>(std::string & var, std::string optarg) {
        var = optarg;
}

#endif
