// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifdef ASTERION_USE_AVX2
#include <immintrin.h>
#endif

namespace asterion {

constexpr int White = 0, Black = 1, Empty = 0;
constexpr int Pawn = 1, Knight = 2, Bishop = 3, Rook = 4, Queen = 5, King = 6;
constexpr int Infinity = 32000, Mate = 30000;
constexpr int WK = 1, WQ = 2, BK = 4, BQ = 8;

int color(int piece) { return piece > 0 ? White : Black; }
int type(int piece) { return piece < 0 ? -piece : piece; }
bool onBoard(int sq) { return sq >= 0 && sq < 64; }
int rankOf(int sq) { return sq >> 3; }
int fileOf(int sq) { return sq & 7; }
int square(int file, int rank) { return rank * 8 + file; }
int popcount(std::uint64_t x) { int n = 0; while (x) { x &= x - 1; ++n; } return n; }

std::string squareName(int sq) {
    std::string out = "a1";
    out[0] = static_cast<char>('a' + fileOf(sq));
    out[1] = static_cast<char>('1' + rankOf(sq));
    return out;
}

int parseSquare(std::string_view s) {
    if (s.size() != 2 || s[0] < 'a' || s[0] > 'h' || s[1] < '1' || s[1] > '8') return -1;
    return square(s[0] - 'a', s[1] - '1');
}

struct Move {
    std::uint8_t from = 0, to = 0, promotion = 0;
    std::uint8_t flags = 0;
    enum : std::uint8_t { Capture = 1, EnPassant = 2, Castle = 4, PawnStart = 8 };
    std::string uci() const {
        std::string text = squareName(from) + squareName(to);
        if (promotion) text += " nbrq"[promotion];
        return text;
    }
};

struct Position {
    std::array<std::int8_t, 64> board{};
    int side = White;
    int castling = WK | WQ | BK | BQ;
    int ep = -1;
    int halfmove = 0;
    int fullmove = 1;

    static Position start();
    static std::optional<Position> fromFen(std::string_view fen);
    std::string fen() const;
    std::uint64_t hash() const;
    bool attacked(int target, int by) const;
    bool inCheck(int who) const;
    void make(const Move& move);
    std::vector<Move> pseudoMoves(bool capturesOnly = false) const;
    std::vector<Move> legalMoves(bool capturesOnly = false) const;
};

Position Position::start() {
    auto p = fromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    return *p;
}

std::optional<Position> Position::fromFen(std::string_view fenText) {
    std::istringstream in{std::string(fenText)};
    std::string placement, turn, rights, epText;
    Position p;
    if (!(in >> placement >> turn >> rights >> epText >> p.halfmove >> p.fullmove)) return std::nullopt;
    int rank = 7, file = 0;
    for (char c : placement) {
        if (c == '/') { if (file != 8 || --rank < 0) return std::nullopt; file = 0; continue; }
        if (c >= '1' && c <= '8') { file += c - '0'; if (file > 8) return std::nullopt; continue; }
        const std::string pieces = "PNBRQKpnbrqk";
        const auto index = pieces.find(c);
        if (index == std::string::npos || file >= 8) return std::nullopt;
        int pc = static_cast<int>(index % 6) + 1;
        p.board[square(file++, rank)] = index < 6 ? pc : -pc;
    }
    if (rank != 0 || file != 8 || (turn != "w" && turn != "b")) return std::nullopt;
    p.side = turn == "w" ? White : Black;
    p.castling = 0;
    if (rights != "-") for (char c : rights) {
        if (c == 'K') p.castling |= WK; else if (c == 'Q') p.castling |= WQ;
        else if (c == 'k') p.castling |= BK; else if (c == 'q') p.castling |= BQ; else return std::nullopt;
    }
    p.ep = epText == "-" ? -1 : parseSquare(epText);
    if (p.ep == -1 && epText != "-") return std::nullopt;
    return p;
}

std::string Position::fen() const {
    std::ostringstream out;
    for (int r = 7; r >= 0; --r) {
        int blanks = 0;
        for (int f = 0; f < 8; ++f) {
            int pc = board[square(f, r)];
            if (!pc) { ++blanks; continue; }
            if (blanks) { out << blanks; blanks = 0; }
            const char* names = " PNBRQK";
            char c = names[type(pc)]; out << (pc > 0 ? c : static_cast<char>(std::tolower(c)));
        }
        if (blanks) out << blanks;
        if (r) out << '/';
    }
    out << (side == White ? " w " : " b ");
    if (!castling) out << '-';
    else { if (castling & WK) out << 'K'; if (castling & WQ) out << 'Q'; if (castling & BK) out << 'k'; if (castling & BQ) out << 'q'; }
    out << ' ' << (ep == -1 ? "-" : squareName(ep)) << ' ' << halfmove << ' ' << fullmove;
    return out.str();
}

std::uint64_t Position::hash() const {
    std::uint64_t h = 1469598103934665603ULL;
    for (int pc : board) { h ^= static_cast<std::uint8_t>(pc + 7); h *= 1099511628211ULL; }
    h ^= static_cast<std::uint64_t>(side + (castling << 1) + ((ep + 1) << 5));
    return h;
}

bool Position::attacked(int target, int by) const {
    const int pawnSourceRank = rankOf(target) + (by == White ? -1 : 1);
    for (int df : {-1, 1}) { int f = fileOf(target) + df; if (f >= 0 && f < 8 && pawnSourceRank >= 0 && pawnSourceRank < 8 && board[square(f, pawnSourceRank)] == (by == White ? Pawn : -Pawn)) return true; }
    static constexpr int knight[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
    static constexpr int king[8][2] = {{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1},{0,-1},{1,-1}};
    for (auto [df, dr] : knight) { int f=fileOf(target)+df,r=rankOf(target)+dr; if(f>=0&&f<8&&r>=0&&r<8&&board[square(f,r)]==(by==White?Knight:-Knight)) return true; }
    for (auto [df, dr] : king) { int f=fileOf(target)+df,r=rankOf(target)+dr; if(f>=0&&f<8&&r>=0&&r<8&&board[square(f,r)]==(by==White?King:-King)) return true; }
    auto ray = [&](int df, int dr, int a, int b) { int f=fileOf(target)+df,r=rankOf(target)+dr; while(f>=0&&f<8&&r>=0&&r<8) { int pc=board[square(f,r)]; if(pc) return pc==(by==White?a:-a)||pc==(by==White?b:-b); f+=df;r+=dr; } return false; };
    return ray(1,0,Rook,Queen)||ray(-1,0,Rook,Queen)||ray(0,1,Rook,Queen)||ray(0,-1,Rook,Queen)||ray(1,1,Bishop,Queen)||ray(-1,1,Bishop,Queen)||ray(1,-1,Bishop,Queen)||ray(-1,-1,Bishop,Queen);
}

bool Position::inCheck(int who) const {
    const int king = who == White ? King : -King;
    for (int sq=0; sq<64; ++sq) if (board[sq] == king) return attacked(sq, who ^ 1);
    return true;
}

void Position::make(const Move& m) {
    const int pc = board[m.from]; const int enemy = side ^ 1;
    const int captured = board[m.to];
    halfmove = (type(pc) == Pawn || captured || (m.flags & Move::EnPassant)) ? 0 : halfmove + 1;
    ep = -1;
    board[m.to] = board[m.from]; board[m.from] = Empty;
    if (m.flags & Move::EnPassant) board[m.to + (side == White ? -8 : 8)] = Empty;
    if (m.promotion) board[m.to] = side == White ? m.promotion : -m.promotion;
    if (m.flags & Move::Castle) {
        if (m.to == square(6,0)) { board[square(5,0)] = Rook; board[square(7,0)] = Empty; }
        if (m.to == square(2,0)) { board[square(3,0)] = Rook; board[square(0,0)] = Empty; }
        if (m.to == square(6,7)) { board[square(5,7)] = -Rook; board[square(7,7)] = Empty; }
        if (m.to == square(2,7)) { board[square(3,7)] = -Rook; board[square(0,7)] = Empty; }
    }
    if (type(pc) == King) castling &= side == White ? ~(WK|WQ) : ~(BK|BQ);
    if (m.from == square(0,0) || m.to == square(0,0)) castling &= ~WQ;
    if (m.from == square(7,0) || m.to == square(7,0)) castling &= ~WK;
    if (m.from == square(0,7) || m.to == square(0,7)) castling &= ~BQ;
    if (m.from == square(7,7) || m.to == square(7,7)) castling &= ~BK;
    if (m.flags & Move::PawnStart) ep = m.from + (side == White ? 8 : -8);
    if (side == Black) ++fullmove;
    side = enemy;
}

std::vector<Move> Position::pseudoMoves(bool capturesOnly) const {
    std::vector<Move> moves; moves.reserve(64);
    const int us = side, forward = us == White ? 1 : -1;
    auto add = [&](int from, int to, int promo=0, int flags=0) {
        if (!onBoard(to)) return;
        int target = board[to];
        if ((target && color(target)==us) || type(target)==King) return;
        if (target) flags |= Move::Capture;
        if (capturesOnly && !(flags & (Move::Capture | Move::EnPassant))) return;
        moves.push_back({static_cast<std::uint8_t>(from),static_cast<std::uint8_t>(to),static_cast<std::uint8_t>(promo),static_cast<std::uint8_t>(flags)});
    };
    static constexpr int knight[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
    static constexpr int king[8][2] = {{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1},{0,-1},{1,-1}};
    for (int from=0; from<64; ++from) {
        int pc=board[from]; if (!pc || color(pc)!=us) continue;
        int f=fileOf(from), r=rankOf(from), pt=type(pc);
        if (pt == Pawn) {
            int one=square(f,r+forward), last=us==White?7:0, home=us==White?1:6;
            if (!capturesOnly && r+forward>=0 && r+forward<8 && !board[one]) { if(r+forward==last) for(int pr:{Queen,Rook,Bishop,Knight}) add(from,one,pr); else add(from,one); int two=square(f,r+2*forward); if(r==home&&!board[two]) add(from,two,0,Move::PawnStart); }
            for(int df:{-1,1}) { int nf=f+df,nr=r+forward; if(nf<0||nf>7||nr<0||nr>7) continue; int to=square(nf,nr); if(board[to] && color(board[to])!=us) { if(nr==last) for(int pr:{Queen,Rook,Bishop,Knight}) add(from,to,pr); else add(from,to); } else if(to==ep) add(from,to,0,Move::EnPassant|Move::Capture); }
        } else if (pt == Knight || pt == King) {
            const auto& jumps = pt == Knight ? knight : king;
            for (auto [df,dr] : jumps) { int nf=f+df,nr=r+dr; if(nf>=0&&nf<8&&nr>=0&&nr<8) add(from,square(nf,nr)); }
            if (pt == King && !capturesOnly && !inCheck(us)) {
                if(us==White && from==square(4,0)) { if((castling&WK)&&board[square(7,0)]==Rook&&!board[square(5,0)]&&!board[square(6,0)]&&!attacked(square(5,0),Black)&&!attacked(square(6,0),Black)) add(from,square(6,0),0,Move::Castle); if((castling&WQ)&&board[square(0,0)]==Rook&&!board[square(1,0)]&&!board[square(2,0)]&&!board[square(3,0)]&&!attacked(square(3,0),Black)&&!attacked(square(2,0),Black)) add(from,square(2,0),0,Move::Castle); }
                if(us==Black && from==square(4,7)) { if((castling&BK)&&board[square(7,7)]==-Rook&&!board[square(5,7)]&&!board[square(6,7)]&&!attacked(square(5,7),White)&&!attacked(square(6,7),White)) add(from,square(6,7),0,Move::Castle); if((castling&BQ)&&board[square(0,7)]==-Rook&&!board[square(1,7)]&&!board[square(2,7)]&&!board[square(3,7)]&&!attacked(square(3,7),White)&&!attacked(square(2,7),White)) add(from,square(2,7),0,Move::Castle); }
            }
        } else {
            const bool diagonal = pt == Bishop || pt == Queen, straight = pt == Rook || pt == Queen;
            for (auto [df,dr] : {std::pair{1,0}, {-1,0}, {0,1}, {0,-1}, {1,1}, {-1,1}, {1,-1}, {-1,-1}}) {
                if ((df==0||dr==0) ? !straight : !diagonal) continue;
                for(int nf=f+df,nr=r+dr;nf>=0&&nf<8&&nr>=0&&nr<8;nf+=df,nr+=dr) { int to=square(nf,nr); if(board[to]) { add(from,to); break; } add(from,to); }
            }
        }
    }
    return moves;
}

std::vector<Move> Position::legalMoves(bool capturesOnly) const {
    auto pseudo = pseudoMoves(capturesOnly); std::vector<Move> legal; legal.reserve(pseudo.size());
    for(const Move& move : pseudo) { Position next=*this; next.make(move); if(!next.inCheck(side)) legal.push_back(move); }
    return legal;
}

struct Model {
    alignas(32) std::array<float, 16> weights{100, 320, 330, 500, 900, 0, -100, -320, -330, -500, -900, 0, 0, 0, 0, 0};
    std::array<float, 16> features(const Position& p) const {
        std::array<float,16> x{};
        for(int pc:p.board) if(pc) x[(pc>0?0:6) + type(pc)-1] += 1.0f;
        return x;
    }
    int evaluateWhite(const Position& p) const {
        const auto x=features(p); float sum=0;
#ifdef ASTERION_USE_AVX2
        __m256 a=_mm256_load_ps(weights.data()), b=_mm256_loadu_ps(x.data());
        __m256 c=_mm256_mul_ps(a,b); alignas(32) float v[8]; _mm256_store_ps(v,c);
        for(float n:v) sum+=n;
        a=_mm256_load_ps(weights.data()+8); b=_mm256_loadu_ps(x.data()+8); c=_mm256_mul_ps(a,b); _mm256_store_ps(v,c); for(float n:v) sum+=n;
#else
        for(int i=0;i<16;++i) sum += weights[i]*x[i];
#endif
        return static_cast<int>(std::lround(sum));
    }
    int evaluate(const Position& p) const { int score=evaluateWhite(p); return p.side==White?score:-score; }
};

struct TTEntry { std::uint64_t key=0; int score=0; std::int16_t depth=-1; std::uint8_t bound=0; Move best{}; };
class Search {
public:
    explicit Search(const Model& model, std::size_t megabytes=32) : model_(model), table_((megabytes*1024*1024)/sizeof(TTEntry)) {}
    Move findBest(const Position& root, int maxDepth, int timeMs=0, bool emitInfo=true) {
        nodes_=0; stopped_=false; deadline_=timeMs?std::chrono::steady_clock::now()+std::chrono::milliseconds(timeMs):std::chrono::steady_clock::time_point::max(); Move best{};
        for(int d=1;d<=maxDepth;++d) { int score=0; Move candidate=best; rootSearch(root,d,candidate,score); if(stopped_) break; best=candidate; if(emitInfo) std::cout<<"info depth "<<d<<" score cp "<<score<<" nodes "<<nodes_<<" pv "<<best.uci()<<"\n"; }
        return best;
    }
    std::uint64_t nodes() const { return nodes_; }
private:
    const Model& model_; std::vector<TTEntry> table_; std::uint64_t nodes_=0; bool stopped_=false; std::chrono::steady_clock::time_point deadline_;
    int scoreMove(const Position& p,const Move& m,const Move& hash) const { if(m.uci()==hash.uci()) return 1000000; int victim=p.board[m.to]; return (m.flags&Move::Capture?10000+10*type(victim)-type(p.board[m.from]):0); }
    int quiesce(const Position& p,int alpha,int beta) { if((++nodes_&2047)==0&&std::chrono::steady_clock::now()>=deadline_) { stopped_=true; return 0; } int stand=model_.evaluate(p); if(stand>=beta)return beta; if(stand>alpha)alpha=stand; auto moves=p.legalMoves(true); for(const auto&m:moves){Position n=p;n.make(m);int s=-quiesce(n,-beta,-alpha);if(stopped_)return 0;if(s>=beta)return beta;if(s>alpha)alpha=s;}return alpha; }
    int negamax(const Position& p,int depth,int alpha,int beta,int ply) { if((++nodes_&2047)==0&&std::chrono::steady_clock::now()>=deadline_) { stopped_=true; return 0; } if(p.halfmove>=100)return 0; if(depth<=0)return quiesce(p,alpha,beta); std::uint64_t key=p.hash(); auto& entry=table_[key%table_.size()]; Move hashMove{}; if(entry.key==key){hashMove=entry.best;if(entry.depth>=depth){if(entry.bound==0)return entry.score;if(entry.bound==1&&entry.score>=beta)return entry.score;if(entry.bound==2&&entry.score<=alpha)return entry.score;}} auto moves=p.legalMoves(); if(moves.empty()) return p.inCheck(p.side)?-Mate+ply:0; std::sort(moves.begin(),moves.end(),[&](const Move&a,const Move&b){return scoreMove(p,a,hashMove)>scoreMove(p,b,hashMove);}); int original=alpha; Move best{}; for(const auto&m:moves){Position n=p;n.make(m);int s=-negamax(n,depth-1,-beta,-alpha,ply+1);if(stopped_)return 0;if(s>alpha){alpha=s;best=m;if(alpha>=beta)break;}} entry={key,alpha,static_cast<std::int16_t>(depth),static_cast<std::uint8_t>(alpha>=beta?1:(alpha<=original?2:0)),best}; return alpha; }
    bool rootSearch(const Position& p,int depth,Move& best,int& score) { auto moves=p.legalMoves(); if(moves.empty()) return true; std::sort(moves.begin(),moves.end(),[&](const Move&a,const Move&b){return scoreMove(p,a,best)>scoreMove(p,b,best);}); int alpha=-Infinity,beta=Infinity; for(const auto&m:moves){Position n=p;n.make(m);int s=-negamax(n,depth-1,-beta,-alpha,1);if(stopped_)return false;if(s>alpha){alpha=s;best=m;}}score=alpha;return true; }
};

struct TrainingState {
    Model model; std::uint64_t completed=0,wins=0,losses=0,draws=0; double loss=0.0, rating=1200.0; std::mt19937_64 rng{0xA57E710ULL};
};
std::atomic_bool pauseRequested=false;
void onInterrupt(int) { pauseRequested=true; }

bool saveCheckpoint(const TrainingState& state,const std::filesystem::path& path) {
    if(path.empty()) return true;
    std::error_code ec; if(path.has_parent_path()) std::filesystem::create_directories(path.parent_path(),ec);
    const auto temp=path.string()+".tmp", backup=path.string()+".bak";
    std::ofstream out(temp,std::ios::trunc); if(!out) return false;
    out<<"ASTERION_CHECKPOINT 1\n"<<std::setprecision(9);
    for (float w : state.model.weights) out << w << ' ';
    out << "\n";
    out<<state.completed<<' '<<state.wins<<' '<<state.losses<<' '<<state.draws<<' '<<state.loss<<' '<<state.rating<<"\n"<<state.rng<<"\n";
    out.flush(); if(!out) return false; out.close();
    if(std::filesystem::exists(path)) { std::filesystem::copy_file(path,backup,std::filesystem::copy_options::overwrite_existing,ec); }
    std::filesystem::rename(temp,path,ec); if(ec) { std::filesystem::remove(path,ec); ec.clear(); std::filesystem::rename(temp,path,ec); } return !ec;
}
std::optional<TrainingState> loadCheckpoint(const std::filesystem::path& path) {
    std::ifstream in(path); TrainingState s; std::string magic; int version=0; if(!(in>>magic>>version)||magic!="ASTERION_CHECKPOINT"||version!=1)return std::nullopt;
    for(float&w:s.model.weights)if(!(in>>w))return std::nullopt;
    if (!(in >> s.completed >> s.wins >> s.losses >> s.draws >> s.loss >> s.rating >> s.rng)) return std::nullopt;
    return s;
}

std::string progressBar(double ratio,int columns=28) { int full=std::clamp(static_cast<int>(ratio*columns),0,columns); return "["+std::string(full,'#')+std::string(columns-full,'.')+"]"; }
void printBoard(const Position& p) { const char* names=" PNBRQK"; std::cout<<"\n  +-----------------+\n";for(int r=7;r>=0;--r){std::cout<<r+1<<" | ";for(int f=0;f<8;++f){int pc=p.board[square(f,r)];char c=pc?names[type(pc)]:'.';std::cout<<(pc<0?static_cast<char>(std::tolower(c)):c)<<' ';}std::cout<<"|\n";}std::cout<<"  +-----------------+\n    a b c d e f g h\n"; }

int optionInt(const std::vector<std::string>& args,std::string_view name,int fallback) { for(std::size_t i=0;i+1<args.size();++i)if(args[i]==name)try{return std::stoi(args[i+1]);}catch(...){return fallback;} return fallback; }
std::string optionText(const std::vector<std::string>& args,std::string_view name) { for(std::size_t i=0;i+1<args.size();++i)if(args[i]==name)return args[i+1]; return {}; }

void train(const std::vector<std::string>& args) {
    int total=std::max(1,optionInt(args,"--games",100)), depth=std::clamp(optionInt(args,"--depth",3),1,12), maxPlies=std::clamp(optionInt(args,"--max-plies",300),20,1000); double rate=std::max(0.00001,static_cast<double>(optionInt(args,"--learning-rate",8))/100000.0);
    std::filesystem::path checkpoint=optionText(args,"--checkpoint"); TrainingState state; const auto resume=optionText(args,"--resume");
    if(!resume.empty()){auto saved=loadCheckpoint(resume);if(!saved){std::cerr<<"error: cannot read checkpoint "<<resume<<"\n";return;}state=*saved;}
    if(auto seed=optionText(args,"--seed");!seed.empty())try{state.rng.seed(std::stoull(seed));}catch(...){std::cerr<<"error: invalid seed\n";return;}
    std::signal(SIGINT,onInterrupt); std::cout<<"Asterion self-play trainer | depth "<<depth<<" | checkpoint "<<(checkpoint.empty()?"disabled":checkpoint.string())<<"\n";
    while(state.completed<static_cast<std::uint64_t>(total)) {
        Position p=Position::start(); std::vector<std::array<float,16>> samples; int result=0;
        // Random opening plies create diverse, reproducible self-play trajectories.
        std::uniform_int_distribution<int> openingLength(0, 5);
        for (int i=0, count=openingLength(state.rng); i<count; ++i) {
            auto openingMoves=p.legalMoves(); if(openingMoves.empty()) break;
            std::uniform_int_distribution<std::size_t> pick(0, openingMoves.size()-1);
            p.make(openingMoves[pick(state.rng)]);
        }
        for(int ply=0;ply<maxPlies;++ply) { samples.push_back(state.model.features(p)); Search search(state.model,8); Move m=search.findBest(p,depth,0,false); if(m.uci()=="a1a1"){result=p.inCheck(p.side)?(p.side==White?-1:1):0;break;} p.make(m); if(p.halfmove>=100){result=0;break;} if(ply==maxPlies-1)result=0; }
        if(result>0)++state.wins;else if(result<0)++state.losses;else ++state.draws;
        float target=(result+1)*0.5f; double gameLoss=0; for(const auto& x:samples){float raw=0;for(int i=0;i<16;++i)raw+=state.model.weights[i]*x[i];float pred=1.0f/(1.0f+std::exp(-raw/400.0f));float err=target-pred;gameLoss+=-target*std::log(std::max(pred,1e-6f))-(1-target)*std::log(std::max(1-pred,1e-6f));for(int i=0;i<16;++i)state.model.weights[i]+=static_cast<float>(rate*err*x[i]);}
        ++state.completed;
        state.loss=(state.loss*(state.completed-1)+gameLoss/std::max<std::size_t>(1,samples.size()))/state.completed; double ratio=static_cast<double>(state.wins+0.5*state.draws)/state.completed;state.rating=800.0+800.0*ratio;
        if(!checkpoint.empty()&&!saveCheckpoint(state,checkpoint)){std::cerr<<"\nerror: checkpoint write failed; stopping to preserve the last good checkpoint\n";return;}
        std::cout<<"\r"<<progressBar(static_cast<double>(state.completed)/total)<<' '<<std::setw(6)<<state.completed<<'/'<<total<<" proxy "<<std::fixed<<std::setprecision(0)<<state.rating<<" loss "<<std::setprecision(4)<<state.loss<<std::flush;
        if(pauseRequested){std::cout<<"\nPaused after a complete game. Resume with --resume "<<checkpoint<<"\n";return;}
    }
    if (!checkpoint.empty()) saveCheckpoint(state, checkpoint);
    std::cout << "\nTraining complete. Games " << state.completed << ", W/D/L " << state.wins << '/' << state.draws << '/' << state.losses << ".\n";
}

void perft(const Position& p,int depth,std::uint64_t& nodes){if(depth==0){++nodes;return;}for(const auto&m:p.legalMoves()){Position n=p;n.make(m);perft(n,depth-1,nodes);}}
void uci(){Model model; Position current=Position::start(); std::string line;while(std::getline(std::cin,line)){std::istringstream in(line);std::string cmd;in>>cmd;if(cmd=="uci"){std::cout<<"id name Asterion 0.1\nid author Asterion contributors\noption name Hash type spin default 32 min 1 max 1024\nuciok\n";}else if(cmd=="isready")std::cout<<"readyok\n";else if(cmd=="ucinewgame"){}else if(cmd=="position"){std::string kind;in>>kind;Position p;if(kind=="startpos")p=Position::start();else if(kind=="fen"){std::string part,fen;for(int i=0;i<6&&in>>part;++i)fen+=(i?" ":"")+part;auto parsed=Position::fromFen(fen);if(!parsed)continue;p=*parsed;}else continue;std::string movesWord;if(in>>movesWord&&movesWord=="moves"){std::string text;while(in>>text){auto moves=p.legalMoves();auto it=std::find_if(moves.begin(),moves.end(),[&](const Move&m){return m.uci()==text;});if(it==moves.end())break;p.make(*it);}} current=p; }else if(cmd=="go"){int depth=6,time=0;std::string key;while(in>>key){if(key=="depth")in>>depth;else if(key=="movetime")in>>time;}Search s(model);Move best=s.findBest(current,std::clamp(depth,1,30),time);std::cout<<"bestmove "<<best.uci()<<"\n";}else if(cmd=="quit")break;}}

void play(const std::vector<std::string>& args){Model model;Position p=Position::start();int depth=std::clamp(optionInt(args,"--depth",4),1,10);std::cout<<"Asterion local board. Enter UCI moves such as e2e4, or 'quit'.\n";while(true){printBoard(p);auto legal=p.legalMoves();if(legal.empty()){std::cout<<(p.inCheck(p.side)?"Checkmate.\n":"Stalemate.\n");return;}if(p.side==White){std::cout<<"white> ";std::string text;if(!(std::cin>>text)||text=="quit")return;auto it=std::find_if(legal.begin(),legal.end(),[&](const Move&m){return m.uci()==text;});if(it==legal.end()){std::cout<<"Illegal move.\n";continue;}p.make(*it);}else{Search s(model);Move move=s.findBest(p,depth);std::cout<<"black: "<<move.uci()<<"\n";p.make(move);}}}

void help(){std::cout<<"Asterion - portable C++ chess engine\n\nCommands:\n  uci\n  bench --depth N\n  perft --depth N\n  play --depth N\n  train --games N --depth N [--checkpoint FILE] [--resume FILE] [--seed N]\n";}
int run(int argc,char** argv){std::vector<std::string>args(argv+1,argv+argc);if(args.empty()){help();return 0;}if(args[0]=="uci"){uci();return 0;}if(args[0]=="train"){train(args);return 0;}if(args[0]=="perft"){std::uint64_t n=0;int d=std::clamp(optionInt(args,"--depth",4),0,8);auto t=std::chrono::steady_clock::now();Position root=Position::start();if(auto fen=optionText(args,"--fen");!fen.empty()){auto parsed=Position::fromFen(fen);if(!parsed){std::cerr<<"error: invalid FEN\n";return 2;}root=*parsed;}if(std::find(args.begin(),args.end(),"--divide")!=args.end()&&d>0){for(const auto&m:root.legalMoves()){Position next=root;next.make(m);std::uint64_t branch=0;perft(next,d-1,branch);std::cout<<m.uci()<<": "<<branch<<"\n";n+=branch;}}else perft(root,d,n);auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t).count();std::cout<<"perft "<<d<<": "<<n<<" nodes in "<<ms<<" ms\n";if(std::find(args.begin(),args.end(),"--verify")!=args.end()){constexpr std::array<std::uint64_t,5> expected{1,20,400,8902,197281};if(d>4||n!=expected[d]||!optionText(args,"--fen").empty()){std::cerr<<"perft verification failed\n";return 2;}}return 0;}if(args[0]=="bench"){Model m;Search s(m);auto t=std::chrono::steady_clock::now();Move best=s.findBest(Position::start(),std::clamp(optionInt(args,"--depth",6),1,12));auto ms=std::max<std::int64_t>(1,std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t).count());std::cout<<"bench bestmove "<<best.uci()<<" nodes "<<s.nodes()<<" nps "<<s.nodes()*1000/ms<<"\n";return 0;}if(args[0]=="play"){play(args);return 0;}help();return 1;}
} // namespace asterion

int main(int argc,char** argv){return asterion::run(argc,argv);}
