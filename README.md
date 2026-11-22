# Asterion

Asterion is a from-scratch, dependency-free C++20 chess-engine research project.
It has a legal chess move generator, iterative alpha-beta search, quiescence,
a fixed-size transposition table, a trainable linear evaluation, self-play,
portable checkpoints, and a UCI interface. The project is intentionally
honest about scope: it is a capable foundation, not a claim of Stockfish-level
strength. Reaching 3000 Elo requires much stronger search/evaluation work and
very large, carefully measured training runs.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The scalar build runs on any C++20-capable target. On a machine known to have
AVX2, opt into its vector evaluation path:

```sh
cmake -S . -B build-avx2 -DCMAKE_BUILD_TYPE=Release -DASTERION_ENABLE_AVX2=ON
cmake --build build-avx2 --config Release
```

Do not distribute an AVX2 binary to CPUs that do not support AVX2; make the
default scalar build for that.

## Commands

```sh
./build/asterion uci
./build/asterion bench --depth 6
./build/asterion perft --depth 5
./build/asterion perft --depth 4 --verify
./build/asterion train --games 10000 --depth 3 --checkpoint runs/latest.ast
./build/asterion train --games 10000 --depth 3 --resume runs/latest.ast --checkpoint runs/latest.ast
./build/asterion play --depth 5
```

`train` checkpoints only between completed games. Press Ctrl-C once to stop at
the next game boundary; the state includes the model, statistics, and full RNG
state, so `--resume` continues exactly from that saved state. `--games` is the
total desired game count, not an additional count after resuming.

The displayed training rating is a progress proxy derived from self-play
results, not a measured Elo. Establish real Elo with a reproducible gauntlet
against fixed opponents and an SPRT-style match procedure.

`perft --verify` validates the standard initial-position node totals through
depth 4. `--divide` prints a root-move breakdown when diagnosing move legality.
The CTest suite also checks the Kiwipete castling and tactical stress position.

## Design notes

- Board state uses a compact 64-square mailbox; a legal move list is formed by
  filtering pseudo-legal moves through king-safety checking.
- The search uses fail-soft negamax alpha-beta, hash move ordering, captures,
  iterative deepening, and quiescence. It is deliberately portable C++.
- The evaluation model uses sixteen material features. Self-play optimizes it
  against game outcomes. The AVX2 path only accelerates that dot product.
- Checkpoints are versioned text files with atomic replacement where the host
  filesystem supports it. A `.bak` copy is retained before replacement.

For performance context, the implementation follows the established practice
of dense board representations and incremental search state described by the
[Chess Programming Wiki](https://www.chessprogramming.org/Board_Representation).
For a genuinely competitive successor, the most productive next major step is
an incrementally updated NNUE evaluator; Stockfish documents why NNUE is
structured around cheaply updating only the changed features
[here](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html).
