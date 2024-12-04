ideas:
- corners are good
- avoid x-squares and c-squares at all costs
- centralization
- maximising disks in sweet 16 (for an 8x8 board)
- fewer surface discs are better
- group discs together in a compact manner

- never play into x-squares or c-squares, unless given no other choice
- given only the choice of an x-square or a c-square pick the c-square
- minimize surface discs at the start until you have a corner
- flip minimal discs per move
- minimize manhattan distance between your discs
- prioritize flipping discs in the inner side

- search algorithms:
  - ids
  - minimax
  - mnimax with alpha beta pruning
  - negamax (with ab)
  - monte carlo sim
  - negascout (pvs)
  - mtd(f)

- features:
  - static weights
  - move ordering
  - transposition table
    - zobrist hashing
  - game stage detection (early game, midgame, endgame)
    - changes weights and evaluations
  - fixed sized data structure (lru) that cycles off the end
  - undo move function instead of copy
  - localized functions instead of helper imports
  - opening book
  - handle "checkmates"
  - why is it on depth 4000 at the end of the game
  - handle timeouts better

- heuristic:
  - stable nodes
  - better weights
  - edges
  - frontier
