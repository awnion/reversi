#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Board {
    pub black: u64,
    pub white: u64,
}

impl Board {
    pub fn new(black: u64, white: u64) -> Self {
        Self { black, white }
    }

    pub fn player_board(&self, is_black: bool) -> u64 {
        if is_black { self.black } else { self.white }
    }

    pub fn opponent_board(&self, is_black: bool) -> u64 {
        if is_black { self.white } else { self.black }
    }

    pub fn empty(&self) -> u64 {
        !(self.black | self.white)
    }

    pub fn legal_moves(&self, is_black: bool) -> u64 {
        let player = self.player_board(is_black);
        let opponent = self.opponent_board(is_black);
        let empty = self.empty();

        let mut moves = 0u64;

        for dir in 0..8 {
            moves |= get_flips_in_dir(player, opponent, empty, dir);
        }

        moves
    }

    pub fn apply_move(&self, is_black: bool, move_mask: u64) -> Board {
        let mut player = self.player_board(is_black);
        let mut opponent = self.opponent_board(is_black);
        
        let mut flipped = 0u64;
        for dir in 0..8 {
            flipped |= get_flips_from_move(player, opponent, move_mask, dir);
        }

        player |= move_mask | flipped;
        opponent &= !flipped;

        if is_black {
            Board::new(player, opponent)
        } else {
            Board::new(opponent, player)
        }
    }
}

fn get_flips_in_dir(player: u64, opponent: u64, empty: u64, dir: u8) -> u64 {
    let shift_op = get_shift(dir);
    let mask = get_mask(dir);

    let mut candidates = shift_op(player) & opponent & mask;
    let mut moves = 0u64;

    while candidates != 0 {
        let next = shift_op(candidates) & mask;
        moves |= next & empty;
        candidates = next & opponent;
    }

    moves
}

fn get_flips_from_move(player: u64, opponent: u64, move_mask: u64, dir: u8) -> u64 {
    let shift_op = get_shift(dir);
    let mask = get_mask(dir);
    
    let mut current = shift_op(move_mask) & opponent & mask;
    let mut flipped = 0u64;
    
    while current != 0 {
        flipped |= current;
        current = shift_op(current) & mask;
        if (current & player) != 0 {
            return flipped;
        }
        current &= opponent;
    }
    
    0
}

// Bit layout: bit i = row (i/8), col (i%8), row 0 = top, col 0 = left.
// shl increases bit index → increases row (S) or col (E).
// shr decreases bit index → decreases row (N) or col (W).
fn get_shift(dir: u8) -> fn(u64) -> u64 {
    match dir {
        0 => |x| x.wrapping_shl(8), // S  (row+1)
        1 => |x| x.wrapping_shr(8), // N  (row-1)
        2 => |x| x.wrapping_shl(1), // E  (col+1)
        3 => |x| x.wrapping_shr(1), // W  (col-1)
        4 => |x| x.wrapping_shl(9), // SE (row+1, col+1)
        5 => |x| x.wrapping_shr(9), // NW (row-1, col-1)
        6 => |x| x.wrapping_shl(7), // SW (row+1, col-1)
        7 => |x| x.wrapping_shr(7), // NE (row-1, col+1)
        _ => unreachable!(),
    }
}

// Prevents bit wrapping across board columns after a directional shift.
// After E/SE shifts (shl 1/9): col 7 wraps to col 0 of next row → remove col 0.
// After W/NW shifts (shr 1/9): col 0 wraps to col 7 of prev row → remove col 7.
// After SW shift  (shl 7):     col 0 wraps to col 7 of same row → remove col 7.
// After NE shift  (shr 7):     col 7 wraps to col 0 of same row → remove col 0.
fn get_mask(dir: u8) -> u64 {
    match dir {
        0 | 1 => 0xFFFFFFFFFFFFFFFF, // S, N:  vertical only, no col wrapping
        2 | 4 => 0xFEFEFEFEFEFEFEFE, // E, SE: remove col 0 (col 7 wrap artifact)
        3 | 5 => 0x7F7F7F7F7F7F7F7F, // W, NW: remove col 7 (col 0 wrap artifact)
        6     => 0x7F7F7F7F7F7F7F7F, // SW:    remove col 7 (col 0 wrap artifact)
        7     => 0xFEFEFEFEFEFEFEFE, // NE:    remove col 0 (col 7 wrap artifact)
        _ => unreachable!(),
    }
}
