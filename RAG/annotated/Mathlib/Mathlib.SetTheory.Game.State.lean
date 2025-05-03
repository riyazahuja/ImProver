/-- `SetTheory.PGame.State S` describes how to interpret `s : S` as a state of a combinatorial game.
Use `SetTheory.PGame.ofState s` or `SetTheory.Game.ofState s` to construct the game.

`SetTheory.PGame.State.l : S → Finset S` and `SetTheory.PGame.State.r : S → Finset S` describe
the states reachable by a move by Left or Right. `SetTheory.PGame.State.turnBound : S → ℕ`
gives an upper bound on the number of possible turns remaining from this state.
-/
class State (S : Type u) where
  turnBound : S → ℕ
  l : S → Finset S
  r : S → Finset S
  left_bound : ∀ {s t : S}, t ∈ l s → turnBound t < turnBound s
  right_bound : ∀ {s t : S}, t ∈ r s → turnBound t < turnBound s


theorem turnBound_ne_zero_of_left_move {s t : S} (m : t ∈ l s) : turnBound s ≠ 0 := by
  /-
    S : Type u
    inst✝ : SetTheory.PGame.State S
    s t : S
    m : Membership.mem (SetTheory.PGame.State.l s) t
    ⊢ Ne (SetTheory.PGame.State.turnBound s) 0
  -/
  intro h
  /-
    S : Type u
    inst✝ : SetTheory.PGame.State S
    s t : S
    m : Membership.mem (SetTheory.PGame.State.l s) t
    h : Eq (SetTheory.PGame.State.turnBound s) 0
    ⊢ False
  -/
  have t := left_bound m
  /-
    S : Type u
    inst✝ : SetTheory.PGame.State S
    s t✝ : S
    m : Membership.mem (SetTheory.PGame.State.l s) t✝
    h : Eq (SetTheory.PGame.State.turnBound s) 0
    t : LT.lt (SetTheory.PGame.State.turnBound t✝) (SetTheory.PGame.State.turnBoun …
    ⊢ False
  -/
  rw [h] at t
  /-
    S : Type u
    inst✝ : SetTheory.PGame.State S
    s t✝ : S
    m : Membership.mem (SetTheory.PGame.State.l s) t✝
    h : Eq (SetTheory.PGame.State.turnBound s) 0
    t : LT.lt (SetTheory.PGame.State.turnBound t✝) 0
    ⊢ False
  -/
  exact Nat.not_succ_le_zero _ t
  /-
    🎉 no goals
  -/


theorem turnBound_ne_zero_of_right_move {s t : S} (m : t ∈ r s) : turnBound s ≠ 0 := by
  /-
    S : Type u
    inst✝ : SetTheory.PGame.State S
    s t : S
    m : Membership.mem (SetTheory.PGame.State.r s) t
    ⊢ Ne (SetTheory.PGame.State.turnBound s) 0
  -/
  intro h
  /-
    S : Type u
    inst✝ : SetTheory.PGame.State S
    s t : S
    m : Membership.mem (SetTheory.PGame.State.r s) t
    h : Eq (SetTheory.PGame.State.turnBound s) 0
    ⊢ False
  -/
  have t := right_bound m
  /-
    S : Type u
    inst✝ : SetTheory.PGame.State S
    s t✝ : S
    m : Membership.mem (SetTheory.PGame.State.r s) t✝
    h : Eq (SetTheory.PGame.State.turnBound s) 0
    t : LT.lt (SetTheory.PGame.State.turnBound t✝) (SetTheory.PGame.State.turnBoun …
    ⊢ False
  -/
  rw [h] at t
  /-
    S : Type u
    inst✝ : SetTheory.PGame.State S
    s t✝ : S
    m : Membership.mem (SetTheory.PGame.State.r s) t✝
    h : Eq (SetTheory.PGame.State.turnBound s) 0
    t : LT.lt (SetTheory.PGame.State.turnBound t✝) 0
    ⊢ False
  -/
  exact Nat.not_succ_le_zero _ t
  /-
    🎉 no goals
  -/


theorem turnBound_of_left {s t : S} (m : t ∈ l s) (n : ℕ) (h : turnBound s ≤ n + 1) :
    turnBound t ≤ n :=
  Nat.le_of_lt_succ (Nat.lt_of_lt_of_le (left_bound m) h)


theorem turnBound_of_right {s t : S} (m : t ∈ r s) (n : ℕ) (h : turnBound s ≤ n + 1) :
    turnBound t ≤ n :=
  Nat.le_of_lt_succ (Nat.lt_of_lt_of_le (right_bound m) h)


/-- Construct a `PGame` from a state and a (not necessarily optimal) bound on the number of
turns remaining.
-/
def ofStateAux : ∀ (n : ℕ) (s : S), turnBound s ≤ n → PGame
  | 0, s, h =>
    PGame.mk { t // t ∈ l s } { t // t ∈ r s }
                   /-
                     S : Type u
                     inst✝ : SetTheory.PGame.State S
                     s : S
                     h : LE.le (SetTheory.PGame.State.turnBound s) 0
                     t : Subtype fun t => Membership.mem (SetTheory.PGame.State.l s) t
                     ⊢ SetTheory.PGame
                   -/
      (fun t => by exfalso; exact turnBound_ne_zero_of_left_move t.2 (nonpos_iff_eq_zero.mp h))
                            /-
                              🎉 no goals
                            -/
                  /-
                    S : Type u
                    inst✝ : SetTheory.PGame.State S
                    s : S
                    h : LE.le (SetTheory.PGame.State.turnBound s) 0
                    t : Subtype fun t => Membership.mem (SetTheory.PGame.State.r s) t
                    ⊢ SetTheory.PGame
                  -/
      fun t => by exfalso; exact turnBound_ne_zero_of_right_move t.2 (nonpos_iff_eq_zero.mp h)
                           /-
                             🎉 no goals
                           -/
  | n + 1, s, h =>
    PGame.mk { t // t ∈ l s } { t // t ∈ r s }
      (fun t => ofStateAux n t (turnBound_of_left t.2 n h)) fun t =>
      ofStateAux n t (turnBound_of_right t.2 n h)


/-- Two different (valid) turn bounds give equivalent games. -/
def ofStateAuxRelabelling :
    ∀ (s : S) (n m : ℕ) (hn : turnBound s ≤ n) (hm : turnBound s ≤ m),
      Relabelling (ofStateAux n s hn) (ofStateAux m s hm)
  | s, 0, 0, hn, hm => by
    /-
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      hn hm : LE.le (SetTheory.PGame.State.turnBound s) 0
      ⊢ (SetTheory.PGame.ofStateAux 0 s hn).Relabelling (SetTheory.PGame.ofStateAux  …
    -/
    dsimp [PGame.ofStateAux]
    /-
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      hn hm : LE.le (SetTheory.PGame.State.turnBound s) 0
      ⊢ (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory.PGame.State. …
    -/
    fconstructor
      /-
        case L
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        hn hm : LE.le (SetTheory.PGame.State.turnBound s) 0
        ⊢ _root_.Equiv (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case R
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        hn hm : LE.le (SetTheory.PGame.State.turnBound s) 0
        ⊢ _root_.Equiv (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case a
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        hn hm : LE.le (SetTheory.PGame.State.turnBound s) 0
        ⊢ (i : (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory.PGame.S …
      -/
    · intro i; dsimp at i; exfalso
      /-
        case a
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        hn hm : LE.le (SetTheory.PGame.State.turnBound s) 0
        i : Subtype fun t => Membership.mem (SetTheory.PGame.State.l s) t
        ⊢ False
      -/
      exact turnBound_ne_zero_of_left_move i.2 (nonpos_iff_eq_zero.mp hn)
      /-
        🎉 no goals
      -/
      /-
        case a
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        hn hm : LE.le (SetTheory.PGame.State.turnBound s) 0
        ⊢ (j : (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory.PGame.S …
      -/
    · intro j; dsimp at j; exfalso
      /-
        case a
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        hn hm : LE.le (SetTheory.PGame.State.turnBound s) 0
        j : Subtype fun t => Membership.mem (SetTheory.PGame.State.r s) t
        ⊢ False
      -/
      exact turnBound_ne_zero_of_right_move j.2 (nonpos_iff_eq_zero.mp hm)
      /-
        🎉 no goals
      -/
  | s, 0, m + 1, hn, hm => by
    /-
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      m : Nat
      hn : LE.le (SetTheory.PGame.State.turnBound s) 0
      hm : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd m 1)
      ⊢ (SetTheory.PGame.ofStateAux 0 s hn).Relabelling (SetTheory.PGame.ofStateAux  …
    -/
    dsimp [PGame.ofStateAux]
    /-
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      m : Nat
      hn : LE.le (SetTheory.PGame.State.turnBound s) 0
      hm : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd m 1)
      ⊢ (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory.PGame.State. …
    -/
    fconstructor
      /-
        case L
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        m : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) 0
        hm : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd m 1)
        ⊢ _root_.Equiv (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case R
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        m : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) 0
        hm : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd m 1)
        ⊢ _root_.Equiv (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case a
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        m : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) 0
        hm : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd m 1)
        ⊢ (i : (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory.PGame.S …
      -/
    · intro i; dsimp at i; exfalso
      /-
        case a
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        m : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) 0
        hm : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd m 1)
        i : Subtype fun t => Membership.mem (SetTheory.PGame.State.l s) t
        ⊢ False
      -/
      exact turnBound_ne_zero_of_left_move i.2 (nonpos_iff_eq_zero.mp hn)
      /-
        🎉 no goals
      -/
      /-
        case a
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        m : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) 0
        hm : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd m 1)
        ⊢ (j : (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory.PGame.S …
      -/
    · intro j; dsimp at j; exfalso
      /-
        case a
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        m : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) 0
        hm : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd m 1)
        j : Subtype fun t => Membership.mem (SetTheory.PGame.State.r s) t
        ⊢ False
      -/
      exact turnBound_ne_zero_of_right_move j.2 (nonpos_iff_eq_zero.mp hn)
      /-
        🎉 no goals
      -/
  | s, n + 1, 0, hn, hm => by
    /-
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      n : Nat
      hn : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n 1)
      hm : LE.le (SetTheory.PGame.State.turnBound s) 0
      ⊢ (SetTheory.PGame.ofStateAux (HAdd.hAdd n 1) s hn).Relabelling (SetTheory.PGa …
    -/
    dsimp [PGame.ofStateAux]
    /-
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      n : Nat
      hn : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n 1)
      hm : LE.le (SetTheory.PGame.State.turnBound s) 0
      ⊢ (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory.PGame.State. …
    -/
    fconstructor
      /-
        case L
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        n : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n 1)
        hm : LE.le (SetTheory.PGame.State.turnBound s) 0
        ⊢ _root_.Equiv (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case R
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        n : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n 1)
        hm : LE.le (SetTheory.PGame.State.turnBound s) 0
        ⊢ _root_.Equiv (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case a
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        n : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n 1)
        hm : LE.le (SetTheory.PGame.State.turnBound s) 0
        ⊢ (i : (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory.PGame.S …
      -/
    · intro i; dsimp at i; exfalso
      /-
        case a
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        n : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n 1)
        hm : LE.le (SetTheory.PGame.State.turnBound s) 0
        i : Subtype fun t => Membership.mem (SetTheory.PGame.State.l s) t
        ⊢ False
      -/
      exact turnBound_ne_zero_of_left_move i.2 (nonpos_iff_eq_zero.mp hm)
      /-
        🎉 no goals
      -/
      /-
        case a
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        n : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n 1)
        hm : LE.le (SetTheory.PGame.State.turnBound s) 0
        ⊢ (j : (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory.PGame.S …
      -/
    · intro j; dsimp at j; exfalso
      /-
        case a
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        n : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n 1)
        hm : LE.le (SetTheory.PGame.State.turnBound s) 0
        j : Subtype fun t => Membership.mem (SetTheory.PGame.State.r s) t
        ⊢ False
      -/
      exact turnBound_ne_zero_of_right_move j.2 (nonpos_iff_eq_zero.mp hm)
      /-
        🎉 no goals
      -/
  | s, n + 1, m + 1, hn, hm => by
    /-
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      n m : Nat
      hn : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n 1)
      hm : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd m 1)
      ⊢ (SetTheory.PGame.ofStateAux (HAdd.hAdd n 1) s hn).Relabelling (SetTheory.PGa …
    -/
    dsimp [PGame.ofStateAux]
    /-
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      n m : Nat
      hn : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n 1)
      hm : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd m 1)
      ⊢ (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory.PGame.State. …
    -/
    fconstructor
      /-
        case L
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        n m : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n 1)
        hm : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd m 1)
        ⊢ _root_.Equiv (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case R
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        n m : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n 1)
        hm : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd m 1)
        ⊢ _root_.Equiv (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case a
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        n m : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n 1)
        hm : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd m 1)
        ⊢ (i : (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory.PGame.S …
      -/
    · intro i
      /-
        case a
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        n m : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n 1)
        hm : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd m 1)
        i : (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory.PGame.Stat …
        ⊢ ((SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory.PGame.State …
      -/
      apply ofStateAuxRelabelling
      /-
        🎉 no goals
      -/
      /-
        case a
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        n m : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n 1)
        hm : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd m 1)
        ⊢ (j : (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory.PGame.S …
      -/
    · intro j
      /-
        case a
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        n m : Nat
        hn : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n 1)
        hm : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd m 1)
        j : (SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory.PGame.Stat …
        ⊢ ((SetTheory.PGame.mk (Subtype fun t => Membership.mem (SetTheory.PGame.State …
      -/
      apply ofStateAuxRelabelling
      /-
        🎉 no goals
      -/


/-- Construct a combinatorial `PGame` from a state. -/
def ofState (s : S) : PGame :=
  ofStateAux (turnBound s) s (refl _)


/-- The equivalence between `leftMoves` for a `PGame` constructed using `ofStateAux _ s _`, and
`L s`. -/
def leftMovesOfStateAux (n : ℕ) {s : S} (h : turnBound s ≤ n) :
                                                          /-
                                                            S : Type u
                                                            inst✝ : SetTheory.PGame.State S
                                                            n : Nat
                                                            s : S
                                                            h : LE.le (SetTheory.PGame.State.turnBound s) n
                                                            ⊢ _root_.Equiv (SetTheory.PGame.ofStateAux n s h).LeftMoves (Subtype fun t =>  …
                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    LeftMoves (ofStateAux n s h) ≃ { t // t ∈ l s } := by induction n <;> rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- The equivalence between `leftMoves` for a `PGame` constructed using `ofState s`, and `l s`. -/
def leftMovesOfState (s : S) : LeftMoves (ofState s) ≃ { t // t ∈ l s } :=
  leftMovesOfStateAux _ _


/-- The equivalence between `rightMoves` for a `PGame` constructed using `ofStateAux _ s _`, and
`R s`. -/
def rightMovesOfStateAux (n : ℕ) {s : S} (h : turnBound s ≤ n) :
                                                           /-
                                                             S : Type u
                                                             inst✝ : SetTheory.PGame.State S
                                                             n : Nat
                                                             s : S
                                                             h : LE.le (SetTheory.PGame.State.turnBound s) n
                                                             ⊢ _root_.Equiv (SetTheory.PGame.ofStateAux n s h).RightMoves (Subtype fun t => …
                                                           -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
    RightMoves (ofStateAux n s h) ≃ { t // t ∈ r s } := by induction n <;> rfl
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- The equivalence between `rightMoves` for a `PGame` constructed using `ofState s`, and
`R s`. -/
def rightMovesOfState (s : S) : RightMoves (ofState s) ≃ { t // t ∈ r s } :=
  rightMovesOfStateAux _ _


/-- The relabelling showing `moveLeft` applied to a game constructed using `ofStateAux`
has itself been constructed using `ofStateAux`.
-/
def relabellingMoveLeftAux (n : ℕ) {s : S} (h : turnBound s ≤ n)
    (t : LeftMoves (ofStateAux n s h)) :
    Relabelling (moveLeft (ofStateAux n s h) t)
      (ofStateAux (n - 1) ((leftMovesOfStateAux n h) t : S)
        (turnBound_of_left ((leftMovesOfStateAux n h) t).2 (n - 1)
          (Nat.le_trans h le_tsub_add))) := by
  /-
    S : Type u
    inst✝ : SetTheory.PGame.State S
    n : Nat
    s : S
    h : LE.le (SetTheory.PGame.State.turnBound s) n
    t : (SetTheory.PGame.ofStateAux n s h).LeftMoves
    ⊢ ((SetTheory.PGame.ofStateAux n s h).moveLeft t).Relabelling (SetTheory.PGame …
  -/
  induction n
    /-
      case zero
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      h : LE.le (SetTheory.PGame.State.turnBound s) 0
      t : (SetTheory.PGame.ofStateAux 0 s h).LeftMoves
      ⊢ ((SetTheory.PGame.ofStateAux 0 s h).moveLeft t).Relabelling (SetTheory.PGame …
    -/
  · have t' := (leftMovesOfStateAux 0 h) t
    /-
      case zero
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      h : LE.le (SetTheory.PGame.State.turnBound s) 0
      t : (SetTheory.PGame.ofStateAux 0 s h).LeftMoves
      t' : Subtype fun t => Membership.mem (SetTheory.PGame.State.l s) t
      ⊢ ((SetTheory.PGame.ofStateAux 0 s h).moveLeft t).Relabelling (SetTheory.PGame …
    -/
    exfalso; exact turnBound_ne_zero_of_left_move t'.2 (nonpos_iff_eq_zero.mp h)
             /-
               🎉 no goals
             -/
    /-
      case succ
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      n✝ : Nat
      a✝ : (h : LE.le (SetTheory.PGame.State.turnBound s) n✝) → (t : (SetTheory.PGam …
      h : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n✝ 1)
      t : (SetTheory.PGame.ofStateAux (HAdd.hAdd n✝ 1) s h).LeftMoves
      ⊢ ((SetTheory.PGame.ofStateAux (HAdd.hAdd n✝ 1) s h).moveLeft t).Relabelling ( …
    -/
  · rfl
    /-
      🎉 no goals
    -/


/-- The relabelling showing `moveLeft` applied to a game constructed using `of`
has itself been constructed using `of`.
-/
def relabellingMoveLeft (s : S) (t : LeftMoves (ofState s)) :
    Relabelling (moveLeft (ofState s) t) (ofState ((leftMovesOfState s).toFun t : S)) := by
  /-
    S : Type u
    inst✝ : SetTheory.PGame.State S
    s : S
    t : (SetTheory.PGame.ofState s).LeftMoves
    ⊢ ((SetTheory.PGame.ofState s).moveLeft t).Relabelling (SetTheory.PGame.ofStat …
  -/
  trans
    /-
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      t : (SetTheory.PGame.ofState s).LeftMoves
      ⊢ ((SetTheory.PGame.ofState s).moveLeft t).Relabelling ?m.6362
    -/
  · apply relabellingMoveLeftAux
    /-
      🎉 no goals
    -/
    /-
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      t : (SetTheory.PGame.ofState s).LeftMoves
      ⊢ (SetTheory.PGame.ofStateAux (HSub.hSub (SetTheory.PGame.State.turnBound s) 1 …
    -/
  · apply ofStateAuxRelabelling
    /-
      🎉 no goals
    -/


/-- The relabelling showing `moveRight` applied to a game constructed using `ofStateAux`
has itself been constructed using `ofStateAux`.
-/
def relabellingMoveRightAux (n : ℕ) {s : S} (h : turnBound s ≤ n)
    (t : RightMoves (ofStateAux n s h)) :
    Relabelling (moveRight (ofStateAux n s h) t)
      (ofStateAux (n - 1) ((rightMovesOfStateAux n h) t : S)
        (turnBound_of_right ((rightMovesOfStateAux n h) t).2 (n - 1)
          (Nat.le_trans h le_tsub_add))) := by
  /-
    S : Type u
    inst✝ : SetTheory.PGame.State S
    n : Nat
    s : S
    h : LE.le (SetTheory.PGame.State.turnBound s) n
    t : (SetTheory.PGame.ofStateAux n s h).RightMoves
    ⊢ ((SetTheory.PGame.ofStateAux n s h).moveRight t).Relabelling (SetTheory.PGam …
  -/
  induction n
    /-
      case zero
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      h : LE.le (SetTheory.PGame.State.turnBound s) 0
      t : (SetTheory.PGame.ofStateAux 0 s h).RightMoves
      ⊢ ((SetTheory.PGame.ofStateAux 0 s h).moveRight t).Relabelling (SetTheory.PGam …
    -/
  · have t' := (rightMovesOfStateAux 0 h) t
    /-
      case zero
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      h : LE.le (SetTheory.PGame.State.turnBound s) 0
      t : (SetTheory.PGame.ofStateAux 0 s h).RightMoves
      t' : Subtype fun t => Membership.mem (SetTheory.PGame.State.r s) t
      ⊢ ((SetTheory.PGame.ofStateAux 0 s h).moveRight t).Relabelling (SetTheory.PGam …
    -/
    exfalso; exact turnBound_ne_zero_of_right_move t'.2 (nonpos_iff_eq_zero.mp h)
             /-
               🎉 no goals
             -/
    /-
      case succ
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      n✝ : Nat
      a✝ : (h : LE.le (SetTheory.PGame.State.turnBound s) n✝) → (t : (SetTheory.PGam …
      h : LE.le (SetTheory.PGame.State.turnBound s) (HAdd.hAdd n✝ 1)
      t : (SetTheory.PGame.ofStateAux (HAdd.hAdd n✝ 1) s h).RightMoves
      ⊢ ((SetTheory.PGame.ofStateAux (HAdd.hAdd n✝ 1) s h).moveRight t).Relabelling  …
    -/
  · rfl
    /-
      🎉 no goals
    -/


/-- The relabelling showing `moveRight` applied to a game constructed using `of`
has itself been constructed using `of`.
-/
def relabellingMoveRight (s : S) (t : RightMoves (ofState s)) :
    Relabelling (moveRight (ofState s) t) (ofState ((rightMovesOfState s).toFun t : S)) := by
  /-
    S : Type u
    inst✝ : SetTheory.PGame.State S
    s : S
    t : (SetTheory.PGame.ofState s).RightMoves
    ⊢ ((SetTheory.PGame.ofState s).moveRight t).Relabelling (SetTheory.PGame.ofSta …
  -/
  trans
    /-
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      t : (SetTheory.PGame.ofState s).RightMoves
      ⊢ ((SetTheory.PGame.ofState s).moveRight t).Relabelling ?m.7620
    -/
  · apply relabellingMoveRightAux
    /-
      🎉 no goals
    -/
    /-
      S : Type u
      inst✝ : SetTheory.PGame.State S
      s : S
      t : (SetTheory.PGame.ofState s).RightMoves
      ⊢ (SetTheory.PGame.ofStateAux (HSub.hSub (SetTheory.PGame.State.turnBound s) 1 …
    -/
  · apply ofStateAuxRelabelling
    /-
      🎉 no goals
    -/


instance fintypeLeftMovesOfStateAux (n : ℕ) (s : S) (h : turnBound s ≤ n) :
    Fintype (LeftMoves (ofStateAux n s h)) := by
  /-
    S : Type u
    inst✝ : SetTheory.PGame.State S
    n : Nat
    s : S
    h : LE.le (SetTheory.PGame.State.turnBound s) n
    ⊢ Fintype (SetTheory.PGame.ofStateAux n s h).LeftMoves
  -/
  apply Fintype.ofEquiv _ (leftMovesOfStateAux _ _).symm
  /-
    🎉 no goals
  -/


instance fintypeRightMovesOfStateAux (n : ℕ) (s : S) (h : turnBound s ≤ n) :
    Fintype (RightMoves (ofStateAux n s h)) := by
  /-
    S : Type u
    inst✝ : SetTheory.PGame.State S
    n : Nat
    s : S
    h : LE.le (SetTheory.PGame.State.turnBound s) n
    ⊢ Fintype (SetTheory.PGame.ofStateAux n s h).RightMoves
  -/
  apply Fintype.ofEquiv _ (rightMovesOfStateAux _ _).symm
  /-
    🎉 no goals
  -/


instance shortOfStateAux : ∀ (n : ℕ) {s : S} (h : turnBound s ≤ n), Short (ofStateAux n s h)
  | 0, s, h =>
    Short.mk'
      (fun i => by
        /-
          S : Type u
          inst✝ : SetTheory.PGame.State S
          s : S
          h : LE.le (SetTheory.PGame.State.turnBound s) 0
          i : (SetTheory.PGame.ofStateAux 0 s h).LeftMoves
          ⊢ ((SetTheory.PGame.ofStateAux 0 s h).moveLeft i).Short
        -/
        have i := (leftMovesOfStateAux _ _).toFun i
        /-
          S : Type u
          inst✝ : SetTheory.PGame.State S
          s : S
          h : LE.le (SetTheory.PGame.State.turnBound s) 0
          i✝ : (SetTheory.PGame.ofStateAux 0 s h).LeftMoves
          i : Subtype fun t => Membership.mem (SetTheory.PGame.State.l s) t
          ⊢ ((SetTheory.PGame.ofStateAux 0 s h).moveLeft i✝).Short
        -/
        exfalso
        /-
          S : Type u
          inst✝ : SetTheory.PGame.State S
          s : S
          h : LE.le (SetTheory.PGame.State.turnBound s) 0
          i✝ : (SetTheory.PGame.ofStateAux 0 s h).LeftMoves
          i : Subtype fun t => Membership.mem (SetTheory.PGame.State.l s) t
          ⊢ False
        -/
        exact turnBound_ne_zero_of_left_move i.2 (nonpos_iff_eq_zero.mp h))
        /-
          🎉 no goals
        -/
      fun j => by
      /-
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        h : LE.le (SetTheory.PGame.State.turnBound s) 0
        j : (SetTheory.PGame.ofStateAux 0 s h).RightMoves
        ⊢ ((SetTheory.PGame.ofStateAux 0 s h).moveRight j).Short
      -/
      have j := (rightMovesOfStateAux _ _).toFun j
      /-
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        h : LE.le (SetTheory.PGame.State.turnBound s) 0
        j✝ : (SetTheory.PGame.ofStateAux 0 s h).RightMoves
        j : Subtype fun t => Membership.mem (SetTheory.PGame.State.r s) t
        ⊢ ((SetTheory.PGame.ofStateAux 0 s h).moveRight j✝).Short
      -/
      exfalso
      /-
        S : Type u
        inst✝ : SetTheory.PGame.State S
        s : S
        h : LE.le (SetTheory.PGame.State.turnBound s) 0
        j✝ : (SetTheory.PGame.ofStateAux 0 s h).RightMoves
        j : Subtype fun t => Membership.mem (SetTheory.PGame.State.r s) t
        ⊢ False
      -/
      exact turnBound_ne_zero_of_right_move j.2 (nonpos_iff_eq_zero.mp h)
      /-
        🎉 no goals
      -/
  | n + 1, _, h =>
    Short.mk'
      (fun i =>
        shortOfRelabelling (relabellingMoveLeftAux (n + 1) h i).symm (shortOfStateAux n _))
      fun j =>
      shortOfRelabelling (relabellingMoveRightAux (n + 1) h j).symm (shortOfStateAux n _)


instance shortOfState (s : S) : Short (ofState s) := by
  /-
    S : Type u
    inst✝ : SetTheory.PGame.State S
    s : S
    ⊢ (SetTheory.PGame.ofState s).Short
  -/
  dsimp [PGame.ofState]
  /-
    S : Type u
    inst✝ : SetTheory.PGame.State S
    s : S
    ⊢ (SetTheory.PGame.ofStateAux (SetTheory.PGame.State.turnBound s) s ⋯).Short
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Construct a combinatorial `Game` from a state. -/
def ofState {S : Type u} [PGame.State S] (s : S) : Game :=
  ⟦PGame.ofState s⟧


