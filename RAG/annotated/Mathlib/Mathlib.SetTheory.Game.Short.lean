/-- A short game is a game with a finite set of moves at every turn. -/
inductive Short : PGame.{u} → Type (u + 1)
  | mk :
    ∀ {α β : Type u} {L : α → PGame.{u}} {R : β → PGame.{u}} (_ : ∀ i : α, Short (L i))
      (_ : ∀ j : β, Short (R j)) [Fintype α] [Fintype β], Short ⟨α, β, L, R⟩


instance subsingleton_short (x : PGame) : Subsingleton (Short x) := by
  induction x with
  | mk xl xr xL xR =>
    constructor
    intro a b
    cases a; cases b
    congr!

-- Porting note: We use `induction` to prove `subsingleton_short` instead of recursion.
-- A proof using recursion generates a harder `decreasing_by` goal than in Lean 3 for some reason:

attribute [-instance] subsingleton_short in
theorem subsingleton_short_example : ∀ x : PGame, Subsingleton (Short x)
  | mk xl xr xL xR =>
    ⟨fun a b => by
      /-
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        a b : (SetTheory.PGame.mk xl xr xL xR).Short
        ⊢ Eq a b
      -/
      cases a; cases b
      /-
        case mk.mk
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        inst✝³ : Fintype xl
        inst✝² : Fintype xr
        x✝³ : (i : xl) → (xL i).Short
        x✝² : (j : xr) → (xR j).Short
        inst✝¹ : Fintype xl
        inst✝ : Fintype xr
        x✝¹ : (i : xl) → (xL i).Short
        x✝ : (j : xr) → (xR j).Short
        ⊢ Eq (SetTheory.PGame.Short.mk x✝³ x✝²) (SetTheory.PGame.Short.mk x✝¹ x✝)
      -/
      congr!
        /-
          case mk.mk.h.e'_5
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          inst✝³ : Fintype xl
          inst✝² : Fintype xr
          x✝³ : (i : xl) → (xL i).Short
          x✝² : (j : xr) → (xR j).Short
          inst✝¹ : Fintype xl
          inst✝ : Fintype xr
          x✝¹ : (i : xl) → (xL i).Short
          x✝ : (j : xr) → (xR j).Short
          ⊢ Eq x✝³ x✝¹
        -/
      · funext x
        /-
          case mk.mk.h.e'_5.h
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          inst✝³ : Fintype xl
          inst✝² : Fintype xr
          x✝³ : (i : xl) → (xL i).Short
          x✝² : (j : xr) → (xR j).Short
          inst✝¹ : Fintype xl
          inst✝ : Fintype xr
          x✝¹ : (i : xl) → (xL i).Short
          x✝ : (j : xr) → (xR j).Short
          x : xl
          ⊢ Eq (x✝³ x) (x✝¹ x)
        -/
        apply @Subsingleton.elim _ (subsingleton_short_example (xL x))
        /-
          🎉 no goals
        -/
        -- Decreasing goal in Lean 4 is `Subsequent (xL x) (mk α β L R)`
        -- where `α`, `β`, `L`, and `R` are fresh hypotheses only propositionally
        -- equal to `xl`, `xr`, `xL`, and `xR`.
        -- (In Lean 3 it was `(mk xl xr xL xR)` instead.)
        /-
          case mk.mk.h.e'_6
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          inst✝³ : Fintype xl
          inst✝² : Fintype xr
          x✝³ : (i : xl) → (xL i).Short
          x✝² : (j : xr) → (xR j).Short
          inst✝¹ : Fintype xl
          inst✝ : Fintype xr
          x✝¹ : (i : xl) → (xL i).Short
          x✝ : (j : xr) → (xR j).Short
          ⊢ Eq x✝² x✝
        -/
      · funext x
        /-
          case mk.mk.h.e'_6.h
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          inst✝³ : Fintype xl
          inst✝² : Fintype xr
          x✝³ : (i : xl) → (xL i).Short
          x✝² : (j : xr) → (xR j).Short
          inst✝¹ : Fintype xl
          inst✝ : Fintype xr
          x✝¹ : (i : xl) → (xL i).Short
          x✝ : (j : xr) → (xR j).Short
          x : xr
          ⊢ Eq (x✝² x) (x✝ x)
        -/
        apply @Subsingleton.elim _ (subsingleton_short_example (xR x))⟩
        /-
          🎉 no goals
        -/
termination_by x => x
-- We need to unify a bunch of hypotheses before `pgame_wf_tac` can work.
decreasing_by all_goals {
  subst_vars
  simp only [mk.injEq, heq_eq_eq, true_and] at *
  casesm* _ ∧ _
  subst_vars
  pgame_wf_tac
}


/-- A synonym for `Short.mk` that specifies the pgame in an implicit argument. -/
def Short.mk' {x : PGame} [Fintype x.LeftMoves] [Fintype x.RightMoves]
    (sL : ∀ i : x.LeftMoves, Short (x.moveLeft i))
    (sR : ∀ j : x.RightMoves, Short (x.moveRight j)) : Short x := by
  -- Porting note: Old proof relied on `unfreezingI`, which doesn't exist in Lean 4.
  /-
    x : SetTheory.PGame
    inst✝¹ : Fintype x.LeftMoves
    inst✝ : Fintype x.RightMoves
    sL : (i : x.LeftMoves) → (x.moveLeft i).Short
    sR : (j : x.RightMoves) → (x.moveRight j).Short
    ⊢ x.Short
  -/
  convert Short.mk sL sR
  /-
    case h.e'_1
    x : SetTheory.PGame
    inst✝¹ : Fintype x.LeftMoves
    inst✝ : Fintype x.RightMoves
    sL : (i : x.LeftMoves) → (x.moveLeft i).Short
    sR : (j : x.RightMoves) → (x.moveRight j).Short
    ⊢ Eq x (SetTheory.PGame.mk x.LeftMoves x.RightMoves x.moveLeft x.moveRight)
  -/
  cases x
  /-
    case h.e'_1.mk
    α✝ β✝ : Type ?u.7018
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    inst✝¹ : Fintype (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).LeftMoves
    inst✝ : Fintype (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).RightMoves
    sL : (i : (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).LeftMoves) → ((SetTheory.PGame.mk  …
    sR : (j : (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).RightMoves) → ((SetTheory.PGame.mk …
    ⊢ Eq (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝) (SetTheory.PGame.mk (SetTheory.PGame.mk …
  -/
  dsimp
  /-
    🎉 no goals
  -/


/-- Extracting the `Fintype` instance for the indexing type for Left's moves in a short game.
This is an unindexed typeclass, so it can't be made a global instance.
-/
def fintypeLeft {α β : Type u} {L : α → PGame.{u}} {R : β → PGame.{u}} [S : Short ⟨α, β, L, R⟩] :
                    /-
                      α β : Type u
                      L : α → SetTheory.PGame
                      R : β → SetTheory.PGame
                      S : (SetTheory.PGame.mk α β L R).Short
                      ⊢ Fintype α
                    -/
    Fintype α := by cases' S with _ _ _ _ _ _ F _; exact F
                                                   /-
                                                     🎉 no goals
                                                   -/


instance fintypeLeftMoves (x : PGame) [S : Short x] : Fintype x.LeftMoves := by
  /-
    x : SetTheory.PGame
    S : x.Short
    ⊢ Fintype x.LeftMoves
  -/
  cases S; assumption
           /-
             🎉 no goals
           -/


/-- Extracting the `Fintype` instance for the indexing type for Right's moves in a short game.
This is an unindexed typeclass, so it can't be made a global instance.
-/
def fintypeRight {α β : Type u} {L : α → PGame.{u}} {R : β → PGame.{u}} [S : Short ⟨α, β, L, R⟩] :
                    /-
                      α β : Type u
                      L : α → SetTheory.PGame
                      R : β → SetTheory.PGame
                      S : (SetTheory.PGame.mk α β L R).Short
                      ⊢ Fintype β
                    -/
    Fintype β := by cases' S with _ _ _ _ _ _ _ F; exact F
                                                   /-
                                                     🎉 no goals
                                                   -/


instance fintypeRightMoves (x : PGame) [S : Short x] : Fintype x.RightMoves := by
  /-
    x : SetTheory.PGame
    S : x.Short
    ⊢ Fintype x.RightMoves
  -/
  cases S; assumption
           /-
             🎉 no goals
           -/


instance moveLeftShort (x : PGame) [S : Short x] (i : x.LeftMoves) : Short (x.moveLeft i) := by
  /-
    x : SetTheory.PGame
    S : x.Short
    i : x.LeftMoves
    ⊢ (x.moveLeft i).Short
  -/
  cases' S with _ _ _ _ L _ _ _; apply L
                                 /-
                                   🎉 no goals
                                 -/


/-- Extracting the `Short` instance for a move by Left.
This would be a dangerous instance potentially introducing new metavariables
in typeclass search, so we only make it an instance locally.
-/
def moveLeftShort' {xl xr} (xL xR) [S : Short (mk xl xr xL xR)] (i : xl) : Short (xL i) := by
  /-
    xl xr : Type ?u.9006
    xL : xl → SetTheory.PGame
    xR : xr → SetTheory.PGame
    S : (SetTheory.PGame.mk xl xr xL xR).Short
    i : xl
    ⊢ (xL i).Short
  -/
  cases' S with _ _ _ _ L _ _ _; apply L
                                 /-
                                   🎉 no goals
                                 -/


instance moveRightShort (x : PGame) [S : Short x] (j : x.RightMoves) : Short (x.moveRight j) := by
  /-
    x : SetTheory.PGame
    S : x.Short
    j : x.RightMoves
    ⊢ (x.moveRight j).Short
  -/
  cases' S with _ _ _ _ _ R _ _; apply R
                                 /-
                                   🎉 no goals
                                 -/


/-- Extracting the `Short` instance for a move by Right.
This would be a dangerous instance potentially introducing new metavariables
in typeclass search, so we only make it an instance locally.
-/
def moveRightShort' {xl xr} (xL xR) [S : Short (mk xl xr xL xR)] (j : xr) : Short (xR j) := by
  /-
    xl xr : Type ?u.9535
    xL : xl → SetTheory.PGame
    xR : xr → SetTheory.PGame
    S : (SetTheory.PGame.mk xl xr xL xR).Short
    j : xr
    ⊢ (xR j).Short
  -/
  cases' S with _ _ _ _ _ R _ _; apply R
                                 /-
                                   🎉 no goals
                                 -/


theorem short_birthday (x : PGame.{u}) : [Short x] → x.birthday < Ordinal.omega0 := by
  -- Porting note: Again `induction` is used instead of `pgame_wf_tac`
  induction x with
  | mk xl xr xL xR ihl ihr =>
    intro hs
    rcases hs with ⟨sL, sR⟩
    rw [birthday, max_lt_iff]
    constructor
    all_goals
      rw [← Cardinal.ord_aleph0]
      refine
        Cardinal.lsub_lt_ord_of_isRegular.{u, u} Cardinal.isRegular_aleph0
          (Cardinal.lt_aleph0_of_finite _) fun i => ?_
      rw [Cardinal.ord_aleph0]
    · apply ihl
    · apply ihr


/-- This leads to infinite loops if made into an instance. -/
def Short.ofIsEmpty {l r xL xR} [IsEmpty l] [IsEmpty r] : Short (PGame.mk l r xL xR) := by
  /-
    l r : Type ?u.10638
    xL : l → SetTheory.PGame
    xR : r → SetTheory.PGame
    inst✝¹ : IsEmpty l
    inst✝ : IsEmpty r
    ⊢ (SetTheory.PGame.mk l r xL xR).Short
  -/
  have : Fintype l := Fintype.ofIsEmpty
  /-
    l r : Type ?u.10638
    xL : l → SetTheory.PGame
    xR : r → SetTheory.PGame
    inst✝¹ : IsEmpty l
    inst✝ : IsEmpty r
    this : Fintype l
    ⊢ (SetTheory.PGame.mk l r xL xR).Short
  -/
  have : Fintype r := Fintype.ofIsEmpty
  /-
    l r : Type ?u.10638
    xL : l → SetTheory.PGame
    xR : r → SetTheory.PGame
    inst✝¹ : IsEmpty l
    inst✝ : IsEmpty r
    this✝ : Fintype l
    this : Fintype r
    ⊢ (SetTheory.PGame.mk l r xL xR).Short
  -/
  exact Short.mk isEmptyElim isEmptyElim
  /-
    🎉 no goals
  -/


instance short0 : Short 0 :=
  Short.ofIsEmpty


instance short1 : Short 1 :=
                        /-
                          i : PUnit.{?u.11016 + 1}
                          ⊢ SetTheory.PGame.Short 0
                        -/
                                 /-
                                   🎉 no goals
                                 -/
  Short.mk (fun i => by cases i; infer_instance) fun j => by cases j
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- Evidence that every `PGame` in a list is `Short`. -/
class inductive ListShort : List PGame.{u} → Type (u + 1)
  | nil : ListShort []
  -- Porting note: We introduce `cons` as a separate instance because attempting to use
  -- `[ListShort tl]` as a constructor argument errors saying that `ListShort tl` is not a class.
  -- Is this a bug in `class inductive`?
  | cons' {hd : PGame.{u}} {tl : List PGame.{u}} : Short hd → ListShort tl → ListShort (hd::tl)


instance ListShort.cons
    (hd : PGame.{u}) [short_hd : Short hd] (tl : List PGame.{u}) [short_tl : ListShort tl] :
    ListShort (hd::tl) :=
  cons' short_hd short_tl


instance listShortGet :
    ∀ (L : List PGame.{u}) [ListShort L] (i : Nat) (h : i < List.length L), Short L[i]
  | _::_, ListShort.cons' S _, 0, _ => S
  | _::tl, ListShort.cons' _ S, n + 1, h =>
    @listShortGet tl S n ((add_lt_add_iff_right 1).mp h)


instance shortOfLists : ∀ (L R : List PGame) [ListShort L] [ListShort R], Short (PGame.ofLists L R)
  | L, R, _, _ => by
    /-
      L R : List SetTheory.PGame
      x✝¹ : SetTheory.PGame.ListShort L
      x✝ : SetTheory.PGame.ListShort R
      ⊢ (SetTheory.PGame.ofLists L R).Short
    -/
    exact Short.mk (fun i ↦ inferInstance) fun j ↦ listShortGet R (↑j.down) (ofLists.proof_2 R j)
    /-
      🎉 no goals
    -/


/-- If `x` is a short game, and `y` is a relabelling of `x`, then `y` is also short. -/
def shortOfRelabelling : ∀ {x y : PGame.{u}}, Relabelling x y → Short x → Short y
  | x, y, ⟨L, R, rL, rR⟩, S => by
    /-
      x y : SetTheory.PGame
      L : _root_.Equiv x.LeftMoves y.LeftMoves
      R : _root_.Equiv x.RightMoves y.RightMoves
      rL : (i : x.LeftMoves) → (x.moveLeft i).Relabelling (y.moveLeft (L i))
      rR : (j : x.RightMoves) → (x.moveRight j).Relabelling (y.moveRight (R j))
      S : x.Short
      ⊢ y.Short
    -/
    haveI := Fintype.ofEquiv _ L
    /-
      x y : SetTheory.PGame
      L : _root_.Equiv x.LeftMoves y.LeftMoves
      R : _root_.Equiv x.RightMoves y.RightMoves
      rL : (i : x.LeftMoves) → (x.moveLeft i).Relabelling (y.moveLeft (L i))
      rR : (j : x.RightMoves) → (x.moveRight j).Relabelling (y.moveRight (R j))
      S : x.Short
      this : Fintype y.LeftMoves
      ⊢ y.Short
    -/
    haveI := Fintype.ofEquiv _ R
    exact
      Short.mk'
        (fun i => by rw [← L.right_inv i]; apply shortOfRelabelling (rL (L.symm i)) inferInstance)
        fun j => by simpa using shortOfRelabelling (rR (R.symm j)) inferInstance


instance shortNeg : ∀ (x : PGame.{u}) [Short x], Short (-x)
  | mk xl xr xL xR, _ => by
    /-
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      x✝ : (SetTheory.PGame.mk xl xr xL xR).Short
      ⊢ (Neg.neg (SetTheory.PGame.mk xl xr xL xR)).Short
    -/
    exact Short.mk (fun i => shortNeg _) fun i => shortNeg _
    /-
      🎉 no goals
    -/


instance shortAdd : ∀ (x y : PGame.{u}) [Short x] [Short y], Short (x + y)
  | mk xl xr xL xR, mk yl yr yL yR, _, _ => by
    /-
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      x✝¹ : (SetTheory.PGame.mk xl xr xL xR).Short
      x✝ : (SetTheory.PGame.mk yl yr yL yR).Short
      ⊢ (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk yl yr yL yR) …
    -/
    apply Short.mk
    all_goals
      rintro ⟨i⟩
      · apply shortAdd
      · change Short (mk xl xr xL xR + _); apply shortAdd
termination_by x y => (x, y)


instance shortNat : ∀ n : ℕ, Short n
  | 0 => PGame.short0
  | n + 1 => @PGame.shortAdd _ _ (shortNat n) PGame.short1


instance shortOfNat (n : ℕ) [Nat.AtLeastTwo n] : Short (no_index (OfNat.ofNat n)) := shortNat n


                                                                   /-
                                                                     x : SetTheory.PGame
                                                                     inst✝ : x.Short
                                                                     ⊢ (HAdd.hAdd x x).Short
                                                                   -/
instance shortBit0 (x : PGame.{u}) [Short x] : Short (x + x) := by infer_instance
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


instance shortBit1 (x : PGame.{u}) [Short x] : Short ((x + x) + 1) := shortAdd _ _


/-- Auxiliary construction of decidability instances.
We build `Decidable (x ≤ y)` and `Decidable (x ⧏ y)` in a simultaneous induction.
Instances for the two projections separately are provided below.
-/
def leLFDecidable : ∀ (x y : PGame.{u}) [Short x] [Short y], Decidable (x ≤ y) × Decidable (x ⧏ y)
  | mk xl xr xL xR, mk yl yr yL yR, shortx, shorty => by
    /-
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      shortx : (SetTheory.PGame.mk xl xr xL xR).Short
      shorty : (SetTheory.PGame.mk yl yr yL yR).Short
      ⊢ Prod (Decidable (LE.le (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk  …
    -/
    constructor
      /-
        case fst
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        shortx : (SetTheory.PGame.mk xl xr xL xR).Short
        shorty : (SetTheory.PGame.mk yl yr yL yR).Short
        ⊢ Decidable (LE.le (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk yl yr  …
      -/
    · refine @decidable_of_iff' _ _ mk_le_mk (id ?_)
      have : Decidable (∀ (i : xl), xL i ⧏ mk yl yr yL yR) := by
        apply @Fintype.decidableForallFintype xl _ ?_ _
        intro i
        apply (leLFDecidable _ _).2
      have : Decidable (∀ (j : yr), mk xl xr xL xR ⧏ yR j) := by
        apply @Fintype.decidableForallFintype yr _ ?_ _
        intro i
        apply (leLFDecidable _ _).2
      /-
        case fst
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        shortx : (SetTheory.PGame.mk xl xr xL xR).Short
        shorty : (SetTheory.PGame.mk yl yr yL yR).Short
        this✝ : Decidable (∀ (i : xl), (xL i).LF (SetTheory.PGame.mk yl yr yL yR))
        this : Decidable (∀ (j : yr), (SetTheory.PGame.mk xl xr xL xR).LF (yR j))
        ⊢ Decidable (And (∀ (i : xl), (xL i).LF (SetTheory.PGame.mk yl yr yL yR)) (∀ ( …
      -/
      exact inferInstanceAs (Decidable (_ ∧ _))
      /-
        🎉 no goals
      -/
      /-
        case snd
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        shortx : (SetTheory.PGame.mk xl xr xL xR).Short
        shorty : (SetTheory.PGame.mk yl yr yL yR).Short
        ⊢ Decidable ((SetTheory.PGame.mk xl xr xL xR).LF (SetTheory.PGame.mk yl yr yL  …
      -/
    · refine @decidable_of_iff' _ _ mk_lf_mk (id ?_)
      have : Decidable (∃ i, mk xl xr xL xR ≤ yL i) := by
        apply @Fintype.decidableExistsFintype yl _ ?_ _
        intro i
        apply (leLFDecidable _ _).1
      have : Decidable (∃ j, xR j ≤ mk yl yr yL yR) := by
        apply @Fintype.decidableExistsFintype xr _ ?_ _
        intro i
        apply (leLFDecidable _ _).1
      /-
        case snd
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        shortx : (SetTheory.PGame.mk xl xr xL xR).Short
        shorty : (SetTheory.PGame.mk yl yr yL yR).Short
        this✝ : Decidable (Exists fun i => LE.le (SetTheory.PGame.mk xl xr xL xR) (yL  …
        this : Decidable (Exists fun j => LE.le (xR j) (SetTheory.PGame.mk yl yr yL yR))
        ⊢ Decidable (Or (Exists fun i => LE.le (SetTheory.PGame.mk xl xr xL xR) (yL i) …
      -/
      exact inferInstanceAs (Decidable (_ ∨ _))
      /-
        🎉 no goals
      -/
termination_by x y => (x, y)


instance leDecidable (x y : PGame.{u}) [Short x] [Short y] : Decidable (x ≤ y) :=
  (leLFDecidable x y).1


instance lfDecidable (x y : PGame.{u}) [Short x] [Short y] : Decidable (x ⧏ y) :=
  (leLFDecidable x y).2


instance ltDecidable (x y : PGame.{u}) [Short x] [Short y] : Decidable (x < y) :=
  inferInstanceAs (Decidable (_ ∧ _))


instance equivDecidable (x y : PGame.{u}) [Short x] [Short y] : Decidable (x ≈ y) :=
  inferInstanceAs (Decidable (_ ∧ _))


