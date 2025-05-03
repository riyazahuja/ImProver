/-- The type of combinatorial games. In ZFC, a combinatorial game is constructed from
  two sets of combinatorial games that have been constructed at an earlier
  stage. To do this in type theory, we say that a combinatorial pre-game is built
  inductively from two families of combinatorial games indexed over any type
  in Type u. The resulting type `PGame.{u}` lives in `Type (u+1)`,
  reflecting that it is a proper class in ZFC.
  A combinatorial game is then constructed by quotienting by the equivalence
  `x ≈ y ↔ x ≤ y ∧ y ≤ x`. -/
abbrev Game :=
  Quotient PGame.setoid


/-- Negation of games. -/
instance : Neg Game where
  neg := Quot.map Neg.neg <| fun _ _ => (neg_equiv_neg_iff).2


instance : Zero Game where zero := ⟦0⟧

instance : Add Game where
  add := Quotient.map₂ HAdd.hAdd <| fun _ _ hx _ _ hy => PGame.add_congr hx hy


instance instAddCommGroupWithOneGame : AddCommGroupWithOne Game where
  zero := ⟦0⟧
  one := ⟦1⟧
  add_zero := by
    /-
      ⊢ ∀ (a : SetTheory.Game), Eq (HAdd.hAdd a 0) a
    -/
    rintro ⟨x⟩
    /-
      case mk
      a✝ : SetTheory.Game
      x : SetTheory.PGame
      ⊢ Eq (HAdd.hAdd (Quot.mk (⇑SetTheory.PGame.setoid) x) 0) (Quot.mk (⇑SetTheory. …
    -/
    /-
      ⊢ ∀ (a : SetTheory.Game), Eq (HAdd.hAdd 0 a) a
    -/
    exact Quot.sound (add_zero_equiv x)
    /-
      case mk
      a✝ : SetTheory.Game
      x : SetTheory.PGame
      ⊢ Eq (HAdd.hAdd 0 (Quot.mk (⇑SetTheory.PGame.setoid) x)) (Quot.mk (⇑SetTheory. …
    -/
    /-
      ⊢ ∀ (a b c : SetTheory.Game), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (H …
    -/
    /-
      🎉 no goals
    -/
    /-
      case mk.mk.mk
      a✝ : SetTheory.Game
      x : SetTheory.PGame
      b✝ : SetTheory.Game
      y : SetTheory.PGame
      c✝ : SetTheory.Game
      z : SetTheory.PGame
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Quot.mk (⇑SetTheory.PGame.setoid) x) (Quot.mk (⇑Se …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  zero_add := by
    rintro ⟨x⟩
    exact Quot.sound (zero_add_equiv x)
  add_assoc := by
    rintro ⟨x⟩ ⟨y⟩ ⟨z⟩
    exact Quot.sound add_assoc_equiv
  neg_add_cancel := Quotient.ind <| fun x => Quot.sound (neg_add_cancel_equiv x)
  add_comm := by
    /-
      ⊢ ∀ (a b : SetTheory.Game), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
    -/
    rintro ⟨x⟩ ⟨y⟩
    /-
      case mk.mk
      a✝ : SetTheory.Game
      x : SetTheory.PGame
      b✝ : SetTheory.Game
      y : SetTheory.PGame
      ⊢ Eq (HAdd.hAdd (Quot.mk (⇑SetTheory.PGame.setoid) x) (Quot.mk (⇑SetTheory.PGa …
    -/
    exact Quot.sound add_comm_equiv
    /-
      🎉 no goals
    -/
  nsmul := nsmulRec
  zsmul := zsmulRec


instance : Inhabited Game :=
  ⟨0⟩


theorem zero_def : (0 : Game) = ⟦0⟧ :=
  rfl


instance instPartialOrderGame : PartialOrder Game where
  le := Quotient.lift₂ (· ≤ ·) fun _ _ _ _ hx hy => propext (le_congr hx hy)
  le_refl := by
    /-
      ⊢ ∀ (a : SetTheory.Game), LE.le a a
    -/
    rintro ⟨x⟩
    /-
      case mk
      a✝ : SetTheory.Game
      x : SetTheory.PGame
      ⊢ LE.le (Quot.mk (⇑SetTheory.PGame.setoid) x) (Quot.mk (⇑SetTheory.PGame.setoi …
    -/
    exact le_refl x
    /-
      🎉 no goals
    -/
  le_trans := by
    /-
      ⊢ ∀ (a b c : SetTheory.Game), LE.le a b → LE.le b c → LE.le a c
    -/
    rintro ⟨x⟩ ⟨y⟩ ⟨z⟩
    /-
      case mk.mk.mk
      a✝ : SetTheory.Game
      x : SetTheory.PGame
      b✝ : SetTheory.Game
      y : SetTheory.PGame
      c✝ : SetTheory.Game
      z : SetTheory.PGame
      ⊢ LE.le (Quot.mk (⇑SetTheory.PGame.setoid) x) (Quot.mk (⇑SetTheory.PGame.setoi …
    -/
    exact @le_trans _ _ x y z
    /-
      🎉 no goals
    -/
  le_antisymm := by
    /-
      ⊢ ∀ (a b : SetTheory.Game), LE.le a b → LE.le b a → Eq a b
    -/
    rintro ⟨x⟩ ⟨y⟩ h₁ h₂
    /-
      case mk.mk
      a✝ : SetTheory.Game
      x : SetTheory.PGame
      b✝ : SetTheory.Game
      y : SetTheory.PGame
      h₁ : LE.le (Quot.mk (⇑SetTheory.PGame.setoid) x) (Quot.mk (⇑SetTheory.PGame.se …
      h₂ : LE.le (Quot.mk (⇑SetTheory.PGame.setoid) y) (Quot.mk (⇑SetTheory.PGame.se …
      ⊢ Eq (Quot.mk (⇑SetTheory.PGame.setoid) x) (Quot.mk (⇑SetTheory.PGame.setoid) y)
    -/
    apply Quot.sound
    /-
      case mk.mk.a
      a✝ : SetTheory.Game
      x : SetTheory.PGame
      b✝ : SetTheory.Game
      y : SetTheory.PGame
      h₁ : LE.le (Quot.mk (⇑SetTheory.PGame.setoid) x) (Quot.mk (⇑SetTheory.PGame.se …
      h₂ : LE.le (Quot.mk (⇑SetTheory.PGame.setoid) y) (Quot.mk (⇑SetTheory.PGame.se …
      ⊢ SetTheory.PGame.setoid x y
    -/
    /-
      ⊢ ∀ (a b : SetTheory.Game), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
    -/
    exact ⟨h₁, h₂⟩
    /-
      case mk.mk
      a✝ : SetTheory.Game
      x : SetTheory.PGame
      b✝ : SetTheory.Game
      y : SetTheory.PGame
      ⊢ Iff (LT.lt (Quot.mk (⇑SetTheory.PGame.setoid) x) (Quot.mk (⇑SetTheory.PGame. …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  lt := Quotient.lift₂ (· < ·) fun _ _ _ _ hx hy => propext (lt_congr hx hy)
  lt_iff_le_not_le := by
    rintro ⟨x⟩ ⟨y⟩
    exact @lt_iff_le_not_le _ _ x y


/-- The less or fuzzy relation on games.

If `0 ⧏ x` (less or fuzzy with), then Left can win `x` as the first player. -/
def LF : Game → Game → Prop :=
  Quotient.lift₂ PGame.LF fun _ _ _ _ hx hy => propext (lf_congr hx hy)


/-- On `Game`, simp-normal inequalities should use as few negations as possible. -/
@[simp]
theorem not_le : ∀ {x y : Game}, ¬x ≤ y ↔ Game.LF y x := by
  /-
    ⊢ ∀ {x y : SetTheory.Game}, Iff (Not (LE.le x y)) (y.LF x)
  -/
  rintro ⟨x⟩ ⟨y⟩
  /-
    case mk.mk
    x✝ : SetTheory.Game
    x : SetTheory.PGame
    y✝ : SetTheory.Game
    y : SetTheory.PGame
    ⊢ Iff (Not (LE.le (Quot.mk (⇑SetTheory.PGame.setoid) x) (Quot.mk (⇑SetTheory.P …
  -/
  exact PGame.not_le
  /-
    🎉 no goals
  -/


/-- On `Game`, simp-normal inequalities should use as few negations as possible. -/
@[simp]
theorem not_lf : ∀ {x y : Game}, ¬Game.LF x y ↔ y ≤ x := by
  /-
    ⊢ ∀ {x y : SetTheory.Game}, Iff (Not (x.LF y)) (LE.le y x)
  -/
  rintro ⟨x⟩ ⟨y⟩
  /-
    case mk.mk
    x✝ : SetTheory.Game
    x : SetTheory.PGame
    y✝ : SetTheory.Game
    y : SetTheory.PGame
    ⊢ Iff (Not (SetTheory.Game.LF (Quot.mk (⇑SetTheory.PGame.setoid) x) (Quot.mk ( …
  -/
  exact PGame.not_lf
  /-
    🎉 no goals
  -/


/-- The fuzzy, confused, or incomparable relation on games.

If `x ‖ 0`, then the first player can always win `x`. -/
def Fuzzy : Game → Game → Prop :=
  Quotient.lift₂ PGame.Fuzzy fun _ _ _ _ hx hy => propext (fuzzy_congr hx hy)

-- Porting note: had to replace ⧏ with LF, otherwise cannot differentiate with the operator on PGame

instance : IsTrichotomous Game LF :=
  ⟨by
    /-
      ⊢ ∀ (a b : SetTheory.Game), Or (a.LF b) (Or (Eq a b) (b.LF a))
    -/
    rintro ⟨x⟩ ⟨y⟩
    /-
      case mk.mk
      a✝ : SetTheory.Game
      x : SetTheory.PGame
      b✝ : SetTheory.Game
      y : SetTheory.PGame
      ⊢ Or (SetTheory.Game.LF (Quot.mk (⇑SetTheory.PGame.setoid) x) (Quot.mk (⇑SetTh …
    -/
    change _ ∨ ⟦x⟧ = ⟦y⟧ ∨ _
    /-
      case mk.mk
      a✝ : SetTheory.Game
      x : SetTheory.PGame
      b✝ : SetTheory.Game
      y : SetTheory.PGame
      ⊢ Or (SetTheory.Game.LF (Quot.mk (⇑SetTheory.PGame.setoid) x) (Quot.mk (⇑SetTh …
    -/
    rw [Quotient.eq]
    /-
      case mk.mk
      a✝ : SetTheory.Game
      x : SetTheory.PGame
      b✝ : SetTheory.Game
      y : SetTheory.PGame
      ⊢ Or (SetTheory.Game.LF (Quot.mk (⇑SetTheory.PGame.setoid) x) (Quot.mk (⇑SetTh …
    -/
    apply lf_or_equiv_or_gf⟩
    /-
      🎉 no goals
    -/


theorem le_iff_game_le {x y : PGame} : x ≤ y ↔ (⟦x⟧ : Game) ≤ ⟦y⟧ :=
  Iff.rfl


theorem lf_iff_game_lf {x y : PGame} : x ⧏ y ↔ Game.LF ⟦x⟧ ⟦y⟧ :=
  Iff.rfl


theorem lt_iff_game_lt {x y : PGame} : x < y ↔ (⟦x⟧ : Game) < ⟦y⟧ :=
  Iff.rfl


theorem equiv_iff_game_eq {x y : PGame} : x ≈ y ↔ (⟦x⟧ : Game) = ⟦y⟧ :=
  (@Quotient.eq' _ _ x y).symm


alias ⟨game_eq, _⟩ := equiv_iff_game_eq


theorem fuzzy_iff_game_fuzzy {x y : PGame} : x ‖ y ↔ Game.Fuzzy ⟦x⟧ ⟦y⟧ :=
  Iff.rfl


local infixl:50 " ⧏ " => LF

local infixl:50 " ‖ " => Fuzzy


instance addLeftMono : AddLeftMono Game :=
  ⟨by
    /-
      ⊢ Covariant SetTheory.Game SetTheory.Game (fun x1 x2 => HAdd.hAdd x1 x2) fun x …
    -/
    rintro ⟨a⟩ ⟨b⟩ ⟨c⟩ h
    /-
      case mk.mk.mk
      m✝ : SetTheory.Game
      a : SetTheory.PGame
      n₁✝ : SetTheory.Game
      b : SetTheory.PGame
      n₂✝ : SetTheory.Game
      c : SetTheory.PGame
      h : LE.le (Quot.mk (⇑SetTheory.PGame.setoid) b) (Quot.mk (⇑SetTheory.PGame.set …
      ⊢ LE.le ((fun x1 x2 => HAdd.hAdd x1 x2) (Quot.mk (⇑SetTheory.PGame.setoid) a)  …
    -/
    exact @add_le_add_left _ _ _ _ b c h a⟩
    /-
      🎉 no goals
    -/


instance addRightMono : AddRightMono Game :=
  ⟨by
    /-
      ⊢ Covariant SetTheory.Game SetTheory.Game (Function.swap fun x1 x2 => HAdd.hAd …
    -/
    rintro ⟨a⟩ ⟨b⟩ ⟨c⟩ h
    /-
      case mk.mk.mk
      m✝ : SetTheory.Game
      a : SetTheory.PGame
      n₁✝ : SetTheory.Game
      b : SetTheory.PGame
      n₂✝ : SetTheory.Game
      c : SetTheory.PGame
      h : LE.le (Quot.mk (⇑SetTheory.PGame.setoid) b) (Quot.mk (⇑SetTheory.PGame.set …
      ⊢ LE.le (Function.swap (fun x1 x2 => HAdd.hAdd x1 x2) (Quot.mk (⇑SetTheory.PGa …
    -/
    exact @add_le_add_right _ _ _ _ b c h a⟩
    /-
      🎉 no goals
    -/


instance addLeftStrictMono : AddLeftStrictMono Game :=
  ⟨by
    /-
      ⊢ Covariant SetTheory.Game SetTheory.Game (fun x1 x2 => HAdd.hAdd x1 x2) fun x …
    -/
    rintro ⟨a⟩ ⟨b⟩ ⟨c⟩ h
    /-
      case mk.mk.mk
      m✝ : SetTheory.Game
      a : SetTheory.PGame
      n₁✝ : SetTheory.Game
      b : SetTheory.PGame
      n₂✝ : SetTheory.Game
      c : SetTheory.PGame
      h : LT.lt (Quot.mk (⇑SetTheory.PGame.setoid) b) (Quot.mk (⇑SetTheory.PGame.set …
      ⊢ LT.lt ((fun x1 x2 => HAdd.hAdd x1 x2) (Quot.mk (⇑SetTheory.PGame.setoid) a)  …
    -/
    exact @add_lt_add_left _ _ _ _ b c h a⟩
    /-
      🎉 no goals
    -/


instance addRightStrictMono : AddRightStrictMono Game :=
  ⟨by
    /-
      ⊢ Covariant SetTheory.Game SetTheory.Game (Function.swap fun x1 x2 => HAdd.hAd …
    -/
    rintro ⟨a⟩ ⟨b⟩ ⟨c⟩ h
    /-
      case mk.mk.mk
      m✝ : SetTheory.Game
      a : SetTheory.PGame
      n₁✝ : SetTheory.Game
      b : SetTheory.PGame
      n₂✝ : SetTheory.Game
      c : SetTheory.PGame
      h : LT.lt (Quot.mk (⇑SetTheory.PGame.setoid) b) (Quot.mk (⇑SetTheory.PGame.set …
      ⊢ LT.lt (Function.swap (fun x1 x2 => HAdd.hAdd x1 x2) (Quot.mk (⇑SetTheory.PGa …
    -/
    exact @add_lt_add_right _ _ _ _ b c h a⟩
    /-
      🎉 no goals
    -/


theorem add_lf_add_right : ∀ {b c : Game} (_ : b ⧏ c) (a), (b + a : Game) ⧏ c + a := by
  /-
    ⊢ ∀ {b c : SetTheory.Game}, b.LF c → ∀ (a : SetTheory.Game), (HAdd.hAdd b a).L …
  -/
  rintro ⟨b⟩ ⟨c⟩ h ⟨a⟩
  /-
    case mk.mk.mk
    b✝ : SetTheory.Game
    b : SetTheory.PGame
    c✝ : SetTheory.Game
    c : SetTheory.PGame
    h : SetTheory.Game.LF (Quot.mk (⇑SetTheory.PGame.setoid) b) (Quot.mk (⇑SetTheo …
    a✝ : SetTheory.Game
    a : SetTheory.PGame
    ⊢ (HAdd.hAdd (Quot.mk (⇑SetTheory.PGame.setoid) b) (Quot.mk (⇑SetTheory.PGame. …
  -/
  apply PGame.add_lf_add_right h
  /-
    🎉 no goals
  -/


theorem add_lf_add_left : ∀ {b c : Game} (_ : b ⧏ c) (a), (a + b : Game) ⧏ a + c := by
  /-
    ⊢ ∀ {b c : SetTheory.Game}, b.LF c → ∀ (a : SetTheory.Game), (HAdd.hAdd a b).L …
  -/
  rintro ⟨b⟩ ⟨c⟩ h ⟨a⟩
  /-
    case mk.mk.mk
    b✝ : SetTheory.Game
    b : SetTheory.PGame
    c✝ : SetTheory.Game
    c : SetTheory.PGame
    h : SetTheory.Game.LF (Quot.mk (⇑SetTheory.PGame.setoid) b) (Quot.mk (⇑SetTheo …
    a✝ : SetTheory.Game
    a : SetTheory.PGame
    ⊢ (HAdd.hAdd (Quot.mk (⇑SetTheory.PGame.setoid) a) (Quot.mk (⇑SetTheory.PGame. …
  -/
  apply PGame.add_lf_add_left h
  /-
    🎉 no goals
  -/


instance orderedAddCommGroup : OrderedAddCommGroup Game :=
  { Game.instAddCommGroupWithOneGame, Game.instPartialOrderGame with
    add_le_add_left := @add_le_add_left _ _ _ Game.addLeftMono }


/-- A small family of games is bounded above. -/
lemma bddAbove_range_of_small {ι : Type*} [Small.{u} ι] (f : ι → Game.{u}) :
    BddAbove (Set.range f) := by
  /-
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → SetTheory.Game
    ⊢ BddAbove (Set.range f)
  -/
  obtain ⟨x, hx⟩ := PGame.bddAbove_range_of_small (Quotient.out ∘ f)
  /-
    case intro
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → SetTheory.Game
    x : SetTheory.PGame
    hx : Membership.mem (upperBounds (Set.range (Function.comp Quotient.out f))) x
    ⊢ BddAbove (Set.range f)
  -/
  refine ⟨⟦x⟧, Set.forall_mem_range.2 fun i ↦ ?_⟩
  /-
    case intro
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → SetTheory.Game
    x : SetTheory.PGame
    hx : Membership.mem (upperBounds (Set.range (Function.comp Quotient.out f))) x
    i : ι
    ⊢ LE.le (f i) (Quotient.mk SetTheory.PGame.setoid x)
  -/
  simpa [PGame.le_iff_game_le] using hx <| Set.mem_range_self i
  /-
    🎉 no goals
  -/


/-- A small set of games is bounded above. -/
lemma bddAbove_of_small (s : Set Game.{u}) [Small.{u} s] : BddAbove s := by
  /-
    s : Set SetTheory.Game
    inst✝ : Small.{u, u + 1} ↑s
    ⊢ BddAbove s
  -/
  simpa using bddAbove_range_of_small (Subtype.val : s → Game.{u})
  /-
    🎉 no goals
  -/


/-- A small family of games is bounded below. -/
lemma bddBelow_range_of_small {ι : Type*} [Small.{u} ι] (f : ι → Game.{u}) :
    BddBelow (Set.range f) := by
  /-
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → SetTheory.Game
    ⊢ BddBelow (Set.range f)
  -/
  obtain ⟨x, hx⟩ := PGame.bddBelow_range_of_small (Quotient.out ∘ f)
  /-
    case intro
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → SetTheory.Game
    x : SetTheory.PGame
    hx : Membership.mem (lowerBounds (Set.range (Function.comp Quotient.out f))) x
    ⊢ BddBelow (Set.range f)
  -/
  refine ⟨⟦x⟧, Set.forall_mem_range.2 fun i ↦ ?_⟩
  /-
    case intro
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → SetTheory.Game
    x : SetTheory.PGame
    hx : Membership.mem (lowerBounds (Set.range (Function.comp Quotient.out f))) x
    i : ι
    ⊢ LE.le (Quotient.mk SetTheory.PGame.setoid x) (f i)
  -/
  simpa [PGame.le_iff_game_le] using hx <| Set.mem_range_self i
  /-
    🎉 no goals
  -/


/-- A small set of games is bounded below. -/
lemma bddBelow_of_small (s : Set Game.{u}) [Small.{u} s] : BddBelow s := by
  /-
    s : Set SetTheory.Game
    inst✝ : Small.{u, u + 1} ↑s
    ⊢ BddBelow s
  -/
  simpa using bddBelow_range_of_small (Subtype.val : s → Game.{u})
  /-
    🎉 no goals
  -/


@[simp] theorem quot_zero : (⟦0⟧ : Game) = 0 := rfl

@[simp] theorem quot_one : (⟦1⟧ : Game) = 1 := rfl

@[simp] theorem quot_neg (a : PGame) : (⟦-a⟧ : Game) = -⟦a⟧ := rfl

@[simp] theorem quot_add (a b : PGame) : ⟦a + b⟧ = (⟦a⟧ : Game) + ⟦b⟧ := rfl

@[simp] theorem quot_sub (a b : PGame) : ⟦a - b⟧ = (⟦a⟧ : Game) - ⟦b⟧ := rfl


@[simp]
theorem quot_natCast : ∀ n : ℕ, ⟦(n : PGame)⟧ = (n : Game)
  | 0 => rfl
  | n + 1 => by
    /-
      n : Nat
      ⊢ Eq (Quotient.mk SetTheory.PGame.setoid ↑(HAdd.hAdd n 1)) ↑(HAdd.hAdd n 1)
    -/
    rw [PGame.nat_succ, quot_add, Nat.cast_add, Nat.cast_one, quot_natCast]
    /-
      n : Nat
      ⊢ Eq (HAdd.hAdd (↑n) (Quotient.mk SetTheory.PGame.setoid 1)) (HAdd.hAdd (↑n) 1)
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem quot_eq_of_mk'_quot_eq {x y : PGame} (L : x.LeftMoves ≃ y.LeftMoves)
    (R : x.RightMoves ≃ y.RightMoves) (hl : ∀ i, (⟦x.moveLeft i⟧ : Game) = ⟦y.moveLeft (L i)⟧)
    (hr : ∀ j, (⟦x.moveRight j⟧ : Game) = ⟦y.moveRight (R j)⟧) : (⟦x⟧ : Game) = ⟦y⟧ :=
  game_eq (.of_equiv L R (fun _ => equiv_iff_game_eq.2 (hl _))
    (fun _ => equiv_iff_game_eq.2 (hr _)))


/-- The product of `x = {xL | xR}` and `y = {yL | yR}` is
`{xL*y + x*yL - xL*yL, xR*y + x*yR - xR*yR | xL*y + x*yR - xL*yR, xR*y + x*yL - xR*yL}`. -/
instance : Mul PGame.{u} :=
  ⟨fun x y => by
    /-
      x y : SetTheory.PGame
      ⊢ SetTheory.PGame
    -/
    induction x generalizing y with | mk xl xr _ _ IHxl IHxr => _
    /-
      case mk
      xl xr : Type u
      a✝¹ : xl → SetTheory.PGame
      a✝ : xr → SetTheory.PGame
      IHxl : xl → SetTheory.PGame → SetTheory.PGame
      IHxr : xr → SetTheory.PGame → SetTheory.PGame
      y : SetTheory.PGame
      ⊢ SetTheory.PGame
    -/
    induction y with | mk yl yr yL yR IHyl IHyr => _
    /-
      case mk.mk
      xl xr : Type u
      a✝¹ : xl → SetTheory.PGame
      a✝ : xr → SetTheory.PGame
      IHxl : xl → SetTheory.PGame → SetTheory.PGame
      IHxr : xr → SetTheory.PGame → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      IHyl : yl → SetTheory.PGame
      IHyr : yr → SetTheory.PGame
      ⊢ SetTheory.PGame
    -/
    have y := mk yl yr yL yR
    /-
      case mk.mk
      xl xr : Type u
      a✝¹ : xl → SetTheory.PGame
      a✝ : xr → SetTheory.PGame
      IHxl : xl → SetTheory.PGame → SetTheory.PGame
      IHxr : xr → SetTheory.PGame → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      IHyl : yl → SetTheory.PGame
      IHyr : yr → SetTheory.PGame
      y : SetTheory.PGame
      ⊢ SetTheory.PGame
    -/
    refine ⟨(xl × yl) ⊕ (xr × yr), (xl × yr) ⊕ (xr × yl), ?_, ?_⟩ <;> rintro (⟨i, j⟩ | ⟨i, j⟩)
      /-
        case mk.mk.refine_1.inl.mk
        xl xr : Type u
        a✝¹ : xl → SetTheory.PGame
        a✝ : xr → SetTheory.PGame
        IHxl : xl → SetTheory.PGame → SetTheory.PGame
        IHxr : xr → SetTheory.PGame → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        IHyl : yl → SetTheory.PGame
        IHyr : yr → SetTheory.PGame
        y : SetTheory.PGame
        i : xl
        j : yl
        ⊢ SetTheory.PGame
      -/
    · exact IHxl i y + IHyl j - IHxl i (yL j)
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.refine_1.inr.mk
        xl xr : Type u
        a✝¹ : xl → SetTheory.PGame
        a✝ : xr → SetTheory.PGame
        IHxl : xl → SetTheory.PGame → SetTheory.PGame
        IHxr : xr → SetTheory.PGame → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        IHyl : yl → SetTheory.PGame
        IHyr : yr → SetTheory.PGame
        y : SetTheory.PGame
        i : xr
        j : yr
        ⊢ SetTheory.PGame
      -/
    · exact IHxr i y + IHyr j - IHxr i (yR j)
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.refine_2.inl.mk
        xl xr : Type u
        a✝¹ : xl → SetTheory.PGame
        a✝ : xr → SetTheory.PGame
        IHxl : xl → SetTheory.PGame → SetTheory.PGame
        IHxr : xr → SetTheory.PGame → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        IHyl : yl → SetTheory.PGame
        IHyr : yr → SetTheory.PGame
        y : SetTheory.PGame
        i : xl
        j : yr
        ⊢ SetTheory.PGame
      -/
    · exact IHxl i y + IHyr j - IHxl i (yR j)
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.refine_2.inr.mk
        xl xr : Type u
        a✝¹ : xl → SetTheory.PGame
        a✝ : xr → SetTheory.PGame
        IHxl : xl → SetTheory.PGame → SetTheory.PGame
        IHxr : xr → SetTheory.PGame → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        IHyl : yl → SetTheory.PGame
        IHyr : yr → SetTheory.PGame
        y : SetTheory.PGame
        i : xr
        j : yl
        ⊢ SetTheory.PGame
      -/
    · exact IHxr i y + IHyl j - IHxr i (yL j)⟩
      /-
        🎉 no goals
      -/


theorem leftMoves_mul :
    ∀ x y : PGame.{u},
      (x * y).LeftMoves = (x.LeftMoves × y.LeftMoves ⊕ x.RightMoves × y.RightMoves)
  | ⟨_, _, _, _⟩, ⟨_, _, _, _⟩ => rfl


theorem rightMoves_mul :
    ∀ x y : PGame.{u},
      (x * y).RightMoves = (x.LeftMoves × y.RightMoves ⊕ x.RightMoves × y.LeftMoves)
  | ⟨_, _, _, _⟩, ⟨_, _, _, _⟩ => rfl


/-- Turns two left or right moves for `x` and `y` into a left move for `x * y` and vice versa.

Even though these types are the same (not definitionally so), this is the preferred way to convert
between them. -/
def toLeftMovesMul {x y : PGame} :
    (x.LeftMoves × y.LeftMoves) ⊕ (x.RightMoves × y.RightMoves) ≃ (x * y).LeftMoves :=
  Equiv.cast (leftMoves_mul x y).symm


/-- Turns a left and a right move for `x` and `y` into a right move for `x * y` and vice versa.

Even though these types are the same (not definitionally so), this is the preferred way to convert
between them. -/
def toRightMovesMul {x y : PGame} :
    (x.LeftMoves × y.RightMoves) ⊕ (x.RightMoves × y.LeftMoves) ≃ (x * y).RightMoves :=
  Equiv.cast (rightMoves_mul x y).symm


@[simp]
theorem mk_mul_moveLeft_inl {xl xr yl yr} {xL xR yL yR} {i j} :
    (mk xl xr xL xR * mk yl yr yL yR).moveLeft (Sum.inl (i, j)) =
      xL i * mk yl yr yL yR + mk xl xr xL xR * yL j - xL i * yL j :=
  rfl


@[simp]
theorem mul_moveLeft_inl {x y : PGame} {i j} :
    (x * y).moveLeft (toLeftMovesMul (Sum.inl (i, j))) =
      x.moveLeft i * y + x * y.moveLeft j - x.moveLeft i * y.moveLeft j := by
  /-
    x y : SetTheory.PGame
    i : x.LeftMoves
    j : y.LeftMoves
    ⊢ Eq ((HMul.hMul x y).moveLeft (SetTheory.PGame.toLeftMovesMul (Sum.inl { fst  …
  -/
  cases x
  /-
    case mk
    y : SetTheory.PGame
    j : y.LeftMoves
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    i : (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).LeftMoves
    ⊢ Eq ((HMul.hMul (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝) y).moveLeft (SetTheory.PGam …
  -/
  cases y
  /-
    case mk.mk
    α✝¹ β✝¹ : Type u_1
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    i : (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²).LeftMoves
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    j : (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).LeftMoves
    ⊢ Eq ((HMul.hMul (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²) (SetTheory.PGame.mk α✝ β …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_mul_moveLeft_inr {xl xr yl yr} {xL xR yL yR} {i j} :
    (mk xl xr xL xR * mk yl yr yL yR).moveLeft (Sum.inr (i, j)) =
      xR i * mk yl yr yL yR + mk xl xr xL xR * yR j - xR i * yR j :=
  rfl


@[simp]
theorem mul_moveLeft_inr {x y : PGame} {i j} :
    (x * y).moveLeft (toLeftMovesMul (Sum.inr (i, j))) =
      x.moveRight i * y + x * y.moveRight j - x.moveRight i * y.moveRight j := by
  /-
    x y : SetTheory.PGame
    i : x.RightMoves
    j : y.RightMoves
    ⊢ Eq ((HMul.hMul x y).moveLeft (SetTheory.PGame.toLeftMovesMul (Sum.inr { fst  …
  -/
  cases x
  /-
    case mk
    y : SetTheory.PGame
    j : y.RightMoves
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    i : (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).RightMoves
    ⊢ Eq ((HMul.hMul (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝) y).moveLeft (SetTheory.PGam …
  -/
  cases y
  /-
    case mk.mk
    α✝¹ β✝¹ : Type u_1
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    i : (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²).RightMoves
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    j : (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).RightMoves
    ⊢ Eq ((HMul.hMul (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²) (SetTheory.PGame.mk α✝ β …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_mul_moveRight_inl {xl xr yl yr} {xL xR yL yR} {i j} :
    (mk xl xr xL xR * mk yl yr yL yR).moveRight (Sum.inl (i, j)) =
      xL i * mk yl yr yL yR + mk xl xr xL xR * yR j - xL i * yR j :=
  rfl


@[simp]
theorem mul_moveRight_inl {x y : PGame} {i j} :
    (x * y).moveRight (toRightMovesMul (Sum.inl (i, j))) =
      x.moveLeft i * y + x * y.moveRight j - x.moveLeft i * y.moveRight j := by
  /-
    x y : SetTheory.PGame
    i : x.LeftMoves
    j : y.RightMoves
    ⊢ Eq ((HMul.hMul x y).moveRight (SetTheory.PGame.toRightMovesMul (Sum.inl { fs …
  -/
  cases x
  /-
    case mk
    y : SetTheory.PGame
    j : y.RightMoves
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    i : (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).LeftMoves
    ⊢ Eq ((HMul.hMul (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝) y).moveRight (SetTheory.PGa …
  -/
  cases y
  /-
    case mk.mk
    α✝¹ β✝¹ : Type u_1
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    i : (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²).LeftMoves
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    j : (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).RightMoves
    ⊢ Eq ((HMul.hMul (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²) (SetTheory.PGame.mk α✝ β …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_mul_moveRight_inr {xl xr yl yr} {xL xR yL yR} {i j} :
    (mk xl xr xL xR * mk yl yr yL yR).moveRight (Sum.inr (i, j)) =
      xR i * mk yl yr yL yR + mk xl xr xL xR * yL j - xR i * yL j :=
  rfl


@[simp]
theorem mul_moveRight_inr {x y : PGame} {i j} :
    (x * y).moveRight (toRightMovesMul (Sum.inr (i, j))) =
      x.moveRight i * y + x * y.moveLeft j - x.moveRight i * y.moveLeft j := by
  /-
    x y : SetTheory.PGame
    i : x.RightMoves
    j : y.LeftMoves
    ⊢ Eq ((HMul.hMul x y).moveRight (SetTheory.PGame.toRightMovesMul (Sum.inr { fs …
  -/
  cases x
  /-
    case mk
    y : SetTheory.PGame
    j : y.LeftMoves
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    i : (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).RightMoves
    ⊢ Eq ((HMul.hMul (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝) y).moveRight (SetTheory.PGa …
  -/
  cases y
  /-
    case mk.mk
    α✝¹ β✝¹ : Type u_1
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    i : (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²).RightMoves
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    j : (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).LeftMoves
    ⊢ Eq ((HMul.hMul (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²) (SetTheory.PGame.mk α✝ β …
  -/
  rfl
  /-
    🎉 no goals
  -/

-- @[simp] -- Porting note: simpNF linter complains

theorem neg_mk_mul_moveLeft_inl {xl xr yl yr} {xL xR yL yR} {i j} :
    (-(mk xl xr xL xR * mk yl yr yL yR)).moveLeft (Sum.inl (i, j)) =
      -(xL i * mk yl yr yL yR + mk xl xr xL xR * yR j - xL i * yR j) :=
  rfl

-- @[simp] -- Porting note: simpNF linter complains

theorem neg_mk_mul_moveLeft_inr {xl xr yl yr} {xL xR yL yR} {i j} :
    (-(mk xl xr xL xR * mk yl yr yL yR)).moveLeft (Sum.inr (i, j)) =
      -(xR i * mk yl yr yL yR + mk xl xr xL xR * yL j - xR i * yL j) :=
  rfl

-- @[simp] -- Porting note: simpNF linter complains

theorem neg_mk_mul_moveRight_inl {xl xr yl yr} {xL xR yL yR} {i j} :
    (-(mk xl xr xL xR * mk yl yr yL yR)).moveRight (Sum.inl (i, j)) =
      -(xL i * mk yl yr yL yR + mk xl xr xL xR * yL j - xL i * yL j) :=
  rfl

-- @[simp] -- Porting note: simpNF linter complains

theorem neg_mk_mul_moveRight_inr {xl xr yl yr} {xL xR yL yR} {i j} :
    (-(mk xl xr xL xR * mk yl yr yL yR)).moveRight (Sum.inr (i, j)) =
      -(xR i * mk yl yr yL yR + mk xl xr xL xR * yR j - xR i * yR j) :=
  rfl


theorem leftMoves_mul_cases {x y : PGame} (k) {P : (x * y).LeftMoves → Prop}
    (hl : ∀ ix iy, P <| toLeftMovesMul (Sum.inl ⟨ix, iy⟩))
    (hr : ∀ jx jy, P <| toLeftMovesMul (Sum.inr ⟨jx, jy⟩)) : P k := by
  /-
    x y : SetTheory.PGame
    k : (HMul.hMul x y).LeftMoves
    P : (HMul.hMul x y).LeftMoves → Prop
    hl : ∀ (ix : x.LeftMoves) (iy : y.LeftMoves), P (SetTheory.PGame.toLeftMovesMu …
    hr : ∀ (jx : x.RightMoves) (jy : y.RightMoves), P (SetTheory.PGame.toLeftMoves …
    ⊢ P k
  -/
  rw [← toLeftMovesMul.apply_symm_apply k]
  /-
    x y : SetTheory.PGame
    k : (HMul.hMul x y).LeftMoves
    P : (HMul.hMul x y).LeftMoves → Prop
    hl : ∀ (ix : x.LeftMoves) (iy : y.LeftMoves), P (SetTheory.PGame.toLeftMovesMu …
    hr : ∀ (jx : x.RightMoves) (jy : y.RightMoves), P (SetTheory.PGame.toLeftMoves …
    ⊢ P (SetTheory.PGame.toLeftMovesMul (SetTheory.PGame.toLeftMovesMul.symm k))
  -/
  rcases toLeftMovesMul.symm k with (⟨ix, iy⟩ | ⟨jx, jy⟩)
    /-
      case inl.mk
      x y : SetTheory.PGame
      k : (HMul.hMul x y).LeftMoves
      P : (HMul.hMul x y).LeftMoves → Prop
      hl : ∀ (ix : x.LeftMoves) (iy : y.LeftMoves), P (SetTheory.PGame.toLeftMovesMu …
      hr : ∀ (jx : x.RightMoves) (jy : y.RightMoves), P (SetTheory.PGame.toLeftMoves …
      ix : x.LeftMoves
      iy : y.LeftMoves
      ⊢ P (SetTheory.PGame.toLeftMovesMul (Sum.inl { fst := ix, snd := iy }))
    -/
  · apply hl
    /-
      🎉 no goals
    -/
    /-
      case inr.mk
      x y : SetTheory.PGame
      k : (HMul.hMul x y).LeftMoves
      P : (HMul.hMul x y).LeftMoves → Prop
      hl : ∀ (ix : x.LeftMoves) (iy : y.LeftMoves), P (SetTheory.PGame.toLeftMovesMu …
      hr : ∀ (jx : x.RightMoves) (jy : y.RightMoves), P (SetTheory.PGame.toLeftMoves …
      jx : x.RightMoves
      jy : y.RightMoves
      ⊢ P (SetTheory.PGame.toLeftMovesMul (Sum.inr { fst := jx, snd := jy }))
    -/
  · apply hr
    /-
      🎉 no goals
    -/


theorem rightMoves_mul_cases {x y : PGame} (k) {P : (x * y).RightMoves → Prop}
    (hl : ∀ ix jy, P <| toRightMovesMul (Sum.inl ⟨ix, jy⟩))
    (hr : ∀ jx iy, P <| toRightMovesMul (Sum.inr ⟨jx, iy⟩)) : P k := by
  /-
    x y : SetTheory.PGame
    k : (HMul.hMul x y).RightMoves
    P : (HMul.hMul x y).RightMoves → Prop
    hl : ∀ (ix : x.LeftMoves) (jy : y.RightMoves), P (SetTheory.PGame.toRightMoves …
    hr : ∀ (jx : x.RightMoves) (iy : y.LeftMoves), P (SetTheory.PGame.toRightMoves …
    ⊢ P k
  -/
  rw [← toRightMovesMul.apply_symm_apply k]
  /-
    x y : SetTheory.PGame
    k : (HMul.hMul x y).RightMoves
    P : (HMul.hMul x y).RightMoves → Prop
    hl : ∀ (ix : x.LeftMoves) (jy : y.RightMoves), P (SetTheory.PGame.toRightMoves …
    hr : ∀ (jx : x.RightMoves) (iy : y.LeftMoves), P (SetTheory.PGame.toRightMoves …
    ⊢ P (SetTheory.PGame.toRightMovesMul (SetTheory.PGame.toRightMovesMul.symm k))
  -/
  rcases toRightMovesMul.symm k with (⟨ix, iy⟩ | ⟨jx, jy⟩)
    /-
      case inl.mk
      x y : SetTheory.PGame
      k : (HMul.hMul x y).RightMoves
      P : (HMul.hMul x y).RightMoves → Prop
      hl : ∀ (ix : x.LeftMoves) (jy : y.RightMoves), P (SetTheory.PGame.toRightMoves …
      hr : ∀ (jx : x.RightMoves) (iy : y.LeftMoves), P (SetTheory.PGame.toRightMoves …
      ix : x.LeftMoves
      iy : y.RightMoves
      ⊢ P (SetTheory.PGame.toRightMovesMul (Sum.inl { fst := ix, snd := iy }))
    -/
  · apply hl
    /-
      🎉 no goals
    -/
    /-
      case inr.mk
      x y : SetTheory.PGame
      k : (HMul.hMul x y).RightMoves
      P : (HMul.hMul x y).RightMoves → Prop
      hl : ∀ (ix : x.LeftMoves) (jy : y.RightMoves), P (SetTheory.PGame.toRightMoves …
      hr : ∀ (jx : x.RightMoves) (iy : y.LeftMoves), P (SetTheory.PGame.toRightMoves …
      jx : x.RightMoves
      jy : y.LeftMoves
      ⊢ P (SetTheory.PGame.toRightMovesMul (Sum.inr { fst := jx, snd := jy }))
    -/
  · apply hr
    /-
      🎉 no goals
    -/


/-- `x * y` and `y * x` have the same moves. -/
def mulCommRelabelling (x y : PGame.{u}) : x * y ≡r y * x :=
  match x, y with
  | ⟨xl, xr, xL, xR⟩, ⟨yl, yr, yL, yR⟩ => by
    refine ⟨Equiv.sumCongr (Equiv.prodComm _ _) (Equiv.prodComm _ _),
      (Equiv.sumComm _ _).trans (Equiv.sumCongr (Equiv.prodComm _ _) (Equiv.prodComm _ _)), ?_, ?_⟩
      <;>
    /-
      case refine_1
      x y : SetTheory.PGame
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ (i : (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk yl yr y …
    -/
    rintro (⟨i, j⟩ | ⟨i, j⟩) <;>
    { dsimp
      exact ((addCommRelabelling _ _).trans <|
        (mulCommRelabelling _ _).addCongr (mulCommRelabelling _ _)).subCongr
        (mulCommRelabelling _ _) }
  termination_by (x, y)


theorem quot_mul_comm (x y : PGame.{u}) : (⟦x * y⟧ : Game) = ⟦y * x⟧ :=
  game_eq (mulCommRelabelling x y).equiv


/-- `x * y` is equivalent to `y * x`. -/
theorem mul_comm_equiv (x y : PGame) : x * y ≈ y * x :=
  Quotient.exact <| quot_mul_comm _ _


instance isEmpty_leftMoves_mul (x y : PGame.{u})
    [IsEmpty (x.LeftMoves × y.LeftMoves ⊕ x.RightMoves × y.RightMoves)] :
    IsEmpty (x * y).LeftMoves := by
  /-
    x y : SetTheory.PGame
    inst✝ : IsEmpty (Sum (Prod x.LeftMoves y.LeftMoves) (Prod x.RightMoves y.Right …
    ⊢ IsEmpty (HMul.hMul x y).LeftMoves
  -/
  cases x
  /-
    case mk
    y : SetTheory.PGame
    α✝ β✝ : Type u
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    inst✝ : IsEmpty (Sum (Prod (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).LeftMoves y.LeftM …
    ⊢ IsEmpty (HMul.hMul (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝) y).LeftMoves
  -/
  cases y
  /-
    case mk.mk
    α✝¹ β✝¹ : Type u
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    α✝ β✝ : Type u
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    inst✝ : IsEmpty (Sum (Prod (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²).LeftMoves (Set …
    ⊢ IsEmpty (HMul.hMul (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²) (SetTheory.PGame.mk  …
  -/
  assumption
  /-
    🎉 no goals
  -/


instance isEmpty_rightMoves_mul (x y : PGame.{u})
    [IsEmpty (x.LeftMoves × y.RightMoves ⊕ x.RightMoves × y.LeftMoves)] :
    IsEmpty (x * y).RightMoves := by
  /-
    x y : SetTheory.PGame
    inst✝ : IsEmpty (Sum (Prod x.LeftMoves y.RightMoves) (Prod x.RightMoves y.Left …
    ⊢ IsEmpty (HMul.hMul x y).RightMoves
  -/
  cases x
  /-
    case mk
    y : SetTheory.PGame
    α✝ β✝ : Type u
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    inst✝ : IsEmpty (Sum (Prod (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).LeftMoves y.Right …
    ⊢ IsEmpty (HMul.hMul (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝) y).RightMoves
  -/
  cases y
  /-
    case mk.mk
    α✝¹ β✝¹ : Type u
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    α✝ β✝ : Type u
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    inst✝ : IsEmpty (Sum (Prod (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²).LeftMoves (Set …
    ⊢ IsEmpty (HMul.hMul (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²) (SetTheory.PGame.mk  …
  -/
  assumption
  /-
    🎉 no goals
  -/


/-- `x * 0` has exactly the same moves as `0`. -/
def mulZeroRelabelling (x : PGame) : x * 0 ≡r 0 :=
  Relabelling.isEmpty _


/-- `x * 0` is equivalent to `0`. -/
theorem mul_zero_equiv (x : PGame) : x * 0 ≈ 0 :=
  (mulZeroRelabelling x).equiv


@[simp]
theorem quot_mul_zero (x : PGame) : (⟦x * 0⟧ : Game) = 0 :=
  game_eq x.mul_zero_equiv


/-- `0 * x` has exactly the same moves as `0`. -/
def zeroMulRelabelling (x : PGame) : 0 * x ≡r 0 :=
  Relabelling.isEmpty _


/-- `0 * x` is equivalent to `0`. -/
theorem zero_mul_equiv (x : PGame) : 0 * x ≈ 0 :=
  (zeroMulRelabelling x).equiv


@[simp]
theorem quot_zero_mul (x : PGame) : (⟦0 * x⟧ : Game) = 0 :=
  game_eq x.zero_mul_equiv


/-- `-x * y` and `-(x * y)` have the same moves. -/
def negMulRelabelling (x y : PGame.{u}) : -x * y ≡r -(x * y) :=
  match x, y with
  | ⟨xl, xr, xL, xR⟩, ⟨yl, yr, yL, yR⟩ => by
      /-
        x y : SetTheory.PGame
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        ⊢ (HMul.hMul (Neg.neg (SetTheory.PGame.mk xl xr xL xR)) (SetTheory.PGame.mk yl …
      -/
      refine ⟨Equiv.sumComm _ _, Equiv.sumComm _ _, ?_, ?_⟩ <;>
      /-
        case refine_1
        x y : SetTheory.PGame
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        ⊢ (i : (HMul.hMul (Neg.neg (SetTheory.PGame.mk xl xr xL xR)) (SetTheory.PGame. …
      -/
      rintro (⟨i, j⟩ | ⟨i, j⟩) <;>
        /-
          case refine_1.inl.mk
          x y : SetTheory.PGame
          xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          i : xr
          j : yl
          ⊢ ((HMul.hMul (Neg.neg (SetTheory.PGame.mk xl xr xL xR)) (SetTheory.PGame.mk y …
        -/
        /-
          case refine_1.inl.mk
          x y : SetTheory.PGame
          xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          i : xr
          j : yl
          ⊢ (HSub.hSub (HAdd.hAdd (HMul.hMul (Neg.neg (xR i)) (SetTheory.PGame.mk yl yr  …
        -/
        /-
          x y : SetTheory.PGame
          xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          i : xr
          j : yl
          ⊢ (HAdd.hAdd (Neg.neg (HAdd.hAdd ((fun a => SetTheory.PGame.rec (motive := fun …
        -/
        apply ((negAddRelabelling _ _).trans _).symm
          /-
            x y : SetTheory.PGame
            xl xr : Type u
            xL : xl → SetTheory.PGame
            xR : xr → SetTheory.PGame
            yl yr : Type u
            yL : yl → SetTheory.PGame
            yR : yr → SetTheory.PGame
            i : xr
            j : yl
            ⊢ (Neg.neg ((fun a => SetTheory.PGame.rec (motive := fun x => SetTheory.PGame  …
          -/
          /-
            🎉 no goals
          -/
          /-
            x y : SetTheory.PGame
            xl xr : Type u
            xL : xl → SetTheory.PGame
            xR : xr → SetTheory.PGame
            yl yr : Type u
            yL : yl → SetTheory.PGame
            yR : yr → SetTheory.PGame
            i : xr
            j : yl
            ⊢ (Neg.neg ((fun a => SetTheory.PGame.rec (motive := fun x => SetTheory.PGame  …
          -/
          /-
            🎉 no goals
          -/
          /-
            x y : SetTheory.PGame
            xl xr : Type u
            xL : xl → SetTheory.PGame
            xR : xr → SetTheory.PGame
            yl yr : Type u
            yL : yl → SetTheory.PGame
            yR : yr → SetTheory.PGame
            i : xl
            j : yr
            ⊢ (Neg.neg ((fun a => SetTheory.PGame.rec (motive := fun x => SetTheory.PGame  …
          -/
          /-
            🎉 no goals
          -/
          /-
            x y : SetTheory.PGame
            xl xr : Type u
            xL : xl → SetTheory.PGame
            xR : xr → SetTheory.PGame
            yl yr : Type u
            yL : yl → SetTheory.PGame
            yR : yr → SetTheory.PGame
            i : xr
            j : yr
            ⊢ (Neg.neg ((fun a => SetTheory.PGame.rec (motive := fun x => SetTheory.PGame  …
          -/
        /-
          x y : SetTheory.PGame
          xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          i : xr
          j : yl
          ⊢ (Neg.neg ((fun a => SetTheory.PGame.rec (fun yl yr yL yR IHyl IHyr => letFun …
        -/
        /-
          x y : SetTheory.PGame
          xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          i : xr
          j : yl
          ⊢ (Neg.neg (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (yL j))).Relabelling (H …
        -/
        /-
          🎉 no goals
        -/
        /-
          x y : SetTheory.PGame
          xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          i : xl
          j : yr
          ⊢ (Neg.neg ((fun a => SetTheory.PGame.rec (fun yl yr yL yR IHyl IHyr => letFun …
        -/
        /-
          x y : SetTheory.PGame
          xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          i : xl
          j : yr
          ⊢ (Neg.neg (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (yR j))).Relabelling (H …
        -/
        /-
          🎉 no goals
        -/
        /-
          x y : SetTheory.PGame
          xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          i : xr
          j : yr
          ⊢ (Neg.neg ((fun a => SetTheory.PGame.rec (fun yl yr yL yR IHyl IHyr => letFun …
        -/
        /-
          x y : SetTheory.PGame
          xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          i : xr
          j : yr
          ⊢ (Neg.neg (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (yR j))).Relabelling (H …
        -/
        /-
          🎉 no goals
        -/
        · exact (negMulRelabelling _ _).symm
          /-
            🎉 no goals
          -/
        -- Porting note: not sure what has gone wrong here.
        -- The goal is hideous here, and the `exact` doesn't work,
        -- but if we just `change` it to look like the mathlib3 goal then we're fine!?
        /-
          x y : SetTheory.PGame
          xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          i : xl
          j : yl
          ⊢ (Neg.neg ((fun a => SetTheory.PGame.rec (fun yl yr yL yR IHyl IHyr => letFun …
        -/
        change -(mk xl xr xL xR * _) ≡r _
        /-
          x y : SetTheory.PGame
          xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          i : xl
          j : yl
          ⊢ (Neg.neg (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (yL j))).Relabelling (H …
        -/
        exact (negMulRelabelling _ _).symm
        /-
          🎉 no goals
        -/
  termination_by (x, y)


@[simp]
theorem quot_neg_mul (x y : PGame) : (⟦-x * y⟧ : Game) = -⟦x * y⟧ :=
  game_eq (negMulRelabelling x y).equiv


/-- `x * -y` and `-(x * y)` have the same moves. -/
def mulNegRelabelling (x y : PGame) : x * -y ≡r -(x * y) :=
  (mulCommRelabelling x _).trans <| (negMulRelabelling _ x).trans (mulCommRelabelling y x).negCongr


@[simp]
theorem quot_mul_neg (x y : PGame) : ⟦x * -y⟧ = (-⟦x * y⟧ : Game) :=
  game_eq (mulNegRelabelling x y).equiv


                                                                            /-
                                                                              x y : SetTheory.PGame
                                                                              ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (Neg.neg x) (Neg.neg y)))  …
                                                                            -/
theorem quot_neg_mul_neg (x y : PGame) : ⟦-x * -y⟧ = (⟦x * y⟧ : Game) := by simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp]
theorem quot_left_distrib (x y z : PGame) : (⟦x * (y + z)⟧ : Game) = ⟦x * y⟧ + ⟦x * z⟧ :=
  match x, y, z with
  | mk xl xr xL xR, mk yl yr yL yR, mk zl zr zL zR => by
    /-
      x y z : SetTheory.PGame
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      zl zr : Type u_1
      zL : zl → SetTheory.PGame
      zR : zr → SetTheory.PGame
      ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (SetTheory.PGame.mk xl xr  …
    -/
    let x := mk xl xr xL xR
    /-
      x✝ y z : SetTheory.PGame
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      zl zr : Type u_1
      zL : zl → SetTheory.PGame
      zR : zr → SetTheory.PGame
      x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
      ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (SetTheory.PGame.mk xl xr  …
    -/
    let y := mk yl yr yL yR
    /-
      x✝ y✝ z : SetTheory.PGame
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      zl zr : Type u_1
      zL : zl → SetTheory.PGame
      zR : zr → SetTheory.PGame
      x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
      y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
      ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (SetTheory.PGame.mk xl xr  …
    -/
    let z := mk zl zr zL zR
    /-
      x✝ y✝ z✝ : SetTheory.PGame
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      zl zr : Type u_1
      zL : zl → SetTheory.PGame
      zR : zr → SetTheory.PGame
      x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
      y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
      z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
      ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (SetTheory.PGame.mk xl xr  …
    -/
    refine quot_eq_of_mk'_quot_eq ?_ ?_ ?_ ?_
      /-
        case refine_1
        x✝ y✝ z✝ : SetTheory.PGame
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u_1
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
        y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
        z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
        ⊢ _root_.Equiv (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (HAdd.hAdd (SetTheo …
      -/
    · fconstructor
        /-
          case refine_1.toFun
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          ⊢ (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (HAdd.hAdd (SetTheory.PGame.mk y …
        -/
      · rintro (⟨_, _ | _⟩ | ⟨_, _ | _⟩) <;>
          -- Porting note: we've increased `maxDepth` here from `5` to `6`.
          -- Likely this sort of off-by-one error is just a change in the implementation
          -- of `solve_by_elim`.
          /-
            case refine_1.toFun.inl.mk.inl
            x✝ y✝ z✝ : SetTheory.PGame
            xl xr : Type u_1
            xL : xl → SetTheory.PGame
            xR : xr → SetTheory.PGame
            yl yr : Type u_1
            yL : yl → SetTheory.PGame
            yR : yr → SetTheory.PGame
            zl zr : Type u_1
            zL : zl → SetTheory.PGame
            zR : zr → SetTheory.PGame
            x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
            y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
            z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
            fst✝ : xl
            val✝ : yl
            ⊢ (HAdd.hAdd (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk y …
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          solve_by_elim (config := { maxDepth := 6 }) [Sum.inl, Sum.inr, Prod.mk]
          /-
            🎉 no goals
          -/
        /-
          case refine_1.invFun
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          ⊢ (HAdd.hAdd (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk y …
        -/
      · rintro (⟨⟨_, _⟩ | ⟨_, _⟩⟩ | ⟨_, _⟩ | ⟨_, _⟩) <;>
          /-
            case refine_1.invFun.inl.inl.mk
            x✝ y✝ z✝ : SetTheory.PGame
            xl xr : Type u_1
            xL : xl → SetTheory.PGame
            xR : xr → SetTheory.PGame
            yl yr : Type u_1
            yL : yl → SetTheory.PGame
            yR : yr → SetTheory.PGame
            zl zr : Type u_1
            zL : zl → SetTheory.PGame
            zR : zr → SetTheory.PGame
            x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
            y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
            z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
            fst✝ : xl
            snd✝ : yl
            ⊢ (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (HAdd.hAdd (SetTheory.PGame.mk y …
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          solve_by_elim (config := { maxDepth := 6 }) [Sum.inl, Sum.inr, Prod.mk]
          /-
            🎉 no goals
          -/
        /-
          case refine_1.left_inv
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          ⊢ Function.LeftInverse (fun a => Sum.casesOn a (fun val => Sum.casesOn val (fu …
        -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
      · rintro (⟨_, _ | _⟩ | ⟨_, _ | _⟩) <;> rfl
                                             /-
                                               🎉 no goals
                                             -/
        /-
          case refine_1.right_inv
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          ⊢ Function.RightInverse (fun a => Sum.casesOn a (fun val => Sum.casesOn val (f …
        -/
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
      · rintro (⟨⟨_, _⟩ | ⟨_, _⟩⟩ | ⟨_, _⟩ | ⟨_, _⟩) <;> rfl
                                                         /-
                                                           🎉 no goals
                                                         -/
      /-
        case refine_2
        x✝ y✝ z✝ : SetTheory.PGame
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u_1
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
        y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
        z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
        ⊢ _root_.Equiv (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (HAdd.hAdd (SetTheo …
      -/
    · fconstructor
        /-
          case refine_2.toFun
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          ⊢ (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (HAdd.hAdd (SetTheory.PGame.mk y …
        -/
      · rintro (⟨_, _ | _⟩ | ⟨_, _ | _⟩) <;>
          /-
            case refine_2.toFun.inl.mk.inl
            x✝ y✝ z✝ : SetTheory.PGame
            xl xr : Type u_1
            xL : xl → SetTheory.PGame
            xR : xr → SetTheory.PGame
            yl yr : Type u_1
            yL : yl → SetTheory.PGame
            yR : yr → SetTheory.PGame
            zl zr : Type u_1
            zL : zl → SetTheory.PGame
            zR : zr → SetTheory.PGame
            x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
            y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
            z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
            fst✝ : xl
            val✝ : yr
            ⊢ (HAdd.hAdd (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk y …
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          solve_by_elim (config := { maxDepth := 6 }) [Sum.inl, Sum.inr, Prod.mk]
          /-
            🎉 no goals
          -/
        /-
          case refine_2.invFun
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          ⊢ (HAdd.hAdd (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk y …
        -/
      · rintro (⟨⟨_, _⟩ | ⟨_, _⟩⟩ | ⟨_, _⟩ | ⟨_, _⟩) <;>
          /-
            case refine_2.invFun.inl.inl.mk
            x✝ y✝ z✝ : SetTheory.PGame
            xl xr : Type u_1
            xL : xl → SetTheory.PGame
            xR : xr → SetTheory.PGame
            yl yr : Type u_1
            yL : yl → SetTheory.PGame
            yR : yr → SetTheory.PGame
            zl zr : Type u_1
            zL : zl → SetTheory.PGame
            zR : zr → SetTheory.PGame
            x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
            y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
            z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
            fst✝ : xl
            snd✝ : yr
            ⊢ (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (HAdd.hAdd (SetTheory.PGame.mk y …
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          solve_by_elim (config := { maxDepth := 6 }) [Sum.inl, Sum.inr, Prod.mk]
          /-
            🎉 no goals
          -/
        /-
          case refine_2.left_inv
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          ⊢ Function.LeftInverse (fun a => Sum.casesOn a (fun val => Sum.casesOn val (fu …
        -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
      · rintro (⟨_, _ | _⟩ | ⟨_, _ | _⟩) <;> rfl
                                             /-
                                               🎉 no goals
                                             -/
        /-
          case refine_2.right_inv
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          ⊢ Function.RightInverse (fun a => Sum.casesOn a (fun val => Sum.casesOn val (f …
        -/
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
      · rintro (⟨⟨_, _⟩ | ⟨_, _⟩⟩ | ⟨_, _⟩ | ⟨_, _⟩) <;> rfl
                                                         /-
                                                           🎉 no goals
                                                         -/
    -- Porting note: explicitly wrote out arguments to each recursive
    -- quot_left_distrib reference below, because otherwise the decreasing_by block
    -- failed. Previously, each branch ended with: `simp [quot_left_distrib]; abel`
    -- See https://github.com/leanprover/lean4/issues/2288
      /-
        case refine_3
        x✝ y✝ z✝ : SetTheory.PGame
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u_1
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
        y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
        z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
        ⊢ ∀ (i : (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (HAdd.hAdd (SetTheory.PGa …
      -/
    · rintro (⟨i, j | k⟩ | ⟨i, j | k⟩)
      · change
          ⟦xL i * (y + z) + x * (yL j + z) - xL i * (yL j + z)⟧ =
            ⟦xL i * y + x * yL j - xL i * yL j + x * z⟧
        /-
          case refine_3.inl.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          j : yl
          ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HSub.hSub (HAdd.hAdd (HMul.hMul (xL  …
        -/
        simp only [quot_sub, quot_add]
        /-
          case refine_3.inl.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (xL  …
        -/
        rw [quot_left_distrib (xL i) (mk yl yr yL yR) (mk zl zr zL zR)]
        /-
          case refine_3.inl.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        rw [quot_left_distrib (mk xl xr xL xR) (yL j) (mk zl zr zL zR)]
        /-
          case refine_3.inl.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        rw [quot_left_distrib (xL i) (yL j) (mk zl zr zL zR)]
        /-
          case refine_3.inl.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
      · change
          ⟦xL i * (y + z) + x * (y + zL k) - xL i * (y + zL k)⟧ =
            ⟦x * y + (xL i * z + x * zL k - xL i * zL k)⟧
        /-
          case refine_3.inl.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          k : zl
          ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HSub.hSub (HAdd.hAdd (HMul.hMul (xL  …
        -/
        simp only [quot_sub, quot_add]
        /-
          case refine_3.inl.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          k : zl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (xL  …
        -/
        rw [quot_left_distrib (xL i) (mk yl yr yL yR) (mk zl zr zL zR)]
        /-
          case refine_3.inl.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          k : zl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        rw [quot_left_distrib (mk xl xr xL xR) (mk yl yr yL yR) (zL k)]
        /-
          case refine_3.inl.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          k : zl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        rw [quot_left_distrib (xL i) (mk yl yr yL yR) (zL k)]
        /-
          case refine_3.inl.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          k : zl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
      · change
          ⟦xR i * (y + z) + x * (yR j + z) - xR i * (yR j + z)⟧ =
            ⟦xR i * y + x * yR j - xR i * yR j + x * z⟧
        /-
          case refine_3.inr.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          j : yr
          ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HSub.hSub (HAdd.hAdd (HMul.hMul (xR  …
        -/
        simp only [quot_sub, quot_add]
        /-
          case refine_3.inr.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (xR  …
        -/
        rw [quot_left_distrib (xR i) (mk yl yr yL yR) (mk zl zr zL zR)]
        /-
          case refine_3.inr.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        rw [quot_left_distrib (mk xl xr xL xR) (yR j) (mk zl zr zL zR)]
        /-
          case refine_3.inr.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        rw [quot_left_distrib (xR i) (yR j) (mk zl zr zL zR)]
        /-
          case refine_3.inr.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
      · change
          ⟦xR i * (y + z) + x * (y + zR k) - xR i * (y + zR k)⟧ =
            ⟦x * y + (xR i * z + x * zR k - xR i * zR k)⟧
        /-
          case refine_3.inr.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          k : zr
          ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HSub.hSub (HAdd.hAdd (HMul.hMul (xR  …
        -/
        simp only [quot_sub, quot_add]
        /-
          case refine_3.inr.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          k : zr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (xR  …
        -/
        rw [quot_left_distrib (xR i) (mk yl yr yL yR) (mk zl zr zL zR)]
        /-
          case refine_3.inr.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          k : zr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        rw [quot_left_distrib (mk xl xr xL xR) (mk yl yr yL yR) (zR k)]
        /-
          case refine_3.inr.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          k : zr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        rw [quot_left_distrib (xR i) (mk yl yr yL yR) (zR k)]
        /-
          case refine_3.inr.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          k : zr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
      /-
        case refine_4
        x✝ y✝ z✝ : SetTheory.PGame
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u_1
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
        y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
        z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
        ⊢ ∀ (j : (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (HAdd.hAdd (SetTheory.PGa …
      -/
    · rintro (⟨i, j | k⟩ | ⟨i, j | k⟩)
      · change
          ⟦xL i * (y + z) + x * (yR j + z) - xL i * (yR j + z)⟧ =
            ⟦xL i * y + x * yR j - xL i * yR j + x * z⟧
        /-
          case refine_4.inl.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          j : yr
          ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HSub.hSub (HAdd.hAdd (HMul.hMul (xL  …
        -/
        simp only [quot_sub, quot_add]
        /-
          case refine_4.inl.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (xL  …
        -/
        rw [quot_left_distrib (xL i) (mk yl yr yL yR) (mk zl zr zL zR)]
        /-
          case refine_4.inl.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        rw [quot_left_distrib (mk xl xr xL xR) (yR j) (mk zl zr zL zR)]
        /-
          case refine_4.inl.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        rw [quot_left_distrib (xL i) (yR j) (mk zl zr zL zR)]
        /-
          case refine_4.inl.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
      · change
          ⟦xL i * (y + z) + x * (y + zR k) - xL i * (y + zR k)⟧ =
            ⟦x * y + (xL i * z + x * zR k - xL i * zR k)⟧
        /-
          case refine_4.inl.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          k : zr
          ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HSub.hSub (HAdd.hAdd (HMul.hMul (xL  …
        -/
        simp only [quot_sub, quot_add]
        /-
          case refine_4.inl.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          k : zr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (xL  …
        -/
        rw [quot_left_distrib (xL i) (mk yl yr yL yR) (mk zl zr zL zR)]
        /-
          case refine_4.inl.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          k : zr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        rw [quot_left_distrib (mk xl xr xL xR) (mk yl yr yL yR) (zR k)]
        /-
          case refine_4.inl.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          k : zr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        rw [quot_left_distrib (xL i) (mk yl yr yL yR) (zR k)]
        /-
          case refine_4.inl.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xl
          k : zr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
      · change
          ⟦xR i * (y + z) + x * (yL j + z) - xR i * (yL j + z)⟧ =
            ⟦xR i * y + x * yL j - xR i * yL j + x * z⟧
        /-
          case refine_4.inr.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          j : yl
          ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HSub.hSub (HAdd.hAdd (HMul.hMul (xR  …
        -/
        simp only [quot_sub, quot_add]
        /-
          case refine_4.inr.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (xR  …
        -/
        rw [quot_left_distrib (xR i) (mk yl yr yL yR) (mk zl zr zL zR)]
        /-
          case refine_4.inr.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        rw [quot_left_distrib (mk xl xr xL xR) (yL j) (mk zl zr zL zR)]
        /-
          case refine_4.inr.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        rw [quot_left_distrib (xR i) (yL j) (mk zl zr zL zR)]
        /-
          case refine_4.inr.mk.inl
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
      · change
          ⟦xR i * (y + z) + x * (y + zL k) - xR i * (y + zL k)⟧ =
            ⟦x * y + (xR i * z + x * zL k - xR i * zL k)⟧
        /-
          case refine_4.inr.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          k : zl
          ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HSub.hSub (HAdd.hAdd (HMul.hMul (xR  …
        -/
        simp only [quot_sub, quot_add]
        /-
          case refine_4.inr.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          k : zl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (xR  …
        -/
        rw [quot_left_distrib (xR i) (mk yl yr yL yR) (mk zl zr zL zR)]
        /-
          case refine_4.inr.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          k : zl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        rw [quot_left_distrib (mk xl xr xL xR) (mk yl yr yL yR) (zL k)]
        /-
          case refine_4.inr.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          k : zl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        rw [quot_left_distrib (xR i) (mk yl yr yL yR) (zL k)]
        /-
          case refine_4.inr.mk.inr
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          i : xr
          k : zl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMu …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
  termination_by (x, y, z)


/-- `x * (y + z)` is equivalent to `x * y + x * z.`-/
theorem left_distrib_equiv (x y z : PGame) : x * (y + z) ≈ x * y + x * z :=
  Quotient.exact <| quot_left_distrib _ _ _


@[simp]
theorem quot_left_distrib_sub (x y z : PGame) : (⟦x * (y - z)⟧ : Game) = ⟦x * y⟧ - ⟦x * z⟧ := by
  /-
    x y z : SetTheory.PGame
    ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x (HSub.hSub y z))) (HSub. …
  -/
  change (⟦x * (y + -z)⟧ : Game) = ⟦x * y⟧ + -⟦x * z⟧
  /-
    x y z : SetTheory.PGame
    ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x (HAdd.hAdd y (Neg.neg z) …
  -/
  rw [quot_left_distrib, quot_mul_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem quot_right_distrib (x y z : PGame) : (⟦(x + y) * z⟧ : Game) = ⟦x * z⟧ + ⟦y * z⟧ := by
  /-
    x y z : SetTheory.PGame
    ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (HAdd.hAdd x y) z)) (HAdd. …
  -/
  simp only [quot_mul_comm, quot_left_distrib]
  /-
    🎉 no goals
  -/


/-- `(x + y) * z` is equivalent to `x * z + y * z.`-/
theorem right_distrib_equiv (x y z : PGame) : (x + y) * z ≈ x * z + y * z :=
  Quotient.exact <| quot_right_distrib _ _ _


@[simp]
theorem quot_right_distrib_sub (x y z : PGame) : (⟦(y - z) * x⟧ : Game) = ⟦y * x⟧ - ⟦z * x⟧ := by
  /-
    x y z : SetTheory.PGame
    ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (HSub.hSub y z) x)) (HSub. …
  -/
  change (⟦(y + -z) * x⟧ : Game) = ⟦y * x⟧ + -⟦z * x⟧
  /-
    x y z : SetTheory.PGame
    ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (HAdd.hAdd y (Neg.neg z))  …
  -/
  rw [quot_right_distrib, quot_neg_mul]
  /-
    🎉 no goals
  -/


/-- `x * 1` has the same moves as `x`. -/
def mulOneRelabelling : ∀ x : PGame.{u}, x * 1 ≡r x
  | ⟨xl, xr, xL, xR⟩ => by
    -- Porting note: the next four lines were just `unfold has_one.one,`
    /-
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      ⊢ (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) 1).Relabelling (SetTheory.PGame. …
    -/
    show _ * One.one ≡r _
    /-
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      ⊢ (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) One.one).Relabelling (SetTheory. …
    -/
    unfold One.one
    /-
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      ⊢ (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) SetTheory.PGame.instOnePGame.1). …
    -/
    unfold instOnePGame
    /-
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      ⊢ (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) { one := SetTheory.PGame.mk PUni …
    -/
    change mk _ _ _ _ * mk _ _ _ _ ≡r _
    refine ⟨(Equiv.sumEmpty _ _).trans (Equiv.prodPUnit _),
      (Equiv.emptySum _ _).trans (Equiv.prodPUnit _), ?_, ?_⟩ <;>
     /-
       case refine_1
       xl xr : Type u
       xL : xl → SetTheory.PGame
       xR : xr → SetTheory.PGame
       ⊢ (i : (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk PUnit.{ …
     -/
    (try rintro (⟨i, ⟨⟩⟩ | ⟨i, ⟨⟩⟩)) <;>
    { dsimp
      apply (Relabelling.subCongr (Relabelling.refl _) (mulZeroRelabelling _)).trans
      rw [sub_zero_eq_add_zero]
      exact (addZeroRelabelling _).trans <|
        (((mulOneRelabelling _).addCongr (mulZeroRelabelling _)).trans <| addZeroRelabelling _) }


@[simp]
theorem quot_mul_one (x : PGame) : (⟦x * 1⟧ : Game) = ⟦x⟧ :=
  game_eq <| PGame.Relabelling.equiv <| mulOneRelabelling x


/-- `x * 1` is equivalent to `x`. -/
theorem mul_one_equiv (x : PGame) : x * 1 ≈ x :=
  Quotient.exact <| quot_mul_one x


/-- `1 * x` has the same moves as `x`. -/
def oneMulRelabelling (x : PGame) : 1 * x ≡r x :=
  (mulCommRelabelling 1 x).trans <| mulOneRelabelling x


@[simp]
theorem quot_one_mul (x : PGame) : (⟦1 * x⟧ : Game) = ⟦x⟧ :=
  game_eq <| PGame.Relabelling.equiv <| oneMulRelabelling x


/-- `1 * x` is equivalent to `x`. -/
theorem one_mul_equiv (x : PGame) : 1 * x ≈ x :=
  Quotient.exact <| quot_one_mul x


theorem quot_mul_assoc (x y z : PGame) : (⟦x * y * z⟧ : Game) = ⟦x * (y * z)⟧ :=
  match x, y, z with
  | mk xl xr xL xR, mk yl yr yL yR, mk zl zr zL zR => by
    /-
      x y z : SetTheory.PGame
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      zl zr : Type u_1
      zL : zl → SetTheory.PGame
      zR : zr → SetTheory.PGame
      ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (HMul.hMul (SetTheory.PGam …
    -/
    let x := mk xl xr xL xR
    /-
      x✝ y z : SetTheory.PGame
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      zl zr : Type u_1
      zL : zl → SetTheory.PGame
      zR : zr → SetTheory.PGame
      x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
      ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (HMul.hMul (SetTheory.PGam …
    -/
    let y := mk yl yr yL yR
    /-
      x✝ y✝ z : SetTheory.PGame
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      zl zr : Type u_1
      zL : zl → SetTheory.PGame
      zR : zr → SetTheory.PGame
      x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
      y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
      ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (HMul.hMul (SetTheory.PGam …
    -/
    let z := mk zl zr zL zR
    /-
      x✝ y✝ z✝ : SetTheory.PGame
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      zl zr : Type u_1
      zL : zl → SetTheory.PGame
      zR : zr → SetTheory.PGame
      x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
      y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
      z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
      ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (HMul.hMul (SetTheory.PGam …
    -/
    refine quot_eq_of_mk'_quot_eq ?_ ?_ ?_ ?_
      /-
        case refine_1
        x✝ y✝ z✝ : SetTheory.PGame
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u_1
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
        y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
        z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
        ⊢ _root_.Equiv (HMul.hMul (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (SetTheo …
      -/
    · fconstructor
        /-
          case refine_1.toFun
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          ⊢ (HMul.hMul (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk y …
        -/
      · rintro (⟨⟨_, _⟩ | ⟨_, _⟩, _⟩ | ⟨⟨_, _⟩ | ⟨_, _⟩, _⟩) <;>
          -- Porting note: as above, increased the `maxDepth` here by 1.
          /-
            case refine_1.toFun.inl.mk.inl.mk
            x✝ y✝ z✝ : SetTheory.PGame
            xl xr : Type u_1
            xL : xl → SetTheory.PGame
            xR : xr → SetTheory.PGame
            yl yr : Type u_1
            yL : yl → SetTheory.PGame
            yR : yr → SetTheory.PGame
            zl zr : Type u_1
            zL : zl → SetTheory.PGame
            zR : zr → SetTheory.PGame
            x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
            y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
            z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
            snd✝¹ : zl
            fst✝ : xl
            snd✝ : yl
            ⊢ (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (HMul.hMul (SetTheory.PGame.mk y …
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          solve_by_elim (config := { maxDepth := 8 }) [Sum.inl, Sum.inr, Prod.mk]
          /-
            🎉 no goals
          -/
        /-
          case refine_1.invFun
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          ⊢ (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (HMul.hMul (SetTheory.PGame.mk y …
        -/
      · rintro (⟨_, ⟨_, _⟩ | ⟨_, _⟩⟩ | ⟨_, ⟨_, _⟩ | ⟨_, _⟩⟩) <;>
          /-
            case refine_1.invFun.inl.mk.inl.mk
            x✝ y✝ z✝ : SetTheory.PGame
            xl xr : Type u_1
            xL : xl → SetTheory.PGame
            xR : xr → SetTheory.PGame
            yl yr : Type u_1
            yL : yl → SetTheory.PGame
            yR : yr → SetTheory.PGame
            zl zr : Type u_1
            zL : zl → SetTheory.PGame
            zR : zr → SetTheory.PGame
            x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
            y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
            z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
            fst✝¹ : xl
            fst✝ : yl
            snd✝ : zl
            ⊢ (HMul.hMul (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk y …
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          solve_by_elim (config := { maxDepth := 8 }) [Sum.inl, Sum.inr, Prod.mk]
          /-
            🎉 no goals
          -/
        /-
          case refine_1.left_inv
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          ⊢ Function.LeftInverse (fun a => Sum.casesOn a (fun val => Prod.casesOn val fu …
        -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
      · rintro (⟨⟨_, _⟩ | ⟨_, _⟩, _⟩ | ⟨⟨_, _⟩ | ⟨_, _⟩, _⟩) <;> rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
        /-
          case refine_1.right_inv
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          ⊢ Function.RightInverse (fun a => Sum.casesOn a (fun val => Prod.casesOn val f …
        -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
      · rintro (⟨_, ⟨_, _⟩ | ⟨_, _⟩⟩ | ⟨_, ⟨_, _⟩ | ⟨_, _⟩⟩) <;> rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
      /-
        case refine_2
        x✝ y✝ z✝ : SetTheory.PGame
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u_1
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
        y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
        z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
        ⊢ _root_.Equiv (HMul.hMul (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (SetTheo …
      -/
    · fconstructor
        /-
          case refine_2.toFun
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          ⊢ (HMul.hMul (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk y …
        -/
      · rintro (⟨⟨_, _⟩ | ⟨_, _⟩, _⟩ | ⟨⟨_, _⟩ | ⟨_, _⟩, _⟩) <;>
          /-
            case refine_2.toFun.inl.mk.inl.mk
            x✝ y✝ z✝ : SetTheory.PGame
            xl xr : Type u_1
            xL : xl → SetTheory.PGame
            xR : xr → SetTheory.PGame
            yl yr : Type u_1
            yL : yl → SetTheory.PGame
            yR : yr → SetTheory.PGame
            zl zr : Type u_1
            zL : zl → SetTheory.PGame
            zR : zr → SetTheory.PGame
            x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
            y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
            z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
            snd✝¹ : zr
            fst✝ : xl
            snd✝ : yl
            ⊢ (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (HMul.hMul (SetTheory.PGame.mk y …
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          solve_by_elim (config := { maxDepth := 8 }) [Sum.inl, Sum.inr, Prod.mk]
          /-
            🎉 no goals
          -/
        /-
          case refine_2.invFun
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          ⊢ (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (HMul.hMul (SetTheory.PGame.mk y …
        -/
      · rintro (⟨_, ⟨_, _⟩ | ⟨_, _⟩⟩ | ⟨_, ⟨_, _⟩ | ⟨_, _⟩⟩) <;>
          /-
            case refine_2.invFun.inl.mk.inl.mk
            x✝ y✝ z✝ : SetTheory.PGame
            xl xr : Type u_1
            xL : xl → SetTheory.PGame
            xR : xr → SetTheory.PGame
            yl yr : Type u_1
            yL : yl → SetTheory.PGame
            yR : yr → SetTheory.PGame
            zl zr : Type u_1
            zL : zl → SetTheory.PGame
            zR : zr → SetTheory.PGame
            x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
            y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
            z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
            fst✝¹ : xl
            fst✝ : yl
            snd✝ : zr
            ⊢ (HMul.hMul (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk y …
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          solve_by_elim (config := { maxDepth := 8 }) [Sum.inl, Sum.inr, Prod.mk]
          /-
            🎉 no goals
          -/
        /-
          case refine_2.left_inv
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          ⊢ Function.LeftInverse (fun a => Sum.casesOn a (fun val => Prod.casesOn val fu …
        -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
      · rintro (⟨⟨_, _⟩ | ⟨_, _⟩, _⟩ | ⟨⟨_, _⟩ | ⟨_, _⟩, _⟩) <;> rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
        /-
          case refine_2.right_inv
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          ⊢ Function.RightInverse (fun a => Sum.casesOn a (fun val => Prod.casesOn val f …
        -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
      · rintro (⟨_, ⟨_, _⟩ | ⟨_, _⟩⟩ | ⟨_, ⟨_, _⟩ | ⟨_, _⟩⟩) <;> rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    -- Porting note: explicitly wrote out arguments to each recursive
    -- quot_mul_assoc reference below, because otherwise the decreasing_by block
    -- failed. Each branch previously ended with: `simp [quot_mul_assoc]; abel`
    -- See https://github.com/leanprover/lean4/issues/2288
      /-
        case refine_3
        x✝ y✝ z✝ : SetTheory.PGame
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u_1
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
        y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
        z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
        ⊢ ∀ (i : (HMul.hMul (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGa …
      -/
    · rintro (⟨⟨i, j⟩ | ⟨i, j⟩, k⟩ | ⟨⟨i, j⟩ | ⟨i, j⟩, k⟩)
      · change
          ⟦(xL i * y + x * yL j - xL i * yL j) * z + x * y * zL k -
                (xL i * y + x * yL j - xL i * yL j) * zL k⟧ =
            ⟦xL i * (y * z) + x * (yL j * z + y * zL k - yL j * zL k) -
                xL i * (yL j * z + y * zL k - yL j * zL k)⟧
        simp only [quot_sub, quot_add, quot_right_distrib_sub, quot_right_distrib,
                   quot_left_distrib_sub, quot_left_distrib]
        /-
          case refine_3.inl.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xL i) (mk yl yr yL yR) (mk zl zr zL zR)]
        /-
          case refine_3.inl.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (yL j) (mk zl zr zL zR)]
        /-
          case refine_3.inl.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xL i) (yL j) (mk zl zr zL zR)]
        /-
          case refine_3.inl.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (mk yl yr yL yR) (zL k)]
        /-
          case refine_3.inl.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xL i) (mk yl yr yL yR) (zL k)]
        /-
          case refine_3.inl.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (yL j) (zL k)]
        /-
          case refine_3.inl.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xL i) (yL j) (zL k)]
        /-
          case refine_3.inl.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
      · change
          ⟦(xR i * y + x * yR j - xR i * yR j) * z + x * y * zL k -
                (xR i * y + x * yR j - xR i * yR j) * zL k⟧ =
            ⟦xR i * (y * z) + x * (yR j * z + y * zL k - yR j * zL k) -
                xR i * (yR j * z + y * zL k - yR j * zL k)⟧
        simp only [quot_sub, quot_add, quot_right_distrib_sub, quot_right_distrib,
                   quot_left_distrib_sub, quot_left_distrib]
        /-
          case refine_3.inl.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xR i) (mk yl yr yL yR) (mk zl zr zL zR)]
        /-
          case refine_3.inl.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (yR j) (mk zl zr zL zR)]
        /-
          case refine_3.inl.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xR i) (yR j) (mk zl zr zL zR)]
        /-
          case refine_3.inl.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (mk yl yr yL yR) (zL k)]
        /-
          case refine_3.inl.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xR i) (mk yl yr yL yR) (zL k)]
        /-
          case refine_3.inl.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (yR j) (zL k)]
        /-
          case refine_3.inl.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xR i) (yR j) (zL k)]
        /-
          case refine_3.inl.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
      · change
          ⟦(xL i * y + x * yR j - xL i * yR j) * z + x * y * zR k -
                (xL i * y + x * yR j - xL i * yR j) * zR k⟧ =
            ⟦xL i * (y * z) + x * (yR j * z + y * zR k - yR j * zR k) -
                xL i * (yR j * z + y * zR k - yR j * zR k)⟧
        simp only [quot_sub, quot_add, quot_right_distrib_sub, quot_right_distrib,
                   quot_left_distrib_sub, quot_left_distrib]
        /-
          case refine_3.inr.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xL i) (mk yl yr yL yR) (mk zl zr zL zR)]
        /-
          case refine_3.inr.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (yR j) (mk zl zr zL zR)]
        /-
          case refine_3.inr.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xL i) (yR j) (mk zl zr zL zR)]
        /-
          case refine_3.inr.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (mk yl yr yL yR) (zR k)]
        /-
          case refine_3.inr.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xL i) (mk yl yr yL yR) (zR k)]
        /-
          case refine_3.inr.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (yR j) (zR k)]
        /-
          case refine_3.inr.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xL i) (yR j) (zR k)]
        /-
          case refine_3.inr.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
      · change
          ⟦(xR i * y + x * yL j - xR i * yL j) * z + x * y * zR k -
                (xR i * y + x * yL j - xR i * yL j) * zR k⟧ =
            ⟦xR i * (y * z) + x * (yL j * z + y * zR k - yL j * zR k) -
                xR i * (yL j * z + y * zR k - yL j * zR k)⟧
        simp only [quot_sub, quot_add, quot_right_distrib_sub, quot_right_distrib,
                   quot_left_distrib_sub, quot_left_distrib]
        /-
          case refine_3.inr.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xR i) (mk yl yr yL yR) (mk zl zr zL zR)]
        /-
          case refine_3.inr.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (yL j) (mk zl zr zL zR)]
        /-
          case refine_3.inr.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xR i) (yL j) (mk zl zr zL zR)]
        /-
          case refine_3.inr.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (mk yl yr yL yR) (zR k)]
        /-
          case refine_3.inr.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xR i) (mk yl yr yL yR) (zR k)]
        /-
          case refine_3.inr.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (yL j) (zR k)]
        /-
          case refine_3.inr.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xR i) (yL j) (zR k)]
        /-
          case refine_3.inr.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
      /-
        case refine_4
        x✝ y✝ z✝ : SetTheory.PGame
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u_1
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
        y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
        z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
        ⊢ ∀ (j : (HMul.hMul (HMul.hMul (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGa …
      -/
    · rintro (⟨⟨i, j⟩ | ⟨i, j⟩, k⟩ | ⟨⟨i, j⟩ | ⟨i, j⟩, k⟩)
      · change
          ⟦(xL i * y + x * yL j - xL i * yL j) * z + x * y * zR k -
                (xL i * y + x * yL j - xL i * yL j) * zR k⟧ =
            ⟦xL i * (y * z) + x * (yL j * z + y * zR k - yL j * zR k) -
                xL i * (yL j * z + y * zR k - yL j * zR k)⟧
        simp only [quot_sub, quot_add, quot_right_distrib_sub, quot_right_distrib,
                   quot_left_distrib_sub, quot_left_distrib]
        /-
          case refine_4.inl.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xL i) (mk yl yr yL yR) (mk zl zr zL zR)]
        /-
          case refine_4.inl.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (yL j) (mk zl zr zL zR)]
        /-
          case refine_4.inl.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xL i) (yL j) (mk zl zr zL zR)]
        /-
          case refine_4.inl.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (mk yl yr yL yR) (zR k)]
        /-
          case refine_4.inl.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xL i) (mk yl yr yL yR) (zR k)]
        /-
          case refine_4.inl.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (yL j) (zR k)]
        /-
          case refine_4.inl.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xL i) (yL j) (zR k)]
        /-
          case refine_4.inl.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xl
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
      · change
          ⟦(xR i * y + x * yR j - xR i * yR j) * z + x * y * zR k -
                (xR i * y + x * yR j - xR i * yR j) * zR k⟧ =
            ⟦xR i * (y * z) + x * (yR j * z + y * zR k - yR j * zR k) -
                xR i * (yR j * z + y * zR k - yR j * zR k)⟧
        simp only [quot_sub, quot_add, quot_right_distrib_sub, quot_right_distrib,
                   quot_left_distrib_sub, quot_left_distrib]
        /-
          case refine_4.inl.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xR i) (mk yl yr yL yR) (mk zl zr zL zR)]
        /-
          case refine_4.inl.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (yR j) (mk zl zr zL zR)]
        /-
          case refine_4.inl.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xR i) (yR j) (mk zl zr zL zR)]
        /-
          case refine_4.inl.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (mk yl yr yL yR) (zR k)]
        /-
          case refine_4.inl.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xR i) (mk yl yr yL yR) (zR k)]
        /-
          case refine_4.inl.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (yR j) (zR k)]
        /-
          case refine_4.inl.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xR i) (yR j) (zR k)]
        /-
          case refine_4.inl.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zr
          i : xr
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
      · change
          ⟦(xL i * y + x * yR j - xL i * yR j) * z + x * y * zL k -
                (xL i * y + x * yR j - xL i * yR j) * zL k⟧ =
            ⟦xL i * (y * z) + x * (yR j * z + y * zL k - yR j * zL k) -
                xL i * (yR j * z + y * zL k - yR j * zL k)⟧
        simp only [quot_sub, quot_add, quot_right_distrib_sub, quot_right_distrib,
                   quot_left_distrib_sub, quot_left_distrib]
        /-
          case refine_4.inr.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xL i) (mk yl yr yL yR) (mk zl zr zL zR)]
        /-
          case refine_4.inr.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (yR j) (mk zl zr zL zR)]
        /-
          case refine_4.inr.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xL i) (yR j) (mk zl zr zL zR)]
        /-
          case refine_4.inr.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (mk yl yr yL yR) (zL k)]
        /-
          case refine_4.inr.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xL i) (mk yl yr yL yR) (zL k)]
        /-
          case refine_4.inr.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (yR j) (zL k)]
        /-
          case refine_4.inr.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xL i) (yR j) (zL k)]
        /-
          case refine_4.inr.mk.inl.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xl
          j : yr
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
      · change
          ⟦(xR i * y + x * yL j - xR i * yL j) * z + x * y * zL k -
                (xR i * y + x * yL j - xR i * yL j) * zL k⟧ =
            ⟦xR i * (y * z) + x * (yL j * z + y * zL k - yL j * zL k) -
                xR i * (yL j * z + y * zL k - yL j * zL k)⟧
        simp only [quot_sub, quot_add, quot_right_distrib_sub, quot_right_distrib,
                   quot_left_distrib_sub, quot_left_distrib]
        /-
          case refine_4.inr.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xR i) (mk yl yr yL yR) (mk zl zr zL zR)]
        /-
          case refine_4.inr.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (yL j) (mk zl zr zL zR)]
        /-
          case refine_4.inr.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xR i) (yL j) (mk zl zr zL zR)]
        /-
          case refine_4.inr.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (mk yl yr yL yR) (zL k)]
        /-
          case refine_4.inr.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xR i) (mk yl yr yL yR) (zL k)]
        /-
          case refine_4.inr.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (mk xl xr xL xR) (yL j) (zL k)]
        /-
          case refine_4.inr.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        rw [quot_mul_assoc (xR i) (yL j) (zL k)]
        /-
          case refine_4.inr.mk.inr.mk
          x✝ y✝ z✝ : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          x : SetTheory.PGame := SetTheory.PGame.mk xl xr xL xR
          y : SetTheory.PGame := SetTheory.PGame.mk yl yr yL yR
          z : SetTheory.PGame := SetTheory.PGame.mk zl zr zL zR
          k : zl
          i : xr
          j : yl
          ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame. …
        -/
        /-
          🎉 no goals
        -/
        abel
        /-
          🎉 no goals
        -/
  termination_by (x, y, z)


/-- `x * y * z` is equivalent to `x * (y * z).`-/
theorem mul_assoc_equiv (x y z : PGame) : x * y * z ≈ x * (y * z) :=
  Quotient.exact <| quot_mul_assoc _ _ _


/-- The left options of `x * y` of the first kind, i.e. of the form `xL * y + x * yL - xL * yL`. -/
def mulOption (x y : PGame) (i : LeftMoves x) (j : LeftMoves y) : PGame :=
  x.moveLeft i * y + x * y.moveLeft j - x.moveLeft i * y.moveLeft j


/-- Any left option of `x * y` of the first kind is also a left option of `x * -(-y)` of
  the first kind. -/
lemma mulOption_neg_neg {x} (y) {i j} :
    mulOption x y i j = mulOption x (-(-y)) i (toLeftMovesNeg <| toRightMovesNeg j) := by
  /-
    x y : SetTheory.PGame
    i : x.LeftMoves
    j : y.LeftMoves
    ⊢ Eq (x.mulOption y i j) (x.mulOption (Neg.neg (Neg.neg y)) i (SetTheory.PGame …
  -/
  simp [mulOption]
  /-
    🎉 no goals
  -/


/-- The left options of `x * y` agree with that of `y * x` up to equivalence. -/
lemma mulOption_symm (x y) {i j} : ⟦mulOption x y i j⟧ = (⟦mulOption y x j i⟧ : Game) := by
  /-
    x y : SetTheory.PGame
    i : x.LeftMoves
    j : y.LeftMoves
    ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (x.mulOption y i j)) (Quotient.mk Set …
  -/
  dsimp only [mulOption, quot_sub, quot_add]
  /-
    x y : SetTheory.PGame
    i : x.LeftMoves
    j : y.LeftMoves
    ⊢ Eq (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (x.m …
  -/
  rw [add_comm]
  /-
    x y : SetTheory.PGame
    i : x.LeftMoves
    j : y.LeftMoves
    ⊢ Eq (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x (y …
  -/
  congr 1
  /-
    case e_a
    x y : SetTheory.PGame
    i : x.LeftMoves
    j : y.LeftMoves
    ⊢ Eq (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x (y.moveLeft j …
  -/
  on_goal 1 => congr 1
  /-
    case e_a.e_a
    x y : SetTheory.PGame
    i : x.LeftMoves
    j : y.LeftMoves
    ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x (y.moveLeft j))) (Quotie …
  -/
  all_goals rw [quot_mul_comm]
  /-
    🎉 no goals
  -/


/-- The left options of `x * y` of the second kind are the left options of `(-x) * (-y)` of the
  first kind, up to equivalence. -/
lemma leftMoves_mul_iff {x y : PGame} (P : Game → Prop) :
    (∀ k, P ⟦(x * y).moveLeft k⟧) ↔
    (∀ i j, P ⟦mulOption x y i j⟧) ∧ (∀ i j, P ⟦mulOption (-x) (-y) i j⟧) := by
  /-
    x y : SetTheory.PGame
    P : SetTheory.Game → Prop
    ⊢ Iff (∀ (k : (HMul.hMul x y).LeftMoves), P (Quotient.mk SetTheory.PGame.setoi …
  -/
  cases x; cases y
  /-
    case mk.mk
    P : SetTheory.Game → Prop
    α✝¹ β✝¹ : Type u_1
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    ⊢ Iff (∀ (k : (HMul.hMul (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²) (SetTheory.PGame …
  -/
  constructor <;> intro h
  on_goal 1 =>
    constructor <;> intros i j
    · exact h (Sum.inl (i, j))
    convert h (Sum.inr (i, j)) using 1
  on_goal 2 =>
    rintro (⟨i, j⟩ | ⟨i, j⟩)
    · exact h.1 i j
    convert h.2 i j using 1
  all_goals
    dsimp only [mk_mul_moveLeft_inr, quot_sub, quot_add, neg_def, mulOption, moveLeft_mk]
    rw [← neg_def, ← neg_def]
    congr 1
    on_goal 1 => congr 1
    all_goals rw [quot_neg_mul_neg]


/-- The right options of `x * y` are the left options of `x * (-y)` and of `(-x) * y` of the first
  kind, up to equivalence. -/
lemma rightMoves_mul_iff {x y : PGame} (P : Game → Prop) :
    (∀ k, P ⟦(x * y).moveRight k⟧) ↔
    (∀ i j, P (-⟦mulOption x (-y) i j⟧)) ∧ (∀ i j, P (-⟦mulOption (-x) y i j⟧)) := by
  /-
    x y : SetTheory.PGame
    P : SetTheory.Game → Prop
    ⊢ Iff (∀ (k : (HMul.hMul x y).RightMoves), P (Quotient.mk SetTheory.PGame.seto …
  -/
  cases x; cases y
  /-
    case mk.mk
    P : SetTheory.Game → Prop
    α✝¹ β✝¹ : Type u_1
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    ⊢ Iff (∀ (k : (HMul.hMul (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²) (SetTheory.PGame …
  -/
  constructor <;> intro h
  on_goal 1 =>
    constructor <;> intros i j
    on_goal 1 => convert h (Sum.inl (i, j))
  /-
    case h.e'_1
    P : SetTheory.Game → Prop
    α✝¹ β✝¹ : Type u_1
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    h : ∀ (k : (HMul.hMul (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²) (SetTheory.PGame.mk …
    i : (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²).LeftMoves
    j : (Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝)).LeftMoves
    ⊢ Eq (Neg.neg (Quotient.mk SetTheory.PGame.setoid ((SetTheory.PGame.mk α✝¹ β✝¹ …
  -/
  on_goal 2 => convert h (Sum.inr (i, j))
  on_goal 3 =>
    rintro (⟨i, j⟩ | ⟨i, j⟩)
    on_goal 1 => convert h.1 i j using 1
    on_goal 2 => convert h.2 i j using 1
  all_goals
    dsimp [mulOption]
    rw [neg_sub', neg_add, ← neg_def]
    congr 1
    on_goal 1 => congr 1
  /-
    case h.e'_1.e_a.e_a
    P : SetTheory.Game → Prop
    α✝¹ β✝¹ : Type u_1
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    h : ∀ (k : (HMul.hMul (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²) (SetTheory.PGame.mk …
    i : (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²).LeftMoves
    j : (Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝)).LeftMoves
    ⊢ Eq (Neg.neg (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (a✝³ i) (Neg.neg  …
  -/
  any_goals rw [quot_neg_mul, neg_neg]
  /-
    case h.e'_1.e_a.e_a
    P : SetTheory.Game → Prop
    α✝¹ β✝¹ : Type u_1
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    h : ∀ (k : (HMul.hMul (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²) (SetTheory.PGame.mk …
    i : (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²).LeftMoves
    j : (Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝)).LeftMoves
    ⊢ Eq (Neg.neg (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (a✝³ i) (Neg.neg  …
  -/
  iterate 6 rw [quot_mul_neg, neg_neg]
  /-
    🎉 no goals
  -/


/-- Because the two halves of the definition of `inv` produce more elements
on each side, we have to define the two families inductively.
This is the indexing set for the function, and `invVal` is the function part. -/
inductive InvTy (l r : Type u) : Bool → Type u
  | zero : InvTy l r false
  | left₁ : r → InvTy l r false → InvTy l r false
  | left₂ : l → InvTy l r true → InvTy l r false
  | right₁ : l → InvTy l r false → InvTy l r true
  | right₂ : r → InvTy l r true → InvTy l r true


instance (l r : Type u) [IsEmpty l] [IsEmpty r] : IsEmpty (InvTy l r true) :=
      /-
        l r : Type u
        inst✝¹ : IsEmpty l
        inst✝ : IsEmpty r
        ⊢ SetTheory.PGame.InvTy l r Bool.true → False
      -/
                                     /-
                                       🎉 no goals
                                     -/
  ⟨by rintro (_ | _ | _ | a | a) <;> exact isEmptyElim a⟩
                                     /-
                                       🎉 no goals
                                     -/


instance InvTy.instInhabited (l r : Type u) : Inhabited (InvTy l r false) :=
  ⟨InvTy.zero⟩


instance uniqueInvTy (l r : Type u) [IsEmpty l] [IsEmpty r] : Unique (InvTy l r false) :=
  { InvTy.instInhabited l r with
    uniq := by
      /-
        l r : Type u
        inst✝¹ : IsEmpty l
        inst✝ : IsEmpty r
        ⊢ ∀ (a : SetTheory.PGame.InvTy l r Bool.false), Eq a Inhabited.default
      -/
      rintro (a | a | a)
        /-
          case zero
          l r : Type u
          inst✝¹ : IsEmpty l
          inst✝ : IsEmpty r
          ⊢ Eq SetTheory.PGame.InvTy.zero Inhabited.default
        -/
      · rfl
        /-
          🎉 no goals
        -/
      /-
        case left₁
        l r : Type u
        inst✝¹ : IsEmpty l
        inst✝ : IsEmpty r
        a : r
        a✝ : SetTheory.PGame.InvTy l r Bool.false
        ⊢ Eq (SetTheory.PGame.InvTy.left₁ a a✝) Inhabited.default
      -/
      all_goals exact isEmptyElim a }
      /-
        🎉 no goals
      -/


/-- Because the two halves of the definition of `inv` produce more elements
of each side, we have to define the two families inductively.
This is the function part, defined by recursion on `InvTy`. -/
def invVal {l r} (L : l → PGame) (R : r → PGame) (IHl : l → PGame) (IHr : r → PGame)
    (x : PGame) : ∀ {b}, InvTy l r b → PGame
  | _, InvTy.zero => 0
  | _, InvTy.left₁ i j => (1 + (R i - x) * invVal L R IHl IHr x j) * IHr i
  | _, InvTy.left₂ i j => (1 + (L i - x) * invVal L R IHl IHr x j) * IHl i
  | _, InvTy.right₁ i j => (1 + (L i - x) * invVal L R IHl IHr x j) * IHl i
  | _, InvTy.right₂ i j => (1 + (R i - x) * invVal L R IHl IHr x j) * IHr i


@[simp]
theorem invVal_isEmpty {l r : Type u} {b} (L R IHl IHr) (i : InvTy l r b) (x) [IsEmpty l]
    [IsEmpty r] : invVal L R IHl IHr x i = 0 := by
  /-
    l r : Type u
    b : Bool
    L : l → SetTheory.PGame
    R : r → SetTheory.PGame
    IHl : l → SetTheory.PGame
    IHr : r → SetTheory.PGame
    i : SetTheory.PGame.InvTy l r b
    x : SetTheory.PGame
    inst✝¹ : IsEmpty l
    inst✝ : IsEmpty r
    ⊢ Eq (SetTheory.PGame.invVal L R IHl IHr x i) 0
  -/
  cases' i with a _ a _ a _ a
    /-
      case zero
      l r : Type u
      L : l → SetTheory.PGame
      R : r → SetTheory.PGame
      IHl : l → SetTheory.PGame
      IHr : r → SetTheory.PGame
      x : SetTheory.PGame
      inst✝¹ : IsEmpty l
      inst✝ : IsEmpty r
      ⊢ Eq (SetTheory.PGame.invVal L R IHl IHr x SetTheory.PGame.InvTy.zero) 0
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case left₁
    l r : Type u
    L : l → SetTheory.PGame
    R : r → SetTheory.PGame
    IHl : l → SetTheory.PGame
    IHr : r → SetTheory.PGame
    x : SetTheory.PGame
    inst✝¹ : IsEmpty l
    inst✝ : IsEmpty r
    a : r
    a✝ : SetTheory.PGame.InvTy l r Bool.false
    ⊢ Eq (SetTheory.PGame.invVal L R IHl IHr x (SetTheory.PGame.InvTy.left₁ a a✝)) 0
  -/
  all_goals exact isEmptyElim a
  /-
    🎉 no goals
  -/


/-- The inverse of a positive surreal number `x = {L | R}` is
given by `x⁻¹ = {0,
  (1 + (R - x) * x⁻¹L) * R, (1 + (L - x) * x⁻¹R) * L |
  (1 + (L - x) * x⁻¹L) * L, (1 + (R - x) * x⁻¹R) * R}`.
Because the two halves `x⁻¹L, x⁻¹R` of `x⁻¹` are used in their own
definition, the sets and elements are inductively generated. -/
def inv' : PGame → PGame
  | ⟨l, r, L, R⟩ =>
    let l' := { i // 0 < L i }
    let L' : l' → PGame := fun i => L i.1
    let IHl' : l' → PGame := fun i => inv' (L i.1)
    let IHr i := inv' (R i)
    let x := mk l r L R
    ⟨InvTy l' r false, InvTy l' r true, invVal L' R IHl' IHr x, invVal L' R IHl' IHr x⟩


theorem zero_lf_inv' : ∀ x : PGame, 0 ⧏ inv' x
  | ⟨xl, xr, xL, xR⟩ => by
    /-
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      ⊢ SetTheory.PGame.LF 0 (SetTheory.PGame.mk xl xr xL xR).inv'
    -/
    convert lf_mk _ _ InvTy.zero
    /-
      case h.e'_1
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      ⊢ Eq 0 (SetTheory.PGame.invVal (fun i => xL ↑i) xR (fun i => (xL ↑i).inv') (fu …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- `inv' 0` has exactly the same moves as `1`. -/
def inv'Zero : inv' 0 ≡r 1 := by
  /-
    ⊢ (SetTheory.PGame.inv' 0).Relabelling 1
  -/
  change mk _ _ _ _ ≡r 1
  /-
    ⊢ (SetTheory.PGame.mk (SetTheory.PGame.InvTy (Subtype fun i => LT.lt 0 i.elim) …
  -/
  refine ⟨?_, ?_, fun i => ?_, IsEmpty.elim ?_⟩
    /-
      case refine_1
      ⊢ _root_.Equiv (SetTheory.PGame.mk (SetTheory.PGame.InvTy (Subtype fun i => LT …
    -/
  · apply Equiv.equivPUnit (InvTy _ _ _)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ⊢ _root_.Equiv (SetTheory.PGame.mk (SetTheory.PGame.InvTy (Subtype fun i => LT …
    -/
  · apply Equiv.equivPEmpty (InvTy _ _ _)
    /-
      🎉 no goals
    -/
  · -- Porting note: had to add `rfl`, because `simp` only uses the built-in `rfl`.
    /-
      case refine_3
      i : (SetTheory.PGame.mk (SetTheory.PGame.InvTy (Subtype fun i => LT.lt 0 i.eli …
      ⊢ ((SetTheory.PGame.mk (SetTheory.PGame.InvTy (Subtype fun i => LT.lt 0 i.elim …
    -/
    simp; rfl
          /-
            🎉 no goals
          -/
    /-
      case refine_4
      ⊢ IsEmpty (SetTheory.PGame.mk (SetTheory.PGame.InvTy (Subtype fun i => LT.lt 0 …
    -/
  · dsimp
    /-
      case refine_4
      ⊢ IsEmpty (SetTheory.PGame.InvTy (Subtype fun i => LT.lt 0 i.elim) PEmpty.{?u. …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


theorem inv'_zero_equiv : inv' 0 ≈ 1 :=
  inv'Zero.equiv


/-- `inv' 1` has exactly the same moves as `1`. -/
def inv'One : inv' 1 ≡r (1 : PGame.{u}) := by
  /-
    ⊢ (SetTheory.PGame.inv' 1).Relabelling 1
  -/
  change Relabelling (mk _ _ _ _) 1
  have : IsEmpty { _i : PUnit.{u + 1} // (0 : PGame.{u}) < 0 } := by
    rw [lt_self_iff_false]
    infer_instance
  /-
    this : IsEmpty (Subtype fun _i => LT.lt 0 0)
    ⊢ (SetTheory.PGame.mk (SetTheory.PGame.InvTy (Subtype fun i => LT.lt 0 ((fun x …
  -/
  refine ⟨?_, ?_, fun i => ?_, IsEmpty.elim ?_⟩ <;> dsimp
    /-
      case refine_1
      this : IsEmpty (Subtype fun _i => LT.lt 0 0)
      ⊢ _root_.Equiv (SetTheory.PGame.InvTy (Subtype fun i => LT.lt 0 0) PEmpty.{u + …
    -/
  · apply Equiv.equivPUnit
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      this : IsEmpty (Subtype fun _i => LT.lt 0 0)
      ⊢ _root_.Equiv (SetTheory.PGame.InvTy (Subtype fun i => LT.lt 0 0) PEmpty.{u + …
    -/
  · apply Equiv.equivOfIsEmpty
    /-
      🎉 no goals
    -/
  · -- Porting note: had to add `rfl`, because `simp` only uses the built-in `rfl`.
    /-
      case refine_3
      this : IsEmpty (Subtype fun _i => LT.lt 0 0)
      i : (SetTheory.PGame.mk (SetTheory.PGame.InvTy (Subtype fun i => LT.lt 0 ((fun …
      ⊢ (SetTheory.PGame.invVal (fun i => 0) PEmpty.elim (fun i => SetTheory.PGame.i …
    -/
    simp; rfl
          /-
            🎉 no goals
          -/
    /-
      case refine_4
      this : IsEmpty (Subtype fun _i => LT.lt 0 0)
      ⊢ IsEmpty (SetTheory.PGame.InvTy (Subtype fun i => LT.lt 0 0) PEmpty.{u + 1} B …
    -/
  · infer_instance
    /-
      🎉 no goals
    -/


theorem inv'_one_equiv : inv' 1 ≈ 1 :=
  inv'One.equiv


/-- The inverse of a pre-game in terms of the inverse on positive pre-games. -/
noncomputable instance : Inv PGame :=
      /-
        ⊢ SetTheory.PGame → SetTheory.PGame
      -/
  ⟨by classical exact fun x => if x ≈ 0 then 0 else if 0 < x then inv' x else -inv' (-x)⟩
      /-
        🎉 no goals
      -/


noncomputable instance : Div PGame :=
  ⟨fun x y => x * y⁻¹⟩


                                                                     /-
                                                                       x : SetTheory.PGame
                                                                       h : HasEquiv.Equiv x 0
                                                                       ⊢ Eq (Inv.inv x) 0
                                                                     -/
theorem inv_eq_of_equiv_zero {x : PGame} (h : x ≈ 0) : x⁻¹ = 0 := by classical exact if_pos h
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
theorem inv_zero : (0 : PGame)⁻¹ = 0 :=
  inv_eq_of_equiv_zero (equiv_refl _)


theorem inv_eq_of_pos {x : PGame} (h : 0 < x) : x⁻¹ = inv' x := by
  /-
    x : SetTheory.PGame
    h : LT.lt 0 x
    ⊢ Eq (Inv.inv x) x.inv'
  -/
  classical exact (if_neg h.lf.not_equiv').trans (if_pos h)
  /-
    🎉 no goals
  -/


theorem inv_eq_of_lf_zero {x : PGame} (h : x ⧏ 0) : x⁻¹ = -inv' (-x) := by
  /-
    x : SetTheory.PGame
    h : x.LF 0
    ⊢ Eq (Inv.inv x) (Neg.neg (Neg.neg x).inv')
  -/
  classical exact (if_neg h.not_equiv).trans (if_neg h.not_gt)
  /-
    🎉 no goals
  -/


/-- `1⁻¹` has exactly the same moves as `1`. -/
def invOne : 1⁻¹ ≡r 1 := by
  /-
    ⊢ (Inv.inv 1).Relabelling 1
  -/
  rw [inv_eq_of_pos PGame.zero_lt_one]
  /-
    ⊢ (SetTheory.PGame.inv' 1).Relabelling 1
  -/
  exact inv'One
  /-
    🎉 no goals
  -/


theorem inv_one_equiv : (1⁻¹ : PGame) ≈ 1 :=
  invOne.equiv


