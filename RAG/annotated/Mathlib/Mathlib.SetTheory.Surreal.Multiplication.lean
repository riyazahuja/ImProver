/-- The nontrivial part of P1 in [SchleicherStoll] says that the left options of `x * y` are less
  than the right options, and this is the general form of these statements. -/
def P1 (x₁ x₂ x₃ y₁ y₂ y₃ : PGame) :=
  ⟦x₁ * y₁⟧ + ⟦x₂ * y₂⟧ - ⟦x₁ * y₂⟧ < ⟦x₃ * y₁⟧ + ⟦x₂ * y₃⟧ - (⟦x₃ * y₃⟧ : Game)


/-- The proposition P2, without numericity assumptions. -/
def P2 (x₁ x₂ y : PGame) := x₁ ≈ x₂ → ⟦x₁ * y⟧ = (⟦x₂ * y⟧ : Game)


/-- The proposition P3, without the `x₁ < x₂` and `y₁ < y₂` assumptions. -/
def P3 (x₁ x₂ y₁ y₂ : PGame) := ⟦x₁ * y₂⟧ + ⟦x₂ * y₁⟧ < ⟦x₁ * y₁⟧ + (⟦x₂ * y₂⟧ : Game)


/-- The proposition P4, without numericity assumptions. In the references, the second part of the
  conjunction is stated as `∀ j, P3 x₁ x₂ y (y.moveRight j)`, which is equivalent to our statement
  by `P3_comm` and `P3_neg`. We choose to state everything in terms of left options for uniform
  treatment. -/
def P4 (x₁ x₂ y : PGame) :=
  x₁ < x₂ → (∀ i, P3 x₁ x₂ (y.moveLeft i) y) ∧ ∀ j, P3 x₁ x₂ ((-y).moveLeft j) (-y)


/-- The conjunction of P2 and P4. -/
def P24 (x₁ x₂ y : PGame) : Prop := P2 x₁ x₂ y ∧ P4 x₁ x₂ y


lemma P3_comm : P3 x₁ x₂ y₁ y₂ ↔ P3 y₁ y₂ x₁ x₂ := by
  /-
    x₁ x₂ y₁ y₂ : SetTheory.PGame
    ⊢ Iff (Surreal.Multiplication.P3 x₁ x₂ y₁ y₂) (Surreal.Multiplication.P3 y₁ y₂ …
  -/
  rw [P3, P3, add_comm]
  /-
    x₁ x₂ y₁ y₂ : SetTheory.PGame
    ⊢ Iff (LT.lt (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x₂ y₁)) …
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
  congr! 2 <;> rw [quot_mul_comm]
               /-
                 🎉 no goals
               -/


lemma P3.trans (h₁ : P3 x₁ x₂ y₁ y₂) (h₂ : P3 x₂ x₃ y₁ y₂) : P3 x₁ x₃ y₁ y₂ := by
  /-
    x₁ x₂ x₃ y₁ y₂ : SetTheory.PGame
    h₁ : Surreal.Multiplication.P3 x₁ x₂ y₁ y₂
    h₂ : Surreal.Multiplication.P3 x₂ x₃ y₁ y₂
    ⊢ Surreal.Multiplication.P3 x₁ x₃ y₁ y₂
  -/
  rw [P3] at h₁ h₂
  /-
    x₁ x₂ x₃ y₁ y₂ : SetTheory.PGame
    h₁ : LT.lt (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x₁ y₂)) ( …
    h₂ : LT.lt (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x₂ y₂)) ( …
    ⊢ Surreal.Multiplication.P3 x₁ x₃ y₁ y₂
  -/
  rw [P3, ← add_lt_add_iff_left (⟦x₂ * y₁⟧ + ⟦x₂ * y₂⟧)]
  /-
    x₁ x₂ x₃ y₁ y₂ : SetTheory.PGame
    h₁ : LT.lt (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x₁ y₂)) ( …
    h₂ : LT.lt (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x₂ y₂)) ( …
    ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x …
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
  convert add_lt_add h₁ h₂ using 1 <;> abel
                                       /-
                                         🎉 no goals
                                       -/


lemma P3_neg : P3 x₁ x₂ y₁ y₂ ↔ P3 (-x₂) (-x₁) y₁ y₂ := by
  /-
    x₁ x₂ y₁ y₂ : SetTheory.PGame
    ⊢ Iff (Surreal.Multiplication.P3 x₁ x₂ y₁ y₂) (Surreal.Multiplication.P3 (Neg. …
  -/
  simp_rw [P3, quot_neg_mul]
  /-
    x₁ x₂ y₁ y₂ : SetTheory.PGame
    ⊢ Iff (LT.lt (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x₁ y₂)) …
  -/
  rw [← _root_.neg_lt_neg_iff]
  /-
    x₁ x₂ y₁ y₂ : SetTheory.PGame
    ⊢ Iff (LT.lt (Neg.neg (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMu …
  -/
  abel_nf
  /-
    🎉 no goals
  -/


lemma P2_neg_left : P2 x₁ x₂ y ↔ P2 (-x₂) (-x₁) y := by
  /-
    x₁ x₂ y : SetTheory.PGame
    ⊢ Iff (Surreal.Multiplication.P2 x₁ x₂ y) (Surreal.Multiplication.P2 (Neg.neg  …
  -/
  rw [P2, P2]
  /-
    x₁ x₂ y : SetTheory.PGame
    ⊢ Iff (HasEquiv.Equiv x₁ x₂ → Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMu …
  -/
  constructor
    /-
      case mp
      x₁ x₂ y : SetTheory.PGame
      ⊢ (HasEquiv.Equiv x₁ x₂ → Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x₁ …
    -/
  · rw [quot_neg_mul, quot_neg_mul, eq_comm, neg_inj, neg_equiv_neg_iff, PGame.equiv_comm]
    /-
      case mp
      x₁ x₂ y : SetTheory.PGame
      ⊢ (HasEquiv.Equiv x₂ x₁ → Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x₂ …
    -/
    exact (· ·)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      x₁ x₂ y : SetTheory.PGame
      ⊢ (HasEquiv.Equiv (Neg.neg x₂) (Neg.neg x₁) → Eq (Quotient.mk SetTheory.PGame. …
    -/
  · rw [PGame.equiv_comm, neg_equiv_neg_iff, quot_neg_mul, quot_neg_mul, neg_inj, eq_comm]
    /-
      case mpr
      x₁ x₂ y : SetTheory.PGame
      ⊢ (HasEquiv.Equiv x₁ x₂ → Eq (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x₁ …
    -/
    exact (· ·)
    /-
      🎉 no goals
    -/


lemma P2_neg_right : P2 x₁ x₂ y ↔ P2 x₁ x₂ (-y) := by
  /-
    x₁ x₂ y : SetTheory.PGame
    ⊢ Iff (Surreal.Multiplication.P2 x₁ x₂ y) (Surreal.Multiplication.P2 x₁ x₂ (Ne …
  -/
  rw [P2, P2, quot_mul_neg, quot_mul_neg, neg_inj]
  /-
    🎉 no goals
  -/


lemma P4_neg_left : P4 x₁ x₂ y ↔ P4 (-x₂) (-x₁) y := by
  /-
    x₁ x₂ y : SetTheory.PGame
    ⊢ Iff (Surreal.Multiplication.P4 x₁ x₂ y) (Surreal.Multiplication.P4 (Neg.neg  …
  -/
  simp_rw [P4, PGame.neg_lt_neg_iff, moveLeft_neg, ← P3_neg]
  /-
    🎉 no goals
  -/


lemma P4_neg_right : P4 x₁ x₂ y ↔ P4 x₁ x₂ (-y) := by
  /-
    x₁ x₂ y : SetTheory.PGame
    ⊢ Iff (Surreal.Multiplication.P4 x₁ x₂ y) (Surreal.Multiplication.P4 x₁ x₂ (Ne …
  -/
  rw [P4, P4, neg_neg, and_comm]
  /-
    🎉 no goals
  -/


                                                           /-
                                                             x₁ x₂ y : SetTheory.PGame
                                                             ⊢ Iff (Surreal.Multiplication.P24 x₁ x₂ y) (Surreal.Multiplication.P24 (Neg.ne …
                                                           -/
lemma P24_neg_left : P24 x₁ x₂ y ↔ P24 (-x₂) (-x₁) y := by rw [P24, P24, P2_neg_left, P4_neg_left]
                                                           /-
                                                             🎉 no goals
                                                           -/

                                                         /-
                                                           x₁ x₂ y : SetTheory.PGame
                                                           ⊢ Iff (Surreal.Multiplication.P24 x₁ x₂ y) (Surreal.Multiplication.P24 x₁ x₂ ( …
                                                         -/
lemma P24_neg_right : P24 x₁ x₂ y ↔ P24 x₁ x₂ (-y) := by rw [P24, P24, P2_neg_right, P4_neg_right]
                                                         /-
                                                           🎉 no goals
                                                         -/


lemma mulOption_lt_iff_P1 {i j k l} :
    (⟦mulOption x y i k⟧ : Game) < -⟦mulOption x (-y) j l⟧ ↔
    P1 (x.moveLeft i) x (x.moveLeft j) y (y.moveLeft k) (-(-y).moveLeft l) := by
  /-
    x y : SetTheory.PGame
    i j : x.LeftMoves
    k : y.LeftMoves
    l : (Neg.neg y).LeftMoves
    ⊢ Iff (LT.lt (Quotient.mk SetTheory.PGame.setoid (x.mulOption y i k)) (Neg.neg …
  -/
  dsimp only [P1, mulOption, quot_sub, quot_add]
  /-
    x y : SetTheory.PGame
    i j : x.LeftMoves
    k : y.LeftMoves
    l : (Neg.neg y).LeftMoves
    ⊢ Iff (LT.lt (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.h …
  -/
  simp_rw [neg_sub', neg_add, quot_mul_neg, neg_neg]
  /-
    🎉 no goals
  -/


lemma mulOption_lt_mul_iff_P3 {i j} :
    ⟦mulOption x y i j⟧ < (⟦x * y⟧ : Game) ↔ P3 (x.moveLeft i) x (y.moveLeft j) y := by
  /-
    x y : SetTheory.PGame
    i : x.LeftMoves
    j : y.LeftMoves
    ⊢ Iff (LT.lt (Quotient.mk SetTheory.PGame.setoid (x.mulOption y i j)) (Quotien …
  -/
  dsimp only [mulOption, quot_sub, quot_add]
  /-
    x y : SetTheory.PGame
    i : x.LeftMoves
    j : y.LeftMoves
    ⊢ Iff (LT.lt (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.h …
  -/
  exact sub_lt_iff_lt_add'
  /-
    🎉 no goals
  -/


lemma P1_of_eq (he : x₁ ≈ x₃) (h₁ : P2 x₁ x₃ y₁) (h₃ : P2 x₁ x₃ y₃) (h3 : P3 x₁ x₂ y₂ y₃) :
    P1 x₁ x₂ x₃ y₁ y₂ y₃ := by
  /-
    x₁ x₂ x₃ y₁ y₂ y₃ : SetTheory.PGame
    he : HasEquiv.Equiv x₁ x₃
    h₁ : Surreal.Multiplication.P2 x₁ x₃ y₁
    h₃ : Surreal.Multiplication.P2 x₁ x₃ y₃
    h3 : Surreal.Multiplication.P3 x₁ x₂ y₂ y₃
    ⊢ Surreal.Multiplication.P1 x₁ x₂ x₃ y₁ y₂ y₃
  -/
  rw [P1, ← h₁ he, ← h₃ he, sub_lt_sub_iff]
  /-
    x₁ x₂ x₃ y₁ y₂ y₃ : SetTheory.PGame
    he : HasEquiv.Equiv x₁ x₃
    h₁ : Surreal.Multiplication.P2 x₁ x₃ y₁
    h₃ : Surreal.Multiplication.P2 x₁ x₃ y₃
    h3 : Surreal.Multiplication.P3 x₁ x₂ y₂ y₃
    ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x …
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
  convert add_lt_add_left h3 ⟦x₁ * y₁⟧ using 1 <;> abel
                                                   /-
                                                     🎉 no goals
                                                   -/


lemma P1_of_lt (h₁ : P3 x₃ x₂ y₂ y₃) (h₂ : P3 x₁ x₃ y₂ y₁) : P1 x₁ x₂ x₃ y₁ y₂ y₃ := by
  /-
    x₁ x₂ x₃ y₁ y₂ y₃ : SetTheory.PGame
    h₁ : Surreal.Multiplication.P3 x₃ x₂ y₂ y₃
    h₂ : Surreal.Multiplication.P3 x₁ x₃ y₂ y₁
    ⊢ Surreal.Multiplication.P1 x₁ x₂ x₃ y₁ y₂ y₃
  -/
  rw [P1, sub_lt_sub_iff, ← add_lt_add_iff_left ⟦x₃ * y₂⟧]
  /-
    x₁ x₂ x₃ y₁ y₂ y₃ : SetTheory.PGame
    h₁ : Surreal.Multiplication.P3 x₃ x₂ y₂ y₃
    h₂ : Surreal.Multiplication.P3 x₁ x₃ y₂ y₁
    ⊢ LT.lt (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x₃ y₂)) (HAd …
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
  convert add_lt_add h₁ h₂ using 1 <;> abel
                                       /-
                                         🎉 no goals
                                       -/


/-- The type of lists of arguments for P1, P2, and P4. -/
inductive Args : Type (u+1)
  | P1 (x y : PGame.{u}) : Args
  | P24 (x₁ x₂ y : PGame.{u}) : Args


/-- The multiset associated to a list of arguments. -/
def Args.toMultiset : Args → Multiset PGame
  | (Args.P1 x y) => {x, y}
  | (Args.P24 x₁ x₂ y) => {x₁, x₂, y}


/-- A list of arguments is numeric if all the arguments are. -/
def Args.Numeric (a : Args) := ∀ x ∈ a.toMultiset, SetTheory.PGame.Numeric x


lemma Args.numeric_P1 {x y} : (Args.P1 x y).Numeric ↔ x.Numeric ∧ y.Numeric := by
  /-
    x y : SetTheory.PGame
    ⊢ Iff (Surreal.Multiplication.Args.P1 x y).Numeric (And x.Numeric y.Numeric)
  -/
  simp [Args.Numeric, Args.toMultiset]
  /-
    🎉 no goals
  -/


lemma Args.numeric_P24 {x₁ x₂ y} :
    (Args.P24 x₁ x₂ y).Numeric ↔ x₁.Numeric ∧ x₂.Numeric ∧ y.Numeric := by
  /-
    x₁ x₂ y : SetTheory.PGame
    ⊢ Iff (Surreal.Multiplication.Args.P24 x₁ x₂ y).Numeric (And x₁.Numeric (And x …
  -/
  simp [Args.Numeric, Args.toMultiset]
  /-
    🎉 no goals
  -/


/-- The relation specifying when a list of (pregame) arguments is considered simpler than another:
  `ArgsRel a₁ a₂` is true if `a₁`, considered as a multiset, can be obtained from `a₂` by
  repeatedly removing a pregame from `a₂` and adding back one or two options of the pregame. -/
def ArgsRel := InvImage (TransGen <| CutExpand IsOption) Args.toMultiset


/-- `ArgsRel` is well-founded. -/
theorem argsRel_wf : WellFounded ArgsRel := InvImage.wf _ wf_isOption.cutExpand.transGen


/-- The statement that we will show by induction using the well-founded relation `ArgsRel`. -/
def P124 : Args → Prop
  | (Args.P1 x y) => Numeric (x * y)
  | (Args.P24 x₁ x₂ y) => P24 x₁ x₂ y


/-- The property that all arguments are numeric is leftward-closed under `ArgsRel`. -/
lemma ArgsRel.numeric_closed {a' a} : ArgsRel a' a → a.Numeric → a'.Numeric :=
  TransGen.closed' <| @cutExpand_closed _ IsOption ⟨wf_isOption.isIrrefl.1⟩ _ Numeric.isOption


/-- A specialized induction hypothesis used to prove P1. -/
def IH1 (x y : PGame) : Prop :=
  ∀ ⦃x₁ x₂ y'⦄, IsOption x₁ x → IsOption x₂ x → (y' = y ∨ IsOption y' y) → P24 x₁ x₂ y'


lemma ih1_neg_left : IH1 x y → IH1 (-x) y :=
  fun h x₁ x₂ y' h₁ h₂ hy ↦ by
    /-
      x y : SetTheory.PGame
      h : Surreal.Multiplication.IH1 x y
      x₁ x₂ y' : SetTheory.PGame
      h₁ : x₁.IsOption (Neg.neg x)
      h₂ : x₂.IsOption (Neg.neg x)
      hy : Or (Eq y' y) (y'.IsOption y)
      ⊢ Surreal.Multiplication.P24 x₁ x₂ y'
    -/
    rw [isOption_neg] at h₁ h₂
    /-
      x y : SetTheory.PGame
      h : Surreal.Multiplication.IH1 x y
      x₁ x₂ y' : SetTheory.PGame
      h₁ : (Neg.neg x₁).IsOption x
      h₂ : (Neg.neg x₂).IsOption x
      hy : Or (Eq y' y) (y'.IsOption y)
      ⊢ Surreal.Multiplication.P24 x₁ x₂ y'
    -/
    exact P24_neg_left.2 (h h₂ h₁ hy)
    /-
      🎉 no goals
    -/


lemma ih1_neg_right : IH1 x y → IH1 x (-y) :=
  fun h x₁ x₂ y' ↦ by
    /-
      x y : SetTheory.PGame
      h : Surreal.Multiplication.IH1 x y
      x₁ x₂ y' : SetTheory.PGame
      ⊢ x₁.IsOption x → x₂.IsOption x → Or (Eq y' (Neg.neg y)) (y'.IsOption (Neg.neg …
    -/
    rw [← neg_eq_iff_eq_neg, isOption_neg, P24_neg_right]
    /-
      x y : SetTheory.PGame
      h : Surreal.Multiplication.IH1 x y
      x₁ x₂ y' : SetTheory.PGame
      ⊢ x₁.IsOption x → x₂.IsOption x → Or (Eq (Neg.neg y') y) ((Neg.neg y').IsOptio …
    -/
    apply h
    /-
      🎉 no goals
    -/


lemma numeric_option_mul (ih : ∀ a, ArgsRel a (Args.P1 x y) → P124 a) (h : IsOption x' x) :
    (x' * y).Numeric :=
  ih (Args.P1 x' y) (TransGen.single <| cutExpand_pair_left h)


lemma numeric_mul_option (ih : ∀ a, ArgsRel a (Args.P1 x y) → P124 a) (h : IsOption y' y) :
    (x * y').Numeric :=
  ih (Args.P1 x y') (TransGen.single <| cutExpand_pair_right h)


lemma numeric_option_mul_option (ih : ∀ a, ArgsRel a (Args.P1 x y) → P124 a) (hx : IsOption x' x)
    (hy : IsOption y' y) : (x' * y').Numeric :=
  ih (Args.P1 x' y') ((TransGen.single <| cutExpand_pair_right hy).tail <| cutExpand_pair_left hx)


lemma ih1 (ih : ∀ a, ArgsRel a (Args.P1 x y) → P124 a) : IH1 x y := by
  /-
    x y : SetTheory.PGame
    ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
    ⊢ Surreal.Multiplication.IH1 x y
  -/
  rintro x₁ x₂ y' h₁ h₂ (rfl|hy) <;> apply ih (Args.P24 _ _ _)
  /-
    case inl
    x x₁ x₂ y' : SetTheory.PGame
    h₁ : x₁.IsOption x
    h₂ : x₂.IsOption x
    ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
    ⊢ Surreal.Multiplication.ArgsRel (Surreal.Multiplication.Args.P24 x₁ x₂ y') (S …
  -/
  on_goal 2 => refine TransGen.tail ?_ (cutExpand_pair_right hy)
  /-
    case inl
    x x₁ x₂ y' : SetTheory.PGame
    h₁ : x₁.IsOption x
    h₂ : x₂.IsOption x
    ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
    ⊢ Surreal.Multiplication.ArgsRel (Surreal.Multiplication.Args.P24 x₁ x₂ y') (S …
  -/
  all_goals exact TransGen.single (cutExpand_double_left h₁ h₂)
  /-
    🎉 no goals
  -/


lemma ih1_swap (ih : ∀ a, ArgsRel a (Args.P1 x y) → P124 a) : IH1 y x := ih1 <| by
  /-
    x y : SetTheory.PGame
    ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
    ⊢ ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Surre …
  -/
  simp_rw [ArgsRel, InvImage, Args.toMultiset, Multiset.pair_comm] at ih ⊢
  /-
    x y : SetTheory.PGame
    ih : ∀ (a : Surreal.Multiplication.Args), Relation.TransGen (Relation.CutExpan …
    ⊢ ∀ (a : Surreal.Multiplication.Args), Relation.TransGen (Relation.CutExpand S …
  -/
  exact ih
  /-
    🎉 no goals
  -/


lemma P3_of_ih (hy : Numeric y) (ihyx : IH1 y x) (i k l) :
    P3 (x.moveLeft i) x (y.moveLeft k) (-(-y).moveLeft l) :=
  P3_comm.2 <| ((ihyx (IsOption.moveLeft k) (isOption_neg.1 <| .moveLeft l) <| Or.inl rfl).2
        /-
          x y : SetTheory.PGame
          hy : y.Numeric
          ihyx : Surreal.Multiplication.IH1 y x
          i : x.LeftMoves
          k : y.LeftMoves
          l : (Neg.neg y).LeftMoves
          ⊢ LT.lt (y.moveLeft k) (Neg.neg ((Neg.neg y).moveLeft l))
        -/
    (by rw [moveLeft_neg, neg_neg]; apply hy.left_lt_right)).1 i
                                    /-
                                      🎉 no goals
                                    -/


lemma P24_of_ih (ihxy : IH1 x y) (i j) : P24 (x.moveLeft i) (x.moveLeft j) y :=
  ihxy (IsOption.moveLeft i) (IsOption.moveLeft j) (Or.inl rfl)


lemma mulOption_lt_of_lt (hy : y.Numeric) (ihxy : IH1 x y) (ihyx : IH1 y x) (i j k l)
    (h : x.moveLeft i < x.moveLeft j) :
    (⟦mulOption x y i k⟧ : Game) < -⟦mulOption x (-y) j l⟧ :=
  mulOption_lt_iff_P1.2 <| P1_of_lt (P3_of_ih hy ihyx j k l) <| ((P24_of_ih ihxy i j).2 h).1 k


lemma mulOption_lt (hx : x.Numeric) (hy : y.Numeric) (ihxy : IH1 x y) (ihyx : IH1 y x) (i j k l) :
    (⟦mulOption x y i k⟧ : Game) < -⟦mulOption x (-y) j l⟧ := by
  /-
    x y : SetTheory.PGame
    hx : x.Numeric
    hy : y.Numeric
    ihxy : Surreal.Multiplication.IH1 x y
    ihyx : Surreal.Multiplication.IH1 y x
    i j : x.LeftMoves
    k : y.LeftMoves
    l : (Neg.neg y).LeftMoves
    ⊢ LT.lt (Quotient.mk SetTheory.PGame.setoid (x.mulOption y i k)) (Neg.neg (Quo …
  -/
  obtain (h|h|h) := lt_or_equiv_or_gt (hx.moveLeft i) (hx.moveLeft j)
    /-
      case inl
      x y : SetTheory.PGame
      hx : x.Numeric
      hy : y.Numeric
      ihxy : Surreal.Multiplication.IH1 x y
      ihyx : Surreal.Multiplication.IH1 y x
      i j : x.LeftMoves
      k : y.LeftMoves
      l : (Neg.neg y).LeftMoves
      h : LT.lt (x.moveLeft i) (x.moveLeft j)
      ⊢ LT.lt (Quotient.mk SetTheory.PGame.setoid (x.mulOption y i k)) (Neg.neg (Quo …
    -/
  · exact mulOption_lt_of_lt hy ihxy ihyx i j k l h
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      x y : SetTheory.PGame
      hx : x.Numeric
      hy : y.Numeric
      ihxy : Surreal.Multiplication.IH1 x y
      ihyx : Surreal.Multiplication.IH1 y x
      i j : x.LeftMoves
      k : y.LeftMoves
      l : (Neg.neg y).LeftMoves
      h : HasEquiv.Equiv (x.moveLeft i) (x.moveLeft j)
      ⊢ LT.lt (Quotient.mk SetTheory.PGame.setoid (x.mulOption y i k)) (Neg.neg (Quo …
    -/
  · have ml := @IsOption.moveLeft
    exact mulOption_lt_iff_P1.2 (P1_of_eq h (P24_of_ih ihxy i j).1
      (ihxy (ml i) (ml j) <| Or.inr <| isOption_neg.1 <| ml l).1 <| P3_of_ih hy ihyx i k l)
    /-
      case inr.inr
      x y : SetTheory.PGame
      hx : x.Numeric
      hy : y.Numeric
      ihxy : Surreal.Multiplication.IH1 x y
      ihyx : Surreal.Multiplication.IH1 y x
      i j : x.LeftMoves
      k : y.LeftMoves
      l : (Neg.neg y).LeftMoves
      h : LT.lt (x.moveLeft j) (x.moveLeft i)
      ⊢ LT.lt (Quotient.mk SetTheory.PGame.setoid (x.mulOption y i k)) (Neg.neg (Quo …
    -/
  · rw [mulOption_neg_neg, lt_neg]
    /-
      case inr.inr
      x y : SetTheory.PGame
      hx : x.Numeric
      hy : y.Numeric
      ihxy : Surreal.Multiplication.IH1 x y
      ihyx : Surreal.Multiplication.IH1 y x
      i j : x.LeftMoves
      k : y.LeftMoves
      l : (Neg.neg y).LeftMoves
      h : LT.lt (x.moveLeft j) (x.moveLeft i)
      ⊢ LT.lt (Quotient.mk SetTheory.PGame.setoid (x.mulOption (Neg.neg y) j l)) (Ne …
    -/
    exact mulOption_lt_of_lt hy.neg (ih1_neg_right ihxy) (ih1_neg_left ihyx) j i l _ h
    /-
      🎉 no goals
    -/


/-- P1 follows from the induction hypothesis. -/
theorem P1_of_ih (ih : ∀ a, ArgsRel a (Args.P1 x y) → P124 a) (hx : x.Numeric) (hy : y.Numeric) :
    (x * y).Numeric := by
  /-
    x y : SetTheory.PGame
    ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
    hx : x.Numeric
    hy : y.Numeric
    ⊢ (HMul.hMul x y).Numeric
  -/
  have ihxy := ih1 ih
  /-
    x y : SetTheory.PGame
    ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
    hx : x.Numeric
    hy : y.Numeric
    ihxy : Surreal.Multiplication.IH1 x y
    ⊢ (HMul.hMul x y).Numeric
  -/
  have ihyx := ih1_swap ih
  /-
    x y : SetTheory.PGame
    ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
    hx : x.Numeric
    hy : y.Numeric
    ihxy : Surreal.Multiplication.IH1 x y
    ihyx : Surreal.Multiplication.IH1 y x
    ⊢ (HMul.hMul x y).Numeric
  -/
  have ihxyn := ih1_neg_left (ih1_neg_right ihxy)
  /-
    x y : SetTheory.PGame
    ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
    hx : x.Numeric
    hy : y.Numeric
    ihxy : Surreal.Multiplication.IH1 x y
    ihyx : Surreal.Multiplication.IH1 y x
    ihxyn : Surreal.Multiplication.IH1 (Neg.neg x) (Neg.neg y)
    ⊢ (HMul.hMul x y).Numeric
  -/
  have ihyxn := ih1_neg_left (ih1_neg_right ihyx)
  /-
    x y : SetTheory.PGame
    ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
    hx : x.Numeric
    hy : y.Numeric
    ihxy : Surreal.Multiplication.IH1 x y
    ihyx : Surreal.Multiplication.IH1 y x
    ihxyn : Surreal.Multiplication.IH1 (Neg.neg x) (Neg.neg y)
    ihyxn : Surreal.Multiplication.IH1 (Neg.neg y) (Neg.neg x)
    ⊢ (HMul.hMul x y).Numeric
  -/
  refine numeric_def.mpr ⟨?_, ?_, ?_⟩
    /-
      case refine_1
      x y : SetTheory.PGame
      ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
      hx : x.Numeric
      hy : y.Numeric
      ihxy : Surreal.Multiplication.IH1 x y
      ihyx : Surreal.Multiplication.IH1 y x
      ihxyn : Surreal.Multiplication.IH1 (Neg.neg x) (Neg.neg y)
      ihyxn : Surreal.Multiplication.IH1 (Neg.neg y) (Neg.neg x)
      ⊢ ∀ (i : (HMul.hMul x y).LeftMoves) (j : (HMul.hMul x y).RightMoves), LT.lt (( …
    -/
  · simp_rw [lt_iff_game_lt]
    /-
      case refine_1
      x y : SetTheory.PGame
      ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
      hx : x.Numeric
      hy : y.Numeric
      ihxy : Surreal.Multiplication.IH1 x y
      ihyx : Surreal.Multiplication.IH1 y x
      ihxyn : Surreal.Multiplication.IH1 (Neg.neg x) (Neg.neg y)
      ihyxn : Surreal.Multiplication.IH1 (Neg.neg y) (Neg.neg x)
      ⊢ ∀ (i : (HMul.hMul x y).LeftMoves) (j : (HMul.hMul x y).RightMoves), LT.lt (Q …
    -/
    intro i
    /-
      case refine_1
      x y : SetTheory.PGame
      ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
      hx : x.Numeric
      hy : y.Numeric
      ihxy : Surreal.Multiplication.IH1 x y
      ihyx : Surreal.Multiplication.IH1 y x
      ihxyn : Surreal.Multiplication.IH1 (Neg.neg x) (Neg.neg y)
      ihyxn : Surreal.Multiplication.IH1 (Neg.neg y) (Neg.neg x)
      i : (HMul.hMul x y).LeftMoves
      ⊢ ∀ (j : (HMul.hMul x y).RightMoves), LT.lt (Quotient.mk SetTheory.PGame.setoi …
    -/
    rw [rightMoves_mul_iff]
    /-
      case refine_1
      x y : SetTheory.PGame
      ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
      hx : x.Numeric
      hy : y.Numeric
      ihxy : Surreal.Multiplication.IH1 x y
      ihyx : Surreal.Multiplication.IH1 y x
      ihxyn : Surreal.Multiplication.IH1 (Neg.neg x) (Neg.neg y)
      ihyxn : Surreal.Multiplication.IH1 (Neg.neg y) (Neg.neg x)
      i : (HMul.hMul x y).LeftMoves
      ⊢ And (∀ (i_1 : x.LeftMoves) (j : (Neg.neg y).LeftMoves), LT.lt (Quotient.mk S …
    -/
    constructor <;> (intro j l; revert i; rw [leftMoves_mul_iff (_ > ·)]; constructor <;> intro i k)
      /-
        case refine_1.left.left
        x y : SetTheory.PGame
        ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
        hx : x.Numeric
        hy : y.Numeric
        ihxy : Surreal.Multiplication.IH1 x y
        ihyx : Surreal.Multiplication.IH1 y x
        ihxyn : Surreal.Multiplication.IH1 (Neg.neg x) (Neg.neg y)
        ihyxn : Surreal.Multiplication.IH1 (Neg.neg y) (Neg.neg x)
        j : x.LeftMoves
        l : (Neg.neg y).LeftMoves
        i : x.LeftMoves
        k : y.LeftMoves
        ⊢ GT.gt (Neg.neg (Quotient.mk SetTheory.PGame.setoid (x.mulOption (Neg.neg y)  …
      -/
    · apply mulOption_lt hx hy ihxy ihyx
      /-
        🎉 no goals
      -/
      /-
        case refine_1.left.right
        x y : SetTheory.PGame
        ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
        hx : x.Numeric
        hy : y.Numeric
        ihxy : Surreal.Multiplication.IH1 x y
        ihyx : Surreal.Multiplication.IH1 y x
        ihxyn : Surreal.Multiplication.IH1 (Neg.neg x) (Neg.neg y)
        ihyxn : Surreal.Multiplication.IH1 (Neg.neg y) (Neg.neg x)
        j : x.LeftMoves
        l : (Neg.neg y).LeftMoves
        i : (Neg.neg x).LeftMoves
        k : (Neg.neg y).LeftMoves
        ⊢ GT.gt (Neg.neg (Quotient.mk SetTheory.PGame.setoid (x.mulOption (Neg.neg y)  …
      -/
    · simp_rw [← mulOption_symm (-y), mulOption_neg_neg x]
      /-
        case refine_1.left.right
        x y : SetTheory.PGame
        ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
        hx : x.Numeric
        hy : y.Numeric
        ihxy : Surreal.Multiplication.IH1 x y
        ihyx : Surreal.Multiplication.IH1 y x
        ihxyn : Surreal.Multiplication.IH1 (Neg.neg x) (Neg.neg y)
        ihyxn : Surreal.Multiplication.IH1 (Neg.neg y) (Neg.neg x)
        j : x.LeftMoves
        l : (Neg.neg y).LeftMoves
        i : (Neg.neg x).LeftMoves
        k : (Neg.neg y).LeftMoves
        ⊢ GT.gt (Neg.neg (Quotient.mk SetTheory.PGame.setoid ((Neg.neg y).mulOption (N …
      -/
      apply mulOption_lt hy.neg hx.neg ihyxn ihxyn
      /-
        🎉 no goals
      -/
      /-
        case refine_1.right.left
        x y : SetTheory.PGame
        ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
        hx : x.Numeric
        hy : y.Numeric
        ihxy : Surreal.Multiplication.IH1 x y
        ihyx : Surreal.Multiplication.IH1 y x
        ihxyn : Surreal.Multiplication.IH1 (Neg.neg x) (Neg.neg y)
        ihyxn : Surreal.Multiplication.IH1 (Neg.neg y) (Neg.neg x)
        j : (Neg.neg x).LeftMoves
        l : y.LeftMoves
        i : x.LeftMoves
        k : y.LeftMoves
        ⊢ GT.gt (Neg.neg (Quotient.mk SetTheory.PGame.setoid ((Neg.neg x).mulOption y  …
      -/
    · simp only [← mulOption_symm y]
      /-
        case refine_1.right.left
        x y : SetTheory.PGame
        ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
        hx : x.Numeric
        hy : y.Numeric
        ihxy : Surreal.Multiplication.IH1 x y
        ihyx : Surreal.Multiplication.IH1 y x
        ihxyn : Surreal.Multiplication.IH1 (Neg.neg x) (Neg.neg y)
        ihyxn : Surreal.Multiplication.IH1 (Neg.neg y) (Neg.neg x)
        j : (Neg.neg x).LeftMoves
        l : y.LeftMoves
        i : x.LeftMoves
        k : y.LeftMoves
        ⊢ GT.gt (Neg.neg (Quotient.mk SetTheory.PGame.setoid (y.mulOption (Neg.neg x)  …
      -/
      apply mulOption_lt hy hx ihyx ihxy
      /-
        🎉 no goals
      -/
      /-
        case refine_1.right.right
        x y : SetTheory.PGame
        ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
        hx : x.Numeric
        hy : y.Numeric
        ihxy : Surreal.Multiplication.IH1 x y
        ihyx : Surreal.Multiplication.IH1 y x
        ihxyn : Surreal.Multiplication.IH1 (Neg.neg x) (Neg.neg y)
        ihyxn : Surreal.Multiplication.IH1 (Neg.neg y) (Neg.neg x)
        j : (Neg.neg x).LeftMoves
        l : y.LeftMoves
        i : (Neg.neg x).LeftMoves
        k : (Neg.neg y).LeftMoves
        ⊢ GT.gt (Neg.neg (Quotient.mk SetTheory.PGame.setoid ((Neg.neg x).mulOption y  …
      -/
    · rw [mulOption_neg_neg y]
      /-
        case refine_1.right.right
        x y : SetTheory.PGame
        ih : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Su …
        hx : x.Numeric
        hy : y.Numeric
        ihxy : Surreal.Multiplication.IH1 x y
        ihyx : Surreal.Multiplication.IH1 y x
        ihxyn : Surreal.Multiplication.IH1 (Neg.neg x) (Neg.neg y)
        ihyxn : Surreal.Multiplication.IH1 (Neg.neg y) (Neg.neg x)
        j : (Neg.neg x).LeftMoves
        l : y.LeftMoves
        i : (Neg.neg x).LeftMoves
        k : (Neg.neg y).LeftMoves
        ⊢ GT.gt (Neg.neg (Quotient.mk SetTheory.PGame.setoid ((Neg.neg x).mulOption (N …
      -/
      apply mulOption_lt hx.neg hy.neg ihxyn ihyxn
      /-
        🎉 no goals
      -/
  all_goals
    cases x; cases y
    rintro (⟨i,j⟩|⟨i,j⟩) <;>
    refine ((numeric_option_mul ih ?_).add <| numeric_mul_option ih ?_).sub
      (numeric_option_mul_option ih ?_ ?_) <;>
    solve_by_elim [IsOption.mk_left, IsOption.mk_right]


/-- A specialized induction hypothesis used to prove P2 and P4. -/
def IH24 (x₁ x₂ y : PGame) : Prop :=
  ∀ ⦃z⦄, (IsOption z x₁ → P24 z x₂ y) ∧ (IsOption z x₂ → P24 x₁ z y) ∧ (IsOption z y → P24 x₁ x₂ z)


/-- A specialized induction hypothesis used to prove P4. -/
def IH4 (x₁ x₂ y : PGame) : Prop :=
  ∀ ⦃z w⦄, IsOption w y → (IsOption z x₁ → P2 z x₂ w) ∧ (IsOption z x₂ → P2 x₁ z w)


lemma ih₁₂ (ih' : ∀ a, ArgsRel a (Args.P24 x₁ x₂ y) → P124 a) : IH24 x₁ x₂ y := by
  /-
    x₁ x₂ y : SetTheory.PGame
    ih' : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (S …
    ⊢ Surreal.Multiplication.IH24 x₁ x₂ y
  -/
  rw [IH24]
  /-
    x₁ x₂ y : SetTheory.PGame
    ih' : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (S …
    ⊢ ∀ ⦃z : SetTheory.PGame⦄, And (z.IsOption x₁ → Surreal.Multiplication.P24 z x …
  -/
  refine fun z ↦ ⟨?_, ?_, ?_⟩ <;>
    /-
      case refine_1
      x₁ x₂ y : SetTheory.PGame
      ih' : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (S …
      z : SetTheory.PGame
      ⊢ z.IsOption x₁ → Surreal.Multiplication.P24 z x₂ y
    -/
    refine fun h ↦ ih' (Args.P24 _ _ _) (TransGen.single ?_)
    /-
      case refine_1
      x₁ x₂ y : SetTheory.PGame
      ih' : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (S …
      z : SetTheory.PGame
      h : z.IsOption x₁
      ⊢ Relation.CutExpand SetTheory.PGame.IsOption (Surreal.Multiplication.Args.P24 …
    -/
  · exact (cutExpand_add_right {y}).2 (cutExpand_pair_left h)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      x₁ x₂ y : SetTheory.PGame
      ih' : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (S …
      z : SetTheory.PGame
      h : z.IsOption x₂
      ⊢ Relation.CutExpand SetTheory.PGame.IsOption (Surreal.Multiplication.Args.P24 …
    -/
  · exact (cutExpand_add_left {x₁}).2 (cutExpand_pair_left h)
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      x₁ x₂ y : SetTheory.PGame
      ih' : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (S …
      z : SetTheory.PGame
      h : z.IsOption y
      ⊢ Relation.CutExpand SetTheory.PGame.IsOption (Surreal.Multiplication.Args.P24 …
    -/
  · exact (cutExpand_add_left {x₁}).2 (cutExpand_pair_right h)
    /-
      🎉 no goals
    -/


lemma ih₂₁ (ih' : ∀ a, ArgsRel a (Args.P24 x₁ x₂ y) → P124 a) : IH24 x₂ x₁ y := ih₁₂ <| by
  /-
    x₁ x₂ y : SetTheory.PGame
    ih' : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (S …
    ⊢ ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (Surre …
  -/
  simp_rw [ArgsRel, InvImage, Args.toMultiset, Multiset.pair_comm] at ih' ⊢
  /-
    x₁ x₂ y : SetTheory.PGame
    ih' : ∀ (a : Surreal.Multiplication.Args), Relation.TransGen (Relation.CutExpa …
    ⊢ ∀ (a : Surreal.Multiplication.Args), Relation.TransGen (Relation.CutExpand S …
  -/
  suffices {x₁, y, x₂} = {x₂, y, x₁} by rwa [← this]
  /-
    x₁ x₂ y : SetTheory.PGame
    ih' : ∀ (a : Surreal.Multiplication.Args), Relation.TransGen (Relation.CutExpa …
    ⊢ Eq (Insert.insert x₁ (Insert.insert y (Singleton.singleton x₂))) (Insert.ins …
  -/
  dsimp only [Multiset.insert_eq_cons, ← Multiset.singleton_add] at ih' ⊢
  /-
    x₁ x₂ y : SetTheory.PGame
    ih' : ∀ (a : Surreal.Multiplication.Args), Relation.TransGen (Relation.CutExpa …
    ⊢ Eq (HAdd.hAdd (Singleton.singleton x₁) (HAdd.hAdd (Singleton.singleton y) (S …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


lemma ih4 (ih' : ∀ a, ArgsRel a (Args.P24 x₁ x₂ y) → P124 a) : IH4 x₁ x₂ y := by
  /-
    x₁ x₂ y : SetTheory.PGame
    ih' : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (S …
    ⊢ Surreal.Multiplication.IH4 x₁ x₂ y
  -/
  refine fun z w h ↦ ⟨?_, ?_⟩
  all_goals
    intro h'
    apply (ih' (Args.P24 _ _ _) <| (TransGen.single _).tail <|
      (cutExpand_add_left {x₁}).2 <| cutExpand_pair_right h).1
    try exact (cutExpand_add_right {w}).2 <| cutExpand_pair_left h'
    try exact (cutExpand_add_right {w}).2 <| cutExpand_pair_right h'


lemma numeric_of_ih (ih' : ∀ a, ArgsRel a (Args.P24 x₁ x₂ y) → P124 a) :
    (x₁ * y).Numeric ∧ (x₂ * y).Numeric := by
  /-
    x₁ x₂ y : SetTheory.PGame
    ih' : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (S …
    ⊢ And (HMul.hMul x₁ y).Numeric (HMul.hMul x₂ y).Numeric
  -/
  constructor <;> refine ih' (Args.P1 _ _) (TransGen.single ?_)
    /-
      case left
      x₁ x₂ y : SetTheory.PGame
      ih' : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (S …
      ⊢ Relation.CutExpand SetTheory.PGame.IsOption (Surreal.Multiplication.Args.P1  …
    -/
  · exact (cutExpand_add_right {y}).2 <| (cutExpand_add_left {x₁}).2 cutExpand_zero
    /-
      🎉 no goals
    -/
    /-
      case right
      x₁ x₂ y : SetTheory.PGame
      ih' : ∀ (a : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel a (S …
      ⊢ Relation.CutExpand SetTheory.PGame.IsOption (Surreal.Multiplication.Args.P1  …
    -/
  · exact (cutExpand_add_right {x₂, y}).2 cutExpand_zero
    /-
      🎉 no goals
    -/


/-- Symmetry properties of `IH24`. -/
lemma ih24_neg : IH24 x₁ x₂ y → IH24 (-x₂) (-x₁) y ∧ IH24 x₁ x₂ (-y) := by
  /-
    x₁ x₂ y : SetTheory.PGame
    ⊢ Surreal.Multiplication.IH24 x₁ x₂ y → And (Surreal.Multiplication.IH24 (Neg. …
  -/
  simp_rw [IH24, ← P24_neg_right, isOption_neg]
  refine fun h ↦ ⟨fun z ↦ ⟨?_, ?_, ?_⟩,
    fun z ↦ ⟨(@h z).1, (@h z).2.1, P24_neg_right.2 ∘ (@h <| -z).2.2⟩⟩
  all_goals
    rw [P24_neg_left]
    simp only [neg_neg]
    first | exact (@h <| -z).2.1 | exact (@h <| -z).1 | exact (@h z).2.2


/-- Symmetry properties of `IH4`. -/
lemma ih4_neg : IH4 x₁ x₂ y → IH4 (-x₂) (-x₁) y ∧ IH4 x₁ x₂ (-y) := by
  /-
    x₁ x₂ y : SetTheory.PGame
    ⊢ Surreal.Multiplication.IH4 x₁ x₂ y → And (Surreal.Multiplication.IH4 (Neg.ne …
  -/
  simp_rw [IH4, isOption_neg]
  /-
    x₁ x₂ y : SetTheory.PGame
    ⊢ (∀ ⦃z w : SetTheory.PGame⦄, w.IsOption y → And (z.IsOption x₁ → Surreal.Mult …
  -/
  refine fun h ↦ ⟨fun z w h' ↦ ?_, fun z w h' ↦ ?_⟩
    /-
      case refine_1
      x₁ x₂ y : SetTheory.PGame
      h : ∀ ⦃z w : SetTheory.PGame⦄, w.IsOption y → And (z.IsOption x₁ → Surreal.Mul …
      z w : SetTheory.PGame
      h' : w.IsOption y
      ⊢ And ((Neg.neg z).IsOption x₂ → Surreal.Multiplication.P2 z (Neg.neg x₁) w) ( …
    -/
                                    /-
                                      🎉 no goals
                                    -/
  · convert (h h').symm using 2 <;> rw [P2_neg_left, neg_neg]
                                    /-
                                      🎉 no goals
                                    -/
    /-
      case refine_2
      x₁ x₂ y : SetTheory.PGame
      h : ∀ ⦃z w : SetTheory.PGame⦄, w.IsOption y → And (z.IsOption x₁ → Surreal.Mul …
      z w : SetTheory.PGame
      h' : (Neg.neg w).IsOption y
      ⊢ And (z.IsOption x₁ → Surreal.Multiplication.P2 z x₂ w) (z.IsOption x₂ → Surr …
    -/
                             /-
                               🎉 no goals
                             -/
  · convert h h' using 2 <;> rw [P2_neg_right]
                             /-
                               🎉 no goals
                             -/


lemma mulOption_lt_mul_of_equiv (hn : x₁.Numeric) (h : IH24 x₁ x₂ y) (he : x₁ ≈ x₂) (i j) :
    ⟦mulOption x₁ y i j⟧ < (⟦x₂ * y⟧ : Game) := by
  /-
    x₁ x₂ y : SetTheory.PGame
    hn : x₁.Numeric
    h : Surreal.Multiplication.IH24 x₁ x₂ y
    he : HasEquiv.Equiv x₁ x₂
    i : x₁.LeftMoves
    j : y.LeftMoves
    ⊢ LT.lt (Quotient.mk SetTheory.PGame.setoid (x₁.mulOption y i j)) (Quotient.mk …
  -/
  convert sub_lt_iff_lt_add'.2 ((((@h _).1 <| IsOption.moveLeft i).2 _).1 j) using 1
    /-
      case h.e'_3
      x₁ x₂ y : SetTheory.PGame
      hn : x₁.Numeric
      h : Surreal.Multiplication.IH24 x₁ x₂ y
      he : HasEquiv.Equiv x₁ x₂
      i : x₁.LeftMoves
      j : y.LeftMoves
      ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (x₁.mulOption y i j)) (HSub.hSub (HAd …
    -/
  · rw [← ((@h _).2.2 <| IsOption.moveLeft j).1 he]
    /-
      case h.e'_3
      x₁ x₂ y : SetTheory.PGame
      hn : x₁.Numeric
      h : Surreal.Multiplication.IH24 x₁ x₂ y
      he : HasEquiv.Equiv x₁ x₂
      i : x₁.LeftMoves
      j : y.LeftMoves
      ⊢ Eq (Quotient.mk SetTheory.PGame.setoid (x₁.mulOption y i j)) (HSub.hSub (HAd …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      x₁ x₂ y : SetTheory.PGame
      hn : x₁.Numeric
      h : Surreal.Multiplication.IH24 x₁ x₂ y
      he : HasEquiv.Equiv x₁ x₂
      i : x₁.LeftMoves
      j : y.LeftMoves
      ⊢ LT.lt (x₁.moveLeft i) x₂
    -/
  · rw [← lt_congr_right he]
    /-
      x₁ x₂ y : SetTheory.PGame
      hn : x₁.Numeric
      h : Surreal.Multiplication.IH24 x₁ x₂ y
      he : HasEquiv.Equiv x₁ x₂
      i : x₁.LeftMoves
      j : y.LeftMoves
      ⊢ LT.lt (x₁.moveLeft i) x₁
    -/
    apply hn.moveLeft_lt
    /-
      🎉 no goals
    -/


/-- P2 follows from specialized induction hypotheses (one half of the equality). -/
theorem mul_right_le_of_equiv (h₁ : x₁.Numeric) (h₂ : x₂.Numeric)
    (h₁₂ : IH24 x₁ x₂ y) (h₂₁ : IH24 x₂ x₁ y) (he : x₁ ≈ x₂) : x₁ * y ≤ x₂ * y := by
  /-
    x₁ x₂ y : SetTheory.PGame
    h₁ : x₁.Numeric
    h₂ : x₂.Numeric
    h₁₂ : Surreal.Multiplication.IH24 x₁ x₂ y
    h₂₁ : Surreal.Multiplication.IH24 x₂ x₁ y
    he : HasEquiv.Equiv x₁ x₂
    ⊢ LE.le (HMul.hMul x₁ y) (HMul.hMul x₂ y)
  -/
  have he' := neg_equiv_neg_iff.2 he
  /-
    x₁ x₂ y : SetTheory.PGame
    h₁ : x₁.Numeric
    h₂ : x₂.Numeric
    h₁₂ : Surreal.Multiplication.IH24 x₁ x₂ y
    h₂₁ : Surreal.Multiplication.IH24 x₂ x₁ y
    he : HasEquiv.Equiv x₁ x₂
    he' : HasEquiv.Equiv (Neg.neg x₁) (Neg.neg x₂)
    ⊢ LE.le (HMul.hMul x₁ y) (HMul.hMul x₂ y)
  -/
  apply PGame.le_of_forall_lt <;> simp_rw [lt_iff_game_lt]
    /-
      case h₁
      x₁ x₂ y : SetTheory.PGame
      h₁ : x₁.Numeric
      h₂ : x₂.Numeric
      h₁₂ : Surreal.Multiplication.IH24 x₁ x₂ y
      h₂₁ : Surreal.Multiplication.IH24 x₂ x₁ y
      he : HasEquiv.Equiv x₁ x₂
      he' : HasEquiv.Equiv (Neg.neg x₁) (Neg.neg x₂)
      ⊢ ∀ (i : (HMul.hMul x₁ y).LeftMoves), LT.lt (Quotient.mk SetTheory.PGame.setoi …
    -/
  · rw [leftMoves_mul_iff (_ > ·)]
    /-
      case h₁
      x₁ x₂ y : SetTheory.PGame
      h₁ : x₁.Numeric
      h₂ : x₂.Numeric
      h₁₂ : Surreal.Multiplication.IH24 x₁ x₂ y
      h₂₁ : Surreal.Multiplication.IH24 x₂ x₁ y
      he : HasEquiv.Equiv x₁ x₂
      he' : HasEquiv.Equiv (Neg.neg x₁) (Neg.neg x₂)
      ⊢ And (∀ (i : x₁.LeftMoves) (j : y.LeftMoves), GT.gt (Quotient.mk SetTheory.PG …
    -/
    refine ⟨mulOption_lt_mul_of_equiv h₁ h₁₂ he, ?_⟩
    /-
      case h₁
      x₁ x₂ y : SetTheory.PGame
      h₁ : x₁.Numeric
      h₂ : x₂.Numeric
      h₁₂ : Surreal.Multiplication.IH24 x₁ x₂ y
      h₂₁ : Surreal.Multiplication.IH24 x₂ x₁ y
      he : HasEquiv.Equiv x₁ x₂
      he' : HasEquiv.Equiv (Neg.neg x₁) (Neg.neg x₂)
      ⊢ ∀ (i : (Neg.neg x₁).LeftMoves) (j : (Neg.neg y).LeftMoves), GT.gt (Quotient. …
    -/
    rw [← quot_neg_mul_neg]
    /-
      case h₁
      x₁ x₂ y : SetTheory.PGame
      h₁ : x₁.Numeric
      h₂ : x₂.Numeric
      h₁₂ : Surreal.Multiplication.IH24 x₁ x₂ y
      h₂₁ : Surreal.Multiplication.IH24 x₂ x₁ y
      he : HasEquiv.Equiv x₁ x₂
      he' : HasEquiv.Equiv (Neg.neg x₁) (Neg.neg x₂)
      ⊢ ∀ (i : (Neg.neg x₁).LeftMoves) (j : (Neg.neg y).LeftMoves), GT.gt (Quotient. …
    -/
    exact mulOption_lt_mul_of_equiv h₁.neg (ih24_neg <| (ih24_neg h₂₁).1).2 he'
    /-
      🎉 no goals
    -/
    /-
      case h₂
      x₁ x₂ y : SetTheory.PGame
      h₁ : x₁.Numeric
      h₂ : x₂.Numeric
      h₁₂ : Surreal.Multiplication.IH24 x₁ x₂ y
      h₂₁ : Surreal.Multiplication.IH24 x₂ x₁ y
      he : HasEquiv.Equiv x₁ x₂
      he' : HasEquiv.Equiv (Neg.neg x₁) (Neg.neg x₂)
      ⊢ ∀ (j : (HMul.hMul x₂ y).RightMoves), LT.lt (Quotient.mk SetTheory.PGame.seto …
    -/
  · rw [rightMoves_mul_iff]
    /-
      case h₂
      x₁ x₂ y : SetTheory.PGame
      h₁ : x₁.Numeric
      h₂ : x₂.Numeric
      h₁₂ : Surreal.Multiplication.IH24 x₁ x₂ y
      h₂₁ : Surreal.Multiplication.IH24 x₂ x₁ y
      he : HasEquiv.Equiv x₁ x₂
      he' : HasEquiv.Equiv (Neg.neg x₁) (Neg.neg x₂)
      ⊢ And (∀ (i : x₂.LeftMoves) (j : (Neg.neg y).LeftMoves), LT.lt (Quotient.mk Se …
    -/
    constructor <;> intros <;> rw [lt_neg]
      /-
        case h₂.left
        x₁ x₂ y : SetTheory.PGame
        h₁ : x₁.Numeric
        h₂ : x₂.Numeric
        h₁₂ : Surreal.Multiplication.IH24 x₁ x₂ y
        h₂₁ : Surreal.Multiplication.IH24 x₂ x₁ y
        he : HasEquiv.Equiv x₁ x₂
        he' : HasEquiv.Equiv (Neg.neg x₁) (Neg.neg x₂)
        i✝ : x₂.LeftMoves
        j✝ : (Neg.neg y).LeftMoves
        ⊢ LT.lt (Quotient.mk SetTheory.PGame.setoid (x₂.mulOption (Neg.neg y) i✝ j✝))  …
      -/
    · rw [← quot_mul_neg]
      /-
        case h₂.left
        x₁ x₂ y : SetTheory.PGame
        h₁ : x₁.Numeric
        h₂ : x₂.Numeric
        h₁₂ : Surreal.Multiplication.IH24 x₁ x₂ y
        h₂₁ : Surreal.Multiplication.IH24 x₂ x₁ y
        he : HasEquiv.Equiv x₁ x₂
        he' : HasEquiv.Equiv (Neg.neg x₁) (Neg.neg x₂)
        i✝ : x₂.LeftMoves
        j✝ : (Neg.neg y).LeftMoves
        ⊢ LT.lt (Quotient.mk SetTheory.PGame.setoid (x₂.mulOption (Neg.neg y) i✝ j✝))  …
      -/
      apply mulOption_lt_mul_of_equiv h₂ (ih24_neg h₂₁).2 (symm he)
      /-
        🎉 no goals
      -/
      /-
        case h₂.right
        x₁ x₂ y : SetTheory.PGame
        h₁ : x₁.Numeric
        h₂ : x₂.Numeric
        h₁₂ : Surreal.Multiplication.IH24 x₁ x₂ y
        h₂₁ : Surreal.Multiplication.IH24 x₂ x₁ y
        he : HasEquiv.Equiv x₁ x₂
        he' : HasEquiv.Equiv (Neg.neg x₁) (Neg.neg x₂)
        i✝ : (Neg.neg x₂).LeftMoves
        j✝ : y.LeftMoves
        ⊢ LT.lt (Quotient.mk SetTheory.PGame.setoid ((Neg.neg x₂).mulOption y i✝ j✝))  …
      -/
    · rw [← quot_neg_mul]
      /-
        case h₂.right
        x₁ x₂ y : SetTheory.PGame
        h₁ : x₁.Numeric
        h₂ : x₂.Numeric
        h₁₂ : Surreal.Multiplication.IH24 x₁ x₂ y
        h₂₁ : Surreal.Multiplication.IH24 x₂ x₁ y
        he : HasEquiv.Equiv x₁ x₂
        he' : HasEquiv.Equiv (Neg.neg x₁) (Neg.neg x₂)
        i✝ : (Neg.neg x₂).LeftMoves
        j✝ : y.LeftMoves
        ⊢ LT.lt (Quotient.mk SetTheory.PGame.setoid ((Neg.neg x₂).mulOption y i✝ j✝))  …
      -/
      apply mulOption_lt_mul_of_equiv h₂.neg (ih24_neg h₁₂).1 (symm he')
      /-
        🎉 no goals
      -/


/-- The statement that all left options of `x * y` of the first kind are less than itself. -/
def MulOptionsLTMul (x y : PGame) : Prop := ∀ ⦃i j⦄, ⟦mulOption x y i j⟧ < (⟦x * y⟧ : Game)


/-- That the left options of `x * y` are less than itself and the right options are greater, which
  is part of the condition that `x * y` is numeric, is equivalent to the conjunction of various
  `MulOptionsLTMul` statements for `x`, `y` and their negations. We only show the forward
  direction. -/
lemma mulOptionsLTMul_of_numeric (hn : (x * y).Numeric) :
    (MulOptionsLTMul x y ∧ MulOptionsLTMul (-x) (-y)) ∧
    (MulOptionsLTMul x (-y) ∧ MulOptionsLTMul (-x) y) := by
  /-
    x y : SetTheory.PGame
    hn : (HMul.hMul x y).Numeric
    ⊢ And (And (Surreal.Multiplication.MulOptionsLTMul x y) (Surreal.Multiplicatio …
  -/
  constructor
    /-
      case left
      x y : SetTheory.PGame
      hn : (HMul.hMul x y).Numeric
      ⊢ And (Surreal.Multiplication.MulOptionsLTMul x y) (Surreal.Multiplication.Mul …
    -/
  · have h := hn.moveLeft_lt
    /-
      case left
      x y : SetTheory.PGame
      hn : (HMul.hMul x y).Numeric
      h : ∀ (i : (HMul.hMul x y).LeftMoves), LT.lt ((HMul.hMul x y).moveLeft i) (HMu …
      ⊢ And (Surreal.Multiplication.MulOptionsLTMul x y) (Surreal.Multiplication.Mul …
    -/
    simp_rw [lt_iff_game_lt] at h
    /-
      case left
      x y : SetTheory.PGame
      hn : (HMul.hMul x y).Numeric
      h : ∀ (i : (HMul.hMul x y).LeftMoves), LT.lt (Quotient.mk SetTheory.PGame.seto …
      ⊢ And (Surreal.Multiplication.MulOptionsLTMul x y) (Surreal.Multiplication.Mul …
    -/
    convert (leftMoves_mul_iff <| GT.gt _).1 h
    /-
      case h.e'_2.a
      x y : SetTheory.PGame
      hn : (HMul.hMul x y).Numeric
      h : ∀ (i : (HMul.hMul x y).LeftMoves), LT.lt (Quotient.mk SetTheory.PGame.seto …
      ⊢ Iff (Surreal.Multiplication.MulOptionsLTMul (Neg.neg x) (Neg.neg y)) (∀ (i : …
    -/
    rw [← quot_neg_mul_neg]
    /-
      case h.e'_2.a
      x y : SetTheory.PGame
      hn : (HMul.hMul x y).Numeric
      h : ∀ (i : (HMul.hMul x y).LeftMoves), LT.lt (Quotient.mk SetTheory.PGame.seto …
      ⊢ Iff (Surreal.Multiplication.MulOptionsLTMul (Neg.neg x) (Neg.neg y)) (∀ (i : …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case right
      x y : SetTheory.PGame
      hn : (HMul.hMul x y).Numeric
      ⊢ And (Surreal.Multiplication.MulOptionsLTMul x (Neg.neg y)) (Surreal.Multipli …
    -/
  · have h := hn.lt_moveRight
    /-
      case right
      x y : SetTheory.PGame
      hn : (HMul.hMul x y).Numeric
      h : ∀ (j : (HMul.hMul x y).RightMoves), LT.lt (HMul.hMul x y) ((HMul.hMul x y) …
      ⊢ And (Surreal.Multiplication.MulOptionsLTMul x (Neg.neg y)) (Surreal.Multipli …
    -/
    simp_rw [lt_iff_game_lt, rightMoves_mul_iff] at h
    /-
      case right
      x y : SetTheory.PGame
      hn : (HMul.hMul x y).Numeric
      h : And (∀ (i : x.LeftMoves) (j : (Neg.neg y).LeftMoves), LT.lt (Quotient.mk S …
      ⊢ And (Surreal.Multiplication.MulOptionsLTMul x (Neg.neg y)) (Surreal.Multipli …
    -/
    refine h.imp ?_ ?_ <;> refine forall₂_imp fun a b ↦ ?_
    all_goals
      rw [lt_neg]
      first | rw [quot_mul_neg] | rw [quot_neg_mul]
      exact id


/-- A condition just enough to deduce P3, which will always be used with `x'` being a left
  option of `x₂`. When `y₁` is a left option of `y₂`, it can be deduced from induction hypotheses
  `IH24 x₁ x₂ y₂`, `IH4 x₁ x₂ y₂`, and `(x₂ * y₂).Numeric` (`ih3_of_ih`); when `y₁` is
  not necessarily an option of `y₂`, it follows from the induction hypothesis for P3 (with `x₂`
  replaced by a left option `x'`) after the `main` theorem (P124) is established, and is used to
  prove P3 in full (`P3_of_lt_of_lt`). -/
def IH3 (x₁ x' x₂ y₁ y₂ : PGame) : Prop :=
    P2 x₁ x' y₁ ∧ P2 x₁ x' y₂ ∧ P3 x' x₂ y₁ y₂ ∧ (x₁ < x' → P3 x₁ x' y₁ y₂)


lemma ih3_of_ih (h24 : IH24 x₁ x₂ y) (h4 : IH4 x₁ x₂ y) (hl : MulOptionsLTMul x₂ y) (i j) :
    IH3 x₁ (x₂.moveLeft i) x₂ (y.moveLeft j) y :=
  have ml := @IsOption.moveLeft
  have h24 := (@h24 _).2.1 (ml i)
  ⟨(h4 <| ml j).2 (ml i), h24.1, mulOption_lt_mul_iff_P3.1 (@hl i j), fun l ↦ (h24.2 l).1 _⟩


lemma P3_of_le_left {y₁ y₂} (i) (h : IH3 x₁ (x₂.moveLeft i) x₂ y₁ y₂) (hl : x₁ ≤ x₂.moveLeft i) :
    P3 x₁ x₂ y₁ y₂ := by
  /-
    x₁ x₂ y₁ y₂ : SetTheory.PGame
    i : x₂.LeftMoves
    h : Surreal.Multiplication.IH3 x₁ (x₂.moveLeft i) x₂ y₁ y₂
    hl : LE.le x₁ (x₂.moveLeft i)
    ⊢ Surreal.Multiplication.P3 x₁ x₂ y₁ y₂
  -/
  obtain (hl|he) := lt_or_equiv_of_le hl
    /-
      case inl
      x₁ x₂ y₁ y₂ : SetTheory.PGame
      i : x₂.LeftMoves
      h : Surreal.Multiplication.IH3 x₁ (x₂.moveLeft i) x₂ y₁ y₂
      hl✝ : LE.le x₁ (x₂.moveLeft i)
      hl : LT.lt x₁ (x₂.moveLeft i)
      ⊢ Surreal.Multiplication.P3 x₁ x₂ y₁ y₂
    -/
  · exact (h.2.2.2 hl).trans h.2.2.1
    /-
      🎉 no goals
    -/
    /-
      case inr
      x₁ x₂ y₁ y₂ : SetTheory.PGame
      i : x₂.LeftMoves
      h : Surreal.Multiplication.IH3 x₁ (x₂.moveLeft i) x₂ y₁ y₂
      hl : LE.le x₁ (x₂.moveLeft i)
      he : HasEquiv.Equiv x₁ (x₂.moveLeft i)
      ⊢ Surreal.Multiplication.P3 x₁ x₂ y₁ y₂
    -/
  · rw [P3, h.1 he, h.2.1 he]
    /-
      case inr
      x₁ x₂ y₁ y₂ : SetTheory.PGame
      i : x₂.LeftMoves
      h : Surreal.Multiplication.IH3 x₁ (x₂.moveLeft i) x₂ y₁ y₂
      hl : LE.le x₁ (x₂.moveLeft i)
      he : HasEquiv.Equiv x₁ (x₂.moveLeft i)
      ⊢ LT.lt (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul (x₂.moveLeft …
    -/
    exact h.2.2.1
    /-
      🎉 no goals
    -/


/-- P3 follows from `IH3` (so P4 (with `y₁` a left option of `y₂`) follows from the induction
  hypothesis). -/
theorem P3_of_lt {y₁ y₂} (h : ∀ i, IH3 x₁ (x₂.moveLeft i) x₂ y₁ y₂)
    (hs : ∀ i, IH3 (-x₂) ((-x₁).moveLeft i) (-x₁) y₁ y₂) (hl : x₁ < x₂) :
    P3 x₁ x₂ y₁ y₂ := by
  /-
    x₁ x₂ y₁ y₂ : SetTheory.PGame
    h : ∀ (i : x₂.LeftMoves), Surreal.Multiplication.IH3 x₁ (x₂.moveLeft i) x₂ y₁ y₂
    hs : ∀ (i : (Neg.neg x₁).LeftMoves), Surreal.Multiplication.IH3 (Neg.neg x₂) ( …
    hl : LT.lt x₁ x₂
    ⊢ Surreal.Multiplication.P3 x₁ x₂ y₁ y₂
  -/
  obtain (⟨i,hi⟩|⟨i,hi⟩) := lf_iff_exists_le.1 (lf_of_lt hl)
    /-
      case inl.intro
      x₁ x₂ y₁ y₂ : SetTheory.PGame
      h : ∀ (i : x₂.LeftMoves), Surreal.Multiplication.IH3 x₁ (x₂.moveLeft i) x₂ y₁ y₂
      hs : ∀ (i : (Neg.neg x₁).LeftMoves), Surreal.Multiplication.IH3 (Neg.neg x₂) ( …
      hl : LT.lt x₁ x₂
      i : x₂.LeftMoves
      hi : LE.le x₁ (x₂.moveLeft i)
      ⊢ Surreal.Multiplication.P3 x₁ x₂ y₁ y₂
    -/
  · exact P3_of_le_left i (h i) hi
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      x₁ x₂ y₁ y₂ : SetTheory.PGame
      h : ∀ (i : x₂.LeftMoves), Surreal.Multiplication.IH3 x₁ (x₂.moveLeft i) x₂ y₁ y₂
      hs : ∀ (i : (Neg.neg x₁).LeftMoves), Surreal.Multiplication.IH3 (Neg.neg x₂) ( …
      hl : LT.lt x₁ x₂
      i : x₁.RightMoves
      hi : LE.le (x₁.moveRight i) x₂
      ⊢ Surreal.Multiplication.P3 x₁ x₂ y₁ y₂
    -/
  · apply P3_neg.2 <| P3_of_le_left _ (hs (toLeftMovesNeg i)) _
    /-
      x₁ x₂ y₁ y₂ : SetTheory.PGame
      h : ∀ (i : x₂.LeftMoves), Surreal.Multiplication.IH3 x₁ (x₂.moveLeft i) x₂ y₁ y₂
      hs : ∀ (i : (Neg.neg x₁).LeftMoves), Surreal.Multiplication.IH3 (Neg.neg x₂) ( …
      hl : LT.lt x₁ x₂
      i : x₁.RightMoves
      hi : LE.le (x₁.moveRight i) x₂
      ⊢ LE.le (Neg.neg x₂) ((Neg.neg x₁).moveLeft (SetTheory.PGame.toLeftMovesNeg i))
    -/
    simpa
    /-
      🎉 no goals
    -/


/-- The main chunk of Theorem 8 in [Conway2001] / Theorem 3.8 in [SchleicherStoll]. -/
theorem main (a : Args) : a.Numeric → P124 a := by
  /-
    a : Surreal.Multiplication.Args
    ⊢ a.Numeric → Surreal.Multiplication.P124 a
  -/
  apply argsRel_wf.induction a
  /-
    a : Surreal.Multiplication.Args
    ⊢ ∀ (x : Surreal.Multiplication.Args), (∀ (y : Surreal.Multiplication.Args), S …
  -/
  intros a ih ha
  /-
    a✝ a : Surreal.Multiplication.Args
    ih : ∀ (y : Surreal.Multiplication.Args), Surreal.Multiplication.ArgsRel y a → …
    ha : a.Numeric
    ⊢ Surreal.Multiplication.P124 a
  -/
  replace ih : ∀ a', ArgsRel a' a → P124 a' := fun a' hr ↦ ih a' hr (hr.numeric_closed ha)
  cases a with
  /- P1 -/
  | P1 x y =>
    rw [Args.numeric_P1] at ha
    exact P1_of_ih ih ha.1 ha.2
  | P24 x₁ x₂ y =>
    have h₁₂ := ih₁₂ ih
    have h₂₁ := ih₂₁ ih
    have h4 := ih4 ih
    obtain ⟨h₁₂x, h₁₂y⟩ := ih24_neg h₁₂
    obtain ⟨h4x, h4y⟩ := ih4_neg h4
    refine ⟨fun he ↦ Quotient.sound ?_, fun hl ↦ ?_⟩
    · /- P2 -/
      rw [Args.numeric_P24] at ha
      exact ⟨mul_right_le_of_equiv ha.1 ha.2.1 h₁₂ h₂₁ he,
        mul_right_le_of_equiv ha.2.1 ha.1 h₂₁ h₁₂ (symm he)⟩
    · /- P4 -/
      obtain ⟨hn₁, hn₂⟩ := numeric_of_ih ih
      obtain ⟨⟨h₁, -⟩, h₂, -⟩ := mulOptionsLTMul_of_numeric hn₂
      obtain ⟨⟨-, h₃⟩, -, h₄⟩ := mulOptionsLTMul_of_numeric hn₁
      constructor <;> intro <;> refine P3_of_lt ?_ ?_ hl <;> intro <;> apply ih3_of_ih
      any_goals assumption
      exacts [(ih24_neg h₁₂y).1, (ih4_neg h4y).1]


theorem Numeric.mul (hx : x.Numeric) (hy : y.Numeric) : Numeric (x * y) :=
  main _ <| Args.numeric_P1.mpr ⟨hx, hy⟩


theorem P24 (hx₁ : x₁.Numeric) (hx₂ : x₂.Numeric) (hy : y.Numeric) : P24 x₁ x₂ y :=
  main _ <| Args.numeric_P24.mpr ⟨hx₁, hx₂, hy⟩


theorem Equiv.mul_congr_left (hx₁ : x₁.Numeric) (hx₂ : x₂.Numeric) (hy : y.Numeric)
    (he : x₁ ≈ x₂) : x₁ * y ≈ x₂ * y :=
  equiv_iff_game_eq.2 <| (P24 hx₁ hx₂ hy).1 he


theorem Equiv.mul_congr_right (hx : x.Numeric) (hy₁ : y₁.Numeric) (hy₂ : y₂.Numeric)
    (he : y₁ ≈ y₂) : x * y₁ ≈ x * y₂ :=
  .trans (mul_comm_equiv _ _) <| .trans (mul_congr_left hy₁ hy₂ hx he) (mul_comm_equiv _ _)


theorem Equiv.mul_congr (hx₁ : x₁.Numeric) (hx₂ : x₂.Numeric)
    (hy₁ : y₁.Numeric) (hy₂ : y₂.Numeric) (hx : x₁ ≈ x₂) (hy : y₁ ≈ y₂) : x₁ * y₁ ≈ x₂ * y₂ :=
  .trans (mul_congr_left hx₁ hx₂ hy₁ hx) (mul_congr_right hx₂ hy₁ hy₂ hy)


/-- One additional inductive argument that supplies the last missing part of Theorem 8. -/
theorem P3_of_lt_of_lt (hx₁ : x₁.Numeric) (hx₂ : x₂.Numeric) (hy₁ : y₁.Numeric) (hy₂ : y₂.Numeric)
    (hx : x₁ < x₂) (hy : y₁ < y₂) : P3 x₁ x₂ y₁ y₂ := by
  /-
    x₁ x₂ y₁ y₂ : SetTheory.PGame
    hx₁ : x₁.Numeric
    hx₂ : x₂.Numeric
    hy₁ : y₁.Numeric
    hy₂ : y₂.Numeric
    hx : LT.lt x₁ x₂
    hy : LT.lt y₁ y₂
    ⊢ Surreal.Multiplication.P3 x₁ x₂ y₁ y₂
  -/
  revert x₁ x₂
  /-
    y₁ y₂ : SetTheory.PGame
    hy₁ : y₁.Numeric
    hy₂ : y₂.Numeric
    hy : LT.lt y₁ y₂
    ⊢ ∀ {x₁ x₂ : SetTheory.PGame}, x₁.Numeric → x₂.Numeric → LT.lt x₁ x₂ → Surreal …
  -/
  rw [← Prod.forall']
  /-
    y₁ y₂ : SetTheory.PGame
    hy₁ : y₁.Numeric
    hy₂ : y₂.Numeric
    hy : LT.lt y₁ y₂
    ⊢ ∀ (x : Prod SetTheory.PGame SetTheory.PGame), x.1.Numeric → x.2.Numeric → LT …
  -/
  refine (wf_isOption.prod_gameAdd wf_isOption).fix ?_
  /-
    y₁ y₂ : SetTheory.PGame
    hy₁ : y₁.Numeric
    hy₂ : y₂.Numeric
    hy : LT.lt y₁ y₂
    ⊢ ∀ (x : Prod SetTheory.PGame SetTheory.PGame), (∀ (y : Prod SetTheory.PGame S …
  -/
  rintro ⟨x₁, x₂⟩ ih hx₁ hx₂ hx
  /-
    case mk
    y₁ y₂ : SetTheory.PGame
    hy₁ : y₁.Numeric
    hy₂ : y₂.Numeric
    hy : LT.lt y₁ y₂
    x₁ x₂ : SetTheory.PGame
    ih : ∀ (y : Prod SetTheory.PGame SetTheory.PGame), Prod.GameAdd SetTheory.PGam …
    hx₁ : { fst := x₁, snd := x₂ }.1.Numeric
    hx₂ : { fst := x₁, snd := x₂ }.2.Numeric
    hx : LT.lt { fst := x₁, snd := x₂ }.1 { fst := x₁, snd := x₂ }.2
    ⊢ Surreal.Multiplication.P3 { fst := x₁, snd := x₂ }.1 { fst := x₁, snd := x₂  …
  -/
  refine P3_of_lt ?_ ?_ hx <;> intro i
    /-
      case mk.refine_1
      y₁ y₂ : SetTheory.PGame
      hy₁ : y₁.Numeric
      hy₂ : y₂.Numeric
      hy : LT.lt y₁ y₂
      x₁ x₂ : SetTheory.PGame
      ih : ∀ (y : Prod SetTheory.PGame SetTheory.PGame), Prod.GameAdd SetTheory.PGam …
      hx₁ : { fst := x₁, snd := x₂ }.1.Numeric
      hx₂ : { fst := x₁, snd := x₂ }.2.Numeric
      hx : LT.lt { fst := x₁, snd := x₂ }.1 { fst := x₁, snd := x₂ }.2
      i : { fst := x₁, snd := x₂ }.2.LeftMoves
      ⊢ Surreal.Multiplication.IH3 { fst := x₁, snd := x₂ }.1 ({ fst := x₁, snd := x …
    -/
  · have hi := hx₂.moveLeft i
    exact ⟨(P24 hx₁ hi hy₁).1, (P24 hx₁ hi hy₂).1,
      P3_comm.2 <| ((P24 hy₁ hy₂ hx₂).2 hy).1 _,
      ih _ (snd <| IsOption.moveLeft i) hx₁ hi⟩
    /-
      case mk.refine_2
      y₁ y₂ : SetTheory.PGame
      hy₁ : y₁.Numeric
      hy₂ : y₂.Numeric
      hy : LT.lt y₁ y₂
      x₁ x₂ : SetTheory.PGame
      ih : ∀ (y : Prod SetTheory.PGame SetTheory.PGame), Prod.GameAdd SetTheory.PGam …
      hx₁ : { fst := x₁, snd := x₂ }.1.Numeric
      hx₂ : { fst := x₁, snd := x₂ }.2.Numeric
      hx : LT.lt { fst := x₁, snd := x₂ }.1 { fst := x₁, snd := x₂ }.2
      i : (Neg.neg { fst := x₁, snd := x₂ }.1).LeftMoves
      ⊢ Surreal.Multiplication.IH3 (Neg.neg { fst := x₁, snd := x₂ }.2) ((Neg.neg {  …
    -/
  · have hi := hx₁.neg.moveLeft i
    exact ⟨(P24 hx₂.neg hi hy₁).1, (P24 hx₂.neg hi hy₂).1,
      P3_comm.2 <| ((P24 hy₁ hy₂ hx₁).2 hy).2 _, by
        rw [moveLeft_neg, ← P3_neg, neg_lt_neg_iff]
        exact ih _ (fst <| IsOption.moveRight _) (hx₁.moveRight _) hx₂⟩


theorem Numeric.mul_pos (hx₁ : x₁.Numeric) (hx₂ : x₂.Numeric) (hp₁ : 0 < x₁) (hp₂ : 0 < x₂) :
    0 < x₁ * x₂ := by
  /-
    x₁ x₂ : SetTheory.PGame
    hx₁ : x₁.Numeric
    hx₂ : x₂.Numeric
    hp₁ : LT.lt 0 x₁
    hp₂ : LT.lt 0 x₂
    ⊢ LT.lt 0 (HMul.hMul x₁ x₂)
  -/
  rw [lt_iff_game_lt]
  /-
    x₁ x₂ : SetTheory.PGame
    hx₁ : x₁.Numeric
    hx₂ : x₂.Numeric
    hp₁ : LT.lt 0 x₁
    hp₂ : LT.lt 0 x₂
    ⊢ LT.lt (Quotient.mk SetTheory.PGame.setoid 0) (Quotient.mk SetTheory.PGame.se …
  -/
  have := P3_of_lt_of_lt numeric_zero hx₁ numeric_zero hx₂ hp₁ hp₂
  /-
    x₁ x₂ : SetTheory.PGame
    hx₁ : x₁.Numeric
    hx₂ : x₂.Numeric
    hp₁ : LT.lt 0 x₁
    hp₂ : LT.lt 0 x₂
    this : Surreal.Multiplication.P3 0 x₁ 0 x₂
    ⊢ LT.lt (Quotient.mk SetTheory.PGame.setoid 0) (Quotient.mk SetTheory.PGame.se …
  -/
  simp_rw [P3, quot_zero_mul, quot_mul_zero, add_lt_add_iff_left] at this
  /-
    x₁ x₂ : SetTheory.PGame
    hx₁ : x₁.Numeric
    hx₂ : x₂.Numeric
    hp₁ : LT.lt 0 x₁
    hp₂ : LT.lt 0 x₂
    this : LT.lt 0 (Quotient.mk SetTheory.PGame.setoid (HMul.hMul x₁ x₂))
    ⊢ LT.lt (Quotient.mk SetTheory.PGame.setoid 0) (Quotient.mk SetTheory.PGame.se …
  -/
  exact this
  /-
    🎉 no goals
  -/


noncomputable instance : LinearOrderedCommRing Surreal where
  __ := Surreal.orderedAddCommGroup
  mul := Surreal.lift₂ (fun x y ox oy ↦ ⟦⟨x * y, ox.mul oy⟩⟧)
    (fun ox₁ oy₁ ox₂ oy₂ hx hy ↦ Quotient.sound <| mul_congr ox₁ ox₂ oy₁ oy₂ hx hy)
                  /-
                    ⊢ ∀ (a b c : Surreal), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul.hMu …
                  -/
  mul_assoc := by rintro ⟨_⟩ ⟨_⟩ ⟨_⟩; exact Quotient.sound (mul_assoc_equiv _ _ _)
                                      /-
                                        🎉 no goals
                                      -/
  one := mk 1 numeric_one
                     /-
                       ⊢ ∀ (a b c : Surreal), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMul.hMul  …
                     -/
                /-
                  ⊢ ∀ (a : Surreal), Eq (HMul.hMul 1 a) a
                -/
                                         /-
                                           🎉 no goals
                                         -/
                      /-
                        ⊢ ∀ (a b c : Surreal), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul  …
                      -/
  one_mul := by rintro ⟨_⟩; exact Quotient.sound (one_mul_equiv _)
                                          /-
                                            🎉 no goals
                                          -/
                            /-
                              🎉 no goals
                            -/
                /-
                  ⊢ ∀ (a : Surreal), Eq (HMul.hMul a 1) a
                -/
  mul_one := by rintro ⟨_⟩; exact Quotient.sound (mul_one_equiv _)
                            /-
                              🎉 no goals
                            -/
  left_distrib := by rintro ⟨_⟩ ⟨_⟩ ⟨_⟩; exact Quotient.sound (left_distrib_equiv _ _ _)
  right_distrib := by rintro ⟨_⟩ ⟨_⟩ ⟨_⟩; exact Quotient.sound (right_distrib_equiv _ _ _)
                 /-
                   ⊢ ∀ (a b : Surreal), Eq (HMul.hMul a b) (HMul.hMul b a)
                 -/
  mul_comm := by rintro ⟨_⟩ ⟨_⟩; exact Quotient.sound (mul_comm_equiv _ _)
                                 /-
                                   🎉 no goals
                                 -/
                 /-
                   ⊢ ∀ (a : Surreal), Eq (HMul.hMul 0 a) 0
                 -/
                /-
                  ⊢ ∀ (a : Surreal), LE.le a a
                -/
                             /-
                               🎉 no goals
                             -/
                 /-
                   ⊢ ∀ (a : Surreal), Eq (HMul.hMul a 0) 0
                 -/
  le := lift₂ (fun x y _ _ ↦ x ≤ y) (fun _ _ _ _ hx hy ↦ propext <| le_congr hx hy)
                             /-
                               🎉 no goals
                             -/
                            /-
                              🎉 no goals
                            -/
                 /-
                   ⊢ ∀ (a b c : Surreal), LE.le a b → LE.le b c → LE.le a c
                 -/
  lt := lift₂ (fun x y _ _ ↦ x < y) (fun _ _ _ _ hx hy ↦ propext <| lt_congr hx hy)
                                     /-
                                       🎉 no goals
                                     -/
                         /-
                           ⊢ ∀ (a b : Surreal), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
                         -/
  le_refl := by rintro ⟨_⟩; apply @le_rfl PGame
                                         /-
                                           🎉 no goals
                                         -/
                    /-
                      ⊢ ∀ (a b : Surreal), LE.le a b → LE.le b a → Eq a b
                    -/
  le_trans := by rintro ⟨_⟩ ⟨_⟩ ⟨_⟩; apply @le_trans PGame
                                          /-
                                            🎉 no goals
                                          -/
                        /-
                          ⊢ ∀ (a b : Surreal), LE.le a b → ∀ (c : Surreal), LE.le (HAdd.hAdd c a) (HAdd. …
                        -/
  lt_iff_le_not_le := by rintro ⟨_⟩ ⟨_⟩; exact lt_iff_le_not_le
                                               /-
                                                 🎉 no goals
                                               -/
  le_antisymm := by rintro ⟨_⟩ ⟨_⟩ h₁ h₂; exact Quotient.sound ⟨h₁, h₂⟩
  add_le_add_left := by rintro ⟨_⟩ ⟨_⟩ hx ⟨_⟩; exact add_le_add_left hx _
  zero_le_one := PGame.zero_lt_one.le
  zero_mul := by rintro ⟨_⟩; exact Quotient.sound (zero_mul_equiv _)
                 /-
                   ⊢ ∀ (a b : Surreal), Or (LE.le a b) (LE.le b a)
                 -/
                /-
                  ⊢ ∀ (a b : Surreal), LT.lt 0 a → LT.lt 0 b → LT.lt 0 (HMul.hMul a b)
                -/
  mul_zero := by rintro ⟨_⟩; exact Quotient.sound (mul_zero_equiv _)
                                /-
                                  🎉 no goals
                                -/
                                 /-
                                   🎉 no goals
                                 -/
  exists_pair_ne := ⟨0, 1, ne_of_lt PGame.zero_lt_one⟩
  le_total := by rintro ⟨x⟩ ⟨y⟩; exact (le_or_gf x.1 y.1).imp id (fun h ↦ h.le y.2 x.2)
  mul_pos := by rintro ⟨x⟩ ⟨y⟩; exact x.2.mul_pos y.2
  decidableLE := Classical.decRel _


