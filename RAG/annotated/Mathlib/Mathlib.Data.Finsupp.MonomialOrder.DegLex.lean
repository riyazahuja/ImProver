/-- A type synonym to equip a type with its lexicographic order sorted by degrees. -/
def DegLex (α : Type*) := α


/-- `toDegLex` is the identity function to the `DegLex` of a type.  -/
@[match_pattern] def toDegLex : α ≃ DegLex α := Equiv.refl _


theorem toDegLex_injective : Function.Injective (toDegLex (α := α)) := fun _ _ ↦ _root_.id


theorem toDegLex_inj {a b : α} : toDegLex a = toDegLex b ↔ a = b := Iff.rfl


/-- `ofDegLex` is the identity function from the `DegLex` of a type.  -/
@[match_pattern] def ofDegLex : DegLex α ≃ α := Equiv.refl _


theorem ofDegLex_injective : Function.Injective (ofDegLex (α := α)) := fun _ _ ↦ _root_.id


theorem ofDegLex_inj {a b : DegLex α} : ofDegLex a = ofDegLex b ↔ a = b := Iff.rfl


@[simp] theorem ofDegLex_symm_eq : (@ofDegLex α).symm = toDegLex := rfl


@[simp] theorem toDegLex_symm_eq : (@toDegLex α).symm = ofDegLex := rfl


@[simp] theorem ofDegLex_toDegLex (a : α) : ofDegLex (toDegLex a) = a := rfl


@[simp] theorem toDegLex_ofDegLex (a : DegLex α) : toDegLex (ofDegLex a) = a := rfl


/-- A recursor for `DegLex`. Use as `induction x`. -/
@[elab_as_elim, induction_eliminator, cases_eliminator]
protected def DegLex.rec {β : DegLex α → Sort*} (h : ∀ a, β (toDegLex a)) :
    ∀ a, β a := fun a => h (ofDegLex a)


@[simp] lemma DegLex.forall_iff {p : DegLex α → Prop} : (∀ a, p a) ↔ ∀ a, p (toDegLex a) := Iff.rfl

@[simp] lemma DegLex.exists_iff {p : DegLex α → Prop} : (∃ a, p a) ↔ ∃ a, p (toDegLex a) := Iff.rfl


noncomputable instance [AddCommMonoid α] :
    AddCommMonoid (DegLex α) := ofDegLex.addCommMonoid


theorem toDegLex_add [AddCommMonoid α] (a b : α) :
    toDegLex (a + b) = toDegLex a + toDegLex b := rfl


theorem ofDegLex_add [AddCommMonoid α] (a b : DegLex α) :
    ofDegLex (a + b) = ofDegLex a + ofDegLex b := rfl


/-- `Finsupp.DegLex r s` is the homogeneous lexicographic order on `α →₀ M`,
where `α` is ordered by `r` and `M` is ordered by `s`.
The type synonym `DegLex (α →₀ M)` has an order given by `Finsupp.DegLex (· < ·) (· < ·)`. -/
protected def DegLex (r : α → α → Prop) (s : ℕ → ℕ → Prop) :
    (α →₀ ℕ) → (α →₀ ℕ) → Prop :=
  (Prod.Lex s (Finsupp.Lex r s)) on (fun x ↦ (x.degree, x))


theorem degLex_def {r : α → α → Prop} {s : ℕ → ℕ → Prop} {a b : α →₀ ℕ} :
    Finsupp.DegLex r s a b ↔ Prod.Lex s (Finsupp.Lex r s) (a.degree, a) (b.degree, b) :=
  Iff.rfl


theorem wellFounded
    {r : α → α → Prop} [IsTrichotomous α r] (hr : WellFounded (Function.swap r))
    {s : ℕ → ℕ → Prop} (hs : WellFounded s) (hs0 : ∀ ⦃n⦄, ¬ s n 0) :
    WellFounded (Finsupp.DegLex r s) := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsTrichotomous α r
    hr : WellFounded (Function.swap r)
    s : Nat → Nat → Prop
    hs : WellFounded s
    hs0 : ∀ ⦃n : Nat⦄, Not (s n 0)
    ⊢ WellFounded (Finsupp.DegLex r s)
  -/
  have wft := WellFounded.prod_lex hs (Finsupp.Lex.wellFounded' hs0 hs hr)
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsTrichotomous α r
    hr : WellFounded (Function.swap r)
    s : Nat → Nat → Prop
    hs : WellFounded s
    hs0 : ∀ ⦃n : Nat⦄, Not (s n 0)
    wft : WellFounded (Prod.Lex s (Finsupp.Lex r s))
    ⊢ WellFounded (Finsupp.DegLex r s)
  -/
  rw [← Set.wellFoundedOn_univ] at wft
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsTrichotomous α r
    hr : WellFounded (Function.swap r)
    s : Nat → Nat → Prop
    hs : WellFounded s
    hs0 : ∀ ⦃n : Nat⦄, Not (s n 0)
    wft : Set.univ.WellFoundedOn (Prod.Lex s (Finsupp.Lex r s))
    ⊢ WellFounded (Finsupp.DegLex r s)
  -/
  unfold Finsupp.DegLex
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsTrichotomous α r
    hr : WellFounded (Function.swap r)
    s : Nat → Nat → Prop
    hs : WellFounded s
    hs0 : ∀ ⦃n : Nat⦄, Not (s n 0)
    wft : Set.univ.WellFoundedOn (Prod.Lex s (Finsupp.Lex r s))
    ⊢ WellFounded (Function.onFun (Prod.Lex s (Finsupp.Lex r s)) fun x => { fst := …
  -/
  rw [← Set.wellFoundedOn_range]
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsTrichotomous α r
    hr : WellFounded (Function.swap r)
    s : Nat → Nat → Prop
    hs : WellFounded s
    hs0 : ∀ ⦃n : Nat⦄, Not (s n 0)
    wft : Set.univ.WellFoundedOn (Prod.Lex s (Finsupp.Lex r s))
    ⊢ (Set.range fun x => { fst := x.degree, snd := x }).WellFoundedOn (Prod.Lex s …
  -/
  exact Set.WellFoundedOn.mono wft (le_refl _) (fun _ _ ↦ trivial)
  /-
    🎉 no goals
  -/


instance [LT α] : LT (DegLex (α →₀ ℕ)) :=
  ⟨fun f g ↦ Finsupp.DegLex (· < ·) (· < ·) (ofDegLex f) (ofDegLex g)⟩


theorem lt_def [LT α] {a b : DegLex (α →₀ ℕ)} :
    a < b ↔ (toLex ((ofDegLex a).degree, toLex (ofDegLex a))) <
        (toLex ((ofDegLex b).degree, toLex (ofDegLex b))) :=
  Iff.rfl


theorem lt_iff [LT α] {a b : DegLex (α →₀ ℕ)} :
    a < b ↔ (ofDegLex a).degree < (ofDegLex b).degree ∨
    (((ofDegLex a).degree = (ofDegLex b).degree) ∧ toLex (ofDegLex a) < toLex (ofDegLex b)) := by
  /-
    α : Type u_1
    inst✝ : LT α
    a b : DegLex (Finsupp α Nat)
    ⊢ Iff (LT.lt a b) (Or (LT.lt (ofDegLex a).degree (ofDegLex b).degree) (And (Eq …
  -/
  simp only [lt_def, Prod.Lex.lt_iff]
  /-
    🎉 no goals
  -/


instance isStrictOrder : IsStrictOrder (DegLex (α →₀ ℕ)) (· < ·) where
                       /-
                         α : Type u_1
                         inst✝ : LinearOrder α
                         a : DegLex (Finsupp α Nat)
                         ⊢ Not (LT.lt a a)
                       -/
  irrefl := fun a ↦ by simp [lt_def]
                       /-
                         🎉 no goals
                       -/
  trans := by
    /-
      α : Type u_1
      inst✝ : LinearOrder α
      ⊢ ∀ (a b c : DegLex (Finsupp α Nat)), LT.lt a b → LT.lt b c → LT.lt a c
    -/
    intro a b c hab hbc
    /-
      α : Type u_1
      inst✝ : LinearOrder α
      a b c : DegLex (Finsupp α Nat)
      hab : LT.lt a b
      hbc : LT.lt b c
      ⊢ LT.lt a c
    -/
    simp only [lt_iff] at hab hbc ⊢
    /-
      α : Type u_1
      inst✝ : LinearOrder α
      a b c : DegLex (Finsupp α Nat)
      hab : Or (LT.lt (ofDegLex a).degree (ofDegLex b).degree) (And (Eq (ofDegLex a) …
      hbc : Or (LT.lt (ofDegLex b).degree (ofDegLex c).degree) (And (Eq (ofDegLex b) …
      ⊢ Or (LT.lt (ofDegLex a).degree (ofDegLex c).degree) (And (Eq (ofDegLex a).deg …
    -/
    rcases hab with (hab | hab)
      /-
        case inl
        α : Type u_1
        inst✝ : LinearOrder α
        a b c : DegLex (Finsupp α Nat)
        hbc : Or (LT.lt (ofDegLex b).degree (ofDegLex c).degree) (And (Eq (ofDegLex b) …
        hab : LT.lt (ofDegLex a).degree (ofDegLex b).degree
        ⊢ Or (LT.lt (ofDegLex a).degree (ofDegLex c).degree) (And (Eq (ofDegLex a).deg …
      -/
    · rcases hbc with (hbc | hbc)
        /-
          case inl.inl
          α : Type u_1
          inst✝ : LinearOrder α
          a b c : DegLex (Finsupp α Nat)
          hab : LT.lt (ofDegLex a).degree (ofDegLex b).degree
          hbc : LT.lt (ofDegLex b).degree (ofDegLex c).degree
          ⊢ Or (LT.lt (ofDegLex a).degree (ofDegLex c).degree) (And (Eq (ofDegLex a).deg …
        -/
      · left; exact lt_trans hab hbc
              /-
                🎉 no goals
              -/
        /-
          case inl.inr
          α : Type u_1
          inst✝ : LinearOrder α
          a b c : DegLex (Finsupp α Nat)
          hab : LT.lt (ofDegLex a).degree (ofDegLex b).degree
          hbc : And (Eq (ofDegLex b).degree (ofDegLex c).degree) (LT.lt (toLex (ofDegLex …
          ⊢ Or (LT.lt (ofDegLex a).degree (ofDegLex c).degree) (And (Eq (ofDegLex a).deg …
        -/
      · left; exact lt_of_lt_of_eq hab hbc.1
              /-
                🎉 no goals
              -/
      /-
        case inr
        α : Type u_1
        inst✝ : LinearOrder α
        a b c : DegLex (Finsupp α Nat)
        hbc : Or (LT.lt (ofDegLex b).degree (ofDegLex c).degree) (And (Eq (ofDegLex b) …
        hab : And (Eq (ofDegLex a).degree (ofDegLex b).degree) (LT.lt (toLex (ofDegLex …
        ⊢ Or (LT.lt (ofDegLex a).degree (ofDegLex c).degree) (And (Eq (ofDegLex a).deg …
      -/
    · rcases hbc with (hbc | hbc)
        /-
          case inr.inl
          α : Type u_1
          inst✝ : LinearOrder α
          a b c : DegLex (Finsupp α Nat)
          hab : And (Eq (ofDegLex a).degree (ofDegLex b).degree) (LT.lt (toLex (ofDegLex …
          hbc : LT.lt (ofDegLex b).degree (ofDegLex c).degree
          ⊢ Or (LT.lt (ofDegLex a).degree (ofDegLex c).degree) (And (Eq (ofDegLex a).deg …
        -/
      · left; exact lt_of_eq_of_lt hab.1 hbc
              /-
                🎉 no goals
              -/
        /-
          case inr.inr
          α : Type u_1
          inst✝ : LinearOrder α
          a b c : DegLex (Finsupp α Nat)
          hab : And (Eq (ofDegLex a).degree (ofDegLex b).degree) (LT.lt (toLex (ofDegLex …
          hbc : And (Eq (ofDegLex b).degree (ofDegLex c).degree) (LT.lt (toLex (ofDegLex …
          ⊢ Or (LT.lt (ofDegLex a).degree (ofDegLex c).degree) (And (Eq (ofDegLex a).deg …
        -/
      · right; exact ⟨Eq.trans hab.1 hbc.1, lt_trans hab.2 hbc.2⟩
               /-
                 🎉 no goals
               -/


/-- The linear order on `Finsupp`s obtained by the homogeneous lexicographic ordering. -/
instance : LinearOrder (DegLex (α →₀ ℕ)) :=
  LinearOrder.lift'
    (fun (f : DegLex (α →₀ ℕ)) ↦ toLex ((ofDegLex f).degree, toLex (ofDegLex f)))
                  /-
                    α : Type u_1
                    inst✝ : LinearOrder α
                    f g : DegLex (Finsupp α Nat)
                    ⊢ Eq ((fun f => toLex { fst := (ofDegLex f).degree, snd := toLex (ofDegLex f)  …
                  -/
    (fun f g ↦ by simp)
                  /-
                    🎉 no goals
                  -/


theorem le_iff {x y : DegLex (α →₀ ℕ)} :
    x ≤ y ↔ (ofDegLex x).degree < (ofDegLex y).degree ∨
      (ofDegLex x).degree = (ofDegLex y).degree ∧ toLex (ofDegLex x) ≤ toLex (ofDegLex y) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    x y : DegLex (Finsupp α Nat)
    ⊢ Iff (LE.le x y) (Or (LT.lt (ofDegLex x).degree (ofDegLex y).degree) (And (Eq …
  -/
  simp only [le_iff_eq_or_lt, lt_iff, EmbeddingLike.apply_eq_iff_eq]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    x y : DegLex (Finsupp α Nat)
    ⊢ Iff (Or (Eq x y) (Or (LT.lt (ofDegLex x).degree (ofDegLex y).degree) (And (E …
  -/
  by_cases h : x = y
    /-
      case pos
      α : Type u_1
      inst✝ : LinearOrder α
      x y : DegLex (Finsupp α Nat)
      h : Eq x y
      ⊢ Iff (Or (Eq x y) (Or (LT.lt (ofDegLex x).degree (ofDegLex y).degree) (And (E …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : LinearOrder α
      x y : DegLex (Finsupp α Nat)
      h : Not (Eq x y)
      ⊢ Iff (Or (Eq x y) (Or (LT.lt (ofDegLex x).degree (ofDegLex y).degree) (And (E …
    -/
  · by_cases k : (ofDegLex x).degree < (ofDegLex y).degree
      /-
        case pos
        α : Type u_1
        inst✝ : LinearOrder α
        x y : DegLex (Finsupp α Nat)
        h : Not (Eq x y)
        k : LT.lt (ofDegLex x).degree (ofDegLex y).degree
        ⊢ Iff (Or (Eq x y) (Or (LT.lt (ofDegLex x).degree (ofDegLex y).degree) (And (E …
      -/
    · simp [k]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝ : LinearOrder α
        x y : DegLex (Finsupp α Nat)
        h : Not (Eq x y)
        k : Not (LT.lt (ofDegLex x).degree (ofDegLex y).degree)
        ⊢ Iff (Or (Eq x y) (Or (LT.lt (ofDegLex x).degree (ofDegLex y).degree) (And (E …
      -/
    · simp only [h, k, false_or]
      /-
        🎉 no goals
      -/


noncomputable instance : OrderedCancelAddCommMonoid (DegLex (α →₀ ℕ)) where
  le_of_add_le_add_left a b c h := by
    /-
      α : Type u_1
      inst✝ : LinearOrder α
      a b c : DegLex (Finsupp α Nat)
      h : LE.le (HAdd.hAdd a b) (HAdd.hAdd a c)
      ⊢ LE.le b c
    -/
    rw [le_iff] at h ⊢
    simpa only [ofDegLex_add, degree_add, add_lt_add_iff_left, add_right_inj, toLex_add,
      add_le_add_iff_left] using h
    /-
      α : Type u_1
      inst✝ : LinearOrder α
      a b : DegLex (Finsupp α Nat)
      h : LE.le a b
      c : DegLex (Finsupp α Nat)
      ⊢ LE.le (HAdd.hAdd c a) (HAdd.hAdd c b)
    -/
  add_le_add_left a b h c := by
    /-
      α : Type u_1
      inst✝ : LinearOrder α
      a b : DegLex (Finsupp α Nat)
      h : Or (LT.lt (ofDegLex a).degree (ofDegLex b).degree) (And (Eq (ofDegLex a).d …
      c : DegLex (Finsupp α Nat)
      ⊢ Or (LT.lt (ofDegLex (HAdd.hAdd c a)).degree (ofDegLex (HAdd.hAdd c b)).degre …
    -/
    rw [le_iff] at h ⊢
    /-
      🎉 no goals
    -/
    simpa [ofDegLex_add, degree_add] using h


/-- The linear order on `Finsupp`s obtained by the homogeneous lexicographic ordering. -/
noncomputable instance :
    LinearOrderedCancelAddCommMonoid (DegLex (α →₀ ℕ)) where
  le_total := instLinearOrderDegLexNat.le_total
  decidableLE := instLinearOrderDegLexNat.decidableLE
  compare_eq_compareOfLessAndEq := instLinearOrderDegLexNat.compare_eq_compareOfLessAndEq


theorem single_strictAnti : StrictAnti (fun (a : α) ↦ toDegLex (single a 1)) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    ⊢ StrictAnti fun a => toDegLex (Finsupp.single a 1)
  -/
  intro _ _ h
  simp only [lt_iff, ofDegLex_toDegLex, degree_single, lt_self_iff_false, Lex.single_lt_iff, h,
    and_self, or_true]


theorem single_antitone : Antitone (fun (a : α) ↦ toDegLex (single a 1)) :=
  single_strictAnti.antitone


theorem single_lt_iff {a b : α} :
    toDegLex (Finsupp.single b 1) < toDegLex (Finsupp.single a 1) ↔ a < b :=
  single_strictAnti.lt_iff_lt


theorem single_le_iff {a b : α} :
    toDegLex (Finsupp.single b 1) ≤ toDegLex (Finsupp.single a 1) ↔ a ≤ b :=
  single_strictAnti.le_iff_le


theorem monotone_degree :
    Monotone (fun (x : DegLex (α →₀ ℕ)) ↦ (ofDegLex x).degree) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    ⊢ Monotone fun x => (ofDegLex x).degree
  -/
  intro x y
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    x y : DegLex (Finsupp α Nat)
    ⊢ LE.le x y → LE.le ((fun x => (ofDegLex x).degree) x) ((fun x => (ofDegLex x) …
  -/
  rw [le_iff]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    x y : DegLex (Finsupp α Nat)
    ⊢ Or (LT.lt (ofDegLex x).degree (ofDegLex y).degree) (And (Eq (ofDegLex x).deg …
  -/
  rintro (h | h)
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrder α
      x y : DegLex (Finsupp α Nat)
      h : LT.lt (ofDegLex x).degree (ofDegLex y).degree
      ⊢ LE.le ((fun x => (ofDegLex x).degree) x) ((fun x => (ofDegLex x).degree) y)
    -/
  · apply le_of_lt h
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrder α
      x y : DegLex (Finsupp α Nat)
      h : And (Eq (ofDegLex x).degree (ofDegLex y).degree) (LE.le (toLex (ofDegLex x …
      ⊢ LE.le ((fun x => (ofDegLex x).degree) x) ((fun x => (ofDegLex x).degree) y)
    -/
  · apply le_of_eq h.1
    /-
      🎉 no goals
    -/


instance orderBot : OrderBot (DegLex (α →₀ ℕ)) where
  bot := toDegLex (0 : α →₀ ℕ)
  bot_le x := by
    /-
      α : Type u_1
      inst✝ : LinearOrder α
      x : DegLex (Finsupp α Nat)
      ⊢ LE.le Bot.bot x
    -/
    simp only [le_iff, ofDegLex_toDegLex, toLex_zero, degree_zero]
    /-
      α : Type u_1
      inst✝ : LinearOrder α
      x : DegLex (Finsupp α Nat)
      ⊢ Or (LT.lt 0 (ofDegLex x).degree) (And (Eq 0 (ofDegLex x).degree) (LE.le 0 (t …
    -/
    rcases eq_zero_or_pos (ofDegLex x).degree with (h | h)
      /-
        case inl
        α : Type u_1
        inst✝ : LinearOrder α
        x : DegLex (Finsupp α Nat)
        h : Eq (ofDegLex x).degree 0
        ⊢ Or (LT.lt 0 (ofDegLex x).degree) (And (Eq 0 (ofDegLex x).degree) (LE.le 0 (t …
      -/
    · simp only [h, lt_self_iff_false, true_and, false_or, ge_iff_le]
      /-
        case inl
        α : Type u_1
        inst✝ : LinearOrder α
        x : DegLex (Finsupp α Nat)
        h : Eq (ofDegLex x).degree 0
        ⊢ LE.le 0 (toLex (ofDegLex x))
      -/
      exact bot_le
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_1
        inst✝ : LinearOrder α
        x : DegLex (Finsupp α Nat)
        h : LT.lt 0 (ofDegLex x).degree
        ⊢ Or (LT.lt 0 (ofDegLex x).degree) (And (Eq 0 (ofDegLex x).degree) (LE.le 0 (t …
      -/
    · simp [h]
      /-
        🎉 no goals
      -/


instance wellFoundedLT [WellFoundedGT α] :
    WellFoundedLT (DegLex (α →₀ ℕ)) :=
  ⟨wellFounded wellFounded_gt wellFounded_lt fun n ↦ (zero_le n).not_lt⟩


/-- The deg-lexicographic order on `σ →₀ ℕ`, as a `MonomialOrder` -/
noncomputable def degLex :
    MonomialOrder σ where
  syn := DegLex (σ →₀ ℕ)
  toSyn := { toEquiv := toDegLex, map_add' := toDegLex_add }
  toSyn_monotone a b h := by
    /-
      α : Type u_1
      σ : Type u_2
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      a b : Finsupp σ Nat
      h : LE.le a b
      ⊢ LE.le ({ toEquiv := toDegLex, map_add' := ⋯ } a) ({ toEquiv := toDegLex, map …
    -/
    simp only [AddEquiv.coe_mk, DegLex.le_iff, ofDegLex_toDegLex]
    /-
      α : Type u_1
      σ : Type u_2
      inst✝¹ : LinearOrder σ
      inst✝ : WellFoundedGT σ
      a b : Finsupp σ Nat
      h : LE.le a b
      ⊢ Or (LT.lt a.degree b.degree) (And (Eq a.degree b.degree) (LE.le (toLex a) (t …
    -/
    by_cases ha : a.degree < b.degree
      /-
        case pos
        α : Type u_1
        σ : Type u_2
        inst✝¹ : LinearOrder σ
        inst✝ : WellFoundedGT σ
        a b : Finsupp σ Nat
        h : LE.le a b
        ha : LT.lt a.degree b.degree
        ⊢ Or (LT.lt a.degree b.degree) (And (Eq a.degree b.degree) (LE.le (toLex a) (t …
      -/
    · exact Or.inl ha
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        σ : Type u_2
        inst✝¹ : LinearOrder σ
        inst✝ : WellFoundedGT σ
        a b : Finsupp σ Nat
        h : LE.le a b
        ha : Not (LT.lt a.degree b.degree)
        ⊢ Or (LT.lt a.degree b.degree) (And (Eq a.degree b.degree) (LE.le (toLex a) (t …
      -/
    · refine Or.inr ⟨le_antisymm ?_ (not_lt.mp ha), toLex_monotone h⟩
      /-
        case neg
        α : Type u_1
        σ : Type u_2
        inst✝¹ : LinearOrder σ
        inst✝ : WellFoundedGT σ
        a b : Finsupp σ Nat
        h : LE.le a b
        ha : Not (LT.lt a.degree b.degree)
        ⊢ LE.le a.degree b.degree
      -/
      rw [← add_tsub_cancel_of_le h, degree_add]
      /-
        case neg
        α : Type u_1
        σ : Type u_2
        inst✝¹ : LinearOrder σ
        inst✝ : WellFoundedGT σ
        a b : Finsupp σ Nat
        h : LE.le a b
        ha : Not (LT.lt a.degree b.degree)
        ⊢ LE.le a.degree (HAdd.hAdd a.degree (HSub.hSub b a).degree)
      -/
      exact Nat.le_add_right a.degree (b - a).degree
      /-
        🎉 no goals
      -/


theorem degLex_le_iff {a b : σ →₀ ℕ} :
    a ≼[degLex] b ↔ toDegLex a ≤ toDegLex b :=
  Iff.rfl


theorem degLex_lt_iff {a b : σ →₀ ℕ} :
    a ≺[degLex] b ↔ toDegLex a < toDegLex b :=
  Iff.rfl


theorem degLex_single_le_iff {a b : σ} :
    single a 1 ≼[degLex] single b 1 ↔ b ≤ a := by
  /-
    σ : Type u_2
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    a b : σ
    ⊢ Iff (LE.le (MonomialOrder.degLex.toSyn (Finsupp.single a 1)) (MonomialOrder. …
  -/
  rw [MonomialOrder.degLex_le_iff, DegLex.single_le_iff]
  /-
    🎉 no goals
  -/


theorem degLex_single_lt_iff {a b : σ} :
    single a 1 ≺[degLex] single b 1 ↔ b < a := by
  /-
    σ : Type u_2
    inst✝¹ : LinearOrder σ
    inst✝ : WellFoundedGT σ
    a b : σ
    ⊢ Iff (LT.lt (MonomialOrder.degLex.toSyn (Finsupp.single a 1)) (MonomialOrder. …
  -/
  rw [MonomialOrder.degLex_lt_iff, DegLex.single_lt_iff]
  /-
    🎉 no goals
  -/


