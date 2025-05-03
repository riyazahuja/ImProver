/-- Monomial orders : equivalence of `σ →₀ ℕ` with a well ordered type -/
structure MonomialOrder (σ : Type*) where
  /-- The synonym type -/
  syn : Type*
  /-- `syn` is a linearly ordered cancellative additive commutative monoid -/
  locacm : LinearOrderedCancelAddCommMonoid syn := by infer_instance
  /-- the additive equivalence from `σ →₀ ℕ` to `syn` -/
  toSyn : (σ →₀ ℕ) ≃+ syn
  /-- `toSyn` is monotone -/
  toSyn_monotone : Monotone toSyn
  /-- `syn` is a well ordering -/
  wf : WellFoundedLT syn := by infer_instance


lemma le_add_right (a b : σ →₀ ℕ) :
    m.toSyn a ≤ m.toSyn a + m.toSyn b := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    a b : Finsupp σ Nat
    ⊢ LE.le (m.toSyn a) (HAdd.hAdd (m.toSyn a) (m.toSyn b))
  -/
  rw [← map_add]
  /-
    σ : Type u_1
    m : MonomialOrder σ
    a b : Finsupp σ Nat
    ⊢ LE.le (m.toSyn a) (m.toSyn (HAdd.hAdd a b))
  -/
  exact m.toSyn_monotone le_self_add
  /-
    🎉 no goals
  -/


instance orderBot : OrderBot (m.syn) where
  bot := 0
  bot_le a := by
    /-
      σ : Type u_1
      m : MonomialOrder σ
      a : m.syn
      ⊢ LE.le Bot.bot a
    -/
    have := m.le_add_right 0 (m.toSyn.symm a)
    /-
      σ : Type u_1
      m : MonomialOrder σ
      a : m.syn
      this : LE.le (m.toSyn 0) (HAdd.hAdd (m.toSyn 0) (m.toSyn (m.toSyn.symm a)))
      ⊢ LE.le Bot.bot a
    -/
    simp [map_add, zero_add] at this
    /-
      σ : Type u_1
      m : MonomialOrder σ
      a : m.syn
      this : LE.le 0 a
      ⊢ LE.le Bot.bot a
    -/
    exact this
    /-
      🎉 no goals
    -/


@[simp]
theorem bot_eq_zero : (⊥ : m.syn) = 0 := rfl


theorem eq_zero_iff {a : m.syn} : a = 0 ↔ a ≤ 0 := eq_bot_iff


lemma toSyn_strictMono : StrictMono (m.toSyn) := by
  /-
    σ : Type u_1
    m : MonomialOrder σ
    ⊢ StrictMono ⇑m.toSyn
  -/
  apply m.toSyn_monotone.strictMono_of_injective m.toSyn.injective
  /-
    🎉 no goals
  -/


/-- Given a monomial order, notation for the corresponding strict order relation on `σ →₀ ℕ` -/
scoped
notation:25 c "≺[" m:25 "]" d:25 => (MonomialOrder.toSyn m c < MonomialOrder.toSyn m d)


/-- Given a monomial order, notation for the corresponding order relation on `σ →₀ ℕ` -/
scoped
notation:25 c "≼[" m:25 "]" d:25 => (MonomialOrder.toSyn m c ≤ MonomialOrder.toSyn m d)


noncomputable instance {α N : Type*} [LinearOrder α] [OrderedCancelAddCommMonoid N] :
    OrderedCancelAddCommMonoid (Lex (α →₀ N)) where
                                      /-
                                        α : Type u_1
                                        N : Type u_2
                                        inst✝¹ : LinearOrder α
                                        inst✝ : OrderedCancelAddCommMonoid N
                                        a b c : Lex (Finsupp α N)
                                        h : LE.le (HAdd.hAdd a b) (HAdd.hAdd a c)
                                        ⊢ LE.le b c
                                      -/
                                /-
                                  α : Type u_1
                                  N : Type u_2
                                  inst✝¹ : LinearOrder α
                                  inst✝ : OrderedCancelAddCommMonoid N
                                  a b : Lex (Finsupp α N)
                                  h : LE.le a b
                                  c : Lex (Finsupp α N)
                                  ⊢ LE.le (HAdd.hAdd c a) (HAdd.hAdd c b)
                                -/
  le_of_add_le_add_left a b c h := by simpa only [add_le_add_iff_left] using h
                                /-
                                  🎉 no goals
                                -/
                                      /-
                                        🎉 no goals
                                      -/
  add_le_add_left a b h c := by simpa only [add_le_add_iff_left] using h


theorem Finsupp.lex_lt_iff {α N : Type*} [LinearOrder α] [LinearOrder N] [Zero N]
    {a b : Lex (α →₀ N)} :
    a < b ↔ ∃ i, (∀ j, j< i → ofLex a j = ofLex b j) ∧ ofLex a i < ofLex b i :=
    Finsupp.lex_def


theorem Finsupp.lex_le_iff {α N : Type*} [LinearOrder α] [LinearOrder N] [Zero N]
    {a b : Lex (α →₀ N)} :
    a ≤ b ↔ a = b ∨ ∃ i, (∀ j, j< i → ofLex a j = ofLex b j) ∧ ofLex a i < ofLex b i := by
    /-
      α : Type u_1
      N : Type u_2
      inst✝² : LinearOrder α
      inst✝¹ : LinearOrder N
      inst✝ : Zero N
      a b : Lex (Finsupp α N)
      ⊢ Iff (LE.le a b) (Or (Eq a b) (Exists fun i => And (∀ (j : α), LT.lt j i → Eq …
    -/
    rw [le_iff_eq_or_lt, Finsupp.lex_lt_iff]
    /-
      🎉 no goals
    -/


/-- The lexicographic order on `σ →₀ ℕ`, as a `MonomialOrder` -/
noncomputable def MonomialOrder.lex [WellFoundedGT σ] :
    MonomialOrder σ where
  syn := Lex (σ →₀ ℕ)
  toSyn :=
  { toEquiv := toLex
    map_add' := toLex_add }
  toSyn_monotone := Finsupp.toLex_monotone


theorem MonomialOrder.lex_le_iff [WellFoundedGT σ] {c d : σ →₀ ℕ} :
    c ≼[lex] d ↔ toLex c ≤ toLex d := Iff.rfl


theorem MonomialOrder.lex_lt_iff [WellFoundedGT σ] {c d : σ →₀ ℕ} :
    c ≺[lex] d ↔ toLex c < toLex d := Iff.rfl


