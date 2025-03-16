instance [h : Nontrivial α] : Nontrivial αᵒᵈ :=
  h


/-- `toDual` is the identity function to the `OrderDual` of a linear order. -/
def toDual : α ≃ αᵒᵈ :=
  Equiv.refl _


/-- `ofDual` is the identity function from the `OrderDual` of a linear order. -/
def ofDual : αᵒᵈ ≃ α :=
  Equiv.refl _


@[simp]
theorem toDual_symm_eq : (@toDual α).symm = ofDual := rfl


@[simp]
theorem ofDual_symm_eq : (@ofDual α).symm = toDual := rfl


@[simp]
theorem toDual_ofDual (a : αᵒᵈ) : toDual (ofDual a) = a :=
  rfl


@[simp]
theorem ofDual_toDual (a : α) : ofDual (toDual a) = a :=
  rfl

-- Porting note:
-- removed @[simp] since this already follows by `simp only [EmbeddingLike.apply_eq_iff_eq]`

theorem toDual_inj {a b : α} : toDual a = toDual b ↔ a = b :=
  Iff.rfl

-- Porting note:
-- removed @[simp] since this already follows by `simp only [EmbeddingLike.apply_eq_iff_eq]`

theorem ofDual_inj {a b : αᵒᵈ} : ofDual a = ofDual b ↔ a = b :=
  Iff.rfl


@[simp]
theorem toDual_le_toDual [LE α] {a b : α} : toDual a ≤ toDual b ↔ b ≤ a :=
  Iff.rfl


@[simp]
theorem toDual_lt_toDual [LT α] {a b : α} : toDual a < toDual b ↔ b < a :=
  Iff.rfl


@[simp]
theorem ofDual_le_ofDual [LE α] {a b : αᵒᵈ} : ofDual a ≤ ofDual b ↔ b ≤ a :=
  Iff.rfl


@[simp]
theorem ofDual_lt_ofDual [LT α] {a b : αᵒᵈ} : ofDual a < ofDual b ↔ b < a :=
  Iff.rfl


theorem le_toDual [LE α] {a : αᵒᵈ} {b : α} : a ≤ toDual b ↔ b ≤ ofDual a :=
  Iff.rfl


theorem lt_toDual [LT α] {a : αᵒᵈ} {b : α} : a < toDual b ↔ b < ofDual a :=
  Iff.rfl


theorem toDual_le [LE α] {a : α} {b : αᵒᵈ} : toDual a ≤ b ↔ ofDual b ≤ a :=
  Iff.rfl


theorem toDual_lt [LT α] {a : α} {b : αᵒᵈ} : toDual a < b ↔ ofDual b < a :=
  Iff.rfl


/-- Recursor for `αᵒᵈ`. -/
@[elab_as_elim]
protected def rec {C : αᵒᵈ → Sort*} (h₂ : ∀ a : α, C (toDual a)) : ∀ a : αᵒᵈ, C a :=
  h₂


@[simp]
protected theorem «forall» {p : αᵒᵈ → Prop} : (∀ a, p a) ↔ ∀ a, p (toDual a) :=
  Iff.rfl


@[simp]
protected theorem «exists» {p : αᵒᵈ → Prop} : (∃ a, p a) ↔ ∃ a, p (toDual a) :=
  Iff.rfl


alias ⟨_, _root_.LE.le.dual⟩ := toDual_le_toDual


alias ⟨_, _root_.LT.lt.dual⟩ := toDual_lt_toDual


alias ⟨_, _root_.LE.le.ofDual⟩ := ofDual_le_ofDual


alias ⟨_, _root_.LT.lt.ofDual⟩ := ofDual_lt_ofDual


/-- A type synonym to equip a type with its lexicographic order. -/
def Lex (α : Type*) :=
  α


/-- `toLex` is the identity function to the `Lex` of a type. -/
@[match_pattern]
def toLex : α ≃ Lex α :=
  Equiv.refl _


/-- `ofLex` is the identity function from the `Lex` of a type. -/
@[match_pattern]
def ofLex : Lex α ≃ α :=
  Equiv.refl _


@[simp]
theorem toLex_symm_eq : (@toLex α).symm = ofLex :=
  rfl


@[simp]
theorem ofLex_symm_eq : (@ofLex α).symm = toLex :=
  rfl


@[simp]
theorem toLex_ofLex (a : Lex α) : toLex (ofLex a) = a :=
  rfl


@[simp]
theorem ofLex_toLex (a : α) : ofLex (toLex a) = a :=
  rfl

-- Porting note:
-- removed @[simp] since this already follows by `simp only [EmbeddingLike.apply_eq_iff_eq]`

theorem toLex_inj {a b : α} : toLex a = toLex b ↔ a = b :=
  Iff.rfl

-- Porting note:
-- removed @[simp] since this already follows by `simp only [EmbeddingLike.apply_eq_iff_eq]`

theorem ofLex_inj {a b : Lex α} : ofLex a = ofLex b ↔ a = b :=
  Iff.rfl


instance (α : Type*) [BEq α] : BEq (Lex α) where
  beq a b := ofLex a == ofLex b


instance (α : Type*) [BEq α] [LawfulBEq α] : LawfulBEq (Lex α) :=
  inferInstanceAs (LawfulBEq α)


instance (α : Type*) [DecidableEq α] : DecidableEq (Lex α) :=
  inferInstanceAs (DecidableEq α)


instance (α : Type*) [Inhabited α] : Inhabited (Lex α) :=
  inferInstanceAs (Inhabited α)


/-- A recursor for `Lex`. Use as `induction x`. -/
@[elab_as_elim, induction_eliminator, cases_eliminator]
protected def Lex.rec {β : Lex α → Sort*} (h : ∀ a, β (toLex a)) : ∀ a, β a := fun a => h (ofLex a)


@[simp] lemma Lex.forall {p : Lex α → Prop} : (∀ a, p a) ↔ ∀ a, p (toLex a) := Iff.rfl

@[simp] lemma Lex.exists {p : Lex α → Prop} : (∃ a, p a) ↔ ∃ a, p (toLex a) := Iff.rfl

