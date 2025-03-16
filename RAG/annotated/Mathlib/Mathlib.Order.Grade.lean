/-- An `𝕆`-graded order is an order `α` equipped with a strictly monotone function
`grade 𝕆 : α → 𝕆` which preserves order covering (`CovBy`). -/
class GradeOrder (𝕆 α : Type*) [Preorder 𝕆] [Preorder α] where
  /-- The grading function. -/
  protected grade : α → 𝕆
  /-- `grade` is strictly monotonic. -/
  grade_strictMono : StrictMono grade
  /-- `grade` preserves `CovBy`. -/
  covBy_grade ⦃a b : α⦄ : a ⋖ b → grade a ⋖ grade b


/-- An `𝕆`-graded order where minimal elements have minimal grades. -/
class GradeMinOrder (𝕆 α : Type*) [Preorder 𝕆] [Preorder α] extends GradeOrder 𝕆 α where
  /-- Minimal elements have minimal grades. -/
  isMin_grade ⦃a : α⦄ : IsMin a → IsMin (grade a)


/-- An `𝕆`-graded order where maximal elements have maximal grades. -/
class GradeMaxOrder (𝕆 α : Type*) [Preorder 𝕆] [Preorder α] extends GradeOrder 𝕆 α where
  /-- Maximal elements have maximal grades. -/
  isMax_grade ⦃a : α⦄ : IsMax a → IsMax (grade a)


/-- An `𝕆`-graded order where minimal elements have minimal grades and maximal elements have maximal
grades. -/
class GradeBoundedOrder (𝕆 α : Type*) [Preorder 𝕆] [Preorder α] extends GradeMinOrder 𝕆 α,
  GradeMaxOrder 𝕆 α


/-- The grade of an element in a graded order. Morally, this is the number of elements you need to
go down by to get to `⊥`. -/
def grade : α → 𝕆 :=
  GradeOrder.grade


protected theorem CovBy.grade (h : a ⋖ b) : grade 𝕆 a ⋖ grade 𝕆 b :=
  GradeOrder.covBy_grade h


theorem grade_strictMono : StrictMono (grade 𝕆 : α → 𝕆) :=
  GradeOrder.grade_strictMono


theorem covBy_iff_lt_covBy_grade : a ⋖ b ↔ a < b ∧ grade 𝕆 a ⋖ grade 𝕆 b :=
  ⟨fun h => ⟨h.1, h.grade _⟩,
    And.imp_right fun h _ ha hb => h.2 (grade_strictMono ha) <| grade_strictMono hb⟩


protected theorem IsMin.grade (h : IsMin a) : IsMin (grade 𝕆 a) :=
  GradeMinOrder.isMin_grade h


@[simp]
theorem isMin_grade_iff : IsMin (grade 𝕆 a) ↔ IsMin a :=
  ⟨grade_strictMono.isMin_of_apply, IsMin.grade _⟩


protected theorem IsMax.grade (h : IsMax a) : IsMax (grade 𝕆 a) :=
  GradeMaxOrder.isMax_grade h


@[simp]
theorem isMax_grade_iff : IsMax (grade 𝕆 a) ↔ IsMax a :=
  ⟨grade_strictMono.isMax_of_apply, IsMax.grade _⟩


theorem grade_mono [PartialOrder α] [GradeOrder 𝕆 α] : Monotone (grade 𝕆 : α → 𝕆) :=
  grade_strictMono.monotone


theorem grade_injective : Function.Injective (grade 𝕆 : α → 𝕆) :=
  grade_strictMono.injective


@[simp]
theorem grade_le_grade_iff : grade 𝕆 a ≤ grade 𝕆 b ↔ a ≤ b :=
  grade_strictMono.le_iff_le


@[simp]
theorem grade_lt_grade_iff : grade 𝕆 a < grade 𝕆 b ↔ a < b :=
  grade_strictMono.lt_iff_lt


@[simp]
theorem grade_eq_grade_iff : grade 𝕆 a = grade 𝕆 b ↔ a = b :=
  grade_injective.eq_iff


theorem grade_ne_grade_iff : grade 𝕆 a ≠ grade 𝕆 b ↔ a ≠ b :=
  grade_injective.ne_iff


theorem grade_covBy_grade_iff : grade 𝕆 a ⋖ grade 𝕆 b ↔ a ⋖ b :=
  (covBy_iff_lt_covBy_grade.trans <| and_iff_right_of_imp fun h => grade_lt_grade_iff.1 h.1).symm


@[simp]
theorem grade_bot [OrderBot 𝕆] [OrderBot α] [GradeMinOrder 𝕆 α] : grade 𝕆 (⊥ : α) = ⊥ :=
  (isMin_bot.grade _).eq_bot


@[simp]
theorem grade_top [OrderTop 𝕆] [OrderTop α] [GradeMaxOrder 𝕆 α] : grade 𝕆 (⊤ : α) = ⊤ :=
  (isMax_top.grade _).eq_top


instance Preorder.toGradeBoundedOrder : GradeBoundedOrder α α where
  grade := id
  isMin_grade _ := id
  isMax_grade _ := id
  grade_strictMono := strictMono_id
  covBy_grade _ _ := id


@[simp]
theorem grade_self (a : α) : grade α a = a :=
  rfl


instance OrderDual.gradeOrder [GradeOrder 𝕆 α] : GradeOrder 𝕆ᵒᵈ αᵒᵈ where
  grade := toDual ∘ grade 𝕆 ∘ ofDual
  grade_strictMono := grade_strictMono.dual
  covBy_grade _ _ h := (h.ofDual.grade _).toDual


instance OrderDual.gradeMinOrder [GradeMaxOrder 𝕆 α] : GradeMinOrder 𝕆ᵒᵈ αᵒᵈ :=
  { OrderDual.gradeOrder with isMin_grade := fun _ => IsMax.grade (α := α) 𝕆 }


instance OrderDual.gradeMaxOrder [GradeMinOrder 𝕆 α] : GradeMaxOrder 𝕆ᵒᵈ αᵒᵈ :=
  { OrderDual.gradeOrder with isMax_grade := fun _ => IsMin.grade (α := α) 𝕆 }


instance [GradeBoundedOrder 𝕆 α] : GradeBoundedOrder 𝕆ᵒᵈ αᵒᵈ :=
  { OrderDual.gradeMinOrder, OrderDual.gradeMaxOrder with }


@[simp]
theorem grade_toDual [GradeOrder 𝕆 α] (a : α) : grade 𝕆ᵒᵈ (toDual a) = toDual (grade 𝕆 a) :=
  rfl


@[simp]
theorem grade_ofDual [GradeOrder 𝕆 α] (a : αᵒᵈ) : grade 𝕆 (ofDual a) = ofDual (grade 𝕆ᵒᵈ a) :=
  rfl


/-- Lifts a graded order along a strictly monotone function. -/
abbrev GradeOrder.liftLeft [GradeOrder 𝕆 α] (f : 𝕆 → ℙ) (hf : StrictMono f)
    (hcovBy : ∀ a b, a ⋖ b → f a ⋖ f b) : GradeOrder ℙ α where
  grade := f ∘ grade 𝕆
  grade_strictMono := hf.comp grade_strictMono
  covBy_grade _ _ h := hcovBy _ _ <| h.grade _

-- See note [reducible non-instances]

/-- Lifts a graded order along a strictly monotone function. -/
abbrev GradeMinOrder.liftLeft [GradeMinOrder 𝕆 α] (f : 𝕆 → ℙ) (hf : StrictMono f)
    (hcovBy : ∀ a b, a ⋖ b → f a ⋖ f b) (hmin : ∀ a, IsMin a → IsMin (f a)) : GradeMinOrder ℙ α :=
  { GradeOrder.liftLeft f hf hcovBy with isMin_grade := fun _ ha => hmin _ <| ha.grade _ }

-- See note [reducible non-instances]

/-- Lifts a graded order along a strictly monotone function. -/
abbrev GradeMaxOrder.liftLeft [GradeMaxOrder 𝕆 α] (f : 𝕆 → ℙ) (hf : StrictMono f)
    (hcovBy : ∀ a b, a ⋖ b → f a ⋖ f b) (hmax : ∀ a, IsMax a → IsMax (f a)) : GradeMaxOrder ℙ α :=
  { GradeOrder.liftLeft f hf hcovBy with isMax_grade := fun _ ha => hmax _ <| ha.grade _ }

-- See note [reducible non-instances]

/-- Lifts a graded order along a strictly monotone function. -/
abbrev GradeBoundedOrder.liftLeft [GradeBoundedOrder 𝕆 α] (f : 𝕆 → ℙ) (hf : StrictMono f)
    (hcovBy : ∀ a b, a ⋖ b → f a ⋖ f b) (hmin : ∀ a, IsMin a → IsMin (f a))
    (hmax : ∀ a, IsMax a → IsMax (f a)) : GradeBoundedOrder ℙ α :=
  { GradeMinOrder.liftLeft f hf hcovBy hmin, GradeMaxOrder.liftLeft f hf hcovBy hmax with }

-- See note [reducible non-instances]

/-- Lifts a graded order along a strictly monotone function. -/
abbrev GradeOrder.liftRight [GradeOrder 𝕆 β] (f : α → β) (hf : StrictMono f)
    (hcovBy : ∀ a b, a ⋖ b → f a ⋖ f b) : GradeOrder 𝕆 α where
  grade := grade 𝕆 ∘ f
  grade_strictMono := grade_strictMono.comp hf
  covBy_grade _ _ h := (hcovBy _ _ h).grade _

-- See note [reducible non-instances]

/-- Lifts a graded order along a strictly monotone function. -/
abbrev GradeMinOrder.liftRight [GradeMinOrder 𝕆 β] (f : α → β) (hf : StrictMono f)
    (hcovBy : ∀ a b, a ⋖ b → f a ⋖ f b) (hmin : ∀ a, IsMin a → IsMin (f a)) : GradeMinOrder 𝕆 α :=
  { GradeOrder.liftRight f hf hcovBy with isMin_grade := fun _ ha => (hmin _ ha).grade _ }

-- See note [reducible non-instances]

/-- Lifts a graded order along a strictly monotone function. -/
abbrev GradeMaxOrder.liftRight [GradeMaxOrder 𝕆 β] (f : α → β) (hf : StrictMono f)
    (hcovBy : ∀ a b, a ⋖ b → f a ⋖ f b) (hmax : ∀ a, IsMax a → IsMax (f a)) : GradeMaxOrder 𝕆 α :=
  { GradeOrder.liftRight f hf hcovBy with isMax_grade := fun _ ha => (hmax _ ha).grade _ }

-- See note [reducible non-instances]

/-- Lifts a graded order along a strictly monotone function. -/
abbrev GradeBoundedOrder.liftRight [GradeBoundedOrder 𝕆 β] (f : α → β) (hf : StrictMono f)
    (hcovBy : ∀ a b, a ⋖ b → f a ⋖ f b) (hmin : ∀ a, IsMin a → IsMin (f a))
    (hmax : ∀ a, IsMax a → IsMax (f a)) : GradeBoundedOrder 𝕆 α :=
  { GradeMinOrder.liftRight f hf hcovBy hmin, GradeMaxOrder.liftRight f hf hcovBy hmax with }


/-- A `Fin n`-graded order is also `ℕ`-graded. We do not mark this an instance because `n` is not
inferable. -/
abbrev GradeOrder.finToNat (n : ℕ) [GradeOrder (Fin n) α] : GradeOrder ℕ α :=
  (GradeOrder.liftLeft (_ : Fin n → ℕ) Fin.val_strictMono) fun _ _ => CovBy.coe_fin

-- See note [reducible non-instances]

/-- A `Fin n`-graded order is also `ℕ`-graded. We do not mark this an instance because `n` is not
inferable. -/
abbrev GradeMinOrder.finToNat (n : ℕ) [GradeMinOrder (Fin n) α] : GradeMinOrder ℕ α :=
  (GradeMinOrder.liftLeft (_ : Fin n → ℕ) Fin.val_strictMono fun _ _ => CovBy.coe_fin) fun a h => by
    /-
      𝕆 : Type u_1
      ℙ : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝⁴ : Preorder 𝕆
      inst✝³ : Preorder ℙ
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      n : Nat
      inst✝ : GradeMinOrder (Fin n) α
      a : Fin n
      h : IsMin a
      ⊢ IsMin ↑a
    -/
    cases n
      /-
        case zero
        𝕆 : Type u_1
        ℙ : Type u_2
        α : Type u_3
        β : Type u_4
        inst✝⁴ : Preorder 𝕆
        inst✝³ : Preorder ℙ
        inst✝² : Preorder α
        inst✝¹ : Preorder β
        inst✝ : GradeMinOrder (Fin 0) α
        a : Fin 0
        h : IsMin a
        ⊢ IsMin ↑a
      -/
    · exact a.elim0
      /-
        🎉 no goals
      -/
    /-
      case succ
      𝕆 : Type u_1
      ℙ : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝⁴ : Preorder 𝕆
      inst✝³ : Preorder ℙ
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      n✝ : Nat
      inst✝ : GradeMinOrder (Fin (HAdd.hAdd n✝ 1)) α
      a : Fin (HAdd.hAdd n✝ 1)
      h : IsMin a
      ⊢ IsMin ↑a
    -/
    rw [h.eq_bot, Fin.bot_eq_zero]
    /-
      case succ
      𝕆 : Type u_1
      ℙ : Type u_2
      α : Type u_3
      β : Type u_4
      inst✝⁴ : Preorder 𝕆
      inst✝³ : Preorder ℙ
      inst✝² : Preorder α
      inst✝¹ : Preorder β
      n✝ : Nat
      inst✝ : GradeMinOrder (Fin (HAdd.hAdd n✝ 1)) α
      a : Fin (HAdd.hAdd n✝ 1)
      h : IsMin a
      ⊢ IsMin ↑0
    -/
    exact isMin_bot
    /-
      🎉 no goals
    -/


instance GradeOrder.natToInt [GradeOrder ℕ α] : GradeOrder ℤ α :=
  (GradeOrder.liftLeft _ Int.natCast_strictMono) fun _ _ => CovBy.intCast


theorem GradeOrder.wellFoundedLT (𝕆 : Type*) [Preorder 𝕆] [GradeOrder 𝕆 α]
    [WellFoundedLT 𝕆] : WellFoundedLT α :=
  (grade_strictMono (𝕆 := 𝕆)).wellFoundedLT


theorem GradeOrder.wellFoundedGT (𝕆 : Type*) [Preorder 𝕆] [GradeOrder 𝕆 α]
    [WellFoundedGT 𝕆] : WellFoundedGT α :=
  (grade_strictMono (𝕆 := 𝕆)).wellFoundedGT


instance [GradeOrder ℕ α] : WellFoundedLT α :=
  GradeOrder.wellFoundedLT ℕ


instance [GradeOrder ℕᵒᵈ α] : WellFoundedGT α :=
  GradeOrder.wellFoundedGT ℕᵒᵈ

