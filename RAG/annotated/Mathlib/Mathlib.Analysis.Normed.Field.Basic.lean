/-- A non-unital seminormed ring is a not-necessarily-unital ring
endowed with a seminorm which satisfies the inequality `‖x y‖ ≤ ‖x‖ ‖y‖`. -/
class NonUnitalSeminormedRing (α : Type*) extends Norm α, NonUnitalRing α,
  PseudoMetricSpace α where
  /-- The distance is induced by the norm. -/
  dist_eq : ∀ x y, dist x y = norm (x - y)
  /-- The norm is submultiplicative. -/
  norm_mul : ∀ a b, norm (a * b) ≤ norm a * norm b


/-- A seminormed ring is a ring endowed with a seminorm which satisfies the inequality
`‖x y‖ ≤ ‖x‖ ‖y‖`. -/
class SeminormedRing (α : Type*) extends Norm α, Ring α, PseudoMetricSpace α where
  /-- The distance is induced by the norm. -/
  dist_eq : ∀ x y, dist x y = norm (x - y)
  /-- The norm is submultiplicative. -/
  norm_mul : ∀ a b, norm (a * b) ≤ norm a * norm b

-- see Note [lower instance priority]

/-- A seminormed ring is a non-unital seminormed ring. -/
instance (priority := 100) SeminormedRing.toNonUnitalSeminormedRing [β : SeminormedRing α] :
    NonUnitalSeminormedRing α :=
  { β with }


/-- A non-unital normed ring is a not-necessarily-unital ring
endowed with a norm which satisfies the inequality `‖x y‖ ≤ ‖x‖ ‖y‖`. -/
class NonUnitalNormedRing (α : Type*) extends Norm α, NonUnitalRing α, MetricSpace α where
  /-- The distance is induced by the norm. -/
  dist_eq : ∀ x y, dist x y = norm (x - y)
  /-- The norm is submultiplicative. -/
  norm_mul : ∀ a b, norm (a * b) ≤ norm a * norm b

-- see Note [lower instance priority]

/-- A non-unital normed ring is a non-unital seminormed ring. -/
instance (priority := 100) NonUnitalNormedRing.toNonUnitalSeminormedRing
    [β : NonUnitalNormedRing α] : NonUnitalSeminormedRing α :=
  { β with }


/-- A normed ring is a ring endowed with a norm which satisfies the inequality `‖x y‖ ≤ ‖x‖ ‖y‖`. -/
class NormedRing (α : Type*) extends Norm α, Ring α, MetricSpace α where
  /-- The distance is induced by the norm. -/
  dist_eq : ∀ x y, dist x y = norm (x - y)
  /-- The norm is submultiplicative. -/
  norm_mul : ∀ a b, norm (a * b) ≤ norm a * norm b


/-- A normed division ring is a division ring endowed with a seminorm which satisfies the equality
`‖x y‖ = ‖x‖ ‖y‖`. -/
class NormedDivisionRing (α : Type*) extends Norm α, DivisionRing α, MetricSpace α where
  /-- The distance is induced by the norm. -/
  dist_eq : ∀ x y, dist x y = norm (x - y)
  /-- The norm is multiplicative. -/
  norm_mul' : ∀ a b, norm (a * b) = norm a * norm b

-- see Note [lower instance priority]

/-- A normed division ring is a normed ring. -/
instance (priority := 100) NormedDivisionRing.toNormedRing [β : NormedDivisionRing α] :
    NormedRing α :=
  { β with norm_mul := fun a b => (NormedDivisionRing.norm_mul' a b).le }

-- see Note [lower instance priority]

/-- A normed ring is a seminormed ring. -/
instance (priority := 100) NormedRing.toSeminormedRing [β : NormedRing α] : SeminormedRing α :=
  { β with }

-- see Note [lower instance priority]

/-- A normed ring is a non-unital normed ring. -/
instance (priority := 100) NormedRing.toNonUnitalNormedRing [β : NormedRing α] :
    NonUnitalNormedRing α :=
  { β with }


/-- A non-unital seminormed commutative ring is a non-unital commutative ring endowed with a
seminorm which satisfies the inequality `‖x y‖ ≤ ‖x‖ ‖y‖`. -/
class NonUnitalSeminormedCommRing (α : Type*) extends NonUnitalSeminormedRing α where
  /-- Multiplication is commutative. -/
  mul_comm : ∀ x y : α, x * y = y * x


/-- A non-unital normed commutative ring is a non-unital commutative ring endowed with a
norm which satisfies the inequality `‖x y‖ ≤ ‖x‖ ‖y‖`. -/
class NonUnitalNormedCommRing (α : Type*) extends NonUnitalNormedRing α where
  /-- Multiplication is commutative. -/
  mul_comm : ∀ x y : α, x * y = y * x

-- see Note [lower instance priority]

/-- A non-unital normed commutative ring is a non-unital seminormed commutative ring. -/
instance (priority := 100) NonUnitalNormedCommRing.toNonUnitalSeminormedCommRing
    [β : NonUnitalNormedCommRing α] : NonUnitalSeminormedCommRing α :=
  { β with }


/-- A seminormed commutative ring is a commutative ring endowed with a seminorm which satisfies
the inequality `‖x y‖ ≤ ‖x‖ ‖y‖`. -/
class SeminormedCommRing (α : Type*) extends SeminormedRing α where
  /-- Multiplication is commutative. -/
  mul_comm : ∀ x y : α, x * y = y * x


/-- A normed commutative ring is a commutative ring endowed with a norm which satisfies
the inequality `‖x y‖ ≤ ‖x‖ ‖y‖`. -/
class NormedCommRing (α : Type*) extends NormedRing α where
  /-- Multiplication is commutative. -/
  mul_comm : ∀ x y : α, x * y = y * x

-- see Note [lower instance priority]

/-- A seminormed commutative ring is a non-unital seminormed commutative ring. -/
instance (priority := 100) SeminormedCommRing.toNonUnitalSeminormedCommRing
    [β : SeminormedCommRing α] : NonUnitalSeminormedCommRing α :=
  { β with }

-- see Note [lower instance priority]

/-- A normed commutative ring is a non-unital normed commutative ring. -/
instance (priority := 100) NormedCommRing.toNonUnitalNormedCommRing
    [β : NormedCommRing α] : NonUnitalNormedCommRing α :=
  { β with }

-- see Note [lower instance priority]

/-- A normed commutative ring is a seminormed commutative ring. -/
instance (priority := 100) NormedCommRing.toSeminormedCommRing [β : NormedCommRing α] :
    SeminormedCommRing α :=
  { β with }


instance PUnit.normedCommRing : NormedCommRing PUnit :=
  { PUnit.normedAddCommGroup, PUnit.commRing with
                              /-
                                α : Type u_1
                                β : Type u_2
                                ι : Type u_3
                                x✝¹ x✝ : PUnit.{?u.5240 + 1}
                                ⊢ LE.le (Norm.norm (HMul.hMul x✝¹ x✝)) (HMul.hMul (Norm.norm x✝¹) (Norm.norm x …
                              -/
    norm_mul := fun _ _ => by simp }
                              /-
                                🎉 no goals
                              -/


/-- A mixin class with the axiom `‖1‖ = 1`. Many `NormedRing`s and all `NormedField`s satisfy this
axiom. -/
class NormOneClass (α : Type*) [Norm α] [One α] : Prop where
  /-- The norm of the multiplicative identity is 1. -/
  norm_one : ‖(1 : α)‖ = 1


@[simp]
theorem nnnorm_one [SeminormedAddCommGroup α] [One α] [NormOneClass α] : ‖(1 : α)‖₊ = 1 :=
  NNReal.eq norm_one


theorem NormOneClass.nontrivial (α : Type*) [SeminormedAddCommGroup α] [One α] [NormOneClass α] :
    Nontrivial α :=
                                                    /-
                                                      α : Type u_4
                                                      inst✝² : SeminormedAddCommGroup α
                                                      inst✝¹ : One α
                                                      inst✝ : NormOneClass α
                                                      ⊢ Ne (Norm.norm 0) (Norm.norm 1)
                                                    -/
  nontrivial_of_ne 0 1 <| ne_of_apply_ne norm <| by simp
                                                    /-
                                                      🎉 no goals
                                                    -/

-- see Note [lower instance priority]

instance (priority := 100) NonUnitalSeminormedCommRing.toNonUnitalCommRing
    [β : NonUnitalSeminormedCommRing α] : NonUnitalCommRing α :=
  { β with }

-- see Note [lower instance priority]

instance (priority := 100) SeminormedCommRing.toCommRing [β : SeminormedCommRing α] : CommRing α :=
  { β with }

-- see Note [lower instance priority]

instance (priority := 100) NonUnitalNormedRing.toNormedAddCommGroup [β : NonUnitalNormedRing α] :
    NormedAddCommGroup α :=
  { β with }

-- see Note [lower instance priority]

instance (priority := 100) NonUnitalSeminormedRing.toSeminormedAddCommGroup
    [NonUnitalSeminormedRing α] : SeminormedAddCommGroup α :=
  { ‹NonUnitalSeminormedRing α› with }


instance ULift.normOneClass [SeminormedAddCommGroup α] [One α] [NormOneClass α] :
    NormOneClass (ULift α) :=
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        inst✝² : SeminormedAddCommGroup α
        inst✝¹ : One α
        inst✝ : NormOneClass α
        ⊢ Eq (Norm.norm 1) 1
      -/
  ⟨by simp [ULift.norm_def]⟩
      /-
        🎉 no goals
      -/


instance Prod.normOneClass [SeminormedAddCommGroup α] [One α] [NormOneClass α]
    [SeminormedAddCommGroup β] [One β] [NormOneClass β] : NormOneClass (α × β) :=
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        inst✝⁵ : SeminormedAddCommGroup α
        inst✝⁴ : One α
        inst✝³ : NormOneClass α
        inst✝² : SeminormedAddCommGroup β
        inst✝¹ : One β
        inst✝ : NormOneClass β
        ⊢ Eq (Norm.norm 1) 1
      -/
  ⟨by simp [Prod.norm_def]⟩
      /-
        🎉 no goals
      -/


instance Pi.normOneClass {ι : Type*} {α : ι → Type*} [Nonempty ι] [Fintype ι]
    [∀ i, SeminormedAddCommGroup (α i)] [∀ i, One (α i)] [∀ i, NormOneClass (α i)] :
    NormOneClass (∀ i, α i) :=
      /-
        α✝ : Type u_1
        β : Type u_2
        ι✝ : Type u_3
        ι : Type u_4
        α : ι → Type u_5
        inst✝⁴ : Nonempty ι
        inst✝³ : Fintype ι
        inst✝² : (i : ι) → SeminormedAddCommGroup (α i)
        inst✝¹ : (i : ι) → One (α i)
        inst✝ : ∀ (i : ι), NormOneClass (α i)
        ⊢ Eq (Norm.norm 1) 1
      -/
  ⟨by simpa [Pi.norm_def] using Finset.sup_const Finset.univ_nonempty 1⟩
      /-
        🎉 no goals
      -/


instance MulOpposite.normOneClass [SeminormedAddCommGroup α] [One α] [NormOneClass α] :
    NormOneClass αᵐᵒᵖ :=
  ⟨@norm_one α _ _ _⟩


theorem norm_mul_le (a b : α) : ‖a * b‖ ≤ ‖a‖ * ‖b‖ :=
  NonUnitalSeminormedRing.norm_mul _ _


theorem nnnorm_mul_le (a b : α) : ‖a * b‖₊ ≤ ‖a‖₊ * ‖b‖₊ := by
  simpa only [← norm_toNNReal, ← Real.toNNReal_mul (norm_nonneg _)] using
    Real.toNNReal_mono (norm_mul_le _ _)


lemma norm_mul_le_of_le {r₁ r₂ : ℝ} (h₁ : ‖a₁‖ ≤ r₁) (h₂ : ‖a₂‖ ≤ r₂) : ‖a₁ * a₂‖ ≤ r₁ * r₂ :=
  (norm_mul_le ..).trans <| mul_le_mul h₁ h₂ (norm_nonneg _) ((norm_nonneg _).trans h₁)


lemma nnnorm_mul_le_of_le {r₁ r₂ : ℝ≥0} (h₁ : ‖a₁‖₊ ≤ r₁) (h₂ : ‖a₂‖₊ ≤ r₂) :
    ‖a₁ * a₂‖₊ ≤ r₁ * r₂ := (nnnorm_mul_le ..).trans <| mul_le_mul' h₁ h₂


lemma norm_mul₃_le : ‖a * b * c‖ ≤ ‖a‖ * ‖b‖ * ‖c‖ := norm_mul_le_of_le (norm_mul_le ..) le_rfl


lemma nnnorm_mul₃_le : ‖a * b * c‖₊ ≤ ‖a‖₊ * ‖b‖₊ * ‖c‖₊ :=
  nnnorm_mul_le_of_le (norm_mul_le ..) le_rfl


theorem one_le_norm_one (β) [NormedRing β] [Nontrivial β] : 1 ≤ ‖(1 : β)‖ :=
  (le_mul_iff_one_le_left <| norm_pos_iff.mpr (one_ne_zero : (1 : β) ≠ 0)).mp
        /-
          β : Type u_4
          inst✝¹ : NormedRing β
          inst✝ : Nontrivial β
          ⊢ LE.le (Norm.norm 1) (HMul.hMul (Norm.norm 1) (Norm.norm 1))
        -/
    (by simpa only [mul_one] using norm_mul_le (1 : β) 1)
        /-
          🎉 no goals
        -/


theorem one_le_nnnorm_one (β) [NormedRing β] [Nontrivial β] : 1 ≤ ‖(1 : β)‖₊ :=
  one_le_norm_one β


/-- In a seminormed ring, the left-multiplication `AddMonoidHom` is bounded. -/
theorem mulLeft_bound (x : α) : ∀ y : α, ‖AddMonoidHom.mulLeft x y‖ ≤ ‖x‖ * ‖y‖ :=
  norm_mul_le x


/-- In a seminormed ring, the right-multiplication `AddMonoidHom` is bounded. -/
theorem mulRight_bound (x : α) : ∀ y : α, ‖AddMonoidHom.mulRight x y‖ ≤ ‖x‖ * ‖y‖ := fun y => by
  /-
    α : Type u_1
    inst✝ : NonUnitalSeminormedRing α
    x y : α
    ⊢ LE.le (Norm.norm ((AddMonoidHom.mulRight x) y)) (HMul.hMul (Norm.norm x) (No …
  -/
  rw [mul_comm]
  /-
    α : Type u_1
    inst✝ : NonUnitalSeminormedRing α
    x y : α
    ⊢ LE.le (Norm.norm ((AddMonoidHom.mulRight x) y)) (HMul.hMul (Norm.norm y) (No …
  -/
  exact norm_mul_le y x
  /-
    🎉 no goals
  -/


/-- A non-unital subalgebra of a non-unital seminormed ring is also a non-unital seminormed ring,
with the restriction of the norm. -/
instance NonUnitalSubalgebra.nonUnitalSeminormedRing {𝕜 : Type*} [CommRing 𝕜] {E : Type*}
    [NonUnitalSeminormedRing E] [Module 𝕜 E] (s : NonUnitalSubalgebra 𝕜 E) :
    NonUnitalSeminormedRing s :=
  { s.toSubmodule.seminormedAddCommGroup, s.toNonUnitalRing with
    norm_mul := fun a b => norm_mul_le a.1 b.1 }


/-- A non-unital subalgebra of a non-unital seminormed ring is also a non-unital seminormed ring,
with the restriction of the norm. -/
-- necessary to require `SMulMemClass S 𝕜 E` so that `𝕜` can be determined as an `outParam`
@[nolint unusedArguments]
instance (priority := 75) NonUnitalSubalgebraClass.nonUnitalSeminormedRing {S 𝕜 E : Type*}
    [CommRing 𝕜] [NonUnitalSeminormedRing E] [Module 𝕜 E] [SetLike S E] [NonUnitalSubringClass S E]
    [SMulMemClass S 𝕜 E] (s : S) :
    NonUnitalSeminormedRing s :=
  { AddSubgroupClass.seminormedAddCommGroup s, NonUnitalSubringClass.toNonUnitalRing s with
    norm_mul := fun a b => norm_mul_le a.1 b.1 }


/-- A non-unital subalgebra of a non-unital normed ring is also a non-unital normed ring, with the
restriction of the norm. -/
instance NonUnitalSubalgebra.nonUnitalNormedRing {𝕜 : Type*} [CommRing 𝕜] {E : Type*}
    [NonUnitalNormedRing E] [Module 𝕜 E] (s : NonUnitalSubalgebra 𝕜 E) : NonUnitalNormedRing s :=
  { s.nonUnitalSeminormedRing with
    eq_of_dist_eq_zero := eq_of_dist_eq_zero }


/-- A non-unital subalgebra of a non-unital normed ring is also a non-unital normed ring,
with the restriction of the norm. -/
instance (priority := 75) NonUnitalSubalgebraClass.nonUnitalNormedRing {S 𝕜 E : Type*}
    [CommRing 𝕜] [NonUnitalNormedRing E] [Module 𝕜 E] [SetLike S E] [NonUnitalSubringClass S E]
    [SMulMemClass S 𝕜 E] (s : S) :
    NonUnitalNormedRing s :=
  { nonUnitalSeminormedRing s with
    eq_of_dist_eq_zero := eq_of_dist_eq_zero }


instance ULift.nonUnitalSeminormedRing : NonUnitalSeminormedRing (ULift α) :=
  { ULift.seminormedAddCommGroup, ULift.nonUnitalRing with
    norm_mul := fun x y => (norm_mul_le x.down y.down : _) }


/-- Non-unital seminormed ring structure on the product of two non-unital seminormed rings,
  using the sup norm. -/
instance Prod.nonUnitalSeminormedRing [NonUnitalSeminormedRing β] :
    NonUnitalSeminormedRing (α × β) :=
  { seminormedAddCommGroup, instNonUnitalRing with
    norm_mul := fun x y =>
      calc
        ‖x * y‖ = ‖(x.1 * y.1, x.2 * y.2)‖ := rfl
        _ = max ‖x.1 * y.1‖ ‖x.2 * y.2‖ := rfl
        _ ≤ max (‖x.1‖ * ‖y.1‖) (‖x.2‖ * ‖y.2‖) :=
          (max_le_max (norm_mul_le x.1 y.1) (norm_mul_le x.2 y.2))
                                                      /-
                                                        α : Type u_1
                                                        β : Type u_2
                                                        ι : Type u_3
                                                        inst✝¹ : NonUnitalSeminormedRing α
                                                        a a₁ a₂ b c : α
                                                        inst✝ : NonUnitalSeminormedRing β
                                                        x y : Prod α β
                                                        ⊢ Eq (Max.max (HMul.hMul (Norm.norm x.1) (Norm.norm y.1)) (HMul.hMul (Norm.nor …
                                                      -/
        _ = max (‖x.1‖ * ‖y.1‖) (‖y.2‖ * ‖x.2‖) := by simp [mul_comm]
                                                      /-
                                                        🎉 no goals
                                                      -/
        _ ≤ max ‖x.1‖ ‖x.2‖ * max ‖y.2‖ ‖y.1‖ := by
          /-
            α : Type u_1
            β : Type u_2
            ι : Type u_3
            inst✝¹ : NonUnitalSeminormedRing α
            a a₁ a₂ b c : α
            inst✝ : NonUnitalSeminormedRing β
            x y : Prod α β
            ⊢ LE.le (Max.max (HMul.hMul (Norm.norm x.1) (Norm.norm y.1)) (HMul.hMul (Norm. …
          -/
                                               /-
                                                 🎉 no goals
                                               -/
          apply max_mul_mul_le_max_mul_max <;> simp [norm_nonneg]
                                               /-
                                                 🎉 no goals
                                               -/
                                                    /-
                                                      α : Type u_1
                                                      β : Type u_2
                                                      ι : Type u_3
                                                      inst✝¹ : NonUnitalSeminormedRing α
                                                      a a₁ a₂ b c : α
                                                      inst✝ : NonUnitalSeminormedRing β
                                                      x y : Prod α β
                                                      ⊢ Eq (HMul.hMul (Max.max (Norm.norm x.1) (Norm.norm x.2)) (Max.max (Norm.norm  …
                                                    -/
        _ = max ‖x.1‖ ‖x.2‖ * max ‖y.1‖ ‖y.2‖ := by simp [max_comm]
                                                    /-
                                                      🎉 no goals
                                                    -/
        _ = ‖x‖ * ‖y‖ := rfl
         }


instance MulOpposite.instNonUnitalSeminormedRing : NonUnitalSeminormedRing αᵐᵒᵖ where
  __ := instNonUnitalRing
  __ := instSeminormedAddCommGroup
  norm_mul := MulOpposite.rec' fun x ↦ MulOpposite.rec' fun y ↦
    (norm_mul_le y x).trans_eq (mul_comm _ _)


/-- A subalgebra of a seminormed ring is also a seminormed ring, with the restriction of the
norm. -/
instance Subalgebra.seminormedRing {𝕜 : Type*} [CommRing 𝕜] {E : Type*} [SeminormedRing E]
    [Algebra 𝕜 E] (s : Subalgebra 𝕜 E) : SeminormedRing s :=
  { s.toSubmodule.seminormedAddCommGroup, s.toRing with
    norm_mul := fun a b => norm_mul_le a.1 b.1 }


/-- A subalgebra of a seminormed ring is also a seminormed ring, with the restriction of the
norm. -/
-- necessary to require `SMulMemClass S 𝕜 E` so that `𝕜` can be determined as an `outParam`
@[nolint unusedArguments]
instance (priority := 75) SubalgebraClass.seminormedRing {S 𝕜 E : Type*} [CommRing 𝕜]
    [SeminormedRing E] [Algebra 𝕜 E] [SetLike S E] [SubringClass S E] [SMulMemClass S 𝕜 E]
    (s : S) : SeminormedRing s :=
  { AddSubgroupClass.seminormedAddCommGroup s, SubringClass.toRing s with
    norm_mul := fun a b => norm_mul_le a.1 b.1 }


/-- A subalgebra of a normed ring is also a normed ring, with the restriction of the norm. -/
instance Subalgebra.normedRing {𝕜 : Type*} [CommRing 𝕜] {E : Type*} [NormedRing E]
    [Algebra 𝕜 E] (s : Subalgebra 𝕜 E) : NormedRing s :=
  { s.seminormedRing with
    eq_of_dist_eq_zero := eq_of_dist_eq_zero }


/-- A subalgebra of a normed ring is also a normed ring, with the restriction of the
norm. -/
instance (priority := 75) SubalgebraClass.normedRing {S 𝕜 E : Type*} [CommRing 𝕜]
    [NormedRing E] [Algebra 𝕜 E] [SetLike S E] [SubringClass S E] [SMulMemClass S 𝕜 E]
    (s : S) : NormedRing s :=
  { seminormedRing s with
    eq_of_dist_eq_zero := eq_of_dist_eq_zero }



theorem Nat.norm_cast_le : ∀ n : ℕ, ‖(n : α)‖ ≤ n * ‖(1 : α)‖
            /-
              α : Type u_1
              inst✝ : SeminormedRing α
              ⊢ LE.le (Norm.norm ↑0) (HMul.hMul (↑0) (Norm.norm 1))
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
  | n + 1 => by
    /-
      α : Type u_1
      inst✝ : SeminormedRing α
      n : Nat
      ⊢ LE.le (Norm.norm ↑(HAdd.hAdd n 1)) (HMul.hMul (↑(HAdd.hAdd n 1)) (Norm.norm  …
    -/
    rw [n.cast_succ, n.cast_succ, add_mul, one_mul]
    /-
      α : Type u_1
      inst✝ : SeminormedRing α
      n : Nat
      ⊢ LE.le (Norm.norm (HAdd.hAdd (↑n) 1)) (HAdd.hAdd (HMul.hMul (↑n) (Norm.norm 1 …
    -/
    exact norm_add_le_of_le (Nat.norm_cast_le n) le_rfl
    /-
      🎉 no goals
    -/


theorem List.norm_prod_le' : ∀ {l : List α}, l ≠ [] → ‖l.prod‖ ≤ (l.map norm).prod
  | [], h => (h rfl).elim
                 /-
                   α : Type u_1
                   inst✝ : SeminormedRing α
                   a : α
                   x✝ : Ne (List.cons a List.nil) List.nil
                   ⊢ LE.le (Norm.norm (List.cons a List.nil).prod) (List.map Norm.norm (List.cons …
                 -/
  | [a], _ => by simp
                 /-
                   🎉 no goals
                 -/
  | a::b::l, _ => by
    /-
      α : Type u_1
      inst✝ : SeminormedRing α
      a b : α
      l : List α
      x✝ : Ne (List.cons a (List.cons b l)) List.nil
      ⊢ LE.le (Norm.norm (List.cons a (List.cons b l)).prod) (List.map Norm.norm (Li …
    -/
    rw [List.map_cons, List.prod_cons, List.prod_cons (a := ‖a‖)]
    /-
      α : Type u_1
      inst✝ : SeminormedRing α
      a b : α
      l : List α
      x✝ : Ne (List.cons a (List.cons b l)) List.nil
      ⊢ LE.le (Norm.norm (HMul.hMul a (List.cons b l).prod)) (HMul.hMul (Norm.norm a …
    -/
    refine le_trans (norm_mul_le _ _) (mul_le_mul_of_nonneg_left ?_ (norm_nonneg _))
    /-
      α : Type u_1
      inst✝ : SeminormedRing α
      a b : α
      l : List α
      x✝ : Ne (List.cons a (List.cons b l)) List.nil
      ⊢ LE.le (Norm.norm (List.cons b l).prod) (List.map Norm.norm (List.cons b l)). …
    -/
    exact List.norm_prod_le' (List.cons_ne_nil b l)
    /-
      🎉 no goals
    -/


theorem List.nnnorm_prod_le' {l : List α} (hl : l ≠ []) : ‖l.prod‖₊ ≤ (l.map nnnorm).prod :=
                                         /-
                                           α : Type u_1
                                           inst✝ : SeminormedRing α
                                           l : List α
                                           hl : Ne l List.nil
                                           ⊢ Eq (List.map Norm.norm l).prod ((fun a => ↑a) (List.map NNNorm.nnnorm l).prod)
                                         -/
  (List.norm_prod_le' hl).trans_eq <| by simp [NNReal.coe_list_prod, List.map_map]
                                         /-
                                           🎉 no goals
                                         -/


theorem List.norm_prod_le [NormOneClass α] : ∀ l : List α, ‖l.prod‖ ≤ (l.map norm).prod
             /-
               α : Type u_1
               inst✝¹ : SeminormedRing α
               inst✝ : NormOneClass α
               ⊢ LE.le (Norm.norm List.nil.prod) (List.map Norm.norm List.nil).prod
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
  | a::l => List.norm_prod_le' (List.cons_ne_nil a l)


theorem List.nnnorm_prod_le [NormOneClass α] (l : List α) : ‖l.prod‖₊ ≤ (l.map nnnorm).prod :=
                                /-
                                  α : Type u_1
                                  inst✝¹ : SeminormedRing α
                                  inst✝ : NormOneClass α
                                  l : List α
                                  ⊢ Eq (List.map Norm.norm l).prod ((fun a => ↑a) (List.map NNNorm.nnnorm l).prod)
                                -/
  l.norm_prod_le.trans_eq <| by simp [NNReal.coe_list_prod, List.map_map]
                                /-
                                  🎉 no goals
                                -/


theorem Finset.norm_prod_le' {α : Type*} [NormedCommRing α] (s : Finset ι) (hs : s.Nonempty)
    (f : ι → α) : ‖∏ i ∈ s, f i‖ ≤ ∏ i ∈ s, ‖f i‖ := by
  /-
    ι : Type u_3
    α : Type u_4
    inst✝ : NormedCommRing α
    s : Finset ι
    hs : s.Nonempty
    f : ι → α
    ⊢ LE.le (Norm.norm (s.prod fun i => f i)) (s.prod fun i => Norm.norm (f i))
  -/
  rcases s with ⟨⟨l⟩, hl⟩
  /-
    case mk.mk
    ι : Type u_3
    α : Type u_4
    inst✝ : NormedCommRing α
    f : ι → α
    val✝ : Multiset ι
    l : List ι
    hl : Multiset.Nodup (Quot.mk (⇑(List.isSetoid ι)) l)
    hs : { val := Quot.mk (⇑(List.isSetoid ι)) l, nodup := hl }.Nonempty
    ⊢ LE.le (Norm.norm ({ val := Quot.mk (⇑(List.isSetoid ι)) l, nodup := hl }.pro …
  -/
  have : l.map f ≠ [] := by simpa using hs
  /-
    case mk.mk
    ι : Type u_3
    α : Type u_4
    inst✝ : NormedCommRing α
    f : ι → α
    val✝ : Multiset ι
    l : List ι
    hl : Multiset.Nodup (Quot.mk (⇑(List.isSetoid ι)) l)
    hs : { val := Quot.mk (⇑(List.isSetoid ι)) l, nodup := hl }.Nonempty
    this : Ne (List.map f l) List.nil
    ⊢ LE.le (Norm.norm ({ val := Quot.mk (⇑(List.isSetoid ι)) l, nodup := hl }.pro …
  -/
  simpa using List.norm_prod_le' this
  /-
    🎉 no goals
  -/


theorem Finset.nnnorm_prod_le' {α : Type*} [NormedCommRing α] (s : Finset ι) (hs : s.Nonempty)
    (f : ι → α) : ‖∏ i ∈ s, f i‖₊ ≤ ∏ i ∈ s, ‖f i‖₊ :=
                                        /-
                                          ι : Type u_3
                                          α : Type u_4
                                          inst✝ : NormedCommRing α
                                          s : Finset ι
                                          hs : s.Nonempty
                                          f : ι → α
                                          ⊢ Eq (s.prod fun i => Norm.norm (f i)) ((fun a => ↑a) (s.prod fun i => NNNorm. …
                                        -/
  (s.norm_prod_le' hs f).trans_eq <| by simp [NNReal.coe_prod]
                                        /-
                                          🎉 no goals
                                        -/


theorem Finset.norm_prod_le {α : Type*} [NormedCommRing α] [NormOneClass α] (s : Finset ι)
    (f : ι → α) : ‖∏ i ∈ s, f i‖ ≤ ∏ i ∈ s, ‖f i‖ := by
  /-
    ι : Type u_3
    α : Type u_4
    inst✝¹ : NormedCommRing α
    inst✝ : NormOneClass α
    s : Finset ι
    f : ι → α
    ⊢ LE.le (Norm.norm (s.prod fun i => f i)) (s.prod fun i => Norm.norm (f i))
  -/
  rcases s with ⟨⟨l⟩, hl⟩
  /-
    case mk.mk
    ι : Type u_3
    α : Type u_4
    inst✝¹ : NormedCommRing α
    inst✝ : NormOneClass α
    f : ι → α
    val✝ : Multiset ι
    l : List ι
    hl : Multiset.Nodup (Quot.mk (⇑(List.isSetoid ι)) l)
    ⊢ LE.le (Norm.norm ({ val := Quot.mk (⇑(List.isSetoid ι)) l, nodup := hl }.pro …
  -/
  simpa using (l.map f).norm_prod_le
  /-
    🎉 no goals
  -/


theorem Finset.nnnorm_prod_le {α : Type*} [NormedCommRing α] [NormOneClass α] (s : Finset ι)
    (f : ι → α) : ‖∏ i ∈ s, f i‖₊ ≤ ∏ i ∈ s, ‖f i‖₊ :=
                                    /-
                                      ι : Type u_3
                                      α : Type u_4
                                      inst✝¹ : NormedCommRing α
                                      inst✝ : NormOneClass α
                                      s : Finset ι
                                      f : ι → α
                                      ⊢ Eq (s.prod fun i => Norm.norm (f i)) ((fun a => ↑a) (s.prod fun i => NNNorm. …
                                    -/
  (s.norm_prod_le f).trans_eq <| by simp [NNReal.coe_prod]
                                    /-
                                      🎉 no goals
                                    -/


/-- If `α` is a seminormed ring, then `‖a ^ n‖₊ ≤ ‖a‖₊ ^ n` for `n > 0`.
See also `nnnorm_pow_le`. -/
theorem nnnorm_pow_le' (a : α) : ∀ {n : ℕ}, 0 < n → ‖a ^ n‖₊ ≤ ‖a‖₊ ^ n
               /-
                 α : Type u_1
                 inst✝ : SeminormedRing α
                 a : α
                 x✝ : LT.lt 0 1
                 ⊢ LE.le (NNNorm.nnnorm (HPow.hPow a 1)) (HPow.hPow (NNNorm.nnnorm a) 1)
               -/
  | 1, _ => by simp only [pow_one, le_rfl]
               /-
                 🎉 no goals
               -/
  | n + 2, _ => by
    simpa only [pow_succ' _ (n + 1)] using
      le_trans (nnnorm_mul_le _ _) (mul_le_mul_left' (nnnorm_pow_le' a n.succ_pos) _)


/-- If `α` is a seminormed ring with `‖1‖₊ = 1`, then `‖a ^ n‖₊ ≤ ‖a‖₊ ^ n`.
See also `nnnorm_pow_le'`. -/
theorem nnnorm_pow_le [NormOneClass α] (a : α) (n : ℕ) : ‖a ^ n‖₊ ≤ ‖a‖₊ ^ n :=
                  /-
                    α : Type u_1
                    inst✝¹ : SeminormedRing α
                    inst✝ : NormOneClass α
                    a : α
                    n : Nat
                    ⊢ LE.le (NNNorm.nnnorm (HPow.hPow a Nat.zero)) (HPow.hPow (NNNorm.nnnorm a) Na …
                  -/
  Nat.recOn n (by simp only [pow_zero, nnnorm_one, le_rfl])
                  /-
                    🎉 no goals
                  -/
    fun k _hk => nnnorm_pow_le' a k.succ_pos


/-- If `α` is a seminormed ring, then `‖a ^ n‖ ≤ ‖a‖ ^ n` for `n > 0`. See also `norm_pow_le`. -/
theorem norm_pow_le' (a : α) {n : ℕ} (h : 0 < n) : ‖a ^ n‖ ≤ ‖a‖ ^ n := by
  /-
    α : Type u_1
    inst✝ : SeminormedRing α
    a : α
    n : Nat
    h : LT.lt 0 n
    ⊢ LE.le (Norm.norm (HPow.hPow a n)) (HPow.hPow (Norm.norm a) n)
  -/
  simpa only [NNReal.coe_pow, coe_nnnorm] using NNReal.coe_mono (nnnorm_pow_le' a h)
  /-
    🎉 no goals
  -/


/-- If `α` is a seminormed ring with `‖1‖ = 1`, then `‖a ^ n‖ ≤ ‖a‖ ^ n`.
See also `norm_pow_le'`. -/
theorem norm_pow_le [NormOneClass α] (a : α) (n : ℕ) : ‖a ^ n‖ ≤ ‖a‖ ^ n :=
                  /-
                    α : Type u_1
                    inst✝¹ : SeminormedRing α
                    inst✝ : NormOneClass α
                    a : α
                    n : Nat
                    ⊢ LE.le (Norm.norm (HPow.hPow a Nat.zero)) (HPow.hPow (Norm.norm a) Nat.zero)
                  -/
  Nat.recOn n (by simp only [pow_zero, norm_one, le_rfl])
                  /-
                    🎉 no goals
                  -/
    fun n _hn => norm_pow_le' a n.succ_pos


theorem eventually_norm_pow_le (a : α) : ∀ᶠ n : ℕ in atTop, ‖a ^ n‖ ≤ ‖a‖ ^ n :=
  eventually_atTop.mpr ⟨1, fun _b h => norm_pow_le' a (Nat.succ_le_iff.mp h)⟩


instance ULift.seminormedRing : SeminormedRing (ULift α) :=
  { ULift.nonUnitalSeminormedRing, ULift.ring with }


/-- Seminormed ring structure on the product of two seminormed rings,
  using the sup norm. -/
instance Prod.seminormedRing [SeminormedRing β] : SeminormedRing (α × β) :=
  { nonUnitalSeminormedRing, instRing with }


instance MulOpposite.instSeminormedRing : SeminormedRing αᵐᵒᵖ where
  __ := instRing
  __ := instNonUnitalSeminormedRing


/-- This inequality is particularly useful when `c = 1` and `‖a‖ = ‖b‖ = 1` as it then shows that
chord length is a metric on the unit complex numbers. -/
lemma norm_sub_mul_le (ha : ‖a‖ ≤ 1) : ‖c - a * b‖ ≤ ‖c - a‖ + ‖1 - b‖ :=
  calc
    _ ≤ ‖c - a‖ + ‖a * (1 - b)‖ := by
        /-
          α : Type u_1
          inst✝ : SeminormedRing α
          a b c : α
          ha : LE.le (Norm.norm a) 1
          ⊢ LE.le (Norm.norm (HSub.hSub c (HMul.hMul a b))) (HAdd.hAdd (Norm.norm (HSub. …
        -/
        simpa [mul_one_sub] using norm_sub_le_norm_sub_add_norm_sub c a (a * b)
        /-
          🎉 no goals
        -/
                                      /-
                                        α : Type u_1
                                        inst✝ : SeminormedRing α
                                        a b c : α
                                        ha : LE.le (Norm.norm a) 1
                                        ⊢ LE.le (HAdd.hAdd (Norm.norm (HSub.hSub c a)) (Norm.norm (HMul.hMul a (HSub.h …
                                      -/
    _ ≤ ‖c - a‖ + ‖a‖ * ‖1 - b‖ := by gcongr; exact norm_mul_le ..
                                              /-
                                                🎉 no goals
                                              -/
                                    /-
                                      α : Type u_1
                                      inst✝ : SeminormedRing α
                                      a b c : α
                                      ha : LE.le (Norm.norm a) 1
                                      ⊢ LE.le (HAdd.hAdd (Norm.norm (HSub.hSub c a)) (HMul.hMul (Norm.norm a) (Norm. …
                                    -/
    _ ≤ ‖c - a‖ + 1 * ‖1 - b‖ := by gcongr
                                    /-
                                      🎉 no goals
                                    -/
                                /-
                                  α : Type u_1
                                  inst✝ : SeminormedRing α
                                  a b c : α
                                  ha : LE.le (Norm.norm a) 1
                                  ⊢ Eq (HAdd.hAdd (Norm.norm (HSub.hSub c a)) (HMul.hMul 1 (Norm.norm (HSub.hSub …
                                -/
    _ = ‖c - a‖ + ‖1 - b‖ := by simp
                                /-
                                  🎉 no goals
                                -/


/-- This inequality is particularly useful when `c = 1` and `‖a‖ = ‖b‖ = 1` as it then shows that
chord length is a metric on the unit complex numbers. -/
lemma norm_sub_mul_le' (hb : ‖b‖ ≤ 1) : ‖c - a * b‖ ≤ ‖1 - a‖ + ‖c - b‖ := by
  /-
    α : Type u_1
    inst✝ : SeminormedRing α
    a b c : α
    hb : LE.le (Norm.norm b) 1
    ⊢ LE.le (Norm.norm (HSub.hSub c (HMul.hMul a b))) (HAdd.hAdd (Norm.norm (HSub. …
  -/
  rw [add_comm]; exact norm_sub_mul_le (α := αᵐᵒᵖ) hb
                 /-
                   🎉 no goals
                 -/


/-- This inequality is particularly useful when `c = 1` and `‖a‖ = ‖b‖ = 1` as it then shows that
chord length is a metric on the unit complex numbers. -/
lemma nnnorm_sub_mul_le (ha : ‖a‖₊ ≤ 1) : ‖c - a * b‖₊ ≤ ‖c - a‖₊ + ‖1 - b‖₊ := norm_sub_mul_le ha


/-- This inequality is particularly useful when `c = 1` and `‖a‖ = ‖b‖ = 1` as it then shows that
chord length is a metric on the unit complex numbers. -/
lemma nnnorm_sub_mul_le' (hb : ‖b‖₊ ≤ 1) : ‖c - a * b‖₊ ≤ ‖1 - a‖₊ + ‖c - b‖₊ := norm_sub_mul_le' hb


lemma norm_commutator_units_sub_one_le (a b : αˣ) :
    ‖(a * b * a⁻¹ * b⁻¹).val - 1‖ ≤ 2 * ‖a⁻¹.val‖ * ‖b⁻¹.val‖ * ‖a.val - 1‖ * ‖b.val - 1‖ :=
  calc
                                                                                /-
                                                                                  α : Type u_1
                                                                                  inst✝ : SeminormedRing α
                                                                                  a b : Units α
                                                                                  ⊢ Eq (Norm.norm (HSub.hSub (↑(HMul.hMul (HMul.hMul (HMul.hMul a b) (Inv.inv a) …
                                                                                -/
    ‖(a * b * a⁻¹ * b⁻¹).val - 1‖ = ‖(a * b - b * a) * a⁻¹.val * b⁻¹.val‖ := by simp [sub_mul, *]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
    _ ≤ ‖(a * b - b * a : α)‖ * ‖a⁻¹.val‖ * ‖b⁻¹.val‖ := norm_mul₃_le
    _ = ‖(a - 1 : α) * (b - 1) - (b - 1) * (a - 1)‖ * ‖a⁻¹.val‖ * ‖b⁻¹.val‖ := by
      /-
        α : Type u_1
        inst✝ : SeminormedRing α
        a b : Units α
        ⊢ Eq (HMul.hMul (HMul.hMul (Norm.norm (HSub.hSub (HMul.hMul ↑a ↑b) (HMul.hMul  …
      -/
      simp_rw [sub_one_mul, mul_sub_one]; abel_nf
                                          /-
                                            🎉 no goals
                                          -/
    _ ≤ (‖(a - 1 : α) * (b - 1)‖ + ‖(b - 1 : α) * (a - 1)‖) * ‖a⁻¹.val‖ * ‖b⁻¹.val‖ := by
      /-
        α : Type u_1
        inst✝ : SeminormedRing α
        a b : Units α
        ⊢ LE.le (HMul.hMul (HMul.hMul (Norm.norm (HSub.hSub (HMul.hMul (HSub.hSub (↑a) …
      -/
      gcongr; exact norm_sub_le ..
              /-
                🎉 no goals
              -/
    _ ≤ (‖a.val - 1‖ * ‖b.val - 1‖ + ‖b.val - 1‖ * ‖a.val - 1‖) * ‖a⁻¹.val‖ * ‖b⁻¹.val‖ := by
      /-
        α : Type u_1
        inst✝ : SeminormedRing α
        a b : Units α
        ⊢ LE.le (HMul.hMul (HMul.hMul (HAdd.hAdd (Norm.norm (HMul.hMul (HSub.hSub (↑a) …
      -/
                 /-
                   🎉 no goals
                 -/
      gcongr <;> exact norm_mul_le ..
                 /-
                   🎉 no goals
                 -/
                                                                    /-
                                                                      α : Type u_1
                                                                      inst✝ : SeminormedRing α
                                                                      a b : Units α
                                                                      ⊢ Eq (HMul.hMul (HMul.hMul (HAdd.hAdd (HMul.hMul (Norm.norm (HSub.hSub (↑a) 1) …
                                                                    -/
    _ = 2 * ‖a⁻¹.val‖ * ‖b⁻¹.val‖ * ‖a.val - 1‖ * ‖b.val - 1‖ := by ring
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


lemma nnnorm_commutator_units_sub_one_le (a b : αˣ) :
    ‖(a * b * a⁻¹ * b⁻¹).val - 1‖₊ ≤ 2 * ‖a⁻¹.val‖₊ * ‖b⁻¹.val‖₊ * ‖a.val - 1‖₊ * ‖b.val - 1‖₊ := by
  /-
    α : Type u_1
    inst✝ : SeminormedRing α
    a b : Units α
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub (↑(HMul.hMul (HMul.hMul (HMul.hMul a b) (Inv …
  -/
  simpa using norm_commutator_units_sub_one_le a b
  /-
    🎉 no goals
  -/


/-- A homomorphism `f` between semi_normed_rings is bounded if there exists a positive
  constant `C` such that for all `x` in `α`, `norm (f x) ≤ C * norm x`. -/
def RingHom.IsBounded {α : Type*} [SeminormedRing α] {β : Type*} [SeminormedRing β]
    (f : α →+* β) : Prop :=
  ∃ C : ℝ, 0 < C ∧ ∀ x : α, norm (f x) ≤ C * norm x


instance ULift.nonUnitalNormedRing : NonUnitalNormedRing (ULift α) :=
  { ULift.nonUnitalSeminormedRing, ULift.normedAddCommGroup with }


/-- Non-unital normed ring structure on the product of two non-unital normed rings,
using the sup norm. -/
instance Prod.nonUnitalNormedRing [NonUnitalNormedRing β] : NonUnitalNormedRing (α × β) :=
  { Prod.nonUnitalSeminormedRing, Prod.normedAddCommGroup with }


instance MulOpposite.instNonUnitalNormedRing : NonUnitalNormedRing αᵐᵒᵖ where
  __ := instNonUnitalRing
  __ := instNonUnitalSeminormedRing
  __ := instNormedAddCommGroup


theorem Units.norm_pos [Nontrivial α] (x : αˣ) : 0 < ‖(x : α)‖ :=
  norm_pos_iff.mpr (Units.ne_zero x)


theorem Units.nnnorm_pos [Nontrivial α] (x : αˣ) : 0 < ‖(x : α)‖₊ :=
  x.norm_pos


instance ULift.normedRing : NormedRing (ULift α) :=
  { ULift.seminormedRing, ULift.normedAddCommGroup with }


/-- Normed ring structure on the product of two normed rings, using the sup norm. -/
instance Prod.normedRing [NormedRing β] : NormedRing (α × β) :=
  { nonUnitalNormedRing, instRing with }


instance MulOpposite.instNormedRing : NormedRing αᵐᵒᵖ where
  __ := instRing
  __ := instSeminormedRing
  __ := instNormedAddCommGroup


instance ULift.nonUnitalSeminormedCommRing : NonUnitalSeminormedCommRing (ULift α) :=
  { ULift.nonUnitalSeminormedRing, ULift.nonUnitalCommRing with }


/-- Non-unital seminormed commutative ring structure on the product of two non-unital seminormed
commutative rings, using the sup norm. -/
instance Prod.nonUnitalSeminormedCommRing [NonUnitalSeminormedCommRing β] :
    NonUnitalSeminormedCommRing (α × β) :=
  { nonUnitalSeminormedRing, instNonUnitalCommRing with }


instance MulOpposite.instNonUnitalSeminormedCommRing : NonUnitalSeminormedCommRing αᵐᵒᵖ where
  __ := instNonUnitalSeminormedRing
  __ := instNonUnitalCommRing


/-- A non-unital subalgebra of a non-unital seminormed commutative ring is also a non-unital
seminormed commutative ring, with the restriction of the norm. -/
instance NonUnitalSubalgebra.nonUnitalSeminormedCommRing {𝕜 : Type*} [CommRing 𝕜] {E : Type*}
    [NonUnitalSeminormedCommRing E] [Module 𝕜 E] (s : NonUnitalSubalgebra 𝕜 E) :
    NonUnitalSeminormedCommRing s :=
  { s.nonUnitalSeminormedRing, s.toNonUnitalCommRing with }


/-- A non-unital subalgebra of a non-unital normed commutative ring is also a non-unital normed
commutative ring, with the restriction of the norm. -/
instance NonUnitalSubalgebra.nonUnitalNormedCommRing {𝕜 : Type*} [CommRing 𝕜] {E : Type*}
    [NonUnitalNormedCommRing E] [Module 𝕜 E] (s : NonUnitalSubalgebra 𝕜 E) :
    NonUnitalNormedCommRing s :=
  { s.nonUnitalSeminormedCommRing, s.nonUnitalNormedRing with }


instance ULift.nonUnitalNormedCommRing : NonUnitalNormedCommRing (ULift α) :=
  { ULift.nonUnitalSeminormedCommRing, ULift.normedAddCommGroup with }


/-- Non-unital normed commutative ring structure on the product of two non-unital normed
commutative rings, using the sup norm. -/
instance Prod.nonUnitalNormedCommRing [NonUnitalNormedCommRing β] :
    NonUnitalNormedCommRing (α × β) :=
  { Prod.nonUnitalSeminormedCommRing, Prod.normedAddCommGroup with }


instance MulOpposite.instNonUnitalNormedCommRing : NonUnitalNormedCommRing αᵐᵒᵖ where
  __ := instNonUnitalNormedRing
  __ := instNonUnitalSeminormedCommRing


instance ULift.seminormedCommRing : SeminormedCommRing (ULift α) :=
  { ULift.nonUnitalSeminormedRing, ULift.commRing with }


/-- Seminormed commutative ring structure on the product of two seminormed commutative rings,
  using the sup norm. -/
instance Prod.seminormedCommRing [SeminormedCommRing β] : SeminormedCommRing (α × β) :=
  { Prod.nonUnitalSeminormedCommRing, instCommRing with }


instance MulOpposite.instSeminormedCommRing : SeminormedCommRing αᵐᵒᵖ where
  __ := instSeminormedRing
  __ := instNonUnitalSeminormedCommRing


/-- A subalgebra of a seminormed commutative ring is also a seminormed commutative ring, with the
restriction of the norm. -/
instance Subalgebra.seminormedCommRing {𝕜 : Type*} [CommRing 𝕜] {E : Type*} [SeminormedCommRing E]
    [Algebra 𝕜 E] (s : Subalgebra 𝕜 E) : SeminormedCommRing s :=
  { s.seminormedRing, s.toCommRing with }


/-- A subalgebra of a normed commutative ring is also a normed commutative ring, with the
restriction of the norm. -/
instance Subalgebra.normedCommRing {𝕜 : Type*} [CommRing 𝕜] {E : Type*} [NormedCommRing E]
    [Algebra 𝕜 E] (s : Subalgebra 𝕜 E) : NormedCommRing s :=
  { s.seminormedCommRing, s.normedRing with }


instance ULift.normedCommRing : NormedCommRing (ULift α) :=
  { ULift.normedRing (α := α), ULift.seminormedCommRing with }


/-- Normed commutative ring structure on the product of two normed commutative rings, using the sup
norm. -/
instance Prod.normedCommRing [NormedCommRing β] : NormedCommRing (α × β) :=
  { nonUnitalNormedRing, instCommRing with }


instance MulOpposite.instNormedCommRing : NormedCommRing αᵐᵒᵖ where
  __ := instNormedRing
  __ := instSeminormedCommRing


/-- The restriction of a power-multiplicative function to a subalgebra is power-multiplicative. -/
theorem IsPowMul.restriction {R S : Type*} [NormedCommRing R] [CommRing S] [Algebra R S]
    (A : Subalgebra R S) {f : S → ℝ} (hf_pm : IsPowMul f) :
    IsPowMul fun x : A => f x.val := fun x n hn => by
  /-
    R : Type u_4
    S : Type u_5
    inst✝² : NormedCommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    A : Subalgebra R S
    f : S → Real
    hf_pm : IsPowMul f
    x : Subtype fun x => Membership.mem A x
    n : Nat
    hn : LE.le 1 n
    ⊢ Eq ((fun x => f ↑x) (HPow.hPow x n)) (HPow.hPow ((fun x => f ↑x) x) n)
  -/
  simpa [SubsemiringClass.coe_pow] using hf_pm (↑x) hn
  /-
    🎉 no goals
  -/


@[simp]
theorem norm_mul (a b : α) : ‖a * b‖ = ‖a‖ * ‖b‖ :=
  NormedDivisionRing.norm_mul' a b


instance (priority := 900) NormedDivisionRing.to_normOneClass : NormOneClass α :=
                                                               /-
                                                                 α : Type u_1
                                                                 β : Type u_2
                                                                 ι : Type u_3
                                                                 inst✝ : NormedDivisionRing α
                                                                 a b : α
                                                                 ⊢ Eq (HMul.hMul (Norm.norm 1) (Norm.norm 1)) (HMul.hMul (Norm.norm 1) 1)
                                                               -/
  ⟨mul_left_cancel₀ (mt norm_eq_zero.1 (one_ne_zero' α)) <| by rw [← norm_mul, mul_one, mul_one]⟩
                                                               /-
                                                                 🎉 no goals
                                                               -/


instance isAbsoluteValue_norm : IsAbsoluteValue (norm : α → ℝ) where
  abv_nonneg' := norm_nonneg
  abv_eq_zero' := norm_eq_zero
  abv_add' := norm_add_le
  abv_mul' := norm_mul


@[simp]
theorem nnnorm_mul (a b : α) : ‖a * b‖₊ = ‖a‖₊ * ‖b‖₊ :=
  NNReal.eq <| norm_mul a b


/-- `norm` as a `MonoidWithZeroHom`. -/
@[simps]
def normHom : α →*₀ ℝ where
  toFun := (‖·‖)
  map_zero' := norm_zero
  map_one' := norm_one
  map_mul' := norm_mul


/-- `nnnorm` as a `MonoidWithZeroHom`. -/
@[simps]
def nnnormHom : α →*₀ ℝ≥0 where
  toFun := (‖·‖₊)
  map_zero' := nnnorm_zero
  map_one' := nnnorm_one
  map_mul' := nnnorm_mul


@[simp]
theorem norm_pow (a : α) : ∀ n : ℕ, ‖a ^ n‖ = ‖a‖ ^ n :=
  (normHom.toMonoidHom : α →* ℝ).map_pow a


@[simp]
theorem nnnorm_pow (a : α) (n : ℕ) : ‖a ^ n‖₊ = ‖a‖₊ ^ n :=
  (nnnormHom.toMonoidHom : α →* ℝ≥0).map_pow a n


protected theorem List.norm_prod (l : List α) : ‖l.prod‖ = (l.map norm).prod :=
  map_list_prod (normHom.toMonoidHom : α →* ℝ) _


protected theorem List.nnnorm_prod (l : List α) : ‖l.prod‖₊ = (l.map nnnorm).prod :=
  map_list_prod (nnnormHom.toMonoidHom : α →* ℝ≥0) _


@[simp]
theorem norm_div (a b : α) : ‖a / b‖ = ‖a‖ / ‖b‖ :=
  map_div₀ (normHom : α →*₀ ℝ) a b


@[simp]
theorem nnnorm_div (a b : α) : ‖a / b‖₊ = ‖a‖₊ / ‖b‖₊ :=
  map_div₀ (nnnormHom : α →*₀ ℝ≥0) a b


@[simp]
theorem norm_inv (a : α) : ‖a⁻¹‖ = ‖a‖⁻¹ :=
  map_inv₀ (normHom : α →*₀ ℝ) a


@[simp]
theorem nnnorm_inv (a : α) : ‖a⁻¹‖₊ = ‖a‖₊⁻¹ :=
                  /-
                    α : Type u_1
                    inst✝ : NormedDivisionRing α
                    a : α
                    ⊢ Eq ↑(NNNorm.nnnorm (Inv.inv a)) ↑(Inv.inv (NNNorm.nnnorm a))
                  -/
  NNReal.eq <| by simp
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem norm_zpow : ∀ (a : α) (n : ℤ), ‖a ^ n‖ = ‖a‖ ^ n :=
  map_zpow₀ (normHom : α →*₀ ℝ)


@[simp]
theorem nnnorm_zpow : ∀ (a : α) (n : ℤ), ‖a ^ n‖₊ = ‖a‖₊ ^ n :=
  map_zpow₀ (nnnormHom : α →*₀ ℝ≥0)


theorem dist_inv_inv₀ {z w : α} (hz : z ≠ 0) (hw : w ≠ 0) :
    dist z⁻¹ w⁻¹ = dist z w / (‖z‖ * ‖w‖) := by
  rw [dist_eq_norm, inv_sub_inv' hz hw, norm_mul, norm_mul, norm_inv, norm_inv, mul_comm ‖z‖⁻¹,
    mul_assoc, dist_eq_norm', div_eq_mul_inv, mul_inv]


theorem nndist_inv_inv₀ {z w : α} (hz : z ≠ 0) (hw : w ≠ 0) :
    nndist z⁻¹ w⁻¹ = nndist z w / (‖z‖₊ * ‖w‖₊) :=
  NNReal.eq <| dist_inv_inv₀ hz hw


lemma norm_commutator_sub_one_le (ha : a ≠ 0) (hb : b ≠ 0) :
    ‖a * b * a⁻¹ * b⁻¹ - 1‖ ≤ 2 * ‖a‖⁻¹ * ‖b‖⁻¹ * ‖a - 1‖ * ‖b - 1‖ := by
  /-
    α : Type u_1
    inst✝ : NormedDivisionRing α
    a b : α
    ha : Ne a 0
    hb : Ne b 0
    ⊢ LE.le (Norm.norm (HSub.hSub (HMul.hMul (HMul.hMul (HMul.hMul a b) (Inv.inv a …
  -/
  simpa using norm_commutator_units_sub_one_le (.mk0 a ha) (.mk0 b hb)
  /-
    🎉 no goals
  -/


lemma nnnorm_commutator_sub_one_le (ha : a ≠ 0) (hb : b ≠ 0) :
    ‖a * b * a⁻¹ * b⁻¹ - 1‖₊ ≤ 2 * ‖a‖₊⁻¹ * ‖b‖₊⁻¹ * ‖a - 1‖₊ * ‖b - 1‖₊ := by
  /-
    α : Type u_1
    inst✝ : NormedDivisionRing α
    a b : α
    ha : Ne a 0
    hb : Ne b 0
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub (HMul.hMul (HMul.hMul (HMul.hMul a b) (Inv.i …
  -/
  simpa using nnnorm_commutator_units_sub_one_le (.mk0 a ha) (.mk0 b hb)
  /-
    🎉 no goals
  -/


lemma norm_eq_one_iff_ne_zero_of_discrete {x : 𝕜} : ‖x‖ = 1 ↔ x ≠ 0 := by
  /-
    𝕜 : Type u_4
    inst✝¹ : NormedDivisionRing 𝕜
    inst✝ : DiscreteTopology 𝕜
    x : 𝕜
    ⊢ Iff (Eq (Norm.norm x) 1) (Ne x 0)
  -/
  constructor <;> intro hx
    /-
      case mp
      𝕜 : Type u_4
      inst✝¹ : NormedDivisionRing 𝕜
      inst✝ : DiscreteTopology 𝕜
      x : 𝕜
      hx : Eq (Norm.norm x) 1
      ⊢ Ne x 0
    -/
  · contrapose! hx
    /-
      case mp
      𝕜 : Type u_4
      inst✝¹ : NormedDivisionRing 𝕜
      inst✝ : DiscreteTopology 𝕜
      x : 𝕜
      hx : Eq x 0
      ⊢ Ne (Norm.norm x) 1
    -/
    simp [hx]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_4
      inst✝¹ : NormedDivisionRing 𝕜
      inst✝ : DiscreteTopology 𝕜
      x : 𝕜
      hx : Ne x 0
      ⊢ Eq (Norm.norm x) 1
    -/
  · have : IsOpen {(0 : 𝕜)} := isOpen_discrete {0}
    /-
      case mpr
      𝕜 : Type u_4
      inst✝¹ : NormedDivisionRing 𝕜
      inst✝ : DiscreteTopology 𝕜
      x : 𝕜
      hx : Ne x 0
      this : IsOpen (Singleton.singleton 0)
      ⊢ Eq (Norm.norm x) 1
    -/
    simp_rw [Metric.isOpen_singleton_iff, dist_eq_norm, sub_zero] at this
    /-
      case mpr
      𝕜 : Type u_4
      inst✝¹ : NormedDivisionRing 𝕜
      inst✝ : DiscreteTopology 𝕜
      x : 𝕜
      hx : Ne x 0
      this : Exists fun ε => And (GT.gt ε 0) (∀ (y : 𝕜), LT.lt (Norm.norm y) ε → Eq  …
      ⊢ Eq (Norm.norm x) 1
    -/
    obtain ⟨ε, εpos, h'⟩ := this
    /-
      case mpr.intro.intro
      𝕜 : Type u_4
      inst✝¹ : NormedDivisionRing 𝕜
      inst✝ : DiscreteTopology 𝕜
      x : 𝕜
      hx : Ne x 0
      ε : Real
      εpos : GT.gt ε 0
      h' : ∀ (y : 𝕜), LT.lt (Norm.norm y) ε → Eq y 0
      ⊢ Eq (Norm.norm x) 1
    -/
    wlog h : ‖x‖ < 1 generalizing 𝕜 with H
      /-
        case mpr.intro.intro.inr
        𝕜 : Type u_4
        inst✝¹ : NormedDivisionRing 𝕜
        inst✝ : DiscreteTopology 𝕜
        x : 𝕜
        hx : Ne x 0
        ε : Real
        εpos : GT.gt ε 0
        h' : ∀ (y : 𝕜), LT.lt (Norm.norm y) ε → Eq y 0
        H : ∀ {𝕜 : Type u_4} [inst : NormedDivisionRing 𝕜] [inst_1 : DiscreteTopology  …
        h : Not (LT.lt (Norm.norm x) 1)
        ⊢ Eq (Norm.norm x) 1
      -/
    · push_neg at h
      /-
        case mpr.intro.intro.inr
        𝕜 : Type u_4
        inst✝¹ : NormedDivisionRing 𝕜
        inst✝ : DiscreteTopology 𝕜
        x : 𝕜
        hx : Ne x 0
        ε : Real
        εpos : GT.gt ε 0
        h' : ∀ (y : 𝕜), LT.lt (Norm.norm y) ε → Eq y 0
        H : ∀ {𝕜 : Type u_4} [inst : NormedDivisionRing 𝕜] [inst_1 : DiscreteTopology  …
        h : LE.le 1 (Norm.norm x)
        ⊢ Eq (Norm.norm x) 1
      -/
      rcases h.eq_or_lt with h|h
        /-
          case mpr.intro.intro.inr.inl
          𝕜 : Type u_4
          inst✝¹ : NormedDivisionRing 𝕜
          inst✝ : DiscreteTopology 𝕜
          x : 𝕜
          hx : Ne x 0
          ε : Real
          εpos : GT.gt ε 0
          h' : ∀ (y : 𝕜), LT.lt (Norm.norm y) ε → Eq y 0
          H : ∀ {𝕜 : Type u_4} [inst : NormedDivisionRing 𝕜] [inst_1 : DiscreteTopology  …
          h✝ : LE.le 1 (Norm.norm x)
          h : Eq 1 (Norm.norm x)
          ⊢ Eq (Norm.norm x) 1
        -/
      · rw [h]
        /-
          🎉 no goals
        -/
      /-
        case mpr.intro.intro.inr.inr
        𝕜 : Type u_4
        inst✝¹ : NormedDivisionRing 𝕜
        inst✝ : DiscreteTopology 𝕜
        x : 𝕜
        hx : Ne x 0
        ε : Real
        εpos : GT.gt ε 0
        h' : ∀ (y : 𝕜), LT.lt (Norm.norm y) ε → Eq y 0
        H : ∀ {𝕜 : Type u_4} [inst : NormedDivisionRing 𝕜] [inst_1 : DiscreteTopology  …
        h✝ : LE.le 1 (Norm.norm x)
        h : LT.lt 1 (Norm.norm x)
        ⊢ Eq (Norm.norm x) 1
      -/
      replace h := norm_inv x ▸ inv_lt_one_of_one_lt₀ h
      /-
        case mpr.intro.intro.inr.inr
        𝕜 : Type u_4
        inst✝¹ : NormedDivisionRing 𝕜
        inst✝ : DiscreteTopology 𝕜
        x : 𝕜
        hx : Ne x 0
        ε : Real
        εpos : GT.gt ε 0
        h' : ∀ (y : 𝕜), LT.lt (Norm.norm y) ε → Eq y 0
        H : ∀ {𝕜 : Type u_4} [inst : NormedDivisionRing 𝕜] [inst_1 : DiscreteTopology  …
        h✝ : LE.le 1 (Norm.norm x)
        h : LT.lt (Norm.norm (Inv.inv x)) 1
        ⊢ Eq (Norm.norm x) 1
      -/
      rw [← inv_inj, inv_one, ← norm_inv]
      /-
        case mpr.intro.intro.inr.inr
        𝕜 : Type u_4
        inst✝¹ : NormedDivisionRing 𝕜
        inst✝ : DiscreteTopology 𝕜
        x : 𝕜
        hx : Ne x 0
        ε : Real
        εpos : GT.gt ε 0
        h' : ∀ (y : 𝕜), LT.lt (Norm.norm y) ε → Eq y 0
        H : ∀ {𝕜 : Type u_4} [inst : NormedDivisionRing 𝕜] [inst_1 : DiscreteTopology  …
        h✝ : LE.le 1 (Norm.norm x)
        h : LT.lt (Norm.norm (Inv.inv x)) 1
        ⊢ Eq (Norm.norm (Inv.inv x)) 1
      -/
      exact H (by simpa) h' h
      /-
        🎉 no goals
      -/
    /-
      𝕜✝ : Type u_4
      inst✝² : NormedDivisionRing 𝕜✝
      ε : Real
      εpos : GT.gt ε 0
      𝕜 : Type u_4
      inst✝¹ : NormedDivisionRing 𝕜
      inst✝ : DiscreteTopology 𝕜
      x : 𝕜
      hx : Ne x 0
      h' : ∀ (y : 𝕜), LT.lt (Norm.norm y) ε → Eq y 0
      h : LT.lt (Norm.norm x) 1
      ⊢ Eq (Norm.norm x) 1
    -/
    obtain ⟨k, hk⟩ : ∃ k : ℕ, ‖x‖ ^ k < ε := exists_pow_lt_of_lt_one εpos h
    /-
      case intro
      𝕜✝ : Type u_4
      inst✝² : NormedDivisionRing 𝕜✝
      ε : Real
      εpos : GT.gt ε 0
      𝕜 : Type u_4
      inst✝¹ : NormedDivisionRing 𝕜
      inst✝ : DiscreteTopology 𝕜
      x : 𝕜
      hx : Ne x 0
      h' : ∀ (y : 𝕜), LT.lt (Norm.norm y) ε → Eq y 0
      h : LT.lt (Norm.norm x) 1
      k : Nat
      hk : LT.lt (HPow.hPow (Norm.norm x) k) ε
      ⊢ Eq (Norm.norm x) 1
    -/
    rw [← norm_pow] at hk
    /-
      case intro
      𝕜✝ : Type u_4
      inst✝² : NormedDivisionRing 𝕜✝
      ε : Real
      εpos : GT.gt ε 0
      𝕜 : Type u_4
      inst✝¹ : NormedDivisionRing 𝕜
      inst✝ : DiscreteTopology 𝕜
      x : 𝕜
      hx : Ne x 0
      h' : ∀ (y : 𝕜), LT.lt (Norm.norm y) ε → Eq y 0
      h : LT.lt (Norm.norm x) 1
      k : Nat
      hk : LT.lt (Norm.norm (HPow.hPow x k)) ε
      ⊢ Eq (Norm.norm x) 1
    -/
    specialize h' _ hk
    /-
      case intro
      𝕜✝ : Type u_4
      inst✝² : NormedDivisionRing 𝕜✝
      ε : Real
      εpos : GT.gt ε 0
      𝕜 : Type u_4
      inst✝¹ : NormedDivisionRing 𝕜
      inst✝ : DiscreteTopology 𝕜
      x : 𝕜
      hx : Ne x 0
      h : LT.lt (Norm.norm x) 1
      k : Nat
      hk : LT.lt (Norm.norm (HPow.hPow x k)) ε
      h' : Eq (HPow.hPow x k) 0
      ⊢ Eq (Norm.norm x) 1
    -/
    simp [hx] at h'
    /-
      🎉 no goals
    -/


@[simp]
lemma norm_le_one_of_discrete
    (x : 𝕜) : ‖x‖ ≤ 1 := by
  /-
    𝕜 : Type u_4
    inst✝¹ : NormedDivisionRing 𝕜
    inst✝ : DiscreteTopology 𝕜
    x : 𝕜
    ⊢ LE.le (Norm.norm x) 1
  -/
  rcases eq_or_ne x 0 with rfl|hx
    /-
      case inl
      𝕜 : Type u_4
      inst✝¹ : NormedDivisionRing 𝕜
      inst✝ : DiscreteTopology 𝕜
      ⊢ LE.le (Norm.norm 0) 1
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_4
      inst✝¹ : NormedDivisionRing 𝕜
      inst✝ : DiscreteTopology 𝕜
      x : 𝕜
      hx : Ne x 0
      ⊢ LE.le (Norm.norm x) 1
    -/
  · simp [norm_eq_one_iff_ne_zero_of_discrete.mpr hx]
    /-
      🎉 no goals
    -/


lemma unitClosedBall_eq_univ_of_discrete : (Metric.closedBall 0 1 : Set 𝕜) = Set.univ := by
  /-
    𝕜 : Type u_4
    inst✝¹ : NormedDivisionRing 𝕜
    inst✝ : DiscreteTopology 𝕜
    ⊢ Eq (Metric.closedBall 0 1) Set.univ
  -/
  ext
  /-
    case h
    𝕜 : Type u_4
    inst✝¹ : NormedDivisionRing 𝕜
    inst✝ : DiscreteTopology 𝕜
    x✝ : 𝕜
    ⊢ Iff (Membership.mem (Metric.closedBall 0 1) x✝) (Membership.mem Set.univ x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-01")]
alias discreteTopology_unit_closedBall_eq_univ := unitClosedBall_eq_univ_of_discrete


/-- A normed field is a field with a norm satisfying ‖x y‖ = ‖x‖ ‖y‖. -/
class NormedField (α : Type*) extends Norm α, Field α, MetricSpace α where
  /-- The distance is induced by the norm. -/
  dist_eq : ∀ x y, dist x y = norm (x - y)
  /-- The norm is multiplicative. -/
  norm_mul' : ∀ a b, norm (a * b) = norm a * norm b


/-- A nontrivially normed field is a normed field in which there is an element of norm different
from `0` and `1`. This makes it possible to bring any element arbitrarily close to `0` by
multiplication by the powers of any element, and thus to relate algebra and topology. -/
class NontriviallyNormedField (α : Type*) extends NormedField α where
  /-- The norm attains a value exceeding 1. -/
  non_trivial : ∃ x : α, 1 < ‖x‖


/-- A densely normed field is a normed field for which the image of the norm is dense in `ℝ≥0`,
which means it is also nontrivially normed. However, not all nontrivally normed fields are densely
normed; in particular, the `Padic`s exhibit this fact. -/
class DenselyNormedField (α : Type*) extends NormedField α where
  /-- The range of the norm is dense in the collection of nonnegative real numbers. -/
  lt_norm_lt : ∀ x y : ℝ, 0 ≤ x → x < y → ∃ a : α, x < ‖a‖ ∧ ‖a‖ < y


/-- A densely normed field is always a nontrivially normed field.
See note [lower instance priority]. -/
instance (priority := 100) DenselyNormedField.toNontriviallyNormedField [DenselyNormedField α] :
    NontriviallyNormedField α where
  non_trivial :=
    let ⟨a, h, _⟩ := DenselyNormedField.lt_norm_lt 1 2 zero_le_one one_lt_two
    ⟨a, h⟩


instance (priority := 100) NormedField.toNormedDivisionRing : NormedDivisionRing α :=
  { ‹NormedField α› with }

-- see Note [lower instance priority]

instance (priority := 100) NormedField.toNormedCommRing : NormedCommRing α :=
  { ‹NormedField α› with norm_mul := fun a b => (norm_mul a b).le }


@[simp]
theorem norm_prod (s : Finset β) (f : β → α) : ‖∏ b ∈ s, f b‖ = ∏ b ∈ s, ‖f b‖ :=
  map_prod normHom.toMonoidHom f s


@[simp]
theorem nnnorm_prod (s : Finset β) (f : β → α) : ‖∏ b ∈ s, f b‖₊ = ∏ b ∈ s, ‖f b‖₊ :=
  map_prod nnnormHom.toMonoidHom f s


theorem exists_one_lt_norm : ∃ x : α, 1 < ‖x‖ :=
  ‹NontriviallyNormedField α›.non_trivial


theorem exists_lt_norm (r : ℝ) : ∃ x : α, r < ‖x‖ :=
  let ⟨w, hw⟩ := exists_one_lt_norm α
  let ⟨n, hn⟩ := pow_unbounded_of_one_lt r hw
             /-
               α : Type u_1
               inst✝ : NontriviallyNormedField α
               r : Real
               w : α
               hw : LT.lt 1 (Norm.norm w)
               n : Nat
               hn : LT.lt r (HPow.hPow (Norm.norm w) n)
               ⊢ LT.lt r (Norm.norm (HPow.hPow w n))
             -/
  ⟨w ^ n, by rwa [norm_pow]⟩
             /-
               🎉 no goals
             -/


theorem exists_norm_lt {r : ℝ} (hr : 0 < r) : ∃ x : α, 0 < ‖x‖ ∧ ‖x‖ < r :=
  let ⟨w, hw⟩ := exists_lt_norm α r⁻¹
           /-
             α : Type u_1
             inst✝ : NontriviallyNormedField α
             r : Real
             hr : LT.lt 0 r
             w : α
             hw : LT.lt (Inv.inv r) (Norm.norm w)
             ⊢ And (LT.lt 0 (Norm.norm (Inv.inv w))) (LT.lt (Norm.norm (Inv.inv w)) r)
           -/
  ⟨w⁻¹, by rwa [← Set.mem_Ioo, norm_inv, ← Set.mem_inv, Set.inv_Ioo_0_left hr]⟩
           /-
             🎉 no goals
           -/


theorem exists_norm_lt_one : ∃ x : α, 0 < ‖x‖ ∧ ‖x‖ < 1 :=
  exists_norm_lt α one_pos


@[instance]
theorem punctured_nhds_neBot (x : α) : NeBot (𝓝[≠] x) := by
  /-
    α : Type u_1
    inst✝ : NontriviallyNormedField α
    x : α
    ⊢ (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).NeBot
  -/
  rw [← mem_closure_iff_nhdsWithin_neBot, Metric.mem_closure_iff]
  /-
    α : Type u_1
    inst✝ : NontriviallyNormedField α
    x : α
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Exists fun b => And (Membership.mem (HasCompl.comp …
  -/
  rintro ε ε0
  /-
    α : Type u_1
    inst✝ : NontriviallyNormedField α
    x : α
    ε : Real
    ε0 : GT.gt ε 0
    ⊢ Exists fun b => And (Membership.mem (HasCompl.compl (Singleton.singleton x)) …
  -/
  rcases exists_norm_lt α ε0 with ⟨b, hb0, hbε⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝ : NontriviallyNormedField α
    x : α
    ε : Real
    ε0 : GT.gt ε 0
    b : α
    hb0 : LT.lt 0 (Norm.norm b)
    hbε : LT.lt (Norm.norm b) ε
    ⊢ Exists fun b => And (Membership.mem (HasCompl.compl (Singleton.singleton x)) …
  -/
  refine ⟨x + b, mt (Set.mem_singleton_iff.trans add_right_eq_self).1 <| norm_pos_iff.1 hb0, ?_⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝ : NontriviallyNormedField α
    x : α
    ε : Real
    ε0 : GT.gt ε 0
    b : α
    hb0 : LT.lt 0 (Norm.norm b)
    hbε : LT.lt (Norm.norm b) ε
    ⊢ LT.lt (Dist.dist x (HAdd.hAdd x b)) ε
  -/
  rwa [dist_comm, dist_eq_norm, add_sub_cancel_left]
  /-
    🎉 no goals
  -/


@[instance]
theorem nhdsWithin_isUnit_neBot : NeBot (𝓝[{ x : α | IsUnit x }] 0) := by
  /-
    α : Type u_1
    inst✝ : NontriviallyNormedField α
    ⊢ (nhdsWithin 0 (setOf fun x => IsUnit x)).NeBot
  -/
  simpa only [isUnit_iff_ne_zero] using punctured_nhds_neBot (0 : α)
  /-
    🎉 no goals
  -/


theorem exists_lt_norm_lt {r₁ r₂ : ℝ} (h₀ : 0 ≤ r₁) (h : r₁ < r₂) : ∃ x : α, r₁ < ‖x‖ ∧ ‖x‖ < r₂ :=
  DenselyNormedField.lt_norm_lt r₁ r₂ h₀ h


theorem exists_lt_nnnorm_lt {r₁ r₂ : ℝ≥0} (h : r₁ < r₂) : ∃ x : α, r₁ < ‖x‖₊ ∧ ‖x‖₊ < r₂ :=
  mod_cast exists_lt_norm_lt α r₁.prop h


instance denselyOrdered_range_norm : DenselyOrdered (Set.range (norm : α → ℝ)) where
  dense := by
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : DenselyNormedField α
      ⊢ ∀ (a₁ a₂ : ↑(Set.range Norm.norm)), LT.lt a₁ a₂ → Exists fun a => And (LT.lt …
    -/
    rintro ⟨-, x, rfl⟩ ⟨-, y, rfl⟩ hxy
    /-
      case mk.intro.mk.intro
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : DenselyNormedField α
      x y : α
      hxy : LT.lt ⟨Norm.norm x, ⋯⟩ ⟨Norm.norm y, ⋯⟩
      ⊢ Exists fun a => And (LT.lt ⟨Norm.norm x, ⋯⟩ a) (LT.lt a ⟨Norm.norm y, ⋯⟩)
    -/
    let ⟨z, h⟩ := exists_lt_norm_lt α (norm_nonneg _) hxy
    /-
      case mk.intro.mk.intro
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : DenselyNormedField α
      x y : α
      hxy : LT.lt ⟨Norm.norm x, ⋯⟩ ⟨Norm.norm y, ⋯⟩
      z : α
      h : And (LT.lt (Norm.norm x) (Norm.norm z)) (LT.lt (Norm.norm z) ↑⟨Norm.norm y …
      ⊢ Exists fun a => And (LT.lt ⟨Norm.norm x, ⋯⟩ a) (LT.lt a ⟨Norm.norm y, ⋯⟩)
    -/
    exact ⟨⟨‖z‖, z, rfl⟩, h⟩
    /-
      🎉 no goals
    -/


instance denselyOrdered_range_nnnorm : DenselyOrdered (Set.range (nnnorm : α → ℝ≥0)) where
  dense := by
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : DenselyNormedField α
      ⊢ ∀ (a₁ a₂ : ↑(Set.range NNNorm.nnnorm)), LT.lt a₁ a₂ → Exists fun a => And (L …
    -/
    rintro ⟨-, x, rfl⟩ ⟨-, y, rfl⟩ hxy
    /-
      case mk.intro.mk.intro
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : DenselyNormedField α
      x y : α
      hxy : LT.lt ⟨NNNorm.nnnorm x, ⋯⟩ ⟨NNNorm.nnnorm y, ⋯⟩
      ⊢ Exists fun a => And (LT.lt ⟨NNNorm.nnnorm x, ⋯⟩ a) (LT.lt a ⟨NNNorm.nnnorm y …
    -/
    let ⟨z, h⟩ := exists_lt_nnnorm_lt α hxy
    /-
      case mk.intro.mk.intro
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      inst✝ : DenselyNormedField α
      x y : α
      hxy : LT.lt ⟨NNNorm.nnnorm x, ⋯⟩ ⟨NNNorm.nnnorm y, ⋯⟩
      z : α
      h : And (LT.lt (↑⟨NNNorm.nnnorm x, ⋯⟩) (NNNorm.nnnorm z)) (LT.lt (NNNorm.nnnor …
      ⊢ Exists fun a => And (LT.lt ⟨NNNorm.nnnorm x, ⋯⟩ a) (LT.lt a ⟨NNNorm.nnnorm y …
    -/
    exact ⟨⟨‖z‖₊, z, rfl⟩, h⟩
    /-
      🎉 no goals
    -/


/-- A normed field is nontrivially normed
provided that the norm of some nonzero element is not one. -/
def NontriviallyNormedField.ofNormNeOne {𝕜 : Type*} [h' : NormedField 𝕜]
    (h : ∃ x : 𝕜, x ≠ 0 ∧ ‖x‖ ≠ 1) : NontriviallyNormedField 𝕜 where
  toNormedField := h'
  non_trivial := by
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      𝕜 : Type u_4
      h' : NormedField 𝕜
      h : Exists fun x => And (Ne x 0) (Ne (Norm.norm x) 1)
      ⊢ Exists fun x => LT.lt 1 (Norm.norm x)
    -/
    rcases h with ⟨x, hx, hx1⟩
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      𝕜 : Type u_4
      h' : NormedField 𝕜
      x : 𝕜
      hx : Ne x 0
      hx1 : Ne (Norm.norm x) 1
      ⊢ Exists fun x => LT.lt 1 (Norm.norm x)
    -/
    rcases hx1.lt_or_lt with hlt | hlt
      /-
        case intro.intro.inl
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        𝕜 : Type u_4
        h' : NormedField 𝕜
        x : 𝕜
        hx : Ne x 0
        hx1 : Ne (Norm.norm x) 1
        hlt : LT.lt (Norm.norm x) 1
        ⊢ Exists fun x => LT.lt 1 (Norm.norm x)
      -/
    · use x⁻¹
      /-
        case h
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        𝕜 : Type u_4
        h' : NormedField 𝕜
        x : 𝕜
        hx : Ne x 0
        hx1 : Ne (Norm.norm x) 1
        hlt : LT.lt (Norm.norm x) 1
        ⊢ LT.lt 1 (Norm.norm (Inv.inv x))
      -/
      rw [norm_inv]
      /-
        case h
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        𝕜 : Type u_4
        h' : NormedField 𝕜
        x : 𝕜
        hx : Ne x 0
        hx1 : Ne (Norm.norm x) 1
        hlt : LT.lt (Norm.norm x) 1
        ⊢ LT.lt 1 (Inv.inv (Norm.norm x))
      -/
      exact (one_lt_inv₀ (norm_pos_iff.2 hx)).2 hlt
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.inr
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        𝕜 : Type u_4
        h' : NormedField 𝕜
        x : 𝕜
        hx : Ne x 0
        hx1 : Ne (Norm.norm x) 1
        hlt : LT.lt 1 (Norm.norm x)
        ⊢ Exists fun x => LT.lt 1 (Norm.norm x)
      -/
    · exact ⟨x, hlt⟩
      /-
        🎉 no goals
      -/


instance Real.normedCommRing : NormedCommRing ℝ :=
  { Real.normedAddCommGroup, Real.commRing with norm_mul := fun x y => (abs_mul x y).le }


noncomputable instance Real.normedField : NormedField ℝ :=
  { Real.normedAddCommGroup, Real.field with
    norm_mul' := abs_mul }


noncomputable instance Real.denselyNormedField : DenselyNormedField ℝ where
  lt_norm_lt _ _ h₀ hr :=
    let ⟨x, h⟩ := exists_between hr
           /-
             α : Type u_1
             β : Type u_2
             ι : Type u_3
             x✝¹ x✝ : Real
             h₀ : LE.le 0 x✝¹
             hr : LT.lt x✝¹ x✝
             x : Real
             h : And (LT.lt x✝¹ x) (LT.lt x x✝)
             ⊢ And (LT.lt x✝¹ (Norm.norm x)) (LT.lt (Norm.norm x) x✝)
           -/
    ⟨x, by rwa [Real.norm_eq_abs, abs_of_nonneg (h₀.trans h.1.le)]⟩
           /-
             🎉 no goals
           -/


theorem toNNReal_mul_nnnorm {x : ℝ} (y : ℝ) (hx : 0 ≤ x) : x.toNNReal * ‖y‖₊ = ‖x * y‖₊ := by
  /-
    x y : Real
    hx : LE.le 0 x
    ⊢ Eq (HMul.hMul x.toNNReal (NNNorm.nnnorm y)) (NNNorm.nnnorm (HMul.hMul x y))
  -/
  ext
  simp only [NNReal.coe_mul, nnnorm_mul, coe_nnnorm, Real.toNNReal_of_nonneg, norm_of_nonneg, hx,
    NNReal.coe_mk]


theorem nnnorm_mul_toNNReal (x : ℝ) {y : ℝ} (hy : 0 ≤ y) : ‖x‖₊ * y.toNNReal = ‖x * y‖₊ := by
  /-
    x y : Real
    hy : LE.le 0 y
    ⊢ Eq (HMul.hMul (NNNorm.nnnorm x) y.toNNReal) (NNNorm.nnnorm (HMul.hMul x y))
  -/
  rw [mul_comm, mul_comm x, toNNReal_mul_nnnorm x hy]
  /-
    🎉 no goals
  -/


                                                /-
                                                  x : NNReal
                                                  ⊢ Eq (Norm.norm ↑x) ↑x
                                                -/
theorem norm_eq (x : ℝ≥0) : ‖(x : ℝ)‖ = x := by rw [Real.norm_eq_abs, x.abs_eq]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem nnnorm_eq (x : ℝ≥0) : ‖(x : ℝ)‖₊ = x :=
  NNReal.eq <| Real.norm_of_nonneg x.2


@[simp 1001] -- Porting note: increase priority so that the LHS doesn't simplify
theorem norm_norm [SeminormedAddCommGroup α] (x : α) : ‖‖x‖‖ = ‖x‖ :=
  Real.norm_of_nonneg (norm_nonneg _)


@[simp]
theorem nnnorm_norm [SeminormedAddCommGroup α] (a : α) : ‖‖a‖‖₊ = ‖a‖₊ := by
  /-
    α : Type u_1
    inst✝ : SeminormedAddCommGroup α
    a : α
    ⊢ Eq (NNNorm.nnnorm (Norm.norm a)) (NNNorm.nnnorm a)
  -/
  rw [Real.nnnorm_of_nonneg (norm_nonneg a)]; rfl
                                              /-
                                                🎉 no goals
                                              -/


/-- A restatement of `MetricSpace.tendsto_atTop` in terms of the norm. -/
theorem NormedAddCommGroup.tendsto_atTop [Nonempty α] [Preorder α] [IsDirected α (· ≤ ·)]
    {β : Type*} [SeminormedAddCommGroup β] {f : α → β} {b : β} :
    Tendsto f atTop (𝓝 b) ↔ ∀ ε, 0 < ε → ∃ N, ∀ n, N ≤ n → ‖f n - b‖ < ε :=
                                                             /-
                                                               α : Type u_1
                                                               inst✝³ : Nonempty α
                                                               inst✝² : Preorder α
                                                               inst✝¹ : IsDirected α fun x1 x2 => LE.le x1 x2
                                                               β : Type u_4
                                                               inst✝ : SeminormedAddCommGroup β
                                                               f : α → β
                                                               b : β
                                                               ⊢ Iff (∀ (ib : Real), LT.lt 0 ib → Exists fun ia => And True (∀ (x : α), Membe …
                                                             -/
  (atTop_basis.tendsto_iff Metric.nhds_basis_ball).trans (by simp [dist_eq_norm])
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- A variant of `NormedAddCommGroup.tendsto_atTop` that
uses `∃ N, ∀ n > N, ...` rather than `∃ N, ∀ n ≥ N, ...`
-/
theorem NormedAddCommGroup.tendsto_atTop' [Nonempty α] [Preorder α] [IsDirected α (· ≤ ·)]
    [NoMaxOrder α] {β : Type*} [SeminormedAddCommGroup β] {f : α → β} {b : β} :
    Tendsto f atTop (𝓝 b) ↔ ∀ ε, 0 < ε → ∃ N, ∀ n, N < n → ‖f n - b‖ < ε :=
                                                                 /-
                                                                   α : Type u_1
                                                                   inst✝⁴ : Nonempty α
                                                                   inst✝³ : Preorder α
                                                                   inst✝² : IsDirected α fun x1 x2 => LE.le x1 x2
                                                                   inst✝¹ : NoMaxOrder α
                                                                   β : Type u_4
                                                                   inst✝ : SeminormedAddCommGroup β
                                                                   f : α → β
                                                                   b : β
                                                                   ⊢ Iff (∀ (ib : Real), LT.lt 0 ib → Exists fun ia => And True (∀ (x : α), Membe …
                                                                 -/
  (atTop_basis_Ioi.tendsto_iff Metric.nhds_basis_ball).trans (by simp [dist_eq_norm])
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- This class states that a ring homomorphism is isometric. This is a sufficient assumption
for a continuous semilinear map to be bounded and this is the main use for this typeclass. -/
class RingHomIsometric [Semiring R₁] [Semiring R₂] [Norm R₁] [Norm R₂] (σ : R₁ →+* R₂) : Prop where
  /-- The ring homomorphism is an isometry. -/
  is_iso : ∀ {x : R₁}, ‖σ x‖ = ‖x‖


instance RingHomIsometric.ids : RingHomIsometric (RingHom.id R₁) :=
  ⟨rfl⟩


/-- A non-unital ring homomorphism from a `NonUnitalRing` to a `NonUnitalSeminormedRing`
induces a `NonUnitalSeminormedRing` structure on the domain.

See note [reducible non-instances] -/
abbrev NonUnitalSeminormedRing.induced [NonUnitalRing R] [NonUnitalSeminormedRing S]
    [NonUnitalRingHomClass F R S] (f : F) : NonUnitalSeminormedRing R :=
  { SeminormedAddCommGroup.induced R S f, ‹NonUnitalRing R› with
    norm_mul := fun x y => by
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        F : Type u_4
        R : Type u_5
        S : Type u_6
        inst✝³ : FunLike F R S
        inst✝² : NonUnitalRing R
        inst✝¹ : NonUnitalSeminormedRing S
        inst✝ : NonUnitalRingHomClass F R S
        f : F
        x y : R
        ⊢ LE.le (Norm.norm (HMul.hMul x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
      -/
      show ‖f (x * y)‖ ≤ ‖f x‖ * ‖f y‖
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        F : Type u_4
        R : Type u_5
        S : Type u_6
        inst✝³ : FunLike F R S
        inst✝² : NonUnitalRing R
        inst✝¹ : NonUnitalSeminormedRing S
        inst✝ : NonUnitalRingHomClass F R S
        f : F
        x y : R
        ⊢ LE.le (Norm.norm (f (HMul.hMul x y))) (HMul.hMul (Norm.norm (f x)) (Norm.nor …
      -/
      exact (map_mul f x y).symm ▸ norm_mul_le (f x) (f y) }
      /-
        🎉 no goals
      -/


/-- An injective non-unital ring homomorphism from a `NonUnitalRing` to a
`NonUnitalNormedRing` induces a `NonUnitalNormedRing` structure on the domain.

See note [reducible non-instances] -/
abbrev NonUnitalNormedRing.induced [NonUnitalRing R] [NonUnitalNormedRing S]
    [NonUnitalRingHomClass F R S] (f : F) (hf : Function.Injective f) : NonUnitalNormedRing R :=
  { NonUnitalSeminormedRing.induced R S f, NormedAddCommGroup.induced R S f hf with }


/-- A non-unital ring homomorphism from a `Ring` to a `SeminormedRing` induces a
`SeminormedRing` structure on the domain.

See note [reducible non-instances] -/
abbrev SeminormedRing.induced [Ring R] [SeminormedRing S] [NonUnitalRingHomClass F R S] (f : F) :
    SeminormedRing R :=
  { NonUnitalSeminormedRing.induced R S f, SeminormedAddCommGroup.induced R S f, ‹Ring R› with }


/-- An injective non-unital ring homomorphism from a `Ring` to a `NormedRing` induces a
`NormedRing` structure on the domain.

See note [reducible non-instances] -/
abbrev NormedRing.induced [Ring R] [NormedRing S] [NonUnitalRingHomClass F R S] (f : F)
    (hf : Function.Injective f) : NormedRing R :=
  { NonUnitalSeminormedRing.induced R S f, NormedAddCommGroup.induced R S f hf, ‹Ring R› with }


/-- A non-unital ring homomorphism from a `NonUnitalCommRing` to a `NonUnitalSeminormedCommRing`
induces a `NonUnitalSeminormedCommRing` structure on the domain.

See note [reducible non-instances] -/
abbrev NonUnitalSeminormedCommRing.induced [NonUnitalCommRing R] [NonUnitalSeminormedCommRing S]
    [NonUnitalRingHomClass F R S] (f : F) : NonUnitalSeminormedCommRing R :=
  { NonUnitalSeminormedRing.induced R S f, ‹NonUnitalCommRing R› with }


/-- An injective non-unital ring homomorphism from a `NonUnitalCommRing` to a
`NonUnitalNormedCommRing` induces a `NonUnitalNormedCommRing` structure on the domain.

See note [reducible non-instances] -/
abbrev NonUnitalNormedCommRing.induced [NonUnitalCommRing R] [NonUnitalNormedCommRing S]
    [NonUnitalRingHomClass F R S] (f : F) (hf : Function.Injective f) : NonUnitalNormedCommRing R :=
  { NonUnitalNormedRing.induced R S f hf, ‹NonUnitalCommRing R› with }

/-- A non-unital ring homomorphism from a `CommRing` to a `SeminormedRing` induces a
`SeminormedCommRing` structure on the domain.

See note [reducible non-instances] -/
abbrev SeminormedCommRing.induced [CommRing R] [SeminormedRing S] [NonUnitalRingHomClass F R S]
    (f : F) : SeminormedCommRing R :=
  { NonUnitalSeminormedRing.induced R S f, SeminormedAddCommGroup.induced R S f, ‹CommRing R› with }


/-- An injective non-unital ring homomorphism from a `CommRing` to a `NormedRing` induces a
`NormedCommRing` structure on the domain.

See note [reducible non-instances] -/
abbrev NormedCommRing.induced [CommRing R] [NormedRing S] [NonUnitalRingHomClass F R S] (f : F)
    (hf : Function.Injective f) : NormedCommRing R :=
  { SeminormedCommRing.induced R S f, NormedAddCommGroup.induced R S f hf with }


/-- An injective non-unital ring homomorphism from a `DivisionRing` to a `NormedRing` induces a
`NormedDivisionRing` structure on the domain.

See note [reducible non-instances] -/
abbrev NormedDivisionRing.induced [DivisionRing R] [NormedDivisionRing S]
    [NonUnitalRingHomClass F R S] (f : F) (hf : Function.Injective f) : NormedDivisionRing R :=
  { NormedAddCommGroup.induced R S f hf, ‹DivisionRing R› with
    norm_mul' := fun x y => by
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        F : Type u_4
        R : Type u_5
        S : Type u_6
        inst✝³ : FunLike F R S
        inst✝² : DivisionRing R
        inst✝¹ : NormedDivisionRing S
        inst✝ : NonUnitalRingHomClass F R S
        f : F
        hf : Function.Injective ⇑f
        x y : R
        ⊢ Eq (Norm.norm (HMul.hMul x y)) (HMul.hMul (Norm.norm x) (Norm.norm y))
      -/
      show ‖f (x * y)‖ = ‖f x‖ * ‖f y‖
      /-
        α : Type u_1
        β : Type u_2
        ι : Type u_3
        F : Type u_4
        R : Type u_5
        S : Type u_6
        inst✝³ : FunLike F R S
        inst✝² : DivisionRing R
        inst✝¹ : NormedDivisionRing S
        inst✝ : NonUnitalRingHomClass F R S
        f : F
        hf : Function.Injective ⇑f
        x y : R
        ⊢ Eq (Norm.norm (f (HMul.hMul x y))) (HMul.hMul (Norm.norm (f x)) (Norm.norm ( …
      -/
      exact (map_mul f x y).symm ▸ norm_mul (f x) (f y) }
      /-
        🎉 no goals
      -/


/-- An injective non-unital ring homomorphism from a `Field` to a `NormedRing` induces a
`NormedField` structure on the domain.

See note [reducible non-instances] -/
abbrev NormedField.induced [Field R] [NormedField S] [NonUnitalRingHomClass F R S] (f : F)
    (hf : Function.Injective f) : NormedField R :=
  { NormedDivisionRing.induced R S f hf with
    mul_comm := mul_comm }


/-- A ring homomorphism from a `Ring R` to a `SeminormedRing S` which induces the norm structure
`SeminormedRing.induced` makes `R` satisfy `‖(1 : R)‖ = 1` whenever `‖(1 : S)‖ = 1`. -/
theorem NormOneClass.induced {F : Type*} (R S : Type*) [Ring R] [SeminormedRing S]
    [NormOneClass S] [FunLike F R S] [RingHomClass F R S] (f : F) :
    @NormOneClass R (SeminormedRing.induced R S f).toNorm _ :=
  -- Porting note: is this `let` a bad idea somehow?
  let _ : SeminormedRing R := SeminormedRing.induced R S f
  { norm_one := (congr_arg norm (map_one f)).trans norm_one }


instance toSeminormedRing [SeminormedRing R] [SubringClass S R] (s : S) : SeminormedRing s :=
  SeminormedRing.induced s R (SubringClass.subtype s)


instance toNormedRing [NormedRing R] [SubringClass S R] (s : S) : NormedRing s :=
  NormedRing.induced s R (SubringClass.subtype s) Subtype.val_injective


instance toSeminormedCommRing [SeminormedCommRing R] [_h : SubringClass S R] (s : S) :
    SeminormedCommRing s :=
  { SubringClass.toSeminormedRing s with mul_comm := mul_comm }


instance toNormedCommRing [NormedCommRing R] [SubringClass S R] (s : S) : NormedCommRing s :=
  { SubringClass.toNormedRing s with mul_comm := mul_comm }


instance toNormOneClass [SeminormedRing R] [NormOneClass R] [SubringClass S R] (s : S) :
    NormOneClass s :=
  .induced s R <| SubringClass.subtype _


/--
If `s` is a subfield of a normed field `F`, then `s` is equipped with an induced normed
field structure.
-/
instance toNormedField [NormedField F] [SubfieldClass S F] (s : S) : NormedField s :=
  NormedField.induced s F (SubringClass.subtype s) Subtype.val_injective


/-- A real absolute value on a ring determines a `NormedRing` structure. -/
noncomputable def toNormedRing {R : Type*} [Ring R] (v : AbsoluteValue R ℝ) : NormedRing R where
  norm := v
  dist_eq _ _ := rfl
                    /-
                      α : Type u_1
                      β : Type u_2
                      ι : Type u_3
                      R : Type u_4
                      inst✝ : Ring R
                      v : AbsoluteValue R Real
                      x : R
                      ⊢ Eq (Dist.dist x x) 0
                    -/
  dist_self x := by simp only [sub_self, MulHom.toFun_eq_coe, AbsoluteValue.coe_toMulHom, map_zero]
                    /-
                      🎉 no goals
                    -/
  dist_comm := v.map_sub
  dist_triangle := v.sub_le
  edist_dist x y := rfl
  norm_mul x y := (v.map_mul x y).le
  eq_of_dist_eq_zero := by simp only [MulHom.toFun_eq_coe, AbsoluteValue.coe_toMulHom,
    AbsoluteValue.map_sub_eq_zero_iff, imp_self, implies_true]


/-- A real absolute value on a field determines a `NormedField` structure. -/
noncomputable def toNormedField {K : Type*} [Field K] (v : AbsoluteValue K ℝ) : NormedField K where
  toField := inferInstanceAs (Field K)
  __ := v.toNormedRing
  norm_mul' := v.map_mul


