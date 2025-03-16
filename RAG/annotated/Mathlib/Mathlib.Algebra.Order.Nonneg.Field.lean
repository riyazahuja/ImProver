lemma NNRat.cast_nonneg (q : ℚ≥0) : 0 ≤ (q : α) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    q : NNRat
    ⊢ LE.le 0 ↑q
  -/
  rw [cast_def]; exact div_nonneg q.num.cast_nonneg q.den.cast_nonneg
                 /-
                   🎉 no goals
                 -/


lemma nnqsmul_nonneg (q : ℚ≥0) (ha : 0 ≤ a) : 0 ≤ q • a := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    a : α
    q : NNRat
    ha : LE.le 0 a
    ⊢ LE.le 0 (HSMul.hSMul q a)
  -/
  rw [NNRat.smul_def]; exact mul_nonneg q.cast_nonneg ha
                       /-
                         🎉 no goals
                       -/


instance inv : Inv { x : α // 0 ≤ x } :=
  ⟨fun x => ⟨x⁻¹, inv_nonneg.2 x.2⟩⟩


@[simp, norm_cast]
protected theorem coe_inv (a : { x : α // 0 ≤ x }) : ((a⁻¹ : { x : α // 0 ≤ x }) : α) = (a : α)⁻¹ :=
  rfl


@[simp]
theorem inv_mk (hx : 0 ≤ x) :
    (⟨x, hx⟩ : { x : α // 0 ≤ x })⁻¹ = ⟨x⁻¹, inv_nonneg.2 hx⟩ :=
  rfl


instance div : Div { x : α // 0 ≤ x } :=
  ⟨fun x y => ⟨x / y, div_nonneg x.2 y.2⟩⟩


@[simp, norm_cast]
protected theorem coe_div (a b : { x : α // 0 ≤ x }) : ((a / b : { x : α // 0 ≤ x }) : α) = a / b :=
  rfl


@[simp]
theorem mk_div_mk (hx : 0 ≤ x) (hy : 0 ≤ y) :
    (⟨x, hx⟩ : { x : α // 0 ≤ x }) / ⟨y, hy⟩ = ⟨x / y, div_nonneg hx hy⟩ :=
  rfl


instance zpow : Pow { x : α // 0 ≤ x } ℤ :=
  ⟨fun a n => ⟨(a : α) ^ n, zpow_nonneg a.2 _⟩⟩


@[simp, norm_cast]
protected theorem coe_zpow (a : { x : α // 0 ≤ x }) (n : ℤ) :
    ((a ^ n : { x : α // 0 ≤ x }) : α) = (a : α) ^ n :=
  rfl


@[simp]
theorem mk_zpow (hx : 0 ≤ x) (n : ℤ) :
    (⟨x, hx⟩ : { x : α // 0 ≤ x }) ^ n = ⟨x ^ n, zpow_nonneg hx n⟩ :=
  rfl


instance instNNRatCast : NNRatCast {x : α // 0 ≤ x} := ⟨fun q ↦ ⟨q, q.cast_nonneg⟩⟩

instance instNNRatSMul : SMul ℚ≥0 {x : α // 0 ≤ x} where
                         /-
                           α : Type u_1
                           inst✝ : LinearOrderedSemifield α
                           x y : α
                           q : NNRat
                           a : Subtype fun x => LE.le 0 x
                           ⊢ LE.le 0 (HSMul.hSMul q ↑a)
                         -/
  smul q a := ⟨q • a, by rw [NNRat.smul_def]; exact mul_nonneg q.cast_nonneg a.2⟩
                                              /-
                                                🎉 no goals
                                              -/


@[simp, norm_cast] lemma coe_nnratCast (q : ℚ≥0) : (q : {x : α // 0 ≤ x}) = (q : α) := rfl

@[simp] lemma mk_nnratCast (q : ℚ≥0) : (⟨q, q.cast_nonneg⟩ : {x : α // 0 ≤ x}) = q := rfl


@[simp, norm_cast] lemma coe_nnqsmul (q : ℚ≥0) (a : {x : α // 0 ≤ x}) :
    ↑(q • a) = (q • a : α) := rfl

@[simp] lemma mk_nnqsmul (q : ℚ≥0) (a : α) (ha : 0 ≤ a) :
                /-
                  α : Type u_1
                  inst✝ : LinearOrderedSemifield α
                  x y : α
                  q : NNRat
                  a : α
                  ha : LE.le 0 a
                  ⊢ LE.le 0 (HSMul.hSMul q a)
                -/
    (⟨q • a, by rw [NNRat.smul_def]; exact mul_nonneg q.cast_nonneg ha⟩ : {x : α // 0 ≤ x}) =
                                     /-
                                       🎉 no goals
                                     -/
      q • a := rfl


instance linearOrderedSemifield : LinearOrderedSemifield { x : α // 0 ≤ x } :=
  Subtype.coe_injective.linearOrderedSemifield _ Nonneg.coe_zero Nonneg.coe_one Nonneg.coe_add
    Nonneg.coe_mul Nonneg.coe_inv Nonneg.coe_div (fun _ _ => rfl) coe_nnqsmul Nonneg.coe_pow
    Nonneg.coe_zpow Nonneg.coe_natCast coe_nnratCast (fun _ _ => rfl) fun _ _ => rfl


instance canonicallyLinearOrderedSemifield [LinearOrderedField α] :
    CanonicallyLinearOrderedSemifield { x : α // 0 ≤ x } :=
  { Nonneg.linearOrderedSemifield, Nonneg.canonicallyOrderedCommSemiring with }


instance linearOrderedCommGroupWithZero [LinearOrderedField α] :
    LinearOrderedCommGroupWithZero { x : α // 0 ≤ x } :=
  inferInstance


