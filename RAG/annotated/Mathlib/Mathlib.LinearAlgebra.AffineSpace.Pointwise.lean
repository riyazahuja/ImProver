/-- The additive action on an affine subspace corresponding to applying the action to every element.

This is available as an instance in the `Pointwise` locale. -/
protected def pointwiseVAdd : VAdd V (AffineSubspace k P) where
  vadd x s := s.map (AffineEquiv.constVAdd k P x)


@[simp, norm_cast] lemma coe_pointwise_vadd (v : V) (s : AffineSubspace k P) :
    ((v +ᵥ s : AffineSubspace k P) : Set P) = v +ᵥ (s : Set P) := rfl


/-- The additive action on an affine subspace corresponding to applying the action to every element.

This is available as an instance in the `Pointwise` locale. -/
protected def pointwiseAddAction : AddAction V (AffineSubspace k P) :=
  SetLike.coe_injective.addAction _ coe_pointwise_vadd


theorem pointwise_vadd_eq_map (v : V) (s : AffineSubspace k P) :
    v +ᵥ s = s.map (AffineEquiv.constVAdd k P v) :=
  rfl


theorem vadd_mem_pointwise_vadd_iff {v : V} {s : AffineSubspace k P} {p : P} :
    v +ᵥ p ∈ v +ᵥ s ↔ p ∈ s :=
  vadd_mem_vadd_set_iff


@[simp] theorem pointwise_vadd_bot (v : V) : v +ᵥ (⊥ : AffineSubspace k P) = ⊥ := by
  /-
    k : Type u_2
    V : Type u_3
    P : Type u_4
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    v : V
    ⊢ Eq (HVAdd.hVAdd v Bot.bot) Bot.bot
  -/
  ext; simp [pointwise_vadd_eq_map, map_bot]
       /-
         🎉 no goals
       -/


@[simp] lemma pointwise_vadd_top (v : V) : v +ᵥ (⊤ : AffineSubspace k P) = ⊤ := by
  /-
    k : Type u_2
    V : Type u_3
    P : Type u_4
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    v : V
    ⊢ Eq (HVAdd.hVAdd v Top.top) Top.top
  -/
  ext; simp [pointwise_vadd_eq_map, map_top, vadd_eq_iff_eq_neg_vadd]
       /-
         🎉 no goals
       -/


theorem pointwise_vadd_direction (v : V) (s : AffineSubspace k P) :
    (v +ᵥ s).direction = s.direction := by
  /-
    k : Type u_2
    V : Type u_3
    P : Type u_4
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    v : V
    s : AffineSubspace k P
    ⊢ Eq (HVAdd.hVAdd v s).direction s.direction
  -/
  rw [pointwise_vadd_eq_map, map_direction]
  /-
    k : Type u_2
    V : Type u_3
    P : Type u_4
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    v : V
    s : AffineSubspace k P
    ⊢ Eq (Submodule.map (↑(AffineEquiv.constVAdd k P v)).linear s.direction) s.dir …
  -/
  exact Submodule.map_id _
  /-
    🎉 no goals
  -/


theorem pointwise_vadd_span (v : V) (s : Set P) : v +ᵥ affineSpan k s = affineSpan k (v +ᵥ s) :=
  map_span _ s


theorem map_pointwise_vadd (f : P₁ →ᵃ[k] P₂) (v : V₁) (s : AffineSubspace k P₁) :
    (v +ᵥ s).map f = f.linear v +ᵥ s.map f := by
  /-
    k : Type u_2
    V₁ : Type u_5
    P₁ : Type u_6
    V₂ : Type u_7
    P₂ : Type u_8
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    f : AffineMap k P₁ P₂
    v : V₁
    s : AffineSubspace k P₁
    ⊢ Eq (AffineSubspace.map f (HVAdd.hVAdd v s)) (HVAdd.hVAdd (f.linear v) (Affin …
  -/
  rw [pointwise_vadd_eq_map, pointwise_vadd_eq_map, map_map, map_map]
  /-
    k : Type u_2
    V₁ : Type u_5
    P₁ : Type u_6
    V₂ : Type u_7
    P₂ : Type u_8
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    f : AffineMap k P₁ P₂
    v : V₁
    s : AffineSubspace k P₁
    ⊢ Eq (AffineSubspace.map (f.comp ↑(AffineEquiv.constVAdd k P₁ v)) s) (AffineSu …
  -/
  congr 1
  /-
    case e_f
    k : Type u_2
    V₁ : Type u_5
    P₁ : Type u_6
    V₂ : Type u_7
    P₂ : Type u_8
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    f : AffineMap k P₁ P₂
    v : V₁
    s : AffineSubspace k P₁
    ⊢ Eq (f.comp ↑(AffineEquiv.constVAdd k P₁ v)) ((↑(AffineEquiv.constVAdd k P₂ ( …
  -/
  ext
  /-
    case e_f.h
    k : Type u_2
    V₁ : Type u_5
    P₁ : Type u_6
    V₂ : Type u_7
    P₂ : Type u_8
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    f : AffineMap k P₁ P₂
    v : V₁
    s : AffineSubspace k P₁
    p✝ : P₁
    ⊢ Eq ((f.comp ↑(AffineEquiv.constVAdd k P₁ v)) p✝) (((↑(AffineEquiv.constVAdd  …
  -/
  exact f.map_vadd _ _
  /-
    🎉 no goals
  -/


/-- The multiplicative action on an affine subspace corresponding to applying the action to every
element.

This is available as an instance in the `Pointwise` locale.

TODO: generalize to include `SMul (P ≃ᵃ[k] P) (AffineSubspace k P)`, which acts on `P` with a
`VAdd` version of a `DistribMulAction`. -/
protected def pointwiseSMul : SMul M (AffineSubspace k V) where
  smul a s := s.map (DistribMulAction.toLinearMap k _ a).toAffineMap


@[simp, norm_cast]
lemma coe_smul (a : M) (s : AffineSubspace k V) : ↑(a • s) = a • (s : Set V) := rfl


/-- The multiplicative action on an affine subspace corresponding to applying the action to every
element.

This is available as an instance in the `Pointwise` locale.

TODO: generalize to include `SMul (P ≃ᵃ[k] P) (AffineSubspace k P)`, which acts on `P` with a
`VAdd` version of a `DistribMulAction`. -/
protected def mulAction : MulAction M (AffineSubspace k V) :=
  SetLike.coe_injective.mulAction _ coe_smul


lemma smul_eq_map (a : M) (s : AffineSubspace k V) :
    a • s = s.map (DistribMulAction.toLinearMap k _ a).toAffineMap := rfl


lemma smul_mem_smul_iff {G : Type*} [Group G] [DistribMulAction G V] [SMulCommClass G k V] {a : G} :
    a • p ∈ a • s ↔ p ∈ s := smul_mem_smul_set_iff


lemma smul_mem_smul_iff_of_isUnit (ha : IsUnit a) : a • p ∈ a • s ↔ p ∈ s :=
  smul_mem_smul_iff (a := ha.unit)


lemma smul_mem_smul_iff₀ {G₀ : Type*} [GroupWithZero G₀] [DistribMulAction G₀ V]
    [SMulCommClass G₀ k V] {a : G₀} (ha : a ≠ 0) : a • p ∈ a • s ↔ p ∈ s :=
  smul_mem_smul_iff_of_isUnit ha.isUnit


@[simp] lemma smul_bot (a : M) : a • (⊥ : AffineSubspace k V) = ⊥ := by
  /-
    M : Type u_1
    k : Type u_2
    V : Type u_3
    inst✝⁵ : Ring k
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module k V
    inst✝² : Monoid M
    inst✝¹ : DistribMulAction M V
    inst✝ : SMulCommClass M k V
    a : M
    ⊢ Eq (HSMul.hSMul a Bot.bot) Bot.bot
  -/
  ext; simp [smul_eq_map, map_bot]
       /-
         🎉 no goals
       -/


@[simp] lemma smul_top (ha : IsUnit a) : a • (⊤ : AffineSubspace k V) = ⊤ := by
  /-
    M : Type u_1
    k : Type u_2
    V : Type u_3
    inst✝⁵ : Ring k
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module k V
    inst✝² : Monoid M
    inst✝¹ : DistribMulAction M V
    inst✝ : SMulCommClass M k V
    a : M
    ha : IsUnit a
    ⊢ Eq (HSMul.hSMul a Top.top) Top.top
  -/
  ext x; simpa [smul_eq_map, map_top] using ⟨ha.unit⁻¹ • x, smul_inv_smul ha.unit _⟩
         /-
           🎉 no goals
         -/


lemma smul_span (a : M) (s : Set V) : a • affineSpan k s = affineSpan k (a • s) := map_span _ s


@[simp]
lemma direction_smul (ha : a ≠ 0) (s : AffineSubspace k V) : (a • s).direction = s.direction := by
  have : DistribMulAction.toLinearMap k V a = a • LinearMap.id := by
    ext; simp
  /-
    k : Type u_2
    V : Type u_3
    inst✝² : Field k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    a : k
    ha : Ne a 0
    s : AffineSubspace k V
    this : Eq (DistribMulAction.toLinearMap k V a) (HSMul.hSMul a LinearMap.id)
    ⊢ Eq (HSMul.hSMul a s).direction s.direction
  -/
  simp [smul_eq_map, map_direction, this, Submodule.map_smul, ha]
  /-
    🎉 no goals
  -/


