instance : CoeFun { f : α → β | Continuous f } fun _ => α → β :=
  ⟨Subtype.val⟩


@[to_additive]
instance instMul [Mul β] [ContinuousMul β] : Mul C(α, β) :=
  ⟨fun f g => ⟨f * g, continuous_mul.comp (f.continuous.prod_mk g.continuous : _)⟩⟩


@[to_additive (attr := norm_cast, simp)]
theorem coe_mul [Mul β] [ContinuousMul β] (f g : C(α, β)) : ⇑(f * g) = f * g :=
  rfl


@[to_additive (attr := simp)]
theorem mul_apply [Mul β] [ContinuousMul β] (f g : C(α, β)) (x : α) : (f * g) x = f x * g x :=
  rfl


@[to_additive (attr := simp)]
theorem mul_comp [Mul γ] [ContinuousMul γ] (f₁ f₂ : C(β, γ)) (g : C(α, β)) :
    (f₁ * f₂).comp g = f₁.comp g * f₂.comp g :=
  rfl


@[to_additive]
instance [One β] : One C(α, β) :=
  ⟨const α 1⟩


@[to_additive (attr := norm_cast, simp)]
theorem coe_one [One β] : ⇑(1 : C(α, β)) = 1 :=
  rfl


@[to_additive (attr := simp)]
theorem one_apply [One β] (x : α) : (1 : C(α, β)) x = 1 :=
  rfl


@[to_additive (attr := simp)]
theorem one_comp [One γ] (g : C(α, β)) : (1 : C(β, γ)).comp g = 1 :=
  rfl


instance [NatCast β] : NatCast C(α, β) :=
  ⟨fun n => ContinuousMap.const _ n⟩


@[simp, norm_cast]
theorem coe_natCast [NatCast β] (n : ℕ) : ((n : C(α, β)) : α → β) = n :=
  rfl


@[deprecated (since := "2024-04-17")]
alias coe_nat_cast := coe_natCast


@[simp]
theorem natCast_apply [NatCast β] (n : ℕ) (x : α) : (n : C(α, β)) x = n :=
  rfl


@[deprecated (since := "2024-04-17")]
alias nat_cast_apply := natCast_apply


instance [IntCast β] : IntCast C(α, β) :=
  ⟨fun n => ContinuousMap.const _ n⟩


@[simp, norm_cast]
theorem coe_intCast [IntCast β] (n : ℤ) : ((n : C(α, β)) : α → β) = n :=
  rfl


@[deprecated (since := "2024-04-17")]
alias coe_int_cast := coe_intCast


@[simp]
theorem intCast_apply [IntCast β] (n : ℤ) (x : α) : (n : C(α, β)) x = n :=
  rfl


@[deprecated (since := "2024-04-17")]
alias int_cast_apply := intCast_apply


instance instNSMul [AddMonoid β] [ContinuousAdd β] : SMul ℕ C(α, β) :=
  ⟨fun n f => ⟨n • ⇑f, f.continuous.nsmul n⟩⟩


@[to_additive existing]
instance instPow [Monoid β] [ContinuousMul β] : Pow C(α, β) ℕ :=
  ⟨fun f n => ⟨(⇑f) ^ n, f.continuous.pow n⟩⟩


@[to_additive (attr := norm_cast) (reorder := 7 8)]
theorem coe_pow [Monoid β] [ContinuousMul β] (f : C(α, β)) (n : ℕ) : ⇑(f ^ n) = (⇑f) ^ n :=
  rfl


@[to_additive (attr := norm_cast)]
theorem pow_apply [Monoid β] [ContinuousMul β] (f : C(α, β)) (n : ℕ) (x : α) :
    (f ^ n) x = f x ^ n :=
  rfl

-- don't make auto-generated `coe_nsmul` and `nsmul_apply` simp, as the linter complains they're
-- redundant WRT `coe_smul`

@[to_additive]
theorem pow_comp [Monoid γ] [ContinuousMul γ] (f : C(β, γ)) (n : ℕ) (g : C(α, β)) :
    (f ^ n).comp g = f.comp g ^ n :=
  rfl

-- don't make `nsmul_comp` simp as the linter complains it's redundant WRT `smul_comp`

@[to_additive]
instance [Inv β] [ContinuousInv β] : Inv C(α, β) where inv f := ⟨f⁻¹, f.continuous.inv⟩


@[to_additive (attr := simp)]
theorem coe_inv [Inv β] [ContinuousInv β] (f : C(α, β)) : ⇑f⁻¹ = (⇑f)⁻¹ :=
  rfl


@[to_additive (attr := simp)]
theorem inv_apply [Inv β] [ContinuousInv β] (f : C(α, β)) (x : α) : f⁻¹ x = (f x)⁻¹ :=
  rfl


@[to_additive (attr := simp)]
theorem inv_comp [Inv γ] [ContinuousInv γ] (f : C(β, γ)) (g : C(α, β)) :
    f⁻¹.comp g = (f.comp g)⁻¹ :=
  rfl


@[to_additive]
instance [Div β] [ContinuousDiv β] : Div C(α, β) where
  div f g := ⟨f / g, f.continuous.div' g.continuous⟩


@[to_additive (attr := norm_cast, simp)]
theorem coe_div [Div β] [ContinuousDiv β] (f g : C(α, β)) : ⇑(f / g) = f / g :=
  rfl


@[to_additive (attr := simp)]
theorem div_apply [Div β] [ContinuousDiv β] (f g : C(α, β)) (x : α) : (f / g) x = f x / g x :=
  rfl


@[to_additive (attr := simp)]
theorem div_comp [Div γ] [ContinuousDiv γ] (f g : C(β, γ)) (h : C(α, β)) :
    (f / g).comp h = f.comp h / g.comp h :=
  rfl


instance instZSMul [AddGroup β] [TopologicalAddGroup β] : SMul ℤ C(α, β) where
  smul z f := ⟨z • ⇑f, f.continuous.zsmul z⟩


@[to_additive existing]
instance instZPow [Group β] [TopologicalGroup β] : Pow C(α, β) ℤ where
  pow f z := ⟨(⇑f) ^ z, f.continuous.zpow z⟩


@[to_additive (attr := norm_cast) (reorder := 7 8)]
theorem coe_zpow [Group β] [TopologicalGroup β] (f : C(α, β)) (z : ℤ) : ⇑(f ^ z) = (⇑f) ^ z :=
  rfl


@[to_additive]
theorem zpow_apply [Group β] [TopologicalGroup β] (f : C(α, β)) (z : ℤ) (x : α) :
    (f ^ z) x = f x ^ z :=
  rfl

-- don't make auto-generated `coe_zsmul` and `zsmul_apply` simp as the linter complains they're
-- redundant WRT `coe_smul`

@[to_additive]
theorem zpow_comp [Group γ] [TopologicalGroup γ] (f : C(β, γ)) (z : ℤ) (g : C(α, β)) :
    (f ^ z).comp g = f.comp g ^ z :=
  rfl

-- don't make `zsmul_comp` simp as the linter complains it's redundant WRT `smul_comp`

/-- The `Submonoid` of continuous maps `α → β`. -/
@[to_additive "The `AddSubmonoid` of continuous maps `α → β`. "]
def continuousSubmonoid (α : Type*) (β : Type*) [TopologicalSpace α] [TopologicalSpace β]
    [MulOneClass β] [ContinuousMul β] : Submonoid (α → β) where
  carrier := { f : α → β | Continuous f }
  one_mem' := @continuous_const _ _ _ _ 1
  mul_mem' fc gc := fc.mul gc


/-- The subgroup of continuous maps `α → β`. -/
@[to_additive "The `AddSubgroup` of continuous maps `α → β`. "]
def continuousSubgroup (α : Type*) (β : Type*) [TopologicalSpace α] [TopologicalSpace β] [Group β]
    [TopologicalGroup β] : Subgroup (α → β) :=
  { continuousSubmonoid α β with inv_mem' := fun fc => Continuous.inv fc }


@[to_additive]
instance [Semigroup β] [ContinuousMul β] : Semigroup C(α, β) :=
  coe_injective.semigroup _ coe_mul


@[to_additive]
instance [CommSemigroup β] [ContinuousMul β] : CommSemigroup C(α, β) :=
  coe_injective.commSemigroup _ coe_mul


@[to_additive]
instance [MulOneClass β] [ContinuousMul β] : MulOneClass C(α, β) :=
  coe_injective.mulOneClass _ coe_one coe_mul


instance [MulZeroClass β] [ContinuousMul β] : MulZeroClass C(α, β) :=
  coe_injective.mulZeroClass _ coe_zero coe_mul


instance [SemigroupWithZero β] [ContinuousMul β] : SemigroupWithZero C(α, β) :=
  coe_injective.semigroupWithZero _ coe_zero coe_mul


@[to_additive]
instance [Monoid β] [ContinuousMul β] : Monoid C(α, β) :=
  coe_injective.monoid _ coe_one coe_mul coe_pow


instance [MonoidWithZero β] [ContinuousMul β] : MonoidWithZero C(α, β) :=
  coe_injective.monoidWithZero _ coe_zero coe_one coe_mul coe_pow


@[to_additive]
instance [CommMonoid β] [ContinuousMul β] : CommMonoid C(α, β) :=
  coe_injective.commMonoid _ coe_one coe_mul coe_pow


instance [CommMonoidWithZero β] [ContinuousMul β] : CommMonoidWithZero C(α, β) :=
  coe_injective.commMonoidWithZero _ coe_zero coe_one coe_mul coe_pow


@[to_additive]
instance [LocallyCompactSpace α] [Mul β] [ContinuousMul β] : ContinuousMul C(α, β) :=
  ⟨by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : TopologicalSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : LocallyCompactSpace α
      inst✝¹ : Mul β
      inst✝ : ContinuousMul β
      ⊢ Continuous fun p => HMul.hMul p.1 p.2
    -/
    refine continuous_of_continuous_uncurry _ ?_
    have h1 : Continuous fun x : (C(α, β) × C(α, β)) × α => x.fst.fst x.snd :=
      continuous_eval.comp (continuous_fst.prodMap continuous_id)
    have h2 : Continuous fun x : (C(α, β) × C(α, β)) × α => x.fst.snd x.snd :=
      continuous_eval.comp (continuous_snd.prodMap continuous_id)
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁴ : TopologicalSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : LocallyCompactSpace α
      inst✝¹ : Mul β
      inst✝ : ContinuousMul β
      h1 : Continuous fun x => x.1.1 x.2
      h2 : Continuous fun x => x.1.2 x.2
      ⊢ Continuous (Function.uncurry fun x y => (HMul.hMul x.1 x.2) y)
    -/
    exact h1.mul h2⟩
    /-
      🎉 no goals
    -/


/-- Coercion to a function as a `MonoidHom`. Similar to `MonoidHom.coeFn`. -/
@[to_additive (attr := simps)
  "Coercion to a function as an `AddMonoidHom`. Similar to `AddMonoidHom.coeFn`." ]
def coeFnMonoidHom [Monoid β] [ContinuousMul β] : C(α, β) →* α → β where
  toFun f := f
  map_one' := coe_one
  map_mul' := coe_mul


/-- Composition on the left by a (continuous) homomorphism of topological monoids, as a
`MonoidHom`. Similar to `MonoidHom.compLeft`. -/
@[to_additive (attr := simps)
"Composition on the left by a (continuous) homomorphism of topological `AddMonoid`s, as an
`AddMonoidHom`. Similar to `AddMonoidHom.comp_left`."]
protected def _root_.MonoidHom.compLeftContinuous {γ : Type*} [Monoid β] [ContinuousMul β]
    [TopologicalSpace γ] [Monoid γ] [ContinuousMul γ] (g : β →* γ) (hg : Continuous g) :
    C(α, β) →* C(α, γ) where
  toFun f := (⟨g, hg⟩ : C(β, γ)).comp f
  map_one' := ext fun _ => g.map_one
  map_mul' _ _ := ext fun _ => g.map_mul _ _


/-- Composition on the right as a `MonoidHom`. Similar to `MonoidHom.compHom'`. -/
@[to_additive (attr := simps)
      "Composition on the right as an `AddMonoidHom`. Similar to `AddMonoidHom.compHom'`."]
def compMonoidHom' {γ : Type*} [TopologicalSpace γ] [MulOneClass γ] [ContinuousMul γ]
    (g : C(α, β)) : C(β, γ) →* C(α, γ) where
  toFun f := f.comp g
  map_one' := one_comp g
  map_mul' f₁ f₂ := mul_comp f₁ f₂ g


@[to_additive (attr := simp)]
theorem coe_prod [CommMonoid β] [ContinuousMul β] {ι : Type*} (s : Finset ι) (f : ι → C(α, β)) :
    ⇑(∏ i ∈ s, f i) = ∏ i ∈ s, (f i : α → β) :=
  map_prod coeFnMonoidHom f s


@[to_additive]
theorem prod_apply [CommMonoid β] [ContinuousMul β] {ι : Type*} (s : Finset ι) (f : ι → C(α, β))
                                                      /-
                                                        α : Type u_1
                                                        β : Type u_2
                                                        inst✝³ : TopologicalSpace α
                                                        inst✝² : TopologicalSpace β
                                                        inst✝¹ : CommMonoid β
                                                        inst✝ : ContinuousMul β
                                                        ι : Type u_3
                                                        s : Finset ι
                                                        f : ι → ContinuousMap α β
                                                        a : α
                                                        ⊢ Eq ((s.prod fun i => f i) a) (s.prod fun i => (f i) a)
                                                      -/
    (a : α) : (∏ i ∈ s, f i) a = ∏ i ∈ s, f i a := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


@[to_additive]
instance [Group β] [TopologicalGroup β] : Group C(α, β) :=
  coe_injective.group _ coe_one coe_mul coe_inv coe_div coe_pow coe_zpow


@[to_additive]
instance instCommGroupContinuousMap [CommGroup β] [TopologicalGroup β] : CommGroup C(α, β) :=
  coe_injective.commGroup _ coe_one coe_mul coe_inv coe_div coe_pow coe_zpow


@[to_additive]
instance [CommGroup β] [TopologicalGroup β] : TopologicalGroup C(α, β) where
  continuous_mul := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : CommGroup β
      inst✝ : TopologicalGroup β
      ⊢ Continuous fun p => HMul.hMul p.1 p.2
    -/
    letI : UniformSpace β := TopologicalGroup.toUniformSpace β
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : CommGroup β
      inst✝ : TopologicalGroup β
      this : UniformSpace β := TopologicalGroup.toUniformSpace β
      ⊢ Continuous fun p => HMul.hMul p.1 p.2
    -/
    have : UniformGroup β := comm_topologicalGroup_is_uniform
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : CommGroup β
      inst✝ : TopologicalGroup β
      this✝ : UniformSpace β := TopologicalGroup.toUniformSpace β
      this : UniformGroup β
      ⊢ Continuous fun p => HMul.hMul p.1 p.2
    -/
    rw [continuous_iff_continuousAt]
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : CommGroup β
      inst✝ : TopologicalGroup β
      this✝ : UniformSpace β := TopologicalGroup.toUniformSpace β
      this : UniformGroup β
      ⊢ ∀ (x : Prod (ContinuousMap α β) (ContinuousMap α β)), ContinuousAt (fun p => …
    -/
    rintro ⟨f, g⟩
    /-
      case mk
      α : Type u_1
      β : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : CommGroup β
      inst✝ : TopologicalGroup β
      this✝ : UniformSpace β := TopologicalGroup.toUniformSpace β
      this : UniformGroup β
      f g : ContinuousMap α β
      ⊢ ContinuousAt (fun p => HMul.hMul p.1 p.2) { fst := f, snd := g }
    -/
    rw [ContinuousAt, tendsto_iff_forall_isCompact_tendstoUniformlyOn, nhds_prod_eq]
    exact fun K hK =>
      uniformContinuous_mul.comp_tendstoUniformlyOn
        ((tendsto_iff_forall_isCompact_tendstoUniformlyOn.mp Filter.tendsto_id K hK).prod
          (tendsto_iff_forall_isCompact_tendstoUniformlyOn.mp Filter.tendsto_id K hK))
  continuous_inv := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : CommGroup β
      inst✝ : TopologicalGroup β
      ⊢ Continuous fun a => Inv.inv a
    -/
    letI : UniformSpace β := TopologicalGroup.toUniformSpace β
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : CommGroup β
      inst✝ : TopologicalGroup β
      this : UniformSpace β := TopologicalGroup.toUniformSpace β
      ⊢ Continuous fun a => Inv.inv a
    -/
    have : UniformGroup β := comm_topologicalGroup_is_uniform
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : CommGroup β
      inst✝ : TopologicalGroup β
      this✝ : UniformSpace β := TopologicalGroup.toUniformSpace β
      this : UniformGroup β
      ⊢ Continuous fun a => Inv.inv a
    -/
    rw [continuous_iff_continuousAt]
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : CommGroup β
      inst✝ : TopologicalGroup β
      this✝ : UniformSpace β := TopologicalGroup.toUniformSpace β
      this : UniformGroup β
      ⊢ ∀ (x : ContinuousMap α β), ContinuousAt (fun a => Inv.inv a) x
    -/
    intro f
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : TopologicalSpace β
      inst✝¹ : CommGroup β
      inst✝ : TopologicalGroup β
      this✝ : UniformSpace β := TopologicalGroup.toUniformSpace β
      this : UniformGroup β
      f : ContinuousMap α β
      ⊢ ContinuousAt (fun a => Inv.inv a) f
    -/
    rw [ContinuousAt, tendsto_iff_forall_isCompact_tendstoUniformlyOn]
    exact fun K hK =>
      uniformContinuous_inv.comp_tendstoUniformlyOn
        (tendsto_iff_forall_isCompact_tendstoUniformlyOn.mp Filter.tendsto_id K hK)


/-- If an infinite product of functions in `C(α, β)` converges to `g`
(for the compact-open topology), then the pointwise product converges to `g x` for all `x ∈ α`. -/
@[to_additive
  "If an infinite sum of functions in `C(α, β)` converges to `g` (for the compact-open topology),
then the pointwise sum converges to `g x` for all `x ∈ α`."]
theorem hasProd_apply {γ : Type*} [CommMonoid β] [ContinuousMul β]
    {f : γ → C(α, β)} {g : C(α, β)} (hf : HasProd f g) (x : α) :
    HasProd (fun i : γ => f i x) (g x) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    γ : Type u_3
    inst✝¹ : CommMonoid β
    inst✝ : ContinuousMul β
    f : γ → ContinuousMap α β
    g : ContinuousMap α β
    hf : HasProd f g
    x : α
    ⊢ HasProd (fun i => (f i) x) (g x)
  -/
  let ev : C(α, β) →* β := (Pi.evalMonoidHom _ x).comp coeFnMonoidHom
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    γ : Type u_3
    inst✝¹ : CommMonoid β
    inst✝ : ContinuousMul β
    f : γ → ContinuousMap α β
    g : ContinuousMap α β
    hf : HasProd f g
    x : α
    ev : MonoidHom (ContinuousMap α β) β := (Pi.evalMonoidHom (fun a => β) x).comp …
    ⊢ HasProd (fun i => (f i) x) (g x)
  -/
  exact hf.map ev (continuous_eval_const x)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem multipliable_apply [CommMonoid β] [ContinuousMul β] {γ : Type*} {f : γ → C(α, β)}
    (hf : Multipliable f) (x : α) : Multipliable fun i : γ => f i x :=
  (hasProd_apply hf.hasProd x).multipliable


@[to_additive]
theorem tprod_apply [T2Space β] [CommMonoid β] [ContinuousMul β] {γ : Type*} {f : γ → C(α, β)}
    (hf : Multipliable f) (x : α) :
    ∏' i : γ, f i x = (∏' i : γ, f i) x :=
  (hasProd_apply hf.hasProd x).tprod_eq


/-- The subsemiring of continuous maps `α → β`. -/
def continuousSubsemiring (α : Type*) (R : Type*) [TopologicalSpace α] [TopologicalSpace R]
    [NonAssocSemiring R] [TopologicalSemiring R] : Subsemiring (α → R) :=
  { continuousAddSubmonoid α R, continuousSubmonoid α R with }


/-- The subring of continuous maps `α → β`. -/
def continuousSubring (α : Type*) (R : Type*) [TopologicalSpace α] [TopologicalSpace R] [Ring R]
    [TopologicalRing R] : Subring (α → R) :=
  { continuousAddSubgroup α R, continuousSubsemiring α R with }


instance {α : Type*} {β : Type*} [TopologicalSpace α] [TopologicalSpace β]
    [NonUnitalNonAssocSemiring β] [TopologicalSemiring β] : NonUnitalNonAssocSemiring C(α, β) :=
  coe_injective.nonUnitalNonAssocSemiring _ coe_zero coe_add coe_mul coe_nsmul


instance {α : Type*} {β : Type*} [TopologicalSpace α] [TopologicalSpace β] [NonUnitalSemiring β]
    [TopologicalSemiring β] : NonUnitalSemiring C(α, β) :=
  coe_injective.nonUnitalSemiring _ coe_zero coe_add coe_mul coe_nsmul


instance {α : Type*} {β : Type*} [TopologicalSpace α] [TopologicalSpace β] [AddMonoidWithOne β]
    [ContinuousAdd β] : AddMonoidWithOne C(α, β) :=
  coe_injective.addMonoidWithOne _ coe_zero coe_one coe_add coe_nsmul coe_natCast


instance {α : Type*} {β : Type*} [TopologicalSpace α] [TopologicalSpace β] [NonAssocSemiring β]
    [TopologicalSemiring β] : NonAssocSemiring C(α, β) :=
  coe_injective.nonAssocSemiring _ coe_zero coe_one coe_add coe_mul coe_nsmul coe_natCast


instance {α : Type*} {β : Type*} [TopologicalSpace α] [TopologicalSpace β] [Semiring β]
    [TopologicalSemiring β] : Semiring C(α, β) :=
  coe_injective.semiring _ coe_zero coe_one coe_add coe_mul coe_nsmul coe_pow coe_natCast


instance {α : Type*} {β : Type*} [TopologicalSpace α] [TopologicalSpace β]
    [NonUnitalNonAssocRing β] [TopologicalRing β] : NonUnitalNonAssocRing C(α, β) :=
  coe_injective.nonUnitalNonAssocRing _ coe_zero coe_add coe_mul coe_neg coe_sub coe_nsmul coe_zsmul


instance {α : Type*} {β : Type*} [TopologicalSpace α] [TopologicalSpace β] [NonUnitalRing β]
    [TopologicalRing β] : NonUnitalRing C(α, β) :=
  coe_injective.nonUnitalRing _ coe_zero coe_add coe_mul coe_neg coe_sub coe_nsmul coe_zsmul


instance {α : Type*} {β : Type*} [TopologicalSpace α] [TopologicalSpace β] [NonAssocRing β]
    [TopologicalRing β] : NonAssocRing C(α, β) :=
  coe_injective.nonAssocRing _ coe_zero coe_one coe_add coe_mul coe_neg coe_sub coe_nsmul coe_zsmul
    coe_natCast coe_intCast


instance instRing {α : Type*} {β : Type*} [TopologicalSpace α] [TopologicalSpace β] [Ring β]
    [TopologicalRing β] : Ring C(α, β) :=
  coe_injective.ring _ coe_zero coe_one coe_add coe_mul coe_neg coe_sub coe_nsmul coe_zsmul coe_pow
    coe_natCast coe_intCast


instance {α : Type*} {β : Type*} [TopologicalSpace α] [TopologicalSpace β]
    [NonUnitalCommSemiring β] [TopologicalSemiring β] : NonUnitalCommSemiring C(α, β) :=
  coe_injective.nonUnitalCommSemiring _ coe_zero coe_add coe_mul coe_nsmul


instance {α : Type*} {β : Type*} [TopologicalSpace α] [TopologicalSpace β] [CommSemiring β]
    [TopologicalSemiring β] : CommSemiring C(α, β) :=
  coe_injective.commSemiring _ coe_zero coe_one coe_add coe_mul coe_nsmul coe_pow coe_natCast


instance {α : Type*} {β : Type*} [TopologicalSpace α] [TopologicalSpace β] [NonUnitalCommRing β]
    [TopologicalRing β] : NonUnitalCommRing C(α, β) :=
  coe_injective.nonUnitalCommRing _ coe_zero coe_add coe_mul coe_neg coe_sub coe_nsmul coe_zsmul


instance {α : Type*} {β : Type*} [TopologicalSpace α] [TopologicalSpace β] [CommRing β]
    [TopologicalRing β] : CommRing C(α, β) :=
  coe_injective.commRing _ coe_zero coe_one coe_add coe_mul coe_neg coe_sub coe_nsmul coe_zsmul
    coe_pow coe_natCast coe_intCast


instance {α : Type*} {β : Type*} [TopologicalSpace α] [TopologicalSpace β] [LocallyCompactSpace α]
    [NonUnitalSemiring β] [TopologicalSemiring β] : TopologicalSemiring C(α, β) where


instance {α : Type*} {β : Type*} [TopologicalSpace α] [TopologicalSpace β] [LocallyCompactSpace α]
    [NonUnitalRing β] [TopologicalRing β] : TopologicalRing C(α, β) where


/-- Composition on the left by a (continuous) homomorphism of topological semirings, as a
`RingHom`.  Similar to `RingHom.compLeft`. -/
@[simps!]
protected def _root_.RingHom.compLeftContinuous (α : Type*) {β : Type*} {γ : Type*}
    [TopologicalSpace α]
    [TopologicalSpace β] [Semiring β] [TopologicalSemiring β] [TopologicalSpace γ] [Semiring γ]
    [TopologicalSemiring γ] (g : β →+* γ) (hg : Continuous g) : C(α, β) →+* C(α, γ) :=
  { g.toMonoidHom.compLeftContinuous α hg, g.toAddMonoidHom.compLeftContinuous α hg with }


/-- Coercion to a function as a `RingHom`. -/
@[simps!]
def coeFnRingHom {α : Type*} {β : Type*} [TopologicalSpace α] [TopologicalSpace β] [Semiring β]
    [TopologicalSemiring β] : C(α, β) →+* α → β :=
  { (coeFnMonoidHom : C(α, β) →* _),
    (coeFnAddMonoidHom : C(α, β) →+ _) with }


/-- The `R`-submodule of continuous maps `α → M`. -/
def continuousSubmodule : Submodule R (α → M) :=
  { continuousAddSubgroup α M with
    carrier := { f : α → M | Continuous f }
    smul_mem' := fun c _ hf => hf.const_smul c }


@[to_additive]
instance instSMul [SMul R M] [ContinuousConstSMul R M] : SMul R C(α, M) :=
  ⟨fun r f => ⟨r • ⇑f, f.continuous.const_smul r⟩⟩


@[to_additive]
instance [LocallyCompactSpace α] [SMul R M] [ContinuousConstSMul R M] :
    ContinuousConstSMul R C(α, M) :=
  ⟨fun γ => continuous_of_continuous_uncurry _ (continuous_eval.const_smul γ)⟩


@[to_additive]
instance [LocallyCompactSpace α] [TopologicalSpace R] [SMul R M] [ContinuousSMul R M] :
    ContinuousSMul R C(α, M) :=
  ⟨by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : TopologicalSpace β
      R : Type u_3
      R₁ : Type u_4
      M : Type u_5
      inst✝⁵ : TopologicalSpace M
      M₂ : Type u_6
      inst✝⁴ : TopologicalSpace M₂
      inst✝³ : LocallyCompactSpace α
      inst✝² : TopologicalSpace R
      inst✝¹ : SMul R M
      inst✝ : ContinuousSMul R M
      ⊢ Continuous fun p => HSMul.hSMul p.1 p.2
    -/
    refine continuous_of_continuous_uncurry _ ?_
    have h : Continuous fun x : (R × C(α, M)) × α => x.fst.snd x.snd :=
      continuous_eval.comp (continuous_snd.prodMap continuous_id)
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : TopologicalSpace β
      R : Type u_3
      R₁ : Type u_4
      M : Type u_5
      inst✝⁵ : TopologicalSpace M
      M₂ : Type u_6
      inst✝⁴ : TopologicalSpace M₂
      inst✝³ : LocallyCompactSpace α
      inst✝² : TopologicalSpace R
      inst✝¹ : SMul R M
      inst✝ : ContinuousSMul R M
      h : Continuous fun x => x.1.2 x.2
      ⊢ Continuous (Function.uncurry fun x y => (HSMul.hSMul x.1 x.2) y)
    -/
    exact (continuous_fst.comp continuous_fst).smul h⟩
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp, norm_cast)]
theorem coe_smul [SMul R M] [ContinuousConstSMul R M] (c : R) (f : C(α, M)) : ⇑(c • f) = c • ⇑f :=
  rfl


@[to_additive]
theorem smul_apply [SMul R M] [ContinuousConstSMul R M] (c : R) (f : C(α, M)) (a : α) :
    (c • f) a = c • f a :=
  rfl


@[to_additive (attr := simp)]
theorem smul_comp [SMul R M] [ContinuousConstSMul R M] (r : R) (f : C(β, M)) (g : C(α, β)) :
    (r • f).comp g = r • f.comp g :=
  rfl


@[to_additive]
instance [SMul R M] [ContinuousConstSMul R M] [SMul R₁ M] [ContinuousConstSMul R₁ M]
    [SMulCommClass R R₁ M] : SMulCommClass R R₁ C(α, M) where
  smul_comm _ _ _ := ext fun _ => smul_comm _ _ _


instance [SMul R M] [ContinuousConstSMul R M] [SMul R₁ M] [ContinuousConstSMul R₁ M] [SMul R R₁]
    [IsScalarTower R R₁ M] : IsScalarTower R R₁ C(α, M) where
  smul_assoc _ _ _ := ext fun _ => smul_assoc _ _ _


instance [SMul R M] [SMul Rᵐᵒᵖ M] [ContinuousConstSMul R M] [IsCentralScalar R M] :
    IsCentralScalar R C(α, M) where op_smul_eq_smul _ _ := ext fun _ => op_smul_eq_smul _ _


instance [SMul R M] [ContinuousConstSMul R M] [Mul M] [ContinuousMul M] [IsScalarTower R M M] :
    IsScalarTower R C(α, M) C(α, M) where
  smul_assoc _ _ _ := ext fun _ => smul_mul_assoc ..


instance [SMul R M] [ContinuousConstSMul R M] [Mul M] [ContinuousMul M] [SMulCommClass R M M] :
    SMulCommClass R C(α, M) C(α, M) where
  smul_comm _ _ _ := ext fun _ => (mul_smul_comm ..).symm


instance [SMul R M] [ContinuousConstSMul R M] [Mul M] [ContinuousMul M] [SMulCommClass M R M] :
    SMulCommClass C(α, M) R C(α, M) where
  smul_comm _ _ _ := ext fun _ => smul_comm (_ : M) ..


instance [Monoid R] [MulAction R M] [ContinuousConstSMul R M] : MulAction R C(α, M) :=
  Function.Injective.mulAction _ coe_injective coe_smul


instance [Monoid R] [AddMonoid M] [DistribMulAction R M] [ContinuousAdd M]
    [ContinuousConstSMul R M] : DistribMulAction R C(α, M) :=
  Function.Injective.distribMulAction coeFnAddMonoidHom coe_injective coe_smul


instance module : Module R C(α, M) :=
  Function.Injective.module R coeFnAddMonoidHom coe_injective coe_smul


/-- Composition on the left by a continuous linear map, as a `LinearMap`.
Similar to `LinearMap.compLeft`. -/
@[simps]
protected def _root_.ContinuousLinearMap.compLeftContinuous (α : Type*) [TopologicalSpace α]
    (g : M →L[R] M₂) : C(α, M) →ₗ[R] C(α, M₂) :=
  { g.toLinearMap.toAddMonoidHom.compLeftContinuous α g.continuous with
    map_smul' := fun c _ => ext fun _ => g.map_smul' c _ }


/-- Coercion to a function as a `LinearMap`. -/
@[simps]
def coeFnLinearMap : C(α, M) →ₗ[R] α → M :=
  { (coeFnAddMonoidHom : C(α, M) →+ _) with
    map_smul' := coe_smul }


/-- Evaluation at a point, as a continuous linear map. -/
@[simps apply]
def evalCLM (x : α) : C(α, M) →L[R] M where
  toFun f := f x
  map_add' _ _ := add_apply _ _ x
  map_smul' _ _ := smul_apply _ _ x


/-- The `R`-subalgebra of continuous maps `α → A`. -/
def continuousSubalgebra : Subalgebra R (α → A) :=
  { continuousSubsemiring α A with
    carrier := { f : α → A | Continuous f }
    algebraMap_mem' := fun r => (continuous_const : Continuous fun _ : α => algebraMap R A r) }


/-- Continuous constant functions as a `RingHom`. -/
def ContinuousMap.C : R →+* C(α, A) where
  toFun := fun c : R => ⟨fun _ : α => (algebraMap R A) c, continuous_const⟩
                 /-
                   α : Type u_1
                   inst✝⁹ : TopologicalSpace α
                   R : Type u_2
                   inst✝⁸ : CommSemiring R
                   A : Type u_3
                   inst✝⁷ : TopologicalSpace A
                   inst✝⁶ : Semiring A
                   inst✝⁵ : Algebra R A
                   inst✝⁴ : TopologicalSemiring A
                   A₂ : Type u_4
                   inst✝³ : TopologicalSpace A₂
                   inst✝² : Semiring A₂
                   inst✝¹ : Algebra R A₂
                   inst✝ : TopologicalSemiring A₂
                   ⊢ Eq ((fun c => { toFun := fun x => (algebraMap R A) c, continuous_toFun := ⋯  …
                 -/
  map_one' := by ext _; exact (algebraMap R A).map_one
                        /-
                          🎉 no goals
                        -/
                       /-
                         α : Type u_1
                         inst✝⁹ : TopologicalSpace α
                         R : Type u_2
                         inst✝⁸ : CommSemiring R
                         A : Type u_3
                         inst✝⁷ : TopologicalSpace A
                         inst✝⁶ : Semiring A
                         inst✝⁵ : Algebra R A
                         inst✝⁴ : TopologicalSemiring A
                         A₂ : Type u_4
                         inst✝³ : TopologicalSpace A₂
                         inst✝² : Semiring A₂
                         inst✝¹ : Algebra R A₂
                         inst✝ : TopologicalSemiring A₂
                         c₁ c₂ : R
                         ⊢ Eq ({ toFun := fun c => { toFun := fun x => (algebraMap R A) c, continuous_t …
                       -/
  map_mul' c₁ c₂ := by ext _; exact (algebraMap R A).map_mul _ _
                              /-
                                🎉 no goals
                              -/
                  /-
                    α : Type u_1
                    inst✝⁹ : TopologicalSpace α
                    R : Type u_2
                    inst✝⁸ : CommSemiring R
                    A : Type u_3
                    inst✝⁷ : TopologicalSpace A
                    inst✝⁶ : Semiring A
                    inst✝⁵ : Algebra R A
                    inst✝⁴ : TopologicalSemiring A
                    A₂ : Type u_4
                    inst✝³ : TopologicalSpace A₂
                    inst✝² : Semiring A₂
                    inst✝¹ : Algebra R A₂
                    inst✝ : TopologicalSemiring A₂
                    ⊢ Eq ((↑{ toFun := fun c => { toFun := fun x => (algebraMap R A) c, continuous …
                  -/
  map_zero' := by ext _; exact (algebraMap R A).map_zero
                         /-
                           🎉 no goals
                         -/
                       /-
                         α : Type u_1
                         inst✝⁹ : TopologicalSpace α
                         R : Type u_2
                         inst✝⁸ : CommSemiring R
                         A : Type u_3
                         inst✝⁷ : TopologicalSpace A
                         inst✝⁶ : Semiring A
                         inst✝⁵ : Algebra R A
                         inst✝⁴ : TopologicalSemiring A
                         A₂ : Type u_4
                         inst✝³ : TopologicalSpace A₂
                         inst✝² : Semiring A₂
                         inst✝¹ : Algebra R A₂
                         inst✝ : TopologicalSemiring A₂
                         c₁ c₂ : R
                         ⊢ Eq ((↑{ toFun := fun c => { toFun := fun x => (algebraMap R A) c, continuous …
                       -/
  map_add' c₁ c₂ := by ext _; exact (algebraMap R A).map_add _ _
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem ContinuousMap.C_apply (r : R) (a : α) : ContinuousMap.C r a = algebraMap R A r :=
  rfl


instance ContinuousMap.algebra : Algebra R C(α, A) where
  toRingHom := ContinuousMap.C
                      /-
                        α : Type u_1
                        inst✝⁹ : TopologicalSpace α
                        R : Type u_2
                        inst✝⁸ : CommSemiring R
                        A : Type u_3
                        inst✝⁷ : TopologicalSpace A
                        inst✝⁶ : Semiring A
                        inst✝⁵ : Algebra R A
                        inst✝⁴ : TopologicalSemiring A
                        A₂ : Type u_4
                        inst✝³ : TopologicalSpace A₂
                        inst✝² : Semiring A₂
                        inst✝¹ : Algebra R A₂
                        inst✝ : TopologicalSemiring A₂
                        c : R
                        f : ContinuousMap α A
                        ⊢ Eq (HMul.hMul (ContinuousMap.C c) f) (HMul.hMul f (ContinuousMap.C c))
                      -/
  commutes' c f := by ext x; exact Algebra.commutes' _ _
                             /-
                               🎉 no goals
                             -/
                      /-
                        α : Type u_1
                        inst✝⁹ : TopologicalSpace α
                        R : Type u_2
                        inst✝⁸ : CommSemiring R
                        A : Type u_3
                        inst✝⁷ : TopologicalSpace A
                        inst✝⁶ : Semiring A
                        inst✝⁵ : Algebra R A
                        inst✝⁴ : TopologicalSemiring A
                        A₂ : Type u_4
                        inst✝³ : TopologicalSpace A₂
                        inst✝² : Semiring A₂
                        inst✝¹ : Algebra R A₂
                        inst✝ : TopologicalSemiring A₂
                        c : R
                        f : ContinuousMap α A
                        ⊢ Eq (HSMul.hSMul c f) (HMul.hMul (ContinuousMap.C c) f)
                      -/
  smul_def' c f := by ext x; exact Algebra.smul_def' _ _
                             /-
                               🎉 no goals
                             -/


/-- Composition on the left by a (continuous) homomorphism of topological `R`-algebras, as an
`AlgHom`. Similar to `AlgHom.compLeft`. -/
@[simps!]
protected def AlgHom.compLeftContinuous {α : Type*} [TopologicalSpace α] (g : A →ₐ[R] A₂)
    (hg : Continuous g) : C(α, A) →ₐ[R] C(α, A₂) :=
  { g.toRingHom.compLeftContinuous α hg with
    commutes' := fun _ => ContinuousMap.ext fun _ => g.commutes' _ }


/-- Precomposition of functions into a topological semiring by a continuous map is an algebra
homomorphism. -/
@[simps]
def ContinuousMap.compRightAlgHom {α β : Type*} [TopologicalSpace α] [TopologicalSpace β]
    (f : C(α, β)) : C(β, A) →ₐ[R] C(α, A) where
  toFun g := g.comp f
  map_zero' := ext fun _ ↦ rfl
  map_add'  _ _ := ext fun _ ↦ rfl
  map_one' := ext fun _ ↦ rfl
  map_mul' _ _ := ext fun _ ↦ rfl
  commutes' _ := ext fun _ ↦ rfl


theorem ContinuousMap.compRightAlgHom_continuous {α β : Type*} [TopologicalSpace α]
    [TopologicalSpace β] (f : C(α, β)) : Continuous (compRightAlgHom R A f) :=
  continuous_precomp f


/-- Coercion to a function as an `AlgHom`. -/
@[simps!]
def ContinuousMap.coeFnAlgHom : C(α, A) →ₐ[R] α → A :=
  { (ContinuousMap.coeFnRingHom : C(α, A) →+* _) with
    commutes' := fun _ => rfl }


/-- A version of `Set.SeparatesPoints` for subalgebras of the continuous functions,
used for stating the Stone-Weierstrass theorem.
-/
abbrev Subalgebra.SeparatesPoints (s : Subalgebra R C(α, A)) : Prop :=
  Set.SeparatesPoints ((fun f : C(α, A) => (f : α → A)) '' (s : Set C(α, A)))


theorem Subalgebra.separatesPoints_monotone :
    Monotone fun s : Subalgebra R C(α, A) => s.SeparatesPoints := fun s s' r h x y n => by
  /-
    α : Type u_1
    inst✝⁵ : TopologicalSpace α
    R : Type u_2
    inst✝⁴ : CommSemiring R
    A : Type u_3
    inst✝³ : TopologicalSpace A
    inst✝² : Semiring A
    inst✝¹ : Algebra R A
    inst✝ : TopologicalSemiring A
    s s' : Subalgebra R (ContinuousMap α A)
    r : LE.le s s'
    h : (fun s => s.SeparatesPoints) s
    x y : α
    n : Ne x y
    ⊢ Exists fun f => And (Membership.mem (Set.image (fun f => ⇑f) ↑s') f) (Ne (f  …
  -/
  obtain ⟨f, m, w⟩ := h n
  /-
    case intro.intro
    α : Type u_1
    inst✝⁵ : TopologicalSpace α
    R : Type u_2
    inst✝⁴ : CommSemiring R
    A : Type u_3
    inst✝³ : TopologicalSpace A
    inst✝² : Semiring A
    inst✝¹ : Algebra R A
    inst✝ : TopologicalSemiring A
    s s' : Subalgebra R (ContinuousMap α A)
    r : LE.le s s'
    h : (fun s => s.SeparatesPoints) s
    x y : α
    n : Ne x y
    f : α → A
    m : Membership.mem (Set.image (fun f => ⇑f) ↑s) f
    w : Ne (f x) (f y)
    ⊢ Exists fun f => And (Membership.mem (Set.image (fun f => ⇑f) ↑s') f) (Ne (f  …
  -/
  rcases m with ⟨f, ⟨m, rfl⟩⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝⁵ : TopologicalSpace α
    R : Type u_2
    inst✝⁴ : CommSemiring R
    A : Type u_3
    inst✝³ : TopologicalSpace A
    inst✝² : Semiring A
    inst✝¹ : Algebra R A
    inst✝ : TopologicalSemiring A
    s s' : Subalgebra R (ContinuousMap α A)
    r : LE.le s s'
    h : (fun s => s.SeparatesPoints) s
    x y : α
    n : Ne x y
    f : ContinuousMap α A
    m : Membership.mem (↑s) f
    w : Ne ((fun f => ⇑f) f x) ((fun f => ⇑f) f y)
    ⊢ Exists fun f => And (Membership.mem (Set.image (fun f => ⇑f) ↑s') f) (Ne (f  …
  -/
  exact ⟨_, ⟨f, ⟨r m, rfl⟩⟩, w⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem algebraMap_apply (k : R) (a : α) : algebraMap R C(α, A) k a = k • (1 : A) := by
  /-
    α : Type u_1
    inst✝⁵ : TopologicalSpace α
    R : Type u_2
    inst✝⁴ : CommSemiring R
    A : Type u_3
    inst✝³ : TopologicalSpace A
    inst✝² : Semiring A
    inst✝¹ : Algebra R A
    inst✝ : TopologicalSemiring A
    k : R
    a : α
    ⊢ Eq (((algebraMap R (ContinuousMap α A)) k) a) (HSMul.hSMul k 1)
  -/
  rw [Algebra.algebraMap_eq_smul_one]
  /-
    α : Type u_1
    inst✝⁵ : TopologicalSpace α
    R : Type u_2
    inst✝⁴ : CommSemiring R
    A : Type u_3
    inst✝³ : TopologicalSpace A
    inst✝² : Semiring A
    inst✝¹ : Algebra R A
    inst✝ : TopologicalSemiring A
    k : R
    a : α
    ⊢ Eq ((HSMul.hSMul k 1) a) (HSMul.hSMul k 1)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A set of continuous maps "separates points strongly"
if for each pair of distinct points there is a function with specified values on them.

We give a slightly unusual formulation, where the specified values are given by some
function `v`, and we ask `f x = v x ∧ f y = v y`. This avoids needing a hypothesis `x ≠ y`.

In fact, this definition would work perfectly well for a set of non-continuous functions,
but as the only current use case is in the Stone-Weierstrass theorem,
writing it this way avoids having to deal with casts inside the set.
(This may need to change if we do Stone-Weierstrass on non-compact spaces,
where the functions would be continuous functions vanishing at infinity.)
-/
def Set.SeparatesPointsStrongly (s : Set C(α, 𝕜)) : Prop :=
  ∀ (v : α → 𝕜) (x y : α), ∃ f ∈ s, (f x : 𝕜) = v x ∧ f y = v y


/-- Working in continuous functions into a topological field,
a subalgebra of functions that separates points also separates points strongly.

By the hypothesis, we can find a function `f` so `f x ≠ f y`.
By an affine transformation in the field we can arrange so that `f x = a` and `f x = b`.
-/
theorem Subalgebra.SeparatesPoints.strongly {s : Subalgebra 𝕜 C(α, 𝕜)} (h : s.SeparatesPoints) :
    (s : Set C(α, 𝕜)).SeparatesPointsStrongly := fun v x y => by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    𝕜 : Type u_5
    inst✝² : TopologicalSpace 𝕜
    inst✝¹ : Field 𝕜
    inst✝ : TopologicalRing 𝕜
    s : Subalgebra 𝕜 (ContinuousMap α 𝕜)
    h : s.SeparatesPoints
    v : α → 𝕜
    x y : α
    ⊢ Exists fun f => And (Membership.mem (↑s) f) (And (Eq (f x) (v x)) (Eq (f y)  …
  -/
  by_cases n : x = y
    /-
      case pos
      α : Type u_1
      inst✝³ : TopologicalSpace α
      𝕜 : Type u_5
      inst✝² : TopologicalSpace 𝕜
      inst✝¹ : Field 𝕜
      inst✝ : TopologicalRing 𝕜
      s : Subalgebra 𝕜 (ContinuousMap α 𝕜)
      h : s.SeparatesPoints
      v : α → 𝕜
      x y : α
      n : Eq x y
      ⊢ Exists fun f => And (Membership.mem (↑s) f) (And (Eq (f x) (v x)) (Eq (f y)  …
    -/
  · subst n
    /-
      case pos
      α : Type u_1
      inst✝³ : TopologicalSpace α
      𝕜 : Type u_5
      inst✝² : TopologicalSpace 𝕜
      inst✝¹ : Field 𝕜
      inst✝ : TopologicalRing 𝕜
      s : Subalgebra 𝕜 (ContinuousMap α 𝕜)
      h : s.SeparatesPoints
      v : α → 𝕜
      x : α
      ⊢ Exists fun f => And (Membership.mem (↑s) f) (And (Eq (f x) (v x)) (Eq (f x)  …
    -/
    exact ⟨_, (v x • (1 : s) : s).prop, mul_one _, mul_one _⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝³ : TopologicalSpace α
    𝕜 : Type u_5
    inst✝² : TopologicalSpace 𝕜
    inst✝¹ : Field 𝕜
    inst✝ : TopologicalRing 𝕜
    s : Subalgebra 𝕜 (ContinuousMap α 𝕜)
    h : s.SeparatesPoints
    v : α → 𝕜
    x y : α
    n : Not (Eq x y)
    ⊢ Exists fun f => And (Membership.mem (↑s) f) (And (Eq (f x) (v x)) (Eq (f y)  …
  -/
  obtain ⟨_, ⟨f, hf, rfl⟩, hxy⟩ := h n
  /-
    case neg.intro.intro.intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    𝕜 : Type u_5
    inst✝² : TopologicalSpace 𝕜
    inst✝¹ : Field 𝕜
    inst✝ : TopologicalRing 𝕜
    s : Subalgebra 𝕜 (ContinuousMap α 𝕜)
    h : s.SeparatesPoints
    v : α → 𝕜
    x y : α
    n : Not (Eq x y)
    f : ContinuousMap α 𝕜
    hf : Membership.mem (↑s) f
    hxy : Ne ((fun f => ⇑f) f x) ((fun f => ⇑f) f y)
    ⊢ Exists fun f => And (Membership.mem (↑s) f) (And (Eq (f x) (v x)) (Eq (f y)  …
  -/
  replace hxy : f x - f y ≠ 0 := sub_ne_zero_of_ne hxy
  /-
    case neg.intro.intro.intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    𝕜 : Type u_5
    inst✝² : TopologicalSpace 𝕜
    inst✝¹ : Field 𝕜
    inst✝ : TopologicalRing 𝕜
    s : Subalgebra 𝕜 (ContinuousMap α 𝕜)
    h : s.SeparatesPoints
    v : α → 𝕜
    x y : α
    n : Not (Eq x y)
    f : ContinuousMap α 𝕜
    hf : Membership.mem (↑s) f
    hxy : Ne (HSub.hSub (f x) (f y)) 0
    ⊢ Exists fun f => And (Membership.mem (↑s) f) (And (Eq (f x) (v x)) (Eq (f y)  …
  -/
  let a := v x
  /-
    case neg.intro.intro.intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    𝕜 : Type u_5
    inst✝² : TopologicalSpace 𝕜
    inst✝¹ : Field 𝕜
    inst✝ : TopologicalRing 𝕜
    s : Subalgebra 𝕜 (ContinuousMap α 𝕜)
    h : s.SeparatesPoints
    v : α → 𝕜
    x y : α
    n : Not (Eq x y)
    f : ContinuousMap α 𝕜
    hf : Membership.mem (↑s) f
    hxy : Ne (HSub.hSub (f x) (f y)) 0
    a : 𝕜 := v x
    ⊢ Exists fun f => And (Membership.mem (↑s) f) (And (Eq (f x) (v x)) (Eq (f y)  …
  -/
  let b := v y
  let f' : s :=
    ((b - a) * (f x - f y)⁻¹) • (algebraMap _ s (f x) - (⟨f, hf⟩ : s)) + algebraMap _ s a
  /-
    case neg.intro.intro.intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    𝕜 : Type u_5
    inst✝² : TopologicalSpace 𝕜
    inst✝¹ : Field 𝕜
    inst✝ : TopologicalRing 𝕜
    s : Subalgebra 𝕜 (ContinuousMap α 𝕜)
    h : s.SeparatesPoints
    v : α → 𝕜
    x y : α
    n : Not (Eq x y)
    f : ContinuousMap α 𝕜
    hf : Membership.mem (↑s) f
    hxy : Ne (HSub.hSub (f x) (f y)) 0
    a : 𝕜 := v x
    b : 𝕜 := v y
    f' : Subtype fun x => Membership.mem s x := HAdd.hAdd (HSMul.hSMul (HMul.hMul  …
    ⊢ Exists fun f => And (Membership.mem (↑s) f) (And (Eq (f x) (v x)) (Eq (f y)  …
  -/
  refine ⟨f', f'.prop, ?_, ?_⟩
    /-
      case neg.intro.intro.intro.intro.refine_1
      α : Type u_1
      inst✝³ : TopologicalSpace α
      𝕜 : Type u_5
      inst✝² : TopologicalSpace 𝕜
      inst✝¹ : Field 𝕜
      inst✝ : TopologicalRing 𝕜
      s : Subalgebra 𝕜 (ContinuousMap α 𝕜)
      h : s.SeparatesPoints
      v : α → 𝕜
      x y : α
      n : Not (Eq x y)
      f : ContinuousMap α 𝕜
      hf : Membership.mem (↑s) f
      hxy : Ne (HSub.hSub (f x) (f y)) 0
      a : 𝕜 := v x
      b : 𝕜 := v y
      f' : Subtype fun x => Membership.mem s x := HAdd.hAdd (HSMul.hSMul (HMul.hMul  …
      ⊢ Eq (↑f' x) (v x)
    -/
  · simp [a, b, f']
    /-
      🎉 no goals
    -/
    /-
      case neg.intro.intro.intro.intro.refine_2
      α : Type u_1
      inst✝³ : TopologicalSpace α
      𝕜 : Type u_5
      inst✝² : TopologicalSpace 𝕜
      inst✝¹ : Field 𝕜
      inst✝ : TopologicalRing 𝕜
      s : Subalgebra 𝕜 (ContinuousMap α 𝕜)
      h : s.SeparatesPoints
      v : α → 𝕜
      x y : α
      n : Not (Eq x y)
      f : ContinuousMap α 𝕜
      hf : Membership.mem (↑s) f
      hxy : Ne (HSub.hSub (f x) (f y)) 0
      a : 𝕜 := v x
      b : 𝕜 := v y
      f' : Subtype fun x => Membership.mem s x := HAdd.hAdd (HSMul.hSMul (HMul.hMul  …
      ⊢ Eq (↑f' y) (v y)
    -/
  · simp [a, b, f', inv_mul_cancel_right₀ hxy]
    /-
      🎉 no goals
    -/


instance ContinuousMap.subsingleton_subalgebra (α : Type*) [TopologicalSpace α] (R : Type*)
    [CommSemiring R] [TopologicalSpace R] [TopologicalSemiring R] [Subsingleton α] :
    Subsingleton (Subalgebra R C(α, R)) :=
  ⟨fun s₁ s₂ => by
    /-
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      R : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : TopologicalSpace R
      inst✝¹ : TopologicalSemiring R
      inst✝ : Subsingleton α
      s₁ s₂ : Subalgebra R (ContinuousMap α R)
      ⊢ Eq s₁ s₂
    -/
    cases isEmpty_or_nonempty α
      /-
        case inl
        α : Type u_1
        inst✝⁴ : TopologicalSpace α
        R : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : TopologicalSpace R
        inst✝¹ : TopologicalSemiring R
        inst✝ : Subsingleton α
        s₁ s₂ : Subalgebra R (ContinuousMap α R)
        h✝ : IsEmpty α
        ⊢ Eq s₁ s₂
      -/
    · have : Subsingleton C(α, R) := DFunLike.coe_injective.subsingleton
      /-
        case inl
        α : Type u_1
        inst✝⁴ : TopologicalSpace α
        R : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : TopologicalSpace R
        inst✝¹ : TopologicalSemiring R
        inst✝ : Subsingleton α
        s₁ s₂ : Subalgebra R (ContinuousMap α R)
        h✝ : IsEmpty α
        this : Subsingleton (ContinuousMap α R)
        ⊢ Eq s₁ s₂
      -/
      subsingleton
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_1
        inst✝⁴ : TopologicalSpace α
        R : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : TopologicalSpace R
        inst✝¹ : TopologicalSemiring R
        inst✝ : Subsingleton α
        s₁ s₂ : Subalgebra R (ContinuousMap α R)
        h✝ : Nonempty α
        ⊢ Eq s₁ s₂
      -/
    · inhabit α
      /-
        case inr
        α : Type u_1
        inst✝⁴ : TopologicalSpace α
        R : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : TopologicalSpace R
        inst✝¹ : TopologicalSemiring R
        inst✝ : Subsingleton α
        s₁ s₂ : Subalgebra R (ContinuousMap α R)
        h✝ : Nonempty α
        inhabited_h : Inhabited α
        ⊢ Eq s₁ s₂
      -/
      ext f
      have h : f = algebraMap R C(α, R) (f default) := by
        ext x'
        simp only [mul_one, Algebra.id.smul_eq_mul, algebraMap_apply]
        congr
        simp [eq_iff_true_of_subsingleton]
      /-
        case inr.h
        α : Type u_1
        inst✝⁴ : TopologicalSpace α
        R : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : TopologicalSpace R
        inst✝¹ : TopologicalSemiring R
        inst✝ : Subsingleton α
        s₁ s₂ : Subalgebra R (ContinuousMap α R)
        h✝ : Nonempty α
        inhabited_h : Inhabited α
        f : ContinuousMap α R
        h : Eq f ((algebraMap R (ContinuousMap α R)) (f Inhabited.default))
        ⊢ Iff (Membership.mem s₁ f) (Membership.mem s₂ f)
      -/
      rw [h]
      /-
        case inr.h
        α : Type u_1
        inst✝⁴ : TopologicalSpace α
        R : Type u_2
        inst✝³ : CommSemiring R
        inst✝² : TopologicalSpace R
        inst✝¹ : TopologicalSemiring R
        inst✝ : Subsingleton α
        s₁ s₂ : Subalgebra R (ContinuousMap α R)
        h✝ : Nonempty α
        inhabited_h : Inhabited α
        f : ContinuousMap α R
        h : Eq f ((algebraMap R (ContinuousMap α R)) (f Inhabited.default))
        ⊢ Iff (Membership.mem s₁ ((algebraMap R (ContinuousMap α R)) (f Inhabited.defa …
      -/
      simp only [Subalgebra.algebraMap_mem]⟩
      /-
        🎉 no goals
      -/


instance instSMul' : SMul C(α, R) C(α, M) :=
  ⟨fun f g => ⟨fun x => f x • g x, Continuous.smul f.2 g.2⟩⟩


/-- Coercion to a function for a scalar-valued continuous map multiplying a vector-valued one
(as opposed to `ContinuousMap.coe_smul` which is multiplication by a constant scalar). -/
@[simp] lemma coe_smul' (f : C(α, R)) (g : C(α, M)) :
    ⇑(f • g) = ⇑f • ⇑g :=
  rfl


/-- Evaluation of a scalar-valued continuous map multiplying a vector-valued one
(as opposed to `ContinuousMap.smul_apply` which is multiplication by a constant scalar). -/
-- (this doesn't need to be @[simp] since it can be derived from `coe_smul'` and `Pi.smul_apply'`)
lemma smul_apply' (f : C(α, R)) (g : C(α, M)) (x : α) :
    (f • g) x = f x • g x :=
  rfl


instance module' [TopologicalSemiring R] [ContinuousAdd M] :
    Module C(α, R) C(α, M) where
  smul := (· • ·)
                       /-
                         α : Type u_1
                         inst✝⁸ : TopologicalSpace α
                         R : Type u_2
                         inst✝⁷ : Semiring R
                         inst✝⁶ : TopologicalSpace R
                         M : Type u_3
                         inst✝⁵ : TopologicalSpace M
                         inst✝⁴ : AddCommMonoid M
                         inst✝³ : Module R M
                         inst✝² : ContinuousSMul R M
                         inst✝¹ : TopologicalSemiring R
                         inst✝ : ContinuousAdd M
                         c : ContinuousMap α R
                         f g : ContinuousMap α M
                         ⊢ Eq (HSMul.hSMul c (HAdd.hAdd f g)) (HAdd.hAdd (HSMul.hSMul c f) (HSMul.hSMul …
                       -/
  smul_add c f g := by ext x; exact smul_add (c x) (f x) (g x)
                         /-
                           α : Type u_1
                           inst✝⁸ : TopologicalSpace α
                           R : Type u_2
                           inst✝⁷ : Semiring R
                           inst✝⁶ : TopologicalSpace R
                           M : Type u_3
                           inst✝⁵ : TopologicalSpace M
                           inst✝⁴ : AddCommMonoid M
                           inst✝³ : Module R M
                           inst✝² : ContinuousSMul R M
                           inst✝¹ : TopologicalSemiring R
                           inst✝ : ContinuousAdd M
                           c₁ c₂ : ContinuousMap α R
                           f : ContinuousMap α M
                           ⊢ Eq (HSMul.hSMul (HMul.hMul c₁ c₂) f) (HSMul.hSMul c₁ (HSMul.hSMul c₂ f))
                         -/
                   /-
                     α : Type u_1
                     inst✝⁸ : TopologicalSpace α
                     R : Type u_2
                     inst✝⁷ : Semiring R
                     inst✝⁶ : TopologicalSpace R
                     M : Type u_3
                     inst✝⁵ : TopologicalSpace M
                     inst✝⁴ : AddCommMonoid M
                     inst✝³ : Module R M
                     inst✝² : ContinuousSMul R M
                     inst✝¹ : TopologicalSemiring R
                     inst✝ : ContinuousAdd M
                     f : ContinuousMap α M
                     ⊢ Eq (HSMul.hSMul 1 f) f
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
                         /-
                           α : Type u_1
                           inst✝⁸ : TopologicalSpace α
                           R : Type u_2
                           inst✝⁷ : Semiring R
                           inst✝⁶ : TopologicalSpace R
                           M : Type u_3
                           inst✝⁵ : TopologicalSpace M
                           inst✝⁴ : AddCommMonoid M
                           inst✝³ : Module R M
                           inst✝² : ContinuousSMul R M
                           inst✝¹ : TopologicalSemiring R
                           inst✝ : ContinuousAdd M
                           c₁ c₂ : ContinuousMap α R
                           f : ContinuousMap α M
                           ⊢ Eq (HSMul.hSMul (HAdd.hAdd c₁ c₂) f) (HAdd.hAdd (HSMul.hSMul c₁ f) (HSMul.hS …
                         -/
  add_smul c₁ c₂ f := by ext x; exact add_smul (c₁ x) (c₂ x) (f x)
                    /-
                      α : Type u_1
                      inst✝⁸ : TopologicalSpace α
                      R : Type u_2
                      inst✝⁷ : Semiring R
                      inst✝⁶ : TopologicalSpace R
                      M : Type u_3
                      inst✝⁵ : TopologicalSpace M
                      inst✝⁴ : AddCommMonoid M
                      inst✝³ : Module R M
                      inst✝² : ContinuousSMul R M
                      inst✝¹ : TopologicalSemiring R
                      inst✝ : ContinuousAdd M
                      r : ContinuousMap α R
                      ⊢ Eq (HSMul.hSMul r 0) 0
                    -/
                                /-
                                  🎉 no goals
                                -/
                           /-
                             🎉 no goals
                           -/
  mul_smul c₁ c₂ f := by ext x; exact mul_smul (c₁ x) (c₂ x) (f x)
  one_smul f := by ext x; exact one_smul R (f x)
                    /-
                      α : Type u_1
                      inst✝⁸ : TopologicalSpace α
                      R : Type u_2
                      inst✝⁷ : Semiring R
                      inst✝⁶ : TopologicalSpace R
                      M : Type u_3
                      inst✝⁵ : TopologicalSpace M
                      inst✝⁴ : AddCommMonoid M
                      inst✝³ : Module R M
                      inst✝² : ContinuousSMul R M
                      inst✝¹ : TopologicalSemiring R
                      inst✝ : ContinuousAdd M
                      f : ContinuousMap α M
                      ⊢ Eq (HSMul.hSMul 0 f) 0
                    -/
  zero_smul f := by ext x; exact zero_smul _ _
                           /-
                             🎉 no goals
                           -/
  smul_zero r := by ext x; exact smul_zero _


/-- Evaluation of continuous maps at a point, bundled as an algebra homomorphism. -/
@[simps]
def ContinuousMap.evalAlgHom (x : X) : C(X, R) →ₐ[S] R where
  toFun f := f x
  map_zero' := rfl
  map_one' := rfl
  map_add' _ _ := rfl
  map_mul' _ _ := rfl
  commutes' _ := rfl

