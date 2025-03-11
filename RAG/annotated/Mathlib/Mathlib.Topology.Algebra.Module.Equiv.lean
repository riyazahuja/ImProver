/-- Continuous linear equivalences between modules. We only put the type classes that are necessary
for the definition, although in applications `M` and `M₂` will be topological modules over the
topological semiring `R`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): linter not ported yet; was @[nolint has_nonempty_instance]
structure ContinuousLinearEquiv {R : Type*} {S : Type*} [Semiring R] [Semiring S] (σ : R →+* S)
    {σ' : S →+* R} [RingHomInvPair σ σ'] [RingHomInvPair σ' σ] (M : Type*) [TopologicalSpace M]
    [AddCommMonoid M] (M₂ : Type*) [TopologicalSpace M₂] [AddCommMonoid M₂] [Module R M]
    [Module S M₂] extends M ≃ₛₗ[σ] M₂ where
  continuous_toFun : Continuous toFun := by continuity
  continuous_invFun : Continuous invFun := by continuity


@[inherit_doc]
notation:50 M " ≃SL[" σ "] " M₂ => ContinuousLinearEquiv σ M M₂


@[inherit_doc]
notation:50 M " ≃L[" R "] " M₂ => ContinuousLinearEquiv (RingHom.id R) M M₂


/-- `ContinuousSemilinearEquivClass F σ M M₂` asserts `F` is a type of bundled continuous
`σ`-semilinear equivs `M → M₂`.  See also `ContinuousLinearEquivClass F R M M₂` for the case
where `σ` is the identity map on `R`.  A map `f` between an `R`-module and an `S`-module over a ring
homomorphism `σ : R →+* S` is semilinear if it satisfies the two properties `f (x + y) = f x + f y`
and `f (c • x) = (σ c) • f x`. -/
class ContinuousSemilinearEquivClass (F : Type*) {R : outParam Type*} {S : outParam Type*}
    [Semiring R] [Semiring S] (σ : outParam <| R →+* S) {σ' : outParam <| S →+* R}
    [RingHomInvPair σ σ'] [RingHomInvPair σ' σ] (M : outParam Type*) [TopologicalSpace M]
    [AddCommMonoid M] (M₂ : outParam Type*) [TopologicalSpace M₂] [AddCommMonoid M₂] [Module R M]
    [Module S M₂] [EquivLike F M M₂] extends SemilinearEquivClass F σ M M₂ : Prop where
  map_continuous : ∀ f : F, Continuous f := by continuity
  inv_continuous : ∀ f : F, Continuous (EquivLike.inv f) := by continuity


/-- `ContinuousLinearEquivClass F σ M M₂` asserts `F` is a type of bundled continuous
`R`-linear equivs `M → M₂`. This is an abbreviation for
`ContinuousSemilinearEquivClass F (RingHom.id R) M M₂`. -/
abbrev ContinuousLinearEquivClass (F : Type*) (R : outParam Type*) [Semiring R]
    (M : outParam Type*) [TopologicalSpace M] [AddCommMonoid M] (M₂ : outParam Type*)
    [TopologicalSpace M₂] [AddCommMonoid M₂] [Module R M] [Module R M₂] [EquivLike F M M₂] :=
  ContinuousSemilinearEquivClass F (RingHom.id R) M M₂


instance (priority := 100) continuousSemilinearMapClass [EquivLike F M M₂]
    [s : ContinuousSemilinearEquivClass F σ M M₂] : ContinuousSemilinearMapClass F σ M M₂ :=
  { s with }


instance (priority := 100) [EquivLike F M M₂]
    [s : ContinuousSemilinearEquivClass F σ M M₂] : HomeomorphClass F M M₂ :=
  { s with }


/-- If `I` and `J` are complementary index sets, the product of the kernels of the `J`th projections
of `φ` is linearly equivalent to the product over `I`. -/
def iInfKerProjEquiv {I J : Set ι} [DecidablePred fun i => i ∈ I] (hd : Disjoint I J)
    (hu : Set.univ ⊆ I ∪ J) :
    (⨅ i ∈ J, ker (proj i : (∀ i, φ i) →L[R] φ i) :
    Submodule R (∀ i, φ i)) ≃L[R] ∀ i : I, φ i where
  toLinearEquiv := LinearMap.iInfKerProjEquiv R φ hd hu
  continuous_toFun :=
    continuous_pi fun i =>
      Continuous.comp (continuous_apply (π := φ) i) <|
        @continuous_subtype_val _ _ fun x =>
          x ∈ (⨅ i ∈ J, ker (proj i : (∀ i, φ i) →L[R] φ i) : Submodule R (∀ i, φ i))
  continuous_invFun :=
    Continuous.subtype_mk
      (continuous_pi fun i => by
        /-
          R✝ : Type u_1
          M✝ : Type u_2
          inst✝¹⁷ : Ring R✝
          inst✝¹⁶ : TopologicalSpace R✝
          inst✝¹⁵ : TopologicalSpace M✝
          inst✝¹⁴ : AddCommGroup M✝
          inst✝¹³ : ContinuousAdd M✝
          inst✝¹² : Module R✝ M✝
          inst✝¹¹ : ContinuousSMul R✝ M✝
          R : Type u_3
          inst✝¹⁰ : Semiring R
          M : Type u_4
          inst✝⁹ : TopologicalSpace M
          inst✝⁸ : AddCommMonoid M
          inst✝⁷ : Module R M
          M₂ : Type u_5
          inst✝⁶ : TopologicalSpace M₂
          inst✝⁵ : AddCommMonoid M₂
          inst✝⁴ : Module R M₂
          ι : Type u_6
          φ : ι → Type u_7
          inst✝³ : (i : ι) → TopologicalSpace (φ i)
          inst✝² : (i : ι) → AddCommMonoid (φ i)
          inst✝¹ : (i : ι) → Module R (φ i)
          I J : Set ι
          inst✝ : DecidablePred fun i => Membership.mem I i
          hd : Disjoint I J
          hu : HasSubset.Subset Set.univ (Union.union I J)
          i : ι
          ⊢ Continuous fun a => (LinearMap.pi fun i => dite (Membership.mem I i) (fun h  …
        -/
        dsimp
        /-
          R✝ : Type u_1
          M✝ : Type u_2
          inst✝¹⁷ : Ring R✝
          inst✝¹⁶ : TopologicalSpace R✝
          inst✝¹⁵ : TopologicalSpace M✝
          inst✝¹⁴ : AddCommGroup M✝
          inst✝¹³ : ContinuousAdd M✝
          inst✝¹² : Module R✝ M✝
          inst✝¹¹ : ContinuousSMul R✝ M✝
          R : Type u_3
          inst✝¹⁰ : Semiring R
          M : Type u_4
          inst✝⁹ : TopologicalSpace M
          inst✝⁸ : AddCommMonoid M
          inst✝⁷ : Module R M
          M₂ : Type u_5
          inst✝⁶ : TopologicalSpace M₂
          inst✝⁵ : AddCommMonoid M₂
          inst✝⁴ : Module R M₂
          ι : Type u_6
          φ : ι → Type u_7
          inst✝³ : (i : ι) → TopologicalSpace (φ i)
          inst✝² : (i : ι) → AddCommMonoid (φ i)
          inst✝¹ : (i : ι) → Module R (φ i)
          I J : Set ι
          inst✝ : DecidablePred fun i => Membership.mem I i
          hd : Disjoint I J
          hu : HasSubset.Subset Set.univ (Union.union I J)
          i : ι
          ⊢ Continuous fun a => (dite (Membership.mem I i) (fun h => LinearMap.proj ⟨i,  …
        -/
        split_ifs <;> [apply continuous_apply; exact continuous_zero])
        /-
          🎉 no goals
        -/
      _


/-- A continuous linear equivalence induces a continuous linear map. -/
@[coe]
def toContinuousLinearMap (e : M₁ ≃SL[σ₁₂] M₂) : M₁ →SL[σ₁₂] M₂ :=
  { e.toLinearEquiv.toLinearMap with cont := e.continuous_toFun }


/-- Coerce continuous linear equivs to continuous linear maps. -/
instance ContinuousLinearMap.coe : Coe (M₁ ≃SL[σ₁₂] M₂) (M₁ →SL[σ₁₂] M₂) :=
  ⟨toContinuousLinearMap⟩


instance equivLike :
    EquivLike (M₁ ≃SL[σ₁₂] M₂) M₁ M₂ where
  coe f := f.toFun
  inv f := f.invFun
  coe_injective' f g h₁ h₂ := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝²⁸ : Ring R
      inst✝²⁷ : TopologicalSpace R
      inst✝²⁶ : TopologicalSpace M
      inst✝²⁵ : AddCommGroup M
      inst✝²⁴ : ContinuousAdd M
      inst✝²³ : Module R M
      inst✝²² : ContinuousSMul R M
      R₁ : Type u_3
      R₂ : Type u_4
      R₃ : Type u_5
      inst✝²¹ : Semiring R₁
      inst✝²⁰ : Semiring R₂
      inst✝¹⁹ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₁ : RingHom R₂ R₁
      inst✝¹⁸ : RingHomInvPair σ₁₂ σ₂₁
      inst✝¹⁷ : RingHomInvPair σ₂₁ σ₁₂
      σ₂₃ : RingHom R₂ R₃
      σ₃₂ : RingHom R₃ R₂
      inst✝¹⁶ : RingHomInvPair σ₂₃ σ₃₂
      inst✝¹⁵ : RingHomInvPair σ₃₂ σ₂₃
      σ₁₃ : RingHom R₁ R₃
      σ₃₁ : RingHom R₃ R₁
      inst✝¹⁴ : RingHomInvPair σ₁₃ σ₃₁
      inst✝¹³ : RingHomInvPair σ₃₁ σ₁₃
      inst✝¹² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝¹¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
      M₁ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₁
      inst✝⁹ : AddCommMonoid M₁
      M₂ : Type u_7
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : AddCommMonoid M₂
      M₃ : Type u_8
      inst✝⁶ : TopologicalSpace M₃
      inst✝⁵ : AddCommMonoid M₃
      M₄ : Type u_9
      inst✝⁴ : TopologicalSpace M₄
      inst✝³ : AddCommMonoid M₄
      inst✝² : Module R₁ M₁
      inst✝¹ : Module R₂ M₂
      inst✝ : Module R₃ M₃
      f g : ContinuousLinearEquiv σ₁₂ M₁ M₂
      h₁ : Eq ((fun f => (↑f.toLinearEquiv).toFun) f) ((fun f => (↑f.toLinearEquiv). …
      h₂ : Eq ((fun f => f.invFun) f) ((fun f => f.invFun) g)
      ⊢ Eq f g
    -/
    cases' f with f' _
    /-
      case mk
      R : Type u_1
      M : Type u_2
      inst✝²⁸ : Ring R
      inst✝²⁷ : TopologicalSpace R
      inst✝²⁶ : TopologicalSpace M
      inst✝²⁵ : AddCommGroup M
      inst✝²⁴ : ContinuousAdd M
      inst✝²³ : Module R M
      inst✝²² : ContinuousSMul R M
      R₁ : Type u_3
      R₂ : Type u_4
      R₃ : Type u_5
      inst✝²¹ : Semiring R₁
      inst✝²⁰ : Semiring R₂
      inst✝¹⁹ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₁ : RingHom R₂ R₁
      inst✝¹⁸ : RingHomInvPair σ₁₂ σ₂₁
      inst✝¹⁷ : RingHomInvPair σ₂₁ σ₁₂
      σ₂₃ : RingHom R₂ R₃
      σ₃₂ : RingHom R₃ R₂
      inst✝¹⁶ : RingHomInvPair σ₂₃ σ₃₂
      inst✝¹⁵ : RingHomInvPair σ₃₂ σ₂₃
      σ₁₃ : RingHom R₁ R₃
      σ₃₁ : RingHom R₃ R₁
      inst✝¹⁴ : RingHomInvPair σ₁₃ σ₃₁
      inst✝¹³ : RingHomInvPair σ₃₁ σ₁₃
      inst✝¹² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝¹¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
      M₁ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₁
      inst✝⁹ : AddCommMonoid M₁
      M₂ : Type u_7
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : AddCommMonoid M₂
      M₃ : Type u_8
      inst✝⁶ : TopologicalSpace M₃
      inst✝⁵ : AddCommMonoid M₃
      M₄ : Type u_9
      inst✝⁴ : TopologicalSpace M₄
      inst✝³ : AddCommMonoid M₄
      inst✝² : Module R₁ M₁
      inst✝¹ : Module R₂ M₂
      inst✝ : Module R₃ M₃
      g : ContinuousLinearEquiv σ₁₂ M₁ M₂
      f' : LinearEquiv σ₁₂ M₁ M₂
      continuous_toFun✝ : Continuous (↑f').toFun
      continuous_invFun✝ : Continuous f'.invFun
      h₁ : Eq ((fun f => (↑f.toLinearEquiv).toFun) { toLinearEquiv := f', continuous …
      h₂ : Eq ((fun f => f.invFun) { toLinearEquiv := f', continuous_toFun := contin …
      ⊢ Eq { toLinearEquiv := f', continuous_toFun := continuous_toFun✝, continuous_ …
    -/
    cases' g with g' _
    /-
      case mk.mk
      R : Type u_1
      M : Type u_2
      inst✝²⁸ : Ring R
      inst✝²⁷ : TopologicalSpace R
      inst✝²⁶ : TopologicalSpace M
      inst✝²⁵ : AddCommGroup M
      inst✝²⁴ : ContinuousAdd M
      inst✝²³ : Module R M
      inst✝²² : ContinuousSMul R M
      R₁ : Type u_3
      R₂ : Type u_4
      R₃ : Type u_5
      inst✝²¹ : Semiring R₁
      inst✝²⁰ : Semiring R₂
      inst✝¹⁹ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₁ : RingHom R₂ R₁
      inst✝¹⁸ : RingHomInvPair σ₁₂ σ₂₁
      inst✝¹⁷ : RingHomInvPair σ₂₁ σ₁₂
      σ₂₃ : RingHom R₂ R₃
      σ₃₂ : RingHom R₃ R₂
      inst✝¹⁶ : RingHomInvPair σ₂₃ σ₃₂
      inst✝¹⁵ : RingHomInvPair σ₃₂ σ₂₃
      σ₁₃ : RingHom R₁ R₃
      σ₃₁ : RingHom R₃ R₁
      inst✝¹⁴ : RingHomInvPair σ₁₃ σ₃₁
      inst✝¹³ : RingHomInvPair σ₃₁ σ₁₃
      inst✝¹² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝¹¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
      M₁ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₁
      inst✝⁹ : AddCommMonoid M₁
      M₂ : Type u_7
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : AddCommMonoid M₂
      M₃ : Type u_8
      inst✝⁶ : TopologicalSpace M₃
      inst✝⁵ : AddCommMonoid M₃
      M₄ : Type u_9
      inst✝⁴ : TopologicalSpace M₄
      inst✝³ : AddCommMonoid M₄
      inst✝² : Module R₁ M₁
      inst✝¹ : Module R₂ M₂
      inst✝ : Module R₃ M₃
      f' : LinearEquiv σ₁₂ M₁ M₂
      continuous_toFun✝¹ : Continuous (↑f').toFun
      continuous_invFun✝¹ : Continuous f'.invFun
      g' : LinearEquiv σ₁₂ M₁ M₂
      continuous_toFun✝ : Continuous (↑g').toFun
      continuous_invFun✝ : Continuous g'.invFun
      h₁ : Eq ((fun f => (↑f.toLinearEquiv).toFun) { toLinearEquiv := f', continuous …
      h₂ : Eq ((fun f => f.invFun) { toLinearEquiv := f', continuous_toFun := contin …
      ⊢ Eq { toLinearEquiv := f', continuous_toFun := continuous_toFun✝¹, continuous …
    -/
    rcases f' with ⟨⟨⟨_, _⟩, _⟩, _⟩
    /-
      case mk.mk.mk.mk.mk
      R : Type u_1
      M : Type u_2
      inst✝²⁸ : Ring R
      inst✝²⁷ : TopologicalSpace R
      inst✝²⁶ : TopologicalSpace M
      inst✝²⁵ : AddCommGroup M
      inst✝²⁴ : ContinuousAdd M
      inst✝²³ : Module R M
      inst✝²² : ContinuousSMul R M
      R₁ : Type u_3
      R₂ : Type u_4
      R₃ : Type u_5
      inst✝²¹ : Semiring R₁
      inst✝²⁰ : Semiring R₂
      inst✝¹⁹ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₁ : RingHom R₂ R₁
      inst✝¹⁸ : RingHomInvPair σ₁₂ σ₂₁
      inst✝¹⁷ : RingHomInvPair σ₂₁ σ₁₂
      σ₂₃ : RingHom R₂ R₃
      σ₃₂ : RingHom R₃ R₂
      inst✝¹⁶ : RingHomInvPair σ₂₃ σ₃₂
      inst✝¹⁵ : RingHomInvPair σ₃₂ σ₂₃
      σ₁₃ : RingHom R₁ R₃
      σ₃₁ : RingHom R₃ R₁
      inst✝¹⁴ : RingHomInvPair σ₁₃ σ₃₁
      inst✝¹³ : RingHomInvPair σ₃₁ σ₁₃
      inst✝¹² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝¹¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
      M₁ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₁
      inst✝⁹ : AddCommMonoid M₁
      M₂ : Type u_7
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : AddCommMonoid M₂
      M₃ : Type u_8
      inst✝⁶ : TopologicalSpace M₃
      inst✝⁵ : AddCommMonoid M₃
      M₄ : Type u_9
      inst✝⁴ : TopologicalSpace M₄
      inst✝³ : AddCommMonoid M₄
      inst✝² : Module R₁ M₁
      inst✝¹ : Module R₂ M₂
      inst✝ : Module R₃ M₃
      g' : LinearEquiv σ₁₂ M₁ M₂
      continuous_toFun✝¹ : Continuous (↑g').toFun
      continuous_invFun✝¹ : Continuous g'.invFun
      invFun✝ : M₂ → M₁
      toFun✝ : M₁ → M₂
      map_add'✝ : ∀ (x y : M₁), Eq (toFun✝ (HAdd.hAdd x y)) (HAdd.hAdd (toFun✝ x) (t …
      map_smul'✝ : ∀ (m : R₁) (x : M₁), Eq ({ toFun := toFun✝, map_add' := map_add'✝ …
      left_inv✝ : Function.LeftInverse invFun✝ { toFun := toFun✝, map_add' := map_ad …
      right_inv✝ : Function.RightInverse invFun✝ { toFun := toFun✝, map_add' := map_ …
      continuous_toFun✝ : Continuous (↑{ toFun := toFun✝, map_add' := map_add'✝, map …
      continuous_invFun✝ : Continuous { toFun := toFun✝, map_add' := map_add'✝, map_ …
      h₁ : Eq ((fun f => (↑f.toLinearEquiv).toFun) { toFun := toFun✝, map_add' := ma …
      h₂ : Eq ((fun f => f.invFun) { toFun := toFun✝, map_add' := map_add'✝, map_smu …
      ⊢ Eq { toFun := toFun✝, map_add' := map_add'✝, map_smul' := map_smul'✝, invFun …
    -/
    rcases g' with ⟨⟨⟨_, _⟩, _⟩, _⟩
    /-
      case mk.mk.mk.mk.mk.mk.mk.mk
      R : Type u_1
      M : Type u_2
      inst✝²⁸ : Ring R
      inst✝²⁷ : TopologicalSpace R
      inst✝²⁶ : TopologicalSpace M
      inst✝²⁵ : AddCommGroup M
      inst✝²⁴ : ContinuousAdd M
      inst✝²³ : Module R M
      inst✝²² : ContinuousSMul R M
      R₁ : Type u_3
      R₂ : Type u_4
      R₃ : Type u_5
      inst✝²¹ : Semiring R₁
      inst✝²⁰ : Semiring R₂
      inst✝¹⁹ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₁ : RingHom R₂ R₁
      inst✝¹⁸ : RingHomInvPair σ₁₂ σ₂₁
      inst✝¹⁷ : RingHomInvPair σ₂₁ σ₁₂
      σ₂₃ : RingHom R₂ R₃
      σ₃₂ : RingHom R₃ R₂
      inst✝¹⁶ : RingHomInvPair σ₂₃ σ₃₂
      inst✝¹⁵ : RingHomInvPair σ₃₂ σ₂₃
      σ₁₃ : RingHom R₁ R₃
      σ₃₁ : RingHom R₃ R₁
      inst✝¹⁴ : RingHomInvPair σ₁₃ σ₃₁
      inst✝¹³ : RingHomInvPair σ₃₁ σ₁₃
      inst✝¹² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝¹¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
      M₁ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₁
      inst✝⁹ : AddCommMonoid M₁
      M₂ : Type u_7
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : AddCommMonoid M₂
      M₃ : Type u_8
      inst✝⁶ : TopologicalSpace M₃
      inst✝⁵ : AddCommMonoid M₃
      M₄ : Type u_9
      inst✝⁴ : TopologicalSpace M₄
      inst✝³ : AddCommMonoid M₄
      inst✝² : Module R₁ M₁
      inst✝¹ : Module R₂ M₂
      inst✝ : Module R₃ M₃
      invFun✝¹ : M₂ → M₁
      toFun✝¹ : M₁ → M₂
      map_add'✝¹ : ∀ (x y : M₁), Eq (toFun✝¹ (HAdd.hAdd x y)) (HAdd.hAdd (toFun✝¹ x) …
      map_smul'✝¹ : ∀ (m : R₁) (x : M₁), Eq ({ toFun := toFun✝¹, map_add' := map_add …
      left_inv✝¹ : Function.LeftInverse invFun✝¹ { toFun := toFun✝¹, map_add' := map …
      right_inv✝¹ : Function.RightInverse invFun✝¹ { toFun := toFun✝¹, map_add' := m …
      continuous_toFun✝¹ : Continuous (↑{ toFun := toFun✝¹, map_add' := map_add'✝¹,  …
      continuous_invFun✝¹ : Continuous { toFun := toFun✝¹, map_add' := map_add'✝¹, m …
      invFun✝ : M₂ → M₁
      toFun✝ : M₁ → M₂
      map_add'✝ : ∀ (x y : M₁), Eq (toFun✝ (HAdd.hAdd x y)) (HAdd.hAdd (toFun✝ x) (t …
      map_smul'✝ : ∀ (m : R₁) (x : M₁), Eq ({ toFun := toFun✝, map_add' := map_add'✝ …
      left_inv✝ : Function.LeftInverse invFun✝ { toFun := toFun✝, map_add' := map_ad …
      right_inv✝ : Function.RightInverse invFun✝ { toFun := toFun✝, map_add' := map_ …
      continuous_toFun✝ : Continuous (↑{ toFun := toFun✝, map_add' := map_add'✝, map …
      continuous_invFun✝ : Continuous { toFun := toFun✝, map_add' := map_add'✝, map_ …
      h₁ : Eq ((fun f => (↑f.toLinearEquiv).toFun) { toFun := toFun✝¹, map_add' := m …
      h₂ : Eq ((fun f => f.invFun) { toFun := toFun✝¹, map_add' := map_add'✝¹, map_s …
      ⊢ Eq { toFun := toFun✝¹, map_add' := map_add'✝¹, map_smul' := map_smul'✝¹, inv …
    -/
    congr
    /-
      🎉 no goals
    -/
  left_inv f := f.left_inv
  right_inv f := f.right_inv


instance continuousSemilinearEquivClass :
    ContinuousSemilinearEquivClass (M₁ ≃SL[σ₁₂] M₂) σ₁₂ M₁ M₂ where
  map_add f := f.map_add'
  map_smulₛₗ f := f.map_smul'
  map_continuous := continuous_toFun
  inv_continuous := continuous_invFun


theorem coe_apply (e : M₁ ≃SL[σ₁₂] M₂) (b : M₁) : (e : M₁ →SL[σ₁₂] M₂) b = e b :=
  rfl


@[simp]
theorem coe_toLinearEquiv (f : M₁ ≃SL[σ₁₂] M₂) : ⇑f.toLinearEquiv = f :=
  rfl


@[simp, norm_cast]
theorem coe_coe (e : M₁ ≃SL[σ₁₂] M₂) : ⇑(e : M₁ →SL[σ₁₂] M₂) = e :=
  rfl


theorem toLinearEquiv_injective :
    Function.Injective (toLinearEquiv : (M₁ ≃SL[σ₁₂] M₂) → M₁ ≃ₛₗ[σ₁₂] M₂) := by
  /-
    R₁ : Type u_3
    R₂ : Type u_4
    inst✝⁹ : Semiring R₁
    inst✝⁸ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    inst✝⁷ : RingHomInvPair σ₁₂ σ₂₁
    inst✝⁶ : RingHomInvPair σ₂₁ σ₁₂
    M₁ : Type u_6
    inst✝⁵ : TopologicalSpace M₁
    inst✝⁴ : AddCommMonoid M₁
    M₂ : Type u_7
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R₁ M₁
    inst✝ : Module R₂ M₂
    ⊢ Function.Injective ContinuousLinearEquiv.toLinearEquiv
  -/
  rintro ⟨e, _, _⟩ ⟨e', _, _⟩ rfl
  /-
    case mk.mk
    R₁ : Type u_3
    R₂ : Type u_4
    inst✝⁹ : Semiring R₁
    inst✝⁸ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    inst✝⁷ : RingHomInvPair σ₁₂ σ₂₁
    inst✝⁶ : RingHomInvPair σ₂₁ σ₁₂
    M₁ : Type u_6
    inst✝⁵ : TopologicalSpace M₁
    inst✝⁴ : AddCommMonoid M₁
    M₂ : Type u_7
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R₁ M₁
    inst✝ : Module R₂ M₂
    e : LinearEquiv σ₁₂ M₁ M₂
    continuous_toFun✝¹ : Continuous (↑e).toFun
    continuous_invFun✝¹ : Continuous e.invFun
    continuous_toFun✝ : Continuous (↑{ toLinearEquiv := e, continuous_toFun := con …
    continuous_invFun✝ : Continuous { toLinearEquiv := e, continuous_toFun := cont …
    ⊢ Eq { toLinearEquiv := e, continuous_toFun := continuous_toFun✝¹, continuous_ …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[ext]
theorem ext {f g : M₁ ≃SL[σ₁₂] M₂} (h : (f : M₁ → M₂) = g) : f = g :=
  toLinearEquiv_injective <| LinearEquiv.ext <| congr_fun h


theorem coe_injective : Function.Injective ((↑) : (M₁ ≃SL[σ₁₂] M₂) → M₁ →SL[σ₁₂] M₂) :=
  fun _e _e' h => ext <| funext <| ContinuousLinearMap.ext_iff.1 h


@[simp, norm_cast]
theorem coe_inj {e e' : M₁ ≃SL[σ₁₂] M₂} : (e : M₁ →SL[σ₁₂] M₂) = e' ↔ e = e' :=
  coe_injective.eq_iff


/-- A continuous linear equivalence induces a homeomorphism. -/
def toHomeomorph (e : M₁ ≃SL[σ₁₂] M₂) : M₁ ≃ₜ M₂ :=
  { e with toEquiv := e.toLinearEquiv.toEquiv }


@[simp]
theorem coe_toHomeomorph (e : M₁ ≃SL[σ₁₂] M₂) : ⇑e.toHomeomorph = e :=
  rfl


theorem isOpenMap (e : M₁ ≃SL[σ₁₂] M₂) : IsOpenMap e :=
  (ContinuousLinearEquiv.toHomeomorph e).isOpenMap


theorem image_closure (e : M₁ ≃SL[σ₁₂] M₂) (s : Set M₁) : e '' closure s = closure (e '' s) :=
  e.toHomeomorph.image_closure s


theorem preimage_closure (e : M₁ ≃SL[σ₁₂] M₂) (s : Set M₂) : e ⁻¹' closure s = closure (e ⁻¹' s) :=
  e.toHomeomorph.preimage_closure s


@[simp]
theorem isClosed_image (e : M₁ ≃SL[σ₁₂] M₂) {s : Set M₁} : IsClosed (e '' s) ↔ IsClosed s :=
  e.toHomeomorph.isClosed_image


theorem map_nhds_eq (e : M₁ ≃SL[σ₁₂] M₂) (x : M₁) : map e (𝓝 x) = 𝓝 (e x) :=
  e.toHomeomorph.map_nhds_eq x

-- Make some straightforward lemmas available to `simp`.

theorem map_zero (e : M₁ ≃SL[σ₁₂] M₂) : e (0 : M₁) = 0 :=
  (e : M₁ →SL[σ₁₂] M₂).map_zero


theorem map_add (e : M₁ ≃SL[σ₁₂] M₂) (x y : M₁) : e (x + y) = e x + e y :=
  (e : M₁ →SL[σ₁₂] M₂).map_add x y


@[simp]
theorem map_smulₛₗ (e : M₁ ≃SL[σ₁₂] M₂) (c : R₁) (x : M₁) : e (c • x) = σ₁₂ c • e x :=
  (e : M₁ →SL[σ₁₂] M₂).map_smulₛₗ c x


theorem map_smul [Module R₁ M₂] (e : M₁ ≃L[R₁] M₂) (c : R₁) (x : M₁) : e (c • x) = c • e x :=
  (e : M₁ →L[R₁] M₂).map_smul c x


theorem map_eq_zero_iff (e : M₁ ≃SL[σ₁₂] M₂) {x : M₁} : e x = 0 ↔ x = 0 :=
  e.toLinearEquiv.map_eq_zero_iff


@[continuity]
protected theorem continuous (e : M₁ ≃SL[σ₁₂] M₂) : Continuous (e : M₁ → M₂) :=
  e.continuous_toFun


protected theorem continuousOn (e : M₁ ≃SL[σ₁₂] M₂) {s : Set M₁} : ContinuousOn (e : M₁ → M₂) s :=
  e.continuous.continuousOn


protected theorem continuousAt (e : M₁ ≃SL[σ₁₂] M₂) {x : M₁} : ContinuousAt (e : M₁ → M₂) x :=
  e.continuous.continuousAt


protected theorem continuousWithinAt (e : M₁ ≃SL[σ₁₂] M₂) {s : Set M₁} {x : M₁} :
    ContinuousWithinAt (e : M₁ → M₂) s x :=
  e.continuous.continuousWithinAt


theorem comp_continuousOn_iff {α : Type*} [TopologicalSpace α] (e : M₁ ≃SL[σ₁₂] M₂) {f : α → M₁}
    {s : Set α} : ContinuousOn (e ∘ f) s ↔ ContinuousOn f s :=
  e.toHomeomorph.comp_continuousOn_iff _ _


theorem comp_continuous_iff {α : Type*} [TopologicalSpace α] (e : M₁ ≃SL[σ₁₂] M₂) {f : α → M₁} :
    Continuous (e ∘ f) ↔ Continuous f :=
  e.toHomeomorph.comp_continuous_iff


/-- An extensionality lemma for `R ≃L[R] M`. -/
theorem ext₁ [TopologicalSpace R₁] {f g : R₁ ≃L[R₁] M₁} (h : f 1 = g 1) : f = g :=
                                        /-
                                          R₁ : Type u_3
                                          inst✝⁴ : Semiring R₁
                                          M₁ : Type u_6
                                          inst✝³ : TopologicalSpace M₁
                                          inst✝² : AddCommMonoid M₁
                                          inst✝¹ : Module R₁ M₁
                                          inst✝ : TopologicalSpace R₁
                                          f g : ContinuousLinearEquiv (RingHom.id R₁) R₁ M₁
                                          h : Eq (f 1) (g 1)
                                          x : R₁
                                          ⊢ Eq (f (HMul.hMul x 1)) (g (HMul.hMul x 1))
                                        -/
  ext <| funext fun x => mul_one x ▸ by rw [← smul_eq_mul, map_smul, h, map_smul]
                                        /-
                                          🎉 no goals
                                        -/


/-- The identity map as a continuous linear equivalence. -/
@[refl]
protected def refl : M₁ ≃L[R₁] M₁ :=
  { LinearEquiv.refl R₁ M₁ with
    continuous_toFun := continuous_id
    continuous_invFun := continuous_id }


@[simp]
theorem refl_apply (x : M₁) :
    ContinuousLinearEquiv.refl R₁ M₁ x = x := rfl


@[simp, norm_cast]
theorem coe_refl : ↑(ContinuousLinearEquiv.refl R₁ M₁) = ContinuousLinearMap.id R₁ M₁ :=
  rfl


@[simp, norm_cast]
theorem coe_refl' : ⇑(ContinuousLinearEquiv.refl R₁ M₁) = id :=
  rfl


/-- The inverse of a continuous linear equivalence as a continuous linear equivalence -/
@[symm]
protected def symm (e : M₁ ≃SL[σ₁₂] M₂) : M₂ ≃SL[σ₂₁] M₁ :=
  { e.toLinearEquiv.symm with
    continuous_toFun := e.continuous_invFun
    continuous_invFun := e.continuous_toFun }


@[simp]
theorem symm_toLinearEquiv (e : M₁ ≃SL[σ₁₂] M₂) : e.symm.toLinearEquiv = e.toLinearEquiv.symm := by
  /-
    R₁ : Type u_3
    R₂ : Type u_4
    inst✝⁹ : Semiring R₁
    inst✝⁸ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    inst✝⁷ : RingHomInvPair σ₁₂ σ₂₁
    inst✝⁶ : RingHomInvPair σ₂₁ σ₁₂
    M₁ : Type u_6
    inst✝⁵ : TopologicalSpace M₁
    inst✝⁴ : AddCommMonoid M₁
    M₂ : Type u_7
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R₁ M₁
    inst✝ : Module R₂ M₂
    e : ContinuousLinearEquiv σ₁₂ M₁ M₂
    ⊢ Eq e.symm.toLinearEquiv e.symm
  -/
  ext
  /-
    case h
    R₁ : Type u_3
    R₂ : Type u_4
    inst✝⁹ : Semiring R₁
    inst✝⁸ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    inst✝⁷ : RingHomInvPair σ₁₂ σ₂₁
    inst✝⁶ : RingHomInvPair σ₂₁ σ₁₂
    M₁ : Type u_6
    inst✝⁵ : TopologicalSpace M₁
    inst✝⁴ : AddCommMonoid M₁
    M₂ : Type u_7
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R₁ M₁
    inst✝ : Module R₂ M₂
    e : ContinuousLinearEquiv σ₁₂ M₁ M₂
    x✝ : M₂
    ⊢ Eq (e.symm.toLinearEquiv x✝) (e.symm x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem symm_toHomeomorph (e : M₁ ≃SL[σ₁₂] M₂) : e.toHomeomorph.symm = e.symm.toHomeomorph :=
  rfl


/-- See Note [custom simps projection]. We need to specify this projection explicitly in this case,
  because it is a composition of multiple projections. -/
def Simps.apply (h : M₁ ≃SL[σ₁₂] M₂) : M₁ → M₂ :=
  h


/-- See Note [custom simps projection] -/
def Simps.symm_apply (h : M₁ ≃SL[σ₁₂] M₂) : M₂ → M₁ :=
  h.symm


theorem symm_map_nhds_eq (e : M₁ ≃SL[σ₁₂] M₂) (x : M₁) : map e.symm (𝓝 (e x)) = 𝓝 x :=
  e.toHomeomorph.symm_map_nhds_eq x


/-- The composition of two continuous linear equivalences as a continuous linear equivalence. -/
@[trans]
protected def trans (e₁ : M₁ ≃SL[σ₁₂] M₂) (e₂ : M₂ ≃SL[σ₂₃] M₃) : M₁ ≃SL[σ₁₃] M₃ :=
  { e₁.toLinearEquiv.trans e₂.toLinearEquiv with
    continuous_toFun := e₂.continuous_toFun.comp e₁.continuous_toFun
    continuous_invFun := e₁.continuous_invFun.comp e₂.continuous_invFun }


@[simp]
theorem trans_toLinearEquiv (e₁ : M₁ ≃SL[σ₁₂] M₂) (e₂ : M₂ ≃SL[σ₂₃] M₃) :
    (e₁.trans e₂).toLinearEquiv = e₁.toLinearEquiv.trans e₂.toLinearEquiv := by
  /-
    R₁ : Type u_3
    R₂ : Type u_4
    R₃ : Type u_5
    inst✝¹⁹ : Semiring R₁
    inst✝¹⁸ : Semiring R₂
    inst✝¹⁷ : Semiring R₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    inst✝¹⁶ : RingHomInvPair σ₁₂ σ₂₁
    inst✝¹⁵ : RingHomInvPair σ₂₁ σ₁₂
    σ₂₃ : RingHom R₂ R₃
    σ₃₂ : RingHom R₃ R₂
    inst✝¹⁴ : RingHomInvPair σ₂₃ σ₃₂
    inst✝¹³ : RingHomInvPair σ₃₂ σ₂₃
    σ₁₃ : RingHom R₁ R₃
    σ₃₁ : RingHom R₃ R₁
    inst✝¹² : RingHomInvPair σ₁₃ σ₃₁
    inst✝¹¹ : RingHomInvPair σ₃₁ σ₁₃
    inst✝¹⁰ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    inst✝⁹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
    M₁ : Type u_6
    inst✝⁸ : TopologicalSpace M₁
    inst✝⁷ : AddCommMonoid M₁
    M₂ : Type u_7
    inst✝⁶ : TopologicalSpace M₂
    inst✝⁵ : AddCommMonoid M₂
    M₃ : Type u_8
    inst✝⁴ : TopologicalSpace M₃
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R₁ M₁
    inst✝¹ : Module R₂ M₂
    inst✝ : Module R₃ M₃
    e₁ : ContinuousLinearEquiv σ₁₂ M₁ M₂
    e₂ : ContinuousLinearEquiv σ₂₃ M₂ M₃
    ⊢ Eq (e₁.trans e₂).toLinearEquiv (e₁.trans e₂.toLinearEquiv)
  -/
  ext
  /-
    case h
    R₁ : Type u_3
    R₂ : Type u_4
    R₃ : Type u_5
    inst✝¹⁹ : Semiring R₁
    inst✝¹⁸ : Semiring R₂
    inst✝¹⁷ : Semiring R₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    inst✝¹⁶ : RingHomInvPair σ₁₂ σ₂₁
    inst✝¹⁵ : RingHomInvPair σ₂₁ σ₁₂
    σ₂₃ : RingHom R₂ R₃
    σ₃₂ : RingHom R₃ R₂
    inst✝¹⁴ : RingHomInvPair σ₂₃ σ₃₂
    inst✝¹³ : RingHomInvPair σ₃₂ σ₂₃
    σ₁₃ : RingHom R₁ R₃
    σ₃₁ : RingHom R₃ R₁
    inst✝¹² : RingHomInvPair σ₁₃ σ₃₁
    inst✝¹¹ : RingHomInvPair σ₃₁ σ₁₃
    inst✝¹⁰ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    inst✝⁹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
    M₁ : Type u_6
    inst✝⁸ : TopologicalSpace M₁
    inst✝⁷ : AddCommMonoid M₁
    M₂ : Type u_7
    inst✝⁶ : TopologicalSpace M₂
    inst✝⁵ : AddCommMonoid M₂
    M₃ : Type u_8
    inst✝⁴ : TopologicalSpace M₃
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R₁ M₁
    inst✝¹ : Module R₂ M₂
    inst✝ : Module R₃ M₃
    e₁ : ContinuousLinearEquiv σ₁₂ M₁ M₂
    e₂ : ContinuousLinearEquiv σ₂₃ M₂ M₃
    x✝ : M₁
    ⊢ Eq ((e₁.trans e₂).toLinearEquiv x✝) ((e₁.trans e₂.toLinearEquiv) x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Product of two continuous linear equivalences. The map comes from `Equiv.prodCongr`. -/
def prod [Module R₁ M₂] [Module R₁ M₃] [Module R₁ M₄] (e : M₁ ≃L[R₁] M₂) (e' : M₃ ≃L[R₁] M₄) :
    (M₁ × M₃) ≃L[R₁] M₂ × M₄ :=
  { e.toLinearEquiv.prod e'.toLinearEquiv with
    continuous_toFun := e.continuous_toFun.prodMap e'.continuous_toFun
    continuous_invFun := e.continuous_invFun.prodMap e'.continuous_invFun }


@[simp, norm_cast]
theorem prod_apply [Module R₁ M₂] [Module R₁ M₃] [Module R₁ M₄] (e : M₁ ≃L[R₁] M₂)
    (e' : M₃ ≃L[R₁] M₄) (x) : e.prod e' x = (e x.1, e' x.2) :=
  rfl


@[simp, norm_cast]
theorem coe_prod [Module R₁ M₂] [Module R₁ M₃] [Module R₁ M₄] (e : M₁ ≃L[R₁] M₂)
    (e' : M₃ ≃L[R₁] M₄) :
    (e.prod e' : M₁ × M₃ →L[R₁] M₂ × M₄) = (e : M₁ →L[R₁] M₂).prodMap (e' : M₃ →L[R₁] M₄) :=
  rfl


theorem prod_symm [Module R₁ M₂] [Module R₁ M₃] [Module R₁ M₄] (e : M₁ ≃L[R₁] M₂)
    (e' : M₃ ≃L[R₁] M₄) : (e.prod e').symm = e.symm.prod e'.symm :=
  rfl


/-- Product of modules is commutative up to continuous linear isomorphism. -/
@[simps! apply toLinearEquiv]
def prodComm [Module R₁ M₂] : (M₁ × M₂) ≃L[R₁] M₂ × M₁ :=
  { LinearEquiv.prodComm R₁ M₁ M₂ with
    continuous_toFun := continuous_swap
    continuous_invFun := continuous_swap }


@[simp] lemma prodComm_symm [Module R₁ M₂] : (prodComm R₁ M₁ M₂).symm = prodComm R₁ M₂ M₁ := rfl


protected theorem bijective (e : M₁ ≃SL[σ₁₂] M₂) : Function.Bijective e :=
  e.toLinearEquiv.toEquiv.bijective


protected theorem injective (e : M₁ ≃SL[σ₁₂] M₂) : Function.Injective e :=
  e.toLinearEquiv.toEquiv.injective


protected theorem surjective (e : M₁ ≃SL[σ₁₂] M₂) : Function.Surjective e :=
  e.toLinearEquiv.toEquiv.surjective


@[simp]
theorem trans_apply (e₁ : M₁ ≃SL[σ₁₂] M₂) (e₂ : M₂ ≃SL[σ₂₃] M₃) (c : M₁) :
    (e₁.trans e₂) c = e₂ (e₁ c) :=
  rfl


@[simp]
theorem apply_symm_apply (e : M₁ ≃SL[σ₁₂] M₂) (c : M₂) : e (e.symm c) = c :=
  e.1.right_inv c


@[simp]
theorem symm_apply_apply (e : M₁ ≃SL[σ₁₂] M₂) (b : M₁) : e.symm (e b) = b :=
  e.1.left_inv b


@[simp]
theorem symm_trans_apply (e₁ : M₂ ≃SL[σ₂₁] M₁) (e₂ : M₃ ≃SL[σ₃₂] M₂) (c : M₁) :
    (e₂.trans e₁).symm c = e₂.symm (e₁.symm c) :=
  rfl


@[simp]
theorem symm_image_image (e : M₁ ≃SL[σ₁₂] M₂) (s : Set M₁) : e.symm '' (e '' s) = s :=
  e.toLinearEquiv.toEquiv.symm_image_image s


@[simp]
theorem image_symm_image (e : M₁ ≃SL[σ₁₂] M₂) (s : Set M₂) : e '' (e.symm '' s) = s :=
  e.symm.symm_image_image s


@[simp, norm_cast]
theorem comp_coe (f : M₁ ≃SL[σ₁₂] M₂) (f' : M₂ ≃SL[σ₂₃] M₃) :
    (f' : M₂ →SL[σ₂₃] M₃).comp (f : M₁ →SL[σ₁₂] M₂) = (f.trans f' : M₁ →SL[σ₁₃] M₃) :=
  rfl

-- Porting note: The priority should be higher than `comp_coe`.

@[simp high]
theorem coe_comp_coe_symm (e : M₁ ≃SL[σ₁₂] M₂) :
    (e : M₁ →SL[σ₁₂] M₂).comp (e.symm : M₂ →SL[σ₂₁] M₁) = ContinuousLinearMap.id R₂ M₂ :=
  ContinuousLinearMap.ext e.apply_symm_apply

-- Porting note: The priority should be higher than `comp_coe`.

@[simp high]
theorem coe_symm_comp_coe (e : M₁ ≃SL[σ₁₂] M₂) :
    (e.symm : M₂ →SL[σ₂₁] M₁).comp (e : M₁ →SL[σ₁₂] M₂) = ContinuousLinearMap.id R₁ M₁ :=
  ContinuousLinearMap.ext e.symm_apply_apply


@[simp]
theorem symm_comp_self (e : M₁ ≃SL[σ₁₂] M₂) : (e.symm : M₂ → M₁) ∘ (e : M₁ → M₂) = id := by
  /-
    R₁ : Type u_3
    R₂ : Type u_4
    inst✝⁹ : Semiring R₁
    inst✝⁸ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    inst✝⁷ : RingHomInvPair σ₁₂ σ₂₁
    inst✝⁶ : RingHomInvPair σ₂₁ σ₁₂
    M₁ : Type u_6
    inst✝⁵ : TopologicalSpace M₁
    inst✝⁴ : AddCommMonoid M₁
    M₂ : Type u_7
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R₁ M₁
    inst✝ : Module R₂ M₂
    e : ContinuousLinearEquiv σ₁₂ M₁ M₂
    ⊢ Eq (Function.comp ⇑e.symm ⇑e) id
  -/
  ext x
  /-
    case h
    R₁ : Type u_3
    R₂ : Type u_4
    inst✝⁹ : Semiring R₁
    inst✝⁸ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    inst✝⁷ : RingHomInvPair σ₁₂ σ₂₁
    inst✝⁶ : RingHomInvPair σ₂₁ σ₁₂
    M₁ : Type u_6
    inst✝⁵ : TopologicalSpace M₁
    inst✝⁴ : AddCommMonoid M₁
    M₂ : Type u_7
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R₁ M₁
    inst✝ : Module R₂ M₂
    e : ContinuousLinearEquiv σ₁₂ M₁ M₂
    x : M₁
    ⊢ Eq (Function.comp (⇑e.symm) (⇑e) x) (id x)
  -/
  exact symm_apply_apply e x
  /-
    🎉 no goals
  -/


@[simp]
theorem self_comp_symm (e : M₁ ≃SL[σ₁₂] M₂) : (e : M₁ → M₂) ∘ (e.symm : M₂ → M₁) = id := by
  /-
    R₁ : Type u_3
    R₂ : Type u_4
    inst✝⁹ : Semiring R₁
    inst✝⁸ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    inst✝⁷ : RingHomInvPair σ₁₂ σ₂₁
    inst✝⁶ : RingHomInvPair σ₂₁ σ₁₂
    M₁ : Type u_6
    inst✝⁵ : TopologicalSpace M₁
    inst✝⁴ : AddCommMonoid M₁
    M₂ : Type u_7
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R₁ M₁
    inst✝ : Module R₂ M₂
    e : ContinuousLinearEquiv σ₁₂ M₁ M₂
    ⊢ Eq (Function.comp ⇑e ⇑e.symm) id
  -/
  ext x
  /-
    case h
    R₁ : Type u_3
    R₂ : Type u_4
    inst✝⁹ : Semiring R₁
    inst✝⁸ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    inst✝⁷ : RingHomInvPair σ₁₂ σ₂₁
    inst✝⁶ : RingHomInvPair σ₂₁ σ₁₂
    M₁ : Type u_6
    inst✝⁵ : TopologicalSpace M₁
    inst✝⁴ : AddCommMonoid M₁
    M₂ : Type u_7
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R₁ M₁
    inst✝ : Module R₂ M₂
    e : ContinuousLinearEquiv σ₁₂ M₁ M₂
    x : M₂
    ⊢ Eq (Function.comp (⇑e) (⇑e.symm) x) (id x)
  -/
  exact apply_symm_apply e x
  /-
    🎉 no goals
  -/


@[simp]
theorem symm_symm (e : M₁ ≃SL[σ₁₂] M₂) : e.symm.symm = e := rfl


@[simp]
theorem refl_symm : (ContinuousLinearEquiv.refl R₁ M₁).symm = ContinuousLinearEquiv.refl R₁ M₁ :=
  rfl


theorem symm_symm_apply (e : M₁ ≃SL[σ₁₂] M₂) (x : M₁) : e.symm.symm x = e x :=
  rfl


theorem symm_apply_eq (e : M₁ ≃SL[σ₁₂] M₂) {x y} : e.symm x = y ↔ x = e y :=
  e.toLinearEquiv.symm_apply_eq


theorem eq_symm_apply (e : M₁ ≃SL[σ₁₂] M₂) {x y} : y = e.symm x ↔ e y = x :=
  e.toLinearEquiv.eq_symm_apply


protected theorem image_eq_preimage (e : M₁ ≃SL[σ₁₂] M₂) (s : Set M₁) : e '' s = e.symm ⁻¹' s :=
  e.toLinearEquiv.toEquiv.image_eq_preimage s


protected theorem image_symm_eq_preimage (e : M₁ ≃SL[σ₁₂] M₂) (s : Set M₂) :
                                /-
                                  R₁ : Type u_3
                                  R₂ : Type u_4
                                  inst✝⁹ : Semiring R₁
                                  inst✝⁸ : Semiring R₂
                                  σ₁₂ : RingHom R₁ R₂
                                  σ₂₁ : RingHom R₂ R₁
                                  inst✝⁷ : RingHomInvPair σ₁₂ σ₂₁
                                  inst✝⁶ : RingHomInvPair σ₂₁ σ₁₂
                                  M₁ : Type u_6
                                  inst✝⁵ : TopologicalSpace M₁
                                  inst✝⁴ : AddCommMonoid M₁
                                  M₂ : Type u_7
                                  inst✝³ : TopologicalSpace M₂
                                  inst✝² : AddCommMonoid M₂
                                  inst✝¹ : Module R₁ M₁
                                  inst✝ : Module R₂ M₂
                                  e : ContinuousLinearEquiv σ₁₂ M₁ M₂
                                  s : Set M₂
                                  ⊢ Eq (Set.image (⇑e.symm) s) (Set.preimage (⇑e) s)
                                -/
    e.symm '' s = e ⁻¹' s := by rw [e.symm.image_eq_preimage, e.symm_symm]
                                /-
                                  🎉 no goals
                                -/


@[simp]
protected theorem symm_preimage_preimage (e : M₁ ≃SL[σ₁₂] M₂) (s : Set M₂) :
    e.symm ⁻¹' (e ⁻¹' s) = s :=
  e.toLinearEquiv.toEquiv.symm_preimage_preimage s


@[simp]
protected theorem preimage_symm_preimage (e : M₁ ≃SL[σ₁₂] M₂) (s : Set M₁) :
    e ⁻¹' (e.symm ⁻¹' s) = s :=
  e.symm.symm_preimage_preimage s


lemma isUniformEmbedding {E₁ E₂ : Type*} [UniformSpace E₁] [UniformSpace E₂]
    [AddCommGroup E₁] [AddCommGroup E₂] [Module R₁ E₁] [Module R₂ E₂] [UniformAddGroup E₁]
    [UniformAddGroup E₂] (e : E₁ ≃SL[σ₁₂] E₂) : IsUniformEmbedding e :=
  e.toLinearEquiv.toEquiv.isUniformEmbedding e.toContinuousLinearMap.uniformContinuous
    e.symm.toContinuousLinearMap.uniformContinuous


@[deprecated (since := "2024-10-01")] alias uniformEmbedding := isUniformEmbedding


protected theorem _root_.LinearEquiv.isUniformEmbedding {E₁ E₂ : Type*} [UniformSpace E₁]
    [UniformSpace E₂] [AddCommGroup E₁] [AddCommGroup E₂] [Module R₁ E₁] [Module R₂ E₂]
    [UniformAddGroup E₁] [UniformAddGroup E₂] (e : E₁ ≃ₛₗ[σ₁₂] E₂)
    (h₁ : Continuous e) (h₂ : Continuous e.symm) : IsUniformEmbedding e :=
  ContinuousLinearEquiv.isUniformEmbedding
    ({ e with
        continuous_toFun := h₁
        continuous_invFun := h₂ } :
      E₁ ≃SL[σ₁₂] E₂)


@[deprecated (since := "2024-10-01")]
alias _root_.LinearEquiv.uniformEmbedding := _root_.LinearEquiv.isUniformEmbedding


/-- Create a `ContinuousLinearEquiv` from two `ContinuousLinearMap`s that are
inverse of each other. See also `equivOfInverse'`. -/
def equivOfInverse (f₁ : M₁ →SL[σ₁₂] M₂) (f₂ : M₂ →SL[σ₂₁] M₁) (h₁ : Function.LeftInverse f₂ f₁)
    (h₂ : Function.RightInverse f₂ f₁) : M₁ ≃SL[σ₁₂] M₂ :=
  { f₁ with
    continuous_toFun := f₁.continuous
    invFun := f₂
    continuous_invFun := f₂.continuous
    left_inv := h₁
    right_inv := h₂ }


@[simp]
theorem equivOfInverse_apply (f₁ : M₁ →SL[σ₁₂] M₂) (f₂ h₁ h₂ x) :
    equivOfInverse f₁ f₂ h₁ h₂ x = f₁ x :=
  rfl


@[simp]
theorem symm_equivOfInverse (f₁ : M₁ →SL[σ₁₂] M₂) (f₂ h₁ h₂) :
    (equivOfInverse f₁ f₂ h₁ h₂).symm = equivOfInverse f₂ f₁ h₂ h₁ :=
  rfl


/-- Create a `ContinuousLinearEquiv` from two `ContinuousLinearMap`s that are
inverse of each other, in the `ContinuousLinearMap.comp` sense. See also `equivOfInverse`. -/
def equivOfInverse' (f₁ : M₁ →SL[σ₁₂] M₂) (f₂ : M₂ →SL[σ₂₁] M₁)
    (h₁ : f₁.comp f₂ = .id R₂ M₂) (h₂ : f₂.comp f₁ = .id R₁ M₁) : M₁ ≃SL[σ₁₂] M₂ :=
  equivOfInverse f₁ f₂
                /-
                  R : Type u_1
                  M : Type u_2
                  inst✝²⁸ : Ring R
                  inst✝²⁷ : TopologicalSpace R
                  inst✝²⁶ : TopologicalSpace M
                  inst✝²⁵ : AddCommGroup M
                  inst✝²⁴ : ContinuousAdd M
                  inst✝²³ : Module R M
                  inst✝²² : ContinuousSMul R M
                  R₁ : Type u_3
                  R₂ : Type u_4
                  R₃ : Type u_5
                  inst✝²¹ : Semiring R₁
                  inst✝²⁰ : Semiring R₂
                  inst✝¹⁹ : Semiring R₃
                  σ₁₂ : RingHom R₁ R₂
                  σ₂₁ : RingHom R₂ R₁
                  inst✝¹⁸ : RingHomInvPair σ₁₂ σ₂₁
                  inst✝¹⁷ : RingHomInvPair σ₂₁ σ₁₂
                  σ₂₃ : RingHom R₂ R₃
                  σ₃₂ : RingHom R₃ R₂
                  inst✝¹⁶ : RingHomInvPair σ₂₃ σ₃₂
                  inst✝¹⁵ : RingHomInvPair σ₃₂ σ₂₃
                  σ₁₃ : RingHom R₁ R₃
                  σ₃₁ : RingHom R₃ R₁
                  inst✝¹⁴ : RingHomInvPair σ₁₃ σ₃₁
                  inst✝¹³ : RingHomInvPair σ₃₁ σ₁₃
                  inst✝¹² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                  inst✝¹¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
                  M₁ : Type u_6
                  inst✝¹⁰ : TopologicalSpace M₁
                  inst✝⁹ : AddCommMonoid M₁
                  M₂ : Type u_7
                  inst✝⁸ : TopologicalSpace M₂
                  inst✝⁷ : AddCommMonoid M₂
                  M₃ : Type u_8
                  inst✝⁶ : TopologicalSpace M₃
                  inst✝⁵ : AddCommMonoid M₃
                  M₄ : Type u_9
                  inst✝⁴ : TopologicalSpace M₄
                  inst✝³ : AddCommMonoid M₄
                  inst✝² : Module R₁ M₁
                  inst✝¹ : Module R₂ M₂
                  inst✝ : Module R₃ M₃
                  f₁ : ContinuousLinearMap σ₁₂ M₁ M₂
                  f₂ : ContinuousLinearMap σ₂₁ M₂ M₁
                  h₁ : Eq (f₁.comp f₂) (ContinuousLinearMap.id R₂ M₂)
                  h₂ : Eq (f₂.comp f₁) (ContinuousLinearMap.id R₁ M₁)
                  x : M₁
                  ⊢ Eq (f₂ (f₁ x)) x
                -/
                /-
                  🎉 no goals
                -/
    (fun x ↦ by simpa using congr($(h₂) x)) (fun x ↦ by simpa using congr($(h₁) x))
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem equivOfInverse'_apply (f₁ : M₁ →SL[σ₁₂] M₂) (f₂ h₁ h₂ x) :
    equivOfInverse' f₁ f₂ h₁ h₂ x = f₁ x :=
  rfl


/-- The inverse of `equivOfInverse'` is obtained by swapping the order of its parameters. -/
@[simp]
theorem symm_equivOfInverse' (f₁ : M₁ →SL[σ₁₂] M₂) (f₂ h₁ h₂) :
    (equivOfInverse' f₁ f₂ h₁ h₂).symm = equivOfInverse' f₂ f₁ h₂ h₁ :=
  rfl


/-- The continuous linear equivalences from `M` to itself form a group under composition. -/
instance automorphismGroup : Group (M₁ ≃L[R₁] M₁) where
  mul f g := g.trans f
  one := ContinuousLinearEquiv.refl R₁ M₁
  inv f := f.symm
  mul_assoc f g h := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝²⁸ : Ring R
      inst✝²⁷ : TopologicalSpace R
      inst✝²⁶ : TopologicalSpace M
      inst✝²⁵ : AddCommGroup M
      inst✝²⁴ : ContinuousAdd M
      inst✝²³ : Module R M
      inst✝²² : ContinuousSMul R M
      R₁ : Type u_3
      R₂ : Type u_4
      R₃ : Type u_5
      inst✝²¹ : Semiring R₁
      inst✝²⁰ : Semiring R₂
      inst✝¹⁹ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₁ : RingHom R₂ R₁
      inst✝¹⁸ : RingHomInvPair σ₁₂ σ₂₁
      inst✝¹⁷ : RingHomInvPair σ₂₁ σ₁₂
      σ₂₃ : RingHom R₂ R₃
      σ₃₂ : RingHom R₃ R₂
      inst✝¹⁶ : RingHomInvPair σ₂₃ σ₃₂
      inst✝¹⁵ : RingHomInvPair σ₃₂ σ₂₃
      σ₁₃ : RingHom R₁ R₃
      σ₃₁ : RingHom R₃ R₁
      inst✝¹⁴ : RingHomInvPair σ₁₃ σ₃₁
      inst✝¹³ : RingHomInvPair σ₃₁ σ₁₃
      inst✝¹² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝¹¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
      M₁ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₁
      inst✝⁹ : AddCommMonoid M₁
      M₂ : Type u_7
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : AddCommMonoid M₂
      M₃ : Type u_8
      inst✝⁶ : TopologicalSpace M₃
      inst✝⁵ : AddCommMonoid M₃
      M₄ : Type u_9
      inst✝⁴ : TopologicalSpace M₄
      inst✝³ : AddCommMonoid M₄
      inst✝² : Module R₁ M₁
      inst✝¹ : Module R₂ M₂
      inst✝ : Module R₃ M₃
      f g h : ContinuousLinearEquiv (RingHom.id R₁) M₁ M₁
      ⊢ Eq (HMul.hMul (HMul.hMul f g) h) (HMul.hMul f (HMul.hMul g h))
    -/
    ext
    /-
      case h.h
      R : Type u_1
      M : Type u_2
      inst✝²⁸ : Ring R
      inst✝²⁷ : TopologicalSpace R
      inst✝²⁶ : TopologicalSpace M
      inst✝²⁵ : AddCommGroup M
      inst✝²⁴ : ContinuousAdd M
      inst✝²³ : Module R M
      inst✝²² : ContinuousSMul R M
      R₁ : Type u_3
      R₂ : Type u_4
      R₃ : Type u_5
      inst✝²¹ : Semiring R₁
      inst✝²⁰ : Semiring R₂
      inst✝¹⁹ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₁ : RingHom R₂ R₁
      inst✝¹⁸ : RingHomInvPair σ₁₂ σ₂₁
      inst✝¹⁷ : RingHomInvPair σ₂₁ σ₁₂
      σ₂₃ : RingHom R₂ R₃
      σ₃₂ : RingHom R₃ R₂
      inst✝¹⁶ : RingHomInvPair σ₂₃ σ₃₂
      inst✝¹⁵ : RingHomInvPair σ₃₂ σ₂₃
      σ₁₃ : RingHom R₁ R₃
      σ₃₁ : RingHom R₃ R₁
      inst✝¹⁴ : RingHomInvPair σ₁₃ σ₃₁
      inst✝¹³ : RingHomInvPair σ₃₁ σ₁₃
      inst✝¹² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝¹¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
      M₁ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₁
      inst✝⁹ : AddCommMonoid M₁
      M₂ : Type u_7
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : AddCommMonoid M₂
      M₃ : Type u_8
      inst✝⁶ : TopologicalSpace M₃
      inst✝⁵ : AddCommMonoid M₃
      M₄ : Type u_9
      inst✝⁴ : TopologicalSpace M₄
      inst✝³ : AddCommMonoid M₄
      inst✝² : Module R₁ M₁
      inst✝¹ : Module R₂ M₂
      inst✝ : Module R₃ M₃
      f g h : ContinuousLinearEquiv (RingHom.id R₁) M₁ M₁
      x✝ : M₁
      ⊢ Eq ((HMul.hMul (HMul.hMul f g) h) x✝) ((HMul.hMul f (HMul.hMul g h)) x✝)
    -/
    rfl
    /-
      🎉 no goals
    -/
  mul_one f := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝²⁸ : Ring R
      inst✝²⁷ : TopologicalSpace R
      inst✝²⁶ : TopologicalSpace M
      inst✝²⁵ : AddCommGroup M
      inst✝²⁴ : ContinuousAdd M
      inst✝²³ : Module R M
      inst✝²² : ContinuousSMul R M
      R₁ : Type u_3
      R₂ : Type u_4
      R₃ : Type u_5
      inst✝²¹ : Semiring R₁
      inst✝²⁰ : Semiring R₂
      inst✝¹⁹ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₁ : RingHom R₂ R₁
      inst✝¹⁸ : RingHomInvPair σ₁₂ σ₂₁
      inst✝¹⁷ : RingHomInvPair σ₂₁ σ₁₂
      σ₂₃ : RingHom R₂ R₃
      σ₃₂ : RingHom R₃ R₂
      inst✝¹⁶ : RingHomInvPair σ₂₃ σ₃₂
      inst✝¹⁵ : RingHomInvPair σ₃₂ σ₂₃
      σ₁₃ : RingHom R₁ R₃
      σ₃₁ : RingHom R₃ R₁
      inst✝¹⁴ : RingHomInvPair σ₁₃ σ₃₁
      inst✝¹³ : RingHomInvPair σ₃₁ σ₁₃
      inst✝¹² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝¹¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
      M₁ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₁
      inst✝⁹ : AddCommMonoid M₁
      M₂ : Type u_7
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : AddCommMonoid M₂
      M₃ : Type u_8
      inst✝⁶ : TopologicalSpace M₃
      inst✝⁵ : AddCommMonoid M₃
      M₄ : Type u_9
      inst✝⁴ : TopologicalSpace M₄
      inst✝³ : AddCommMonoid M₄
      inst✝² : Module R₁ M₁
      inst✝¹ : Module R₂ M₂
      inst✝ : Module R₃ M₃
      f : ContinuousLinearEquiv (RingHom.id R₁) M₁ M₁
      ⊢ Eq (HMul.hMul f 1) f
    -/
    ext
    /-
      case h.h
      R : Type u_1
      M : Type u_2
      inst✝²⁸ : Ring R
      inst✝²⁷ : TopologicalSpace R
      inst✝²⁶ : TopologicalSpace M
      inst✝²⁵ : AddCommGroup M
      inst✝²⁴ : ContinuousAdd M
      inst✝²³ : Module R M
      inst✝²² : ContinuousSMul R M
      R₁ : Type u_3
      R₂ : Type u_4
      R₃ : Type u_5
      inst✝²¹ : Semiring R₁
      inst✝²⁰ : Semiring R₂
      inst✝¹⁹ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₁ : RingHom R₂ R₁
      inst✝¹⁸ : RingHomInvPair σ₁₂ σ₂₁
      inst✝¹⁷ : RingHomInvPair σ₂₁ σ₁₂
      σ₂₃ : RingHom R₂ R₃
      σ₃₂ : RingHom R₃ R₂
      inst✝¹⁶ : RingHomInvPair σ₂₃ σ₃₂
      inst✝¹⁵ : RingHomInvPair σ₃₂ σ₂₃
      σ₁₃ : RingHom R₁ R₃
      σ₃₁ : RingHom R₃ R₁
      inst✝¹⁴ : RingHomInvPair σ₁₃ σ₃₁
      inst✝¹³ : RingHomInvPair σ₃₁ σ₁₃
      inst✝¹² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝¹¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
      M₁ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₁
      inst✝⁹ : AddCommMonoid M₁
      M₂ : Type u_7
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : AddCommMonoid M₂
      M₃ : Type u_8
      inst✝⁶ : TopologicalSpace M₃
      inst✝⁵ : AddCommMonoid M₃
      M₄ : Type u_9
      inst✝⁴ : TopologicalSpace M₄
      inst✝³ : AddCommMonoid M₄
      inst✝² : Module R₁ M₁
      inst✝¹ : Module R₂ M₂
      inst✝ : Module R₃ M₃
      f : ContinuousLinearEquiv (RingHom.id R₁) M₁ M₁
      x✝ : M₁
      ⊢ Eq ((HMul.hMul f 1) x✝) (f x✝)
    -/
    /-
      R : Type u_1
      M : Type u_2
      inst✝²⁸ : Ring R
      inst✝²⁷ : TopologicalSpace R
      inst✝²⁶ : TopologicalSpace M
      inst✝²⁵ : AddCommGroup M
      inst✝²⁴ : ContinuousAdd M
      inst✝²³ : Module R M
      inst✝²² : ContinuousSMul R M
      R₁ : Type u_3
      R₂ : Type u_4
      R₃ : Type u_5
      inst✝²¹ : Semiring R₁
      inst✝²⁰ : Semiring R₂
      inst✝¹⁹ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₁ : RingHom R₂ R₁
      inst✝¹⁸ : RingHomInvPair σ₁₂ σ₂₁
      inst✝¹⁷ : RingHomInvPair σ₂₁ σ₁₂
      σ₂₃ : RingHom R₂ R₃
      σ₃₂ : RingHom R₃ R₂
      inst✝¹⁶ : RingHomInvPair σ₂₃ σ₃₂
      inst✝¹⁵ : RingHomInvPair σ₃₂ σ₂₃
      σ₁₃ : RingHom R₁ R₃
      σ₃₁ : RingHom R₃ R₁
      inst✝¹⁴ : RingHomInvPair σ₁₃ σ₃₁
      inst✝¹³ : RingHomInvPair σ₃₁ σ₁₃
      inst✝¹² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝¹¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
      M₁ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₁
      inst✝⁹ : AddCommMonoid M₁
      M₂ : Type u_7
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : AddCommMonoid M₂
      M₃ : Type u_8
      inst✝⁶ : TopologicalSpace M₃
      inst✝⁵ : AddCommMonoid M₃
      M₄ : Type u_9
      inst✝⁴ : TopologicalSpace M₄
      inst✝³ : AddCommMonoid M₄
      inst✝² : Module R₁ M₁
      inst✝¹ : Module R₂ M₂
      inst✝ : Module R₃ M₃
      f : ContinuousLinearEquiv (RingHom.id R₁) M₁ M₁
      ⊢ Eq (HMul.hMul 1 f) f
    -/
    rfl
    /-
      case h.h
      R : Type u_1
      M : Type u_2
      inst✝²⁸ : Ring R
      inst✝²⁷ : TopologicalSpace R
      inst✝²⁶ : TopologicalSpace M
      inst✝²⁵ : AddCommGroup M
      inst✝²⁴ : ContinuousAdd M
      inst✝²³ : Module R M
      inst✝²² : ContinuousSMul R M
      R₁ : Type u_3
      R₂ : Type u_4
      R₃ : Type u_5
      inst✝²¹ : Semiring R₁
      inst✝²⁰ : Semiring R₂
      inst✝¹⁹ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₁ : RingHom R₂ R₁
      inst✝¹⁸ : RingHomInvPair σ₁₂ σ₂₁
      inst✝¹⁷ : RingHomInvPair σ₂₁ σ₁₂
      σ₂₃ : RingHom R₂ R₃
      σ₃₂ : RingHom R₃ R₂
      inst✝¹⁶ : RingHomInvPair σ₂₃ σ₃₂
      inst✝¹⁵ : RingHomInvPair σ₃₂ σ₂₃
      σ₁₃ : RingHom R₁ R₃
      σ₃₁ : RingHom R₃ R₁
      inst✝¹⁴ : RingHomInvPair σ₁₃ σ₃₁
      inst✝¹³ : RingHomInvPair σ₃₁ σ₁₃
      inst✝¹² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝¹¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
      M₁ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₁
      inst✝⁹ : AddCommMonoid M₁
      M₂ : Type u_7
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : AddCommMonoid M₂
      M₃ : Type u_8
      inst✝⁶ : TopologicalSpace M₃
      inst✝⁵ : AddCommMonoid M₃
      M₄ : Type u_9
      inst✝⁴ : TopologicalSpace M₄
      inst✝³ : AddCommMonoid M₄
      inst✝² : Module R₁ M₁
      inst✝¹ : Module R₂ M₂
      inst✝ : Module R₃ M₃
      f : ContinuousLinearEquiv (RingHom.id R₁) M₁ M₁
      x✝ : M₁
      ⊢ Eq ((HMul.hMul 1 f) x✝) (f x✝)
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  one_mul f := by
    ext
    rfl
  inv_mul_cancel f := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝²⁸ : Ring R
      inst✝²⁷ : TopologicalSpace R
      inst✝²⁶ : TopologicalSpace M
      inst✝²⁵ : AddCommGroup M
      inst✝²⁴ : ContinuousAdd M
      inst✝²³ : Module R M
      inst✝²² : ContinuousSMul R M
      R₁ : Type u_3
      R₂ : Type u_4
      R₃ : Type u_5
      inst✝²¹ : Semiring R₁
      inst✝²⁰ : Semiring R₂
      inst✝¹⁹ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₁ : RingHom R₂ R₁
      inst✝¹⁸ : RingHomInvPair σ₁₂ σ₂₁
      inst✝¹⁷ : RingHomInvPair σ₂₁ σ₁₂
      σ₂₃ : RingHom R₂ R₃
      σ₃₂ : RingHom R₃ R₂
      inst✝¹⁶ : RingHomInvPair σ₂₃ σ₃₂
      inst✝¹⁵ : RingHomInvPair σ₃₂ σ₂₃
      σ₁₃ : RingHom R₁ R₃
      σ₃₁ : RingHom R₃ R₁
      inst✝¹⁴ : RingHomInvPair σ₁₃ σ₃₁
      inst✝¹³ : RingHomInvPair σ₃₁ σ₁₃
      inst✝¹² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝¹¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
      M₁ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₁
      inst✝⁹ : AddCommMonoid M₁
      M₂ : Type u_7
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : AddCommMonoid M₂
      M₃ : Type u_8
      inst✝⁶ : TopologicalSpace M₃
      inst✝⁵ : AddCommMonoid M₃
      M₄ : Type u_9
      inst✝⁴ : TopologicalSpace M₄
      inst✝³ : AddCommMonoid M₄
      inst✝² : Module R₁ M₁
      inst✝¹ : Module R₂ M₂
      inst✝ : Module R₃ M₃
      f : ContinuousLinearEquiv (RingHom.id R₁) M₁ M₁
      ⊢ Eq (HMul.hMul (Inv.inv f) f) 1
    -/
    ext x
    /-
      case h.h
      R : Type u_1
      M : Type u_2
      inst✝²⁸ : Ring R
      inst✝²⁷ : TopologicalSpace R
      inst✝²⁶ : TopologicalSpace M
      inst✝²⁵ : AddCommGroup M
      inst✝²⁴ : ContinuousAdd M
      inst✝²³ : Module R M
      inst✝²² : ContinuousSMul R M
      R₁ : Type u_3
      R₂ : Type u_4
      R₃ : Type u_5
      inst✝²¹ : Semiring R₁
      inst✝²⁰ : Semiring R₂
      inst✝¹⁹ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₁ : RingHom R₂ R₁
      inst✝¹⁸ : RingHomInvPair σ₁₂ σ₂₁
      inst✝¹⁷ : RingHomInvPair σ₂₁ σ₁₂
      σ₂₃ : RingHom R₂ R₃
      σ₃₂ : RingHom R₃ R₂
      inst✝¹⁶ : RingHomInvPair σ₂₃ σ₃₂
      inst✝¹⁵ : RingHomInvPair σ₃₂ σ₂₃
      σ₁₃ : RingHom R₁ R₃
      σ₃₁ : RingHom R₃ R₁
      inst✝¹⁴ : RingHomInvPair σ₁₃ σ₃₁
      inst✝¹³ : RingHomInvPair σ₃₁ σ₁₃
      inst✝¹² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      inst✝¹¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
      M₁ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₁
      inst✝⁹ : AddCommMonoid M₁
      M₂ : Type u_7
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : AddCommMonoid M₂
      M₃ : Type u_8
      inst✝⁶ : TopologicalSpace M₃
      inst✝⁵ : AddCommMonoid M₃
      M₄ : Type u_9
      inst✝⁴ : TopologicalSpace M₄
      inst✝³ : AddCommMonoid M₄
      inst✝² : Module R₁ M₁
      inst✝¹ : Module R₂ M₂
      inst✝ : Module R₃ M₃
      f : ContinuousLinearEquiv (RingHom.id R₁) M₁ M₁
      x : M₁
      ⊢ Eq ((HMul.hMul (Inv.inv f) f) x) (1 x)
    -/
    exact f.left_inv x
    /-
      🎉 no goals
    -/


/-- The continuous linear equivalence between `ULift M₁` and `M₁`.

This is a continuous version of `ULift.moduleEquiv`. -/
def ulift : ULift M₁ ≃L[R₁] M₁ :=
  { ULift.moduleEquiv with
    continuous_toFun := continuous_uLift_down
    continuous_invFun := continuous_uLift_up }


/-- A pair of continuous (semi)linear equivalences generates an equivalence between the spaces of
continuous linear maps. See also `ContinuousLinearEquiv.arrowCongr`. -/
@[simps]
def arrowCongrEquiv (e₁₂ : M₁ ≃SL[σ₁₂] M₂) (e₄₃ : M₄ ≃SL[σ₄₃] M₃) :
    (M₁ →SL[σ₁₄] M₄) ≃ (M₂ →SL[σ₂₃] M₃) where
  toFun f := (e₄₃ : M₄ →SL[σ₄₃] M₃).comp (f.comp (e₁₂.symm : M₂ →SL[σ₂₁] M₁))
  invFun f := (e₄₃.symm : M₃ →SL[σ₃₄] M₄).comp (f.comp (e₁₂ : M₁ →SL[σ₁₂] M₂))
  left_inv f :=
    ContinuousLinearMap.ext fun x => by
      /-
        R : Type u_1
        M : Type u_2
        inst✝³⁵ : Ring R
        inst✝³⁴ : TopologicalSpace R
        inst✝³³ : TopologicalSpace M
        inst✝³² : AddCommGroup M
        inst✝³¹ : ContinuousAdd M
        inst✝³⁰ : Module R M
        inst✝²⁹ : ContinuousSMul R M
        R₁ : Type u_3
        R₂ : Type u_4
        R₃ : Type u_5
        inst✝²⁸ : Semiring R₁
        inst✝²⁷ : Semiring R₂
        inst✝²⁶ : Semiring R₃
        σ₁₂ : RingHom R₁ R₂
        σ₂₁ : RingHom R₂ R₁
        inst✝²⁵ : RingHomInvPair σ₁₂ σ₂₁
        inst✝²⁴ : RingHomInvPair σ₂₁ σ₁₂
        σ₂₃ : RingHom R₂ R₃
        σ₃₂ : RingHom R₃ R₂
        inst✝²³ : RingHomInvPair σ₂₃ σ₃₂
        inst✝²² : RingHomInvPair σ₃₂ σ₂₃
        σ₁₃ : RingHom R₁ R₃
        σ₃₁ : RingHom R₃ R₁
        inst✝²¹ : RingHomInvPair σ₁₃ σ₃₁
        inst✝²⁰ : RingHomInvPair σ₃₁ σ₁₃
        inst✝¹⁹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        inst✝¹⁸ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
        M₁ : Type u_6
        inst✝¹⁷ : TopologicalSpace M₁
        inst✝¹⁶ : AddCommMonoid M₁
        M₂ : Type u_7
        inst✝¹⁵ : TopologicalSpace M₂
        inst✝¹⁴ : AddCommMonoid M₂
        M₃ : Type u_8
        inst✝¹³ : TopologicalSpace M₃
        inst✝¹² : AddCommMonoid M₃
        M₄ : Type u_9
        inst✝¹¹ : TopologicalSpace M₄
        inst✝¹⁰ : AddCommMonoid M₄
        inst✝⁹ : Module R₁ M₁
        inst✝⁸ : Module R₂ M₂
        inst✝⁷ : Module R₃ M₃
        R₄ : Type u_10
        inst✝⁶ : Semiring R₄
        inst✝⁵ : Module R₄ M₄
        σ₃₄ : RingHom R₃ R₄
        σ₄₃ : RingHom R₄ R₃
        inst✝⁴ : RingHomInvPair σ₃₄ σ₄₃
        inst✝³ : RingHomInvPair σ₄₃ σ₃₄
        σ₂₄ : RingHom R₂ R₄
        σ₁₄ : RingHom R₁ R₄
        inst✝² : RingHomCompTriple σ₂₁ σ₁₄ σ₂₄
        inst✝¹ : RingHomCompTriple σ₂₄ σ₄₃ σ₂₃
        inst✝ : RingHomCompTriple σ₁₃ σ₃₄ σ₁₄
        e₁₂ : ContinuousLinearEquiv σ₁₂ M₁ M₂
        e₄₃ : ContinuousLinearEquiv σ₄₃ M₄ M₃
        f : ContinuousLinearMap σ₁₄ M₁ M₄
        x : M₁
        ⊢ Eq (((fun f => (↑e₄₃.symm).comp (f.comp ↑e₁₂)) ((fun f => (↑e₄₃).comp (f.com …
      -/
      simp only [ContinuousLinearMap.comp_apply, symm_apply_apply, coe_coe]
      /-
        🎉 no goals
      -/
  right_inv f :=
    ContinuousLinearMap.ext fun x => by
      /-
        R : Type u_1
        M : Type u_2
        inst✝³⁵ : Ring R
        inst✝³⁴ : TopologicalSpace R
        inst✝³³ : TopologicalSpace M
        inst✝³² : AddCommGroup M
        inst✝³¹ : ContinuousAdd M
        inst✝³⁰ : Module R M
        inst✝²⁹ : ContinuousSMul R M
        R₁ : Type u_3
        R₂ : Type u_4
        R₃ : Type u_5
        inst✝²⁸ : Semiring R₁
        inst✝²⁷ : Semiring R₂
        inst✝²⁶ : Semiring R₃
        σ₁₂ : RingHom R₁ R₂
        σ₂₁ : RingHom R₂ R₁
        inst✝²⁵ : RingHomInvPair σ₁₂ σ₂₁
        inst✝²⁴ : RingHomInvPair σ₂₁ σ₁₂
        σ₂₃ : RingHom R₂ R₃
        σ₃₂ : RingHom R₃ R₂
        inst✝²³ : RingHomInvPair σ₂₃ σ₃₂
        inst✝²² : RingHomInvPair σ₃₂ σ₂₃
        σ₁₃ : RingHom R₁ R₃
        σ₃₁ : RingHom R₃ R₁
        inst✝²¹ : RingHomInvPair σ₁₃ σ₃₁
        inst✝²⁰ : RingHomInvPair σ₃₁ σ₁₃
        inst✝¹⁹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        inst✝¹⁸ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
        M₁ : Type u_6
        inst✝¹⁷ : TopologicalSpace M₁
        inst✝¹⁶ : AddCommMonoid M₁
        M₂ : Type u_7
        inst✝¹⁵ : TopologicalSpace M₂
        inst✝¹⁴ : AddCommMonoid M₂
        M₃ : Type u_8
        inst✝¹³ : TopologicalSpace M₃
        inst✝¹² : AddCommMonoid M₃
        M₄ : Type u_9
        inst✝¹¹ : TopologicalSpace M₄
        inst✝¹⁰ : AddCommMonoid M₄
        inst✝⁹ : Module R₁ M₁
        inst✝⁸ : Module R₂ M₂
        inst✝⁷ : Module R₃ M₃
        R₄ : Type u_10
        inst✝⁶ : Semiring R₄
        inst✝⁵ : Module R₄ M₄
        σ₃₄ : RingHom R₃ R₄
        σ₄₃ : RingHom R₄ R₃
        inst✝⁴ : RingHomInvPair σ₃₄ σ₄₃
        inst✝³ : RingHomInvPair σ₄₃ σ₃₄
        σ₂₄ : RingHom R₂ R₄
        σ₁₄ : RingHom R₁ R₄
        inst✝² : RingHomCompTriple σ₂₁ σ₁₄ σ₂₄
        inst✝¹ : RingHomCompTriple σ₂₄ σ₄₃ σ₂₃
        inst✝ : RingHomCompTriple σ₁₃ σ₃₄ σ₁₄
        e₁₂ : ContinuousLinearEquiv σ₁₂ M₁ M₂
        e₄₃ : ContinuousLinearEquiv σ₄₃ M₄ M₃
        f : ContinuousLinearMap σ₂₃ M₂ M₃
        x : M₂
        ⊢ Eq (((fun f => (↑e₄₃).comp (f.comp ↑e₁₂.symm)) ((fun f => (↑e₄₃.symm).comp ( …
      -/
      simp only [ContinuousLinearMap.comp_apply, apply_symm_apply, coe_coe]
      /-
        🎉 no goals
      -/


/-- Combine a family of linear equivalences into a linear equivalence of `pi`-types.
This is `Equiv.piCongrLeft` as a `ContinuousLinearEquiv`.
-/
def piCongrLeft (R : Type*) [Semiring R] {ι ι' : Type*}
    (φ : ι → Type*) [∀ i, AddCommMonoid (φ i)] [∀ i, Module R (φ i)]
    [∀ i, TopologicalSpace (φ i)]
    (e : ι' ≃ ι) : ((i' : ι') → φ (e i')) ≃L[R] (i : ι) → φ i where
  __ := Homeomorph.piCongrLeft e
  __ := LinearEquiv.piCongrLeft R φ e


/-- The product over `S ⊕ T` of a family of topological modules
is isomorphic (topologically and alegbraically) to the product of
(the product over `S`) and (the product over `T`).

This is `Equiv.sumPiEquivProdPi` as a `ContinuousLinearEquiv`.
-/
def sumPiEquivProdPi (R : Type*) [Semiring R] (S T : Type*)
    (A : S ⊕ T → Type*) [∀ st, AddCommMonoid (A st)] [∀ st, Module R (A st)]
    [∀ st, TopologicalSpace (A st)] :
    ((st : S ⊕ T) → A st) ≃L[R] ((s : S) → A (Sum.inl s)) × ((t : T) → A (Sum.inr t)) where
  __ := LinearEquiv.sumPiEquivProdPi R S T A
  __ := Homeomorph.sumPiEquivProdPi S T A


/-- The product `Π t : α, f t` of a family of topological modules is isomorphic
(both topologically and algebraically) to the space `f ⬝` when `α` only contains `⬝`.

This is `Equiv.piUnique` as a `ContinuousLinearEquiv`.
-/
@[simps! (config := .asFn)]
def piUnique {α : Type*} [Unique α] (R : Type*) [Semiring R] (f : α → Type*)
    [∀ x, AddCommMonoid (f x)] [∀ x, Module R (f x)] [∀ x, TopologicalSpace (f x)] :
    (Π t, f t) ≃L[R] f default where
  __ := LinearEquiv.piUnique R f
  __ := Homeomorph.piUnique f


/-- Combine a family of continuous linear equivalences into a continuous linear equivalence of
pi-types. -/
def piCongrRight : ((i : ι) → M i) ≃L[R₁] (i : ι) → N i :=
  { LinearEquiv.piCongrRight fun i ↦ f i with
    continuous_toFun := by
      /-
        R : Type u_1
        M✝ : Type u_2
        inst✝⁴¹ : Ring R
        inst✝⁴⁰ : TopologicalSpace R
        inst✝³⁹ : TopologicalSpace M✝
        inst✝³⁸ : AddCommGroup M✝
        inst✝³⁷ : ContinuousAdd M✝
        inst✝³⁶ : Module R M✝
        inst✝³⁵ : ContinuousSMul R M✝
        R₁ : Type u_3
        R₂ : Type u_4
        R₃ : Type u_5
        inst✝³⁴ : Semiring R₁
        inst✝³³ : Semiring R₂
        inst✝³² : Semiring R₃
        σ₁₂ : RingHom R₁ R₂
        σ₂₁ : RingHom R₂ R₁
        inst✝³¹ : RingHomInvPair σ₁₂ σ₂₁
        inst✝³⁰ : RingHomInvPair σ₂₁ σ₁₂
        σ₂₃ : RingHom R₂ R₃
        σ₃₂ : RingHom R₃ R₂
        inst✝²⁹ : RingHomInvPair σ₂₃ σ₃₂
        inst✝²⁸ : RingHomInvPair σ₃₂ σ₂₃
        σ₁₃ : RingHom R₁ R₃
        σ₃₁ : RingHom R₃ R₁
        inst✝²⁷ : RingHomInvPair σ₁₃ σ₃₁
        inst✝²⁶ : RingHomInvPair σ₃₁ σ₁₃
        inst✝²⁵ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        inst✝²⁴ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
        M₁ : Type u_6
        inst✝²³ : TopologicalSpace M₁
        inst✝²² : AddCommMonoid M₁
        M₂ : Type u_7
        inst✝²¹ : TopologicalSpace M₂
        inst✝²⁰ : AddCommMonoid M₂
        M₃ : Type u_8
        inst✝¹⁹ : TopologicalSpace M₃
        inst✝¹⁸ : AddCommMonoid M₃
        M₄ : Type u_9
        inst✝¹⁷ : TopologicalSpace M₄
        inst✝¹⁶ : AddCommMonoid M₄
        inst✝¹⁵ : Module R₁ M₁
        inst✝¹⁴ : Module R₂ M₂
        inst✝¹³ : Module R₃ M₃
        R₄ : Type u_10
        inst✝¹² : Semiring R₄
        inst✝¹¹ : Module R₄ M₄
        σ₃₄ : RingHom R₃ R₄
        σ₄₃ : RingHom R₄ R₃
        inst✝¹⁰ : RingHomInvPair σ₃₄ σ₄₃
        inst✝⁹ : RingHomInvPair σ₄₃ σ₃₄
        σ₂₄ : RingHom R₂ R₄
        σ₁₄ : RingHom R₁ R₄
        inst✝⁸ : RingHomCompTriple σ₂₁ σ₁₄ σ₂₄
        inst✝⁷ : RingHomCompTriple σ₂₄ σ₄₃ σ₂₃
        inst✝⁶ : RingHomCompTriple σ₁₃ σ₃₄ σ₁₄
        ι : Type u_11
        M : ι → Type u_12
        inst✝⁵ : (i : ι) → TopologicalSpace (M i)
        inst✝⁴ : (i : ι) → AddCommMonoid (M i)
        inst✝³ : (i : ι) → Module R₁ (M i)
        N : ι → Type u_13
        inst✝² : (i : ι) → TopologicalSpace (N i)
        inst✝¹ : (i : ι) → AddCommMonoid (N i)
        inst✝ : (i : ι) → Module R₁ (N i)
        f : (i : ι) → ContinuousLinearEquiv (RingHom.id R₁) (M i) (N i)
        ⊢ Continuous (↑__src✝).toFun
      -/
      exact continuous_pi fun i ↦ (f i).continuous_toFun.comp (continuous_apply i)
      /-
        🎉 no goals
      -/
    continuous_invFun := by
      /-
        R : Type u_1
        M✝ : Type u_2
        inst✝⁴¹ : Ring R
        inst✝⁴⁰ : TopologicalSpace R
        inst✝³⁹ : TopologicalSpace M✝
        inst✝³⁸ : AddCommGroup M✝
        inst✝³⁷ : ContinuousAdd M✝
        inst✝³⁶ : Module R M✝
        inst✝³⁵ : ContinuousSMul R M✝
        R₁ : Type u_3
        R₂ : Type u_4
        R₃ : Type u_5
        inst✝³⁴ : Semiring R₁
        inst✝³³ : Semiring R₂
        inst✝³² : Semiring R₃
        σ₁₂ : RingHom R₁ R₂
        σ₂₁ : RingHom R₂ R₁
        inst✝³¹ : RingHomInvPair σ₁₂ σ₂₁
        inst✝³⁰ : RingHomInvPair σ₂₁ σ₁₂
        σ₂₃ : RingHom R₂ R₃
        σ₃₂ : RingHom R₃ R₂
        inst✝²⁹ : RingHomInvPair σ₂₃ σ₃₂
        inst✝²⁸ : RingHomInvPair σ₃₂ σ₂₃
        σ₁₃ : RingHom R₁ R₃
        σ₃₁ : RingHom R₃ R₁
        inst✝²⁷ : RingHomInvPair σ₁₃ σ₃₁
        inst✝²⁶ : RingHomInvPair σ₃₁ σ₁₃
        inst✝²⁵ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
        inst✝²⁴ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
        M₁ : Type u_6
        inst✝²³ : TopologicalSpace M₁
        inst✝²² : AddCommMonoid M₁
        M₂ : Type u_7
        inst✝²¹ : TopologicalSpace M₂
        inst✝²⁰ : AddCommMonoid M₂
        M₃ : Type u_8
        inst✝¹⁹ : TopologicalSpace M₃
        inst✝¹⁸ : AddCommMonoid M₃
        M₄ : Type u_9
        inst✝¹⁷ : TopologicalSpace M₄
        inst✝¹⁶ : AddCommMonoid M₄
        inst✝¹⁵ : Module R₁ M₁
        inst✝¹⁴ : Module R₂ M₂
        inst✝¹³ : Module R₃ M₃
        R₄ : Type u_10
        inst✝¹² : Semiring R₄
        inst✝¹¹ : Module R₄ M₄
        σ₃₄ : RingHom R₃ R₄
        σ₄₃ : RingHom R₄ R₃
        inst✝¹⁰ : RingHomInvPair σ₃₄ σ₄₃
        inst✝⁹ : RingHomInvPair σ₄₃ σ₃₄
        σ₂₄ : RingHom R₂ R₄
        σ₁₄ : RingHom R₁ R₄
        inst✝⁸ : RingHomCompTriple σ₂₁ σ₁₄ σ₂₄
        inst✝⁷ : RingHomCompTriple σ₂₄ σ₄₃ σ₂₃
        inst✝⁶ : RingHomCompTriple σ₁₃ σ₃₄ σ₁₄
        ι : Type u_11
        M : ι → Type u_12
        inst✝⁵ : (i : ι) → TopologicalSpace (M i)
        inst✝⁴ : (i : ι) → AddCommMonoid (M i)
        inst✝³ : (i : ι) → Module R₁ (M i)
        N : ι → Type u_13
        inst✝² : (i : ι) → TopologicalSpace (N i)
        inst✝¹ : (i : ι) → AddCommMonoid (N i)
        inst✝ : (i : ι) → Module R₁ (N i)
        f : (i : ι) → ContinuousLinearEquiv (RingHom.id R₁) (M i) (N i)
        ⊢ Continuous __src✝.invFun
      -/
      exact continuous_pi fun i => (f i).continuous_invFun.comp (continuous_apply i) }
      /-
        🎉 no goals
      -/


@[simp]
theorem piCongrRight_apply (m : (i : ι) → M i) (i : ι) :
    piCongrRight f m i = (f i) (m i) := rfl


@[simp]
theorem piCongrRight_symm_apply (n : (i : ι) → N i) (i : ι) :
    (piCongrRight f).symm n i = (f i).symm (n i) := rfl


/-- Equivalence given by a block lower diagonal matrix. `e` and `e'` are diagonal square blocks,
  and `f` is a rectangular block below the diagonal. -/
def skewProd (e : M ≃L[R] M₂) (e' : M₃ ≃L[R] M₄) (f : M →L[R] M₄) : (M × M₃) ≃L[R] M₂ × M₄ :=
  {
    e.toLinearEquiv.skewProd e'.toLinearEquiv
      ↑f with
    continuous_toFun :=
      (e.continuous_toFun.comp continuous_fst).prod_mk
        ((e'.continuous_toFun.comp continuous_snd).add <| f.continuous.comp continuous_fst)
    continuous_invFun :=
      (e.continuous_invFun.comp continuous_fst).prod_mk
        (e'.continuous_invFun.comp <|
          continuous_snd.sub <| f.continuous.comp <| e.continuous_invFun.comp continuous_fst) }


@[simp]
theorem skewProd_apply (e : M ≃L[R] M₂) (e' : M₃ ≃L[R] M₄) (f : M →L[R] M₄) (x) :
    e.skewProd e' f x = (e x.1, e' x.2 + f x.1) :=
  rfl


@[simp]
theorem skewProd_symm_apply (e : M ≃L[R] M₂) (e' : M₃ ≃L[R] M₄) (f : M →L[R] M₄) (x) :
    (e.skewProd e' f).symm x = (e.symm x.1, e'.symm (x.2 - f (e.symm x.1))) :=
  rfl


variable (R) in
/-- The negation map as a continuous linear equivalence. -/
def neg [ContinuousNeg M] :
    M ≃L[R] M :=
  { LinearEquiv.neg R with
    continuous_toFun := continuous_neg
    continuous_invFun := continuous_neg }


@[simp]
theorem coe_neg [ContinuousNeg M] :
    (neg R : M → M) = -id := rfl


@[simp]
theorem neg_apply [ContinuousNeg M] (x : M) :
                       /-
                         R : Type u_3
                         inst✝⁴ : Semiring R
                         M : Type u_4
                         inst✝³ : TopologicalSpace M
                         inst✝² : AddCommGroup M
                         inst✝¹ : Module R M
                         inst✝ : ContinuousNeg M
                         x : M
                         ⊢ Eq ((ContinuousLinearEquiv.neg R) x) (Neg.neg x)
                       -/
    neg R x = -x := by simp
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem symm_neg [ContinuousNeg M] :
    (neg R : M ≃L[R] M).symm = neg R := rfl


theorem map_sub (e : M ≃SL[σ₁₂] M₂) (x y : M) : e (x - y) = e x - e y :=
  (e : M →SL[σ₁₂] M₂).map_sub x y


theorem map_neg (e : M ≃SL[σ₁₂] M₂) (x : M) : e (-x) = -e x :=
  (e : M →SL[σ₁₂] M₂).map_neg x


/-- An invertible continuous linear map `f` determines a continuous equivalence from `M` to itself.
-/
def ofUnit (f : (M →L[R] M)ˣ) : M ≃L[R] M where
  toLinearEquiv :=
    { toFun := f.val
                     /-
                       R✝ : Type u_1
                       M✝ : Type u_2
                       inst✝¹⁶ : Ring R✝
                       inst✝¹⁵ : TopologicalSpace R✝
                       inst✝¹⁴ : TopologicalSpace M✝
                       inst✝¹³ : AddCommGroup M✝
                       inst✝¹² : ContinuousAdd M✝
                       inst✝¹¹ : Module R✝ M✝
                       inst✝¹⁰ : ContinuousSMul R✝ M✝
                       R : Type u_3
                       inst✝⁹ : Ring R
                       R₂ : Type u_4
                       inst✝⁸ : Ring R₂
                       M : Type u_5
                       inst✝⁷ : TopologicalSpace M
                       inst✝⁶ : AddCommGroup M
                       inst✝⁵ : Module R M
                       M₂ : Type u_6
                       inst✝⁴ : TopologicalSpace M₂
                       inst✝³ : AddCommGroup M₂
                       inst✝² : Module R₂ M₂
                       σ₁₂ : RingHom R R₂
                       σ₂₁ : RingHom R₂ R
                       inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
                       inst✝ : RingHomInvPair σ₂₁ σ₁₂
                       f : Units (ContinuousLinearMap (RingHom.id R) M M)
                       ⊢ ∀ (x y : M), Eq (↑f (HAdd.hAdd x y)) (HAdd.hAdd (↑f x) (↑f y))
                     -/
      map_add' := by simp
                     /-
                       🎉 no goals
                     -/
                      /-
                        R✝ : Type u_1
                        M✝ : Type u_2
                        inst✝¹⁶ : Ring R✝
                        inst✝¹⁵ : TopologicalSpace R✝
                        inst✝¹⁴ : TopologicalSpace M✝
                        inst✝¹³ : AddCommGroup M✝
                        inst✝¹² : ContinuousAdd M✝
                        inst✝¹¹ : Module R✝ M✝
                        inst✝¹⁰ : ContinuousSMul R✝ M✝
                        R : Type u_3
                        inst✝⁹ : Ring R
                        R₂ : Type u_4
                        inst✝⁸ : Ring R₂
                        M : Type u_5
                        inst✝⁷ : TopologicalSpace M
                        inst✝⁶ : AddCommGroup M
                        inst✝⁵ : Module R M
                        M₂ : Type u_6
                        inst✝⁴ : TopologicalSpace M₂
                        inst✝³ : AddCommGroup M₂
                        inst✝² : Module R₂ M₂
                        σ₁₂ : RingHom R R₂
                        σ₂₁ : RingHom R₂ R
                        inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
                        inst✝ : RingHomInvPair σ₂₁ σ₁₂
                        f : Units (ContinuousLinearMap (RingHom.id R) M M)
                        ⊢ ∀ (m : R) (x : M), Eq ({ toFun := ⇑↑f, map_add' := ⋯ }.toFun (HSMul.hSMul m  …
                      -/
      map_smul' := by simp
                      /-
                        🎉 no goals
                      -/
      invFun := f.inv
      left_inv := fun x =>
        show (f.inv * f.val) x = x by
          /-
            R✝ : Type u_1
            M✝ : Type u_2
            inst✝¹⁶ : Ring R✝
            inst✝¹⁵ : TopologicalSpace R✝
            inst✝¹⁴ : TopologicalSpace M✝
            inst✝¹³ : AddCommGroup M✝
            inst✝¹² : ContinuousAdd M✝
            inst✝¹¹ : Module R✝ M✝
            inst✝¹⁰ : ContinuousSMul R✝ M✝
            R : Type u_3
            inst✝⁹ : Ring R
            R₂ : Type u_4
            inst✝⁸ : Ring R₂
            M : Type u_5
            inst✝⁷ : TopologicalSpace M
            inst✝⁶ : AddCommGroup M
            inst✝⁵ : Module R M
            M₂ : Type u_6
            inst✝⁴ : TopologicalSpace M₂
            inst✝³ : AddCommGroup M₂
            inst✝² : Module R₂ M₂
            σ₁₂ : RingHom R R₂
            σ₂₁ : RingHom R₂ R
            inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
            inst✝ : RingHomInvPair σ₂₁ σ₁₂
            f : Units (ContinuousLinearMap (RingHom.id R) M M)
            x : M
            ⊢ Eq ((HMul.hMul f.inv ↑f) x) x
          -/
          rw [f.inv_val]
          /-
            R✝ : Type u_1
            M✝ : Type u_2
            inst✝¹⁶ : Ring R✝
            inst✝¹⁵ : TopologicalSpace R✝
            inst✝¹⁴ : TopologicalSpace M✝
            inst✝¹³ : AddCommGroup M✝
            inst✝¹² : ContinuousAdd M✝
            inst✝¹¹ : Module R✝ M✝
            inst✝¹⁰ : ContinuousSMul R✝ M✝
            R : Type u_3
            inst✝⁹ : Ring R
            R₂ : Type u_4
            inst✝⁸ : Ring R₂
            M : Type u_5
            inst✝⁷ : TopologicalSpace M
            inst✝⁶ : AddCommGroup M
            inst✝⁵ : Module R M
            M₂ : Type u_6
            inst✝⁴ : TopologicalSpace M₂
            inst✝³ : AddCommGroup M₂
            inst✝² : Module R₂ M₂
            σ₁₂ : RingHom R R₂
            σ₂₁ : RingHom R₂ R
            inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
            inst✝ : RingHomInvPair σ₂₁ σ₁₂
            f : Units (ContinuousLinearMap (RingHom.id R) M M)
            x : M
            ⊢ Eq (1 x) x
          -/
          simp
          /-
            🎉 no goals
          -/
      right_inv := fun x =>
        show (f.val * f.inv) x = x by
          /-
            R✝ : Type u_1
            M✝ : Type u_2
            inst✝¹⁶ : Ring R✝
            inst✝¹⁵ : TopologicalSpace R✝
            inst✝¹⁴ : TopologicalSpace M✝
            inst✝¹³ : AddCommGroup M✝
            inst✝¹² : ContinuousAdd M✝
            inst✝¹¹ : Module R✝ M✝
            inst✝¹⁰ : ContinuousSMul R✝ M✝
            R : Type u_3
            inst✝⁹ : Ring R
            R₂ : Type u_4
            inst✝⁸ : Ring R₂
            M : Type u_5
            inst✝⁷ : TopologicalSpace M
            inst✝⁶ : AddCommGroup M
            inst✝⁵ : Module R M
            M₂ : Type u_6
            inst✝⁴ : TopologicalSpace M₂
            inst✝³ : AddCommGroup M₂
            inst✝² : Module R₂ M₂
            σ₁₂ : RingHom R R₂
            σ₂₁ : RingHom R₂ R
            inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
            inst✝ : RingHomInvPair σ₂₁ σ₁₂
            f : Units (ContinuousLinearMap (RingHom.id R) M M)
            x : M
            ⊢ Eq ((HMul.hMul (↑f) f.inv) x) x
          -/
          rw [f.val_inv]
          /-
            R✝ : Type u_1
            M✝ : Type u_2
            inst✝¹⁶ : Ring R✝
            inst✝¹⁵ : TopologicalSpace R✝
            inst✝¹⁴ : TopologicalSpace M✝
            inst✝¹³ : AddCommGroup M✝
            inst✝¹² : ContinuousAdd M✝
            inst✝¹¹ : Module R✝ M✝
            inst✝¹⁰ : ContinuousSMul R✝ M✝
            R : Type u_3
            inst✝⁹ : Ring R
            R₂ : Type u_4
            inst✝⁸ : Ring R₂
            M : Type u_5
            inst✝⁷ : TopologicalSpace M
            inst✝⁶ : AddCommGroup M
            inst✝⁵ : Module R M
            M₂ : Type u_6
            inst✝⁴ : TopologicalSpace M₂
            inst✝³ : AddCommGroup M₂
            inst✝² : Module R₂ M₂
            σ₁₂ : RingHom R R₂
            σ₂₁ : RingHom R₂ R
            inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
            inst✝ : RingHomInvPair σ₂₁ σ₁₂
            f : Units (ContinuousLinearMap (RingHom.id R) M M)
            x : M
            ⊢ Eq (1 x) x
          -/
          simp }
          /-
            🎉 no goals
          -/
  continuous_toFun := f.val.continuous
  continuous_invFun := f.inv.continuous


/-- A continuous equivalence from `M` to itself determines an invertible continuous linear map. -/
def toUnit (f : M ≃L[R] M) : (M →L[R] M)ˣ where
  val := f
  inv := f.symm
  val_inv := by
    /-
      R✝ : Type u_1
      M✝ : Type u_2
      inst✝¹⁶ : Ring R✝
      inst✝¹⁵ : TopologicalSpace R✝
      inst✝¹⁴ : TopologicalSpace M✝
      inst✝¹³ : AddCommGroup M✝
      inst✝¹² : ContinuousAdd M✝
      inst✝¹¹ : Module R✝ M✝
      inst✝¹⁰ : ContinuousSMul R✝ M✝
      R : Type u_3
      inst✝⁹ : Ring R
      R₂ : Type u_4
      inst✝⁸ : Ring R₂
      M : Type u_5
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      M₂ : Type u_6
      inst✝⁴ : TopologicalSpace M₂
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R₂ M₂
      σ₁₂ : RingHom R R₂
      σ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
      inst✝ : RingHomInvPair σ₂₁ σ₁₂
      f : ContinuousLinearEquiv (RingHom.id R) M M
      ⊢ Eq (HMul.hMul ↑f ↑f.symm) 1
    -/
    ext
    /-
      case h
      R✝ : Type u_1
      M✝ : Type u_2
      inst✝¹⁶ : Ring R✝
      inst✝¹⁵ : TopologicalSpace R✝
      inst✝¹⁴ : TopologicalSpace M✝
      inst✝¹³ : AddCommGroup M✝
      inst✝¹² : ContinuousAdd M✝
      inst✝¹¹ : Module R✝ M✝
      inst✝¹⁰ : ContinuousSMul R✝ M✝
      R : Type u_3
      inst✝⁹ : Ring R
      R₂ : Type u_4
      inst✝⁸ : Ring R₂
      M : Type u_5
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      M₂ : Type u_6
      inst✝⁴ : TopologicalSpace M₂
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R₂ M₂
      σ₁₂ : RingHom R R₂
      σ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
      inst✝ : RingHomInvPair σ₂₁ σ₁₂
      f : ContinuousLinearEquiv (RingHom.id R) M M
      x✝ : M
      ⊢ Eq ((HMul.hMul ↑f ↑f.symm) x✝) (1 x✝)
    -/
    simp
    /-
      🎉 no goals
    -/
  inv_val := by
    /-
      R✝ : Type u_1
      M✝ : Type u_2
      inst✝¹⁶ : Ring R✝
      inst✝¹⁵ : TopologicalSpace R✝
      inst✝¹⁴ : TopologicalSpace M✝
      inst✝¹³ : AddCommGroup M✝
      inst✝¹² : ContinuousAdd M✝
      inst✝¹¹ : Module R✝ M✝
      inst✝¹⁰ : ContinuousSMul R✝ M✝
      R : Type u_3
      inst✝⁹ : Ring R
      R₂ : Type u_4
      inst✝⁸ : Ring R₂
      M : Type u_5
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      M₂ : Type u_6
      inst✝⁴ : TopologicalSpace M₂
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R₂ M₂
      σ₁₂ : RingHom R R₂
      σ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
      inst✝ : RingHomInvPair σ₂₁ σ₁₂
      f : ContinuousLinearEquiv (RingHom.id R) M M
      ⊢ Eq (HMul.hMul ↑f.symm ↑f) 1
    -/
    ext
    /-
      case h
      R✝ : Type u_1
      M✝ : Type u_2
      inst✝¹⁶ : Ring R✝
      inst✝¹⁵ : TopologicalSpace R✝
      inst✝¹⁴ : TopologicalSpace M✝
      inst✝¹³ : AddCommGroup M✝
      inst✝¹² : ContinuousAdd M✝
      inst✝¹¹ : Module R✝ M✝
      inst✝¹⁰ : ContinuousSMul R✝ M✝
      R : Type u_3
      inst✝⁹ : Ring R
      R₂ : Type u_4
      inst✝⁸ : Ring R₂
      M : Type u_5
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      M₂ : Type u_6
      inst✝⁴ : TopologicalSpace M₂
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R₂ M₂
      σ₁₂ : RingHom R R₂
      σ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
      inst✝ : RingHomInvPair σ₂₁ σ₁₂
      f : ContinuousLinearEquiv (RingHom.id R) M M
      x✝ : M
      ⊢ Eq ((HMul.hMul ↑f.symm ↑f) x✝) (1 x✝)
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The units of the algebra of continuous `R`-linear endomorphisms of `M` is multiplicatively
equivalent to the type of continuous linear equivalences between `M` and itself. -/
def unitsEquiv : (M →L[R] M)ˣ ≃* M ≃L[R] M where
  toFun := ofUnit
  invFun := toUnit
  left_inv f := by
    /-
      R✝ : Type u_1
      M✝ : Type u_2
      inst✝¹⁶ : Ring R✝
      inst✝¹⁵ : TopologicalSpace R✝
      inst✝¹⁴ : TopologicalSpace M✝
      inst✝¹³ : AddCommGroup M✝
      inst✝¹² : ContinuousAdd M✝
      inst✝¹¹ : Module R✝ M✝
      inst✝¹⁰ : ContinuousSMul R✝ M✝
      R : Type u_3
      inst✝⁹ : Ring R
      R₂ : Type u_4
      inst✝⁸ : Ring R₂
      M : Type u_5
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      M₂ : Type u_6
      inst✝⁴ : TopologicalSpace M₂
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R₂ M₂
      σ₁₂ : RingHom R R₂
      σ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
      inst✝ : RingHomInvPair σ₂₁ σ₁₂
      f : Units (ContinuousLinearMap (RingHom.id R) M M)
      ⊢ Eq (ContinuousLinearEquiv.ofUnit f).toUnit f
    -/
    ext
    /-
      case a.h
      R✝ : Type u_1
      M✝ : Type u_2
      inst✝¹⁶ : Ring R✝
      inst✝¹⁵ : TopologicalSpace R✝
      inst✝¹⁴ : TopologicalSpace M✝
      inst✝¹³ : AddCommGroup M✝
      inst✝¹² : ContinuousAdd M✝
      inst✝¹¹ : Module R✝ M✝
      inst✝¹⁰ : ContinuousSMul R✝ M✝
      R : Type u_3
      inst✝⁹ : Ring R
      R₂ : Type u_4
      inst✝⁸ : Ring R₂
      M : Type u_5
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      M₂ : Type u_6
      inst✝⁴ : TopologicalSpace M₂
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R₂ M₂
      σ₁₂ : RingHom R R₂
      σ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
      inst✝ : RingHomInvPair σ₂₁ σ₁₂
      f : Units (ContinuousLinearMap (RingHom.id R) M M)
      x✝ : M
      ⊢ Eq (↑(ContinuousLinearEquiv.ofUnit f).toUnit x✝) (↑f x✝)
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      R✝ : Type u_1
      M✝ : Type u_2
      inst✝¹⁶ : Ring R✝
      inst✝¹⁵ : TopologicalSpace R✝
      inst✝¹⁴ : TopologicalSpace M✝
      inst✝¹³ : AddCommGroup M✝
      inst✝¹² : ContinuousAdd M✝
      inst✝¹¹ : Module R✝ M✝
      inst✝¹⁰ : ContinuousSMul R✝ M✝
      R : Type u_3
      inst✝⁹ : Ring R
      R₂ : Type u_4
      inst✝⁸ : Ring R₂
      M : Type u_5
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      M₂ : Type u_6
      inst✝⁴ : TopologicalSpace M₂
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R₂ M₂
      σ₁₂ : RingHom R R₂
      σ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
      inst✝ : RingHomInvPair σ₂₁ σ₁₂
      f : ContinuousLinearEquiv (RingHom.id R) M M
      ⊢ Eq (ContinuousLinearEquiv.ofUnit f.toUnit) f
    -/
    ext
    /-
      case h.h
      R✝ : Type u_1
      M✝ : Type u_2
      inst✝¹⁶ : Ring R✝
      inst✝¹⁵ : TopologicalSpace R✝
      inst✝¹⁴ : TopologicalSpace M✝
      inst✝¹³ : AddCommGroup M✝
      inst✝¹² : ContinuousAdd M✝
      inst✝¹¹ : Module R✝ M✝
      inst✝¹⁰ : ContinuousSMul R✝ M✝
      R : Type u_3
      inst✝⁹ : Ring R
      R₂ : Type u_4
      inst✝⁸ : Ring R₂
      M : Type u_5
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      M₂ : Type u_6
      inst✝⁴ : TopologicalSpace M₂
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R₂ M₂
      σ₁₂ : RingHom R R₂
      σ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
      inst✝ : RingHomInvPair σ₂₁ σ₁₂
      f : ContinuousLinearEquiv (RingHom.id R) M M
      x✝ : M
      ⊢ Eq ((ContinuousLinearEquiv.ofUnit f.toUnit) x✝) (f x✝)
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_mul' x y := by
    /-
      R✝ : Type u_1
      M✝ : Type u_2
      inst✝¹⁶ : Ring R✝
      inst✝¹⁵ : TopologicalSpace R✝
      inst✝¹⁴ : TopologicalSpace M✝
      inst✝¹³ : AddCommGroup M✝
      inst✝¹² : ContinuousAdd M✝
      inst✝¹¹ : Module R✝ M✝
      inst✝¹⁰ : ContinuousSMul R✝ M✝
      R : Type u_3
      inst✝⁹ : Ring R
      R₂ : Type u_4
      inst✝⁸ : Ring R₂
      M : Type u_5
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      M₂ : Type u_6
      inst✝⁴ : TopologicalSpace M₂
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R₂ M₂
      σ₁₂ : RingHom R R₂
      σ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
      inst✝ : RingHomInvPair σ₂₁ σ₁₂
      x y : Units (ContinuousLinearMap (RingHom.id R) M M)
      ⊢ Eq ({ toFun := ContinuousLinearEquiv.ofUnit, invFun := ContinuousLinearEquiv …
    -/
    ext
    /-
      case h.h
      R✝ : Type u_1
      M✝ : Type u_2
      inst✝¹⁶ : Ring R✝
      inst✝¹⁵ : TopologicalSpace R✝
      inst✝¹⁴ : TopologicalSpace M✝
      inst✝¹³ : AddCommGroup M✝
      inst✝¹² : ContinuousAdd M✝
      inst✝¹¹ : Module R✝ M✝
      inst✝¹⁰ : ContinuousSMul R✝ M✝
      R : Type u_3
      inst✝⁹ : Ring R
      R₂ : Type u_4
      inst✝⁸ : Ring R₂
      M : Type u_5
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Module R M
      M₂ : Type u_6
      inst✝⁴ : TopologicalSpace M₂
      inst✝³ : AddCommGroup M₂
      inst✝² : Module R₂ M₂
      σ₁₂ : RingHom R R₂
      σ₂₁ : RingHom R₂ R
      inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
      inst✝ : RingHomInvPair σ₂₁ σ₁₂
      x y : Units (ContinuousLinearMap (RingHom.id R) M M)
      x✝ : M
      ⊢ Eq (({ toFun := ContinuousLinearEquiv.ofUnit, invFun := ContinuousLinearEqui …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem unitsEquiv_apply (f : (M →L[R] M)ˣ) (x : M) : unitsEquiv R M f x = (f : M →L[R] M) x :=
  rfl


/-- Continuous linear equivalences `R ≃L[R] R` are enumerated by `Rˣ`. -/
def unitsEquivAut : Rˣ ≃ R ≃L[R] R where
  toFun u :=
    equivOfInverse (ContinuousLinearMap.smulRight (1 : R →L[R] R) ↑u)
                                                                        /-
                                                                          R✝ : Type u_1
                                                                          M✝ : Type u_2
                                                                          inst✝¹⁸ : Ring R✝
                                                                          inst✝¹⁷ : TopologicalSpace R✝
                                                                          inst✝¹⁶ : TopologicalSpace M✝
                                                                          inst✝¹⁵ : AddCommGroup M✝
                                                                          inst✝¹⁴ : ContinuousAdd M✝
                                                                          inst✝¹³ : Module R✝ M✝
                                                                          inst✝¹² : ContinuousSMul R✝ M✝
                                                                          R : Type u_3
                                                                          inst✝¹¹ : Ring R
                                                                          R₂ : Type u_4
                                                                          inst✝¹⁰ : Ring R₂
                                                                          M : Type u_5
                                                                          inst✝⁹ : TopologicalSpace M
                                                                          inst✝⁸ : AddCommGroup M
                                                                          inst✝⁷ : Module R M
                                                                          M₂ : Type u_6
                                                                          inst✝⁶ : TopologicalSpace M₂
                                                                          inst✝⁵ : AddCommGroup M₂
                                                                          inst✝⁴ : Module R₂ M₂
                                                                          σ₁₂ : RingHom R R₂
                                                                          σ₂₁ : RingHom R₂ R
                                                                          inst✝³ : RingHomInvPair σ₁₂ σ₂₁
                                                                          inst✝² : RingHomInvPair σ₂₁ σ₁₂
                                                                          inst✝¹ : TopologicalSpace R
                                                                          inst✝ : ContinuousMul R
                                                                          u : Units R
                                                                          x : R
                                                                          ⊢ Eq ((ContinuousLinearMap.smulRight 1 ↑(Inv.inv u)) ((ContinuousLinearMap.smu …
                                                                        -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
      (ContinuousLinearMap.smulRight (1 : R →L[R] R) ↑u⁻¹) (fun x => by simp) fun x => by simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
  invFun e :=
                       /-
                         R✝ : Type u_1
                         M✝ : Type u_2
                         inst✝¹⁸ : Ring R✝
                         inst✝¹⁷ : TopologicalSpace R✝
                         inst✝¹⁶ : TopologicalSpace M✝
                         inst✝¹⁵ : AddCommGroup M✝
                         inst✝¹⁴ : ContinuousAdd M✝
                         inst✝¹³ : Module R✝ M✝
                         inst✝¹² : ContinuousSMul R✝ M✝
                         R : Type u_3
                         inst✝¹¹ : Ring R
                         R₂ : Type u_4
                         inst✝¹⁰ : Ring R₂
                         M : Type u_5
                         inst✝⁹ : TopologicalSpace M
                         inst✝⁸ : AddCommGroup M
                         inst✝⁷ : Module R M
                         M₂ : Type u_6
                         inst✝⁶ : TopologicalSpace M₂
                         inst✝⁵ : AddCommGroup M₂
                         inst✝⁴ : Module R₂ M₂
                         σ₁₂ : RingHom R R₂
                         σ₂₁ : RingHom R₂ R
                         inst✝³ : RingHomInvPair σ₁₂ σ₂₁
                         inst✝² : RingHomInvPair σ₂₁ σ₁₂
                         inst✝¹ : TopologicalSpace R
                         inst✝ : ContinuousMul R
                         e : ContinuousLinearEquiv (RingHom.id R) R R
                         ⊢ Eq (HMul.hMul (e 1) (e.symm 1)) 1
                       -/
    ⟨e 1, e.symm 1, by rw [← smul_eq_mul, ← map_smul, smul_eq_mul, mul_one, symm_apply_apply], by
                       /-
                         🎉 no goals
                       -/
      /-
        R✝ : Type u_1
        M✝ : Type u_2
        inst✝¹⁸ : Ring R✝
        inst✝¹⁷ : TopologicalSpace R✝
        inst✝¹⁶ : TopologicalSpace M✝
        inst✝¹⁵ : AddCommGroup M✝
        inst✝¹⁴ : ContinuousAdd M✝
        inst✝¹³ : Module R✝ M✝
        inst✝¹² : ContinuousSMul R✝ M✝
        R : Type u_3
        inst✝¹¹ : Ring R
        R₂ : Type u_4
        inst✝¹⁰ : Ring R₂
        M : Type u_5
        inst✝⁹ : TopologicalSpace M
        inst✝⁸ : AddCommGroup M
        inst✝⁷ : Module R M
        M₂ : Type u_6
        inst✝⁶ : TopologicalSpace M₂
        inst✝⁵ : AddCommGroup M₂
        inst✝⁴ : Module R₂ M₂
        σ₁₂ : RingHom R R₂
        σ₂₁ : RingHom R₂ R
        inst✝³ : RingHomInvPair σ₁₂ σ₂₁
        inst✝² : RingHomInvPair σ₂₁ σ₁₂
        inst✝¹ : TopologicalSpace R
        inst✝ : ContinuousMul R
        e : ContinuousLinearEquiv (RingHom.id R) R R
        ⊢ Eq (HMul.hMul (e.symm 1) (e 1)) 1
      -/
      rw [← smul_eq_mul, ← map_smul, smul_eq_mul, mul_one, apply_symm_apply]⟩
      /-
        🎉 no goals
      -/
                                /-
                                  R✝ : Type u_1
                                  M✝ : Type u_2
                                  inst✝¹⁸ : Ring R✝
                                  inst✝¹⁷ : TopologicalSpace R✝
                                  inst✝¹⁶ : TopologicalSpace M✝
                                  inst✝¹⁵ : AddCommGroup M✝
                                  inst✝¹⁴ : ContinuousAdd M✝
                                  inst✝¹³ : Module R✝ M✝
                                  inst✝¹² : ContinuousSMul R✝ M✝
                                  R : Type u_3
                                  inst✝¹¹ : Ring R
                                  R₂ : Type u_4
                                  inst✝¹⁰ : Ring R₂
                                  M : Type u_5
                                  inst✝⁹ : TopologicalSpace M
                                  inst✝⁸ : AddCommGroup M
                                  inst✝⁷ : Module R M
                                  M₂ : Type u_6
                                  inst✝⁶ : TopologicalSpace M₂
                                  inst✝⁵ : AddCommGroup M₂
                                  inst✝⁴ : Module R₂ M₂
                                  σ₁₂ : RingHom R R₂
                                  σ₂₁ : RingHom R₂ R
                                  inst✝³ : RingHomInvPair σ₁₂ σ₂₁
                                  inst✝² : RingHomInvPair σ₂₁ σ₁₂
                                  inst✝¹ : TopologicalSpace R
                                  inst✝ : ContinuousMul R
                                  u : Units R
                                  ⊢ Eq ↑((fun e => { val := e 1, inv := e.symm 1, val_inv := ⋯, inv_val := ⋯ })  …
                                -/
  left_inv u := Units.ext <| by simp
                                /-
                                  🎉 no goals
                                -/
                            /-
                              R✝ : Type u_1
                              M✝ : Type u_2
                              inst✝¹⁸ : Ring R✝
                              inst✝¹⁷ : TopologicalSpace R✝
                              inst✝¹⁶ : TopologicalSpace M✝
                              inst✝¹⁵ : AddCommGroup M✝
                              inst✝¹⁴ : ContinuousAdd M✝
                              inst✝¹³ : Module R✝ M✝
                              inst✝¹² : ContinuousSMul R✝ M✝
                              R : Type u_3
                              inst✝¹¹ : Ring R
                              R₂ : Type u_4
                              inst✝¹⁰ : Ring R₂
                              M : Type u_5
                              inst✝⁹ : TopologicalSpace M
                              inst✝⁸ : AddCommGroup M
                              inst✝⁷ : Module R M
                              M₂ : Type u_6
                              inst✝⁶ : TopologicalSpace M₂
                              inst✝⁵ : AddCommGroup M₂
                              inst✝⁴ : Module R₂ M₂
                              σ₁₂ : RingHom R R₂
                              σ₂₁ : RingHom R₂ R
                              inst✝³ : RingHomInvPair σ₁₂ σ₂₁
                              inst✝² : RingHomInvPair σ₂₁ σ₁₂
                              inst✝¹ : TopologicalSpace R
                              inst✝ : ContinuousMul R
                              e : ContinuousLinearEquiv (RingHom.id R) R R
                              ⊢ Eq (((fun u => ContinuousLinearEquiv.equivOfInverse (ContinuousLinearMap.smu …
                            -/
  right_inv e := ext₁ <| by simp
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem unitsEquivAut_apply (u : Rˣ) (x : R) : unitsEquivAut R u x = x * u :=
  rfl


@[simp]
theorem unitsEquivAut_apply_symm (u : Rˣ) (x : R) : (unitsEquivAut R u).symm x = x * ↑u⁻¹ :=
  rfl


@[simp]
theorem unitsEquivAut_symm_apply (e : R ≃L[R] R) : ↑((unitsEquivAut R).symm e) = e 1 :=
  rfl


/-- A pair of continuous linear maps such that `f₁ ∘ f₂ = id` generates a continuous
linear equivalence `e` between `M` and `M₂ × f₁.ker` such that `(e x).2 = x` for `x ∈ f₁.ker`,
`(e x).1 = f₁ x`, and `(e (f₂ y)).2 = 0`. The map is given by `e x = (f₁ x, x - f₂ (f₁ x))`. -/
def equivOfRightInverse (f₁ : M →L[R] M₂) (f₂ : M₂ →L[R] M) (h : Function.RightInverse f₂ f₁) :
    M ≃L[R] M₂ × ker f₁ :=
  equivOfInverse (f₁.prod (f₁.projKerOfRightInverse f₂ h)) (f₂.coprod (ker f₁).subtypeL)
                 /-
                   R✝ : Type u_1
                   M✝ : Type u_2
                   inst✝¹⁸ : Ring R✝
                   inst✝¹⁷ : TopologicalSpace R✝
                   inst✝¹⁶ : TopologicalSpace M✝
                   inst✝¹⁵ : AddCommGroup M✝
                   inst✝¹⁴ : ContinuousAdd M✝
                   inst✝¹³ : Module R✝ M✝
                   inst✝¹² : ContinuousSMul R✝ M✝
                   R : Type u_3
                   inst✝¹¹ : Ring R
                   R₂ : Type u_4
                   inst✝¹⁰ : Ring R₂
                   M : Type u_5
                   inst✝⁹ : TopologicalSpace M
                   inst✝⁸ : AddCommGroup M
                   inst✝⁷ : Module R M
                   M₂ : Type u_6
                   inst✝⁶ : TopologicalSpace M₂
                   inst✝⁵ : AddCommGroup M₂
                   inst✝⁴ : Module R₂ M₂
                   σ₁₂ : RingHom R R₂
                   σ₂₁ : RingHom R₂ R
                   inst✝³ : RingHomInvPair σ₁₂ σ₂₁
                   inst✝² : RingHomInvPair σ₂₁ σ₁₂
                   inst✝¹ : Module R M₂
                   inst✝ : TopologicalAddGroup M
                   f₁ : ContinuousLinearMap (RingHom.id R) M M₂
                   f₂ : ContinuousLinearMap (RingHom.id R) M₂ M
                   h : Function.RightInverse ⇑f₂ ⇑f₁
                   x : M
                   ⊢ Eq ((f₂.coprod (LinearMap.ker f₁).subtypeL) ((f₁.prod (f₁.projKerOfRightInve …
                 -/
    (fun x => by simp) fun ⟨x, y⟩ => by
                 /-
                   🎉 no goals
                 -/
      -- Porting note: `simp` timeouts.
      rw [ContinuousLinearMap.coprod_apply,
        Submodule.subtypeL_apply, _root_.map_add, ContinuousLinearMap.prod_apply, h x,
        ContinuousLinearMap.projKerOfRightInverse_comp_inv,
        ContinuousLinearMap.prod_apply, LinearMap.map_coe_ker,
        ContinuousLinearMap.projKerOfRightInverse_apply_idem, Prod.mk_add_mk, add_zero, zero_add]


@[simp]
theorem fst_equivOfRightInverse (f₁ : M →L[R] M₂) (f₂ : M₂ →L[R] M)
    (h : Function.RightInverse f₂ f₁) (x : M) : (equivOfRightInverse f₁ f₂ h x).1 = f₁ x :=
  rfl


@[simp]
theorem snd_equivOfRightInverse (f₁ : M →L[R] M₂) (f₂ : M₂ →L[R] M)
    (h : Function.RightInverse f₂ f₁) (x : M) :
    ((equivOfRightInverse f₁ f₂ h x).2 : M) = x - f₂ (f₁ x) :=
  rfl


@[simp]
theorem equivOfRightInverse_symm_apply (f₁ : M →L[R] M₂) (f₂ : M₂ →L[R] M)
    (h : Function.RightInverse f₂ f₁) (y : M₂ × ker f₁) :
    (equivOfRightInverse f₁ f₂ h).symm y = f₂ y.1 + y.2 :=
  rfl


/-- If `ι` has a unique element, then `ι → M` is continuously linear equivalent to `M`. -/
def funUnique : (ι → M) ≃L[R] M :=
  { Homeomorph.funUnique ι M with toLinearEquiv := LinearEquiv.funUnique ι R M }


@[simp]
theorem coe_funUnique : ⇑(funUnique ι R M) = Function.eval default :=
  rfl


@[simp]
theorem coe_funUnique_symm : ⇑(funUnique ι R M).symm = Function.const ι :=
  rfl


/-- Continuous linear equivalence between dependent functions `(i : Fin 2) → M i` and `M 0 × M 1`.
-/
@[simps! (config := .asFn) apply symm_apply]
def piFinTwo (M : Fin 2 → Type*) [∀ i, AddCommMonoid (M i)] [∀ i, Module R (M i)]
    [∀ i, TopologicalSpace (M i)] : ((i : _) → M i) ≃L[R] M 0 × M 1 :=
  { Homeomorph.piFinTwo M with toLinearEquiv := LinearEquiv.piFinTwo R M }


/-- Continuous linear equivalence between vectors in `M² = Fin 2 → M` and `M × M`. -/
@[simps! (config := .asFn) apply symm_apply]
def finTwoArrow : (Fin 2 → M) ≃L[R] M × M :=
  { piFinTwo R fun _ => M with toLinearEquiv := LinearEquiv.finTwoArrow R M }


/-- A continuous linear map is invertible if it is the forward direction of a continuous linear
equivalence. -/
def IsInvertible (f : M →L[R] M₂) : Prop :=
  ∃ (A : M ≃L[R] M₂), A = f


open Classical in
/-- Introduce a function `inverse` from `M →L[R] M₂` to `M₂ →L[R] M`, which sends `f` to `f.symm` if
`f` is a continuous linear equivalence and to `0` otherwise.  This definition is somewhat ad hoc,
but one needs a fully (rather than partially) defined inverse function for some purposes, including
for calculus. -/
noncomputable def inverse : (M →L[R] M₂) → M₂ →L[R] M := fun f =>
  if h : f.IsInvertible then ((Classical.choose h).symm : M₂ →L[R] M) else 0


@[simp] lemma isInvertible_equiv {f : M ≃L[R] M₂} : IsInvertible (f : M →L[R] M₂) := ⟨f, rfl⟩


/-- By definition, if `f` is invertible then `inverse f = f.symm`. -/
@[simp]
theorem inverse_equiv (e : M ≃L[R] M₂) : inverse (e : M →L[R] M₂) = e.symm := by
  /-
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : TopologicalSpace M₂
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    e : ContinuousLinearEquiv (RingHom.id R) M M₂
    ⊢ Eq (↑e).inverse ↑e.symm
  -/
  simp [inverse]
  /-
    🎉 no goals
  -/


/-- By definition, if `f` is not invertible then `inverse f = 0`. -/
@[simp] lemma inverse_of_not_isInvertible
    {f : M →L[R] M₂} (hf : ¬ f.IsInvertible) : f.inverse = 0 :=
  dif_neg hf


@[deprecated (since := "2024-10-29")] alias inverse_non_equiv := inverse_of_not_isInvertible


@[simp] theorem inverse_zero : inverse (0 : M →L[R] M₂) = 0 := by
  /-
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : TopologicalSpace M₂
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    ⊢ Eq (ContinuousLinearMap.inverse 0) 0
  -/
  by_cases h : IsInvertible (0 : M →L[R] M₂)
    /-
      case pos
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M₂
      inst✝ : Module R M₂
      h : ContinuousLinearMap.IsInvertible 0
      ⊢ Eq (ContinuousLinearMap.inverse 0) 0
    -/
  · rcases h with ⟨e', he'⟩
    /-
      case pos.intro
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M₂
      inst✝ : Module R M₂
      e' : ContinuousLinearEquiv (RingHom.id R) M M₂
      he' : Eq (↑e') 0
      ⊢ Eq (ContinuousLinearMap.inverse 0) 0
    -/
    simp only [← he', inverse_equiv]
    /-
      case pos.intro
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M₂
      inst✝ : Module R M₂
      e' : ContinuousLinearEquiv (RingHom.id R) M M₂
      he' : Eq (↑e') 0
      ⊢ Eq (↑e'.symm) 0
    -/
    ext v
    /-
      case pos.intro.h
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M₂
      inst✝ : Module R M₂
      e' : ContinuousLinearEquiv (RingHom.id R) M M₂
      he' : Eq (↑e') 0
      v : M₂
      ⊢ Eq (↑e'.symm v) (0 v)
    -/
    apply e'.injective
    /-
      case pos.intro.h.a
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M₂
      inst✝ : Module R M₂
      e' : ContinuousLinearEquiv (RingHom.id R) M M₂
      he' : Eq (↑e') 0
      v : M₂
      ⊢ Eq (e' (↑e'.symm v)) (e' (0 v))
    -/
    rw [← ContinuousLinearEquiv.coe_coe, he']
    /-
      case pos.intro.h.a
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M₂
      inst✝ : Module R M₂
      e' : ContinuousLinearEquiv (RingHom.id R) M M₂
      he' : Eq (↑e') 0
      v : M₂
      ⊢ Eq (0 (↑e'.symm v)) (0 (0 v))
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M₂
      inst✝ : Module R M₂
      h : Not (ContinuousLinearMap.IsInvertible 0)
      ⊢ Eq (ContinuousLinearMap.inverse 0) 0
    -/
  · exact inverse_of_not_isInvertible h
    /-
      🎉 no goals
    -/


lemma IsInvertible.comp {g : M₂ →L[R] M₃} {f : M →L[R] M₂}
    (hg : g.IsInvertible) (hf : f.IsInvertible) : (g ∘L f).IsInvertible := by
  /-
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    g : ContinuousLinearMap (RingHom.id R) M₂ M₃
    f : ContinuousLinearMap (RingHom.id R) M M₂
    hg : g.IsInvertible
    hf : f.IsInvertible
    ⊢ (g.comp f).IsInvertible
  -/
  rcases hg with ⟨N, rfl⟩
  /-
    case intro
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    f : ContinuousLinearMap (RingHom.id R) M M₂
    hf : f.IsInvertible
    N : ContinuousLinearEquiv (RingHom.id R) M₂ M₃
    ⊢ ((↑N).comp f).IsInvertible
  -/
  rcases hf with ⟨M, rfl⟩
  /-
    case intro.intro
    R : Type u_3
    M✝ : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁹ : TopologicalSpace M✝
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M✝
    inst✝⁴ : Module R M✝
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    N : ContinuousLinearEquiv (RingHom.id R) M₂ M₃
    M : ContinuousLinearEquiv (RingHom.id R) M✝ M₂
    ⊢ ((↑N).comp ↑M).IsInvertible
  -/
  exact ⟨M.trans N, rfl⟩
  /-
    🎉 no goals
  -/


lemma IsInvertible.of_inverse {f : M →L[R] M₂} {g : M₂ →L[R] M}
    (hf : f ∘L g = id R M₂) (hg : g ∘L f = id R M) :
    f.IsInvertible :=
  ⟨ContinuousLinearEquiv.equivOfInverse' _ _ hf hg, rfl⟩


lemma inverse_eq {f : M →L[R] M₂} {g : M₂ →L[R] M} (hf : f ∘L g = id R M₂) (hg : g ∘L f = id R M) :
    f.inverse = g := by
  /-
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : TopologicalSpace M₂
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f : ContinuousLinearMap (RingHom.id R) M M₂
    g : ContinuousLinearMap (RingHom.id R) M₂ M
    hf : Eq (f.comp g) (ContinuousLinearMap.id R M₂)
    hg : Eq (g.comp f) (ContinuousLinearMap.id R M)
    ⊢ Eq f.inverse g
  -/
  have : f = ContinuousLinearEquiv.equivOfInverse' f g hf hg := rfl
  /-
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : TopologicalSpace M₂
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f : ContinuousLinearMap (RingHom.id R) M M₂
    g : ContinuousLinearMap (RingHom.id R) M₂ M
    hf : Eq (f.comp g) (ContinuousLinearMap.id R M₂)
    hg : Eq (g.comp f) (ContinuousLinearMap.id R M)
    this : Eq f ↑(ContinuousLinearEquiv.equivOfInverse' f g hf hg)
    ⊢ Eq f.inverse g
  -/
  rw [this, inverse_equiv]
  /-
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : TopologicalSpace M₂
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f : ContinuousLinearMap (RingHom.id R) M M₂
    g : ContinuousLinearMap (RingHom.id R) M₂ M
    hf : Eq (f.comp g) (ContinuousLinearMap.id R M₂)
    hg : Eq (g.comp f) (ContinuousLinearMap.id R M)
    this : Eq f ↑(ContinuousLinearEquiv.equivOfInverse' f g hf hg)
    ⊢ Eq (↑(ContinuousLinearEquiv.equivOfInverse' f g hf hg).symm) g
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma IsInvertible.inverse_apply_eq {f : M →L[R] M₂} {x : M} {y : M₂} (hf : f.IsInvertible) :
    f.inverse y = x ↔ y = f x := by
  /-
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : TopologicalSpace M₂
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    f : ContinuousLinearMap (RingHom.id R) M M₂
    x : M
    y : M₂
    hf : f.IsInvertible
    ⊢ Iff (Eq (f.inverse y) x) (Eq y (f x))
  -/
  rcases hf with ⟨M, rfl⟩
  /-
    case intro
    R : Type u_3
    M✝ : Type u_4
    M₂ : Type u_5
    inst✝⁶ : TopologicalSpace M✝
    inst✝⁵ : TopologicalSpace M₂
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M✝
    inst✝² : Module R M✝
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    x : M✝
    y : M₂
    M : ContinuousLinearEquiv (RingHom.id R) M✝ M₂
    ⊢ Iff (Eq ((↑M).inverse y) x) (Eq y (↑M x))
  -/
  simp only [inverse_equiv, ContinuousLinearEquiv.coe_coe]
  /-
    case intro
    R : Type u_3
    M✝ : Type u_4
    M₂ : Type u_5
    inst✝⁶ : TopologicalSpace M✝
    inst✝⁵ : TopologicalSpace M₂
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M✝
    inst✝² : Module R M✝
    inst✝¹ : AddCommMonoid M₂
    inst✝ : Module R M₂
    x : M✝
    y : M₂
    M : ContinuousLinearEquiv (RingHom.id R) M✝ M₂
    ⊢ Iff (Eq (M.symm y) x) (Eq y (M x))
  -/
  exact ContinuousLinearEquiv.symm_apply_eq M
  /-
    🎉 no goals
  -/


@[simp] lemma isInvertible_equiv_comp {e : M₂ ≃L[R] M₃} {f : M →L[R] M₂} :
    ((e : M₂ →L[R] M₃) ∘L f).IsInvertible ↔ f.IsInvertible := by
  /-
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    e : ContinuousLinearEquiv (RingHom.id R) M₂ M₃
    f : ContinuousLinearMap (RingHom.id R) M M₂
    ⊢ Iff ((↑e).comp f).IsInvertible f.IsInvertible
  -/
  constructor
    /-
      case mp
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₂ M₃
      f : ContinuousLinearMap (RingHom.id R) M M₂
      ⊢ ((↑e).comp f).IsInvertible → f.IsInvertible
    -/
  · rintro ⟨A, hA⟩
    /-
      case mp.intro
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₂ M₃
      f : ContinuousLinearMap (RingHom.id R) M M₂
      A : ContinuousLinearEquiv (RingHom.id R) M M₃
      hA : Eq (↑A) ((↑e).comp f)
      ⊢ f.IsInvertible
    -/
    have : f = e.symm ∘L ((e : M₂ →L[R] M₃) ∘L f) := by ext; simp
    /-
      case mp.intro
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₂ M₃
      f : ContinuousLinearMap (RingHom.id R) M M₂
      A : ContinuousLinearEquiv (RingHom.id R) M M₃
      hA : Eq (↑A) ((↑e).comp f)
      this : Eq f ((↑e.symm).comp ((↑e).comp f))
      ⊢ f.IsInvertible
    -/
    rw [this, ← hA]
    /-
      case mp.intro
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₂ M₃
      f : ContinuousLinearMap (RingHom.id R) M M₂
      A : ContinuousLinearEquiv (RingHom.id R) M M₃
      hA : Eq (↑A) ((↑e).comp f)
      this : Eq f ((↑e.symm).comp ((↑e).comp f))
      ⊢ ((↑e.symm).comp ↑A).IsInvertible
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₂ M₃
      f : ContinuousLinearMap (RingHom.id R) M M₂
      ⊢ f.IsInvertible → ((↑e).comp f).IsInvertible
    -/
  · rintro ⟨M, rfl⟩
    /-
      case mpr.intro
      R : Type u_3
      M✝ : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M✝
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M✝
      inst✝⁴ : Module R M✝
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₂ M₃
      M : ContinuousLinearEquiv (RingHom.id R) M✝ M₂
      ⊢ ((↑e).comp ↑M).IsInvertible
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp] lemma isInvertible_comp_equiv {e : M₃ ≃L[R] M} {f : M →L[R] M₂} :
    (f ∘L (e : M₃ →L[R] M)).IsInvertible ↔ f.IsInvertible := by
  /-
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    e : ContinuousLinearEquiv (RingHom.id R) M₃ M
    f : ContinuousLinearMap (RingHom.id R) M M₂
    ⊢ Iff (f.comp ↑e).IsInvertible f.IsInvertible
  -/
  constructor
    /-
      case mp
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₃ M
      f : ContinuousLinearMap (RingHom.id R) M M₂
      ⊢ (f.comp ↑e).IsInvertible → f.IsInvertible
    -/
  · rintro ⟨A, hA⟩
    /-
      case mp.intro
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₃ M
      f : ContinuousLinearMap (RingHom.id R) M M₂
      A : ContinuousLinearEquiv (RingHom.id R) M₃ M₂
      hA : Eq (↑A) (f.comp ↑e)
      ⊢ f.IsInvertible
    -/
    have : f = (f ∘L (e : M₃ →L[R] M)) ∘L e.symm := by ext; simp
    /-
      case mp.intro
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₃ M
      f : ContinuousLinearMap (RingHom.id R) M M₂
      A : ContinuousLinearEquiv (RingHom.id R) M₃ M₂
      hA : Eq (↑A) (f.comp ↑e)
      this : Eq f ((f.comp ↑e).comp ↑e.symm)
      ⊢ f.IsInvertible
    -/
    rw [this, ← hA]
    /-
      case mp.intro
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₃ M
      f : ContinuousLinearMap (RingHom.id R) M M₂
      A : ContinuousLinearEquiv (RingHom.id R) M₃ M₂
      hA : Eq (↑A) (f.comp ↑e)
      this : Eq f ((f.comp ↑e).comp ↑e.symm)
      ⊢ ((↑A).comp ↑e.symm).IsInvertible
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₃ M
      f : ContinuousLinearMap (RingHom.id R) M M₂
      ⊢ f.IsInvertible → (f.comp ↑e).IsInvertible
    -/
  · rintro ⟨M, rfl⟩
    /-
      case mpr.intro
      R : Type u_3
      M✝ : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M✝
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M✝
      inst✝⁴ : Module R M✝
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₃ M✝
      M : ContinuousLinearEquiv (RingHom.id R) M✝ M₂
      ⊢ ((↑M).comp ↑e).IsInvertible
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp] lemma inverse_equiv_comp {e : M₂ ≃L[R] M₃} {f : M →L[R] M₂} :
    (e ∘L f).inverse = f.inverse ∘L (e.symm : M₃ →L[R] M₂) := by
  /-
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    e : ContinuousLinearEquiv (RingHom.id R) M₂ M₃
    f : ContinuousLinearMap (RingHom.id R) M M₂
    ⊢ Eq ((↑e).comp f).inverse (f.inverse.comp ↑e.symm)
  -/
  by_cases hf : f.IsInvertible
    /-
      case pos
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₂ M₃
      f : ContinuousLinearMap (RingHom.id R) M M₂
      hf : f.IsInvertible
      ⊢ Eq ((↑e).comp f).inverse (f.inverse.comp ↑e.symm)
    -/
  · rcases hf with ⟨A, rfl⟩
    /-
      case pos.intro
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₂ M₃
      A : ContinuousLinearEquiv (RingHom.id R) M M₂
      ⊢ Eq ((↑e).comp ↑A).inverse ((↑A).inverse.comp ↑e.symm)
    -/
    simp only [ContinuousLinearEquiv.comp_coe, inverse_equiv, ContinuousLinearEquiv.coe_inj]
    /-
      case pos.intro
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₂ M₃
      A : ContinuousLinearEquiv (RingHom.id R) M M₂
      ⊢ Eq (A.trans e).symm (e.symm.trans A.symm)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₂ M₃
      f : ContinuousLinearMap (RingHom.id R) M M₂
      hf : Not f.IsInvertible
      ⊢ Eq ((↑e).comp f).inverse (f.inverse.comp ↑e.symm)
    -/
  · rw [inverse_of_not_isInvertible (by simp [hf]), inverse_of_not_isInvertible hf]
    /-
      case neg
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₂ M₃
      f : ContinuousLinearMap (RingHom.id R) M M₂
      hf : Not f.IsInvertible
      ⊢ Eq 0 (ContinuousLinearMap.comp 0 ↑e.symm)
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp] lemma inverse_comp_equiv {e : M₃ ≃L[R] M} {f : M →L[R] M₂} :
    (f ∘L e).inverse = (e.symm : M →L[R] M₃) ∘L f.inverse := by
  /-
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    e : ContinuousLinearEquiv (RingHom.id R) M₃ M
    f : ContinuousLinearMap (RingHom.id R) M M₂
    ⊢ Eq (f.comp ↑e).inverse ((↑e.symm).comp f.inverse)
  -/
  by_cases hf : f.IsInvertible
    /-
      case pos
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₃ M
      f : ContinuousLinearMap (RingHom.id R) M M₂
      hf : f.IsInvertible
      ⊢ Eq (f.comp ↑e).inverse ((↑e.symm).comp f.inverse)
    -/
  · rcases hf with ⟨A, rfl⟩
    /-
      case pos.intro
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₃ M
      A : ContinuousLinearEquiv (RingHom.id R) M M₂
      ⊢ Eq ((↑A).comp ↑e).inverse ((↑e.symm).comp (↑A).inverse)
    -/
    simp only [ContinuousLinearEquiv.comp_coe, inverse_equiv, ContinuousLinearEquiv.coe_inj]
    /-
      case pos.intro
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₃ M
      A : ContinuousLinearEquiv (RingHom.id R) M M₂
      ⊢ Eq (e.trans A).symm (A.symm.trans e.symm)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₃ M
      f : ContinuousLinearMap (RingHom.id R) M M₂
      hf : Not f.IsInvertible
      ⊢ Eq (f.comp ↑e).inverse ((↑e.symm).comp f.inverse)
    -/
  · rw [inverse_of_not_isInvertible (by simp [hf]), inverse_of_not_isInvertible hf]
    /-
      case neg
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      M₃ : Type u_6
      inst✝⁹ : TopologicalSpace M
      inst✝⁸ : TopologicalSpace M₂
      inst✝⁷ : TopologicalSpace M₃
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid M₂
      inst✝² : Module R M₂
      inst✝¹ : AddCommMonoid M₃
      inst✝ : Module R M₃
      e : ContinuousLinearEquiv (RingHom.id R) M₃ M
      f : ContinuousLinearMap (RingHom.id R) M M₂
      hf : Not f.IsInvertible
      ⊢ Eq 0 ((↑e.symm).comp 0)
    -/
    simp
    /-
      🎉 no goals
    -/


lemma IsInvertible.inverse_comp_of_left {g : M₂ →L[R] M₃} {f : M →L[R] M₂}
    (hg : g.IsInvertible) : (g ∘L f).inverse = f.inverse ∘L g.inverse := by
  /-
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    g : ContinuousLinearMap (RingHom.id R) M₂ M₃
    f : ContinuousLinearMap (RingHom.id R) M M₂
    hg : g.IsInvertible
    ⊢ Eq (g.comp f).inverse (f.inverse.comp g.inverse)
  -/
  rcases hg with ⟨N, rfl⟩
  /-
    case intro
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    f : ContinuousLinearMap (RingHom.id R) M M₂
    N : ContinuousLinearEquiv (RingHom.id R) M₂ M₃
    ⊢ Eq ((↑N).comp f).inverse (f.inverse.comp (↑N).inverse)
  -/
  simp
  /-
    🎉 no goals
  -/


lemma IsInvertible.inverse_comp_apply_of_left {g : M₂ →L[R] M₃} {f : M →L[R] M₂} {v : M₃}
    (hg : g.IsInvertible) : (g ∘L f).inverse v = f.inverse (g.inverse v) := by
  /-
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    g : ContinuousLinearMap (RingHom.id R) M₂ M₃
    f : ContinuousLinearMap (RingHom.id R) M M₂
    v : M₃
    hg : g.IsInvertible
    ⊢ Eq ((g.comp f).inverse v) (f.inverse (g.inverse v))
  -/
  simp only [hg.inverse_comp_of_left, coe_comp', Function.comp_apply]
  /-
    🎉 no goals
  -/


lemma IsInvertible.inverse_comp_of_right {g : M₂ →L[R] M₃} {f : M →L[R] M₂}
    (hf : f.IsInvertible) : (g ∘L f).inverse = f.inverse ∘L g.inverse := by
  /-
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    g : ContinuousLinearMap (RingHom.id R) M₂ M₃
    f : ContinuousLinearMap (RingHom.id R) M M₂
    hf : f.IsInvertible
    ⊢ Eq (g.comp f).inverse (f.inverse.comp g.inverse)
  -/
  rcases hf with ⟨M, rfl⟩
  /-
    case intro
    R : Type u_3
    M✝ : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁹ : TopologicalSpace M✝
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M✝
    inst✝⁴ : Module R M✝
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    g : ContinuousLinearMap (RingHom.id R) M₂ M₃
    M : ContinuousLinearEquiv (RingHom.id R) M✝ M₂
    ⊢ Eq (g.comp ↑M).inverse ((↑M).inverse.comp g.inverse)
  -/
  simp
  /-
    🎉 no goals
  -/


lemma IsInvertible.inverse_comp_apply_of_right {g : M₂ →L[R] M₃} {f : M →L[R] M₂} {v : M₃}
    (hf : f.IsInvertible) : (g ∘L f).inverse v = f.inverse (g.inverse v) := by
  /-
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    M₃ : Type u_6
    inst✝⁹ : TopologicalSpace M
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    inst✝¹ : AddCommMonoid M₃
    inst✝ : Module R M₃
    g : ContinuousLinearMap (RingHom.id R) M₂ M₃
    f : ContinuousLinearMap (RingHom.id R) M M₂
    v : M₃
    hf : f.IsInvertible
    ⊢ Eq ((g.comp f).inverse v) (f.inverse (g.inverse v))
  -/
  simp only [hf.inverse_comp_of_right, coe_comp', Function.comp_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem ring_inverse_equiv (e : M ≃L[R] M) : Ring.inverse ↑e = inverse (e : M →L[R] M) := by
  suffices Ring.inverse ((ContinuousLinearEquiv.unitsEquiv _ _).symm e : M →L[R] M) = inverse ↑e by
    convert this
  /-
    R : Type u_3
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    e : ContinuousLinearEquiv (RingHom.id R) M M
    ⊢ Eq (Ring.inverse ↑((ContinuousLinearEquiv.unitsEquiv R M).symm e)) (↑e).inve …
  -/
  simp
  /-
    R : Type u_3
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    e : ContinuousLinearEquiv (RingHom.id R) M M
    ⊢ Eq ↑(Inv.inv ((ContinuousLinearEquiv.unitsEquiv R M).symm e)) ↑e.symm
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The function `ContinuousLinearEquiv.inverse` can be written in terms of `Ring.inverse` for the
ring of self-maps of the domain. -/
theorem to_ring_inverse (e : M ≃L[R] M₂) (f : M →L[R] M₂) :
    inverse f = Ring.inverse ((e.symm : M₂ →L[R] M).comp f) ∘L e.symm := by
  /-
    R : Type u_3
    M : Type u_4
    M₂ : Type u_5
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : TopologicalSpace M₂
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M₂
    inst✝ : Module R M₂
    e : ContinuousLinearEquiv (RingHom.id R) M M₂
    f : ContinuousLinearMap (RingHom.id R) M M₂
    ⊢ Eq f.inverse ((Ring.inverse ((↑e.symm).comp f)).comp ↑e.symm)
  -/
  by_cases h₁ : f.IsInvertible
    /-
      case pos
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M₂
      inst✝ : Module R M₂
      e : ContinuousLinearEquiv (RingHom.id R) M M₂
      f : ContinuousLinearMap (RingHom.id R) M M₂
      h₁ : f.IsInvertible
      ⊢ Eq f.inverse ((Ring.inverse ((↑e.symm).comp f)).comp ↑e.symm)
    -/
  · obtain ⟨e', he'⟩ := h₁
    /-
      case pos.intro
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M₂
      inst✝ : Module R M₂
      e : ContinuousLinearEquiv (RingHom.id R) M M₂
      f : ContinuousLinearMap (RingHom.id R) M M₂
      e' : ContinuousLinearEquiv (RingHom.id R) M M₂
      he' : Eq (↑e') f
      ⊢ Eq f.inverse ((Ring.inverse ((↑e.symm).comp f)).comp ↑e.symm)
    -/
    rw [← he']
    /-
      case pos.intro
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M₂
      inst✝ : Module R M₂
      e : ContinuousLinearEquiv (RingHom.id R) M M₂
      f : ContinuousLinearMap (RingHom.id R) M M₂
      e' : ContinuousLinearEquiv (RingHom.id R) M M₂
      he' : Eq (↑e') f
      ⊢ Eq (↑e').inverse ((Ring.inverse ((↑e.symm).comp ↑e')).comp ↑e.symm)
    -/
    change _ = Ring.inverse (e'.trans e.symm : M →L[R] M) ∘L (e.symm : M₂ →L[R] M)
    /-
      case pos.intro
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M₂
      inst✝ : Module R M₂
      e : ContinuousLinearEquiv (RingHom.id R) M M₂
      f : ContinuousLinearMap (RingHom.id R) M M₂
      e' : ContinuousLinearEquiv (RingHom.id R) M M₂
      he' : Eq (↑e') f
      ⊢ Eq (↑e').inverse ((Ring.inverse ↑(e'.trans e.symm)).comp ↑e.symm)
    -/
    ext
    /-
      case pos.intro.h
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M₂
      inst✝ : Module R M₂
      e : ContinuousLinearEquiv (RingHom.id R) M M₂
      f : ContinuousLinearMap (RingHom.id R) M M₂
      e' : ContinuousLinearEquiv (RingHom.id R) M M₂
      he' : Eq (↑e') f
      x✝ : M₂
      ⊢ Eq ((↑e').inverse x✝) (((Ring.inverse ↑(e'.trans e.symm)).comp ↑e.symm) x✝)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M₂
      inst✝ : Module R M₂
      e : ContinuousLinearEquiv (RingHom.id R) M M₂
      f : ContinuousLinearMap (RingHom.id R) M M₂
      h₁ : Not f.IsInvertible
      ⊢ Eq f.inverse ((Ring.inverse ((↑e.symm).comp f)).comp ↑e.symm)
    -/
  · suffices ¬IsUnit ((e.symm : M₂ →L[R] M).comp f) by simp [this, h₁]
    /-
      case neg
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M₂
      inst✝ : Module R M₂
      e : ContinuousLinearEquiv (RingHom.id R) M M₂
      f : ContinuousLinearMap (RingHom.id R) M M₂
      h₁ : Not f.IsInvertible
      ⊢ Not (IsUnit ((↑e.symm).comp f))
    -/
    contrapose! h₁
    /-
      case neg
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M₂
      inst✝ : Module R M₂
      e : ContinuousLinearEquiv (RingHom.id R) M M₂
      f : ContinuousLinearMap (RingHom.id R) M M₂
      h₁ : IsUnit ((↑e.symm).comp f)
      ⊢ f.IsInvertible
    -/
    rcases h₁ with ⟨F, hF⟩
    /-
      case neg.intro
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M₂
      inst✝ : Module R M₂
      e : ContinuousLinearEquiv (RingHom.id R) M M₂
      f : ContinuousLinearMap (RingHom.id R) M M₂
      F : Units (ContinuousLinearMap (RingHom.id R) M M)
      hF : Eq (↑F) ((↑e.symm).comp f)
      ⊢ f.IsInvertible
    -/
    use (ContinuousLinearEquiv.unitsEquiv _ _ F).trans e
    /-
      case h
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M₂
      inst✝ : Module R M₂
      e : ContinuousLinearEquiv (RingHom.id R) M M₂
      f : ContinuousLinearMap (RingHom.id R) M M₂
      F : Units (ContinuousLinearMap (RingHom.id R) M M)
      hF : Eq (↑F) ((↑e.symm).comp f)
      ⊢ Eq (↑(((ContinuousLinearEquiv.unitsEquiv R M) F).trans e)) f
    -/
    ext
    /-
      case h.h
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M₂
      inst✝ : Module R M₂
      e : ContinuousLinearEquiv (RingHom.id R) M M₂
      f : ContinuousLinearMap (RingHom.id R) M M₂
      F : Units (ContinuousLinearMap (RingHom.id R) M M)
      hF : Eq (↑F) ((↑e.symm).comp f)
      x✝ : M
      ⊢ Eq (↑(((ContinuousLinearEquiv.unitsEquiv R M) F).trans e) x✝) (f x✝)
    -/
    dsimp
    /-
      case h.h
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M₂
      inst✝ : Module R M₂
      e : ContinuousLinearEquiv (RingHom.id R) M M₂
      f : ContinuousLinearMap (RingHom.id R) M M₂
      F : Units (ContinuousLinearMap (RingHom.id R) M M)
      hF : Eq (↑F) ((↑e.symm).comp f)
      x✝ : M
      ⊢ Eq (e (↑F x✝)) (f x✝)
    -/
    rw [hF]
    /-
      case h.h
      R : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : Ring R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M₂
      inst✝ : Module R M₂
      e : ContinuousLinearEquiv (RingHom.id R) M M₂
      f : ContinuousLinearMap (RingHom.id R) M M₂
      F : Units (ContinuousLinearMap (RingHom.id R) M M)
      hF : Eq (↑F) ((↑e.symm).comp f)
      x✝ : M
      ⊢ Eq (e (((↑e.symm).comp f) x✝)) (f x✝)
    -/
    simp
    /-
      🎉 no goals
    -/


theorem ring_inverse_eq_map_inverse : Ring.inverse = @inverse R M M _ _ _ _ _ _ _ := by
  /-
    R : Type u_3
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Eq Ring.inverse ContinuousLinearMap.inverse
  -/
  ext
  /-
    case h.h
    R : Type u_3
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x✝¹ : ContinuousLinearMap (RingHom.id R) M M
    x✝ : M
    ⊢ Eq ((Ring.inverse x✝¹) x✝) (x✝¹.inverse x✝)
  -/
  simp [to_ring_inverse (ContinuousLinearEquiv.refl R M)]
  /-
    🎉 no goals
  -/


@[simp] theorem inverse_id : (id R M).inverse = id R M := by
  /-
    R : Type u_3
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Eq (ContinuousLinearMap.id R M).inverse (ContinuousLinearMap.id R M)
  -/
  rw [← ring_inverse_eq_map_inverse]
  /-
    R : Type u_3
    M : Type u_4
    inst✝³ : TopologicalSpace M
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Eq (Ring.inverse (ContinuousLinearMap.id R M)) (ContinuousLinearMap.id R M)
  -/
  exact Ring.inverse_one _
  /-
    🎉 no goals
  -/


/-- If `p` is a closed complemented submodule,
then there exists a submodule `q` and a continuous linear equivalence `M ≃L[R] (p × q)` such that
`e (x : p) = (x, 0)`, `e (y : q) = (0, y)`, and `e.symm x = x.1 + x.2`.

In fact, the properties of `e` imply the properties of `e.symm` and vice versa,
but we provide both for convenience. -/
lemma ClosedComplemented.exists_submodule_equiv_prod [TopologicalAddGroup M]
    {p : Submodule R M} (hp : p.ClosedComplemented) :
    ∃ (q : Submodule R M) (e : M ≃L[R] (p × q)),
      (∀ x : p, e x = (x, 0)) ∧ (∀ y : q, e y = (0, y)) ∧ (∀ x, e.symm x = x.1 + x.2) :=
  let ⟨f, hf⟩ := hp
  ⟨LinearMap.ker f, .equivOfRightInverse _ p.subtypeL hf,
               /-
                 R : Type u_3
                 inst✝⁴ : Ring R
                 M : Type u_4
                 inst✝³ : TopologicalSpace M
                 inst✝² : AddCommGroup M
                 inst✝¹ : Module R M
                 inst✝ : TopologicalAddGroup M
                 p : Submodule R M
                 hp : p.ClosedComplemented
                 f : ContinuousLinearMap (RingHom.id R) M (Subtype fun x => Membership.mem p x)
                 hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
                 x✝ : Subtype fun x => Membership.mem p x
                 ⊢ Eq ((ContinuousLinearEquiv.equivOfRightInverse f p.subtypeL hf) ↑x✝) { fst : …
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
    fun _ ↦ by ext <;> simp [hf], fun _ ↦ by ext <;> simp [hf], fun _ ↦ rfl⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- The function `op` is a continuous linear equivalence. -/
@[simps!]
def opContinuousLinearEquiv : M ≃L[R] Mᵐᵒᵖ where
  __ := MulOpposite.opLinearEquiv R


