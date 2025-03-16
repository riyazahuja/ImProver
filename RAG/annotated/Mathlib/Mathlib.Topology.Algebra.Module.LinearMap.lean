/-- Continuous linear maps between modules. We only put the type classes that are necessary for the
definition, although in applications `M` and `M₂` will be topological modules over the topological
ring `R`. -/
structure ContinuousLinearMap {R : Type*} {S : Type*} [Semiring R] [Semiring S] (σ : R →+* S)
    (M : Type*) [TopologicalSpace M] [AddCommMonoid M] (M₂ : Type*) [TopologicalSpace M₂]
    [AddCommMonoid M₂] [Module R M] [Module S M₂] extends M →ₛₗ[σ] M₂ where
  cont : Continuous toFun := by continuity


@[inherit_doc]
notation:25 M " →SL[" σ "] " M₂ => ContinuousLinearMap σ M M₂


@[inherit_doc]
notation:25 M " →L[" R "] " M₂ => ContinuousLinearMap (RingHom.id R) M M₂


/-- `ContinuousSemilinearMapClass F σ M M₂` asserts `F` is a type of bundled continuous
`σ`-semilinear maps `M → M₂`.  See also `ContinuousLinearMapClass F R M M₂` for the case where
`σ` is the identity map on `R`.  A map `f` between an `R`-module and an `S`-module over a ring
homomorphism `σ : R →+* S` is semilinear if it satisfies the two properties `f (x + y) = f x + f y`
and `f (c • x) = (σ c) • f x`. -/
class ContinuousSemilinearMapClass (F : Type*) {R S : outParam Type*} [Semiring R] [Semiring S]
    (σ : outParam <| R →+* S) (M : outParam Type*) [TopologicalSpace M] [AddCommMonoid M]
    (M₂ : outParam Type*) [TopologicalSpace M₂] [AddCommMonoid M₂] [Module R M]
    [Module S M₂] [FunLike F M M₂]
    extends SemilinearMapClass F σ M M₂, ContinuousMapClass F M M₂ : Prop


/-- `ContinuousLinearMapClass F R M M₂` asserts `F` is a type of bundled continuous
`R`-linear maps `M → M₂`.  This is an abbreviation for
`ContinuousSemilinearMapClass F (RingHom.id R) M M₂`. -/
abbrev ContinuousLinearMapClass (F : Type*) (R : outParam Type*) [Semiring R]
    (M : outParam Type*) [TopologicalSpace M] [AddCommMonoid M] (M₂ : outParam Type*)
    [TopologicalSpace M₂] [AddCommMonoid M₂] [Module R M] [Module R M₂] [FunLike F M M₂] :=
  ContinuousSemilinearMapClass F (RingHom.id R) M M₂


/-- Coerce continuous linear maps to linear maps. -/
instance LinearMap.coe : Coe (M₁ →SL[σ₁₂] M₂) (M₁ →ₛₗ[σ₁₂] M₂) := ⟨toLinearMap⟩


theorem coe_injective : Function.Injective ((↑) : (M₁ →SL[σ₁₂] M₂) → M₁ →ₛₗ[σ₁₂] M₂) := by
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝⁷ : Semiring R₁
    inst✝⁶ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    M₁ : Type u_4
    inst✝⁵ : TopologicalSpace M₁
    inst✝⁴ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R₁ M₁
    inst✝ : Module R₂ M₂
    ⊢ Function.Injective ContinuousLinearMap.toLinearMap
  -/
  intro f g H
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝⁷ : Semiring R₁
    inst✝⁶ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    M₁ : Type u_4
    inst✝⁵ : TopologicalSpace M₁
    inst✝⁴ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R₁ M₁
    inst✝ : Module R₂ M₂
    f g : ContinuousLinearMap σ₁₂ M₁ M₂
    H : Eq ↑f ↑g
    ⊢ Eq f g
  -/
  cases f
  /-
    case mk
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝⁷ : Semiring R₁
    inst✝⁶ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    M₁ : Type u_4
    inst✝⁵ : TopologicalSpace M₁
    inst✝⁴ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R₁ M₁
    inst✝ : Module R₂ M₂
    g : ContinuousLinearMap σ₁₂ M₁ M₂
    toLinearMap✝ : LinearMap σ₁₂ M₁ M₂
    cont✝ : Continuous toLinearMap✝.toFun
    H : Eq ↑{ toLinearMap := toLinearMap✝, cont := cont✝ } ↑g
    ⊢ Eq { toLinearMap := toLinearMap✝, cont := cont✝ } g
  -/
  cases g
  /-
    case mk.mk
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝⁷ : Semiring R₁
    inst✝⁶ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    M₁ : Type u_4
    inst✝⁵ : TopologicalSpace M₁
    inst✝⁴ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R₁ M₁
    inst✝ : Module R₂ M₂
    toLinearMap✝¹ : LinearMap σ₁₂ M₁ M₂
    cont✝¹ : Continuous toLinearMap✝¹.toFun
    toLinearMap✝ : LinearMap σ₁₂ M₁ M₂
    cont✝ : Continuous toLinearMap✝.toFun
    H : Eq ↑{ toLinearMap := toLinearMap✝¹, cont := cont✝¹ } ↑{ toLinearMap := toL …
    ⊢ Eq { toLinearMap := toLinearMap✝¹, cont := cont✝¹ } { toLinearMap := toLinea …
  -/
  congr
  /-
    🎉 no goals
  -/


instance funLike : FunLike (M₁ →SL[σ₁₂] M₂) M₁ M₂ where
  coe f := f.toLinearMap
  coe_injective' _ _ h := coe_injective (DFunLike.coe_injective h)


instance continuousSemilinearMapClass :
    ContinuousSemilinearMapClass (M₁ →SL[σ₁₂] M₂) σ₁₂ M₁ M₂ where
  map_add f := map_add f.toLinearMap
  map_continuous f := f.2
  map_smulₛₗ f := f.toLinearMap.map_smul'


theorem coe_mk (f : M₁ →ₛₗ[σ₁₂] M₂) (h) : (mk f h : M₁ →ₛₗ[σ₁₂] M₂) = f :=
  rfl


@[simp]
theorem coe_mk' (f : M₁ →ₛₗ[σ₁₂] M₂) (h) : (mk f h : M₁ → M₂) = f :=
  rfl


@[continuity, fun_prop]
protected theorem continuous (f : M₁ →SL[σ₁₂] M₂) : Continuous f :=
  f.2


protected theorem uniformContinuous {E₁ E₂ : Type*} [UniformSpace E₁] [UniformSpace E₂]
    [AddCommGroup E₁] [AddCommGroup E₂] [Module R₁ E₁] [Module R₂ E₂] [UniformAddGroup E₁]
    [UniformAddGroup E₂] (f : E₁ →SL[σ₁₂] E₂) : UniformContinuous f :=
  uniformContinuous_addMonoidHom_of_continuous f.continuous


@[simp, norm_cast]
theorem coe_inj {f g : M₁ →SL[σ₁₂] M₂} : (f : M₁ →ₛₗ[σ₁₂] M₂) = g ↔ f = g :=
  coe_injective.eq_iff


theorem coeFn_injective : @Function.Injective (M₁ →SL[σ₁₂] M₂) (M₁ → M₂) (↑) :=
  DFunLike.coe_injective


/-- See Note [custom simps projection]. We need to specify this projection explicitly in this case,
  because it is a composition of multiple projections. -/
def Simps.apply (h : M₁ →SL[σ₁₂] M₂) : M₁ → M₂ :=
  h


/-- See Note [custom simps projection]. -/
def Simps.coe (h : M₁ →SL[σ₁₂] M₂) : M₁ →ₛₗ[σ₁₂] M₂ :=
  h


@[ext]
theorem ext {f g : M₁ →SL[σ₁₂] M₂} (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext f g h


/-- Copy of a `ContinuousLinearMap` with a new `toFun` equal to the old one. Useful to fix
definitional equalities. -/
protected def copy (f : M₁ →SL[σ₁₂] M₂) (f' : M₁ → M₂) (h : f' = ⇑f) : M₁ →SL[σ₁₂] M₂ where
  toLinearMap := f.toLinearMap.copy f' h
  cont := show Continuous f' from h.symm ▸ f.continuous


@[simp]
theorem coe_copy (f : M₁ →SL[σ₁₂] M₂) (f' : M₁ → M₂) (h : f' = ⇑f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : M₁ →SL[σ₁₂] M₂) (f' : M₁ → M₂) (h : f' = ⇑f) : f.copy f' h = f :=
  DFunLike.ext' h


theorem range_coeFn_eq :
    Set.range ((⇑) : (M₁ →SL[σ₁₂] M₂) → (M₁ → M₂)) =
      {f | Continuous f} ∩ Set.range ((⇑) : (M₁ →ₛₗ[σ₁₂] M₂) → (M₁ → M₂)) := by
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝⁷ : Semiring R₁
    inst✝⁶ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    M₁ : Type u_4
    inst✝⁵ : TopologicalSpace M₁
    inst✝⁴ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R₁ M₁
    inst✝ : Module R₂ M₂
    ⊢ Eq (Set.range DFunLike.coe) (Inter.inter (setOf fun f => Continuous f) (Set. …
  -/
  ext f
  /-
    case h
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝⁷ : Semiring R₁
    inst✝⁶ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    M₁ : Type u_4
    inst✝⁵ : TopologicalSpace M₁
    inst✝⁴ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R₁ M₁
    inst✝ : Module R₂ M₂
    f : M₁ → M₂
    ⊢ Iff (Membership.mem (Set.range DFunLike.coe) f) (Membership.mem (Inter.inter …
  -/
  constructor
    /-
      case h.mp
      R₁ : Type u_1
      R₂ : Type u_2
      inst✝⁷ : Semiring R₁
      inst✝⁶ : Semiring R₂
      σ₁₂ : RingHom R₁ R₂
      M₁ : Type u_4
      inst✝⁵ : TopologicalSpace M₁
      inst✝⁴ : AddCommMonoid M₁
      M₂ : Type u_6
      inst✝³ : TopologicalSpace M₂
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R₁ M₁
      inst✝ : Module R₂ M₂
      f : M₁ → M₂
      ⊢ Membership.mem (Set.range DFunLike.coe) f → Membership.mem (Inter.inter (set …
    -/
  · rintro ⟨f, rfl⟩
    /-
      case h.mp.intro
      R₁ : Type u_1
      R₂ : Type u_2
      inst✝⁷ : Semiring R₁
      inst✝⁶ : Semiring R₂
      σ₁₂ : RingHom R₁ R₂
      M₁ : Type u_4
      inst✝⁵ : TopologicalSpace M₁
      inst✝⁴ : AddCommMonoid M₁
      M₂ : Type u_6
      inst✝³ : TopologicalSpace M₂
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R₁ M₁
      inst✝ : Module R₂ M₂
      f : ContinuousLinearMap σ₁₂ M₁ M₂
      ⊢ Membership.mem (Inter.inter (setOf fun f => Continuous f) (Set.range DFunLik …
    -/
    exact ⟨f.continuous, f, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R₁ : Type u_1
      R₂ : Type u_2
      inst✝⁷ : Semiring R₁
      inst✝⁶ : Semiring R₂
      σ₁₂ : RingHom R₁ R₂
      M₁ : Type u_4
      inst✝⁵ : TopologicalSpace M₁
      inst✝⁴ : AddCommMonoid M₁
      M₂ : Type u_6
      inst✝³ : TopologicalSpace M₂
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R₁ M₁
      inst✝ : Module R₂ M₂
      f : M₁ → M₂
      ⊢ Membership.mem (Inter.inter (setOf fun f => Continuous f) (Set.range DFunLik …
    -/
  · rintro ⟨hfc, f, rfl⟩
    /-
      case h.mpr.intro.intro
      R₁ : Type u_1
      R₂ : Type u_2
      inst✝⁷ : Semiring R₁
      inst✝⁶ : Semiring R₂
      σ₁₂ : RingHom R₁ R₂
      M₁ : Type u_4
      inst✝⁵ : TopologicalSpace M₁
      inst✝⁴ : AddCommMonoid M₁
      M₂ : Type u_6
      inst✝³ : TopologicalSpace M₂
      inst✝² : AddCommMonoid M₂
      inst✝¹ : Module R₁ M₁
      inst✝ : Module R₂ M₂
      f : LinearMap σ₁₂ M₁ M₂
      hfc : Membership.mem (setOf fun f => Continuous f) ⇑f
      ⊢ Membership.mem (Set.range DFunLike.coe) ⇑f
    -/
    exact ⟨⟨f, hfc⟩, rfl⟩
    /-
      🎉 no goals
    -/

-- make some straightforward lemmas available to `simp`.

protected theorem map_zero (f : M₁ →SL[σ₁₂] M₂) : f (0 : M₁) = 0 :=
  map_zero f


protected theorem map_add (f : M₁ →SL[σ₁₂] M₂) (x y : M₁) : f (x + y) = f x + f y :=
  map_add f x y


@[simp]
protected theorem map_smulₛₗ (f : M₁ →SL[σ₁₂] M₂) (c : R₁) (x : M₁) : f (c • x) = σ₁₂ c • f x :=
  (toLinearMap _).map_smulₛₗ _ _


protected theorem map_smul [Module R₁ M₂] (f : M₁ →L[R₁] M₂) (c : R₁) (x : M₁) :
                              /-
                                R₁ : Type u_1
                                inst✝⁶ : Semiring R₁
                                M₁ : Type u_4
                                inst✝⁵ : TopologicalSpace M₁
                                inst✝⁴ : AddCommMonoid M₁
                                M₂ : Type u_6
                                inst✝³ : TopologicalSpace M₂
                                inst✝² : AddCommMonoid M₂
                                inst✝¹ : Module R₁ M₁
                                inst✝ : Module R₁ M₂
                                f : ContinuousLinearMap (RingHom.id R₁) M₁ M₂
                                c : R₁
                                x : M₁
                                ⊢ Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                              -/
    f (c • x) = c • f x := by simp only [RingHom.id_apply, ContinuousLinearMap.map_smulₛₗ]
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem map_smul_of_tower {R S : Type*} [Semiring S] [SMul R M₁] [Module S M₁] [SMul R M₂]
    [Module S M₂] [LinearMap.CompatibleSMul M₁ M₂ R S] (f : M₁ →L[S] M₂) (c : R) (x : M₁) :
    f (c • x) = c • f x :=
  LinearMap.CompatibleSMul.map_smul (f : M₁ →ₗ[S] M₂) c x


@[simp, norm_cast]
theorem coe_coe (f : M₁ →SL[σ₁₂] M₂) : ⇑(f : M₁ →ₛₗ[σ₁₂] M₂) = f :=
  rfl


@[ext]
theorem ext_ring [TopologicalSpace R₁] {f g : R₁ →L[R₁] M₁} (h : f 1 = g 1) : f = g :=
  coe_inj.1 <| LinearMap.ext_ring h


/-- If two continuous linear maps are equal on a set `s`, then they are equal on the closure
of the `Submodule.span` of this set. -/
theorem eqOn_closure_span [T2Space M₂] {s : Set M₁} {f g : M₁ →SL[σ₁₂] M₂} (h : Set.EqOn f g s) :
    Set.EqOn f g (closure (Submodule.span R₁ s : Set M₁)) :=
  (LinearMap.eqOn_span' h).closure f.continuous g.continuous


/-- If the submodule generated by a set `s` is dense in the ambient module, then two continuous
linear maps equal on `s` are equal. -/
theorem ext_on [T2Space M₂] {s : Set M₁} (hs : Dense (Submodule.span R₁ s : Set M₁))
    {f g : M₁ →SL[σ₁₂] M₂} (h : Set.EqOn f g s) : f = g :=
  ext fun x => eqOn_closure_span h (hs x)


/-- Under a continuous linear map, the image of the `TopologicalClosure` of a submodule is
contained in the `TopologicalClosure` of its image. -/
theorem _root_.Submodule.topologicalClosure_map [RingHomSurjective σ₁₂] [TopologicalSpace R₁]
    [TopologicalSpace R₂] [ContinuousSMul R₁ M₁] [ContinuousAdd M₁] [ContinuousSMul R₂ M₂]
    [ContinuousAdd M₂] (f : M₁ →SL[σ₁₂] M₂) (s : Submodule R₁ M₁) :
    s.topologicalClosure.map (f : M₁ →ₛₗ[σ₁₂] M₂) ≤
      (s.map (f : M₁ →ₛₗ[σ₁₂] M₂)).topologicalClosure :=
  image_closure_subset_closure_image f.continuous


/-- Under a dense continuous linear map, a submodule whose `TopologicalClosure` is `⊤` is sent to
another such submodule.  That is, the image of a dense set under a map with dense range is dense.
-/
theorem _root_.DenseRange.topologicalClosure_map_submodule [RingHomSurjective σ₁₂]
    [TopologicalSpace R₁] [TopologicalSpace R₂] [ContinuousSMul R₁ M₁] [ContinuousAdd M₁]
    [ContinuousSMul R₂ M₂] [ContinuousAdd M₂] {f : M₁ →SL[σ₁₂] M₂} (hf' : DenseRange f)
    {s : Submodule R₁ M₁} (hs : s.topologicalClosure = ⊤) :
    (s.map (f : M₁ →ₛₗ[σ₁₂] M₂)).topologicalClosure = ⊤ := by
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝¹⁴ : Semiring R₁
    inst✝¹³ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    M₁ : Type u_4
    inst✝¹² : TopologicalSpace M₁
    inst✝¹¹ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝¹⁰ : TopologicalSpace M₂
    inst✝⁹ : AddCommMonoid M₂
    inst✝⁸ : Module R₁ M₁
    inst✝⁷ : Module R₂ M₂
    inst✝⁶ : RingHomSurjective σ₁₂
    inst✝⁵ : TopologicalSpace R₁
    inst✝⁴ : TopologicalSpace R₂
    inst✝³ : ContinuousSMul R₁ M₁
    inst✝² : ContinuousAdd M₁
    inst✝¹ : ContinuousSMul R₂ M₂
    inst✝ : ContinuousAdd M₂
    f : ContinuousLinearMap σ₁₂ M₁ M₂
    hf' : DenseRange ⇑f
    s : Submodule R₁ M₁
    hs : Eq s.topologicalClosure Top.top
    ⊢ Eq (Submodule.map (↑f) s).topologicalClosure Top.top
  -/
  rw [SetLike.ext'_iff] at hs ⊢
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝¹⁴ : Semiring R₁
    inst✝¹³ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    M₁ : Type u_4
    inst✝¹² : TopologicalSpace M₁
    inst✝¹¹ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝¹⁰ : TopologicalSpace M₂
    inst✝⁹ : AddCommMonoid M₂
    inst✝⁸ : Module R₁ M₁
    inst✝⁷ : Module R₂ M₂
    inst✝⁶ : RingHomSurjective σ₁₂
    inst✝⁵ : TopologicalSpace R₁
    inst✝⁴ : TopologicalSpace R₂
    inst✝³ : ContinuousSMul R₁ M₁
    inst✝² : ContinuousAdd M₁
    inst✝¹ : ContinuousSMul R₂ M₂
    inst✝ : ContinuousAdd M₂
    f : ContinuousLinearMap σ₁₂ M₁ M₂
    hf' : DenseRange ⇑f
    s : Submodule R₁ M₁
    hs : Eq ↑s.topologicalClosure ↑Top.top
    ⊢ Eq ↑(Submodule.map (↑f) s).topologicalClosure ↑Top.top
  -/
  simp only [Submodule.topologicalClosure_coe, Submodule.top_coe, ← dense_iff_closure_eq] at hs ⊢
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝¹⁴ : Semiring R₁
    inst✝¹³ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    M₁ : Type u_4
    inst✝¹² : TopologicalSpace M₁
    inst✝¹¹ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝¹⁰ : TopologicalSpace M₂
    inst✝⁹ : AddCommMonoid M₂
    inst✝⁸ : Module R₁ M₁
    inst✝⁷ : Module R₂ M₂
    inst✝⁶ : RingHomSurjective σ₁₂
    inst✝⁵ : TopologicalSpace R₁
    inst✝⁴ : TopologicalSpace R₂
    inst✝³ : ContinuousSMul R₁ M₁
    inst✝² : ContinuousAdd M₁
    inst✝¹ : ContinuousSMul R₂ M₂
    inst✝ : ContinuousAdd M₂
    f : ContinuousLinearMap σ₁₂ M₁ M₂
    hf' : DenseRange ⇑f
    s : Submodule R₁ M₁
    hs : Dense ↑s
    ⊢ Dense ↑(Submodule.map (↑f) s)
  -/
  exact hf'.dense_image f.continuous hs
  /-
    🎉 no goals
  -/


instance instSMul : SMul S₂ (M₁ →SL[σ₁₂] M₂) where
  smul c f := ⟨c • (f : M₁ →ₛₗ[σ₁₂] M₂), (f.2.const_smul _ : Continuous fun x => c • f x)⟩


instance mulAction : MulAction S₂ (M₁ →SL[σ₁₂] M₂) where
  one_smul _f := ext fun _x => one_smul _ _
  mul_smul _a _b _f := ext fun _x => mul_smul _ _ _


theorem smul_apply (c : S₂) (f : M₁ →SL[σ₁₂] M₂) (x : M₁) : (c • f) x = c • f x :=
  rfl


@[simp, norm_cast]
theorem coe_smul (c : S₂) (f : M₁ →SL[σ₁₂] M₂) :
    ↑(c • f) = c • (f : M₁ →ₛₗ[σ₁₂] M₂) :=
  rfl


@[simp, norm_cast]
theorem coe_smul' (c : S₂) (f : M₁ →SL[σ₁₂] M₂) :
    ↑(c • f) = c • (f : M₁ → M₂) :=
  rfl


instance isScalarTower [SMul S₂ T₂] [IsScalarTower S₂ T₂ M₂] :
    IsScalarTower S₂ T₂ (M₁ →SL[σ₁₂] M₂) :=
  ⟨fun a b f => ext fun x => smul_assoc a b (f x)⟩


instance smulCommClass [SMulCommClass S₂ T₂ M₂] : SMulCommClass S₂ T₂ (M₁ →SL[σ₁₂] M₂) :=
  ⟨fun a b f => ext fun x => smul_comm a b (f x)⟩


/-- The continuous map that is constantly zero. -/
instance zero : Zero (M₁ →SL[σ₁₂] M₂) :=
  ⟨⟨0, continuous_zero⟩⟩


instance inhabited : Inhabited (M₁ →SL[σ₁₂] M₂) :=
  ⟨0⟩


@[simp]
theorem default_def : (default : M₁ →SL[σ₁₂] M₂) = 0 :=
  rfl


@[simp]
theorem zero_apply (x : M₁) : (0 : M₁ →SL[σ₁₂] M₂) x = 0 :=
  rfl


@[simp, norm_cast]
theorem coe_zero : ((0 : M₁ →SL[σ₁₂] M₂) : M₁ →ₛₗ[σ₁₂] M₂) = 0 :=
  rfl

/- no simp attribute on the next line as simp does not always simplify `0 x` to `0`
when `0` is the zero function, while it does for the zero continuous linear map,
and this is the most important property we care about. -/

@[norm_cast]
theorem coe_zero' : ⇑(0 : M₁ →SL[σ₁₂] M₂) = 0 :=
  rfl


instance uniqueOfLeft [Subsingleton M₁] : Unique (M₁ →SL[σ₁₂] M₂) :=
  coe_injective.unique


instance uniqueOfRight [Subsingleton M₂] : Unique (M₁ →SL[σ₁₂] M₂) :=
  coe_injective.unique


theorem exists_ne_zero {f : M₁ →SL[σ₁₂] M₂} (hf : f ≠ 0) : ∃ x, f x ≠ 0 := by
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝⁷ : Semiring R₁
    inst✝⁶ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    M₁ : Type u_4
    inst✝⁵ : TopologicalSpace M₁
    inst✝⁴ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R₁ M₁
    inst✝ : Module R₂ M₂
    f : ContinuousLinearMap σ₁₂ M₁ M₂
    hf : Ne f 0
    ⊢ Exists fun x => Ne (f x) 0
  -/
  by_contra! h
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    inst✝⁷ : Semiring R₁
    inst✝⁶ : Semiring R₂
    σ₁₂ : RingHom R₁ R₂
    M₁ : Type u_4
    inst✝⁵ : TopologicalSpace M₁
    inst✝⁴ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R₁ M₁
    inst✝ : Module R₂ M₂
    f : ContinuousLinearMap σ₁₂ M₁ M₂
    hf : Ne f 0
    h : ∀ (x : M₁), Eq (f x) 0
    ⊢ False
  -/
  exact hf (ContinuousLinearMap.ext h)
  /-
    🎉 no goals
  -/


/-- the identity map as a continuous linear map. -/
def id : M₁ →L[R₁] M₁ :=
  ⟨LinearMap.id, continuous_id⟩


instance one : One (M₁ →L[R₁] M₁) :=
  ⟨id R₁ M₁⟩


theorem one_def : (1 : M₁ →L[R₁] M₁) = id R₁ M₁ :=
  rfl


theorem id_apply (x : M₁) : id R₁ M₁ x = x :=
  rfl


@[simp, norm_cast]
theorem coe_id : (id R₁ M₁ : M₁ →ₗ[R₁] M₁) = LinearMap.id :=
  rfl


@[simp, norm_cast]
theorem coe_id' : ⇑(id R₁ M₁) = _root_.id :=
  rfl


@[simp, norm_cast]
theorem coe_eq_id {f : M₁ →L[R₁] M₁} : (f : M₁ →ₗ[R₁] M₁) = LinearMap.id ↔ f = id _ _ := by
  /-
    R₁ : Type u_1
    inst✝³ : Semiring R₁
    M₁ : Type u_4
    inst✝² : TopologicalSpace M₁
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    f : ContinuousLinearMap (RingHom.id R₁) M₁ M₁
    ⊢ Iff (Eq (↑f) LinearMap.id) (Eq f (ContinuousLinearMap.id R₁ M₁))
  -/
  rw [← coe_id, coe_inj]
  /-
    🎉 no goals
  -/


@[simp]
theorem one_apply (x : M₁) : (1 : M₁ →L[R₁] M₁) x = x :=
  rfl


instance [Nontrivial M₁] : Nontrivial (M₁ →L[R₁] M₁) :=
  ⟨0, 1, fun e ↦
                                               /-
                                                 R₁ : Type u_1
                                                 R₂ : Type u_2
                                                 R₃ : Type u_3
                                                 inst✝¹⁷ : Semiring R₁
                                                 inst✝¹⁶ : Semiring R₂
                                                 inst✝¹⁵ : Semiring R₃
                                                 σ₁₂ : RingHom R₁ R₂
                                                 σ₂₃ : RingHom R₂ R₃
                                                 σ₁₃ : RingHom R₁ R₃
                                                 M₁ : Type u_4
                                                 inst✝¹⁴ : TopologicalSpace M₁
                                                 inst✝¹³ : AddCommMonoid M₁
                                                 M'₁ : Type u_5
                                                 inst✝¹² : TopologicalSpace M'₁
                                                 inst✝¹¹ : AddCommMonoid M'₁
                                                 M₂ : Type u_6
                                                 inst✝¹⁰ : TopologicalSpace M₂
                                                 inst✝⁹ : AddCommMonoid M₂
                                                 M₃ : Type u_7
                                                 inst✝⁸ : TopologicalSpace M₃
                                                 inst✝⁷ : AddCommMonoid M₃
                                                 M₄ : Type u_8
                                                 inst✝⁶ : TopologicalSpace M₄
                                                 inst✝⁵ : AddCommMonoid M₄
                                                 inst✝⁴ : Module R₁ M₁
                                                 inst✝³ : Module R₁ M'₁
                                                 inst✝² : Module R₂ M₂
                                                 inst✝¹ : Module R₃ M₃
                                                 inst✝ : Nontrivial M₁
                                                 e : Eq 0 1
                                                 x : M₁
                                                 hx : Ne x 0
                                                 ⊢ Eq x 0
                                               -/
    have ⟨x, hx⟩ := exists_ne (0 : M₁); hx (by simpa using DFunLike.congr_fun e.symm x)⟩
                                               /-
                                                 🎉 no goals
                                               -/


instance add : Add (M₁ →SL[σ₁₂] M₂) :=
  ⟨fun f g => ⟨f + g, f.2.add g.2⟩⟩


@[simp]
theorem add_apply (f g : M₁ →SL[σ₁₂] M₂) (x : M₁) : (f + g) x = f x + g x :=
  rfl


@[simp, norm_cast]
theorem coe_add (f g : M₁ →SL[σ₁₂] M₂) : (↑(f + g) : M₁ →ₛₗ[σ₁₂] M₂) = f + g :=
  rfl


@[norm_cast]
theorem coe_add' (f g : M₁ →SL[σ₁₂] M₂) : ⇑(f + g) = f + g :=
  rfl


instance addCommMonoid : AddCommMonoid (M₁ →SL[σ₁₂] M₂) where
  zero_add := by
    /-
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝¹⁷ : Semiring R₁
      inst✝¹⁶ : Semiring R₂
      inst✝¹⁵ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      M₁ : Type u_4
      inst✝¹⁴ : TopologicalSpace M₁
      inst✝¹³ : AddCommMonoid M₁
      M'₁ : Type u_5
      inst✝¹² : TopologicalSpace M'₁
      inst✝¹¹ : AddCommMonoid M'₁
      M₂ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₂
      inst✝⁹ : AddCommMonoid M₂
      M₃ : Type u_7
      inst✝⁸ : TopologicalSpace M₃
      inst✝⁷ : AddCommMonoid M₃
      M₄ : Type u_8
      inst✝⁶ : TopologicalSpace M₄
      inst✝⁵ : AddCommMonoid M₄
      inst✝⁴ : Module R₁ M₁
      inst✝³ : Module R₁ M'₁
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      inst✝ : ContinuousAdd M₂
      ⊢ ∀ (a : ContinuousLinearMap σ₁₂ M₁ M₂), Eq (HAdd.hAdd 0 a) a
    -/
    intros
    /-
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝¹⁷ : Semiring R₁
      inst✝¹⁶ : Semiring R₂
      inst✝¹⁵ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      M₁ : Type u_4
      inst✝¹⁴ : TopologicalSpace M₁
      inst✝¹³ : AddCommMonoid M₁
      M'₁ : Type u_5
      inst✝¹² : TopologicalSpace M'₁
      inst✝¹¹ : AddCommMonoid M'₁
      M₂ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₂
      inst✝⁹ : AddCommMonoid M₂
      M₃ : Type u_7
      inst✝⁸ : TopologicalSpace M₃
      inst✝⁷ : AddCommMonoid M₃
      M₄ : Type u_8
      inst✝⁶ : TopologicalSpace M₄
      inst✝⁵ : AddCommMonoid M₄
      inst✝⁴ : Module R₁ M₁
      inst✝³ : Module R₁ M'₁
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      inst✝ : ContinuousAdd M₂
      a✝ : ContinuousLinearMap σ₁₂ M₁ M₂
      ⊢ Eq (HAdd.hAdd 0 a✝) a✝
    -/
    ext
    /-
      case h
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝¹⁷ : Semiring R₁
      inst✝¹⁶ : Semiring R₂
      inst✝¹⁵ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      M₁ : Type u_4
      inst✝¹⁴ : TopologicalSpace M₁
      inst✝¹³ : AddCommMonoid M₁
      M'₁ : Type u_5
      inst✝¹² : TopologicalSpace M'₁
      inst✝¹¹ : AddCommMonoid M'₁
      M₂ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₂
      inst✝⁹ : AddCommMonoid M₂
      M₃ : Type u_7
      inst✝⁸ : TopologicalSpace M₃
      inst✝⁷ : AddCommMonoid M₃
      M₄ : Type u_8
      inst✝⁶ : TopologicalSpace M₄
      inst✝⁵ : AddCommMonoid M₄
      inst✝⁴ : Module R₁ M₁
      inst✝³ : Module R₁ M'₁
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      inst✝ : ContinuousAdd M₂
      a✝ : ContinuousLinearMap σ₁₂ M₁ M₂
      x✝ : M₁
      ⊢ Eq ((HAdd.hAdd 0 a✝) x✝) (a✝ x✝)
    -/
    apply_rules [zero_add, add_assoc, add_zero, neg_add_cancel, add_comm]
    /-
      🎉 no goals
    -/
  add_zero := by
    /-
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝¹⁷ : Semiring R₁
      inst✝¹⁶ : Semiring R₂
      inst✝¹⁵ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      M₁ : Type u_4
      inst✝¹⁴ : TopologicalSpace M₁
      inst✝¹³ : AddCommMonoid M₁
      M'₁ : Type u_5
      inst✝¹² : TopologicalSpace M'₁
      inst✝¹¹ : AddCommMonoid M'₁
      M₂ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₂
      inst✝⁹ : AddCommMonoid M₂
      M₃ : Type u_7
      inst✝⁸ : TopologicalSpace M₃
      inst✝⁷ : AddCommMonoid M₃
      M₄ : Type u_8
      inst✝⁶ : TopologicalSpace M₄
      inst✝⁵ : AddCommMonoid M₄
      inst✝⁴ : Module R₁ M₁
      inst✝³ : Module R₁ M'₁
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      inst✝ : ContinuousAdd M₂
      ⊢ ∀ (a : ContinuousLinearMap σ₁₂ M₁ M₂), Eq (HAdd.hAdd a 0) a
    -/
    intros
    /-
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝¹⁷ : Semiring R₁
      inst✝¹⁶ : Semiring R₂
      inst✝¹⁵ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      M₁ : Type u_4
      inst✝¹⁴ : TopologicalSpace M₁
      inst✝¹³ : AddCommMonoid M₁
      M'₁ : Type u_5
      inst✝¹² : TopologicalSpace M'₁
      inst✝¹¹ : AddCommMonoid M'₁
      M₂ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₂
      inst✝⁹ : AddCommMonoid M₂
      M₃ : Type u_7
      inst✝⁸ : TopologicalSpace M₃
      inst✝⁷ : AddCommMonoid M₃
      M₄ : Type u_8
      inst✝⁶ : TopologicalSpace M₄
      inst✝⁵ : AddCommMonoid M₄
      inst✝⁴ : Module R₁ M₁
      inst✝³ : Module R₁ M'₁
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      inst✝ : ContinuousAdd M₂
      a✝ : ContinuousLinearMap σ₁₂ M₁ M₂
      ⊢ Eq (HAdd.hAdd a✝ 0) a✝
    -/
    ext
    /-
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝¹⁷ : Semiring R₁
      inst✝¹⁶ : Semiring R₂
      inst✝¹⁵ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      M₁ : Type u_4
      inst✝¹⁴ : TopologicalSpace M₁
      inst✝¹³ : AddCommMonoid M₁
      M'₁ : Type u_5
      inst✝¹² : TopologicalSpace M'₁
      inst✝¹¹ : AddCommMonoid M'₁
      M₂ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₂
      inst✝⁹ : AddCommMonoid M₂
      M₃ : Type u_7
      inst✝⁸ : TopologicalSpace M₃
      inst✝⁷ : AddCommMonoid M₃
      M₄ : Type u_8
      inst✝⁶ : TopologicalSpace M₄
      inst✝⁵ : AddCommMonoid M₄
      inst✝⁴ : Module R₁ M₁
      inst✝³ : Module R₁ M'₁
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      inst✝ : ContinuousAdd M₂
      ⊢ ∀ (a b c : ContinuousLinearMap σ₁₂ M₁ M₂), Eq (HAdd.hAdd (HAdd.hAdd a b) c)  …
    -/
    /-
      case h
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝¹⁷ : Semiring R₁
      inst✝¹⁶ : Semiring R₂
      inst✝¹⁵ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      M₁ : Type u_4
      inst✝¹⁴ : TopologicalSpace M₁
      inst✝¹³ : AddCommMonoid M₁
      M'₁ : Type u_5
      inst✝¹² : TopologicalSpace M'₁
      inst✝¹¹ : AddCommMonoid M'₁
      M₂ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₂
      inst✝⁹ : AddCommMonoid M₂
      M₃ : Type u_7
      inst✝⁸ : TopologicalSpace M₃
      inst✝⁷ : AddCommMonoid M₃
      M₄ : Type u_8
      inst✝⁶ : TopologicalSpace M₄
      inst✝⁵ : AddCommMonoid M₄
      inst✝⁴ : Module R₁ M₁
      inst✝³ : Module R₁ M'₁
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      inst✝ : ContinuousAdd M₂
      a✝ : ContinuousLinearMap σ₁₂ M₁ M₂
      x✝ : M₁
      ⊢ Eq ((HAdd.hAdd a✝ 0) x✝) (a✝ x✝)
    -/
    /-
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝¹⁷ : Semiring R₁
      inst✝¹⁶ : Semiring R₂
      inst✝¹⁵ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      M₁ : Type u_4
      inst✝¹⁴ : TopologicalSpace M₁
      inst✝¹³ : AddCommMonoid M₁
      M'₁ : Type u_5
      inst✝¹² : TopologicalSpace M'₁
      inst✝¹¹ : AddCommMonoid M'₁
      M₂ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₂
      inst✝⁹ : AddCommMonoid M₂
      M₃ : Type u_7
      inst✝⁸ : TopologicalSpace M₃
      inst✝⁷ : AddCommMonoid M₃
      M₄ : Type u_8
      inst✝⁶ : TopologicalSpace M₄
      inst✝⁵ : AddCommMonoid M₄
      inst✝⁴ : Module R₁ M₁
      inst✝³ : Module R₁ M'₁
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      inst✝ : ContinuousAdd M₂
      a✝ b✝ c✝ : ContinuousLinearMap σ₁₂ M₁ M₂
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd a✝ b✝) c✝) (HAdd.hAdd a✝ (HAdd.hAdd b✝ c✝))
    -/
    apply_rules [zero_add, add_assoc, add_zero, neg_add_cancel, add_comm]
    /-
      case h
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝¹⁷ : Semiring R₁
      inst✝¹⁶ : Semiring R₂
      inst✝¹⁵ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      M₁ : Type u_4
      inst✝¹⁴ : TopologicalSpace M₁
      inst✝¹³ : AddCommMonoid M₁
      M'₁ : Type u_5
      inst✝¹² : TopologicalSpace M'₁
      inst✝¹¹ : AddCommMonoid M'₁
      M₂ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₂
      inst✝⁹ : AddCommMonoid M₂
      M₃ : Type u_7
      inst✝⁸ : TopologicalSpace M₃
      inst✝⁷ : AddCommMonoid M₃
      M₄ : Type u_8
      inst✝⁶ : TopologicalSpace M₄
      inst✝⁵ : AddCommMonoid M₄
      inst✝⁴ : Module R₁ M₁
      inst✝³ : Module R₁ M'₁
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      inst✝ : ContinuousAdd M₂
      a✝ b✝ c✝ : ContinuousLinearMap σ₁₂ M₁ M₂
      x✝ : M₁
      ⊢ Eq ((HAdd.hAdd (HAdd.hAdd a✝ b✝) c✝) x✝) ((HAdd.hAdd a✝ (HAdd.hAdd b✝ c✝)) x✝)
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  add_comm := by
    /-
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝¹⁷ : Semiring R₁
      inst✝¹⁶ : Semiring R₂
      inst✝¹⁵ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      M₁ : Type u_4
      inst✝¹⁴ : TopologicalSpace M₁
      inst✝¹³ : AddCommMonoid M₁
      M'₁ : Type u_5
      inst✝¹² : TopologicalSpace M'₁
      inst✝¹¹ : AddCommMonoid M'₁
      M₂ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₂
      inst✝⁹ : AddCommMonoid M₂
      M₃ : Type u_7
      inst✝⁸ : TopologicalSpace M₃
      inst✝⁷ : AddCommMonoid M₃
      M₄ : Type u_8
      inst✝⁶ : TopologicalSpace M₄
      inst✝⁵ : AddCommMonoid M₄
      inst✝⁴ : Module R₁ M₁
      inst✝³ : Module R₁ M'₁
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      inst✝ : ContinuousAdd M₂
      ⊢ ∀ (a b : ContinuousLinearMap σ₁₂ M₁ M₂), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
    -/
    intros
    /-
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝¹⁷ : Semiring R₁
      inst✝¹⁶ : Semiring R₂
      inst✝¹⁵ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      M₁ : Type u_4
      inst✝¹⁴ : TopologicalSpace M₁
      inst✝¹³ : AddCommMonoid M₁
      M'₁ : Type u_5
      inst✝¹² : TopologicalSpace M'₁
      inst✝¹¹ : AddCommMonoid M'₁
      M₂ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₂
      inst✝⁹ : AddCommMonoid M₂
      M₃ : Type u_7
      inst✝⁸ : TopologicalSpace M₃
      inst✝⁷ : AddCommMonoid M₃
      M₄ : Type u_8
      inst✝⁶ : TopologicalSpace M₄
      inst✝⁵ : AddCommMonoid M₄
      inst✝⁴ : Module R₁ M₁
      inst✝³ : Module R₁ M'₁
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      inst✝ : ContinuousAdd M₂
      a✝ b✝ : ContinuousLinearMap σ₁₂ M₁ M₂
      ⊢ Eq (HAdd.hAdd a✝ b✝) (HAdd.hAdd b✝ a✝)
    -/
    ext
    /-
      case h
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝¹⁷ : Semiring R₁
      inst✝¹⁶ : Semiring R₂
      inst✝¹⁵ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      M₁ : Type u_4
      inst✝¹⁴ : TopologicalSpace M₁
      inst✝¹³ : AddCommMonoid M₁
      M'₁ : Type u_5
      inst✝¹² : TopologicalSpace M'₁
      inst✝¹¹ : AddCommMonoid M'₁
      M₂ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₂
      inst✝⁹ : AddCommMonoid M₂
      M₃ : Type u_7
      inst✝⁸ : TopologicalSpace M₃
      inst✝⁷ : AddCommMonoid M₃
      M₄ : Type u_8
      inst✝⁶ : TopologicalSpace M₄
      inst✝⁵ : AddCommMonoid M₄
      inst✝⁴ : Module R₁ M₁
      inst✝³ : Module R₁ M'₁
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      inst✝ : ContinuousAdd M₂
      a✝ b✝ : ContinuousLinearMap σ₁₂ M₁ M₂
      x✝ : M₁
      ⊢ Eq ((HAdd.hAdd a✝ b✝) x✝) ((HAdd.hAdd b✝ a✝) x✝)
    -/
    apply_rules [zero_add, add_assoc, add_zero, neg_add_cancel, add_comm]
    /-
      🎉 no goals
    -/
  add_assoc := by
    intros
    /-
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝¹⁷ : Semiring R₁
      inst✝¹⁶ : Semiring R₂
      inst✝¹⁵ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      M₁ : Type u_4
      inst✝¹⁴ : TopologicalSpace M₁
      inst✝¹³ : AddCommMonoid M₁
      M'₁ : Type u_5
      inst✝¹² : TopologicalSpace M'₁
      inst✝¹¹ : AddCommMonoid M'₁
      M₂ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₂
      inst✝⁹ : AddCommMonoid M₂
      M₃ : Type u_7
      inst✝⁸ : TopologicalSpace M₃
      inst✝⁷ : AddCommMonoid M₃
      M₄ : Type u_8
      inst✝⁶ : TopologicalSpace M₄
      inst✝⁵ : AddCommMonoid M₄
      inst✝⁴ : Module R₁ M₁
      inst✝³ : Module R₁ M'₁
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      inst✝ : ContinuousAdd M₂
      f : ContinuousLinearMap σ₁₂ M₁ M₂
      ⊢ Eq ((fun x1 x2 => HSMul.hSMul x1 x2) 0 f) 0
    -/
    ext
    /-
      case h
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝¹⁷ : Semiring R₁
      inst✝¹⁶ : Semiring R₂
      inst✝¹⁵ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      M₁ : Type u_4
      inst✝¹⁴ : TopologicalSpace M₁
      inst✝¹³ : AddCommMonoid M₁
      M'₁ : Type u_5
      inst✝¹² : TopologicalSpace M'₁
      inst✝¹¹ : AddCommMonoid M'₁
      M₂ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₂
      inst✝⁹ : AddCommMonoid M₂
      M₃ : Type u_7
      inst✝⁸ : TopologicalSpace M₃
      inst✝⁷ : AddCommMonoid M₃
      M₄ : Type u_8
      inst✝⁶ : TopologicalSpace M₄
      inst✝⁵ : AddCommMonoid M₄
      inst✝⁴ : Module R₁ M₁
      inst✝³ : Module R₁ M'₁
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      inst✝ : ContinuousAdd M₂
      f : ContinuousLinearMap σ₁₂ M₁ M₂
      x✝ : M₁
      ⊢ Eq (((fun x1 x2 => HSMul.hSMul x1 x2) 0 f) x✝) (0 x✝)
    -/
    apply_rules [zero_add, add_assoc, add_zero, neg_add_cancel, add_comm]
    /-
      🎉 no goals
    -/
  nsmul := (· • ·)
    /-
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝¹⁷ : Semiring R₁
      inst✝¹⁶ : Semiring R₂
      inst✝¹⁵ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      M₁ : Type u_4
      inst✝¹⁴ : TopologicalSpace M₁
      inst✝¹³ : AddCommMonoid M₁
      M'₁ : Type u_5
      inst✝¹² : TopologicalSpace M'₁
      inst✝¹¹ : AddCommMonoid M'₁
      M₂ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₂
      inst✝⁹ : AddCommMonoid M₂
      M₃ : Type u_7
      inst✝⁸ : TopologicalSpace M₃
      inst✝⁷ : AddCommMonoid M₃
      M₄ : Type u_8
      inst✝⁶ : TopologicalSpace M₄
      inst✝⁵ : AddCommMonoid M₄
      inst✝⁴ : Module R₁ M₁
      inst✝³ : Module R₁ M'₁
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      inst✝ : ContinuousAdd M₂
      n : Nat
      f : ContinuousLinearMap σ₁₂ M₁ M₂
      ⊢ Eq ((fun x1 x2 => HSMul.hSMul x1 x2) (HAdd.hAdd n 1) f) (HAdd.hAdd ((fun x1  …
    -/
  nsmul_zero f := by
    /-
      case h
      R₁ : Type u_1
      R₂ : Type u_2
      R₃ : Type u_3
      inst✝¹⁷ : Semiring R₁
      inst✝¹⁶ : Semiring R₂
      inst✝¹⁵ : Semiring R₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      M₁ : Type u_4
      inst✝¹⁴ : TopologicalSpace M₁
      inst✝¹³ : AddCommMonoid M₁
      M'₁ : Type u_5
      inst✝¹² : TopologicalSpace M'₁
      inst✝¹¹ : AddCommMonoid M'₁
      M₂ : Type u_6
      inst✝¹⁰ : TopologicalSpace M₂
      inst✝⁹ : AddCommMonoid M₂
      M₃ : Type u_7
      inst✝⁸ : TopologicalSpace M₃
      inst✝⁷ : AddCommMonoid M₃
      M₄ : Type u_8
      inst✝⁶ : TopologicalSpace M₄
      inst✝⁵ : AddCommMonoid M₄
      inst✝⁴ : Module R₁ M₁
      inst✝³ : Module R₁ M'₁
      inst✝² : Module R₂ M₂
      inst✝¹ : Module R₃ M₃
      inst✝ : ContinuousAdd M₂
      n : Nat
      f : ContinuousLinearMap σ₁₂ M₁ M₂
      x✝ : M₁
      ⊢ Eq (((fun x1 x2 => HSMul.hSMul x1 x2) (HAdd.hAdd n 1) f) x✝) ((HAdd.hAdd ((f …
    -/
    ext
    /-
      🎉 no goals
    -/
    simp
  nsmul_succ n f := by
    ext
    simp [add_smul]


@[simp, norm_cast]
theorem coe_sum {ι : Type*} (t : Finset ι) (f : ι → M₁ →SL[σ₁₂] M₂) :
    ↑(∑ d ∈ t, f d) = (∑ d ∈ t, f d : M₁ →ₛₗ[σ₁₂] M₂) :=
  map_sum (AddMonoidHom.mk ⟨((↑) : (M₁ →SL[σ₁₂] M₂) → M₁ →ₛₗ[σ₁₂] M₂), rfl⟩ fun _ _ => rfl) _ _


@[simp, norm_cast]
theorem coe_sum' {ι : Type*} (t : Finset ι) (f : ι → M₁ →SL[σ₁₂] M₂) :
                                            /-
                                              R₁ : Type u_1
                                              R₂ : Type u_2
                                              inst✝⁸ : Semiring R₁
                                              inst✝⁷ : Semiring R₂
                                              σ₁₂ : RingHom R₁ R₂
                                              M₁ : Type u_4
                                              inst✝⁶ : TopologicalSpace M₁
                                              inst✝⁵ : AddCommMonoid M₁
                                              M₂ : Type u_6
                                              inst✝⁴ : TopologicalSpace M₂
                                              inst✝³ : AddCommMonoid M₂
                                              inst✝² : Module R₁ M₁
                                              inst✝¹ : Module R₂ M₂
                                              inst✝ : ContinuousAdd M₂
                                              ι : Type u_9
                                              t : Finset ι
                                              f : ι → ContinuousLinearMap σ₁₂ M₁ M₂
                                              ⊢ Eq (⇑(t.sum fun d => f d)) (t.sum fun d => ⇑(f d))
                                            -/
    ⇑(∑ d ∈ t, f d) = ∑ d ∈ t, ⇑(f d) := by simp only [← coe_coe, coe_sum, LinearMap.coeFn_sum]
                                            /-
                                              🎉 no goals
                                            -/


theorem sum_apply {ι : Type*} (t : Finset ι) (f : ι → M₁ →SL[σ₁₂] M₂) (b : M₁) :
                                            /-
                                              R₁ : Type u_1
                                              R₂ : Type u_2
                                              inst✝⁸ : Semiring R₁
                                              inst✝⁷ : Semiring R₂
                                              σ₁₂ : RingHom R₁ R₂
                                              M₁ : Type u_4
                                              inst✝⁶ : TopologicalSpace M₁
                                              inst✝⁵ : AddCommMonoid M₁
                                              M₂ : Type u_6
                                              inst✝⁴ : TopologicalSpace M₂
                                              inst✝³ : AddCommMonoid M₂
                                              inst✝² : Module R₁ M₁
                                              inst✝¹ : Module R₂ M₂
                                              inst✝ : ContinuousAdd M₂
                                              ι : Type u_9
                                              t : Finset ι
                                              f : ι → ContinuousLinearMap σ₁₂ M₁ M₂
                                              b : M₁
                                              ⊢ Eq ((t.sum fun d => f d) b) (t.sum fun d => (f d) b)
                                            -/
    (∑ d ∈ t, f d) b = ∑ d ∈ t, f d b := by simp only [coe_sum', Finset.sum_apply]
                                            /-
                                              🎉 no goals
                                            -/


/-- Composition of bounded linear maps. -/
def comp (g : M₂ →SL[σ₂₃] M₃) (f : M₁ →SL[σ₁₂] M₂) : M₁ →SL[σ₁₃] M₃ :=
  ⟨(g : M₂ →ₛₗ[σ₂₃] M₃).comp (f : M₁ →ₛₗ[σ₁₂] M₂), g.2.comp f.2⟩


@[inherit_doc comp]
infixr:80 " ∘L " =>
  @ContinuousLinearMap.comp _ _ _ _ _ _ (RingHom.id _) (RingHom.id _) (RingHom.id _) _ _ _ _ _ _ _ _
    _ _ _ _ RingHomCompTriple.ids


@[simp, norm_cast]
theorem coe_comp (h : M₂ →SL[σ₂₃] M₃) (f : M₁ →SL[σ₁₂] M₂) :
    (h.comp f : M₁ →ₛₗ[σ₁₃] M₃) = (h : M₂ →ₛₗ[σ₂₃] M₃).comp (f : M₁ →ₛₗ[σ₁₂] M₂) :=
  rfl


@[simp, norm_cast]
theorem coe_comp' (h : M₂ →SL[σ₂₃] M₃) (f : M₁ →SL[σ₁₂] M₂) : ⇑(h.comp f) = h ∘ f :=
  rfl


theorem comp_apply (g : M₂ →SL[σ₂₃] M₃) (f : M₁ →SL[σ₁₂] M₂) (x : M₁) : (g.comp f) x = g (f x) :=
  rfl


@[simp]
theorem comp_id (f : M₁ →SL[σ₁₂] M₂) : f.comp (id R₁ M₁) = f :=
  ext fun _x => rfl


@[simp]
theorem id_comp (f : M₁ →SL[σ₁₂] M₂) : (id R₂ M₂).comp f = f :=
  ext fun _x => rfl


@[simp]
theorem comp_zero (g : M₂ →SL[σ₂₃] M₃) : g.comp (0 : M₁ →SL[σ₁₂] M₂) = 0 := by
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    R₃ : Type u_3
    inst✝¹² : Semiring R₁
    inst✝¹¹ : Semiring R₂
    inst✝¹⁰ : Semiring R₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R₁ R₃
    M₁ : Type u_4
    inst✝⁹ : TopologicalSpace M₁
    inst✝⁸ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝⁷ : TopologicalSpace M₂
    inst✝⁶ : AddCommMonoid M₂
    M₃ : Type u_7
    inst✝⁵ : TopologicalSpace M₃
    inst✝⁴ : AddCommMonoid M₃
    inst✝³ : Module R₁ M₁
    inst✝² : Module R₂ M₂
    inst✝¹ : Module R₃ M₃
    inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    g : ContinuousLinearMap σ₂₃ M₂ M₃
    ⊢ Eq (g.comp 0) 0
  -/
  ext
  /-
    case h
    R₁ : Type u_1
    R₂ : Type u_2
    R₃ : Type u_3
    inst✝¹² : Semiring R₁
    inst✝¹¹ : Semiring R₂
    inst✝¹⁰ : Semiring R₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R₁ R₃
    M₁ : Type u_4
    inst✝⁹ : TopologicalSpace M₁
    inst✝⁸ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝⁷ : TopologicalSpace M₂
    inst✝⁶ : AddCommMonoid M₂
    M₃ : Type u_7
    inst✝⁵ : TopologicalSpace M₃
    inst✝⁴ : AddCommMonoid M₃
    inst✝³ : Module R₁ M₁
    inst✝² : Module R₂ M₂
    inst✝¹ : Module R₃ M₃
    inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    g : ContinuousLinearMap σ₂₃ M₂ M₃
    x✝ : M₁
    ⊢ Eq ((g.comp 0) x✝) (0 x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_comp (f : M₁ →SL[σ₁₂] M₂) : (0 : M₂ →SL[σ₂₃] M₃).comp f = 0 := by
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    R₃ : Type u_3
    inst✝¹² : Semiring R₁
    inst✝¹¹ : Semiring R₂
    inst✝¹⁰ : Semiring R₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R₁ R₃
    M₁ : Type u_4
    inst✝⁹ : TopologicalSpace M₁
    inst✝⁸ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝⁷ : TopologicalSpace M₂
    inst✝⁶ : AddCommMonoid M₂
    M₃ : Type u_7
    inst✝⁵ : TopologicalSpace M₃
    inst✝⁴ : AddCommMonoid M₃
    inst✝³ : Module R₁ M₁
    inst✝² : Module R₂ M₂
    inst✝¹ : Module R₃ M₃
    inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    f : ContinuousLinearMap σ₁₂ M₁ M₂
    ⊢ Eq (ContinuousLinearMap.comp 0 f) 0
  -/
  ext
  /-
    case h
    R₁ : Type u_1
    R₂ : Type u_2
    R₃ : Type u_3
    inst✝¹² : Semiring R₁
    inst✝¹¹ : Semiring R₂
    inst✝¹⁰ : Semiring R₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R₁ R₃
    M₁ : Type u_4
    inst✝⁹ : TopologicalSpace M₁
    inst✝⁸ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝⁷ : TopologicalSpace M₂
    inst✝⁶ : AddCommMonoid M₂
    M₃ : Type u_7
    inst✝⁵ : TopologicalSpace M₃
    inst✝⁴ : AddCommMonoid M₃
    inst✝³ : Module R₁ M₁
    inst✝² : Module R₂ M₂
    inst✝¹ : Module R₃ M₃
    inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    f : ContinuousLinearMap σ₁₂ M₁ M₂
    x✝ : M₁
    ⊢ Eq ((ContinuousLinearMap.comp 0 f) x✝) (0 x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_add [ContinuousAdd M₂] [ContinuousAdd M₃] (g : M₂ →SL[σ₂₃] M₃)
    (f₁ f₂ : M₁ →SL[σ₁₂] M₂) : g.comp (f₁ + f₂) = g.comp f₁ + g.comp f₂ := by
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    R₃ : Type u_3
    inst✝¹⁴ : Semiring R₁
    inst✝¹³ : Semiring R₂
    inst✝¹² : Semiring R₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R₁ R₃
    M₁ : Type u_4
    inst✝¹¹ : TopologicalSpace M₁
    inst✝¹⁰ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝⁹ : TopologicalSpace M₂
    inst✝⁸ : AddCommMonoid M₂
    M₃ : Type u_7
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : AddCommMonoid M₃
    inst✝⁵ : Module R₁ M₁
    inst✝⁴ : Module R₂ M₂
    inst✝³ : Module R₃ M₃
    inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    inst✝¹ : ContinuousAdd M₂
    inst✝ : ContinuousAdd M₃
    g : ContinuousLinearMap σ₂₃ M₂ M₃
    f₁ f₂ : ContinuousLinearMap σ₁₂ M₁ M₂
    ⊢ Eq (g.comp (HAdd.hAdd f₁ f₂)) (HAdd.hAdd (g.comp f₁) (g.comp f₂))
  -/
  ext
  /-
    case h
    R₁ : Type u_1
    R₂ : Type u_2
    R₃ : Type u_3
    inst✝¹⁴ : Semiring R₁
    inst✝¹³ : Semiring R₂
    inst✝¹² : Semiring R₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R₁ R₃
    M₁ : Type u_4
    inst✝¹¹ : TopologicalSpace M₁
    inst✝¹⁰ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝⁹ : TopologicalSpace M₂
    inst✝⁸ : AddCommMonoid M₂
    M₃ : Type u_7
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : AddCommMonoid M₃
    inst✝⁵ : Module R₁ M₁
    inst✝⁴ : Module R₂ M₂
    inst✝³ : Module R₃ M₃
    inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    inst✝¹ : ContinuousAdd M₂
    inst✝ : ContinuousAdd M₃
    g : ContinuousLinearMap σ₂₃ M₂ M₃
    f₁ f₂ : ContinuousLinearMap σ₁₂ M₁ M₂
    x✝ : M₁
    ⊢ Eq ((g.comp (HAdd.hAdd f₁ f₂)) x✝) ((HAdd.hAdd (g.comp f₁) (g.comp f₂)) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem add_comp [ContinuousAdd M₃] (g₁ g₂ : M₂ →SL[σ₂₃] M₃) (f : M₁ →SL[σ₁₂] M₂) :
    (g₁ + g₂).comp f = g₁.comp f + g₂.comp f := by
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    R₃ : Type u_3
    inst✝¹³ : Semiring R₁
    inst✝¹² : Semiring R₂
    inst✝¹¹ : Semiring R₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R₁ R₃
    M₁ : Type u_4
    inst✝¹⁰ : TopologicalSpace M₁
    inst✝⁹ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : AddCommMonoid M₂
    M₃ : Type u_7
    inst✝⁶ : TopologicalSpace M₃
    inst✝⁵ : AddCommMonoid M₃
    inst✝⁴ : Module R₁ M₁
    inst✝³ : Module R₂ M₂
    inst✝² : Module R₃ M₃
    inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    inst✝ : ContinuousAdd M₃
    g₁ g₂ : ContinuousLinearMap σ₂₃ M₂ M₃
    f : ContinuousLinearMap σ₁₂ M₁ M₂
    ⊢ Eq ((HAdd.hAdd g₁ g₂).comp f) (HAdd.hAdd (g₁.comp f) (g₂.comp f))
  -/
  ext
  /-
    case h
    R₁ : Type u_1
    R₂ : Type u_2
    R₃ : Type u_3
    inst✝¹³ : Semiring R₁
    inst✝¹² : Semiring R₂
    inst✝¹¹ : Semiring R₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R₁ R₃
    M₁ : Type u_4
    inst✝¹⁰ : TopologicalSpace M₁
    inst✝⁹ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : AddCommMonoid M₂
    M₃ : Type u_7
    inst✝⁶ : TopologicalSpace M₃
    inst✝⁵ : AddCommMonoid M₃
    inst✝⁴ : Module R₁ M₁
    inst✝³ : Module R₂ M₂
    inst✝² : Module R₃ M₃
    inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    inst✝ : ContinuousAdd M₃
    g₁ g₂ : ContinuousLinearMap σ₂₃ M₂ M₃
    f : ContinuousLinearMap σ₁₂ M₁ M₂
    x✝ : M₁
    ⊢ Eq (((HAdd.hAdd g₁ g₂).comp f) x✝) ((HAdd.hAdd (g₁.comp f) (g₂.comp f)) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem comp_finset_sum {ι : Type*} {s : Finset ι}
    [ContinuousAdd M₂] [ContinuousAdd M₃] (g : M₂ →SL[σ₂₃] M₃)
    (f : ι → M₁ →SL[σ₁₂] M₂) : g.comp (∑ i ∈ s, f i) = ∑ i ∈ s, g.comp (f i) := by
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    R₃ : Type u_3
    inst✝¹⁴ : Semiring R₁
    inst✝¹³ : Semiring R₂
    inst✝¹² : Semiring R₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R₁ R₃
    M₁ : Type u_4
    inst✝¹¹ : TopologicalSpace M₁
    inst✝¹⁰ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝⁹ : TopologicalSpace M₂
    inst✝⁸ : AddCommMonoid M₂
    M₃ : Type u_7
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : AddCommMonoid M₃
    inst✝⁵ : Module R₁ M₁
    inst✝⁴ : Module R₂ M₂
    inst✝³ : Module R₃ M₃
    inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    ι : Type u_9
    s : Finset ι
    inst✝¹ : ContinuousAdd M₂
    inst✝ : ContinuousAdd M₃
    g : ContinuousLinearMap σ₂₃ M₂ M₃
    f : ι → ContinuousLinearMap σ₁₂ M₁ M₂
    ⊢ Eq (g.comp (s.sum fun i => f i)) (s.sum fun i => g.comp (f i))
  -/
  ext
  /-
    case h
    R₁ : Type u_1
    R₂ : Type u_2
    R₃ : Type u_3
    inst✝¹⁴ : Semiring R₁
    inst✝¹³ : Semiring R₂
    inst✝¹² : Semiring R₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R₁ R₃
    M₁ : Type u_4
    inst✝¹¹ : TopologicalSpace M₁
    inst✝¹⁰ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝⁹ : TopologicalSpace M₂
    inst✝⁸ : AddCommMonoid M₂
    M₃ : Type u_7
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : AddCommMonoid M₃
    inst✝⁵ : Module R₁ M₁
    inst✝⁴ : Module R₂ M₂
    inst✝³ : Module R₃ M₃
    inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    ι : Type u_9
    s : Finset ι
    inst✝¹ : ContinuousAdd M₂
    inst✝ : ContinuousAdd M₃
    g : ContinuousLinearMap σ₂₃ M₂ M₃
    f : ι → ContinuousLinearMap σ₁₂ M₁ M₂
    x✝ : M₁
    ⊢ Eq ((g.comp (s.sum fun i => f i)) x✝) ((s.sum fun i => g.comp (f i)) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem finset_sum_comp {ι : Type*} {s : Finset ι}
    [ContinuousAdd M₃] (g : ι → M₂ →SL[σ₂₃] M₃)
    (f : M₁ →SL[σ₁₂] M₂) : (∑ i ∈ s, g i).comp f = ∑ i ∈ s, (g i).comp f := by
  /-
    R₁ : Type u_1
    R₂ : Type u_2
    R₃ : Type u_3
    inst✝¹³ : Semiring R₁
    inst✝¹² : Semiring R₂
    inst✝¹¹ : Semiring R₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R₁ R₃
    M₁ : Type u_4
    inst✝¹⁰ : TopologicalSpace M₁
    inst✝⁹ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : AddCommMonoid M₂
    M₃ : Type u_7
    inst✝⁶ : TopologicalSpace M₃
    inst✝⁵ : AddCommMonoid M₃
    inst✝⁴ : Module R₁ M₁
    inst✝³ : Module R₂ M₂
    inst✝² : Module R₃ M₃
    inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    ι : Type u_9
    s : Finset ι
    inst✝ : ContinuousAdd M₃
    g : ι → ContinuousLinearMap σ₂₃ M₂ M₃
    f : ContinuousLinearMap σ₁₂ M₁ M₂
    ⊢ Eq ((s.sum fun i => g i).comp f) (s.sum fun i => (g i).comp f)
  -/
  ext
  /-
    case h
    R₁ : Type u_1
    R₂ : Type u_2
    R₃ : Type u_3
    inst✝¹³ : Semiring R₁
    inst✝¹² : Semiring R₂
    inst✝¹¹ : Semiring R₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R₁ R₃
    M₁ : Type u_4
    inst✝¹⁰ : TopologicalSpace M₁
    inst✝⁹ : AddCommMonoid M₁
    M₂ : Type u_6
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : AddCommMonoid M₂
    M₃ : Type u_7
    inst✝⁶ : TopologicalSpace M₃
    inst✝⁵ : AddCommMonoid M₃
    inst✝⁴ : Module R₁ M₁
    inst✝³ : Module R₂ M₂
    inst✝² : Module R₃ M₃
    inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    ι : Type u_9
    s : Finset ι
    inst✝ : ContinuousAdd M₃
    g : ι → ContinuousLinearMap σ₂₃ M₂ M₃
    f : ContinuousLinearMap σ₁₂ M₁ M₂
    x✝ : M₁
    ⊢ Eq (((s.sum fun i => g i).comp f) x✝) ((s.sum fun i => (g i).comp f) x✝)
  -/
  simp only [coe_comp', coe_sum', Function.comp_apply, Finset.sum_apply]
  /-
    🎉 no goals
  -/


theorem comp_assoc {R₄ : Type*} [Semiring R₄] [Module R₄ M₄] {σ₁₄ : R₁ →+* R₄} {σ₂₄ : R₂ →+* R₄}
    {σ₃₄ : R₃ →+* R₄} [RingHomCompTriple σ₁₃ σ₃₄ σ₁₄] [RingHomCompTriple σ₂₃ σ₃₄ σ₂₄]
    [RingHomCompTriple σ₁₂ σ₂₄ σ₁₄] (h : M₃ →SL[σ₃₄] M₄) (g : M₂ →SL[σ₂₃] M₃) (f : M₁ →SL[σ₁₂] M₂) :
    (h.comp g).comp f = h.comp (g.comp f) :=
  rfl


instance instMul : Mul (M₁ →L[R₁] M₁) :=
  ⟨comp⟩


theorem mul_def (f g : M₁ →L[R₁] M₁) : f * g = f.comp g :=
  rfl


@[simp]
theorem coe_mul (f g : M₁ →L[R₁] M₁) : ⇑(f * g) = f ∘ g :=
  rfl


theorem mul_apply (f g : M₁ →L[R₁] M₁) (x : M₁) : (f * g) x = f (g x) :=
  rfl


instance monoidWithZero : MonoidWithZero (M₁ →L[R₁] M₁) where
  mul_zero f := ext fun _ => map_zero f
  zero_mul _ := ext fun _ => rfl
  mul_one _ := ext fun _ => rfl
  one_mul _ := ext fun _ => rfl
  mul_assoc _ _ _ := ext fun _ => rfl


theorem coe_pow (f : M₁ →L[R₁] M₁) (n : ℕ) : ⇑(f ^ n) = f^[n] :=
  hom_coe_pow _ rfl (fun _ _ ↦ rfl) _ _


instance instNatCast [ContinuousAdd M₁] : NatCast (M₁ →L[R₁] M₁) where
  natCast n := n • (1 : M₁ →L[R₁] M₁)


instance semiring [ContinuousAdd M₁] : Semiring (M₁ →L[R₁] M₁) where
  __ := ContinuousLinearMap.monoidWithZero
  __ := ContinuousLinearMap.addCommMonoid
  left_distrib f g h := ext fun x => map_add f (g x) (h x)
  right_distrib _ _ _ := ext fun _ => LinearMap.add_apply _ _ _
  toNatCast := instNatCast
  natCast_zero := zero_smul ℕ (1 : M₁ →L[R₁] M₁)
  natCast_succ n := AddMonoid.nsmul_succ n (1 : M₁ →L[R₁] M₁)


/-- `ContinuousLinearMap.toLinearMap` as a `RingHom`. -/
@[simps]
def toLinearMapRingHom [ContinuousAdd M₁] : (M₁ →L[R₁] M₁) →+* M₁ →ₗ[R₁] M₁ where
  toFun := toLinearMap
  map_zero' := rfl
  map_one' := rfl
  map_add' _ _ := rfl
  map_mul' _ _ := rfl


@[simp]
theorem natCast_apply [ContinuousAdd M₁] (n : ℕ) (m : M₁) : (↑n : M₁ →L[R₁] M₁) m = n • m :=
  rfl


@[simp]
theorem ofNat_apply [ContinuousAdd M₁] (n : ℕ) [n.AtLeastTwo] (m : M₁) :
    ((no_index (OfNat.ofNat n) : M₁ →L[R₁] M₁)) m = OfNat.ofNat n • m :=
  rfl


/-- The tautological action by `M₁ →L[R₁] M₁` on `M`.

This generalizes `Function.End.applyMulAction`. -/
instance applyModule : Module (M₁ →L[R₁] M₁) M₁ :=
  Module.compHom _ toLinearMapRingHom


@[simp]
protected theorem smul_def (f : M₁ →L[R₁] M₁) (a : M₁) : f • a = f a :=
  rfl


/-- `ContinuousLinearMap.applyModule` is faithful. -/
instance applyFaithfulSMul : FaithfulSMul (M₁ →L[R₁] M₁) M₁ :=
  ⟨fun {_ _} => ContinuousLinearMap.ext⟩


instance applySMulCommClass : SMulCommClass R₁ (M₁ →L[R₁] M₁) M₁ where
  smul_comm r e m := (e.map_smul r m).symm


instance applySMulCommClass' : SMulCommClass (M₁ →L[R₁] M₁) R₁ M₁ where
  smul_comm := ContinuousLinearMap.map_smul


instance continuousConstSMul_apply : ContinuousConstSMul (M₁ →L[R₁] M₁) M₁ :=
  ⟨ContinuousLinearMap.continuous⟩


theorem isClosed_ker [T1Space M₂] [FunLike F M₁ M₂] [ContinuousSemilinearMapClass F σ₁₂ M₁ M₂]
    (f : F) :
    IsClosed (ker f : Set M₁) :=
  continuous_iff_isClosed.1 (map_continuous f) _ isClosed_singleton


theorem isComplete_ker {M' : Type*} [UniformSpace M'] [CompleteSpace M'] [AddCommMonoid M']
    [Module R₁ M'] [T1Space M₂] [FunLike F M' M₂] [ContinuousSemilinearMapClass F σ₁₂ M' M₂]
    (f : F) :
    IsComplete (ker f : Set M') :=
  (isClosed_ker f).isComplete


instance completeSpace_ker {M' : Type*} [UniformSpace M'] [CompleteSpace M']
    [AddCommMonoid M'] [Module R₁ M'] [T1Space M₂]
    [FunLike F M' M₂] [ContinuousSemilinearMapClass F σ₁₂ M' M₂]
    (f : F) : CompleteSpace (ker f) :=
  (isComplete_ker f).completeSpace_coe


instance completeSpace_eqLocus {M' : Type*} [UniformSpace M'] [CompleteSpace M']
    [AddCommMonoid M'] [Module R₁ M'] [T2Space M₂]
    [FunLike F M' M₂] [ContinuousSemilinearMapClass F σ₁₂ M' M₂]
    (f g : F) : CompleteSpace (LinearMap.eqLocus f g) :=
  IsClosed.completeSpace_coe <| isClosed_eq (map_continuous f) (map_continuous g)


/-- Restrict codomain of a continuous linear map. -/
def codRestrict (f : M₁ →SL[σ₁₂] M₂) (p : Submodule R₂ M₂) (h : ∀ x, f x ∈ p) :
    M₁ →SL[σ₁₂] p where
  cont := f.continuous.subtype_mk _
  toLinearMap := (f : M₁ →ₛₗ[σ₁₂] M₂).codRestrict p h


@[norm_cast]
theorem coe_codRestrict (f : M₁ →SL[σ₁₂] M₂) (p : Submodule R₂ M₂) (h : ∀ x, f x ∈ p) :
    (f.codRestrict p h : M₁ →ₛₗ[σ₁₂] p) = (f : M₁ →ₛₗ[σ₁₂] M₂).codRestrict p h :=
  rfl


@[simp]
theorem coe_codRestrict_apply (f : M₁ →SL[σ₁₂] M₂) (p : Submodule R₂ M₂) (h : ∀ x, f x ∈ p) (x) :
    (f.codRestrict p h x : M₂) = f x :=
  rfl


@[simp]
theorem ker_codRestrict (f : M₁ →SL[σ₁₂] M₂) (p : Submodule R₂ M₂) (h : ∀ x, f x ∈ p) :
    ker (f.codRestrict p h) = ker f :=
  (f : M₁ →ₛₗ[σ₁₂] M₂).ker_codRestrict p h


/-- Restrict the codomain of a continuous linear map `f` to `f.range`. -/
abbrev rangeRestrict [RingHomSurjective σ₁₂] (f : M₁ →SL[σ₁₂] M₂) :=
  f.codRestrict (LinearMap.range f) (LinearMap.mem_range_self f)


@[simp]
theorem coe_rangeRestrict [RingHomSurjective σ₁₂] (f : M₁ →SL[σ₁₂] M₂) :
    (f.rangeRestrict : M₁ →ₛₗ[σ₁₂] LinearMap.range f) = (f : M₁ →ₛₗ[σ₁₂] M₂).rangeRestrict :=
  rfl


/-- `Submodule.subtype` as a `ContinuousLinearMap`. -/
def _root_.Submodule.subtypeL (p : Submodule R₁ M₁) : p →L[R₁] M₁ where
  cont := continuous_subtype_val
  toLinearMap := p.subtype


@[simp, norm_cast]
theorem _root_.Submodule.coe_subtypeL (p : Submodule R₁ M₁) :
    (p.subtypeL : p →ₗ[R₁] M₁) = p.subtype :=
  rfl


@[simp]
theorem _root_.Submodule.coe_subtypeL' (p : Submodule R₁ M₁) : ⇑p.subtypeL = p.subtype :=
  rfl


@[simp] -- @[norm_cast] -- Porting note: A theorem with this can't have a rhs starting with `↑`.
theorem _root_.Submodule.subtypeL_apply (p : Submodule R₁ M₁) (x : p) : p.subtypeL x = x :=
  rfl


@[simp]
theorem _root_.Submodule.range_subtypeL (p : Submodule R₁ M₁) : range p.subtypeL = p :=
  Submodule.range_subtype _


@[simp]
theorem _root_.Submodule.ker_subtypeL (p : Submodule R₁ M₁) : ker p.subtypeL = ⊥ :=
  Submodule.ker_subtype _


/-- The linear map `fun x => c x • f`.  Associates to a scalar-valued linear map and an element of
`M₂` the `M₂`-valued linear map obtained by multiplying the two (a.k.a. tensoring by `M₂`).
See also `ContinuousLinearMap.smulRightₗ` and `ContinuousLinearMap.smulRightL`. -/
def smulRight (c : M₁ →L[R] S) (f : M₂) : M₁ →L[R] M₂ :=
  { c.toLinearMap.smulRight f with cont := c.2.smul continuous_const }


@[simp]
theorem smulRight_apply {c : M₁ →L[R] S} {f : M₂} {x : M₁} :
    (smulRight c f : M₁ → M₂) x = c x • f :=
  rfl


@[simp]
theorem smulRight_one_one (c : R₁ →L[R₁] M₂) : smulRight (1 : R₁ →L[R₁] R₁) (c 1) = c := by
  /-
    R₁ : Type u_1
    inst✝⁵ : Semiring R₁
    M₂ : Type u_6
    inst✝⁴ : TopologicalSpace M₂
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R₁ M₂
    inst✝¹ : TopologicalSpace R₁
    inst✝ : ContinuousSMul R₁ M₂
    c : ContinuousLinearMap (RingHom.id R₁) R₁ M₂
    ⊢ Eq (ContinuousLinearMap.smulRight 1 (c 1)) c
  -/
  ext
  /-
    case h
    R₁ : Type u_1
    inst✝⁵ : Semiring R₁
    M₂ : Type u_6
    inst✝⁴ : TopologicalSpace M₂
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R₁ M₂
    inst✝¹ : TopologicalSpace R₁
    inst✝ : ContinuousSMul R₁ M₂
    c : ContinuousLinearMap (RingHom.id R₁) R₁ M₂
    ⊢ Eq ((ContinuousLinearMap.smulRight 1 (c 1)) 1) (c 1)
  -/
  simp [← ContinuousLinearMap.map_smul_of_tower]
  /-
    🎉 no goals
  -/


@[simp]
theorem smulRight_one_eq_iff {f f' : M₂} :
    smulRight (1 : R₁ →L[R₁] R₁) f = smulRight (1 : R₁ →L[R₁] R₁) f' ↔ f = f' := by
  /-
    R₁ : Type u_1
    inst✝⁵ : Semiring R₁
    M₂ : Type u_6
    inst✝⁴ : TopologicalSpace M₂
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R₁ M₂
    inst✝¹ : TopologicalSpace R₁
    inst✝ : ContinuousSMul R₁ M₂
    f f' : M₂
    ⊢ Iff (Eq (ContinuousLinearMap.smulRight 1 f) (ContinuousLinearMap.smulRight 1 …
  -/
  simp only [ContinuousLinearMap.ext_ring_iff, smulRight_apply, one_apply, one_smul]
  /-
    🎉 no goals
  -/


theorem smulRight_comp [ContinuousMul R₁] {x : M₂} {c : R₁} :
    (smulRight (1 : R₁ →L[R₁] R₁) x).comp (smulRight (1 : R₁ →L[R₁] R₁) c) =
      smulRight (1 : R₁ →L[R₁] R₁) (c • x) := by
  /-
    R₁ : Type u_1
    inst✝⁶ : Semiring R₁
    M₂ : Type u_6
    inst✝⁵ : TopologicalSpace M₂
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R₁ M₂
    inst✝² : TopologicalSpace R₁
    inst✝¹ : ContinuousSMul R₁ M₂
    inst✝ : ContinuousMul R₁
    x : M₂
    c : R₁
    ⊢ Eq ((ContinuousLinearMap.smulRight 1 x).comp (ContinuousLinearMap.smulRight  …
  -/
  ext
  /-
    case h
    R₁ : Type u_1
    inst✝⁶ : Semiring R₁
    M₂ : Type u_6
    inst✝⁵ : TopologicalSpace M₂
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R₁ M₂
    inst✝² : TopologicalSpace R₁
    inst✝¹ : ContinuousSMul R₁ M₂
    inst✝ : ContinuousMul R₁
    x : M₂
    c : R₁
    ⊢ Eq (((ContinuousLinearMap.smulRight 1 x).comp (ContinuousLinearMap.smulRight …
  -/
  simp [mul_smul]
  /-
    🎉 no goals
  -/


/-- Given an element `x` of a topological space `M` over a semiring `R`, the natural continuous
linear map from `R` to `M` by taking multiples of `x`. -/
def toSpanSingleton (x : M₁) : R₁ →L[R₁] M₁ where
  toLinearMap := LinearMap.toSpanSingleton R₁ M₁ x
  cont := continuous_id.smul continuous_const


theorem toSpanSingleton_apply (x : M₁) (r : R₁) : toSpanSingleton R₁ x r = r • x :=
  rfl


theorem toSpanSingleton_add [ContinuousAdd M₁] (x y : M₁) :
    toSpanSingleton R₁ (x + y) = toSpanSingleton R₁ x + toSpanSingleton R₁ y := by
  /-
    R₁ : Type u_1
    inst✝⁶ : Semiring R₁
    M₁ : Type u_4
    inst✝⁵ : TopologicalSpace M₁
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : Module R₁ M₁
    inst✝² : TopologicalSpace R₁
    inst✝¹ : ContinuousSMul R₁ M₁
    inst✝ : ContinuousAdd M₁
    x y : M₁
    ⊢ Eq (ContinuousLinearMap.toSpanSingleton R₁ (HAdd.hAdd x y)) (HAdd.hAdd (Cont …
  -/
  ext1; simp [toSpanSingleton_apply]
        /-
          🎉 no goals
        -/


theorem toSpanSingleton_smul' {α} [Monoid α] [DistribMulAction α M₁] [ContinuousConstSMul α M₁]
    [SMulCommClass R₁ α M₁] (c : α) (x : M₁) :
    toSpanSingleton R₁ (c • x) = c • toSpanSingleton R₁ x := by
  /-
    R₁ : Type u_1
    inst✝⁹ : Semiring R₁
    M₁ : Type u_4
    inst✝⁸ : TopologicalSpace M₁
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : Module R₁ M₁
    inst✝⁵ : TopologicalSpace R₁
    inst✝⁴ : ContinuousSMul R₁ M₁
    α : Type u_10
    inst✝³ : Monoid α
    inst✝² : DistribMulAction α M₁
    inst✝¹ : ContinuousConstSMul α M₁
    inst✝ : SMulCommClass R₁ α M₁
    c : α
    x : M₁
    ⊢ Eq (ContinuousLinearMap.toSpanSingleton R₁ (HSMul.hSMul c x)) (HSMul.hSMul c …
  -/
  ext1; rw [toSpanSingleton_apply, smul_apply, toSpanSingleton_apply, smul_comm]
        /-
          🎉 no goals
        -/


/-- A special case of `to_span_singleton_smul'` for when `R` is commutative. -/
theorem toSpanSingleton_smul (R) {M₁} [CommSemiring R] [AddCommMonoid M₁] [Module R M₁]
    [TopologicalSpace R] [TopologicalSpace M₁] [ContinuousSMul R M₁] (c : R) (x : M₁) :
    toSpanSingleton R (c • x) = c • toSpanSingleton R x :=
  toSpanSingleton_smul' R c x


protected theorem map_neg (f : M →SL[σ₁₂] M₂) (x : M) : f (-x) = -f x := by
  /-
    R : Type u_1
    inst✝⁷ : Ring R
    R₂ : Type u_2
    inst✝⁶ : Ring R₂
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : AddCommGroup M
    M₂ : Type u_5
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M
    inst✝ : Module R₂ M₂
    σ₁₂ : RingHom R R₂
    f : ContinuousLinearMap σ₁₂ M M₂
    x : M
    ⊢ Eq (f (Neg.neg x)) (Neg.neg (f x))
  -/
  exact map_neg f x
  /-
    🎉 no goals
  -/


protected theorem map_sub (f : M →SL[σ₁₂] M₂) (x y : M) : f (x - y) = f x - f y := by
  /-
    R : Type u_1
    inst✝⁷ : Ring R
    R₂ : Type u_2
    inst✝⁶ : Ring R₂
    M : Type u_4
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : AddCommGroup M
    M₂ : Type u_5
    inst✝³ : TopologicalSpace M₂
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M
    inst✝ : Module R₂ M₂
    σ₁₂ : RingHom R R₂
    f : ContinuousLinearMap σ₁₂ M M₂
    x y : M
    ⊢ Eq (f (HSub.hSub x y)) (HSub.hSub (f x) (f y))
  -/
  exact map_sub f x y
  /-
    🎉 no goals
  -/


@[simp]
theorem sub_apply' (f g : M →SL[σ₁₂] M₂) (x : M) : ((f : M →ₛₗ[σ₁₂] M₂) - g) x = f x - g x :=
  rfl


instance neg : Neg (M →SL[σ₁₂] M₂) :=
  ⟨fun f => ⟨-f, f.2.neg⟩⟩


@[simp]
theorem neg_apply (f : M →SL[σ₁₂] M₂) (x : M) : (-f) x = -f x :=
  rfl


@[simp, norm_cast]
theorem coe_neg (f : M →SL[σ₁₂] M₂) : (↑(-f) : M →ₛₗ[σ₁₂] M₂) = -f :=
  rfl


@[norm_cast]
theorem coe_neg' (f : M →SL[σ₁₂] M₂) : ⇑(-f) = -f :=
  rfl


instance sub : Sub (M →SL[σ₁₂] M₂) :=
  ⟨fun f g => ⟨f - g, f.2.sub g.2⟩⟩


instance addCommGroup : AddCommGroup (M →SL[σ₁₂] M₂) where
  __ := ContinuousLinearMap.addCommMonoid
  neg := (-·)
  sub := (· - ·)
                           /-
                             R : Type u_1
                             inst✝¹⁴ : Ring R
                             R₂ : Type u_2
                             inst✝¹³ : Ring R₂
                             R₃ : Type u_3
                             inst✝¹² : Ring R₃
                             M : Type u_4
                             inst✝¹¹ : TopologicalSpace M
                             inst✝¹⁰ : AddCommGroup M
                             M₂ : Type u_5
                             inst✝⁹ : TopologicalSpace M₂
                             inst✝⁸ : AddCommGroup M₂
                             M₃ : Type u_6
                             inst✝⁷ : TopologicalSpace M₃
                             inst✝⁶ : AddCommGroup M₃
                             M₄ : Type u_7
                             inst✝⁵ : TopologicalSpace M₄
                             inst✝⁴ : AddCommGroup M₄
                             inst✝³ : Module R M
                             inst✝² : Module R₂ M₂
                             inst✝¹ : Module R₃ M₃
                             σ₁₂ : RingHom R R₂
                             σ₂₃ : RingHom R₂ R₃
                             σ₁₃ : RingHom R R₃
                             inst✝ : TopologicalAddGroup M₂
                             x✝¹ x✝ : ContinuousLinearMap σ₁₂ M M₂
                             ⊢ Eq (HSub.hSub x✝¹ x✝) (HAdd.hAdd x✝¹ (Neg.neg x✝))
                           -/
  sub_eq_add_neg _ _ := by ext; apply sub_eq_add_neg
                                /-
                                  🎉 no goals
                                -/
  nsmul := (· • ·)
  zsmul := (· • ·)
                      /-
                        R : Type u_1
                        inst✝¹⁴ : Ring R
                        R₂ : Type u_2
                        inst✝¹³ : Ring R₂
                        R₃ : Type u_3
                        inst✝¹² : Ring R₃
                        M : Type u_4
                        inst✝¹¹ : TopologicalSpace M
                        inst✝¹⁰ : AddCommGroup M
                        M₂ : Type u_5
                        inst✝⁹ : TopologicalSpace M₂
                        inst✝⁸ : AddCommGroup M₂
                        M₃ : Type u_6
                        inst✝⁷ : TopologicalSpace M₃
                        inst✝⁶ : AddCommGroup M₃
                        M₄ : Type u_7
                        inst✝⁵ : TopologicalSpace M₄
                        inst✝⁴ : AddCommGroup M₄
                        inst✝³ : Module R M
                        inst✝² : Module R₂ M₂
                        inst✝¹ : Module R₃ M₃
                        σ₁₂ : RingHom R R₂
                        σ₂₃ : RingHom R₂ R₃
                        σ₁₃ : RingHom R R₃
                        inst✝ : TopologicalAddGroup M₂
                        f : ContinuousLinearMap σ₁₂ M M₂
                        ⊢ Eq ((fun x1 x2 => HSMul.hSMul x1 x2) 0 f) 0
                      -/
  zsmul_zero' f := by ext; simp
                           /-
                             🎉 no goals
                           -/
                        /-
                          R : Type u_1
                          inst✝¹⁴ : Ring R
                          R₂ : Type u_2
                          inst✝¹³ : Ring R₂
                          R₃ : Type u_3
                          inst✝¹² : Ring R₃
                          M : Type u_4
                          inst✝¹¹ : TopologicalSpace M
                          inst✝¹⁰ : AddCommGroup M
                          M₂ : Type u_5
                          inst✝⁹ : TopologicalSpace M₂
                          inst✝⁸ : AddCommGroup M₂
                          M₃ : Type u_6
                          inst✝⁷ : TopologicalSpace M₃
                          inst✝⁶ : AddCommGroup M₃
                          M₄ : Type u_7
                          inst✝⁵ : TopologicalSpace M₄
                          inst✝⁴ : AddCommGroup M₄
                          inst✝³ : Module R M
                          inst✝² : Module R₂ M₂
                          inst✝¹ : Module R₃ M₃
                          σ₁₂ : RingHom R R₂
                          σ₂₃ : RingHom R₂ R₃
                          σ₁₃ : RingHom R R₃
                          inst✝ : TopologicalAddGroup M₂
                          n : Nat
                          f : ContinuousLinearMap σ₁₂ M M₂
                          ⊢ Eq ((fun x1 x2 => HSMul.hSMul x1 x2) (↑n.succ) f) (HAdd.hAdd ((fun x1 x2 =>  …
                        -/
  zsmul_succ' n f := by ext; simp [add_smul, add_comm]
                             /-
                               🎉 no goals
                             -/
                       /-
                         R : Type u_1
                         inst✝¹⁴ : Ring R
                         R₂ : Type u_2
                         inst✝¹³ : Ring R₂
                         R₃ : Type u_3
                         inst✝¹² : Ring R₃
                         M : Type u_4
                         inst✝¹¹ : TopologicalSpace M
                         inst✝¹⁰ : AddCommGroup M
                         M₂ : Type u_5
                         inst✝⁹ : TopologicalSpace M₂
                         inst✝⁸ : AddCommGroup M₂
                         M₃ : Type u_6
                         inst✝⁷ : TopologicalSpace M₃
                         inst✝⁶ : AddCommGroup M₃
                         M₄ : Type u_7
                         inst✝⁵ : TopologicalSpace M₄
                         inst✝⁴ : AddCommGroup M₄
                         inst✝³ : Module R M
                         inst✝² : Module R₂ M₂
                         inst✝¹ : Module R₃ M₃
                         σ₁₂ : RingHom R R₂
                         σ₂₃ : RingHom R₂ R₃
                         σ₁₃ : RingHom R R₃
                         inst✝ : TopologicalAddGroup M₂
                         n : Nat
                         f : ContinuousLinearMap σ₁₂ M M₂
                         ⊢ Eq ((fun x1 x2 => HSMul.hSMul x1 x2) (Int.negSucc n) f) (Neg.neg ((fun x1 x2 …
                       -/
  zsmul_neg' n f := by ext; simp [add_smul]
                            /-
                              🎉 no goals
                            -/
                         /-
                           R : Type u_1
                           inst✝¹⁴ : Ring R
                           R₂ : Type u_2
                           inst✝¹³ : Ring R₂
                           R₃ : Type u_3
                           inst✝¹² : Ring R₃
                           M : Type u_4
                           inst✝¹¹ : TopologicalSpace M
                           inst✝¹⁰ : AddCommGroup M
                           M₂ : Type u_5
                           inst✝⁹ : TopologicalSpace M₂
                           inst✝⁸ : AddCommGroup M₂
                           M₃ : Type u_6
                           inst✝⁷ : TopologicalSpace M₃
                           inst✝⁶ : AddCommGroup M₃
                           M₄ : Type u_7
                           inst✝⁵ : TopologicalSpace M₄
                           inst✝⁴ : AddCommGroup M₄
                           inst✝³ : Module R M
                           inst✝² : Module R₂ M₂
                           inst✝¹ : Module R₃ M₃
                           σ₁₂ : RingHom R R₂
                           σ₂₃ : RingHom R₂ R₃
                           σ₁₃ : RingHom R R₃
                           inst✝ : TopologicalAddGroup M₂
                           x✝ : ContinuousLinearMap σ₁₂ M M₂
                           ⊢ Eq (HAdd.hAdd (Neg.neg x✝) x✝) 0
                         -/
  neg_add_cancel _ := by ext; apply neg_add_cancel
                              /-
                                🎉 no goals
                              -/


theorem sub_apply (f g : M →SL[σ₁₂] M₂) (x : M) : (f - g) x = f x - g x :=
  rfl


@[simp, norm_cast]
theorem coe_sub (f g : M →SL[σ₁₂] M₂) : (↑(f - g) : M →ₛₗ[σ₁₂] M₂) = f - g :=
  rfl


@[simp, norm_cast]
theorem coe_sub' (f g : M →SL[σ₁₂] M₂) : ⇑(f - g) = f - g :=
  rfl


@[simp]
theorem comp_neg [RingHomCompTriple σ₁₂ σ₂₃ σ₁₃] [TopologicalAddGroup M₂] [TopologicalAddGroup M₃]
    (g : M₂ →SL[σ₂₃] M₃) (f : M →SL[σ₁₂] M₂) : g.comp (-f) = -g.comp f := by
  /-
    R : Type u_1
    inst✝¹⁴ : Ring R
    R₂ : Type u_2
    inst✝¹³ : Ring R₂
    R₃ : Type u_3
    inst✝¹² : Ring R₃
    M : Type u_4
    inst✝¹¹ : TopologicalSpace M
    inst✝¹⁰ : AddCommGroup M
    M₂ : Type u_5
    inst✝⁹ : TopologicalSpace M₂
    inst✝⁸ : AddCommGroup M₂
    M₃ : Type u_6
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : AddCommGroup M₃
    inst✝⁵ : Module R M
    inst✝⁴ : Module R₂ M₂
    inst✝³ : Module R₃ M₃
    σ₁₂ : RingHom R R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R R₃
    inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    inst✝¹ : TopologicalAddGroup M₂
    inst✝ : TopologicalAddGroup M₃
    g : ContinuousLinearMap σ₂₃ M₂ M₃
    f : ContinuousLinearMap σ₁₂ M M₂
    ⊢ Eq (g.comp (Neg.neg f)) (Neg.neg (g.comp f))
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝¹⁴ : Ring R
    R₂ : Type u_2
    inst✝¹³ : Ring R₂
    R₃ : Type u_3
    inst✝¹² : Ring R₃
    M : Type u_4
    inst✝¹¹ : TopologicalSpace M
    inst✝¹⁰ : AddCommGroup M
    M₂ : Type u_5
    inst✝⁹ : TopologicalSpace M₂
    inst✝⁸ : AddCommGroup M₂
    M₃ : Type u_6
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : AddCommGroup M₃
    inst✝⁵ : Module R M
    inst✝⁴ : Module R₂ M₂
    inst✝³ : Module R₃ M₃
    σ₁₂ : RingHom R R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R R₃
    inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    inst✝¹ : TopologicalAddGroup M₂
    inst✝ : TopologicalAddGroup M₃
    g : ContinuousLinearMap σ₂₃ M₂ M₃
    f : ContinuousLinearMap σ₁₂ M M₂
    x : M
    ⊢ Eq ((g.comp (Neg.neg f)) x) ((Neg.neg (g.comp f)) x)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem neg_comp [RingHomCompTriple σ₁₂ σ₂₃ σ₁₃] [TopologicalAddGroup M₃] (g : M₂ →SL[σ₂₃] M₃)
    (f : M →SL[σ₁₂] M₂) : (-g).comp f = -g.comp f := by
  /-
    R : Type u_1
    inst✝¹³ : Ring R
    R₂ : Type u_2
    inst✝¹² : Ring R₂
    R₃ : Type u_3
    inst✝¹¹ : Ring R₃
    M : Type u_4
    inst✝¹⁰ : TopologicalSpace M
    inst✝⁹ : AddCommGroup M
    M₂ : Type u_5
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : AddCommGroup M₂
    M₃ : Type u_6
    inst✝⁶ : TopologicalSpace M₃
    inst✝⁵ : AddCommGroup M₃
    inst✝⁴ : Module R M
    inst✝³ : Module R₂ M₂
    inst✝² : Module R₃ M₃
    σ₁₂ : RingHom R R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R R₃
    inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    inst✝ : TopologicalAddGroup M₃
    g : ContinuousLinearMap σ₂₃ M₂ M₃
    f : ContinuousLinearMap σ₁₂ M M₂
    ⊢ Eq ((Neg.neg g).comp f) (Neg.neg (g.comp f))
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝¹³ : Ring R
    R₂ : Type u_2
    inst✝¹² : Ring R₂
    R₃ : Type u_3
    inst✝¹¹ : Ring R₃
    M : Type u_4
    inst✝¹⁰ : TopologicalSpace M
    inst✝⁹ : AddCommGroup M
    M₂ : Type u_5
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : AddCommGroup M₂
    M₃ : Type u_6
    inst✝⁶ : TopologicalSpace M₃
    inst✝⁵ : AddCommGroup M₃
    inst✝⁴ : Module R M
    inst✝³ : Module R₂ M₂
    inst✝² : Module R₃ M₃
    σ₁₂ : RingHom R R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R R₃
    inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    inst✝ : TopologicalAddGroup M₃
    g : ContinuousLinearMap σ₂₃ M₂ M₃
    f : ContinuousLinearMap σ₁₂ M M₂
    x✝ : M
    ⊢ Eq (((Neg.neg g).comp f) x✝) ((Neg.neg (g.comp f)) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_sub [RingHomCompTriple σ₁₂ σ₂₃ σ₁₃] [TopologicalAddGroup M₂] [TopologicalAddGroup M₃]
    (g : M₂ →SL[σ₂₃] M₃) (f₁ f₂ : M →SL[σ₁₂] M₂) : g.comp (f₁ - f₂) = g.comp f₁ - g.comp f₂ := by
  /-
    R : Type u_1
    inst✝¹⁴ : Ring R
    R₂ : Type u_2
    inst✝¹³ : Ring R₂
    R₃ : Type u_3
    inst✝¹² : Ring R₃
    M : Type u_4
    inst✝¹¹ : TopologicalSpace M
    inst✝¹⁰ : AddCommGroup M
    M₂ : Type u_5
    inst✝⁹ : TopologicalSpace M₂
    inst✝⁸ : AddCommGroup M₂
    M₃ : Type u_6
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : AddCommGroup M₃
    inst✝⁵ : Module R M
    inst✝⁴ : Module R₂ M₂
    inst✝³ : Module R₃ M₃
    σ₁₂ : RingHom R R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R R₃
    inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    inst✝¹ : TopologicalAddGroup M₂
    inst✝ : TopologicalAddGroup M₃
    g : ContinuousLinearMap σ₂₃ M₂ M₃
    f₁ f₂ : ContinuousLinearMap σ₁₂ M M₂
    ⊢ Eq (g.comp (HSub.hSub f₁ f₂)) (HSub.hSub (g.comp f₁) (g.comp f₂))
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝¹⁴ : Ring R
    R₂ : Type u_2
    inst✝¹³ : Ring R₂
    R₃ : Type u_3
    inst✝¹² : Ring R₃
    M : Type u_4
    inst✝¹¹ : TopologicalSpace M
    inst✝¹⁰ : AddCommGroup M
    M₂ : Type u_5
    inst✝⁹ : TopologicalSpace M₂
    inst✝⁸ : AddCommGroup M₂
    M₃ : Type u_6
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : AddCommGroup M₃
    inst✝⁵ : Module R M
    inst✝⁴ : Module R₂ M₂
    inst✝³ : Module R₃ M₃
    σ₁₂ : RingHom R R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R R₃
    inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    inst✝¹ : TopologicalAddGroup M₂
    inst✝ : TopologicalAddGroup M₃
    g : ContinuousLinearMap σ₂₃ M₂ M₃
    f₁ f₂ : ContinuousLinearMap σ₁₂ M M₂
    x✝ : M
    ⊢ Eq ((g.comp (HSub.hSub f₁ f₂)) x✝) ((HSub.hSub (g.comp f₁) (g.comp f₂)) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem sub_comp [RingHomCompTriple σ₁₂ σ₂₃ σ₁₃] [TopologicalAddGroup M₃] (g₁ g₂ : M₂ →SL[σ₂₃] M₃)
    (f : M →SL[σ₁₂] M₂) : (g₁ - g₂).comp f = g₁.comp f - g₂.comp f := by
  /-
    R : Type u_1
    inst✝¹³ : Ring R
    R₂ : Type u_2
    inst✝¹² : Ring R₂
    R₃ : Type u_3
    inst✝¹¹ : Ring R₃
    M : Type u_4
    inst✝¹⁰ : TopologicalSpace M
    inst✝⁹ : AddCommGroup M
    M₂ : Type u_5
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : AddCommGroup M₂
    M₃ : Type u_6
    inst✝⁶ : TopologicalSpace M₃
    inst✝⁵ : AddCommGroup M₃
    inst✝⁴ : Module R M
    inst✝³ : Module R₂ M₂
    inst✝² : Module R₃ M₃
    σ₁₂ : RingHom R R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R R₃
    inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    inst✝ : TopologicalAddGroup M₃
    g₁ g₂ : ContinuousLinearMap σ₂₃ M₂ M₃
    f : ContinuousLinearMap σ₁₂ M M₂
    ⊢ Eq ((HSub.hSub g₁ g₂).comp f) (HSub.hSub (g₁.comp f) (g₂.comp f))
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝¹³ : Ring R
    R₂ : Type u_2
    inst✝¹² : Ring R₂
    R₃ : Type u_3
    inst✝¹¹ : Ring R₃
    M : Type u_4
    inst✝¹⁰ : TopologicalSpace M
    inst✝⁹ : AddCommGroup M
    M₂ : Type u_5
    inst✝⁸ : TopologicalSpace M₂
    inst✝⁷ : AddCommGroup M₂
    M₃ : Type u_6
    inst✝⁶ : TopologicalSpace M₃
    inst✝⁵ : AddCommGroup M₃
    inst✝⁴ : Module R M
    inst✝³ : Module R₂ M₂
    inst✝² : Module R₃ M₃
    σ₁₂ : RingHom R R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R R₃
    inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    inst✝ : TopologicalAddGroup M₃
    g₁ g₂ : ContinuousLinearMap σ₂₃ M₂ M₃
    f : ContinuousLinearMap σ₁₂ M M₂
    x✝ : M
    ⊢ Eq (((HSub.hSub g₁ g₂).comp f) x✝) ((HSub.hSub (g₁.comp f) (g₂.comp f)) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


instance ring [TopologicalAddGroup M] : Ring (M →L[R] M) where
  __ := ContinuousLinearMap.semiring
  __ := ContinuousLinearMap.addCommGroup
  intCast z := z • (1 : M →L[R] M)
  intCast_ofNat := natCast_zsmul _
  intCast_negSucc := negSucc_zsmul _


@[simp]
theorem intCast_apply [TopologicalAddGroup M] (z : ℤ) (m : M) : (↑z : M →L[R] M) m = z • m :=
  rfl


theorem smulRight_one_pow [TopologicalSpace R] [TopologicalRing R] (c : R) (n : ℕ) :
    smulRight (1 : R →L[R] R) c ^ n = smulRight (1 : R →L[R] R) (c ^ n) := by
  induction n with
  | zero => ext; simp
  | succ n ihn => rw [pow_succ, ihn, mul_def, smulRight_comp, smul_eq_mul, pow_succ']


/-- Given a right inverse `f₂ : M₂ →L[R] M` to `f₁ : M →L[R] M₂`,
`projKerOfRightInverse f₁ f₂ h` is the projection `M →L[R] LinearMap.ker f₁` along
`LinearMap.range f₂`. -/
def projKerOfRightInverse [TopologicalAddGroup M] (f₁ : M →SL[σ₁₂] M₂) (f₂ : M₂ →SL[σ₂₁] M)
    (h : Function.RightInverse f₂ f₁) : M →L[R] LinearMap.ker f₁ :=
                                                                   /-
                                                                     R : Type u_1
                                                                     inst✝¹⁵ : Ring R
                                                                     R₂ : Type u_2
                                                                     inst✝¹⁴ : Ring R₂
                                                                     R₃ : Type u_3
                                                                     inst✝¹³ : Ring R₃
                                                                     M : Type u_4
                                                                     inst✝¹² : TopologicalSpace M
                                                                     inst✝¹¹ : AddCommGroup M
                                                                     M₂ : Type u_5
                                                                     inst✝¹⁰ : TopologicalSpace M₂
                                                                     inst✝⁹ : AddCommGroup M₂
                                                                     M₃ : Type u_6
                                                                     inst✝⁸ : TopologicalSpace M₃
                                                                     inst✝⁷ : AddCommGroup M₃
                                                                     M₄ : Type u_7
                                                                     inst✝⁶ : TopologicalSpace M₄
                                                                     inst✝⁵ : AddCommGroup M₄
                                                                     inst✝⁴ : Module R M
                                                                     inst✝³ : Module R₂ M₂
                                                                     inst✝² : Module R₃ M₃
                                                                     σ₁₂ : RingHom R R₂
                                                                     σ₂₃ : RingHom R₂ R₃
                                                                     σ₁₃ : RingHom R R₃
                                                                     σ₂₁ : RingHom R₂ R
                                                                     inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
                                                                     inst✝ : TopologicalAddGroup M
                                                                     f₁ : ContinuousLinearMap σ₁₂ M M₂
                                                                     f₂ : ContinuousLinearMap σ₂₁ M₂ M
                                                                     h : Function.RightInverse ⇑f₂ ⇑f₁
                                                                     x : M
                                                                     ⊢ Membership.mem (LinearMap.ker f₁) ((HSub.hSub (ContinuousLinearMap.id R M) ( …
                                                                   -/
  (id R M - f₂.comp f₁).codRestrict (LinearMap.ker f₁) fun x => by simp [h (f₁ x)]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem coe_projKerOfRightInverse_apply [TopologicalAddGroup M] (f₁ : M →SL[σ₁₂] M₂)
    (f₂ : M₂ →SL[σ₂₁] M) (h : Function.RightInverse f₂ f₁) (x : M) :
    (f₁.projKerOfRightInverse f₂ h x : M) = x - f₂ (f₁ x) :=
  rfl


@[simp]
theorem projKerOfRightInverse_apply_idem [TopologicalAddGroup M] (f₁ : M →SL[σ₁₂] M₂)
    (f₂ : M₂ →SL[σ₂₁] M) (h : Function.RightInverse f₂ f₁) (x : LinearMap.ker f₁) :
    f₁.projKerOfRightInverse f₂ h x = x := by
  /-
    R : Type u_1
    inst✝⁹ : Ring R
    R₂ : Type u_2
    inst✝⁸ : Ring R₂
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommGroup M
    M₂ : Type u_5
    inst✝⁵ : TopologicalSpace M₂
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : Module R M
    inst✝² : Module R₂ M₂
    σ₁₂ : RingHom R R₂
    σ₂₁ : RingHom R₂ R
    inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
    inst✝ : TopologicalAddGroup M
    f₁ : ContinuousLinearMap σ₁₂ M M₂
    f₂ : ContinuousLinearMap σ₂₁ M₂ M
    h : Function.RightInverse ⇑f₂ ⇑f₁
    x : Subtype fun x => Membership.mem (LinearMap.ker f₁) x
    ⊢ Eq ((f₁.projKerOfRightInverse f₂ h) ↑x) x
  -/
  ext1
  /-
    case a
    R : Type u_1
    inst✝⁹ : Ring R
    R₂ : Type u_2
    inst✝⁸ : Ring R₂
    M : Type u_4
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommGroup M
    M₂ : Type u_5
    inst✝⁵ : TopologicalSpace M₂
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : Module R M
    inst✝² : Module R₂ M₂
    σ₁₂ : RingHom R R₂
    σ₂₁ : RingHom R₂ R
    inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
    inst✝ : TopologicalAddGroup M
    f₁ : ContinuousLinearMap σ₁₂ M M₂
    f₂ : ContinuousLinearMap σ₂₁ M₂ M
    h : Function.RightInverse ⇑f₂ ⇑f₁
    x : Subtype fun x => Membership.mem (LinearMap.ker f₁) x
    ⊢ Eq ↑((f₁.projKerOfRightInverse f₂ h) ↑x) ↑x
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem projKerOfRightInverse_comp_inv [TopologicalAddGroup M] (f₁ : M →SL[σ₁₂] M₂)
    (f₂ : M₂ →SL[σ₂₁] M) (h : Function.RightInverse f₂ f₁) (y : M₂) :
    f₁.projKerOfRightInverse f₂ h (f₂ y) = 0 :=
                              /-
                                R : Type u_1
                                inst✝⁹ : Ring R
                                R₂ : Type u_2
                                inst✝⁸ : Ring R₂
                                M : Type u_4
                                inst✝⁷ : TopologicalSpace M
                                inst✝⁶ : AddCommGroup M
                                M₂ : Type u_5
                                inst✝⁵ : TopologicalSpace M₂
                                inst✝⁴ : AddCommGroup M₂
                                inst✝³ : Module R M
                                inst✝² : Module R₂ M₂
                                σ₁₂ : RingHom R R₂
                                σ₂₁ : RingHom R₂ R
                                inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
                                inst✝ : TopologicalAddGroup M
                                f₁ : ContinuousLinearMap σ₁₂ M M₂
                                f₂ : ContinuousLinearMap σ₂₁ M₂ M
                                h : Function.RightInverse ⇑f₂ ⇑f₁
                                y : M₂
                                ⊢ Eq ↑((f₁.projKerOfRightInverse f₂ h) (f₂ y)) ↑0
                              -/
  Subtype.ext_iff_val.2 <| by simp [h y]
                              /-
                                🎉 no goals
                              -/


/-- A nonzero continuous linear functional is open. -/
protected theorem isOpenMap_of_ne_zero [TopologicalSpace R] [DivisionRing R] [ContinuousSub R]
    [AddCommGroup M] [TopologicalSpace M] [ContinuousAdd M] [Module R M] [ContinuousSMul R M]
    (f : M →L[R] R) (hf : f ≠ 0) : IsOpenMap f :=
  let ⟨x, hx⟩ := exists_ne_zero hf
  IsOpenMap.of_sections fun y =>
                                                                         /-
                                                                           R : Type u_1
                                                                           M : Type u_2
                                                                           inst✝⁷ : TopologicalSpace R
                                                                           inst✝⁶ : DivisionRing R
                                                                           inst✝⁵ : ContinuousSub R
                                                                           inst✝⁴ : AddCommGroup M
                                                                           inst✝³ : TopologicalSpace M
                                                                           inst✝² : ContinuousAdd M
                                                                           inst✝¹ : Module R M
                                                                           inst✝ : ContinuousSMul R M
                                                                           f : ContinuousLinearMap (RingHom.id R) M R
                                                                           hf : Ne f 0
                                                                           x : M
                                                                           hx : Ne (f x) 0
                                                                           y : M
                                                                           ⊢ Continuous fun a => HAdd.hAdd y (HSMul.hSMul (HSub.hSub a (f y)) (HSMul.hSMu …
                                                                         -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
    ⟨fun a => y + (a - f y) • (f x)⁻¹ • x, Continuous.continuousAt <| by continuity, by simp,
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
                  /-
                    R : Type u_1
                    M : Type u_2
                    inst✝⁷ : TopologicalSpace R
                    inst✝⁶ : DivisionRing R
                    inst✝⁵ : ContinuousSub R
                    inst✝⁴ : AddCommGroup M
                    inst✝³ : TopologicalSpace M
                    inst✝² : ContinuousAdd M
                    inst✝¹ : Module R M
                    inst✝ : ContinuousSMul R M
                    f : ContinuousLinearMap (RingHom.id R) M R
                    hf : Ne f 0
                    x : M
                    hx : Ne (f x) 0
                    y : M
                    a : R
                    ⊢ Eq (f ((fun a => HAdd.hAdd y (HSMul.hSMul (HSub.hSub a (f y)) (HSMul.hSMul ( …
                  -/
      fun a => by simp [hx]⟩
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem smul_comp (c : S₃) (h : M₂ →SL[σ₂₃] M₃) (f : M →SL[σ₁₂] M₂) :
    (c • h).comp f = c • h.comp f :=
  rfl


@[simp]
theorem comp_smul [LinearMap.CompatibleSMul N₂ N₃ S R] (hₗ : N₂ →L[R] N₃) (c : S)
    (fₗ : M →L[R] N₂) : hₗ.comp (c • fₗ) = c • hₗ.comp fₗ := by
  /-
    R : Type u_1
    S : Type u_4
    inst✝¹⁷ : Semiring R
    inst✝¹⁶ : Monoid S
    M : Type u_6
    inst✝¹⁵ : TopologicalSpace M
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    N₂ : Type u_9
    inst✝¹² : TopologicalSpace N₂
    inst✝¹¹ : AddCommMonoid N₂
    inst✝¹⁰ : Module R N₂
    N₃ : Type u_10
    inst✝⁹ : TopologicalSpace N₃
    inst✝⁸ : AddCommMonoid N₃
    inst✝⁷ : Module R N₃
    inst✝⁶ : DistribMulAction S N₃
    inst✝⁵ : SMulCommClass R S N₃
    inst✝⁴ : ContinuousConstSMul S N₃
    inst✝³ : DistribMulAction S N₂
    inst✝² : ContinuousConstSMul S N₂
    inst✝¹ : SMulCommClass R S N₂
    inst✝ : LinearMap.CompatibleSMul N₂ N₃ S R
    hₗ : ContinuousLinearMap (RingHom.id R) N₂ N₃
    c : S
    fₗ : ContinuousLinearMap (RingHom.id R) M N₂
    ⊢ Eq (hₗ.comp (HSMul.hSMul c fₗ)) (HSMul.hSMul c (hₗ.comp fₗ))
  -/
  ext x
  /-
    case h
    R : Type u_1
    S : Type u_4
    inst✝¹⁷ : Semiring R
    inst✝¹⁶ : Monoid S
    M : Type u_6
    inst✝¹⁵ : TopologicalSpace M
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    N₂ : Type u_9
    inst✝¹² : TopologicalSpace N₂
    inst✝¹¹ : AddCommMonoid N₂
    inst✝¹⁰ : Module R N₂
    N₃ : Type u_10
    inst✝⁹ : TopologicalSpace N₃
    inst✝⁸ : AddCommMonoid N₃
    inst✝⁷ : Module R N₃
    inst✝⁶ : DistribMulAction S N₃
    inst✝⁵ : SMulCommClass R S N₃
    inst✝⁴ : ContinuousConstSMul S N₃
    inst✝³ : DistribMulAction S N₂
    inst✝² : ContinuousConstSMul S N₂
    inst✝¹ : SMulCommClass R S N₂
    inst✝ : LinearMap.CompatibleSMul N₂ N₃ S R
    hₗ : ContinuousLinearMap (RingHom.id R) N₂ N₃
    c : S
    fₗ : ContinuousLinearMap (RingHom.id R) M N₂
    x : M
    ⊢ Eq ((hₗ.comp (HSMul.hSMul c fₗ)) x) ((HSMul.hSMul c (hₗ.comp fₗ)) x)
  -/
  exact hₗ.map_smul_of_tower c (fₗ x)
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_smulₛₗ [SMulCommClass R₂ R₂ M₂] [SMulCommClass R₃ R₃ M₃] [ContinuousConstSMul R₂ M₂]
    [ContinuousConstSMul R₃ M₃] (h : M₂ →SL[σ₂₃] M₃) (c : R₂) (f : M →SL[σ₁₂] M₂) :
    h.comp (c • f) = σ₂₃ c • h.comp f := by
  /-
    R : Type u_1
    R₂ : Type u_2
    R₃ : Type u_3
    inst✝¹⁶ : Semiring R
    inst✝¹⁵ : Semiring R₂
    inst✝¹⁴ : Semiring R₃
    M : Type u_6
    inst✝¹³ : TopologicalSpace M
    inst✝¹² : AddCommMonoid M
    inst✝¹¹ : Module R M
    M₂ : Type u_7
    inst✝¹⁰ : TopologicalSpace M₂
    inst✝⁹ : AddCommMonoid M₂
    inst✝⁸ : Module R₂ M₂
    M₃ : Type u_8
    inst✝⁷ : TopologicalSpace M₃
    inst✝⁶ : AddCommMonoid M₃
    inst✝⁵ : Module R₃ M₃
    σ₁₂ : RingHom R R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R R₃
    inst✝⁴ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    inst✝³ : SMulCommClass R₂ R₂ M₂
    inst✝² : SMulCommClass R₃ R₃ M₃
    inst✝¹ : ContinuousConstSMul R₂ M₂
    inst✝ : ContinuousConstSMul R₃ M₃
    h : ContinuousLinearMap σ₂₃ M₂ M₃
    c : R₂
    f : ContinuousLinearMap σ₁₂ M M₂
    ⊢ Eq (h.comp (HSMul.hSMul c f)) (HSMul.hSMul (σ₂₃ c) (h.comp f))
  -/
  ext x
  simp only [coe_smul', coe_comp', Function.comp_apply, Pi.smul_apply,
    ContinuousLinearMap.map_smulₛₗ]


instance distribMulAction [ContinuousAdd M₂] : DistribMulAction S₃ (M →SL[σ₁₂] M₂) where
  smul_add a f g := ext fun x => smul_add a (f x) (g x)
  smul_zero a := ext fun _ => smul_zero a


instance module : Module S₃ (M →SL[σ₁₃] M₃) where
  zero_smul _ := ext fun _ => zero_smul S₃ _
  add_smul _ _ _ := ext fun _ => add_smul _ _ _


instance isCentralScalar [Module S₃ᵐᵒᵖ M₃] [IsCentralScalar S₃ M₃] :
    IsCentralScalar S₃ (M →SL[σ₁₃] M₃) where
  op_smul_eq_smul _ _ := ext fun _ => op_smul_eq_smul _ _


/-- The coercion from `M →L[R] M₂` to `M →ₗ[R] M₂`, as a linear map. -/
@[simps]
def coeLM : (M →L[R] N₃) →ₗ[S] M →ₗ[R] N₃ where
  toFun := (↑)
  map_add' f g := coe_add f g
  map_smul' c f := coe_smul c f


/-- The coercion from `M →SL[σ] M₂` to `M →ₛₗ[σ] M₂`, as a linear map. -/
@[simps]
def coeLMₛₗ : (M →SL[σ₁₃] M₃) →ₗ[S₃] M →ₛₗ[σ₁₃] M₃ where
  toFun := (↑)
  map_add' f g := coe_add f g
  map_smul' c f := coe_smul c f


/-- Given `c : E →L[R] S`, `c.smulRightₗ` is the linear map from `F` to `E →L[R] F`
sending `f` to `fun e => c e • f`. See also `ContinuousLinearMap.smulRightL`. -/
def smulRightₗ (c : M →L[R] S) : M₂ →ₗ[T] M →L[R] M₂ where
  toFun := c.smulRight
  map_add' x y := by
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝¹⁸ : Semiring R
      inst✝¹⁷ : Semiring S
      inst✝¹⁶ : Semiring T
      inst✝¹⁵ : Module R S
      inst✝¹⁴ : AddCommMonoid M₂
      inst✝¹³ : Module R M₂
      inst✝¹² : Module S M₂
      inst✝¹¹ : IsScalarTower R S M₂
      inst✝¹⁰ : TopologicalSpace S
      inst✝⁹ : TopologicalSpace M₂
      inst✝⁸ : ContinuousSMul S M₂
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      inst✝⁴ : ContinuousAdd M₂
      inst✝³ : Module T M₂
      inst✝² : ContinuousConstSMul T M₂
      inst✝¹ : SMulCommClass R T M₂
      inst✝ : SMulCommClass S T M₂
      c : ContinuousLinearMap (RingHom.id R) M S
      x y : M₂
      ⊢ Eq (c.smulRight (HAdd.hAdd x y)) (HAdd.hAdd (c.smulRight x) (c.smulRight y))
    -/
    ext e
    /-
      case h
      R : Type u_1
      S : Type u_2
      T : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝¹⁸ : Semiring R
      inst✝¹⁷ : Semiring S
      inst✝¹⁶ : Semiring T
      inst✝¹⁵ : Module R S
      inst✝¹⁴ : AddCommMonoid M₂
      inst✝¹³ : Module R M₂
      inst✝¹² : Module S M₂
      inst✝¹¹ : IsScalarTower R S M₂
      inst✝¹⁰ : TopologicalSpace S
      inst✝⁹ : TopologicalSpace M₂
      inst✝⁸ : ContinuousSMul S M₂
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      inst✝⁴ : ContinuousAdd M₂
      inst✝³ : Module T M₂
      inst✝² : ContinuousConstSMul T M₂
      inst✝¹ : SMulCommClass R T M₂
      inst✝ : SMulCommClass S T M₂
      c : ContinuousLinearMap (RingHom.id R) M S
      x y : M₂
      e : M
      ⊢ Eq ((c.smulRight (HAdd.hAdd x y)) e) ((HAdd.hAdd (c.smulRight x) (c.smulRigh …
    -/
    apply smul_add (c e)
    /-
      🎉 no goals
    -/
  map_smul' a x := by
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝¹⁸ : Semiring R
      inst✝¹⁷ : Semiring S
      inst✝¹⁶ : Semiring T
      inst✝¹⁵ : Module R S
      inst✝¹⁴ : AddCommMonoid M₂
      inst✝¹³ : Module R M₂
      inst✝¹² : Module S M₂
      inst✝¹¹ : IsScalarTower R S M₂
      inst✝¹⁰ : TopologicalSpace S
      inst✝⁹ : TopologicalSpace M₂
      inst✝⁸ : ContinuousSMul S M₂
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      inst✝⁴ : ContinuousAdd M₂
      inst✝³ : Module T M₂
      inst✝² : ContinuousConstSMul T M₂
      inst✝¹ : SMulCommClass R T M₂
      inst✝ : SMulCommClass S T M₂
      c : ContinuousLinearMap (RingHom.id R) M S
      a : T
      x : M₂
      ⊢ Eq ({ toFun := c.smulRight, map_add' := ⋯ }.toFun (HSMul.hSMul a x)) (HSMul. …
    -/
    ext e
    /-
      case h
      R : Type u_1
      S : Type u_2
      T : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝¹⁸ : Semiring R
      inst✝¹⁷ : Semiring S
      inst✝¹⁶ : Semiring T
      inst✝¹⁵ : Module R S
      inst✝¹⁴ : AddCommMonoid M₂
      inst✝¹³ : Module R M₂
      inst✝¹² : Module S M₂
      inst✝¹¹ : IsScalarTower R S M₂
      inst✝¹⁰ : TopologicalSpace S
      inst✝⁹ : TopologicalSpace M₂
      inst✝⁸ : ContinuousSMul S M₂
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      inst✝⁴ : ContinuousAdd M₂
      inst✝³ : Module T M₂
      inst✝² : ContinuousConstSMul T M₂
      inst✝¹ : SMulCommClass R T M₂
      inst✝ : SMulCommClass S T M₂
      c : ContinuousLinearMap (RingHom.id R) M S
      a : T
      x : M₂
      e : M
      ⊢ Eq (({ toFun := c.smulRight, map_add' := ⋯ }.toFun (HSMul.hSMul a x)) e) ((H …
    -/
    dsimp
    /-
      case h
      R : Type u_1
      S : Type u_2
      T : Type u_3
      M : Type u_4
      M₂ : Type u_5
      inst✝¹⁸ : Semiring R
      inst✝¹⁷ : Semiring S
      inst✝¹⁶ : Semiring T
      inst✝¹⁵ : Module R S
      inst✝¹⁴ : AddCommMonoid M₂
      inst✝¹³ : Module R M₂
      inst✝¹² : Module S M₂
      inst✝¹¹ : IsScalarTower R S M₂
      inst✝¹⁰ : TopologicalSpace S
      inst✝⁹ : TopologicalSpace M₂
      inst✝⁸ : ContinuousSMul S M₂
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      inst✝⁴ : ContinuousAdd M₂
      inst✝³ : Module T M₂
      inst✝² : ContinuousConstSMul T M₂
      inst✝¹ : SMulCommClass R T M₂
      inst✝ : SMulCommClass S T M₂
      c : ContinuousLinearMap (RingHom.id R) M S
      a : T
      x : M₂
      e : M
      ⊢ Eq (HSMul.hSMul (c e) (HSMul.hSMul a x)) (HSMul.hSMul a (HSMul.hSMul (c e) x))
    -/
    apply smul_comm
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_smulRightₗ (c : M →L[R] S) : ⇑(smulRightₗ c : M₂ →ₗ[T] M →L[R] M₂) = c.smulRight :=
  rfl


instance algebra : Algebra R (M₂ →L[R] M₂) :=
  Algebra.ofModule smul_comp fun _ _ _ => comp_smul _ _ _


@[simp] theorem algebraMap_apply (r : R) (m : M₂) : algebraMap R (M₂ →L[R] M₂) r m = r • m := rfl


/-- If `A` is an `R`-algebra, then a continuous `A`-linear map can be interpreted as a continuous
`R`-linear map. We assume `LinearMap.CompatibleSMul M M₂ R A` to match assumptions of
`LinearMap.map_smul_of_tower`. -/
def restrictScalars (f : M →L[A] M₂) : M →L[R] M₂ :=
  ⟨(f : M →ₗ[A] M₂).restrictScalars R, f.continuous⟩


@[simp] -- @[norm_cast] -- Porting note: This theorem can't be a `norm_cast` theorem.
theorem coe_restrictScalars (f : M →L[A] M₂) :
    (f.restrictScalars R : M →ₗ[R] M₂) = (f : M →ₗ[A] M₂).restrictScalars R :=
  rfl


@[simp]
theorem coe_restrictScalars' (f : M →L[A] M₂) : ⇑(f.restrictScalars R) = f :=
  rfl


@[simp]
theorem restrictScalars_zero : (0 : M →L[A] M₂).restrictScalars R = 0 :=
  rfl


@[simp]
theorem restrictScalars_add (f g : M →L[A] M₂) :
    (f + g).restrictScalars R = f.restrictScalars R + g.restrictScalars R :=
  rfl


@[simp]
theorem restrictScalars_neg (f : M →L[A] M₂) : (-f).restrictScalars R = -f.restrictScalars R :=
  rfl


@[simp]
theorem restrictScalars_smul (c : S) (f : M →L[A] M₂) :
    (c • f).restrictScalars R = c • f.restrictScalars R :=
  rfl


/-- `ContinuousLinearMap.restrictScalars` as a `LinearMap`. See also
`ContinuousLinearMap.restrictScalarsL`. -/
def restrictScalarsₗ : (M →L[A] M₂) →ₗ[S] M →L[R] M₂ where
  toFun := restrictScalars R
  map_add' := restrictScalars_add
  map_smul' := restrictScalars_smul


@[simp]
theorem coe_restrictScalarsₗ : ⇑(restrictScalarsₗ A M M₂ R S) = restrictScalars R :=
  rfl


/-- A submodule `p` is called *complemented* if there exists a continuous projection `M →ₗ[R] p`. -/
def ClosedComplemented (p : Submodule R M) : Prop :=
  ∃ f : M →L[R] p, ∀ x : p, f x = x


theorem ClosedComplemented.exists_isClosed_isCompl {p : Submodule R M} [T1Space p]
    (h : ClosedComplemented p) :
    ∃ q : Submodule R M, IsClosed (q : Set M) ∧ IsCompl p q :=
  Exists.elim h fun f hf => ⟨ker f, isClosed_ker f, LinearMap.isCompl_of_proj hf⟩


protected theorem ClosedComplemented.isClosed [TopologicalAddGroup M] [T1Space M]
    {p : Submodule R M} (h : ClosedComplemented p) : IsClosed (p : Set M) := by
  /-
    R : Type u_1
    inst✝⁵ : Ring R
    M : Type u_2
    inst✝⁴ : TopologicalSpace M
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : TopologicalAddGroup M
    inst✝ : T1Space M
    p : Submodule R M
    h : p.ClosedComplemented
    ⊢ IsClosed ↑p
  -/
  rcases h with ⟨f, hf⟩
  /-
    case intro
    R : Type u_1
    inst✝⁵ : Ring R
    M : Type u_2
    inst✝⁴ : TopologicalSpace M
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : TopologicalAddGroup M
    inst✝ : T1Space M
    p : Submodule R M
    f : ContinuousLinearMap (RingHom.id R) M (Subtype fun x => Membership.mem p x)
    hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
    ⊢ IsClosed ↑p
  -/
  have : ker (id R M - p.subtypeL.comp f) = p := LinearMap.ker_id_sub_eq_of_proj hf
  /-
    case intro
    R : Type u_1
    inst✝⁵ : Ring R
    M : Type u_2
    inst✝⁴ : TopologicalSpace M
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : TopologicalAddGroup M
    inst✝ : T1Space M
    p : Submodule R M
    f : ContinuousLinearMap (RingHom.id R) M (Subtype fun x => Membership.mem p x)
    hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
    this : Eq (LinearMap.ker (HSub.hSub (ContinuousLinearMap.id R M) (p.subtypeL.c …
    ⊢ IsClosed ↑p
  -/
  exact this ▸ isClosed_ker _
  /-
    🎉 no goals
  -/


@[simp]
theorem closedComplemented_bot : ClosedComplemented (⊥ : Submodule R M) :=
                  /-
                    R : Type u_1
                    inst✝³ : Ring R
                    M : Type u_2
                    inst✝² : TopologicalSpace M
                    inst✝¹ : AddCommGroup M
                    inst✝ : Module R M
                    x : Subtype fun x => Membership.mem Bot.bot x
                    ⊢ Eq (0 ↑x) x
                  -/
  ⟨0, fun x => by simp only [zero_apply, eq_zero_of_bot_submodule x]⟩
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem closedComplemented_top : ClosedComplemented (⊤ : Submodule R M) :=
                                                                                  /-
                                                                                    R : Type u_1
                                                                                    inst✝³ : Ring R
                                                                                    M : Type u_2
                                                                                    inst✝² : TopologicalSpace M
                                                                                    inst✝¹ : AddCommGroup M
                                                                                    inst✝ : Module R M
                                                                                    x : Subtype fun x => Membership.mem Top.top x
                                                                                    ⊢ Eq ↑(((ContinuousLinearMap.id R M).codRestrict Top.top ⋯) ↑x) ↑x
                                                                                  -/
  ⟨(id R M).codRestrict ⊤ fun _x => trivial, fun x => Subtype.ext_iff_val.2 <| by simp⟩
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem ContinuousLinearMap.closedComplemented_ker_of_rightInverse {R : Type*} [Ring R]
    {M : Type*} [TopologicalSpace M] [AddCommGroup M] {M₂ : Type*} [TopologicalSpace M₂]
    [AddCommGroup M₂] [Module R M] [Module R M₂] [TopologicalAddGroup M] (f₁ : M →L[R] M₂)
    (f₂ : M₂ →L[R] M) (h : Function.RightInverse f₂ f₁) : (ker f₁).ClosedComplemented :=
  ⟨f₁.projKerOfRightInverse f₂ h, f₁.projKerOfRightInverse_apply_idem f₂ h⟩

