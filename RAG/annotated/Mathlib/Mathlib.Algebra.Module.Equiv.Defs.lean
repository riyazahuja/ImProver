/-- A linear equivalence is an invertible linear map. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO @[nolint has_nonempty_instance]
structure LinearEquiv {R : Type*} {S : Type*} [Semiring R] [Semiring S] (σ : R →+* S)
  {σ' : S →+* R} [RingHomInvPair σ σ'] [RingHomInvPair σ' σ] (M : Type*) (M₂ : Type*)
  [AddCommMonoid M] [AddCommMonoid M₂] [Module R M] [Module S M₂] extends LinearMap σ M M₂, M ≃+ M₂


/-- The notation `M ≃ₛₗ[σ] M₂` denotes the type of linear equivalences between `M` and `M₂` over a
ring homomorphism `σ`. -/
notation:50 M " ≃ₛₗ[" σ "] " M₂ => LinearEquiv σ M M₂


/-- The notation `M ≃ₗ [R] M₂` denotes the type of linear equivalences between `M` and `M₂` over
a plain linear map `M →ₗ M₂`. -/
notation:50 M " ≃ₗ[" R "] " M₂ => LinearEquiv (RingHom.id R) M M₂


/-- `SemilinearEquivClass F σ M M₂` asserts `F` is a type of bundled `σ`-semilinear equivs
`M → M₂`.

See also `LinearEquivClass F R M M₂` for the case where `σ` is the identity map on `R`.

A map `f` between an `R`-module and an `S`-module over a ring homomorphism `σ : R →+* S`
is semilinear if it satisfies the two properties `f (x + y) = f x + f y` and
`f (c • x) = (σ c) • f x`. -/
class SemilinearEquivClass (F : Type*) {R S : outParam Type*} [Semiring R] [Semiring S]
  (σ : outParam <| R →+* S) {σ' : outParam <| S →+* R} [RingHomInvPair σ σ'] [RingHomInvPair σ' σ]
  (M M₂ : outParam Type*) [AddCommMonoid M] [AddCommMonoid M₂] [Module R M] [Module S M₂]
  [EquivLike F M M₂]
  extends AddEquivClass F M M₂ : Prop where
  /-- Applying a semilinear equivalence `f` over `σ` to `r • x` equals `σ r • f x`. -/
  map_smulₛₗ : ∀ (f : F) (r : R) (x : M), f (r • x) = σ r • f x

-- `R, S, σ, σ'` become metavars, but it's OK since they are outparams.


/-- `LinearEquivClass F R M M₂` asserts `F` is a type of bundled `R`-linear equivs `M → M₂`.
This is an abbreviation for `SemilinearEquivClass F (RingHom.id R) M M₂`.
-/
abbrev LinearEquivClass (F : Type*) (R M M₂ : outParam Type*) [Semiring R] [AddCommMonoid M]
    [AddCommMonoid M₂] [Module R M] [Module R M₂] [EquivLike F M M₂] :=
  SemilinearEquivClass F (RingHom.id R) M M₂


instance (priority := 100) [RingHomInvPair σ σ'] [RingHomInvPair σ' σ]
  [EquivLike F M M₂] [s : SemilinearEquivClass F σ M M₂] : SemilinearMapClass F σ M M₂ :=
  { s with }


/-- Reinterpret an element of a type of semilinear equivalences as a semilinear equivalence. -/
@[coe]
def semilinearEquiv [RingHomInvPair σ σ'] [RingHomInvPair σ' σ]
    [EquivLike F M M₂] [SemilinearEquivClass F σ M M₂] (f : F) : M ≃ₛₗ[σ] M₂ :=
  { (f : M ≃+ M₂), (f : M →ₛₗ[σ] M₂) with }


/-- Reinterpret an element of a type of semilinear equivalences as a semilinear equivalence. -/
instance instCoeToSemilinearEquiv [RingHomInvPair σ σ'] [RingHomInvPair σ' σ]
    [EquivLike F M M₂] [SemilinearEquivClass F σ M M₂] : CoeHead F (M ≃ₛₗ[σ] M₂) where
  coe f := semilinearEquiv f


instance : Coe (M ≃ₛₗ[σ] M₂) (M →ₛₗ[σ] M₂) :=
  ⟨toLinearMap⟩

-- This exists for compatibility, previously `≃ₗ[R]` extended `≃` instead of `≃+`.

/-- The equivalence of types underlying a linear equivalence. -/
def toEquiv : (M ≃ₛₗ[σ] M₂) → M ≃ M₂ := fun f ↦ f.toAddEquiv.toEquiv


theorem toEquiv_injective : Function.Injective (toEquiv : (M ≃ₛₗ[σ] M₂) → M ≃ M₂) :=
  fun ⟨⟨⟨_, _⟩, _⟩, _, _, _⟩ ⟨⟨⟨_, _⟩, _⟩, _, _, _⟩ h ↦
    (LinearEquiv.mk.injEq _ _ _ _ _ _ _ _).mpr
      ⟨LinearMap.ext (congr_fun (Equiv.mk.inj h).1), (Equiv.mk.inj h).2⟩


@[simp]
theorem toEquiv_inj {e₁ e₂ : M ≃ₛₗ[σ] M₂} : e₁.toEquiv = e₂.toEquiv ↔ e₁ = e₂ :=
  toEquiv_injective.eq_iff


theorem toLinearMap_injective : Injective (toLinearMap : (M ≃ₛₗ[σ] M₂) → M →ₛₗ[σ] M₂) :=
  fun _ _ H ↦ toEquiv_injective <| Equiv.ext <| LinearMap.congr_fun H


@[simp, norm_cast]
theorem toLinearMap_inj {e₁ e₂ : M ≃ₛₗ[σ] M₂} : (↑e₁ : M →ₛₗ[σ] M₂) = e₂ ↔ e₁ = e₂ :=
  toLinearMap_injective.eq_iff


instance : EquivLike (M ≃ₛₗ[σ] M₂) M M₂ where
  inv := LinearEquiv.invFun
  coe_injective' _ _ h _ := toLinearMap_injective (DFunLike.coe_injective h)
  left_inv := LinearEquiv.left_inv
  right_inv := LinearEquiv.right_inv


instance : SemilinearEquivClass (M ≃ₛₗ[σ] M₂) σ M M₂ where
  map_add := (·.map_add') --map_add' Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO why did I need to change this?
  map_smulₛₗ := (·.map_smul') --map_smul' Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO why did I need to change this?

-- Porting note: moved to a lower line since there is no shortcut `CoeFun` instance any more

@[simp]
theorem coe_mk {to_fun inv_fun map_add map_smul left_inv right_inv} :
    (⟨⟨⟨to_fun, map_add⟩, map_smul⟩, inv_fun, left_inv, right_inv⟩ : M ≃ₛₗ[σ] M₂) = to_fun := rfl


theorem coe_injective : @Injective (M ≃ₛₗ[σ] M₂) (M → M₂) CoeFun.coe :=
  DFunLike.coe_injective


@[simp, norm_cast]
theorem coe_coe : ⇑(e : M →ₛₗ[σ] M₂) = e :=
  rfl


@[simp]
theorem coe_toEquiv : ⇑(e.toEquiv) = e :=
  rfl


@[simp]
theorem coe_toLinearMap : ⇑e.toLinearMap = e :=
  rfl

-- Porting note: no longer a `simp`

theorem toFun_eq_coe : e.toFun = e := rfl


@[ext]
theorem ext (h : ∀ x, e x = e' x) : e = e' :=
  DFunLike.ext _ _ h


protected theorem congr_arg {x x'} : x = x' → e x = e x' :=
  DFunLike.congr_arg e


protected theorem congr_fun (h : e = e') (x : M) : e x = e' x :=
  DFunLike.congr_fun h x


/-- The identity map is a linear equivalence. -/
@[refl]
def refl [Module R M] : M ≃ₗ[R] M :=
  { LinearMap.id, Equiv.refl M with }


@[simp]
theorem refl_apply [Module R M] (x : M) : refl R M x = x :=
  rfl


/-- Linear equivalences are symmetric. -/
@[symm]
def symm (e : M ≃ₛₗ[σ] M₂) : M₂ ≃ₛₗ[σ'] M :=
  { e.toLinearMap.inverse e.invFun e.left_inv e.right_inv,
    e.toEquiv.symm with
    toFun := e.toLinearMap.inverse e.invFun e.left_inv e.right_inv
    invFun := e.toEquiv.symm.invFun
                              /-
                                R : Type u_1
                                R₁ : Type u_2
                                R₂ : Type u_3
                                R₃ : Type u_4
                                S : Type u_5
                                M : Type u_6
                                M₁ : Type u_7
                                M₂ : Type u_8
                                M₃ : Type u_9
                                N₁ : Type u_10
                                N₂ : Type u_11
                                inst✝¹⁰ : Semiring R
                                inst✝⁹ : Semiring S
                                inst✝⁸ : Semiring R₁
                                inst✝⁷ : Semiring R₂
                                inst✝⁶ : Semiring R₃
                                inst✝⁵ : AddCommMonoid M
                                inst✝⁴ : AddCommMonoid M₁
                                inst✝³ : AddCommMonoid M₂
                                inst✝² : AddCommMonoid M₃
                                inst✝¹ : AddCommMonoid N₁
                                inst✝ : AddCommMonoid N₂
                                module_M : Module R M
                                module_S_M₂ : Module S M₂
                                σ : RingHom R S
                                σ' : RingHom S R
                                re₁ : RingHomInvPair σ σ'
                                re₂ : RingHomInvPair σ' σ
                                e✝ e' e : LinearEquiv σ M M₂
                                r : S
                                x : M₂
                                ⊢ Eq ({ toFun := ⇑((↑e).inverse e.invFun ⋯ ⋯), map_add' := ⋯ }.toFun (HSMul.hS …
                              -/
    map_smul' := fun r x ↦ by dsimp only; rw [map_smulₛₗ] }
                                          /-
                                            🎉 no goals
                                          -/

-- Porting note: this is new

/-- See Note [custom simps projection] -/
def Simps.apply {R : Type*} {S : Type*} [Semiring R] [Semiring S]
    {σ : R →+* S} {σ' : S →+* R} [RingHomInvPair σ σ'] [RingHomInvPair σ' σ]
    {M : Type*} {M₂ : Type*} [AddCommMonoid M] [AddCommMonoid M₂] [Module R M] [Module S M₂]
    (e : M ≃ₛₗ[σ] M₂) : M → M₂ :=
  e


/-- See Note [custom simps projection] -/
def Simps.symm_apply {R : Type*} {S : Type*} [Semiring R] [Semiring S]
    {σ : R →+* S} {σ' : S →+* R} [RingHomInvPair σ σ'] [RingHomInvPair σ' σ]
    {M : Type*} {M₂ : Type*} [AddCommMonoid M] [AddCommMonoid M₂] [Module R M] [Module S M₂]
    (e : M ≃ₛₗ[σ] M₂) : M₂ → M :=
  e.symm


@[simp]
theorem invFun_eq_symm : e.invFun = e.symm :=
  rfl


@[simp]
theorem coe_toEquiv_symm : e.toEquiv.symm = e.symm :=
  rfl


set_option linter.unusedVariables false in
/-- Linear equivalences are transitive. -/
-- Note: the `RingHomCompTriple σ₃₂ σ₂₁ σ₃₁` is unused, but is convenient to carry around
-- implicitly for lemmas like `LinearEquiv.self_trans_symm`.
@[trans, nolint unusedArguments]
def trans
    [RingHomCompTriple σ₁₂ σ₂₃ σ₁₃] [RingHomCompTriple σ₃₂ σ₂₁ σ₃₁]
    {re₁₂ : RingHomInvPair σ₁₂ σ₂₁} {re₂₃ : RingHomInvPair σ₂₃ σ₃₂}
    [RingHomInvPair σ₁₃ σ₃₁] {re₂₁ : RingHomInvPair σ₂₁ σ₁₂}
    {re₃₂ : RingHomInvPair σ₃₂ σ₂₃} [RingHomInvPair σ₃₁ σ₁₃]
    (e₁₂ : M₁ ≃ₛₗ[σ₁₂] M₂) (e₂₃ : M₂ ≃ₛₗ[σ₂₃] M₃) : M₁ ≃ₛₗ[σ₁₃] M₃ :=
  { e₂₃.toLinearMap.comp e₁₂.toLinearMap, e₁₂.toEquiv.trans e₂₃.toEquiv with }


/-- The notation `e₁ ≪≫ₗ e₂` denotes the composition of the linear equivalences `e₁` and `e₂`. -/
notation3:80 (name := transNotation) e₁:80 " ≪≫ₗ " e₂:81 =>
  @LinearEquiv.trans _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ (RingHom.id _) (RingHom.id _) (RingHom.id _)
    (RingHom.id _) (RingHom.id _) (RingHom.id _) RingHomCompTriple.ids RingHomCompTriple.ids
    RingHomInvPair.ids RingHomInvPair.ids RingHomInvPair.ids RingHomInvPair.ids RingHomInvPair.ids
    RingHomInvPair.ids e₁ e₂


@[simp]
theorem coe_toAddEquiv : e.toAddEquiv = e :=
  rfl


/-- The two paths coercion can take to an `AddMonoidHom` are equivalent -/
theorem toAddMonoidHom_commutes : e.toLinearMap.toAddMonoidHom = e.toAddEquiv.toAddMonoidHom :=
  rfl


lemma coe_toAddEquiv_symm : (e₁₂.symm : M₂ ≃+ M₁) = (e₁₂ : M₁ ≃+ M₂).symm := by
  /-
    R₁ : Type u_2
    R₂ : Type u_3
    M₁ : Type u_7
    M₂ : Type u_8
    inst✝³ : Semiring R₁
    inst✝² : Semiring R₂
    inst✝¹ : AddCommMonoid M₁
    inst✝ : AddCommMonoid M₂
    module_M₁ : Module R₁ M₁
    module_M₂ : Module R₂ M₂
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    e₁₂ : LinearEquiv σ₁₂ M₁ M₂
    ⊢ Eq (↑e₁₂.symm) (↑e₁₂).symm
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem trans_apply (c : M₁) : (e₁₂.trans e₂₃ : M₁ ≃ₛₗ[σ₁₃] M₃) c = e₂₃ (e₁₂ c) :=
  rfl


theorem coe_trans :
    (e₁₂.trans e₂₃ : M₁ →ₛₗ[σ₁₃] M₃) = (e₂₃ : M₂ →ₛₗ[σ₂₃] M₃).comp (e₁₂ : M₁ →ₛₗ[σ₁₂] M₂) :=
  rfl


@[simp]
theorem apply_symm_apply (c : M₂) : e (e.symm c) = c :=
  e.right_inv c


@[simp]
theorem symm_apply_apply (b : M) : e.symm (e b) = b :=
  e.left_inv b


@[simp]
theorem trans_symm : (e₁₂.trans e₂₃ : M₁ ≃ₛₗ[σ₁₃] M₃).symm = e₂₃.symm.trans e₁₂.symm :=
  rfl


theorem symm_trans_apply (c : M₃) :
    (e₁₂.trans e₂₃ : M₁ ≃ₛₗ[σ₁₃] M₃).symm c = e₁₂.symm (e₂₃.symm c) :=
  rfl


@[simp]
theorem trans_refl : e.trans (refl S M₂) = e :=
  toEquiv_injective e.toEquiv.trans_refl


@[simp]
theorem refl_trans : (refl R M).trans e = e :=
  toEquiv_injective e.toEquiv.refl_trans


theorem symm_apply_eq {x y} : e.symm x = y ↔ x = e y :=
  e.toEquiv.symm_apply_eq


theorem eq_symm_apply {x y} : y = e.symm x ↔ e y = x :=
  e.toEquiv.eq_symm_apply


theorem eq_comp_symm {α : Type*} (f : M₂ → α) (g : M₁ → α) : f = g ∘ e₁₂.symm ↔ f ∘ e₁₂ = g :=
  e₁₂.toEquiv.eq_comp_symm f g


theorem comp_symm_eq {α : Type*} (f : M₂ → α) (g : M₁ → α) : g ∘ e₁₂.symm = f ↔ g = f ∘ e₁₂ :=
  e₁₂.toEquiv.comp_symm_eq f g


theorem eq_symm_comp {α : Type*} (f : α → M₁) (g : α → M₂) : f = e₁₂.symm ∘ g ↔ e₁₂ ∘ f = g :=
  e₁₂.toEquiv.eq_symm_comp f g


theorem symm_comp_eq {α : Type*} (f : α → M₁) (g : α → M₂) : e₁₂.symm ∘ g = f ↔ g = e₁₂ ∘ f :=
  e₁₂.toEquiv.symm_comp_eq f g


theorem eq_comp_toLinearMap_symm (f : M₂ →ₛₗ[σ₂₃] M₃) (g : M₁ →ₛₗ[σ₁₃] M₃) :
    f = g.comp e₁₂.symm.toLinearMap ↔ f.comp e₁₂.toLinearMap = g := by
  /-
    R₁ : Type u_2
    R₂ : Type u_3
    R₃ : Type u_4
    M₁ : Type u_7
    M₂ : Type u_8
    M₃ : Type u_9
    inst✝⁷ : Semiring R₁
    inst✝⁶ : Semiring R₂
    inst✝⁵ : Semiring R₃
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : AddCommMonoid M₂
    inst✝² : AddCommMonoid M₃
    module_M₁ : Module R₁ M₁
    module_M₂ : Module R₂ M₂
    module_M₃ : Module R₃ M₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R₁ R₃
    σ₂₁ : RingHom R₂ R₁
    inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    e₁₂ : LinearEquiv σ₁₂ M₁ M₂
    inst✝ : RingHomCompTriple σ₂₁ σ₁₃ σ₂₃
    f : LinearMap σ₂₃ M₂ M₃
    g : LinearMap σ₁₃ M₁ M₃
    ⊢ Iff (Eq f (g.comp ↑e₁₂.symm)) (Eq (f.comp ↑e₁₂) g)
  -/
  constructor <;> intro H <;> ext
    /-
      case mp.h
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M₁ : Type u_7
      M₂ : Type u_8
      M₃ : Type u_9
      inst✝⁷ : Semiring R₁
      inst✝⁶ : Semiring R₂
      inst✝⁵ : Semiring R₃
      inst✝⁴ : AddCommMonoid M₁
      inst✝³ : AddCommMonoid M₂
      inst✝² : AddCommMonoid M₃
      module_M₁ : Module R₁ M₁
      module_M₂ : Module R₂ M₂
      module_M₃ : Module R₃ M₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      σ₂₁ : RingHom R₂ R₁
      inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      re₁₂ : RingHomInvPair σ₁₂ σ₂₁
      re₂₁ : RingHomInvPair σ₂₁ σ₁₂
      e₁₂ : LinearEquiv σ₁₂ M₁ M₂
      inst✝ : RingHomCompTriple σ₂₁ σ₁₃ σ₂₃
      f : LinearMap σ₂₃ M₂ M₃
      g : LinearMap σ₁₃ M₁ M₃
      H : Eq f (g.comp ↑e₁₂.symm)
      x✝ : M₁
      ⊢ Eq ((f.comp ↑e₁₂) x✝) (g x✝)
    -/
  · simp [H, e₁₂.toEquiv.eq_comp_symm f g]
    /-
      🎉 no goals
    -/
    /-
      case mpr.h
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M₁ : Type u_7
      M₂ : Type u_8
      M₃ : Type u_9
      inst✝⁷ : Semiring R₁
      inst✝⁶ : Semiring R₂
      inst✝⁵ : Semiring R₃
      inst✝⁴ : AddCommMonoid M₁
      inst✝³ : AddCommMonoid M₂
      inst✝² : AddCommMonoid M₃
      module_M₁ : Module R₁ M₁
      module_M₂ : Module R₂ M₂
      module_M₃ : Module R₃ M₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      σ₂₁ : RingHom R₂ R₁
      inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      re₁₂ : RingHomInvPair σ₁₂ σ₂₁
      re₂₁ : RingHomInvPair σ₂₁ σ₁₂
      e₁₂ : LinearEquiv σ₁₂ M₁ M₂
      inst✝ : RingHomCompTriple σ₂₁ σ₁₃ σ₂₃
      f : LinearMap σ₂₃ M₂ M₃
      g : LinearMap σ₁₃ M₁ M₃
      H : Eq (f.comp ↑e₁₂) g
      x✝ : M₂
      ⊢ Eq (f x✝) ((g.comp ↑e₁₂.symm) x✝)
    -/
  · simp [← H, ← e₁₂.toEquiv.eq_comp_symm f g]
    /-
      🎉 no goals
    -/


theorem comp_toLinearMap_symm_eq (f : M₂ →ₛₗ[σ₂₃] M₃) (g : M₁ →ₛₗ[σ₁₃] M₃) :
    g.comp e₁₂.symm.toLinearMap = f ↔ g = f.comp e₁₂.toLinearMap := by
  /-
    R₁ : Type u_2
    R₂ : Type u_3
    R₃ : Type u_4
    M₁ : Type u_7
    M₂ : Type u_8
    M₃ : Type u_9
    inst✝⁷ : Semiring R₁
    inst✝⁶ : Semiring R₂
    inst✝⁵ : Semiring R₃
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : AddCommMonoid M₂
    inst✝² : AddCommMonoid M₃
    module_M₁ : Module R₁ M₁
    module_M₂ : Module R₂ M₂
    module_M₃ : Module R₃ M₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R₁ R₃
    σ₂₁ : RingHom R₂ R₁
    inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    e₁₂ : LinearEquiv σ₁₂ M₁ M₂
    inst✝ : RingHomCompTriple σ₂₁ σ₁₃ σ₂₃
    f : LinearMap σ₂₃ M₂ M₃
    g : LinearMap σ₁₃ M₁ M₃
    ⊢ Iff (Eq (g.comp ↑e₁₂.symm) f) (Eq g (f.comp ↑e₁₂))
  -/
  constructor <;> intro H <;> ext
    /-
      case mp.h
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M₁ : Type u_7
      M₂ : Type u_8
      M₃ : Type u_9
      inst✝⁷ : Semiring R₁
      inst✝⁶ : Semiring R₂
      inst✝⁵ : Semiring R₃
      inst✝⁴ : AddCommMonoid M₁
      inst✝³ : AddCommMonoid M₂
      inst✝² : AddCommMonoid M₃
      module_M₁ : Module R₁ M₁
      module_M₂ : Module R₂ M₂
      module_M₃ : Module R₃ M₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      σ₂₁ : RingHom R₂ R₁
      inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      re₁₂ : RingHomInvPair σ₁₂ σ₂₁
      re₂₁ : RingHomInvPair σ₂₁ σ₁₂
      e₁₂ : LinearEquiv σ₁₂ M₁ M₂
      inst✝ : RingHomCompTriple σ₂₁ σ₁₃ σ₂₃
      f : LinearMap σ₂₃ M₂ M₃
      g : LinearMap σ₁₃ M₁ M₃
      H : Eq (g.comp ↑e₁₂.symm) f
      x✝ : M₁
      ⊢ Eq (g x✝) ((f.comp ↑e₁₂) x✝)
    -/
  · simp [← H, ← e₁₂.toEquiv.comp_symm_eq f g]
    /-
      🎉 no goals
    -/
    /-
      case mpr.h
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M₁ : Type u_7
      M₂ : Type u_8
      M₃ : Type u_9
      inst✝⁷ : Semiring R₁
      inst✝⁶ : Semiring R₂
      inst✝⁵ : Semiring R₃
      inst✝⁴ : AddCommMonoid M₁
      inst✝³ : AddCommMonoid M₂
      inst✝² : AddCommMonoid M₃
      module_M₁ : Module R₁ M₁
      module_M₂ : Module R₂ M₂
      module_M₃ : Module R₃ M₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₃ : RingHom R₂ R₃
      σ₁₃ : RingHom R₁ R₃
      σ₂₁ : RingHom R₂ R₁
      inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
      re₁₂ : RingHomInvPair σ₁₂ σ₂₁
      re₂₁ : RingHomInvPair σ₂₁ σ₁₂
      e₁₂ : LinearEquiv σ₁₂ M₁ M₂
      inst✝ : RingHomCompTriple σ₂₁ σ₁₃ σ₂₃
      f : LinearMap σ₂₃ M₂ M₃
      g : LinearMap σ₁₃ M₁ M₃
      H : Eq g (f.comp ↑e₁₂)
      x✝ : M₂
      ⊢ Eq ((g.comp ↑e₁₂.symm) x✝) (f x✝)
    -/
  · simp [H, e₁₂.toEquiv.comp_symm_eq f g]
    /-
      🎉 no goals
    -/


theorem eq_toLinearMap_symm_comp (f : M₃ →ₛₗ[σ₃₁] M₁) (g : M₃ →ₛₗ[σ₃₂] M₂) :
    f = e₁₂.symm.toLinearMap.comp g ↔ e₁₂.toLinearMap.comp f = g := by
  /-
    R₁ : Type u_2
    R₂ : Type u_3
    R₃ : Type u_4
    M₁ : Type u_7
    M₂ : Type u_8
    M₃ : Type u_9
    inst✝⁷ : Semiring R₁
    inst✝⁶ : Semiring R₂
    inst✝⁵ : Semiring R₃
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : AddCommMonoid M₂
    inst✝² : AddCommMonoid M₃
    module_M₁ : Module R₁ M₁
    module_M₂ : Module R₂ M₂
    module_M₃ : Module R₃ M₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    σ₃₂ : RingHom R₃ R₂
    σ₃₁ : RingHom R₃ R₁
    inst✝¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    e₁₂ : LinearEquiv σ₁₂ M₁ M₂
    inst✝ : RingHomCompTriple σ₃₁ σ₁₂ σ₃₂
    f : LinearMap σ₃₁ M₃ M₁
    g : LinearMap σ₃₂ M₃ M₂
    ⊢ Iff (Eq f ((↑e₁₂.symm).comp g)) (Eq ((↑e₁₂).comp f) g)
  -/
  constructor <;> intro H <;> ext
    /-
      case mp.h
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M₁ : Type u_7
      M₂ : Type u_8
      M₃ : Type u_9
      inst✝⁷ : Semiring R₁
      inst✝⁶ : Semiring R₂
      inst✝⁵ : Semiring R₃
      inst✝⁴ : AddCommMonoid M₁
      inst✝³ : AddCommMonoid M₂
      inst✝² : AddCommMonoid M₃
      module_M₁ : Module R₁ M₁
      module_M₂ : Module R₂ M₂
      module_M₃ : Module R₃ M₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₁ : RingHom R₂ R₁
      σ₃₂ : RingHom R₃ R₂
      σ₃₁ : RingHom R₃ R₁
      inst✝¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
      re₁₂ : RingHomInvPair σ₁₂ σ₂₁
      re₂₁ : RingHomInvPair σ₂₁ σ₁₂
      e₁₂ : LinearEquiv σ₁₂ M₁ M₂
      inst✝ : RingHomCompTriple σ₃₁ σ₁₂ σ₃₂
      f : LinearMap σ₃₁ M₃ M₁
      g : LinearMap σ₃₂ M₃ M₂
      H : Eq f ((↑e₁₂.symm).comp g)
      x✝ : M₃
      ⊢ Eq (((↑e₁₂).comp f) x✝) (g x✝)
    -/
  · simp [H, e₁₂.toEquiv.eq_symm_comp f g]
    /-
      🎉 no goals
    -/
    /-
      case mpr.h
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M₁ : Type u_7
      M₂ : Type u_8
      M₃ : Type u_9
      inst✝⁷ : Semiring R₁
      inst✝⁶ : Semiring R₂
      inst✝⁵ : Semiring R₃
      inst✝⁴ : AddCommMonoid M₁
      inst✝³ : AddCommMonoid M₂
      inst✝² : AddCommMonoid M₃
      module_M₁ : Module R₁ M₁
      module_M₂ : Module R₂ M₂
      module_M₃ : Module R₃ M₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₁ : RingHom R₂ R₁
      σ₃₂ : RingHom R₃ R₂
      σ₃₁ : RingHom R₃ R₁
      inst✝¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
      re₁₂ : RingHomInvPair σ₁₂ σ₂₁
      re₂₁ : RingHomInvPair σ₂₁ σ₁₂
      e₁₂ : LinearEquiv σ₁₂ M₁ M₂
      inst✝ : RingHomCompTriple σ₃₁ σ₁₂ σ₃₂
      f : LinearMap σ₃₁ M₃ M₁
      g : LinearMap σ₃₂ M₃ M₂
      H : Eq ((↑e₁₂).comp f) g
      x✝ : M₃
      ⊢ Eq (f x✝) (((↑e₁₂.symm).comp g) x✝)
    -/
  · simp [← H, ← e₁₂.toEquiv.eq_symm_comp f g]
    /-
      🎉 no goals
    -/


theorem toLinearMap_symm_comp_eq (f : M₃ →ₛₗ[σ₃₁] M₁) (g : M₃ →ₛₗ[σ₃₂] M₂) :
    e₁₂.symm.toLinearMap.comp g = f ↔ g = e₁₂.toLinearMap.comp f := by
  /-
    R₁ : Type u_2
    R₂ : Type u_3
    R₃ : Type u_4
    M₁ : Type u_7
    M₂ : Type u_8
    M₃ : Type u_9
    inst✝⁷ : Semiring R₁
    inst✝⁶ : Semiring R₂
    inst✝⁵ : Semiring R₃
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : AddCommMonoid M₂
    inst✝² : AddCommMonoid M₃
    module_M₁ : Module R₁ M₁
    module_M₂ : Module R₂ M₂
    module_M₃ : Module R₃ M₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    σ₃₂ : RingHom R₃ R₂
    σ₃₁ : RingHom R₃ R₁
    inst✝¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    e₁₂ : LinearEquiv σ₁₂ M₁ M₂
    inst✝ : RingHomCompTriple σ₃₁ σ₁₂ σ₃₂
    f : LinearMap σ₃₁ M₃ M₁
    g : LinearMap σ₃₂ M₃ M₂
    ⊢ Iff (Eq ((↑e₁₂.symm).comp g) f) (Eq g ((↑e₁₂).comp f))
  -/
  constructor <;> intro H <;> ext
    /-
      case mp.h
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M₁ : Type u_7
      M₂ : Type u_8
      M₃ : Type u_9
      inst✝⁷ : Semiring R₁
      inst✝⁶ : Semiring R₂
      inst✝⁵ : Semiring R₃
      inst✝⁴ : AddCommMonoid M₁
      inst✝³ : AddCommMonoid M₂
      inst✝² : AddCommMonoid M₃
      module_M₁ : Module R₁ M₁
      module_M₂ : Module R₂ M₂
      module_M₃ : Module R₃ M₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₁ : RingHom R₂ R₁
      σ₃₂ : RingHom R₃ R₂
      σ₃₁ : RingHom R₃ R₁
      inst✝¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
      re₁₂ : RingHomInvPair σ₁₂ σ₂₁
      re₂₁ : RingHomInvPair σ₂₁ σ₁₂
      e₁₂ : LinearEquiv σ₁₂ M₁ M₂
      inst✝ : RingHomCompTriple σ₃₁ σ₁₂ σ₃₂
      f : LinearMap σ₃₁ M₃ M₁
      g : LinearMap σ₃₂ M₃ M₂
      H : Eq ((↑e₁₂.symm).comp g) f
      x✝ : M₃
      ⊢ Eq (g x✝) (((↑e₁₂).comp f) x✝)
    -/
  · simp [← H, ← e₁₂.toEquiv.symm_comp_eq f g]
    /-
      🎉 no goals
    -/
    /-
      case mpr.h
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      M₁ : Type u_7
      M₂ : Type u_8
      M₃ : Type u_9
      inst✝⁷ : Semiring R₁
      inst✝⁶ : Semiring R₂
      inst✝⁵ : Semiring R₃
      inst✝⁴ : AddCommMonoid M₁
      inst✝³ : AddCommMonoid M₂
      inst✝² : AddCommMonoid M₃
      module_M₁ : Module R₁ M₁
      module_M₂ : Module R₂ M₂
      module_M₃ : Module R₃ M₃
      σ₁₂ : RingHom R₁ R₂
      σ₂₁ : RingHom R₂ R₁
      σ₃₂ : RingHom R₃ R₂
      σ₃₁ : RingHom R₃ R₁
      inst✝¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
      re₁₂ : RingHomInvPair σ₁₂ σ₂₁
      re₂₁ : RingHomInvPair σ₂₁ σ₁₂
      e₁₂ : LinearEquiv σ₁₂ M₁ M₂
      inst✝ : RingHomCompTriple σ₃₁ σ₁₂ σ₃₂
      f : LinearMap σ₃₁ M₃ M₁
      g : LinearMap σ₃₂ M₃ M₂
      H : Eq g ((↑e₁₂).comp f)
      x✝ : M₃
      ⊢ Eq (((↑e₁₂.symm).comp g) x✝) (f x✝)
    -/
  · simp [H, e₁₂.toEquiv.symm_comp_eq f g]
    /-
      🎉 no goals
    -/


@[simp]
theorem comp_toLinearMap_eq_iff (f g : M₃ →ₛₗ[σ₃₁] M₁) :
    e₁₂.toLinearMap.comp f = e₁₂.toLinearMap.comp g ↔ f = g := by
  /-
    R₁ : Type u_2
    R₂ : Type u_3
    R₃ : Type u_4
    M₁ : Type u_7
    M₂ : Type u_8
    M₃ : Type u_9
    inst✝⁷ : Semiring R₁
    inst✝⁶ : Semiring R₂
    inst✝⁵ : Semiring R₃
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : AddCommMonoid M₂
    inst✝² : AddCommMonoid M₃
    module_M₁ : Module R₁ M₁
    module_M₂ : Module R₂ M₂
    module_M₃ : Module R₃ M₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    σ₃₂ : RingHom R₃ R₂
    σ₃₁ : RingHom R₃ R₁
    inst✝¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    e₁₂ : LinearEquiv σ₁₂ M₁ M₂
    inst✝ : RingHomCompTriple σ₃₁ σ₁₂ σ₃₂
    f g : LinearMap σ₃₁ M₃ M₁
    ⊢ Iff (Eq ((↑e₁₂).comp f) ((↑e₁₂).comp g)) (Eq f g)
  -/
  refine ⟨fun h => ?_, congrArg e₁₂.comp⟩
  /-
    R₁ : Type u_2
    R₂ : Type u_3
    R₃ : Type u_4
    M₁ : Type u_7
    M₂ : Type u_8
    M₃ : Type u_9
    inst✝⁷ : Semiring R₁
    inst✝⁶ : Semiring R₂
    inst✝⁵ : Semiring R₃
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : AddCommMonoid M₂
    inst✝² : AddCommMonoid M₃
    module_M₁ : Module R₁ M₁
    module_M₂ : Module R₂ M₂
    module_M₃ : Module R₃ M₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    σ₃₂ : RingHom R₃ R₂
    σ₃₁ : RingHom R₃ R₁
    inst✝¹ : RingHomCompTriple σ₃₂ σ₂₁ σ₃₁
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    e₁₂ : LinearEquiv σ₁₂ M₁ M₂
    inst✝ : RingHomCompTriple σ₃₁ σ₁₂ σ₃₂
    f g : LinearMap σ₃₁ M₃ M₁
    h : Eq ((↑e₁₂).comp f) ((↑e₁₂).comp g)
    ⊢ Eq f g
  -/
  rw [← (toLinearMap_symm_comp_eq g (e₁₂.toLinearMap.comp f)).mpr h, eq_toLinearMap_symm_comp]
  /-
    🎉 no goals
  -/


@[simp]
theorem eq_comp_toLinearMap_iff (f g : M₂ →ₛₗ[σ₂₃] M₃) :
    f.comp e₁₂.toLinearMap = g.comp e₁₂.toLinearMap ↔ f = g := by
  /-
    R₁ : Type u_2
    R₂ : Type u_3
    R₃ : Type u_4
    M₁ : Type u_7
    M₂ : Type u_8
    M₃ : Type u_9
    inst✝⁷ : Semiring R₁
    inst✝⁶ : Semiring R₂
    inst✝⁵ : Semiring R₃
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : AddCommMonoid M₂
    inst✝² : AddCommMonoid M₃
    module_M₁ : Module R₁ M₁
    module_M₂ : Module R₂ M₂
    module_M₃ : Module R₃ M₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R₁ R₃
    σ₂₁ : RingHom R₂ R₁
    inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    e₁₂ : LinearEquiv σ₁₂ M₁ M₂
    inst✝ : RingHomCompTriple σ₂₁ σ₁₃ σ₂₃
    f g : LinearMap σ₂₃ M₂ M₃
    ⊢ Iff (Eq (f.comp ↑e₁₂) (g.comp ↑e₁₂)) (Eq f g)
  -/
  refine ⟨fun h => ?_, fun a ↦ congrFun (congrArg LinearMap.comp a) e₁₂.toLinearMap⟩
  /-
    R₁ : Type u_2
    R₂ : Type u_3
    R₃ : Type u_4
    M₁ : Type u_7
    M₂ : Type u_8
    M₃ : Type u_9
    inst✝⁷ : Semiring R₁
    inst✝⁶ : Semiring R₂
    inst✝⁵ : Semiring R₃
    inst✝⁴ : AddCommMonoid M₁
    inst✝³ : AddCommMonoid M₂
    inst✝² : AddCommMonoid M₃
    module_M₁ : Module R₁ M₁
    module_M₂ : Module R₂ M₂
    module_M₃ : Module R₃ M₃
    σ₁₂ : RingHom R₁ R₂
    σ₂₃ : RingHom R₂ R₃
    σ₁₃ : RingHom R₁ R₃
    σ₂₁ : RingHom R₂ R₁
    inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    e₁₂ : LinearEquiv σ₁₂ M₁ M₂
    inst✝ : RingHomCompTriple σ₂₁ σ₁₃ σ₂₃
    f g : LinearMap σ₂₃ M₂ M₃
    h : Eq (f.comp ↑e₁₂) (g.comp ↑e₁₂)
    ⊢ Eq f g
  -/
  rw [(eq_comp_toLinearMap_symm g (f.comp e₁₂.toLinearMap)).mpr h.symm, eq_comp_toLinearMap_symm]
  /-
    🎉 no goals
  -/


@[simp]
theorem refl_symm [Module R M] : (refl R M).symm = LinearEquiv.refl R M :=
  rfl


@[simp]
theorem self_trans_symm (f : M₁ ≃ₛₗ[σ₁₂] M₂) : f.trans f.symm = LinearEquiv.refl R₁ M₁ := by
  /-
    R₁ : Type u_2
    R₂ : Type u_3
    M₁ : Type u_7
    M₂ : Type u_8
    inst✝³ : Semiring R₁
    inst✝² : Semiring R₂
    inst✝¹ : AddCommMonoid M₁
    inst✝ : AddCommMonoid M₂
    module_M₁ : Module R₁ M₁
    module_M₂ : Module R₂ M₂
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    f : LinearEquiv σ₁₂ M₁ M₂
    ⊢ Eq (f.trans f.symm) (LinearEquiv.refl R₁ M₁)
  -/
  ext x
  /-
    case h
    R₁ : Type u_2
    R₂ : Type u_3
    M₁ : Type u_7
    M₂ : Type u_8
    inst✝³ : Semiring R₁
    inst✝² : Semiring R₂
    inst✝¹ : AddCommMonoid M₁
    inst✝ : AddCommMonoid M₂
    module_M₁ : Module R₁ M₁
    module_M₂ : Module R₂ M₂
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    f : LinearEquiv σ₁₂ M₁ M₂
    x : M₁
    ⊢ Eq ((f.trans f.symm) x) ((LinearEquiv.refl R₁ M₁) x)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem symm_trans_self (f : M₁ ≃ₛₗ[σ₁₂] M₂) : f.symm.trans f = LinearEquiv.refl R₂ M₂ := by
  /-
    R₁ : Type u_2
    R₂ : Type u_3
    M₁ : Type u_7
    M₂ : Type u_8
    inst✝³ : Semiring R₁
    inst✝² : Semiring R₂
    inst✝¹ : AddCommMonoid M₁
    inst✝ : AddCommMonoid M₂
    module_M₁ : Module R₁ M₁
    module_M₂ : Module R₂ M₂
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    f : LinearEquiv σ₁₂ M₁ M₂
    ⊢ Eq (f.symm.trans f) (LinearEquiv.refl R₂ M₂)
  -/
  ext x
  /-
    case h
    R₁ : Type u_2
    R₂ : Type u_3
    M₁ : Type u_7
    M₂ : Type u_8
    inst✝³ : Semiring R₁
    inst✝² : Semiring R₂
    inst✝¹ : AddCommMonoid M₁
    inst✝ : AddCommMonoid M₂
    module_M₁ : Module R₁ M₁
    module_M₂ : Module R₂ M₂
    σ₁₂ : RingHom R₁ R₂
    σ₂₁ : RingHom R₂ R₁
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    f : LinearEquiv σ₁₂ M₁ M₂
    x : M₂
    ⊢ Eq ((f.symm.trans f) x) ((LinearEquiv.refl R₂ M₂) x)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]  -- Porting note: norm_cast
theorem refl_toLinearMap [Module R M] : (LinearEquiv.refl R M : M →ₗ[R] M) = LinearMap.id :=
  rfl


@[simp]  -- Porting note: norm_cast
theorem comp_coe [Module R M] [Module R M₂] [Module R M₃] (f : M ≃ₗ[R] M₂) (f' : M₂ ≃ₗ[R] M₃) :
    (f' : M₂ →ₗ[R] M₃).comp (f : M →ₗ[R] M₂) = (f.trans f' : M ≃ₗ[R] M₃) :=
  rfl


@[simp]
theorem mk_coe (f h₁ h₂) : (LinearEquiv.mk e f h₁ h₂ : M ≃ₛₗ[σ] M₂) = e :=
  ext fun _ ↦ rfl


protected theorem map_add (a b : M) : e (a + b) = e a + e b :=
  map_add e a b


protected theorem map_zero : e 0 = 0 :=
  map_zero e


protected theorem map_smulₛₗ (c : R) (x : M) : e (c • x) = (σ : R → S) c • e x :=
  e.map_smul' c x


theorem map_smul (e : N₁ ≃ₗ[R₁] N₂) (c : R₁) (x : N₁) : e (c • x) = c • e x :=
  map_smulₛₗ e c x


theorem map_eq_zero_iff {x : M} : e x = 0 ↔ x = 0 :=
  e.toAddEquiv.map_eq_zero_iff


theorem map_ne_zero_iff {x : M} : e x ≠ 0 ↔ x ≠ 0 :=
  e.toAddEquiv.map_ne_zero_iff


@[simp]
theorem symm_symm (e : M ≃ₛₗ[σ] M₂) : e.symm.symm = e := rfl


theorem symm_bijective [Module R M] [Module S M₂] [RingHomInvPair σ' σ] [RingHomInvPair σ σ'] :
    Function.Bijective (symm : (M ≃ₛₗ[σ] M₂) → M₂ ≃ₛₗ[σ'] M) :=
  Function.bijective_iff_has_inverse.mpr ⟨_, symm_symm, symm_symm⟩


@[simp]
theorem mk_coe' (f h₁ h₂ h₃ h₄) :
    (LinearEquiv.mk ⟨⟨f, h₁⟩, h₂⟩ (⇑e) h₃ h₄ : M₂ ≃ₛₗ[σ'] M) = e.symm :=
  symm_bijective.injective <| ext fun _ ↦ rfl


/-- Auxiliary definition to avoid looping in `dsimp` with `LinearEquiv.symm_mk`. -/
protected def symm_mk.aux (f h₁ h₂ h₃ h₄) := (⟨⟨⟨e, h₁⟩, h₂⟩, f, h₃, h₄⟩ : M ≃ₛₗ[σ] M₂).symm


@[simp]
theorem symm_mk (f h₁ h₂ h₃ h₄) :
    (⟨⟨⟨e, h₁⟩, h₂⟩, f, h₃, h₄⟩ : M ≃ₛₗ[σ] M₂).symm =
      { symm_mk.aux e f h₁ h₂ h₃ h₄ with
        toFun := f
        invFun := e } :=
  rfl


@[simp]
theorem coe_symm_mk [Module R M] [Module R M₂]
    {to_fun inv_fun map_add map_smul left_inv right_inv} :
    ⇑(⟨⟨⟨to_fun, map_add⟩, map_smul⟩, inv_fun, left_inv, right_inv⟩ : M ≃ₗ[R] M₂).symm = inv_fun :=
  rfl


protected theorem bijective : Function.Bijective e :=
  e.toEquiv.bijective


protected theorem injective : Function.Injective e :=
  e.toEquiv.injective


protected theorem surjective : Function.Surjective e :=
  e.toEquiv.surjective


protected theorem image_eq_preimage (s : Set M) : e '' s = e.symm ⁻¹' s :=
  e.toEquiv.image_eq_preimage s


protected theorem image_symm_eq_preimage (s : Set M₂) : e.symm '' s = e ⁻¹' s :=
  e.toEquiv.symm.image_eq_preimage s


/-- Interpret a `RingEquiv` `f` as an `f`-semilinear equiv. -/
@[simps]
def _root_.RingEquiv.toSemilinearEquiv (f : R ≃+* S) :
    haveI := RingHomInvPair.of_ringEquiv f
    haveI := RingHomInvPair.symm (↑f : R →+* S) (f.symm : S →+* R)
    R ≃ₛₗ[(↑f : R →+* S)] S :=
  haveI := RingHomInvPair.of_ringEquiv f
  haveI := RingHomInvPair.symm (↑f : R →+* S) (f.symm : S →+* R)
  { f with
    toFun := f
    map_smul' := f.map_mul }


/-- An involutive linear map is a linear equivalence. -/
def ofInvolutive {σ σ' : R →+* R} [RingHomInvPair σ σ'] [RingHomInvPair σ' σ]
    {_ : Module R M} (f : M →ₛₗ[σ] M) (hf : Involutive f) : M ≃ₛₗ[σ] M :=
  { f, hf.toPerm f with }


@[simp]
theorem coe_ofInvolutive {σ σ' : R →+* R} [RingHomInvPair σ σ'] [RingHomInvPair σ' σ]
    {_ : Module R M} (f : M →ₛₗ[σ] M) (hf : Involutive f) : ⇑(ofInvolutive f hf) = f :=
  rfl


