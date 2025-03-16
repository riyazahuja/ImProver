/-- If `M` and `M₂` are both `R`-semimodules and `S`-semimodules and `R`-semimodule structures
are defined by an action of `R` on `S` (formally, we have two scalar towers), then any `S`-linear
equivalence from `M` to `M₂` is also an `R`-linear equivalence.

See also `LinearMap.restrictScalars`. -/
@[simps]
def restrictScalars (f : M ≃ₗ[S] M₂) : M ≃ₗ[R] M₂ :=
  { f.toLinearMap.restrictScalars R with
    toFun := f
    invFun := f.symm
    left_inv := f.left_inv
    right_inv := f.right_inv }


theorem restrictScalars_injective :
    Function.Injective (restrictScalars R : (M ≃ₗ[S] M₂) → M ≃ₗ[R] M₂) := fun _ _ h ↦
  ext (LinearEquiv.congr_fun h : _)


@[simp]
theorem restrictScalars_inj (f g : M ≃ₗ[S] M₂) :
    f.restrictScalars R = g.restrictScalars R ↔ f = g :=
  (restrictScalars_injective R).eq_iff


theorem _root_.Module.End_isUnit_iff [Module R M] (f : Module.End R M) :
    IsUnit f ↔ Function.Bijective f :=
  ⟨fun h ↦
    Function.bijective_iff_has_inverse.mpr <|
      ⟨h.unit.inv,
        ⟨Module.End_isUnit_inv_apply_apply_of_isUnit h,
        Module.End_isUnit_apply_inv_apply_of_isUnit h⟩⟩,
    fun H ↦
    let e : M ≃ₗ[R] M := { f, Equiv.ofBijective f H with }
    ⟨⟨_, e.symm, LinearMap.ext e.right_inv, LinearMap.ext e.left_inv⟩, rfl⟩⟩


instance automorphismGroup : Group (M ≃ₗ[R] M) where
  mul f g := g.trans f
  one := LinearEquiv.refl R M
  inv f := f.symm
  mul_assoc _ _ _ := rfl
  mul_one _ := ext fun _ ↦ rfl
  one_mul _ := ext fun _ ↦ rfl
  inv_mul_cancel f := ext <| f.left_inv


@[simp]
lemma coe_one : ↑(1 : M ≃ₗ[R] M) = id := rfl


@[simp]
lemma coe_toLinearMap_one : (↑(1 : M ≃ₗ[R] M) : M →ₗ[R] M) = LinearMap.id := rfl


@[simp]
lemma coe_toLinearMap_mul {e₁ e₂ : M ≃ₗ[R] M} :
    (↑(e₁ * e₂) : M →ₗ[R] M) = (e₁ : M →ₗ[R] M) * (e₂ : M →ₗ[R] M) :=
  rfl


theorem coe_pow (e : M ≃ₗ[R] M) (n : ℕ) : ⇑(e ^ n) = e^[n] := hom_coe_pow _ rfl (fun _ _ ↦ rfl) _ _


theorem pow_apply (e : M ≃ₗ[R] M) (n : ℕ) (m : M) : (e ^ n) m = e^[n] m := congr_fun (coe_pow e n) m


@[simp] lemma mul_apply (f : M ≃ₗ[R] M) (g : M ≃ₗ[R] M) (x : M) : (f * g) x = f (g x) := rfl


/-- Restriction from `R`-linear automorphisms of `M` to `R`-linear endomorphisms of `M`,
promoted to a monoid hom. -/
@[simps]
def automorphismGroup.toLinearMapMonoidHom : (M ≃ₗ[R] M) →* M →ₗ[R] M where
  toFun e := e.toLinearMap
  map_one' := rfl
  map_mul' _ _ := rfl


/-- The tautological action by `M ≃ₗ[R] M` on `M`.

This generalizes `Function.End.applyMulAction`. -/
instance applyDistribMulAction : DistribMulAction (M ≃ₗ[R] M) M where
  smul := (· <| ·)
  smul_zero := LinearEquiv.map_zero
  smul_add := LinearEquiv.map_add
  one_smul _ := rfl
  mul_smul _ _ _ := rfl


@[simp]
protected theorem smul_def (f : M ≃ₗ[R] M) (a : M) : f • a = f a :=
  rfl


/-- `LinearEquiv.applyDistribMulAction` is faithful. -/
instance apply_faithfulSMul : FaithfulSMul (M ≃ₗ[R] M) M :=
  ⟨LinearEquiv.ext⟩


instance apply_smulCommClass [SMul S R] [SMul S M] [IsScalarTower S R M] :
    SMulCommClass S (M ≃ₗ[R] M) M where
  smul_comm r e m := (e.map_smul_of_tower r m).symm


instance apply_smulCommClass' [SMul S R] [SMul S M] [IsScalarTower S R M] :
    SMulCommClass (M ≃ₗ[R] M) S M :=
  SMulCommClass.symm _ _ _


/-- Any two modules that are subsingletons are isomorphic. -/
@[simps]
def ofSubsingleton : M ≃ₗ[R] M₂ :=
  { (0 : M →ₗ[R] M₂) with
    toFun := fun _ ↦ 0
    invFun := fun _ ↦ 0
    left_inv := fun _ ↦ Subsingleton.elim _ _
    right_inv := fun _ ↦ Subsingleton.elim _ _ }


@[simp]
theorem ofSubsingleton_self : ofSubsingleton M M = refl R M := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Subsingleton M
    ⊢ Eq (LinearEquiv.ofSubsingleton M M) (LinearEquiv.refl R M)
  -/
  ext
  /-
    case h
    R : Type u_1
    M : Type u_5
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Subsingleton M
    x✝ : M
    ⊢ Eq ((LinearEquiv.ofSubsingleton M M) x✝) ((LinearEquiv.refl R M) x✝)
  -/
  simp [eq_iff_true_of_subsingleton]
  /-
    🎉 no goals
  -/


/-- `g : R ≃+* S` is `R`-linear when the module structure on `S` is `Module.compHom S g` . -/
@[simps]
def compHom.toLinearEquiv {R S : Type*} [Semiring R] [Semiring S] (g : R ≃+* S) :
    haveI := compHom S (↑g : R →+* S)
    R ≃ₗ[R] S :=
  letI := compHom S (↑g : R →+* S)
  { g with
    toFun := (g : R → S)
    invFun := (g.symm : S → R)
    map_smul' := g.map_mul }


/-- Each element of the group defines a linear equivalence.

This is a stronger version of `DistribMulAction.toAddEquiv`. -/
@[simps!]
def toLinearEquiv (s : S) : M ≃ₗ[R] M :=
  { toAddEquiv M s, toLinearMap R M s with }


/-- Each element of the group defines a module automorphism.

This is a stronger version of `DistribMulAction.toAddAut`. -/
@[simps]
def toModuleAut : S →* M ≃ₗ[R] M where
  toFun := toLinearEquiv R M
  map_one' := LinearEquiv.ext <| one_smul _
  map_mul' _ _ := LinearEquiv.ext <| mul_smul _ _


/-- An additive equivalence whose underlying function preserves `smul` is a linear equivalence. -/
def toLinearEquiv (h : ∀ (c : R) (x), e (c • x) = c • e x) : M ≃ₗ[R] M₂ :=
  { e with map_smul' := h }


@[simp]
theorem coe_toLinearEquiv (h : ∀ (c : R) (x), e (c • x) = c • e x) : ⇑(e.toLinearEquiv h) = e :=
  rfl


@[simp]
theorem coe_toLinearEquiv_symm (h : ∀ (c : R) (x), e (c • x) = c • e x) :
    ⇑(e.toLinearEquiv h).symm = e.symm :=
  rfl


/-- An additive equivalence between commutative additive monoids is a linear equivalence between
ℕ-modules -/
def toNatLinearEquiv : M ≃ₗ[ℕ] M₂ :=
                               /-
                                 R : Type u_1
                                 R₂ : Type u_2
                                 K : Type u_3
                                 S : Type u_4
                                 M : Type u_5
                                 M₁ : Type u_6
                                 M₂ : Type u_7
                                 M₃ : Type u_8
                                 inst✝⁵ : Semiring R
                                 inst✝⁴ : AddCommMonoid M
                                 inst✝³ : AddCommMonoid M₂
                                 inst✝² : AddCommMonoid M₃
                                 inst✝¹ : Module R M
                                 inst✝ : Module R M₂
                                 e : AddEquiv M M₂
                                 c : Nat
                                 a : M
                                 ⊢ Eq (e (HSMul.hSMul c a)) (HSMul.hSMul c (e a))
                               -/
  e.toLinearEquiv fun c a ↦ by rw [map_nsmul]
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem coe_toNatLinearEquiv : ⇑e.toNatLinearEquiv = e :=
  rfl


@[simp]
theorem toNatLinearEquiv_toAddEquiv : ↑e.toNatLinearEquiv = e :=
  rfl


@[simp]
theorem _root_.LinearEquiv.toAddEquiv_toNatLinearEquiv (e : M ≃ₗ[ℕ] M₂) :
    AddEquiv.toNatLinearEquiv ↑e = e :=
  DFunLike.coe_injective rfl


@[simp]
theorem toNatLinearEquiv_symm : e.toNatLinearEquiv.symm = e.symm.toNatLinearEquiv :=
  rfl


@[simp]
theorem toNatLinearEquiv_refl : (AddEquiv.refl M).toNatLinearEquiv = LinearEquiv.refl ℕ M :=
  rfl


@[simp]
theorem toNatLinearEquiv_trans (e₂ : M₂ ≃+ M₃) :
    e.toNatLinearEquiv.trans e₂.toNatLinearEquiv = (e.trans e₂).toNatLinearEquiv :=
  rfl


/-- An additive equivalence between commutative additive groups is a linear
equivalence between ℤ-modules -/
def toIntLinearEquiv : M ≃ₗ[ℤ] M₂ :=
  e.toLinearEquiv fun c a ↦ e.toAddMonoidHom.map_zsmul a c


@[simp]
theorem coe_toIntLinearEquiv : ⇑e.toIntLinearEquiv = e :=
  rfl


@[simp]
theorem toIntLinearEquiv_toAddEquiv : ↑e.toIntLinearEquiv = e := by
  /-
    M : Type u_5
    M₂ : Type u_7
    inst✝¹ : AddCommGroup M
    inst✝ : AddCommGroup M₂
    e : AddEquiv M M₂
    ⊢ Eq (↑e.toIntLinearEquiv) e
  -/
  ext
  /-
    case h
    M : Type u_5
    M₂ : Type u_7
    inst✝¹ : AddCommGroup M
    inst✝ : AddCommGroup M₂
    e : AddEquiv M M₂
    x✝ : M
    ⊢ Eq (↑e.toIntLinearEquiv x✝) (e x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.LinearEquiv.toAddEquiv_toIntLinearEquiv (e : M ≃ₗ[ℤ] M₂) :
    AddEquiv.toIntLinearEquiv (e : M ≃+ M₂) = e :=
  DFunLike.coe_injective rfl


@[simp]
theorem toIntLinearEquiv_symm : e.toIntLinearEquiv.symm = e.symm.toIntLinearEquiv :=
  rfl


@[simp]
theorem toIntLinearEquiv_refl : (AddEquiv.refl M).toIntLinearEquiv = LinearEquiv.refl ℤ M :=
  rfl


@[simp]
theorem toIntLinearEquiv_trans (e₂ : M₂ ≃+ M₃) :
    e.toIntLinearEquiv.trans e₂.toIntLinearEquiv = (e.trans e₂).toIntLinearEquiv :=
  rfl


/-- The equivalence between R-linear maps from `R` to `M`, and points of `M` itself.
This says that the forgetful functor from `R`-modules to types is representable, by `R`.

This is an `S`-linear equivalence, under the assumption that `S` acts on `M` commuting with `R`.
When `R` is commutative, we can take this to be the usual action with `S = R`.
Otherwise, `S = ℕ` shows that the equivalence is additive.
See note [bundled maps over different rings].
-/
@[simps]
def ringLmapEquivSelf [Module S M] [SMulCommClass R S M] : (R →ₗ[R] M) ≃ₗ[S] M :=
  { applyₗ' S (1 : R) with
    toFun := fun f ↦ f 1
    invFun := smulRight (1 : R →ₗ[R] R)
    left_inv := fun f ↦ by
      /-
        R : Type u_1
        R₂ : Type u_2
        K : Type u_3
        S : Type u_4
        M : Type u_5
        M₁ : Type u_6
        M₂ : Type u_7
        M₃ : Type u_8
        inst✝⁵ : Semiring R
        inst✝⁴ : Semiring S
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : SMulCommClass R S M
        f : LinearMap (RingHom.id R) R M
        ⊢ Eq (LinearMap.smulRight 1 ({ toFun := fun f => f 1, map_add' := ⋯, map_smul' …
      -/
      ext
      /-
        case h
        R : Type u_1
        R₂ : Type u_2
        K : Type u_3
        S : Type u_4
        M : Type u_5
        M₁ : Type u_6
        M₂ : Type u_7
        M₃ : Type u_8
        inst✝⁵ : Semiring R
        inst✝⁴ : Semiring S
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        inst✝¹ : Module S M
        inst✝ : SMulCommClass R S M
        f : LinearMap (RingHom.id R) R M
        ⊢ Eq ((LinearMap.smulRight 1 ({ toFun := fun f => f 1, map_add' := ⋯, map_smul …
      -/
      simp only [coe_smulRight, one_apply, smul_eq_mul, ← map_smul f, mul_one]
      /-
        🎉 no goals
      -/
                            /-
                              R : Type u_1
                              R₂ : Type u_2
                              K : Type u_3
                              S : Type u_4
                              M : Type u_5
                              M₁ : Type u_6
                              M₂ : Type u_7
                              M₃ : Type u_8
                              inst✝⁵ : Semiring R
                              inst✝⁴ : Semiring S
                              inst✝³ : AddCommMonoid M
                              inst✝² : Module R M
                              inst✝¹ : Module S M
                              inst✝ : SMulCommClass R S M
                              x : M
                              ⊢ Eq ({ toFun := fun f => f 1, map_add' := ⋯, map_smul' := ⋯ }.toFun (LinearMa …
                            -/
    right_inv := fun x ↦ by simp }
                            /-
                              🎉 no goals
                            -/


/--
The `R`-linear equivalence between additive morphisms `A →+ B` and `ℕ`-linear morphisms `A →ₗ[ℕ] B`.
-/
@[simps]
def addMonoidHomLequivNat {A B : Type*} (R : Type*) [Semiring R] [AddCommMonoid A]
    [AddCommMonoid B] [Module R B] : (A →+ B) ≃ₗ[R] A →ₗ[ℕ] B
    where
  toFun := AddMonoidHom.toNatLinearMap
  invFun := LinearMap.toAddMonoidHom
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  left_inv _ := rfl
  right_inv _ := rfl


/--
The `R`-linear equivalence between additive morphisms `A →+ B` and `ℤ`-linear morphisms `A →ₗ[ℤ] B`.
-/
@[simps]
def addMonoidHomLequivInt {A B : Type*} (R : Type*) [Semiring R] [AddCommGroup A] [AddCommGroup B]
    [Module R B] : (A →+ B) ≃ₗ[R] A →ₗ[ℤ] B
    where
  toFun := AddMonoidHom.toIntLinearMap
  invFun := LinearMap.toAddMonoidHom
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  left_inv _ := rfl
  right_inv _ := rfl


/-- Ring equivalence between additive group endomorphisms of an `AddCommGroup` `A` and
`ℤ`-module endomorphisms of `A.` -/
@[simps] def addMonoidEndRingEquivInt (A : Type*) [AddCommGroup A] :
    AddMonoid.End A ≃+* Module.End ℤ A :=
  { addMonoidHomLequivInt (B := A) ℤ with
    map_mul' := fun _ _ ↦ rfl }


/-- Between two zero modules, the zero map is an equivalence. -/
instance : Zero (M ≃ₛₗ[σ₁₂] M₂) :=
  ⟨{ (0 : M →ₛₗ[σ₁₂] M₂) with
      toFun := 0
      invFun := 0
      right_inv := Subsingleton.elim _
      left_inv := Subsingleton.elim _ }⟩

-- Even though these are implied by `Subsingleton.elim` via the `Unique` instance below, they're
-- nice to have as `rfl`-lemmas for `dsimp`.

@[simp]
theorem zero_symm : (0 : M ≃ₛₗ[σ₁₂] M₂).symm = 0 :=
  rfl


@[simp]
theorem coe_zero : ⇑(0 : M ≃ₛₗ[σ₁₂] M₂) = 0 :=
  rfl


theorem zero_apply (x : M) : (0 : M ≃ₛₗ[σ₁₂] M₂) x = 0 :=
  rfl


/-- Between two zero modules, the zero map is the only equivalence. -/
instance : Unique (M ≃ₛₗ[σ₁₂] M₂) where
  uniq _ := toLinearMap_injective (Subsingleton.elim _ _)
  default := 0


instance uniqueOfSubsingleton [Subsingleton R] [Subsingleton R₂] : Unique (M ≃ₛₗ[σ₁₂] M₂) := by
  /-
    R : Type u_1
    R₂ : Type u_2
    K : Type u_3
    S : Type u_4
    M : Type u_5
    M₁ : Type u_6
    M₂ : Type u_7
    M₃ : Type u_8
    inst✝⁹ : Semiring R
    inst✝⁸ : Semiring R₂
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid M₂
    inst✝⁵ : Module R M
    inst✝⁴ : Module R₂ M₂
    σ₁₂ : RingHom R R₂
    σ₂₁ : RingHom R₂ R
    inst✝³ : RingHomInvPair σ₁₂ σ₂₁
    inst✝² : RingHomInvPair σ₂₁ σ₁₂
    inst✝¹ : Subsingleton R
    inst✝ : Subsingleton R₂
    ⊢ Unique (LinearEquiv σ₁₂ M M₂)
  -/
  haveI := Module.subsingleton R M
  /-
    R : Type u_1
    R₂ : Type u_2
    K : Type u_3
    S : Type u_4
    M : Type u_5
    M₁ : Type u_6
    M₂ : Type u_7
    M₃ : Type u_8
    inst✝⁹ : Semiring R
    inst✝⁸ : Semiring R₂
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid M₂
    inst✝⁵ : Module R M
    inst✝⁴ : Module R₂ M₂
    σ₁₂ : RingHom R R₂
    σ₂₁ : RingHom R₂ R
    inst✝³ : RingHomInvPair σ₁₂ σ₂₁
    inst✝² : RingHomInvPair σ₂₁ σ₁₂
    inst✝¹ : Subsingleton R
    inst✝ : Subsingleton R₂
    this : Subsingleton M
    ⊢ Unique (LinearEquiv σ₁₂ M M₂)
  -/
  haveI := Module.subsingleton R₂ M₂
  /-
    R : Type u_1
    R₂ : Type u_2
    K : Type u_3
    S : Type u_4
    M : Type u_5
    M₁ : Type u_6
    M₂ : Type u_7
    M₃ : Type u_8
    inst✝⁹ : Semiring R
    inst✝⁸ : Semiring R₂
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid M₂
    inst✝⁵ : Module R M
    inst✝⁴ : Module R₂ M₂
    σ₁₂ : RingHom R R₂
    σ₂₁ : RingHom R₂ R
    inst✝³ : RingHomInvPair σ₁₂ σ₂₁
    inst✝² : RingHomInvPair σ₂₁ σ₁₂
    inst✝¹ : Subsingleton R
    inst✝ : Subsingleton R₂
    this✝ : Subsingleton M
    this : Subsingleton M₂
    ⊢ Unique (LinearEquiv σ₁₂ M M₂)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Linear equivalence between a curried and uncurried function.
  Differs from `TensorProduct.curry`. -/
protected def curry : (V × V₂ → M) ≃ₗ[R] V → V₂ → M :=
  { Equiv.curry _ _ _ with
    map_add' := fun _ _ ↦ rfl
    map_smul' := fun _ _ ↦ rfl }


@[simp]
theorem coe_curry : ⇑(LinearEquiv.curry R M V V₂) = curry :=
  rfl


@[simp]
theorem coe_curry_symm : ⇑(LinearEquiv.curry R M V V₂).symm = uncurry :=
  rfl


/-- If a linear map has an inverse, it is a linear equivalence. -/
def ofLinear (h₁ : f.comp g = LinearMap.id) (h₂ : g.comp f = LinearMap.id) : M ≃ₛₗ[σ₁₂] M₂ :=
  { f with
    invFun := g
    left_inv := LinearMap.ext_iff.1 h₂
    right_inv := LinearMap.ext_iff.1 h₁ }


@[simp]
theorem ofLinear_apply {h₁ h₂} (x : M) : (ofLinear f g h₁ h₂ : M ≃ₛₗ[σ₁₂] M₂) x = f x :=
  rfl


@[simp]
theorem ofLinear_symm_apply {h₁ h₂} (x : M₂) : (ofLinear f g h₁ h₂ : M ≃ₛₗ[σ₁₂] M₂).symm x = g x :=
  rfl


@[simp]
theorem ofLinear_toLinearMap {h₁ h₂} : (ofLinear f g h₁ h₂ : M ≃ₛₗ[σ₁₂] M₂) = f := rfl


@[simp]
theorem ofLinear_symm_toLinearMap {h₁ h₂} : (ofLinear f g h₁ h₂ : M ≃ₛₗ[σ₁₂] M₂).symm = g := rfl


/-- `x ↦ -x` as a `LinearEquiv` -/
def neg : M ≃ₗ[R] M :=
  { Equiv.neg M, (-LinearMap.id : M →ₗ[R] M) with }


@[simp]
theorem coe_neg : ⇑(neg R : M ≃ₗ[R] M) = -id :=
  rfl


                                               /-
                                                 R : Type u_1
                                                 M : Type u_5
                                                 inst✝² : Semiring R
                                                 inst✝¹ : AddCommGroup M
                                                 inst✝ : Module R M
                                                 x : M
                                                 ⊢ Eq ((LinearEquiv.neg R) x) (Neg.neg x)
                                               -/
theorem neg_apply (x : M) : neg R x = -x := by simp
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem symm_neg : (neg R : M ≃ₗ[R] M).symm = neg R :=
  rfl


/-- Multiplying by a unit `a` of the ring `R` is a linear equivalence. -/
def smulOfUnit (a : Rˣ) : M ≃ₗ[R] M :=
  DistribMulAction.toLinearEquiv R M a


/-- A linear isomorphism between the domains and codomains of two spaces of linear maps gives a
linear isomorphism between the two function spaces. -/
def arrowCongr {R M₁ M₂ M₂₁ M₂₂ : Sort _} [CommSemiring R] [AddCommMonoid M₁] [AddCommMonoid M₂]
    [AddCommMonoid M₂₁] [AddCommMonoid M₂₂] [Module R M₁] [Module R M₂] [Module R M₂₁]
    [Module R M₂₂] (e₁ : M₁ ≃ₗ[R] M₂) (e₂ : M₂₁ ≃ₗ[R] M₂₂) : (M₁ →ₗ[R] M₂₁) ≃ₗ[R] M₂ →ₗ[R] M₂₂ where
  toFun := fun f : M₁ →ₗ[R] M₂₁ ↦ (e₂ : M₂₁ →ₗ[R] M₂₂).comp <| f.comp (e₁.symm : M₂ →ₗ[R] M₁)
  invFun f := (e₂.symm : M₂₂ →ₗ[R] M₂₁).comp <| f.comp (e₁ : M₁ →ₗ[R] M₂)
  left_inv f := by
    /-
      R✝ : Type u_1
      R₂ : Type u_2
      K : Type u_3
      S : Type u_4
      M : Type u_5
      M₁✝ : Type u_6
      M₂✝ : Type u_7
      M₃ : Type u_8
      inst✝¹⁵ : CommSemiring R✝
      inst✝¹⁴ : AddCommMonoid M
      inst✝¹³ : AddCommMonoid M₂✝
      inst✝¹² : AddCommMonoid M₃
      inst✝¹¹ : Module R✝ M
      inst✝¹⁰ : Module R✝ M₂✝
      inst✝⁹ : Module R✝ M₃
      R : Type ?u.89211
      M₁ : Type ?u.89214
      M₂ : Type ?u.89217
      M₂₁ : Type ?u.89220
      M₂₂ : Type ?u.89223
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M₁
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : AddCommMonoid M₂₁
      inst✝⁴ : AddCommMonoid M₂₂
      inst✝³ : Module R M₁
      inst✝² : Module R M₂
      inst✝¹ : Module R M₂₁
      inst✝ : Module R M₂₂
      e₁ : LinearEquiv (RingHom.id R) M₁ M₂
      e₂ : LinearEquiv (RingHom.id R) M₂₁ M₂₂
      f : LinearMap (RingHom.id R) M₁ M₂₁
      ⊢ Eq ((fun f => (↑e₂.symm).comp (f.comp ↑e₁)) ({ toFun := fun f => (↑e₂).comp  …
    -/
    ext x
    /-
      case h
      R✝ : Type u_1
      R₂ : Type u_2
      K : Type u_3
      S : Type u_4
      M : Type u_5
      M₁✝ : Type u_6
      M₂✝ : Type u_7
      M₃ : Type u_8
      inst✝¹⁵ : CommSemiring R✝
      inst✝¹⁴ : AddCommMonoid M
      inst✝¹³ : AddCommMonoid M₂✝
      inst✝¹² : AddCommMonoid M₃
      inst✝¹¹ : Module R✝ M
      inst✝¹⁰ : Module R✝ M₂✝
      inst✝⁹ : Module R✝ M₃
      R : Type ?u.89211
      M₁ : Type ?u.89214
      M₂ : Type ?u.89217
      M₂₁ : Type ?u.89220
      M₂₂ : Type ?u.89223
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M₁
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : AddCommMonoid M₂₁
      inst✝⁴ : AddCommMonoid M₂₂
      inst✝³ : Module R M₁
      inst✝² : Module R M₂
      inst✝¹ : Module R M₂₁
      inst✝ : Module R M₂₂
      e₁ : LinearEquiv (RingHom.id R) M₁ M₂
      e₂ : LinearEquiv (RingHom.id R) M₂₁ M₂₂
      f : LinearMap (RingHom.id R) M₁ M₂₁
      x : M₁
      ⊢ Eq (((fun f => (↑e₂.symm).comp (f.comp ↑e₁)) ({ toFun := fun f => (↑e₂).comp …
    -/
    simp only [symm_apply_apply, Function.comp_apply, coe_comp, coe_coe]
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      R✝ : Type u_1
      R₂ : Type u_2
      K : Type u_3
      S : Type u_4
      M : Type u_5
      M₁✝ : Type u_6
      M₂✝ : Type u_7
      M₃ : Type u_8
      inst✝¹⁵ : CommSemiring R✝
      inst✝¹⁴ : AddCommMonoid M
      inst✝¹³ : AddCommMonoid M₂✝
      inst✝¹² : AddCommMonoid M₃
      inst✝¹¹ : Module R✝ M
      inst✝¹⁰ : Module R✝ M₂✝
      inst✝⁹ : Module R✝ M₃
      R : Type ?u.89211
      M₁ : Type ?u.89214
      M₂ : Type ?u.89217
      M₂₁ : Type ?u.89220
      M₂₂ : Type ?u.89223
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M₁
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : AddCommMonoid M₂₁
      inst✝⁴ : AddCommMonoid M₂₂
      inst✝³ : Module R M₁
      inst✝² : Module R M₂
      inst✝¹ : Module R M₂₁
      inst✝ : Module R M₂₂
      e₁ : LinearEquiv (RingHom.id R) M₁ M₂
      e₂ : LinearEquiv (RingHom.id R) M₂₁ M₂₂
      f g : LinearMap (RingHom.id R) M₁ M₂₁
      ⊢ Eq ((fun f => (↑e₂).comp (f.comp ↑e₁.symm)) (HAdd.hAdd f g)) (HAdd.hAdd ((fu …
    -/
    /-
      R✝ : Type u_1
      R₂ : Type u_2
      K : Type u_3
      S : Type u_4
      M : Type u_5
      M₁✝ : Type u_6
      M₂✝ : Type u_7
      M₃ : Type u_8
      inst✝¹⁵ : CommSemiring R✝
      inst✝¹⁴ : AddCommMonoid M
      inst✝¹³ : AddCommMonoid M₂✝
      inst✝¹² : AddCommMonoid M₃
      inst✝¹¹ : Module R✝ M
      inst✝¹⁰ : Module R✝ M₂✝
      inst✝⁹ : Module R✝ M₃
      R : Type ?u.89211
      M₁ : Type ?u.89214
      M₂ : Type ?u.89217
      M₂₁ : Type ?u.89220
      M₂₂ : Type ?u.89223
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M₁
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : AddCommMonoid M₂₁
      inst✝⁴ : AddCommMonoid M₂₂
      inst✝³ : Module R M₁
      inst✝² : Module R M₂
      inst✝¹ : Module R M₂₁
      inst✝ : Module R M₂₂
      e₁ : LinearEquiv (RingHom.id R) M₁ M₂
      e₂ : LinearEquiv (RingHom.id R) M₂₁ M₂₂
      f : LinearMap (RingHom.id R) M₂ M₂₂
      ⊢ Eq ({ toFun := fun f => (↑e₂).comp (f.comp ↑e₁.symm), map_add' := ⋯, map_smu …
    -/
    /-
      case h
      R✝ : Type u_1
      R₂ : Type u_2
      K : Type u_3
      S : Type u_4
      M : Type u_5
      M₁✝ : Type u_6
      M₂✝ : Type u_7
      M₃ : Type u_8
      inst✝¹⁵ : CommSemiring R✝
      inst✝¹⁴ : AddCommMonoid M
      inst✝¹³ : AddCommMonoid M₂✝
      inst✝¹² : AddCommMonoid M₃
      inst✝¹¹ : Module R✝ M
      inst✝¹⁰ : Module R✝ M₂✝
      inst✝⁹ : Module R✝ M₃
      R : Type ?u.89211
      M₁ : Type ?u.89214
      M₂ : Type ?u.89217
      M₂₁ : Type ?u.89220
      M₂₂ : Type ?u.89223
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M₁
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : AddCommMonoid M₂₁
      inst✝⁴ : AddCommMonoid M₂₂
      inst✝³ : Module R M₁
      inst✝² : Module R M₂
      inst✝¹ : Module R M₂₁
      inst✝ : Module R M₂₂
      e₁ : LinearEquiv (RingHom.id R) M₁ M₂
      e₂ : LinearEquiv (RingHom.id R) M₂₁ M₂₂
      f g : LinearMap (RingHom.id R) M₁ M₂₁
      x : M₂
      ⊢ Eq (((fun f => (↑e₂).comp (f.comp ↑e₁.symm)) (HAdd.hAdd f g)) x) ((HAdd.hAdd …
    -/
    ext x
    /-
      🎉 no goals
    -/
    /-
      case h
      R✝ : Type u_1
      R₂ : Type u_2
      K : Type u_3
      S : Type u_4
      M : Type u_5
      M₁✝ : Type u_6
      M₂✝ : Type u_7
      M₃ : Type u_8
      inst✝¹⁵ : CommSemiring R✝
      inst✝¹⁴ : AddCommMonoid M
      inst✝¹³ : AddCommMonoid M₂✝
      inst✝¹² : AddCommMonoid M₃
      inst✝¹¹ : Module R✝ M
      inst✝¹⁰ : Module R✝ M₂✝
      inst✝⁹ : Module R✝ M₃
      R : Type ?u.89211
      M₁ : Type ?u.89214
      M₂ : Type ?u.89217
      M₂₁ : Type ?u.89220
      M₂₂ : Type ?u.89223
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M₁
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : AddCommMonoid M₂₁
      inst✝⁴ : AddCommMonoid M₂₂
      inst✝³ : Module R M₁
      inst✝² : Module R M₂
      inst✝¹ : Module R M₂₁
      inst✝ : Module R M₂₂
      e₁ : LinearEquiv (RingHom.id R) M₁ M₂
      e₂ : LinearEquiv (RingHom.id R) M₂₁ M₂₂
      f : LinearMap (RingHom.id R) M₂ M₂₂
      x : M₂
      ⊢ Eq (({ toFun := fun f => (↑e₂).comp (f.comp ↑e₁.symm), map_add' := ⋯, map_sm …
    -/
    /-
      R✝ : Type u_1
      R₂ : Type u_2
      K : Type u_3
      S : Type u_4
      M : Type u_5
      M₁✝ : Type u_6
      M₂✝ : Type u_7
      M₃ : Type u_8
      inst✝¹⁵ : CommSemiring R✝
      inst✝¹⁴ : AddCommMonoid M
      inst✝¹³ : AddCommMonoid M₂✝
      inst✝¹² : AddCommMonoid M₃
      inst✝¹¹ : Module R✝ M
      inst✝¹⁰ : Module R✝ M₂✝
      inst✝⁹ : Module R✝ M₃
      R : Type ?u.89211
      M₁ : Type ?u.89214
      M₂ : Type ?u.89217
      M₂₁ : Type ?u.89220
      M₂₂ : Type ?u.89223
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M₁
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : AddCommMonoid M₂₁
      inst✝⁴ : AddCommMonoid M₂₂
      inst✝³ : Module R M₁
      inst✝² : Module R M₂
      inst✝¹ : Module R M₂₁
      inst✝ : Module R M₂₂
      e₁ : LinearEquiv (RingHom.id R) M₁ M₂
      e₂ : LinearEquiv (RingHom.id R) M₂₁ M₂₂
      c : R
      f : LinearMap (RingHom.id R) M₁ M₂₁
      ⊢ Eq ({ toFun := fun f => (↑e₂).comp (f.comp ↑e₁.symm), map_add' := ⋯ }.toFun  …
    -/
    simp only [Function.comp_apply, apply_symm_apply, coe_comp, coe_coe]
    /-
      case h
      R✝ : Type u_1
      R₂ : Type u_2
      K : Type u_3
      S : Type u_4
      M : Type u_5
      M₁✝ : Type u_6
      M₂✝ : Type u_7
      M₃ : Type u_8
      inst✝¹⁵ : CommSemiring R✝
      inst✝¹⁴ : AddCommMonoid M
      inst✝¹³ : AddCommMonoid M₂✝
      inst✝¹² : AddCommMonoid M₃
      inst✝¹¹ : Module R✝ M
      inst✝¹⁰ : Module R✝ M₂✝
      inst✝⁹ : Module R✝ M₃
      R : Type ?u.89211
      M₁ : Type ?u.89214
      M₂ : Type ?u.89217
      M₂₁ : Type ?u.89220
      M₂₂ : Type ?u.89223
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M₁
      inst✝⁶ : AddCommMonoid M₂
      inst✝⁵ : AddCommMonoid M₂₁
      inst✝⁴ : AddCommMonoid M₂₂
      inst✝³ : Module R M₁
      inst✝² : Module R M₂
      inst✝¹ : Module R M₂₁
      inst✝ : Module R M₂₂
      e₁ : LinearEquiv (RingHom.id R) M₁ M₂
      e₂ : LinearEquiv (RingHom.id R) M₂₁ M₂₂
      c : R
      f : LinearMap (RingHom.id R) M₁ M₂₁
      x : M₂
      ⊢ Eq (({ toFun := fun f => (↑e₂).comp (f.comp ↑e₁.symm), map_add' := ⋯ }.toFun …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  map_add' f g := by
    ext x
    simp only [map_add, add_apply, Function.comp_apply, coe_comp, coe_coe]
  map_smul' c f := by
    ext x
    simp only [smul_apply, Function.comp_apply, coe_comp, map_smulₛₗ e₂, coe_coe]


@[simp]
theorem arrowCongr_apply {R M₁ M₂ M₂₁ M₂₂ : Sort _} [CommSemiring R] [AddCommMonoid M₁]
    [AddCommMonoid M₂] [AddCommMonoid M₂₁] [AddCommMonoid M₂₂] [Module R M₁] [Module R M₂]
    [Module R M₂₁] [Module R M₂₂] (e₁ : M₁ ≃ₗ[R] M₂) (e₂ : M₂₁ ≃ₗ[R] M₂₂) (f : M₁ →ₗ[R] M₂₁)
    (x : M₂) : arrowCongr e₁ e₂ f x = e₂ (f (e₁.symm x)) :=
  rfl


@[simp]
theorem arrowCongr_symm_apply {R M₁ M₂ M₂₁ M₂₂ : Sort _} [CommSemiring R] [AddCommMonoid M₁]
    [AddCommMonoid M₂] [AddCommMonoid M₂₁] [AddCommMonoid M₂₂] [Module R M₁] [Module R M₂]
    [Module R M₂₁] [Module R M₂₂] (e₁ : M₁ ≃ₗ[R] M₂) (e₂ : M₂₁ ≃ₗ[R] M₂₂) (f : M₂ →ₗ[R] M₂₂)
    (x : M₁) : (arrowCongr e₁ e₂).symm f x = e₂.symm (f (e₁ x)) :=
  rfl


theorem arrowCongr_comp {N N₂ N₃ : Sort _} [AddCommMonoid N] [AddCommMonoid N₂] [AddCommMonoid N₃]
    [Module R N] [Module R N₂] [Module R N₃] (e₁ : M ≃ₗ[R] N) (e₂ : M₂ ≃ₗ[R] N₂) (e₃ : M₃ ≃ₗ[R] N₃)
    (f : M →ₗ[R] M₂) (g : M₂ →ₗ[R] M₃) :
    arrowCongr e₁ e₃ (g.comp f) = (arrowCongr e₂ e₃ g).comp (arrowCongr e₁ e₂ f) := by
  /-
    R : Type u_1
    M : Type u_5
    M₂ : Type u_7
    M₃ : Type u_8
    inst✝¹² : CommSemiring R
    inst✝¹¹ : AddCommMonoid M
    inst✝¹⁰ : AddCommMonoid M₂
    inst✝⁹ : AddCommMonoid M₃
    inst✝⁸ : Module R M
    inst✝⁷ : Module R M₂
    inst✝⁶ : Module R M₃
    N : Type u_9
    N₂ : Type u_10
    N₃ : Type u_11
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid N₂
    inst✝³ : AddCommMonoid N₃
    inst✝² : Module R N
    inst✝¹ : Module R N₂
    inst✝ : Module R N₃
    e₁ : LinearEquiv (RingHom.id R) M N
    e₂ : LinearEquiv (RingHom.id R) M₂ N₂
    e₃ : LinearEquiv (RingHom.id R) M₃ N₃
    f : LinearMap (RingHom.id R) M M₂
    g : LinearMap (RingHom.id R) M₂ M₃
    ⊢ Eq ((e₁.arrowCongr e₃) (g.comp f)) (((e₂.arrowCongr e₃) g).comp ((e₁.arrowCo …
  -/
  ext
  /-
    case h
    R : Type u_1
    M : Type u_5
    M₂ : Type u_7
    M₃ : Type u_8
    inst✝¹² : CommSemiring R
    inst✝¹¹ : AddCommMonoid M
    inst✝¹⁰ : AddCommMonoid M₂
    inst✝⁹ : AddCommMonoid M₃
    inst✝⁸ : Module R M
    inst✝⁷ : Module R M₂
    inst✝⁶ : Module R M₃
    N : Type u_9
    N₂ : Type u_10
    N₃ : Type u_11
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid N₂
    inst✝³ : AddCommMonoid N₃
    inst✝² : Module R N
    inst✝¹ : Module R N₂
    inst✝ : Module R N₃
    e₁ : LinearEquiv (RingHom.id R) M N
    e₂ : LinearEquiv (RingHom.id R) M₂ N₂
    e₃ : LinearEquiv (RingHom.id R) M₃ N₃
    f : LinearMap (RingHom.id R) M M₂
    g : LinearMap (RingHom.id R) M₂ M₃
    x✝ : N
    ⊢ Eq (((e₁.arrowCongr e₃) (g.comp f)) x✝) ((((e₂.arrowCongr e₃) g).comp ((e₁.a …
  -/
  simp only [symm_apply_apply, arrowCongr_apply, LinearMap.comp_apply]
  /-
    🎉 no goals
  -/


theorem arrowCongr_trans {M₁ M₂ M₃ N₁ N₂ N₃ : Sort _} [AddCommMonoid M₁] [Module R M₁]
    [AddCommMonoid M₂] [Module R M₂] [AddCommMonoid M₃] [Module R M₃] [AddCommMonoid N₁]
    [Module R N₁] [AddCommMonoid N₂] [Module R N₂] [AddCommMonoid N₃] [Module R N₃]
    (e₁ : M₁ ≃ₗ[R] M₂) (e₂ : N₁ ≃ₗ[R] N₂) (e₃ : M₂ ≃ₗ[R] M₃) (e₄ : N₂ ≃ₗ[R] N₃) :
    (arrowCongr e₁ e₂).trans (arrowCongr e₃ e₄) = arrowCongr (e₁.trans e₃) (e₂.trans e₄) :=
  rfl


/-- If `M₂` and `M₃` are linearly isomorphic then the two spaces of linear maps from `M` into `M₂`
and `M` into `M₃` are linearly isomorphic. -/
def congrRight (f : M₂ ≃ₗ[R] M₃) : (M →ₗ[R] M₂) ≃ₗ[R] M →ₗ[R] M₃ :=
  arrowCongr (LinearEquiv.refl R M) f


/-- If `M` and `M₂` are linearly isomorphic then the two spaces of linear maps from `M` and `M₂` to
themselves are linearly isomorphic. -/
def conj (e : M ≃ₗ[R] M₂) : Module.End R M ≃ₗ[R] Module.End R M₂ :=
  arrowCongr e e


theorem conj_apply (e : M ≃ₗ[R] M₂) (f : Module.End R M) :
    e.conj f = ((↑e : M →ₗ[R] M₂).comp f).comp (e.symm : M₂ →ₗ[R] M) :=
  rfl


theorem conj_apply_apply (e : M ≃ₗ[R] M₂) (f : Module.End R M) (x : M₂) :
    e.conj f x = e (f (e.symm x)) :=
  rfl


theorem symm_conj_apply (e : M ≃ₗ[R] M₂) (f : Module.End R M₂) :
    e.symm.conj f = ((↑e.symm : M₂ →ₗ[R] M).comp f).comp (e : M →ₗ[R] M₂) :=
  rfl


theorem conj_comp (e : M ≃ₗ[R] M₂) (f g : Module.End R M) :
    e.conj (g.comp f) = (e.conj g).comp (e.conj f) :=
  arrowCongr_comp e e e f g


theorem conj_trans (e₁ : M ≃ₗ[R] M₂) (e₂ : M₂ ≃ₗ[R] M₃) :
    e₁.conj.trans e₂.conj = (e₁.trans e₂).conj :=
  rfl


@[simp]
theorem conj_id (e : M ≃ₗ[R] M₂) : e.conj LinearMap.id = LinearMap.id := by
  /-
    R : Type u_1
    M : Type u_5
    M₂ : Type u_7
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    e : LinearEquiv (RingHom.id R) M M₂
    ⊢ Eq (e.conj LinearMap.id) LinearMap.id
  -/
  ext
  /-
    case h
    R : Type u_1
    M : Type u_5
    M₂ : Type u_7
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    inst✝¹ : Module R M
    inst✝ : Module R M₂
    e : LinearEquiv (RingHom.id R) M M₂
    x✝ : M₂
    ⊢ Eq ((e.conj LinearMap.id) x✝) (LinearMap.id x✝)
  -/
  simp [conj_apply]
  /-
    🎉 no goals
  -/


variable (M) in
/-- An `R`-linear isomorphism between two `R`-modules `M₂` and `M₃` induces an `S`-linear
isomorphism between `M₂ →ₗ[R] M` and `M₃ →ₗ[R] M`, if `M` is both an `R`-module and an
`S`-module and their actions commute. -/
def congrLeft {R} (S) [Semiring R] [Semiring S] [Module R M₂] [Module R M₃] [Module R M]
    [Module S M] [SMulCommClass R S M] (e : M₂ ≃ₗ[R] M₃) : (M₂ →ₗ[R] M) ≃ₗ[S] (M₃ →ₗ[R] M) where
  toFun f := f.comp e.symm.toLinearMap
  invFun f := f.comp e.toLinearMap
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
                   /-
                     R✝ : Type u_1
                     R₂ : Type u_2
                     K : Type u_3
                     S✝ : Type u_4
                     M : Type u_5
                     M₁ : Type u_6
                     M₂ : Type u_7
                     M₃ : Type u_8
                     inst✝¹³ : CommSemiring R✝
                     inst✝¹² : AddCommMonoid M
                     inst✝¹¹ : AddCommMonoid M₂
                     inst✝¹⁰ : AddCommMonoid M₃
                     inst✝⁹ : Module R✝ M
                     inst✝⁸ : Module R✝ M₂
                     inst✝⁷ : Module R✝ M₃
                     R : Type ?u.121321
                     S : Type ?u.121324
                     inst✝⁶ : Semiring R
                     inst✝⁵ : Semiring S
                     inst✝⁴ : Module R M₂
                     inst✝³ : Module R M₃
                     inst✝² : Module R M
                     inst✝¹ : Module S M
                     inst✝ : SMulCommClass R S M
                     e : LinearEquiv (RingHom.id R) M₂ M₃
                     f : LinearMap (RingHom.id R) M₂ M
                     ⊢ Eq ((fun f => f.comp ↑e) ({ toFun := fun f => f.comp ↑e.symm, map_add' := ⋯, …
                   -/
  left_inv f := by dsimp only; apply DFunLike.ext; exact (congr_arg f <| e.left_inv ·)
                                                   /-
                                                     🎉 no goals
                                                   -/
                    /-
                      R✝ : Type u_1
                      R₂ : Type u_2
                      K : Type u_3
                      S✝ : Type u_4
                      M : Type u_5
                      M₁ : Type u_6
                      M₂ : Type u_7
                      M₃ : Type u_8
                      inst✝¹³ : CommSemiring R✝
                      inst✝¹² : AddCommMonoid M
                      inst✝¹¹ : AddCommMonoid M₂
                      inst✝¹⁰ : AddCommMonoid M₃
                      inst✝⁹ : Module R✝ M
                      inst✝⁸ : Module R✝ M₂
                      inst✝⁷ : Module R✝ M₃
                      R : Type ?u.121321
                      S : Type ?u.121324
                      inst✝⁶ : Semiring R
                      inst✝⁵ : Semiring S
                      inst✝⁴ : Module R M₂
                      inst✝³ : Module R M₃
                      inst✝² : Module R M
                      inst✝¹ : Module S M
                      inst✝ : SMulCommClass R S M
                      e : LinearEquiv (RingHom.id R) M₂ M₃
                      f : LinearMap (RingHom.id R) M₃ M
                      ⊢ Eq ({ toFun := fun f => f.comp ↑e.symm, map_add' := ⋯, map_smul' := ⋯ }.toFu …
                    -/
  right_inv f := by dsimp only; apply DFunLike.ext; exact (congr_arg f <| e.right_inv ·)
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- Multiplying by a nonzero element `a` of the field `K` is a linear equivalence. -/
@[simps!]
def smulOfNeZero (a : K) (ha : a ≠ 0) : M ≃ₗ[K] M :=
  smulOfUnit <| Units.mk0 a ha


/-- An equivalence whose underlying function is linear is a linear equivalence. -/
def toLinearEquiv (e : M ≃ M₂) (h : IsLinearMap R (e : M → M₂)) : M ≃ₗ[R] M₂ :=
  { e, h.mk' e with }


/-- Given an `R`-module `M` and a function `m → n` between arbitrary types,
construct a linear map `(n → M) →ₗ[R] (m → M)` -/
def funLeft (f : m → n) : (n → M) →ₗ[R] m → M where
  toFun := (· ∘ f)
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


@[simp]
theorem funLeft_apply (f : m → n) (g : n → M) (i : m) : funLeft R M f g i = g (f i) :=
  rfl


@[simp]
theorem funLeft_id (g : n → M) : funLeft R M _root_.id g = g :=
  rfl


theorem funLeft_comp (f₁ : n → p) (f₂ : m → n) :
    funLeft R M (f₁ ∘ f₂) = (funLeft R M f₂).comp (funLeft R M f₁) :=
  rfl


theorem funLeft_surjective_of_injective (f : m → n) (hf : Injective f) :
    Surjective (funLeft R M f) := by
  classical
    intro g
    refine ⟨fun x ↦ if h : ∃ y, f y = x then g h.choose else 0, ?_⟩
    ext
    dsimp only [funLeft_apply]
    split_ifs with w
    · congr
      exact hf w.choose_spec
    · simp only [not_true, exists_apply_eq_apply] at w


theorem funLeft_injective_of_surjective (f : m → n) (hf : Surjective f) :
    Injective (funLeft R M f) := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    m : Type u_9
    n : Type u_10
    f : m → n
    hf : Function.Surjective f
    ⊢ Function.Injective ⇑(LinearMap.funLeft R M f)
  -/
  obtain ⟨g, hg⟩ := hf.hasRightInverse
  /-
    case intro
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    m : Type u_9
    n : Type u_10
    f : m → n
    hf : Function.Surjective f
    g : n → m
    hg : Function.RightInverse g f
    ⊢ Function.Injective ⇑(LinearMap.funLeft R M f)
  -/
  suffices LeftInverse (funLeft R M g) (funLeft R M f) by exact this.injective
  /-
    case intro
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    m : Type u_9
    n : Type u_10
    f : m → n
    hf : Function.Surjective f
    g : n → m
    hg : Function.RightInverse g f
    ⊢ Function.LeftInverse ⇑(LinearMap.funLeft R M g) ⇑(LinearMap.funLeft R M f)
  -/
  intro x
  /-
    case intro
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    m : Type u_9
    n : Type u_10
    f : m → n
    hf : Function.Surjective f
    g : n → m
    hg : Function.RightInverse g f
    x : n → M
    ⊢ Eq ((LinearMap.funLeft R M g) ((LinearMap.funLeft R M f) x)) x
  -/
  rw [← LinearMap.comp_apply, ← funLeft_comp, hg.id, funLeft_id]
  /-
    🎉 no goals
  -/


/-- Given an `R`-module `M` and an equivalence `m ≃ n` between arbitrary types,
construct a linear equivalence `(n → M) ≃ₗ[R] (m → M)` -/
def funCongrLeft (e : m ≃ n) : (n → M) ≃ₗ[R] m → M :=
  LinearEquiv.ofLinear (funLeft R M e) (funLeft R M e.symm)
    (LinearMap.ext fun x ↦
                        /-
                          R : Type u_1
                          R₂ : Type u_2
                          K : Type u_3
                          S : Type u_4
                          M : Type u_5
                          M₁ : Type u_6
                          M₂ : Type u_7
                          M₃ : Type u_8
                          inst✝² : Semiring R
                          inst✝¹ : AddCommMonoid M
                          inst✝ : Module R M
                          m : Type u_9
                          n : Type u_10
                          p : Type u_11
                          e : Equiv m n
                          x : m → M
                          i : m
                          ⊢ Eq (((LinearMap.funLeft R M ⇑e).comp (LinearMap.funLeft R M ⇑e.symm)) x i) ( …
                        -/
      funext fun i ↦ by rw [id_apply, ← funLeft_comp, Equiv.symm_comp_self, LinearMap.funLeft_id])
                        /-
                          🎉 no goals
                        -/
    (LinearMap.ext fun x ↦
                        /-
                          R : Type u_1
                          R₂ : Type u_2
                          K : Type u_3
                          S : Type u_4
                          M : Type u_5
                          M₁ : Type u_6
                          M₂ : Type u_7
                          M₃ : Type u_8
                          inst✝² : Semiring R
                          inst✝¹ : AddCommMonoid M
                          inst✝ : Module R M
                          m : Type u_9
                          n : Type u_10
                          p : Type u_11
                          e : Equiv m n
                          x : n → M
                          i : n
                          ⊢ Eq (((LinearMap.funLeft R M ⇑e.symm).comp (LinearMap.funLeft R M ⇑e)) x i) ( …
                        -/
      funext fun i ↦ by rw [id_apply, ← funLeft_comp, Equiv.self_comp_symm, LinearMap.funLeft_id])
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem funCongrLeft_apply (e : m ≃ n) (x : n → M) : funCongrLeft R M e x = funLeft R M e x :=
  rfl


@[simp]
theorem funCongrLeft_id : funCongrLeft R M (Equiv.refl n) = LinearEquiv.refl R (n → M) :=
  rfl


@[simp]
theorem funCongrLeft_comp (e₁ : m ≃ n) (e₂ : n ≃ p) :
    funCongrLeft R M (Equiv.trans e₁ e₂) =
      LinearEquiv.trans (funCongrLeft R M e₂) (funCongrLeft R M e₁) :=
  rfl


@[simp]
theorem funCongrLeft_symm (e : m ≃ n) : (funCongrLeft R M e).symm = funCongrLeft R M e.symm :=
  rfl


/-- The product over `S ⊕ T` of a family of modules is isomorphic to the product of
(the product over `S`) and (the product over `T`).

This is `Equiv.sumPiEquivProdPi` as a `LinearEquiv`.
-/
def sumPiEquivProdPi (R : Type*) [Semiring R] (S T : Type*) (A : S ⊕ T → Type*)
    [∀ st, AddCommMonoid (A st)] [∀ st, Module R (A st)] :
    (Π (st : S ⊕ T), A st) ≃ₗ[R] (Π (s : S), A (.inl s)) × (Π (t : T), A (.inr t)) where
  __ := Equiv.sumPiEquivProdPi _
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


/-- The product `Π t : α, f t` of a family of modules is linearly isomorphic to the module
`f ⬝` when `α` only contains `⬝`.

This is `Equiv.piUnique` as a `LinearEquiv`.
-/
@[simps (config := .asFn)]
def piUnique {α : Type*} [Unique α] (R : Type*) [Semiring R] (f : α → Type*)
    [∀ x, AddCommMonoid (f x)] [∀ x, Module R (f x)] : (Π t : α, f t) ≃ₗ[R] f default where
  __ := Equiv.piUnique _
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


