/-- A map `f` between modules over a semiring is linear if it satisfies the two properties
`f (x + y) = f x + f y` and `f (c • x) = c • f x`. The predicate `IsLinearMap R f` asserts this
property. A bundled version is available with `LinearMap`, and should be favored over
`IsLinearMap` most of the time. -/
structure IsLinearMap (R : Type u) {M : Type v} {M₂ : Type w} [Semiring R] [AddCommMonoid M]
  [AddCommMonoid M₂] [Module R M] [Module R M₂] (f : M → M₂) : Prop where
  /-- A linear map preserves addition. -/
  map_add : ∀ x y, f (x + y) = f x + f y
  /-- A linear map preserves scalar multiplication. -/
  map_smul : ∀ (c : R) (x), f (c • x) = c • f x


/-- A map `f` between an `R`-module and an `S`-module over a ring homomorphism `σ : R →+* S`
is semilinear if it satisfies the two properties `f (x + y) = f x + f y` and
`f (c • x) = (σ c) • f x`. Elements of `LinearMap σ M M₂` (available under the notation
`M →ₛₗ[σ] M₂`) are bundled versions of such maps. For plain linear maps (i.e. for which
`σ = RingHom.id R`), the notation `M →ₗ[R] M₂` is available. An unbundled version of plain linear
maps is available with the predicate `IsLinearMap`, but it should be avoided most of the time. -/
structure LinearMap {R S : Type*} [Semiring R] [Semiring S] (σ : R →+* S) (M : Type*)
    (M₂ : Type*) [AddCommMonoid M] [AddCommMonoid M₂] [Module R M] [Module S M₂] extends
    AddHom M M₂, MulActionHom σ M M₂


/-- `M →ₛₗ[σ] N` is the type of `σ`-semilinear maps from `M` to `N`. -/
notation:25 M " →ₛₗ[" σ:25 "] " M₂:0 => LinearMap σ M M₂


/-- `M →ₗ[R] N` is the type of `R`-linear maps from `M` to `N`. -/
notation:25 M " →ₗ[" R:25 "] " M₂:0 => LinearMap (RingHom.id R) M M₂


/-- `SemilinearMapClass F σ M M₂` asserts `F` is a type of bundled `σ`-semilinear maps `M → M₂`.

See also `LinearMapClass F R M M₂` for the case where `σ` is the identity map on `R`.

A map `f` between an `R`-module and an `S`-module over a ring homomorphism `σ : R →+* S`
is semilinear if it satisfies the two properties `f (x + y) = f x + f y` and
`f (c • x) = (σ c) • f x`. -/
class SemilinearMapClass (F : Type*) {R S : outParam Type*} [Semiring R] [Semiring S]
  (σ : outParam (R →+* S)) (M M₂ : outParam Type*) [AddCommMonoid M] [AddCommMonoid M₂]
    [Module R M] [Module S M₂] [FunLike F M M₂]
    extends AddHomClass F M M₂, MulActionSemiHomClass F σ M M₂ : Prop


/-- `LinearMapClass F R M M₂` asserts `F` is a type of bundled `R`-linear maps `M → M₂`.

This is an abbreviation for `SemilinearMapClass F (RingHom.id R) M M₂`.
-/
abbrev LinearMapClass (F : Type*) (R : outParam Type*) (M M₂ : Type*)
    [Semiring R] [AddCommMonoid M] [AddCommMonoid M₂] [Module R M] [Module R M₂]
    [FunLike F M M₂] :=
  SemilinearMapClass F (RingHom.id R) M M₂


protected lemma LinearMapClass.map_smul {R M M₂ : outParam Type*} [Semiring R] [AddCommMonoid M]
    [AddCommMonoid M₂] [Module R M] [Module R M₂]
    {F : Type*} [FunLike F M M₂] [LinearMapClass F R M M₂] (f : F) (r : R) (x : M) :
                              /-
                                R : outParam (Type u_14)
                                M : outParam (Type u_15)
                                M₂ : outParam (Type u_16)
                                inst✝⁶ : Semiring R
                                inst✝⁵ : AddCommMonoid M
                                inst✝⁴ : AddCommMonoid M₂
                                inst✝³ : Module R M
                                inst✝² : Module R M₂
                                F : Type u_17
                                inst✝¹ : FunLike F M M₂
                                inst✝ : LinearMapClass F R M M₂
                                f : F
                                r : R
                                x : M
                                ⊢ Eq (f (HSMul.hSMul r x)) (HSMul.hSMul r (f x))
                              -/
    f (r • x) = r • f x := by rw [_root_.map_smul]
                              /-
                                🎉 no goals
                              -/


instance (priority := 100) instAddMonoidHomClass [FunLike F M M₃] [SemilinearMapClass F σ M M₃] :
    AddMonoidHomClass F M M₃ :=
  { SemilinearMapClass.toAddHomClass with
    map_zero := fun f ↦
      show f 0 = 0 by
        /-
          R : Type u_1
          R₁ : Type u_2
          R₂ : Type u_3
          R₃ : Type u_4
          S : Type u_5
          S₃ : Type u_6
          T : Type u_7
          M : Type u_8
          M₁ : Type u_9
          M₂ : Type u_10
          M₃ : Type u_11
          N₂ : Type u_12
          N₃ : Type u_13
          F : Type u_14
          inst✝¹⁰ : Semiring R
          inst✝⁹ : Semiring S
          inst✝⁸ : AddCommMonoid M
          inst✝⁷ : AddCommMonoid M₁
          inst✝⁶ : AddCommMonoid M₂
          inst✝⁵ : AddCommMonoid M₃
          inst✝⁴ : Module R M
          inst✝³ : Module R M₂
          inst✝² : Module S M₃
          σ : RingHom R S
          inst✝¹ : FunLike F M M₃
          inst✝ : SemilinearMapClass F σ M M₃
          f : F
          ⊢ Eq (f 0) 0
        -/
        rw [← zero_smul R (0 : M), map_smulₛₗ]
        /-
          R : Type u_1
          R₁ : Type u_2
          R₂ : Type u_3
          R₃ : Type u_4
          S : Type u_5
          S₃ : Type u_6
          T : Type u_7
          M : Type u_8
          M₁ : Type u_9
          M₂ : Type u_10
          M₃ : Type u_11
          N₂ : Type u_12
          N₃ : Type u_13
          F : Type u_14
          inst✝¹⁰ : Semiring R
          inst✝⁹ : Semiring S
          inst✝⁸ : AddCommMonoid M
          inst✝⁷ : AddCommMonoid M₁
          inst✝⁶ : AddCommMonoid M₂
          inst✝⁵ : AddCommMonoid M₃
          inst✝⁴ : Module R M
          inst✝³ : Module R M₂
          inst✝² : Module S M₃
          σ : RingHom R S
          inst✝¹ : FunLike F M M₃
          inst✝ : SemilinearMapClass F σ M M₃
          f : F
          ⊢ Eq (HSMul.hSMul (σ 0) (f 0)) 0
        -/
        simp }
        /-
          🎉 no goals
        -/


instance (priority := 100) distribMulActionSemiHomClass
    [FunLike F M M₃] [SemilinearMapClass F σ M M₃] :
    DistribMulActionSemiHomClass F σ M M₃ :=
  { SemilinearMapClass.toAddHomClass with
                                 /-
                                   R : Type u_1
                                   R₁ : Type u_2
                                   R₂ : Type u_3
                                   R₃ : Type u_4
                                   S : Type u_5
                                   S₃ : Type u_6
                                   T : Type u_7
                                   M : Type u_8
                                   M₁ : Type u_9
                                   M₂ : Type u_10
                                   M₃ : Type u_11
                                   N₂ : Type u_12
                                   N₃ : Type u_13
                                   F : Type u_14
                                   inst✝¹⁰ : Semiring R
                                   inst✝⁹ : Semiring S
                                   inst✝⁸ : AddCommMonoid M
                                   inst✝⁷ : AddCommMonoid M₁
                                   inst✝⁶ : AddCommMonoid M₂
                                   inst✝⁵ : AddCommMonoid M₃
                                   inst✝⁴ : Module R M
                                   inst✝³ : Module R M₂
                                   inst✝² : Module S M₃
                                   σ : RingHom R S
                                   inst✝¹ : FunLike F M M₃
                                   inst✝ : SemilinearMapClass F σ M M₃
                                   f : F
                                   c : R
                                   x : M
                                   ⊢ Eq (f (HSMul.hSMul c x)) (HSMul.hSMul (σ c) (f x))
                                 -/
    map_smulₛₗ := fun f c x ↦ by rw [map_smulₛₗ] }
                                 /-
                                   🎉 no goals
                                 -/


theorem map_smul_inv {σ' : S →+* R} [RingHomInvPair σ σ'] (c : S) (x : M) :
                                 /-
                                   R : Type u_1
                                   S : Type u_5
                                   M : Type u_8
                                   M₃ : Type u_11
                                   F : Type u_14
                                   inst✝⁸ : Semiring R
                                   inst✝⁷ : Semiring S
                                   inst✝⁶ : AddCommMonoid M
                                   inst✝⁵ : AddCommMonoid M₃
                                   inst✝⁴ : Module R M
                                   inst✝³ : Module S M₃
                                   σ : RingHom R S
                                   f : F
                                   inst✝² : FunLike F M M₃
                                   inst✝¹ : SemilinearMapClass F σ M M₃
                                   σ' : RingHom S R
                                   inst✝ : RingHomInvPair σ σ'
                                   c : S
                                   x : M
                                   ⊢ Eq (HSMul.hSMul c (f x)) (f (HSMul.hSMul (σ' c) x))
                                 -/
    c • f x = f (σ' c • x) := by simp [map_smulₛₗ _]
                                 /-
                                   🎉 no goals
                                 -/


/-- Reinterpret an element of a type of semilinear maps as a semilinear map. -/
@[coe]
def semilinearMap : M →ₛₗ[σ] M₃ where
  toFun := f
  map_add' := map_add f
  map_smul' := map_smulₛₗ f


/-- Reinterpret an element of a type of semilinear maps as a semilinear map. -/
instance instCoeToSemilinearMap : CoeHead F (M →ₛₗ[σ] M₃) where
  coe f := semilinearMap f


/-- Reinterpret an element of a type of linear maps as a linear map. -/
abbrev linearMap : M₁ →ₗ[R] M₂ := SemilinearMapClass.semilinearMap f


/-- Reinterpret an element of a type of linear maps as a linear map. -/
instance instCoeToLinearMap : CoeHead F (M₁ →ₗ[R] M₂) where
  coe f := SemilinearMapClass.semilinearMap f


instance instFunLike : FunLike (M →ₛₗ[σ] M₃) M M₃ where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      S : Type u_5
      S₃ : Type u_6
      T : Type u_7
      M : Type u_8
      M₁ : Type u_9
      M₂ : Type u_10
      M₃ : Type u_11
      N₂ : Type u_12
      N₃ : Type u_13
      inst✝⁸ : Semiring R
      inst✝⁷ : Semiring S
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module S M₃
      σ : RingHom R S
      f g : LinearMap σ M M₃
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      S : Type u_5
      S₃ : Type u_6
      T : Type u_7
      M : Type u_8
      M₁ : Type u_9
      M₂ : Type u_10
      M₃ : Type u_11
      N₂ : Type u_12
      N₃ : Type u_13
      inst✝⁸ : Semiring R
      inst✝⁷ : Semiring S
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module S M₃
      σ : RingHom R S
      g : LinearMap σ M M₃
      toAddHom✝ : AddHom M M₃
      map_smul'✝ : ∀ (m : R) (x : M), Eq (toAddHom✝.toFun (HSMul.hSMul m x)) (HSMul. …
      h : Eq ((fun f => f.toFun) { toAddHom := toAddHom✝, map_smul' := map_smul'✝ }) …
      ⊢ Eq { toAddHom := toAddHom✝, map_smul' := map_smul'✝ } g
    -/
    cases g
    /-
      case mk.mk
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      S : Type u_5
      S₃ : Type u_6
      T : Type u_7
      M : Type u_8
      M₁ : Type u_9
      M₂ : Type u_10
      M₃ : Type u_11
      N₂ : Type u_12
      N₃ : Type u_13
      inst✝⁸ : Semiring R
      inst✝⁷ : Semiring S
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module S M₃
      σ : RingHom R S
      toAddHom✝¹ : AddHom M M₃
      map_smul'✝¹ : ∀ (m : R) (x : M), Eq (toAddHom✝¹.toFun (HSMul.hSMul m x)) (HSMu …
      toAddHom✝ : AddHom M M₃
      map_smul'✝ : ∀ (m : R) (x : M), Eq (toAddHom✝.toFun (HSMul.hSMul m x)) (HSMul. …
      h : Eq ((fun f => f.toFun) { toAddHom := toAddHom✝¹, map_smul' := map_smul'✝¹  …
      ⊢ Eq { toAddHom := toAddHom✝¹, map_smul' := map_smul'✝¹ } { toAddHom := toAddH …
    -/
    congr
    /-
      case mk.mk.e_toAddHom
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      S : Type u_5
      S₃ : Type u_6
      T : Type u_7
      M : Type u_8
      M₁ : Type u_9
      M₂ : Type u_10
      M₃ : Type u_11
      N₂ : Type u_12
      N₃ : Type u_13
      inst✝⁸ : Semiring R
      inst✝⁷ : Semiring S
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module S M₃
      σ : RingHom R S
      toAddHom✝¹ : AddHom M M₃
      map_smul'✝¹ : ∀ (m : R) (x : M), Eq (toAddHom✝¹.toFun (HSMul.hSMul m x)) (HSMu …
      toAddHom✝ : AddHom M M₃
      map_smul'✝ : ∀ (m : R) (x : M), Eq (toAddHom✝.toFun (HSMul.hSMul m x)) (HSMul. …
      h : Eq ((fun f => f.toFun) { toAddHom := toAddHom✝¹, map_smul' := map_smul'✝¹  …
      ⊢ Eq toAddHom✝¹ toAddHom✝
    -/
    apply DFunLike.coe_injective'
    /-
      case mk.mk.e_toAddHom.a
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      S : Type u_5
      S₃ : Type u_6
      T : Type u_7
      M : Type u_8
      M₁ : Type u_9
      M₂ : Type u_10
      M₃ : Type u_11
      N₂ : Type u_12
      N₃ : Type u_13
      inst✝⁸ : Semiring R
      inst✝⁷ : Semiring S
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module S M₃
      σ : RingHom R S
      toAddHom✝¹ : AddHom M M₃
      map_smul'✝¹ : ∀ (m : R) (x : M), Eq (toAddHom✝¹.toFun (HSMul.hSMul m x)) (HSMu …
      toAddHom✝ : AddHom M M₃
      map_smul'✝ : ∀ (m : R) (x : M), Eq (toAddHom✝.toFun (HSMul.hSMul m x)) (HSMul. …
      h : Eq ((fun f => f.toFun) { toAddHom := toAddHom✝¹, map_smul' := map_smul'✝¹  …
      ⊢ Eq ⇑toAddHom✝¹ ⇑toAddHom✝
    -/
    exact h
    /-
      🎉 no goals
    -/


instance semilinearMapClass : SemilinearMapClass (M →ₛₗ[σ] M₃) σ M M₃ where
  map_add f := f.map_add'
  map_smulₛₗ := LinearMap.map_smul'


@[simp, norm_cast]
lemma coe_coe {F : Type*} [FunLike F M M₃] [SemilinearMapClass F σ M M₃] {f : F} :
    ⇑(f : M →ₛₗ[σ] M₃) = f :=
  rfl


/-- The `DistribMulActionHom` underlying a `LinearMap`. -/
def toDistribMulActionHom (f : M →ₛₗ[σ] M₃) : DistribMulActionHom σ.toMonoidHom M M₃ :=
  { f with map_zero' := show f 0 = 0 from map_zero f }


@[simp]
theorem coe_toAddHom (f : M →ₛₗ[σ] M₃) : ⇑f.toAddHom = f := rfl

-- Porting note: no longer a `simp`

theorem toFun_eq_coe {f : M →ₛₗ[σ] M₃} : f.toFun = (f : M → M₃) := rfl


@[ext]
theorem ext {f g : M →ₛₗ[σ] M₃} (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext f g h


/-- Copy of a `LinearMap` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (f : M →ₛₗ[σ] M₃) (f' : M → M₃) (h : f' = ⇑f) : M →ₛₗ[σ] M₃ where
  toFun := f'
  map_add' := h.symm ▸ f.map_add'
  map_smul' := h.symm ▸ f.map_smul'


@[simp]
theorem coe_copy (f : M →ₛₗ[σ] M₃) (f' : M → M₃) (h : f' = ⇑f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : M →ₛₗ[σ] M₃) (f' : M → M₃) (h : f' = ⇑f) : f.copy f' h = f :=
  DFunLike.ext' h


@[simp]
theorem coe_mk {σ : R →+* S} (f : AddHom M M₃) (h) :
    ((LinearMap.mk f h : M →ₛₗ[σ] M₃) : M → M₃) = f :=
  rfl

-- Porting note: This theorem is new.

@[simp]
theorem coe_addHom_mk {σ : R →+* S} (f : AddHom M M₃) (h) :
    ((LinearMap.mk f h : M →ₛₗ[σ] M₃) : AddHom M M₃) = f :=
  rfl


theorem coe_semilinearMap {F : Type*} [FunLike F M M₃] [SemilinearMapClass F σ M M₃] (f : F) :
    ((f : M →ₛₗ[σ] M₃) : M → M₃) = f :=
  rfl


theorem toLinearMap_injective {F : Type*} [FunLike F M M₃] [SemilinearMapClass F σ M M₃]
    {f g : F} (h : (f : M →ₛₗ[σ] M₃) = (g : M →ₛₗ[σ] M₃)) :
    f = g := by
  /-
    R : Type u_1
    S : Type u_5
    M : Type u_8
    M₃ : Type u_11
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring S
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₃
    inst✝³ : Module R M
    inst✝² : Module S M₃
    σ : RingHom R S
    F : Type u_14
    inst✝¹ : FunLike F M M₃
    inst✝ : SemilinearMapClass F σ M M₃
    f g : F
    h : Eq ↑f ↑g
    ⊢ Eq f g
  -/
  apply DFunLike.ext
  /-
    case h
    R : Type u_1
    S : Type u_5
    M : Type u_8
    M₃ : Type u_11
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring S
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₃
    inst✝³ : Module R M
    inst✝² : Module S M₃
    σ : RingHom R S
    F : Type u_14
    inst✝¹ : FunLike F M M₃
    inst✝ : SemilinearMapClass F σ M M₃
    f g : F
    h : Eq ↑f ↑g
    ⊢ ∀ (x : M), Eq (f x) (g x)
  -/
  intro m
  /-
    case h
    R : Type u_1
    S : Type u_5
    M : Type u_8
    M₃ : Type u_11
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring S
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₃
    inst✝³ : Module R M
    inst✝² : Module S M₃
    σ : RingHom R S
    F : Type u_14
    inst✝¹ : FunLike F M M₃
    inst✝ : SemilinearMapClass F σ M M₃
    f g : F
    h : Eq ↑f ↑g
    m : M
    ⊢ Eq (f m) (g m)
  -/
  exact DFunLike.congr_fun h m
  /-
    🎉 no goals
  -/


/-- Identity map as a `LinearMap` -/
def id : M →ₗ[R] M :=
  { DistribMulActionHom.id R with toFun := _root_.id }


theorem id_apply (x : M) : @id R M _ _ _ x = x :=
  rfl


@[simp, norm_cast]
theorem id_coe : ((LinearMap.id : M →ₗ[R] M) : M → M) = _root_.id :=
  rfl


/-- A generalisation of `LinearMap.id` that constructs the identity function
as a `σ`-semilinear map for any ring homomorphism `σ` which we know is the identity. -/
@[simps]
def id' {σ : R →+* R} [RingHomId σ] : M →ₛₗ[σ] M where
  toFun x := x
  map_add' _ _ := rfl
  map_smul' r x := by
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      S : Type u_5
      S₃ : Type u_6
      T : Type u_7
      M : Type u_8
      M₁ : Type u_9
      M₂ : Type u_10
      M₃ : Type u_11
      N₂ : Type u_12
      N₃ : Type u_13
      inst✝⁹ : Semiring R
      inst✝⁸ : Semiring S
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : AddCommMonoid M₁
      inst✝⁵ : AddCommMonoid M₂
      inst✝⁴ : AddCommMonoid M₃
      inst✝³ : Module R M
      inst✝² : Module R M₂
      inst✝¹ : Module S M₃
      σ✝ : RingHom R S
      σ : RingHom R R
      inst✝ : RingHomId σ
      r : R
      x : M
      ⊢ Eq ({ toFun := fun x => x, map_add' := ⋯ }.toFun (HSMul.hSMul r x)) (HSMul.h …
    -/
    have := (RingHomId.eq_id : σ = _)
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      S : Type u_5
      S₃ : Type u_6
      T : Type u_7
      M : Type u_8
      M₁ : Type u_9
      M₂ : Type u_10
      M₃ : Type u_11
      N₂ : Type u_12
      N₃ : Type u_13
      inst✝⁹ : Semiring R
      inst✝⁸ : Semiring S
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : AddCommMonoid M₁
      inst✝⁵ : AddCommMonoid M₂
      inst✝⁴ : AddCommMonoid M₃
      inst✝³ : Module R M
      inst✝² : Module R M₂
      inst✝¹ : Module S M₃
      σ✝ : RingHom R S
      σ : RingHom R R
      inst✝ : RingHomId σ
      r : R
      x : M
      this : Eq σ (RingHom.id R)
      ⊢ Eq ({ toFun := fun x => x, map_add' := ⋯ }.toFun (HSMul.hSMul r x)) (HSMul.h …
    -/
    subst this
    /-
      R : Type u_1
      R₁ : Type u_2
      R₂ : Type u_3
      R₃ : Type u_4
      S : Type u_5
      S₃ : Type u_6
      T : Type u_7
      M : Type u_8
      M₁ : Type u_9
      M₂ : Type u_10
      M₃ : Type u_11
      N₂ : Type u_12
      N₃ : Type u_13
      inst✝⁹ : Semiring R
      inst✝⁸ : Semiring S
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : AddCommMonoid M₁
      inst✝⁵ : AddCommMonoid M₂
      inst✝⁴ : AddCommMonoid M₃
      inst✝³ : Module R M
      inst✝² : Module R M₂
      inst✝¹ : Module S M₃
      σ : RingHom R S
      r : R
      x : M
      inst✝ : RingHomId (RingHom.id R)
      ⊢ Eq ({ toFun := fun x => x, map_add' := ⋯ }.toFun (HSMul.hSMul r x)) (HSMul.h …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem id'_coe {σ : R →+* R} [RingHomId σ] : ((id' : M →ₛₗ[σ] M) : M → M) = _root_.id :=
  rfl


theorem isLinear : IsLinearMap R fₗ :=
  ⟨fₗ.map_add', fₗ.map_smul'⟩


theorem coe_injective : Injective (DFunLike.coe : (M →ₛₗ[σ] M₃) → _) :=
  DFunLike.coe_injective


protected theorem congr_arg {x x' : M} : x = x' → f x = f x' :=
  DFunLike.congr_arg f


/-- If two linear maps are equal, they are equal at each point. -/
protected theorem congr_fun (h : f = g) (x : M) : f x = g x :=
  DFunLike.congr_fun h x


@[simp]
theorem mk_coe (f : M →ₛₗ[σ] M₃) (h) : (LinearMap.mk f h : M →ₛₗ[σ] M₃) = f :=
  rfl


protected theorem map_add (x y : M) : f (x + y) = f x + f y :=
  map_add f x y


protected theorem map_zero : f 0 = 0 :=
  map_zero f

-- Porting note: `simp` wasn't picking up `map_smulₛₗ` for `LinearMap`s without specifying
-- `map_smulₛₗ f`, so we marked this as `@[simp]` in Mathlib3.
-- For Mathlib4, let's try without the `@[simp]` attribute and hope it won't need to be re-enabled.
-- This has to be re-tagged as `@[simp]` in https://github.com/leanprover-community/mathlib4/pull/8386 (see also https://github.com/leanprover/lean4/issues/3107).

@[simp]
protected theorem map_smulₛₗ (c : R) (x : M) : f (c • x) = σ c • f x :=
  map_smulₛₗ f c x


protected theorem map_smul (c : R) (x : M) : fₗ (c • x) = c • fₗ x :=
  map_smul fₗ c x


protected theorem map_smul_inv {σ' : S →+* R} [RingHomInvPair σ σ'] (c : S) (x : M) :
                                 /-
                                   R : Type u_1
                                   S : Type u_5
                                   M : Type u_8
                                   M₃ : Type u_11
                                   inst✝⁶ : Semiring R
                                   inst✝⁵ : Semiring S
                                   inst✝⁴ : AddCommMonoid M
                                   inst✝³ : AddCommMonoid M₃
                                   inst✝² : Module R M
                                   inst✝¹ : Module S M₃
                                   σ : RingHom R S
                                   f : LinearMap σ M M₃
                                   σ' : RingHom S R
                                   inst✝ : RingHomInvPair σ σ'
                                   c : S
                                   x : M
                                   ⊢ Eq (HSMul.hSMul c (f x)) (f (HSMul.hSMul (σ' c) x))
                                 -/
    c • f x = f (σ' c • x) := by simp
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem map_eq_zero_iff (h : Function.Injective f) {x : M} : f x = 0 ↔ x = 0 :=
  _root_.map_eq_zero_iff f h


/-- A typeclass for `SMul` structures which can be moved through a `LinearMap`.
This typeclass is generated automatically from an `IsScalarTower` instance, but exists so that
we can also add an instance for `AddCommGroup.toIntModule`, allowing `z •` to be moved even if
`S` does not support negation.
-/
class CompatibleSMul (R S : Type*) [Semiring S] [SMul R M] [Module S M] [SMul R M₂]
  [Module S M₂] : Prop where
  /-- Scalar multiplication by `R` of `M` can be moved through linear maps. -/
  map_smul : ∀ (fₗ : M →ₗ[S] M₂) (c : R) (x : M), fₗ (c • x) = c • fₗ x


instance (priority := 100) IsScalarTower.compatibleSMul [SMul R S]
    [IsScalarTower R S M] [IsScalarTower R S M₂] :
    CompatibleSMul M M₂ R S :=
                   /-
                     R✝ : Type u_1
                     R₁ : Type u_2
                     R₂ : Type u_3
                     R₃ : Type u_4
                     S✝ : Type u_5
                     S₃ : Type u_6
                     T : Type u_7
                     M : Type u_8
                     M₁ : Type u_9
                     M₂ : Type u_10
                     M₃ : Type u_11
                     N₂ : Type u_12
                     N₃ : Type u_13
                     inst✝¹⁶ : Semiring R✝
                     inst✝¹⁵ : Semiring S✝
                     inst✝¹⁴ : AddCommMonoid M
                     inst✝¹³ : AddCommMonoid M₁
                     inst✝¹² : AddCommMonoid M₂
                     inst✝¹¹ : AddCommMonoid M₃
                     inst✝¹⁰ : Module R✝ M
                     inst✝⁹ : Module R✝ M₂
                     inst✝⁸ : Module S✝ M₃
                     σ : RingHom R✝ S✝
                     fₗ✝ : LinearMap (RingHom.id R✝) M M₂
                     f g : LinearMap σ M M₃
                     R : Type u_14
                     S : Type u_15
                     inst✝⁷ : Semiring S
                     inst✝⁶ : SMul R M
                     inst✝⁵ : Module S M
                     inst✝⁴ : SMul R M₂
                     inst✝³ : Module S M₂
                     inst✝² : SMul R S
                     inst✝¹ : IsScalarTower R S M
                     inst✝ : IsScalarTower R S M₂
                     fₗ : LinearMap (RingHom.id S) M M₂
                     c : R
                     x : M
                     ⊢ Eq (fₗ (HSMul.hSMul c x)) (HSMul.hSMul c (fₗ x))
                   -/
  ⟨fun fₗ c x ↦ by rw [← smul_one_smul S c x, ← smul_one_smul S c (fₗ x), map_smul]⟩
                   /-
                     🎉 no goals
                   -/


instance IsScalarTower.compatibleSMul' [SMul R S] [IsScalarTower R S M] :
    CompatibleSMul S M R S where
  map_smul := (IsScalarTower.smulHomClass R S M (S →ₗ[S] M)).map_smulₛₗ


@[simp]
theorem map_smul_of_tower [CompatibleSMul M M₂ R S] (fₗ : M →ₗ[S] M₂) (c : R) (x : M) :
    fₗ (c • x) = c • fₗ x :=
  CompatibleSMul.map_smul fₗ c x


variable (R R) in
theorem isScalarTower_of_injective [SMul R S] [CompatibleSMul M M₂ R S] [IsScalarTower R S M₂]
    (f : M →ₗ[S] M₂) (hf : Function.Injective f) : IsScalarTower R S M where
                               /-
                                 M : Type u_8
                                 M₂ : Type u_10
                                 inst✝⁹ : AddCommMonoid M
                                 inst✝⁸ : AddCommMonoid M₂
                                 R : Type u_14
                                 S : Type u_15
                                 inst✝⁷ : Semiring S
                                 inst✝⁶ : SMul R M
                                 inst✝⁵ : Module S M
                                 inst✝⁴ : SMul R M₂
                                 inst✝³ : Module S M₂
                                 inst✝² : SMul R S
                                 inst✝¹ : LinearMap.CompatibleSMul M M₂ R S
                                 inst✝ : IsScalarTower R S M₂
                                 f : LinearMap (RingHom.id S) M M₂
                                 hf : Function.Injective ⇑f
                                 r : R
                                 s : S
                                 x✝ : M
                                 ⊢ Eq (f (HSMul.hSMul (HSMul.hSMul r s) x✝)) (f (HSMul.hSMul r (HSMul.hSMul s x …
                               -/
  smul_assoc r s _ := hf <| by rw [f.map_smul_of_tower r, map_smul, map_smul, smul_assoc]
                               /-
                                 🎉 no goals
                               -/


variable (R) in
theorem isLinearMap_of_compatibleSMul [Module S M] [Module S M₂] [CompatibleSMul M M₂ R S]
    (f : M →ₗ[S] M₂) : IsLinearMap R f where
  map_add := map_add f
  map_smul := map_smul_of_tower f


/-- convert a linear map to an additive map -/
def toAddMonoidHom : M →+ M₃ where
  toFun := f
  map_zero' := f.map_zero
  map_add' := f.map_add


@[simp]
theorem toAddMonoidHom_coe : ⇑f.toAddMonoidHom = f :=
  rfl


/-- If `M` and `M₂` are both `R`-modules and `S`-modules and `R`-module structures
are defined by an action of `R` on `S` (formally, we have two scalar towers), then any `S`-linear
map from `M` to `M₂` is `R`-linear.

See also `LinearMap.map_smul_of_tower`. -/
@[coe] def restrictScalars (fₗ : M →ₗ[S] M₂) : M →ₗ[R] M₂ where
  toFun := fₗ
  map_add' := fₗ.map_add
  map_smul' := fₗ.map_smul_of_tower

-- Porting note: generalized from `Algebra` to `CompatibleSMul`

instance coeIsScalarTower : CoeHTCT (M →ₗ[S] M₂) (M →ₗ[R] M₂) :=
  ⟨restrictScalars R⟩


@[simp, norm_cast]
theorem coe_restrictScalars (f : M →ₗ[S] M₂) : ((f : M →ₗ[R] M₂) : M → M₂) = f :=
  rfl


theorem restrictScalars_apply (fₗ : M →ₗ[S] M₂) (x) : restrictScalars R fₗ x = fₗ x :=
  rfl


theorem restrictScalars_injective :
    Function.Injective (restrictScalars R : (M →ₗ[S] M₂) → M →ₗ[R] M₂) := fun _ _ h ↦
  ext (LinearMap.congr_fun h : _)


@[simp]
theorem restrictScalars_inj (fₗ gₗ : M →ₗ[S] M₂) :
    fₗ.restrictScalars R = gₗ.restrictScalars R ↔ fₗ = gₗ :=
  (restrictScalars_injective R).eq_iff


theorem toAddMonoidHom_injective :
    Function.Injective (toAddMonoidHom : (M →ₛₗ[σ] M₃) → M →+ M₃) := fun fₗ gₗ h ↦
  ext <| (DFunLike.congr_fun h : ∀ x, fₗ.toAddMonoidHom x = gₗ.toAddMonoidHom x)


/-- If two `σ`-linear maps from `R` are equal on `1`, then they are equal. -/
@[ext high]
theorem ext_ring {f g : R →ₛₗ[σ] M₃} (h : f 1 = g 1) : f = g :=
                 /-
                   R : Type u_1
                   S : Type u_5
                   M₃ : Type u_11
                   inst✝³ : Semiring R
                   inst✝² : Semiring S
                   inst✝¹ : AddCommMonoid M₃
                   inst✝ : Module S M₃
                   σ : RingHom R S
                   f g : LinearMap σ R M₃
                   h : Eq (f 1) (g 1)
                   x : R
                   ⊢ Eq (f x) (g x)
                 -/
  ext fun x ↦ by rw [← mul_one x, ← smul_eq_mul, f.map_smulₛₗ, g.map_smulₛₗ, h]
                 /-
                   🎉 no goals
                 -/


/-- Interpret a `RingHom` `f` as an `f`-semilinear map. -/
@[simps]
def _root_.RingHom.toSemilinearMap (f : R →+* S) : R →ₛₗ[f] S :=
  { f with
    map_smul' := f.map_mul }


/-- Composition of two linear maps is a linear map -/
def comp [RingHomCompTriple σ₁₂ σ₂₃ σ₁₃] (f : M₂ →ₛₗ[σ₂₃] M₃) (g : M₁ →ₛₗ[σ₁₂] M₂) :
    M₁ →ₛₗ[σ₁₃] M₃ where
  toFun := f ∘ g
                 /-
                   R : Type u_1
                   R₁ : Type u_2
                   R₂ : Type u_3
                   R₃ : Type u_4
                   S : Type u_5
                   S₃ : Type u_6
                   T : Type u_7
                   M : Type u_8
                   M₁ : Type u_9
                   M₂ : Type u_10
                   M₃ : Type u_11
                   N₂ : Type u_12
                   N₃ : Type u_13
                   inst✝⁹ : Semiring R
                   inst✝⁸ : Semiring S
                   inst✝⁷ : Semiring R₁
                   inst✝⁶ : Semiring R₂
                   inst✝⁵ : Semiring R₃
                   inst✝⁴ : AddCommMonoid M
                   inst✝³ : AddCommMonoid M₁
                   inst✝² : AddCommMonoid M₂
                   inst✝¹ : AddCommMonoid M₃
                   module_M₁ : Module R₁ M₁
                   module_M₂ : Module R₂ M₂
                   module_M₃ : Module R₃ M₃
                   σ₁₂ : RingHom R₁ R₂
                   σ₂₃ : RingHom R₂ R₃
                   σ₁₃ : RingHom R₁ R₃
                   inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                   f : LinearMap σ₂₃ M₂ M₃
                   g : LinearMap σ₁₂ M₁ M₂
                   ⊢ ∀ (x y : M₁), Eq (Function.comp (⇑f) (⇑g) (HAdd.hAdd x y)) (HAdd.hAdd (Funct …
                 -/
  map_add' := by simp only [map_add, forall_const, Function.comp_apply]
                 /-
                   🎉 no goals
                 -/
  -- Note that https://github.com/leanprover-community/mathlib4/pull/8386 changed `map_smulₛₗ` to `map_smulₛₗ _`
                      /-
                        R : Type u_1
                        R₁ : Type u_2
                        R₂ : Type u_3
                        R₃ : Type u_4
                        S : Type u_5
                        S₃ : Type u_6
                        T : Type u_7
                        M : Type u_8
                        M₁ : Type u_9
                        M₂ : Type u_10
                        M₃ : Type u_11
                        N₂ : Type u_12
                        N₃ : Type u_13
                        inst✝⁹ : Semiring R
                        inst✝⁸ : Semiring S
                        inst✝⁷ : Semiring R₁
                        inst✝⁶ : Semiring R₂
                        inst✝⁵ : Semiring R₃
                        inst✝⁴ : AddCommMonoid M
                        inst✝³ : AddCommMonoid M₁
                        inst✝² : AddCommMonoid M₂
                        inst✝¹ : AddCommMonoid M₃
                        module_M₁ : Module R₁ M₁
                        module_M₂ : Module R₂ M₂
                        module_M₃ : Module R₃ M₃
                        σ₁₂ : RingHom R₁ R₂
                        σ₂₃ : RingHom R₂ R₃
                        σ₁₃ : RingHom R₁ R₃
                        inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                        f : LinearMap σ₂₃ M₂ M₃
                        g : LinearMap σ₁₂ M₁ M₂
                        r : R₁
                        x : M₁
                        ⊢ Eq ({ toFun := Function.comp ⇑f ⇑g, map_add' := ⋯ }.toFun (HSMul.hSMul r x)) …
                      -/
  map_smul' r x := by simp only [Function.comp_apply, map_smulₛₗ _, RingHomCompTriple.comp_apply]
                      /-
                        🎉 no goals
                      -/


/-- `∘ₗ` is notation for composition of two linear (not semilinear!) maps into a linear map.
This is useful when Lean is struggling to infer the `RingHomCompTriple` instance. -/
notation3:80 (name := compNotation) f:81 " ∘ₗ " g:80 =>
  LinearMap.comp (σ₁₂ := RingHom.id _) (σ₂₃ := RingHom.id _) (σ₁₃ := RingHom.id _) f g


theorem comp_apply (x : M₁) : f.comp g x = f (g x) :=
  rfl


@[simp, norm_cast]
theorem coe_comp : (f.comp g : M₁ → M₃) = f ∘ g :=
  rfl


@[simp]
theorem comp_id : f.comp id = f :=
  rfl


@[simp]
theorem id_comp : id.comp f = f :=
  rfl


theorem comp_assoc
    {R₄ M₄ : Type*} [Semiring R₄] [AddCommMonoid M₄] [Module R₄ M₄]
    {σ₃₄ : R₃ →+* R₄} {σ₂₄ : R₂ →+* R₄} {σ₁₄ : R₁ →+* R₄}
    [RingHomCompTriple σ₂₃ σ₃₄ σ₂₄] [RingHomCompTriple σ₁₃ σ₃₄ σ₁₄] [RingHomCompTriple σ₁₂ σ₂₄ σ₁₄]
    (f : M₁ →ₛₗ[σ₁₂] M₂) (g : M₂ →ₛₗ[σ₂₃] M₃) (h : M₃ →ₛₗ[σ₃₄] M₄) :
    ((h.comp g : M₂ →ₛₗ[σ₂₄] M₄).comp f : M₁ →ₛₗ[σ₁₄] M₄) = h.comp (g.comp f : M₁ →ₛₗ[σ₁₃] M₃) :=
  rfl


/-- The linear map version of `Function.Surjective.injective_comp_right` -/
lemma _root_.Function.Surjective.injective_linearMapComp_right (hg : Surjective g) :
    Injective fun f : M₂ →ₛₗ[σ₂₃] M₃ ↦ f.comp g :=
  fun _ _ h ↦ ext <| hg.forall.2 (LinearMap.ext_iff.1 h)


@[simp]
theorem cancel_right (hg : Surjective g) : f.comp g = f'.comp g ↔ f = f' :=
  hg.injective_linearMapComp_right.eq_iff


/-- The linear map version of `Function.Injective.comp_left` -/
lemma _root_.Function.Injective.injective_linearMapComp_left (hf : Injective f) :
    Injective fun g : M₁ →ₛₗ[σ₁₂] M₂ ↦ f.comp g :=
                                                               /-
                                                                 R₁ : Type u_2
                                                                 R₂ : Type u_3
                                                                 R₃ : Type u_4
                                                                 M₁ : Type u_9
                                                                 M₂ : Type u_10
                                                                 M₃ : Type u_11
                                                                 inst✝⁶ : Semiring R₁
                                                                 inst✝⁵ : Semiring R₂
                                                                 inst✝⁴ : Semiring R₃
                                                                 inst✝³ : AddCommMonoid M₁
                                                                 inst✝² : AddCommMonoid M₂
                                                                 inst✝¹ : AddCommMonoid M₃
                                                                 module_M₁ : Module R₁ M₁
                                                                 module_M₂ : Module R₂ M₂
                                                                 module_M₃ : Module R₃ M₃
                                                                 σ₁₂ : RingHom R₁ R₂
                                                                 σ₂₃ : RingHom R₂ R₃
                                                                 σ₁₃ : RingHom R₁ R₃
                                                                 inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                                                                 f : LinearMap σ₂₃ M₂ M₃
                                                                 hf : Function.Injective ⇑f
                                                                 g₁ g₂ : LinearMap σ₁₂ M₁ M₂
                                                                 h : Eq (f.comp g₁) (f.comp g₂)
                                                                 x : M₁
                                                                 ⊢ Eq (f (g₁ x)) (f (g₂ x))
                                                               -/
  fun g₁ g₂ (h : f.comp g₁ = f.comp g₂) ↦ ext fun x ↦ hf <| by rw [← comp_apply, h, comp_apply]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
theorem cancel_left (hf : Injective f) : f.comp g = f.comp g' ↔ g = g' :=
  hf.injective_linearMapComp_left.eq_iff


/-- If a function `g` is a left and right inverse of a linear map `f`, then `g` is linear itself. -/
def inverse (f : M →ₛₗ[σ] M₂) (g : M₂ → M) (h₁ : LeftInverse g f) (h₂ : RightInverse g f) :
    M₂ →ₛₗ[σ'] M := by
  /-
    R : Type u_1
    R₁ : Type u_2
    R₂ : Type u_3
    R₃ : Type u_4
    S : Type u_5
    S₃ : Type u_6
    T : Type u_7
    M : Type u_8
    M₁ : Type u_9
    M₂ : Type u_10
    M₃ : Type u_11
    N₂ : Type u_12
    N₃ : Type u_13
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring S
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M
    inst✝¹ : Module S M₂
    σ : RingHom R S
    σ' : RingHom S R
    inst✝ : RingHomInvPair σ σ'
    f : LinearMap σ M M₂
    g : M₂ → M
    h₁ : Function.LeftInverse g ⇑f
    h₂ : Function.RightInverse g ⇑f
    ⊢ LinearMap σ' M₂ M
  -/
  dsimp [LeftInverse, Function.RightInverse] at h₁ h₂
  exact
    { toFun := g
      map_add' := fun x y ↦ by rw [← h₁ (g (x + y)), ← h₁ (g x + g y)]; simp [h₂]
      map_smul' := fun a b ↦ by
        dsimp only
        rw [← h₁ (g (a • b)), ← h₁ (σ' a • g b)]
        simp [h₂] }


theorem injective_of_comp_eq_id : Injective f :=
                          /-
                            R : Type u_1
                            S : Type u_5
                            M : Type u_8
                            M₂ : Type u_10
                            inst✝⁶ : Semiring R
                            inst✝⁵ : Semiring S
                            inst✝⁴ : AddCommMonoid M
                            inst✝³ : AddCommMonoid M₂
                            inst✝² : Module R M
                            inst✝¹ : Module S M₂
                            σ : RingHom R S
                            σ' : RingHom S R
                            inst✝ : RingHomInvPair σ σ'
                            f : LinearMap σ M M₂
                            g : LinearMap σ' M₂ M
                            h : Eq (g.comp f) LinearMap.id
                            ⊢ Function.Injective (Function.comp ⇑g ⇑f)
                          -/
  .of_comp (f := g) <| by simp_rw [← coe_comp, h, id_coe, bijective_id.1]
                          /-
                            🎉 no goals
                          -/


theorem surjective_of_comp_eq_id : Surjective g :=
                          /-
                            R : Type u_1
                            S : Type u_5
                            M : Type u_8
                            M₂ : Type u_10
                            inst✝⁶ : Semiring R
                            inst✝⁵ : Semiring S
                            inst✝⁴ : AddCommMonoid M
                            inst✝³ : AddCommMonoid M₂
                            inst✝² : Module R M
                            inst✝¹ : Module S M₂
                            σ : RingHom R S
                            σ' : RingHom S R
                            inst✝ : RingHomInvPair σ σ'
                            f : LinearMap σ M M₂
                            g : LinearMap σ' M₂ M
                            h : Eq (g.comp f) LinearMap.id
                            ⊢ Function.Surjective (Function.comp ⇑g ⇑f)
                          -/
  .of_comp (g := f) <| by simp_rw [← coe_comp, h, id_coe, bijective_id.2]
                          /-
                            🎉 no goals
                          -/


protected theorem map_neg (x : M) : f (-x) = -f x :=
  map_neg f x


protected theorem map_sub (x y : M) : f (x - y) = f x - f y :=
  map_sub f x y


instance CompatibleSMul.intModule {S : Type*} [Semiring S] [Module S M] [Module S M₂] :
    CompatibleSMul M M₂ ℤ S :=
  ⟨fun fₗ c x ↦ by
    induction c using Int.induction_on with
    | hz => simp
    | hp n ih => simp [add_smul, ih]
    | hn n ih => simp [sub_smul, ih]⟩


instance CompatibleSMul.units {R S : Type*} [Monoid R] [MulAction R M] [MulAction R M₂]
    [Semiring S] [Module S M] [Module S M₂] [CompatibleSMul M M₂ R S] : CompatibleSMul M M₂ Rˣ S :=
  ⟨fun fₗ c x ↦ (CompatibleSMul.map_smul fₗ (c : R) x : _)⟩


/-- `g : R →+* S` is `R`-linear when the module structure on `S` is `Module.compHom S g` . -/
@[simps]
def compHom.toLinearMap {R S : Type*} [Semiring R] [Semiring S] (g : R →+* S) :
    letI := compHom S g; R →ₗ[R] S :=
  letI := compHom S g
  { toFun := (g : R → S)
    map_add' := g.map_add
    map_smul' := g.map_mul }


/-- A `DistribMulActionHom` between two modules is a linear map. -/
@[deprecated "No deprecation message was provided." (since := "2024-11-08")]
def toSemilinearMap (fₗ : M →ₑ+[σ.toMonoidHom] M₂) : M →ₛₗ[σ] M₂ :=
  { fₗ with }


instance : SemilinearMapClass (M →ₑ+[σ.toMonoidHom] M₂) σ M M₂ where


/-- A `DistribMulActionHom` between two modules is a linear map. -/
@[deprecated "No deprecation message was provided." (since := "2024-11-08")]
def toLinearMap (fₗ : M →+[R] M₃) : M →ₗ[R] M₃ :=
  { fₗ with }


/-- A `DistribMulActionHom` between two modules is a linear map. -/
instance : LinearMapClass (M →+[R] M₃) R M M₃ where

-- Porting note: because coercions get unfolded, there is no need for this rewrite

-- Porting note: removed @[norm_cast] attribute due to error:
-- norm_cast: badly shaped lemma, rhs can't start with coe

@[simp]
theorem coe_toLinearMap (f : M →ₑ+[σ.toMonoidHom] M₂) : ((f : M →ₛₗ[σ] M₂) : M → M₂) = f :=
  rfl


theorem toLinearMap_injective {f g : M →ₑ+[σ.toMonoidHom] M₂}
    (h : (f : M →ₛₗ[σ] M₂) = (g : M →ₛₗ[σ] M₂)) :
    f = g := by
  /-
    R : Type u_1
    S : Type u_5
    M : Type u_8
    M₂ : Type u_10
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Semiring R
    inst✝² : Module R M
    inst✝¹ : Semiring S
    inst✝ : Module S M₂
    σ : RingHom R S
    f g : DistribMulActionHom (↑σ) M M₂
    h : Eq ↑f ↑g
    ⊢ Eq f g
  -/
  ext m
  /-
    case a
    R : Type u_1
    S : Type u_5
    M : Type u_8
    M₂ : Type u_10
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Semiring R
    inst✝² : Module R M
    inst✝¹ : Semiring S
    inst✝ : Module S M₂
    σ : RingHom R S
    f g : DistribMulActionHom (↑σ) M M₂
    h : Eq ↑f ↑g
    m : M
    ⊢ Eq (f m) (g m)
  -/
  exact LinearMap.congr_fun h m
  /-
    🎉 no goals
  -/


/-- Convert an `IsLinearMap` predicate to a `LinearMap` -/
def mk' (f : M → M₂) (lin : IsLinearMap R f) : M →ₗ[R] M₂ where
  toFun := f
  map_add' := lin.1
  map_smul' := lin.2


@[simp]
theorem mk'_apply {f : M → M₂} (lin : IsLinearMap R f) (x : M) : mk' f lin x = f x :=
  rfl


theorem isLinearMap_smul {R M : Type*} [CommSemiring R] [AddCommMonoid M] [Module R M] (c : R) :
    IsLinearMap R fun z : M ↦ c • z := by
  /-
    R : Type u_14
    M : Type u_15
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    c : R
    ⊢ IsLinearMap R fun z => HSMul.hSMul c z
  -/
  refine IsLinearMap.mk (smul_add c) ?_
  /-
    R : Type u_14
    M : Type u_15
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    c : R
    ⊢ ∀ (c_1 : R) (x : M), Eq (HSMul.hSMul c (HSMul.hSMul c_1 x)) (HSMul.hSMul c_1 …
  -/
  intro _ _
  /-
    R : Type u_14
    M : Type u_15
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    c c✝ : R
    x✝ : M
    ⊢ Eq (HSMul.hSMul c (HSMul.hSMul c✝ x✝)) (HSMul.hSMul c✝ (HSMul.hSMul c x✝))
  -/
  simp only [smul_smul, mul_comm]
  /-
    🎉 no goals
  -/


theorem isLinearMap_smul' {R M : Type*} [Semiring R] [AddCommMonoid M] [Module R M] (a : M) :
    IsLinearMap R fun c : R ↦ c • a :=
  IsLinearMap.mk (fun x y ↦ add_smul x y a) fun x y ↦ mul_smul x y a


theorem map_zero {f : M → M₂} (lin : IsLinearMap R f) : f (0 : M) = (0 : M₂) :=
  (lin.mk' f).map_zero


theorem isLinearMap_neg : IsLinearMap R fun z : M ↦ -z :=
  IsLinearMap.mk neg_add fun x y ↦ (smul_neg x y).symm


theorem map_neg {f : M → M₂} (lin : IsLinearMap R f) (x : M) : f (-x) = -f x :=
  (lin.mk' f).map_neg x


theorem map_sub {f : M → M₂} (lin : IsLinearMap R f) (x y : M) : f (x - y) = f x - f y :=
  (lin.mk' f).map_sub x y


/-- Reinterpret an additive homomorphism as an `ℕ`-linear map. -/
def AddMonoidHom.toNatLinearMap [AddCommMonoid M] [AddCommMonoid M₂] (f : M →+ M₂) :
    M →ₗ[ℕ] M₂ where
  toFun := f
  map_add' := f.map_add
  map_smul' := map_nsmul f


theorem AddMonoidHom.toNatLinearMap_injective [AddCommMonoid M] [AddCommMonoid M₂] :
    Function.Injective (@AddMonoidHom.toNatLinearMap M M₂ _ _) := by
  /-
    M : Type u_8
    M₂ : Type u_10
    inst✝¹ : AddCommMonoid M
    inst✝ : AddCommMonoid M₂
    ⊢ Function.Injective AddMonoidHom.toNatLinearMap
  -/
  intro f g h
  /-
    M : Type u_8
    M₂ : Type u_10
    inst✝¹ : AddCommMonoid M
    inst✝ : AddCommMonoid M₂
    f g : AddMonoidHom M M₂
    h : Eq f.toNatLinearMap g.toNatLinearMap
    ⊢ Eq f g
  -/
  ext x
  /-
    case h
    M : Type u_8
    M₂ : Type u_10
    inst✝¹ : AddCommMonoid M
    inst✝ : AddCommMonoid M₂
    f g : AddMonoidHom M M₂
    h : Eq f.toNatLinearMap g.toNatLinearMap
    x : M
    ⊢ Eq (f x) (g x)
  -/
  exact LinearMap.congr_fun h x
  /-
    🎉 no goals
  -/


/-- Reinterpret an additive homomorphism as a `ℤ`-linear map. -/
def AddMonoidHom.toIntLinearMap [AddCommGroup M] [AddCommGroup M₂] (f : M →+ M₂) : M →ₗ[ℤ] M₂ where
  toFun := f
  map_add' := f.map_add
  map_smul' := map_zsmul f


theorem AddMonoidHom.toIntLinearMap_injective [AddCommGroup M] [AddCommGroup M₂] :
    Function.Injective (@AddMonoidHom.toIntLinearMap M M₂ _ _) := by
  /-
    M : Type u_8
    M₂ : Type u_10
    inst✝¹ : AddCommGroup M
    inst✝ : AddCommGroup M₂
    ⊢ Function.Injective AddMonoidHom.toIntLinearMap
  -/
  intro f g h
  /-
    M : Type u_8
    M₂ : Type u_10
    inst✝¹ : AddCommGroup M
    inst✝ : AddCommGroup M₂
    f g : AddMonoidHom M M₂
    h : Eq f.toIntLinearMap g.toIntLinearMap
    ⊢ Eq f g
  -/
  ext x
  /-
    case h
    M : Type u_8
    M₂ : Type u_10
    inst✝¹ : AddCommGroup M
    inst✝ : AddCommGroup M₂
    f g : AddMonoidHom M M₂
    h : Eq f.toIntLinearMap g.toIntLinearMap
    x : M
    ⊢ Eq (f x) (g x)
  -/
  exact LinearMap.congr_fun h x
  /-
    🎉 no goals
  -/


@[simp]
theorem AddMonoidHom.coe_toIntLinearMap [AddCommGroup M] [AddCommGroup M₂] (f : M →+ M₂) :
    ⇑f.toIntLinearMap = f :=
  rfl


instance : SMul S (M →ₛₗ[σ₁₂] M₂) :=
  ⟨fun a f ↦
    { toFun := a • (f : M → M₂)
                               /-
                                 R : Type u_1
                                 R₁ : Type u_2
                                 R₂ : Type u_3
                                 R₃ : Type u_4
                                 S : Type u_5
                                 S₃ : Type u_6
                                 T : Type u_7
                                 M : Type u_8
                                 M₁ : Type u_9
                                 M₂ : Type u_10
                                 M₃ : Type u_11
                                 N₂ : Type u_12
                                 N₃ : Type u_13
                                 inst✝¹¹ : Semiring R
                                 inst✝¹⁰ : Semiring R₂
                                 inst✝⁹ : AddCommMonoid M
                                 inst✝⁸ : AddCommMonoid M₂
                                 inst✝⁷ : Module R M
                                 inst✝⁶ : Module R₂ M₂
                                 σ₁₂ : RingHom R R₂
                                 inst✝⁵ : Monoid S
                                 inst✝⁴ : DistribMulAction S M₂
                                 inst✝³ : SMulCommClass R₂ S M₂
                                 inst✝² : Monoid T
                                 inst✝¹ : DistribMulAction T M₂
                                 inst✝ : SMulCommClass R₂ T M₂
                                 a : S
                                 f : LinearMap σ₁₂ M M₂
                                 x y : M
                                 ⊢ Eq (HSMul.hSMul a (⇑f) (HAdd.hAdd x y)) (HAdd.hAdd (HSMul.hSMul a (⇑f) x) (H …
                               -/
      map_add' := fun x y ↦ by simp only [Pi.smul_apply, f.map_add, smul_add]
                               /-
                                 🎉 no goals
                               -/
                                /-
                                  R : Type u_1
                                  R₁ : Type u_2
                                  R₂ : Type u_3
                                  R₃ : Type u_4
                                  S : Type u_5
                                  S₃ : Type u_6
                                  T : Type u_7
                                  M : Type u_8
                                  M₁ : Type u_9
                                  M₂ : Type u_10
                                  M₃ : Type u_11
                                  N₂ : Type u_12
                                  N₃ : Type u_13
                                  inst✝¹¹ : Semiring R
                                  inst✝¹⁰ : Semiring R₂
                                  inst✝⁹ : AddCommMonoid M
                                  inst✝⁸ : AddCommMonoid M₂
                                  inst✝⁷ : Module R M
                                  inst✝⁶ : Module R₂ M₂
                                  σ₁₂ : RingHom R R₂
                                  inst✝⁵ : Monoid S
                                  inst✝⁴ : DistribMulAction S M₂
                                  inst✝³ : SMulCommClass R₂ S M₂
                                  inst✝² : Monoid T
                                  inst✝¹ : DistribMulAction T M₂
                                  inst✝ : SMulCommClass R₂ T M₂
                                  a : S
                                  f : LinearMap σ₁₂ M M₂
                                  c : R
                                  x : M
                                  ⊢ Eq ({ toFun := HSMul.hSMul a ⇑f, map_add' := ⋯ }.toFun (HSMul.hSMul c x)) (H …
                                -/
      map_smul' := fun c x ↦ by simp [Pi.smul_apply, smul_comm] }⟩
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem smul_apply (a : S) (f : M →ₛₗ[σ₁₂] M₂) (x : M) : (a • f) x = a • f x :=
  rfl


theorem coe_smul (a : S) (f : M →ₛₗ[σ₁₂] M₂) : (a • f : M →ₛₗ[σ₁₂] M₂) = a • (f : M → M₂) :=
  rfl


instance [SMulCommClass S T M₂] : SMulCommClass S T (M →ₛₗ[σ₁₂] M₂) :=
  ⟨fun _ _ _ ↦ ext fun _ ↦ smul_comm _ _ _⟩

-- example application of this instance: if S -> T -> R are homomorphisms of commutative rings and
-- M and M₂ are R-modules then the S-module and T-module structures on Hom_R(M,M₂) are compatible.

instance [SMul S T] [IsScalarTower S T M₂] : IsScalarTower S T (M →ₛₗ[σ₁₂] M₂) where
  smul_assoc _ _ _ := ext fun _ ↦ smul_assoc _ _ _


instance [DistribMulAction Sᵐᵒᵖ M₂] [SMulCommClass R₂ Sᵐᵒᵖ M₂] [IsCentralScalar S M₂] :
    IsCentralScalar S (M →ₛₗ[σ₁₂] M₂) where
  op_smul_eq_smul _ _ := ext fun _ ↦ op_smul_eq_smul _ _


/-- The constant 0 map is linear. -/
instance : Zero (M →ₛₗ[σ₁₂] M₂) :=
  ⟨{  toFun := 0
                     /-
                       R : Type u_1
                       R₁ : Type u_2
                       R₂ : Type u_3
                       R₃ : Type u_4
                       S : Type u_5
                       S₃ : Type u_6
                       T : Type u_7
                       M : Type u_8
                       M₁ : Type u_9
                       M₂ : Type u_10
                       M₃ : Type u_11
                       N₂ : Type u_12
                       N₃ : Type u_13
                       inst✝¹³ : Semiring R₁
                       inst✝¹² : Semiring R₂
                       inst✝¹¹ : Semiring R₃
                       inst✝¹⁰ : AddCommMonoid M
                       inst✝⁹ : AddCommMonoid M₂
                       inst✝⁸ : AddCommMonoid M₃
                       inst✝⁷ : AddCommGroup N₂
                       inst✝⁶ : AddCommGroup N₃
                       inst✝⁵ : Module R₁ M
                       inst✝⁴ : Module R₂ M₂
                       inst✝³ : Module R₃ M₃
                       inst✝² : Module R₂ N₂
                       inst✝¹ : Module R₃ N₃
                       σ₁₂ : RingHom R₁ R₂
                       σ₂₃ : RingHom R₂ R₃
                       σ₁₃ : RingHom R₁ R₃
                       inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                       ⊢ ∀ (x y : M), Eq (0 (HAdd.hAdd x y)) (HAdd.hAdd (0 x) (0 y))
                     -/
      map_add' := by simp
                     /-
                       🎉 no goals
                     -/
                      /-
                        R : Type u_1
                        R₁ : Type u_2
                        R₂ : Type u_3
                        R₃ : Type u_4
                        S : Type u_5
                        S₃ : Type u_6
                        T : Type u_7
                        M : Type u_8
                        M₁ : Type u_9
                        M₂ : Type u_10
                        M₃ : Type u_11
                        N₂ : Type u_12
                        N₃ : Type u_13
                        inst✝¹³ : Semiring R₁
                        inst✝¹² : Semiring R₂
                        inst✝¹¹ : Semiring R₃
                        inst✝¹⁰ : AddCommMonoid M
                        inst✝⁹ : AddCommMonoid M₂
                        inst✝⁸ : AddCommMonoid M₃
                        inst✝⁷ : AddCommGroup N₂
                        inst✝⁶ : AddCommGroup N₃
                        inst✝⁵ : Module R₁ M
                        inst✝⁴ : Module R₂ M₂
                        inst✝³ : Module R₃ M₃
                        inst✝² : Module R₂ N₂
                        inst✝¹ : Module R₃ N₃
                        σ₁₂ : RingHom R₁ R₂
                        σ₂₃ : RingHom R₂ R₃
                        σ₁₃ : RingHom R₁ R₃
                        inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                        ⊢ ∀ (m : R₁) (x : M), Eq ({ toFun := 0, map_add' := ⋯ }.toFun (HSMul.hSMul m x …
                      -/
      map_smul' := by simp }⟩
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem zero_apply (x : M) : (0 : M →ₛₗ[σ₁₂] M₂) x = 0 :=
  rfl


@[simp]
theorem comp_zero (g : M₂ →ₛₗ[σ₂₃] M₃) : (g.comp (0 : M →ₛₗ[σ₁₂] M₂) : M →ₛₗ[σ₁₃] M₃) = 0 :=
                 /-
                   R₁ : Type u_2
                   R₂ : Type u_3
                   R₃ : Type u_4
                   M : Type u_8
                   M₂ : Type u_10
                   M₃ : Type u_11
                   inst✝⁹ : Semiring R₁
                   inst✝⁸ : Semiring R₂
                   inst✝⁷ : Semiring R₃
                   inst✝⁶ : AddCommMonoid M
                   inst✝⁵ : AddCommMonoid M₂
                   inst✝⁴ : AddCommMonoid M₃
                   inst✝³ : Module R₁ M
                   inst✝² : Module R₂ M₂
                   inst✝¹ : Module R₃ M₃
                   σ₁₂ : RingHom R₁ R₂
                   σ₂₃ : RingHom R₂ R₃
                   σ₁₃ : RingHom R₁ R₃
                   inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                   g : LinearMap σ₂₃ M₂ M₃
                   c : M
                   ⊢ Eq ((g.comp 0) c) (0 c)
                 -/
  ext fun c ↦ by rw [comp_apply, zero_apply, zero_apply, g.map_zero]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem zero_comp (f : M →ₛₗ[σ₁₂] M₂) : ((0 : M₂ →ₛₗ[σ₂₃] M₃).comp f : M →ₛₗ[σ₁₃] M₃) = 0 :=
  rfl


instance : Inhabited (M →ₛₗ[σ₁₂] M₂) :=
  ⟨0⟩


@[simp]
theorem default_def : (default : M →ₛₗ[σ₁₂] M₂) = 0 :=
  rfl


instance uniqueOfLeft [Subsingleton M] : Unique (M →ₛₗ[σ₁₂] M₂) :=
  { inferInstanceAs (Inhabited (M →ₛₗ[σ₁₂] M₂)) with
                                     /-
                                       R : Type u_1
                                       R₁ : Type u_2
                                       R₂ : Type u_3
                                       R₃ : Type u_4
                                       S : Type u_5
                                       S₃ : Type u_6
                                       T : Type u_7
                                       M : Type u_8
                                       M₁ : Type u_9
                                       M₂ : Type u_10
                                       M₃ : Type u_11
                                       N₂ : Type u_12
                                       N₃ : Type u_13
                                       inst✝¹⁴ : Semiring R₁
                                       inst✝¹³ : Semiring R₂
                                       inst✝¹² : Semiring R₃
                                       inst✝¹¹ : AddCommMonoid M
                                       inst✝¹⁰ : AddCommMonoid M₂
                                       inst✝⁹ : AddCommMonoid M₃
                                       inst✝⁸ : AddCommGroup N₂
                                       inst✝⁷ : AddCommGroup N₃
                                       inst✝⁶ : Module R₁ M
                                       inst✝⁵ : Module R₂ M₂
                                       inst✝⁴ : Module R₃ M₃
                                       inst✝³ : Module R₂ N₂
                                       inst✝² : Module R₃ N₃
                                       σ₁₂ : RingHom R₁ R₂
                                       σ₂₃ : RingHom R₂ R₃
                                       σ₁₃ : RingHom R₁ R₃
                                       inst✝¹ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                                       inst✝ : Subsingleton M
                                       f : LinearMap σ₁₂ M M₂
                                       x : M
                                       ⊢ Eq (f x) (Inhabited.default x)
                                     -/
    uniq := fun f => ext fun x => by rw [Subsingleton.elim x 0, map_zero, map_zero] }
                                     /-
                                       🎉 no goals
                                     -/


instance uniqueOfRight [Subsingleton M₂] : Unique (M →ₛₗ[σ₁₂] M₂) :=
  coe_injective.unique


/-- The sum of two linear maps is linear. -/
instance : Add (M →ₛₗ[σ₁₂] M₂) :=
  ⟨fun f g ↦
    { toFun := f + g
                     /-
                       R : Type u_1
                       R₁ : Type u_2
                       R₂ : Type u_3
                       R₃ : Type u_4
                       S : Type u_5
                       S₃ : Type u_6
                       T : Type u_7
                       M : Type u_8
                       M₁ : Type u_9
                       M₂ : Type u_10
                       M₃ : Type u_11
                       N₂ : Type u_12
                       N₃ : Type u_13
                       inst✝¹³ : Semiring R₁
                       inst✝¹² : Semiring R₂
                       inst✝¹¹ : Semiring R₃
                       inst✝¹⁰ : AddCommMonoid M
                       inst✝⁹ : AddCommMonoid M₂
                       inst✝⁸ : AddCommMonoid M₃
                       inst✝⁷ : AddCommGroup N₂
                       inst✝⁶ : AddCommGroup N₃
                       inst✝⁵ : Module R₁ M
                       inst✝⁴ : Module R₂ M₂
                       inst✝³ : Module R₃ M₃
                       inst✝² : Module R₂ N₂
                       inst✝¹ : Module R₃ N₃
                       σ₁₂ : RingHom R₁ R₂
                       σ₂₃ : RingHom R₂ R₃
                       σ₁₃ : RingHom R₁ R₃
                       inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                       f g : LinearMap σ₁₂ M M₂
                       ⊢ ∀ (x y : M), Eq (HAdd.hAdd (⇑f) (⇑g) (HAdd.hAdd x y)) (HAdd.hAdd (HAdd.hAdd  …
                     -/
      map_add' := by simp [add_comm, add_left_comm]
                     /-
                       🎉 no goals
                     -/
                      /-
                        R : Type u_1
                        R₁ : Type u_2
                        R₂ : Type u_3
                        R₃ : Type u_4
                        S : Type u_5
                        S₃ : Type u_6
                        T : Type u_7
                        M : Type u_8
                        M₁ : Type u_9
                        M₂ : Type u_10
                        M₃ : Type u_11
                        N₂ : Type u_12
                        N₃ : Type u_13
                        inst✝¹³ : Semiring R₁
                        inst✝¹² : Semiring R₂
                        inst✝¹¹ : Semiring R₃
                        inst✝¹⁰ : AddCommMonoid M
                        inst✝⁹ : AddCommMonoid M₂
                        inst✝⁸ : AddCommMonoid M₃
                        inst✝⁷ : AddCommGroup N₂
                        inst✝⁶ : AddCommGroup N₃
                        inst✝⁵ : Module R₁ M
                        inst✝⁴ : Module R₂ M₂
                        inst✝³ : Module R₃ M₃
                        inst✝² : Module R₂ N₂
                        inst✝¹ : Module R₃ N₃
                        σ₁₂ : RingHom R₁ R₂
                        σ₂₃ : RingHom R₂ R₃
                        σ₁₃ : RingHom R₁ R₃
                        inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                        f g : LinearMap σ₁₂ M M₂
                        ⊢ ∀ (m : R₁) (x : M), Eq ({ toFun := HAdd.hAdd ⇑f ⇑g, map_add' := ⋯ }.toFun (H …
                      -/
      map_smul' := by simp [smul_add] }⟩
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem add_apply (f g : M →ₛₗ[σ₁₂] M₂) (x : M) : (f + g) x = f x + g x :=
  rfl


theorem add_comp (f : M →ₛₗ[σ₁₂] M₂) (g h : M₂ →ₛₗ[σ₂₃] M₃) :
    ((h + g).comp f : M →ₛₗ[σ₁₃] M₃) = h.comp f + g.comp f :=
  rfl


theorem comp_add (f g : M →ₛₗ[σ₁₂] M₂) (h : M₂ →ₛₗ[σ₂₃] M₃) :
    (h.comp (f + g) : M →ₛₗ[σ₁₃] M₃) = h.comp f + h.comp g :=
  ext fun _ ↦ h.map_add _ _


/-- The type of linear maps is an additive monoid. -/
instance addCommMonoid : AddCommMonoid (M →ₛₗ[σ₁₂] M₂) :=
  DFunLike.coe_injective.addCommMonoid _ rfl (fun _ _ ↦ rfl) fun _ _ ↦ rfl


/-- The negation of a linear map is linear. -/
instance : Neg (M →ₛₗ[σ₁₂] N₂) :=
  ⟨fun f ↦
    { toFun := -f
                     /-
                       R : Type u_1
                       R₁ : Type u_2
                       R₂ : Type u_3
                       R₃ : Type u_4
                       S : Type u_5
                       S₃ : Type u_6
                       T : Type u_7
                       M : Type u_8
                       M₁ : Type u_9
                       M₂ : Type u_10
                       M₃ : Type u_11
                       N₂ : Type u_12
                       N₃ : Type u_13
                       inst✝¹³ : Semiring R₁
                       inst✝¹² : Semiring R₂
                       inst✝¹¹ : Semiring R₃
                       inst✝¹⁰ : AddCommMonoid M
                       inst✝⁹ : AddCommMonoid M₂
                       inst✝⁸ : AddCommMonoid M₃
                       inst✝⁷ : AddCommGroup N₂
                       inst✝⁶ : AddCommGroup N₃
                       inst✝⁵ : Module R₁ M
                       inst✝⁴ : Module R₂ M₂
                       inst✝³ : Module R₃ M₃
                       inst✝² : Module R₂ N₂
                       inst✝¹ : Module R₃ N₃
                       σ₁₂ : RingHom R₁ R₂
                       σ₂₃ : RingHom R₂ R₃
                       σ₁₃ : RingHom R₁ R₃
                       inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                       f : LinearMap σ₁₂ M N₂
                       ⊢ ∀ (x y : M), Eq (Neg.neg (⇑f) (HAdd.hAdd x y)) (HAdd.hAdd (Neg.neg (⇑f) x) ( …
                     -/
      map_add' := by simp [add_comm]
                     /-
                       🎉 no goals
                     -/
                      /-
                        R : Type u_1
                        R₁ : Type u_2
                        R₂ : Type u_3
                        R₃ : Type u_4
                        S : Type u_5
                        S₃ : Type u_6
                        T : Type u_7
                        M : Type u_8
                        M₁ : Type u_9
                        M₂ : Type u_10
                        M₃ : Type u_11
                        N₂ : Type u_12
                        N₃ : Type u_13
                        inst✝¹³ : Semiring R₁
                        inst✝¹² : Semiring R₂
                        inst✝¹¹ : Semiring R₃
                        inst✝¹⁰ : AddCommMonoid M
                        inst✝⁹ : AddCommMonoid M₂
                        inst✝⁸ : AddCommMonoid M₃
                        inst✝⁷ : AddCommGroup N₂
                        inst✝⁶ : AddCommGroup N₃
                        inst✝⁵ : Module R₁ M
                        inst✝⁴ : Module R₂ M₂
                        inst✝³ : Module R₃ M₃
                        inst✝² : Module R₂ N₂
                        inst✝¹ : Module R₃ N₃
                        σ₁₂ : RingHom R₁ R₂
                        σ₂₃ : RingHom R₂ R₃
                        σ₁₃ : RingHom R₁ R₃
                        inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                        f : LinearMap σ₁₂ M N₂
                        ⊢ ∀ (m : R₁) (x : M), Eq ({ toFun := Neg.neg ⇑f, map_add' := ⋯ }.toFun (HSMul. …
                      -/
      map_smul' := by simp }⟩
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem neg_apply (f : M →ₛₗ[σ₁₂] N₂) (x : M) : (-f) x = -f x :=
  rfl


@[simp]
theorem neg_comp (f : M →ₛₗ[σ₁₂] M₂) (g : M₂ →ₛₗ[σ₂₃] N₃) : (-g).comp f = -g.comp f :=
  rfl


@[simp]
theorem comp_neg (f : M →ₛₗ[σ₁₂] N₂) (g : N₂ →ₛₗ[σ₂₃] N₃) : g.comp (-f) = -g.comp f :=
  ext fun _ ↦ g.map_neg _


/-- The subtraction of two linear maps is linear. -/
instance : Sub (M →ₛₗ[σ₁₂] N₂) :=
  ⟨fun f g ↦
    { toFun := f - g
                               /-
                                 R : Type u_1
                                 R₁ : Type u_2
                                 R₂ : Type u_3
                                 R₃ : Type u_4
                                 S : Type u_5
                                 S₃ : Type u_6
                                 T : Type u_7
                                 M : Type u_8
                                 M₁ : Type u_9
                                 M₂ : Type u_10
                                 M₃ : Type u_11
                                 N₂ : Type u_12
                                 N₃ : Type u_13
                                 inst✝¹³ : Semiring R₁
                                 inst✝¹² : Semiring R₂
                                 inst✝¹¹ : Semiring R₃
                                 inst✝¹⁰ : AddCommMonoid M
                                 inst✝⁹ : AddCommMonoid M₂
                                 inst✝⁸ : AddCommMonoid M₃
                                 inst✝⁷ : AddCommGroup N₂
                                 inst✝⁶ : AddCommGroup N₃
                                 inst✝⁵ : Module R₁ M
                                 inst✝⁴ : Module R₂ M₂
                                 inst✝³ : Module R₃ M₃
                                 inst✝² : Module R₂ N₂
                                 inst✝¹ : Module R₃ N₃
                                 σ₁₂ : RingHom R₁ R₂
                                 σ₂₃ : RingHom R₂ R₃
                                 σ₁₃ : RingHom R₁ R₃
                                 inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                                 f g : LinearMap σ₁₂ M N₂
                                 x y : M
                                 ⊢ Eq (HSub.hSub (⇑f) (⇑g) (HAdd.hAdd x y)) (HAdd.hAdd (HSub.hSub (⇑f) (⇑g) x)  …
                               -/
      map_add' := fun x y ↦ by simp only [Pi.sub_apply, map_add, add_sub_add_comm]
                               /-
                                 🎉 no goals
                               -/
                                /-
                                  R : Type u_1
                                  R₁ : Type u_2
                                  R₂ : Type u_3
                                  R₃ : Type u_4
                                  S : Type u_5
                                  S₃ : Type u_6
                                  T : Type u_7
                                  M : Type u_8
                                  M₁ : Type u_9
                                  M₂ : Type u_10
                                  M₃ : Type u_11
                                  N₂ : Type u_12
                                  N₃ : Type u_13
                                  inst✝¹³ : Semiring R₁
                                  inst✝¹² : Semiring R₂
                                  inst✝¹¹ : Semiring R₃
                                  inst✝¹⁰ : AddCommMonoid M
                                  inst✝⁹ : AddCommMonoid M₂
                                  inst✝⁸ : AddCommMonoid M₃
                                  inst✝⁷ : AddCommGroup N₂
                                  inst✝⁶ : AddCommGroup N₃
                                  inst✝⁵ : Module R₁ M
                                  inst✝⁴ : Module R₂ M₂
                                  inst✝³ : Module R₃ M₃
                                  inst✝² : Module R₂ N₂
                                  inst✝¹ : Module R₃ N₃
                                  σ₁₂ : RingHom R₁ R₂
                                  σ₂₃ : RingHom R₂ R₃
                                  σ₁₃ : RingHom R₁ R₃
                                  inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                                  f g : LinearMap σ₁₂ M N₂
                                  r : R₁
                                  x : M
                                  ⊢ Eq ({ toFun := HSub.hSub ⇑f ⇑g, map_add' := ⋯ }.toFun (HSMul.hSMul r x)) (HS …
                                -/
      map_smul' := fun r x ↦ by simp [Pi.sub_apply, map_smul, smul_sub] }⟩
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem sub_apply (f g : M →ₛₗ[σ₁₂] N₂) (x : M) : (f - g) x = f x - g x :=
  rfl


theorem sub_comp (f : M →ₛₗ[σ₁₂] M₂) (g h : M₂ →ₛₗ[σ₂₃] N₃) :
    (g - h).comp f = g.comp f - h.comp f :=
  rfl


theorem comp_sub (f g : M →ₛₗ[σ₁₂] N₂) (h : N₂ →ₛₗ[σ₂₃] N₃) :
    h.comp (g - f) = h.comp g - h.comp f :=
  ext fun _ ↦ h.map_sub _ _


/-- The type of linear maps is an additive group. -/
instance addCommGroup : AddCommGroup (M →ₛₗ[σ₁₂] N₂) :=
  DFunLike.coe_injective.addCommGroup _ rfl (fun _ _ ↦ rfl) (fun _ ↦ rfl) (fun _ _ ↦ rfl)
    (fun _ _ ↦ rfl) fun _ _ ↦ rfl


/-- Evaluation of a `σ₁₂`-linear map at a fixed `a`, as an `AddMonoidHom`. -/
@[simps]
def evalAddMonoidHom (a : M) : (M →ₛₗ[σ₁₂] M₂) →+ M₂ where
  toFun f := f a
  map_add' f g := LinearMap.add_apply f g a
  map_zero' := rfl


/-- `LinearMap.toAddMonoidHom` promoted to an `AddMonoidHom`. -/
@[simps]
def toAddMonoidHom' : (M →ₛₗ[σ₁₂] M₂) →+ M →+ M₂ where
  toFun := toAddMonoidHom
                  /-
                    R : Type u_1
                    R₁ : Type u_2
                    R₂ : Type u_3
                    R₃ : Type u_4
                    S : Type u_5
                    S₃ : Type u_6
                    T : Type u_7
                    M : Type u_8
                    M₁ : Type u_9
                    M₂ : Type u_10
                    M₃ : Type u_11
                    N₂ : Type u_12
                    N₃ : Type u_13
                    inst✝¹³ : Semiring R₁
                    inst✝¹² : Semiring R₂
                    inst✝¹¹ : Semiring R₃
                    inst✝¹⁰ : AddCommMonoid M
                    inst✝⁹ : AddCommMonoid M₂
                    inst✝⁸ : AddCommMonoid M₃
                    inst✝⁷ : AddCommGroup N₂
                    inst✝⁶ : AddCommGroup N₃
                    inst✝⁵ : Module R₁ M
                    inst✝⁴ : Module R₂ M₂
                    inst✝³ : Module R₃ M₃
                    inst✝² : Module R₂ N₂
                    inst✝¹ : Module R₃ N₃
                    σ₁₂ : RingHom R₁ R₂
                    σ₂₃ : RingHom R₂ R₃
                    σ₁₃ : RingHom R₁ R₃
                    inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                    ⊢ Eq (LinearMap.toAddMonoidHom 0) 0
                  -/
  map_zero' := by ext; rfl
                       /-
                         🎉 no goals
                       -/
                 /-
                   R : Type u_1
                   R₁ : Type u_2
                   R₂ : Type u_3
                   R₃ : Type u_4
                   S : Type u_5
                   S₃ : Type u_6
                   T : Type u_7
                   M : Type u_8
                   M₁ : Type u_9
                   M₂ : Type u_10
                   M₃ : Type u_11
                   N₂ : Type u_12
                   N₃ : Type u_13
                   inst✝¹³ : Semiring R₁
                   inst✝¹² : Semiring R₂
                   inst✝¹¹ : Semiring R₃
                   inst✝¹⁰ : AddCommMonoid M
                   inst✝⁹ : AddCommMonoid M₂
                   inst✝⁸ : AddCommMonoid M₃
                   inst✝⁷ : AddCommGroup N₂
                   inst✝⁶ : AddCommGroup N₃
                   inst✝⁵ : Module R₁ M
                   inst✝⁴ : Module R₂ M₂
                   inst✝³ : Module R₃ M₃
                   inst✝² : Module R₂ N₂
                   inst✝¹ : Module R₃ N₃
                   σ₁₂ : RingHom R₁ R₂
                   σ₂₃ : RingHom R₂ R₃
                   σ₁₃ : RingHom R₁ R₃
                   inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                   ⊢ ∀ (x y : LinearMap σ₁₂ M M₂), Eq ({ toFun := LinearMap.toAddMonoidHom, map_z …
                 -/
  map_add' := by intros; ext; rfl
                              /-
                                🎉 no goals
                              -/


/-- If `M` is the zero module, then the identity map of `M` is the zero map. -/
@[simp]
theorem identityMapOfZeroModuleIsZero [Subsingleton M] : id (R := R₁) (M := M) = 0 :=
  Subsingleton.eq_zero id


instance : DistribMulAction S (M →ₛₗ[σ₁₂] M₂) where
  one_smul _ := ext fun _ ↦ one_smul _ _
  mul_smul _ _ _ := ext fun _ ↦ mul_smul _ _ _
  smul_add _ _ _ := ext fun _ ↦ smul_add _ _ _
  smul_zero _ := ext fun _ ↦ smul_zero _


theorem smul_comp (a : S₃) (g : M₂ →ₛₗ[σ₂₃] M₃) (f : M →ₛₗ[σ₁₂] M₂) :
    (a • g).comp f = a • g.comp f :=
  rfl

-- TODO: generalize this to semilinear maps

theorem comp_smul [Module R M₂] [Module R M₃] [SMulCommClass R S M₂] [DistribMulAction S M₃]
    [SMulCommClass R S M₃] [CompatibleSMul M₃ M₂ S R] (g : M₃ →ₗ[R] M₂) (a : S) (f : M →ₗ[R] M₃) :
    g.comp (a • f) = a • g.comp f :=
  ext fun _ ↦ g.map_smul_of_tower _ _


instance module : Module S (M →ₛₗ[σ₁₂] M₂) where
  add_smul _ _ _ := ext fun _ ↦ add_smul _ _ _
  zero_smul _ := ext fun _ ↦ zero_smul _ _


variable (R S M N) in
@[simp]
lemma restrictScalars_zero : (0 : M →ₗ[S] N).restrictScalars R = 0 :=
  rfl


@[simp]
theorem restrictScalars_add (f g : M →ₗ[S] N) :
    (f + g).restrictScalars R = f.restrictScalars R + g.restrictScalars R :=
  rfl


@[simp]
theorem restrictScalars_neg {M N : Type*} [AddCommGroup M] [AddCommGroup N]
    [Module R M] [Module R N] [Module S M] [Module S N] [CompatibleSMul M N R S]
    (f : M →ₗ[S] N) : (-f).restrictScalars R = -f.restrictScalars R :=
  rfl


@[simp]
theorem restrictScalars_smul (c : R₁) (f : M →ₗ[S] N) :
    (c • f).restrictScalars R = c • f.restrictScalars R :=
  rfl


@[simp]
lemma restrictScalars_comp [AddCommMonoid P] [Module S P] [Module R P]
    [CompatibleSMul N P R S] [CompatibleSMul M P R S] (f : N →ₗ[S] P) (g : M →ₗ[S] N) :
    (f ∘ₗ g).restrictScalars R = f.restrictScalars R ∘ₗ g.restrictScalars R := by
  /-
    R : Type u_14
    S : Type u_15
    M : Type u_16
    N : Type u_17
    P : Type u_18
    inst✝¹³ : Semiring R
    inst✝¹² : Semiring S
    inst✝¹¹ : AddCommMonoid M
    inst✝¹⁰ : AddCommMonoid N
    inst✝⁹ : Module R M
    inst✝⁸ : Module R N
    inst✝⁷ : Module S M
    inst✝⁶ : Module S N
    inst✝⁵ : LinearMap.CompatibleSMul M N R S
    inst✝⁴ : AddCommMonoid P
    inst✝³ : Module S P
    inst✝² : Module R P
    inst✝¹ : LinearMap.CompatibleSMul N P R S
    inst✝ : LinearMap.CompatibleSMul M P R S
    f : LinearMap (RingHom.id S) N P
    g : LinearMap (RingHom.id S) M N
    ⊢ Eq (↑R (f.comp g)) ((↑R f).comp (↑R g))
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma restrictScalars_trans {T : Type*} [CommSemiring T] [Module T M] [Module T N]
    [CompatibleSMul M N S T] [CompatibleSMul M N R T] (f : M →ₗ[T] N) :
    (f.restrictScalars S).restrictScalars R = f.restrictScalars R :=
  rfl


/-- `LinearMap.restrictScalars` as a `LinearMap`. -/
@[simps apply]
def restrictScalarsₗ : (M →ₗ[S] N) →ₗ[R₁] M →ₗ[R] N where
  toFun := restrictScalars R
  map_add' := restrictScalars_add
  map_smul' := restrictScalars_smul


