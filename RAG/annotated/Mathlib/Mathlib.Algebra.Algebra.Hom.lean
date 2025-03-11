/-- Defining the homomorphism in the category R-Alg. -/
-- @[nolint has_nonempty_instance] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): linter not ported yet
structure AlgHom (R : Type u) (A : Type v) (B : Type w) [CommSemiring R] [Semiring A] [Semiring B]
  [Algebra R A] [Algebra R B] extends RingHom A B where
  commutes' : ∀ r : R, toFun (algebraMap R A r) = algebraMap R B r


@[inherit_doc AlgHom]
infixr:25 " →ₐ " => AlgHom _


@[inherit_doc]
notation:25 A " →ₐ[" R "] " B => AlgHom R A B


/-- `AlgHomClass F R A B` asserts `F` is a type of bundled algebra homomorphisms
from `A` to `B`. -/
class AlgHomClass (F : Type*) (R A B : outParam Type*)
  [CommSemiring R] [Semiring A] [Semiring B] [Algebra R A] [Algebra R B]
  [FunLike F A B] extends RingHomClass F A B : Prop where
  commutes : ∀ (f : F) (r : R), f (algebraMap R A r) = algebraMap R B r

-- For now, don't replace `AlgHom.commutes` and `AlgHomClass.commutes` with the more generic lemma.
-- The file `Mathlib.NumberTheory.NumberField.CanonicalEmbedding.FundamentalCone` slows down by
-- 15% if we would do so (see benchmark on PR https://github.com/leanprover-community/mathlib4/pull/18040).
-- attribute [simp] AlgHomClass.commutes


instance (priority := 100) linearMapClass [AlgHomClass F R A B] : LinearMapClass F R A B :=
  { ‹AlgHomClass F R A B› with
    map_smulₛₗ := fun f r x => by
      /-
        R : Type u_1
        A : Type u_2
        B : Type u_3
        F : Type u_4
        inst✝⁶ : CommSemiring R
        inst✝⁵ : Semiring A
        inst✝⁴ : Semiring B
        inst✝³ : Algebra R A
        inst✝² : Algebra R B
        inst✝¹ : FunLike F A B
        inst✝ : AlgHomClass F R A B
        f : F
        r : R
        x : A
        ⊢ Eq (f (HSMul.hSMul r x)) (HSMul.hSMul ((RingHom.id R) r) (f x))
      -/
      simp only [Algebra.smul_def, map_mul, commutes, RingHom.id_apply] }
      /-
        🎉 no goals
      -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11445): A new definition underlying a coercion `↑`.

/-- Turn an element of a type `F` satisfying `AlgHomClass F α β` into an actual
`AlgHom`. This is declared as the default coercion from `F` to `α →+* β`. -/
@[coe]
def toAlgHom {F : Type*} [FunLike F A B] [AlgHomClass F R A B] (f : F) : A →ₐ[R] B where
  __ := (f : A →+* B)
  toFun := f
  commutes' := AlgHomClass.commutes f


instance coeTC {F : Type*} [FunLike F A B] [AlgHomClass F R A B] : CoeTC F (A →ₐ[R] B) :=
  ⟨AlgHomClass.toAlgHom⟩


instance funLike : FunLike (A →ₐ[R] B) A B where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      R : Type u
      A : Type v
      B : Type w
      C : Type u₁
      D : Type v₁
      inst✝⁸ : CommSemiring R
      inst✝⁷ : Semiring A
      inst✝⁶ : Semiring B
      inst✝⁵ : Semiring C
      inst✝⁴ : Semiring D
      inst✝³ : Algebra R A
      inst✝² : Algebra R B
      inst✝¹ : Algebra R C
      inst✝ : Algebra R D
      f g : AlgHom R A B
      h : Eq ((fun f => (↑↑f.toRingHom).toFun) f) ((fun f => (↑↑f.toRingHom).toFun) g)
      ⊢ Eq f g
    -/
    rcases f with ⟨⟨⟨⟨_, _⟩, _⟩, _, _⟩, _⟩
    /-
      case mk.mk.mk.mk
      R : Type u
      A : Type v
      B : Type w
      C : Type u₁
      D : Type v₁
      inst✝⁸ : CommSemiring R
      inst✝⁷ : Semiring A
      inst✝⁶ : Semiring B
      inst✝⁵ : Semiring C
      inst✝⁴ : Semiring D
      inst✝³ : Algebra R A
      inst✝² : Algebra R B
      inst✝¹ : Algebra R C
      inst✝ : Algebra R D
      g : AlgHom R A B
      toFun✝ : A → B
      map_one'✝ : Eq (toFun✝ 1) 1
      map_mul'✝ : ∀ (x y : A), Eq ({ toFun := toFun✝, map_one' := map_one'✝ }.toFun  …
      map_zero'✝ : Eq ((↑{ toFun := toFun✝, map_one' := map_one'✝, map_mul' := map_m …
      map_add'✝ : ∀ (x y : A), Eq ((↑{ toFun := toFun✝, map_one' := map_one'✝, map_m …
      commutes'✝ : ∀ (r : R), Eq ((↑↑{ toFun := toFun✝, map_one' := map_one'✝, map_m …
      h : Eq ((fun f => (↑↑f.toRingHom).toFun) { toFun := toFun✝, map_one' := map_on …
      ⊢ Eq { toFun := toFun✝, map_one' := map_one'✝, map_mul' := map_mul'✝, map_zero …
    -/
    rcases g with ⟨⟨⟨⟨_, _⟩, _⟩, _, _⟩, _⟩
    /-
      case mk.mk.mk.mk.mk.mk.mk.mk
      R : Type u
      A : Type v
      B : Type w
      C : Type u₁
      D : Type v₁
      inst✝⁸ : CommSemiring R
      inst✝⁷ : Semiring A
      inst✝⁶ : Semiring B
      inst✝⁵ : Semiring C
      inst✝⁴ : Semiring D
      inst✝³ : Algebra R A
      inst✝² : Algebra R B
      inst✝¹ : Algebra R C
      inst✝ : Algebra R D
      toFun✝¹ : A → B
      map_one'✝¹ : Eq (toFun✝¹ 1) 1
      map_mul'✝¹ : ∀ (x y : A), Eq ({ toFun := toFun✝¹, map_one' := map_one'✝¹ }.toF …
      map_zero'✝¹ : Eq ((↑{ toFun := toFun✝¹, map_one' := map_one'✝¹, map_mul' := ma …
      map_add'✝¹ : ∀ (x y : A), Eq ((↑{ toFun := toFun✝¹, map_one' := map_one'✝¹, ma …
      commutes'✝¹ : ∀ (r : R), Eq ((↑↑{ toFun := toFun✝¹, map_one' := map_one'✝¹, ma …
      toFun✝ : A → B
      map_one'✝ : Eq (toFun✝ 1) 1
      map_mul'✝ : ∀ (x y : A), Eq ({ toFun := toFun✝, map_one' := map_one'✝ }.toFun  …
      map_zero'✝ : Eq ((↑{ toFun := toFun✝, map_one' := map_one'✝, map_mul' := map_m …
      map_add'✝ : ∀ (x y : A), Eq ((↑{ toFun := toFun✝, map_one' := map_one'✝, map_m …
      commutes'✝ : ∀ (r : R), Eq ((↑↑{ toFun := toFun✝, map_one' := map_one'✝, map_m …
      h : Eq ((fun f => (↑↑f.toRingHom).toFun) { toFun := toFun✝¹, map_one' := map_o …
      ⊢ Eq { toFun := toFun✝¹, map_one' := map_one'✝¹, map_mul' := map_mul'✝¹, map_z …
    -/
    congr
    /-
      🎉 no goals
    -/

-- Porting note: This instance is moved.

instance algHomClass : AlgHomClass (A →ₐ[R] B) R A B where
  map_add f := f.map_add'
  map_zero f := f.map_zero'
  map_mul f := f.map_mul'
  map_one f := f.map_one'
  commutes f := f.commutes'


/-- See Note [custom simps projection] -/
def Simps.apply {R : Type u} {α : Type v} {β : Type w} [CommSemiring R]
    [Semiring α] [Semiring β] [Algebra R α] [Algebra R β] (f : α →ₐ[R] β) : α → β := f


@[simp]
protected theorem coe_coe {F : Type*} [FunLike F A B] [AlgHomClass F R A B] (f : F) :
    ⇑(f : A →ₐ[R] B) = f :=
  rfl


@[simp]
theorem toFun_eq_coe (f : A →ₐ[R] B) : f.toFun = f :=
  rfl

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11445): A new definition underlying a coercion `↑`.

@[coe]
def toMonoidHom' (f : A →ₐ[R] B) : A →* B := (f : A →+* B)


instance coeOutMonoidHom : CoeOut (A →ₐ[R] B) (A →* B) :=
  ⟨AlgHom.toMonoidHom'⟩

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11445): A new definition underlying a coercion `↑`.

@[coe]
def toAddMonoidHom' (f : A →ₐ[R] B) : A →+ B := (f : A →+* B)


instance coeOutAddMonoidHom : CoeOut (A →ₐ[R] B) (A →+ B) :=
  ⟨AlgHom.toAddMonoidHom'⟩

-- Porting note: Lean 3: `@[simp, norm_cast] coe_mk`
--               Lean 4: `@[simp] coe_mk` & `@[norm_cast] coe_mks`

@[simp]
theorem coe_mk {f : A →+* B} (h) : ((⟨f, h⟩ : A →ₐ[R] B) : A → B) = f :=
  rfl


@[norm_cast]
theorem coe_mks {f : A → B} (h₁ h₂ h₃ h₄ h₅) : ⇑(⟨⟨⟨⟨f, h₁⟩, h₂⟩, h₃, h₄⟩, h₅⟩ : A →ₐ[R] B) = f :=
  rfl

-- Porting note: This theorem is new.

@[simp, norm_cast]
theorem coe_ringHom_mk {f : A →+* B} (h) : ((⟨f, h⟩ : A →ₐ[R] B) : A →+* B) = f :=
  rfl

-- make the coercion the simp-normal form

@[simp]
theorem toRingHom_eq_coe (f : A →ₐ[R] B) : f.toRingHom = f :=
  rfl


@[simp, norm_cast]
theorem coe_toRingHom (f : A →ₐ[R] B) : ⇑(f : A →+* B) = f :=
  rfl


@[simp, norm_cast]
theorem coe_toMonoidHom (f : A →ₐ[R] B) : ⇑(f : A →* B) = f :=
  rfl


@[simp, norm_cast]
theorem coe_toAddMonoidHom (f : A →ₐ[R] B) : ⇑(f : A →+ B) = f :=
  rfl


theorem coe_fn_injective : @Function.Injective (A →ₐ[R] B) (A → B) (↑) :=
  DFunLike.coe_injective


theorem coe_fn_inj {φ₁ φ₂ : A →ₐ[R] B} : (φ₁ : A → B) = φ₂ ↔ φ₁ = φ₂ :=
  DFunLike.coe_fn_eq


theorem coe_ringHom_injective : Function.Injective ((↑) : (A →ₐ[R] B) → A →+* B) := fun φ₁ φ₂ H =>
  coe_fn_injective <| show ((φ₁ : A →+* B) : A → B) = ((φ₂ : A →+* B) : A → B) from congr_arg _ H


theorem coe_monoidHom_injective : Function.Injective ((↑) : (A →ₐ[R] B) → A →* B) :=
  RingHom.coe_monoidHom_injective.comp coe_ringHom_injective


theorem coe_addMonoidHom_injective : Function.Injective ((↑) : (A →ₐ[R] B) → A →+ B) :=
  RingHom.coe_addMonoidHom_injective.comp coe_ringHom_injective


protected theorem congr_fun {φ₁ φ₂ : A →ₐ[R] B} (H : φ₁ = φ₂) (x : A) : φ₁ x = φ₂ x :=
  DFunLike.congr_fun H x


protected theorem congr_arg (φ : A →ₐ[R] B) {x y : A} (h : x = y) : φ x = φ y :=
  DFunLike.congr_arg φ h


@[ext]
theorem ext {φ₁ φ₂ : A →ₐ[R] B} (H : ∀ x, φ₁ x = φ₂ x) : φ₁ = φ₂ :=
  DFunLike.ext _ _ H


@[simp]
theorem mk_coe {f : A →ₐ[R] B} (h₁ h₂ h₃ h₄ h₅) : (⟨⟨⟨⟨f, h₁⟩, h₂⟩, h₃, h₄⟩, h₅⟩ : A →ₐ[R] B) = f :=
  rfl


@[simp]
theorem commutes (r : R) : φ (algebraMap R A r) = algebraMap R B r :=
  φ.commutes' r


theorem comp_algebraMap : (φ : A →+* B).comp (algebraMap R A) = algebraMap R B :=
  RingHom.ext <| φ.commutes


@[deprecated map_add (since := "2024-06-26")]
protected theorem map_add (r s : A) : φ (r + s) = φ r + φ s :=
  map_add _ _ _


@[deprecated map_zero (since := "2024-06-26")]
protected theorem map_zero : φ 0 = 0 :=
  map_zero _


@[deprecated map_mul (since := "2024-06-26")]
protected theorem map_mul (x y) : φ (x * y) = φ x * φ y :=
  map_mul _ _ _


@[deprecated map_one (since := "2024-06-26")]
protected theorem map_one : φ 1 = 1 :=
  map_one _


@[deprecated map_pow (since := "2024-06-26")]
protected theorem map_pow (x : A) (n : ℕ) : φ (x ^ n) = φ x ^ n :=
  map_pow _ _ _


@[deprecated map_smul (since := "2024-06-26")]
protected theorem map_smul (r : R) (x : A) : φ (r • x) = r • φ x :=
  map_smul _ _ _


@[deprecated map_sum (since := "2024-06-26")]
protected theorem map_sum {ι : Type*} (f : ι → A) (s : Finset ι) :
    φ (∑ x ∈ s, f x) = ∑ x ∈ s, φ (f x) :=
  map_sum _ _ _


/-- If a `RingHom` is `R`-linear, then it is an `AlgHom`. -/
def mk' (f : A →+* B) (h : ∀ (c : R) (x), f (c • x) = c • f x) : A →ₐ[R] B :=
  { f with
    toFun := f
                             /-
                               R : Type u
                               A : Type v
                               B : Type w
                               C : Type u₁
                               D : Type v₁
                               inst✝⁸ : CommSemiring R
                               inst✝⁷ : Semiring A
                               inst✝⁶ : Semiring B
                               inst✝⁵ : Semiring C
                               inst✝⁴ : Semiring D
                               inst✝³ : Algebra R A
                               inst✝² : Algebra R B
                               inst✝¹ : Algebra R C
                               inst✝ : Algebra R D
                               φ : AlgHom R A B
                               f : RingHom A B
                               h : ∀ (c : R) (x : A), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                               c : R
                               ⊢ Eq ((↑↑{ toFun := ⇑f, map_one' := ⋯, map_mul' := ⋯, map_zero' := ⋯, map_add' …
                             -/
    commutes' := fun c => by simp only [Algebra.algebraMap_eq_smul_one, h, f.map_one] }
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem coe_mk' (f : A →+* B) (h : ∀ (c : R) (x), f (c • x) = c • f x) : ⇑(mk' f h) = f :=
  rfl


/-- Identity map as an `AlgHom`. -/
protected def id : A →ₐ[R] A :=
  { RingHom.id A with commutes' := fun _ => rfl }


@[simp]
theorem coe_id : ⇑(AlgHom.id R A) = id :=
  rfl


@[simp]
theorem id_toRingHom : (AlgHom.id R A : A →+* A) = RingHom.id _ :=
  rfl


theorem id_apply (p : A) : AlgHom.id R A p = p :=
  rfl


/-- Composition of algebra homeomorphisms. -/
def comp (φ₁ : B →ₐ[R] C) (φ₂ : A →ₐ[R] B) : A →ₐ[R] C :=
  { φ₁.toRingHom.comp ↑φ₂ with
                                 /-
                                   R : Type u
                                   A : Type v
                                   B : Type w
                                   C : Type u₁
                                   D : Type v₁
                                   inst✝⁸ : CommSemiring R
                                   inst✝⁷ : Semiring A
                                   inst✝⁶ : Semiring B
                                   inst✝⁵ : Semiring C
                                   inst✝⁴ : Semiring D
                                   inst✝³ : Algebra R A
                                   inst✝² : Algebra R B
                                   inst✝¹ : Algebra R C
                                   inst✝ : Algebra R D
                                   φ : AlgHom R A B
                                   φ₁ : AlgHom R B C
                                   φ₂ : AlgHom R A B
                                   r : R
                                   ⊢ Eq ((↑↑__src✝).toFun ((algebraMap R A) r)) ((algebraMap R C) r)
                                 -/
    commutes' := fun r : R => by rw [← φ₁.commutes, ← φ₂.commutes]; rfl }
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
theorem coe_comp (φ₁ : B →ₐ[R] C) (φ₂ : A →ₐ[R] B) : ⇑(φ₁.comp φ₂) = φ₁ ∘ φ₂ :=
  rfl


theorem comp_apply (φ₁ : B →ₐ[R] C) (φ₂ : A →ₐ[R] B) (p : A) : φ₁.comp φ₂ p = φ₁ (φ₂ p) :=
  rfl


theorem comp_toRingHom (φ₁ : B →ₐ[R] C) (φ₂ : A →ₐ[R] B) :
    (φ₁.comp φ₂ : A →+* C) = (φ₁ : B →+* C).comp ↑φ₂ :=
  rfl


@[simp]
theorem comp_id : φ.comp (AlgHom.id R A) = φ :=
  rfl


@[simp]
theorem id_comp : (AlgHom.id R B).comp φ = φ :=
  rfl


theorem comp_assoc (φ₁ : C →ₐ[R] D) (φ₂ : B →ₐ[R] C) (φ₃ : A →ₐ[R] B) :
    (φ₁.comp φ₂).comp φ₃ = φ₁.comp (φ₂.comp φ₃) :=
  rfl


/-- R-Alg ⥤ R-Mod -/
def toLinearMap : A →ₗ[R] B where
  toFun := φ
  map_add' := map_add _
  map_smul' := map_smul _


@[simp]
theorem toLinearMap_apply (p : A) : φ.toLinearMap p = φ p :=
  rfl


theorem toLinearMap_injective :
    Function.Injective (toLinearMap : _ → A →ₗ[R] B) := fun _φ₁ _φ₂ h =>
  ext <| LinearMap.congr_fun h


@[simp]
theorem comp_toLinearMap (f : A →ₐ[R] B) (g : B →ₐ[R] C) :
    (g.comp f).toLinearMap = g.toLinearMap.comp f.toLinearMap :=
  rfl


@[simp]
theorem toLinearMap_id : toLinearMap (AlgHom.id R A) = LinearMap.id :=
  rfl


/-- Promote a `LinearMap` to an `AlgHom` by supplying proofs about the behavior on `1` and `*`. -/
@[simps]
def ofLinearMap (f : A →ₗ[R] B) (map_one : f 1 = 1) (map_mul : ∀ x y, f (x * y) = f x * f y) :
    A →ₐ[R] B :=
  { f.toAddMonoidHom with
    toFun := f
    map_one' := map_one
    map_mul' := map_mul
                             /-
                               R : Type u
                               A : Type v
                               B : Type w
                               C : Type u₁
                               D : Type v₁
                               inst✝⁸ : CommSemiring R
                               inst✝⁷ : Semiring A
                               inst✝⁶ : Semiring B
                               inst✝⁵ : Semiring C
                               inst✝⁴ : Semiring D
                               inst✝³ : Algebra R A
                               inst✝² : Algebra R B
                               inst✝¹ : Algebra R C
                               inst✝ : Algebra R D
                               φ : AlgHom R A B
                               f : LinearMap (RingHom.id R) A B
                               map_one : Eq (f 1) 1
                               map_mul : ∀ (x y : A), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
                               c : R
                               ⊢ Eq ((↑↑{ toFun := ⇑f, map_one' := map_one, map_mul' := map_mul, map_zero' := …
                             -/
    commutes' := fun c => by simp only [Algebra.algebraMap_eq_smul_one, f.map_smul, map_one] }
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem ofLinearMap_toLinearMap (map_one) (map_mul) :
    ofLinearMap φ.toLinearMap map_one map_mul = φ :=
  rfl


@[simp]
theorem toLinearMap_ofLinearMap (f : A →ₗ[R] B) (map_one) (map_mul) :
    toLinearMap (ofLinearMap f map_one map_mul) = f :=
  rfl


@[simp]
theorem ofLinearMap_id (map_one) (map_mul) :
    ofLinearMap LinearMap.id map_one map_mul = AlgHom.id R A :=
  rfl


theorem map_smul_of_tower {R'} [SMul R' A] [SMul R' B] [LinearMap.CompatibleSMul A B R' R] (r : R')
    (x : A) : φ (r • x) = r • φ x :=
  φ.toLinearMap.map_smul_of_tower r x


@[deprecated map_list_prod (since := "2024-06-26")]
protected theorem map_list_prod (s : List A) : φ s.prod = (s.map φ).prod :=
  map_list_prod φ s


@[simps (config := .lemmasOnly) toSemigroup_toMul_mul toOne_one]
instance End : Monoid (A →ₐ[R] A) where
  mul := comp
  mul_assoc _ _ _ := rfl
  one := AlgHom.id R A
  one_mul _ := rfl
  mul_one _ := rfl


@[simp]
theorem one_apply (x : A) : (1 : A →ₐ[R] A) x = x :=
  rfl


@[simp]
theorem mul_apply (φ ψ : A →ₐ[R] A) (x : A) : (φ * ψ) x = φ (ψ x) :=
  rfl


theorem algebraMap_eq_apply (f : A →ₐ[R] B) {y : R} {x : A} (h : algebraMap R A y = x) :
    algebraMap R B y = f x :=
  h ▸ (f.commutes _).symm


@[deprecated map_neg (since := "2024-06-26")]
protected theorem map_neg (x) : φ (-x) = -φ x :=
  map_neg _ _


@[deprecated map_sub (since := "2024-06-26")]
protected theorem map_sub (x y) : φ (x - y) = φ x - φ y :=
  map_sub _ _ _


/-- Reinterpret a `RingHom` as an `ℕ`-algebra homomorphism. -/
def toNatAlgHom [Semiring R] [Semiring S] (f : R →+* S) : R →ₐ[ℕ] S :=
  { f with
    toFun := f
                             /-
                               R : Type u_1
                               S : Type u_2
                               inst✝¹ : Semiring R
                               inst✝ : Semiring S
                               f : RingHom R S
                               n : Nat
                               ⊢ Eq ((↑↑{ toFun := ⇑f, map_one' := ⋯, map_mul' := ⋯, map_zero' := ⋯, map_add' …
                             -/
    commutes' := fun n => by simp }
                             /-
                               🎉 no goals
                             -/


@[simp]
lemma toNatAlgHom_coe [Semiring R] [Semiring S] (f : R →+* S) :
    ⇑f.toNatAlgHom = ⇑f := rfl


lemma toNatAlgHom_apply [Semiring R] [Semiring S] (f : R →+* S) (x : R) :
    f.toNatAlgHom x = f x := rfl


/-- Reinterpret a `RingHom` as a `ℤ`-algebra homomorphism. -/
def toIntAlgHom [Ring R] [Ring S] (f : R →+* S) : R →ₐ[ℤ] S :=
                                    /-
                                      R : Type u_1
                                      S : Type u_2
                                      inst✝¹ : Ring R
                                      inst✝ : Ring S
                                      f : RingHom R S
                                      n : Int
                                      ⊢ Eq ((↑↑f).toFun ((algebraMap Int R) n)) ((algebraMap Int S) n)
                                    -/
  { f with commutes' := fun n => by simp }
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
lemma toIntAlgHom_coe [Ring R] [Ring S] (f : R →+* S) :
    ⇑f.toIntAlgHom = ⇑f := rfl


lemma toIntAlgHom_apply [Ring R] [Ring S] (f : R →+* S) (x : R) :
    f.toIntAlgHom x = f x := rfl


lemma toIntAlgHom_injective [Ring R] [Ring S] :
    Function.Injective (RingHom.toIntAlgHom : (R →+* S) → _) :=
  fun _ _ e ↦ DFunLike.ext _ _ (fun x ↦ DFunLike.congr_fun e x)


/-- `AlgebraMap` as an `AlgHom`. -/
def ofId : R →ₐ[R] A :=
  { algebraMap R A with commutes' := fun _ => rfl }


theorem ofId_apply (r) : ofId R A r = algebraMap R A r :=
  rfl


/-- This is a special case of a more general instance that we define in a later file. -/
instance subsingleton_id : Subsingleton (R →ₐ[R] A) :=
  ⟨fun f g => AlgHom.ext fun _ => (f.commutes _).trans (g.commutes _).symm⟩


/-- This ext lemma closes trivial subgoals create when chaining heterobasic ext lemmas. -/
@[ext high]
theorem ext_id (f g : R →ₐ[R] A) : f = g := Subsingleton.elim _ _


instance : MulDistribMulAction (A →ₐ[R] A) Aˣ where
  smul f := Units.map f
                   /-
                     R : Type u
                     A : Type v
                     inst✝² : CommSemiring R
                     inst✝¹ : Semiring A
                     inst✝ : Algebra R A
                     x✝ : Units A
                     ⊢ Eq (HSMul.hSMul 1 x✝) x✝
                   -/
  one_smul _ := by ext; rfl
                        /-
                          🎉 no goals
                        -/
                       /-
                         R : Type u
                         A : Type v
                         inst✝² : CommSemiring R
                         inst✝¹ : Semiring A
                         inst✝ : Algebra R A
                         x✝² x✝¹ : AlgHom R A A
                         x✝ : Units A
                         ⊢ Eq (HSMul.hSMul (HMul.hMul x✝² x✝¹) x✝) (HSMul.hSMul x✝² (HSMul.hSMul x✝¹ x✝))
                       -/
  mul_smul _ _ _ := by ext; rfl
                            /-
                              🎉 no goals
                            -/
                       /-
                         R : Type u
                         A : Type v
                         inst✝² : CommSemiring R
                         inst✝¹ : Semiring A
                         inst✝ : Algebra R A
                         x✝² : AlgHom R A A
                         x✝¹ x✝ : Units A
                         ⊢ Eq (HSMul.hSMul x✝² (HMul.hMul x✝¹ x✝)) (HMul.hMul (HSMul.hSMul x✝² x✝¹) (HS …
                       -/
  smul_mul _ _ _ := by ext; exact map_mul _ _ _
                            /-
                              🎉 no goals
                            -/
                   /-
                     R : Type u
                     A : Type v
                     inst✝² : CommSemiring R
                     inst✝¹ : Semiring A
                     inst✝ : Algebra R A
                     x✝ : AlgHom R A A
                     ⊢ Eq (HSMul.hSMul x✝ 1) 1
                   -/
  smul_one _ := by ext; exact map_one _
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem smul_units_def (f : A →ₐ[R] A) (x : Aˣ) :
    f • x = Units.map (f : A →* A) x := rfl


lemma algebraMapSubmonoid_map_eq (f : A →ₐ[R] B) :
    (algebraMapSubmonoid A M).map f = algebraMapSubmonoid B M := by
  /-
    R : Type u
    A : Type v
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    M : Submonoid R
    B : Type w
    inst✝¹ : CommRing B
    inst✝ : Algebra R B
    f : AlgHom R A B
    ⊢ Eq (Submonoid.map f (Algebra.algebraMapSubmonoid A M)) (Algebra.algebraMapSu …
  -/
  ext x
  /-
    case h
    R : Type u
    A : Type v
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    M : Submonoid R
    B : Type w
    inst✝¹ : CommRing B
    inst✝ : Algebra R B
    f : AlgHom R A B
    x : B
    ⊢ Iff (Membership.mem (Submonoid.map f (Algebra.algebraMapSubmonoid A M)) x) ( …
  -/
  constructor
    /-
      case h.mp
      R : Type u
      A : Type v
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      M : Submonoid R
      B : Type w
      inst✝¹ : CommRing B
      inst✝ : Algebra R B
      f : AlgHom R A B
      x : B
      ⊢ Membership.mem (Submonoid.map f (Algebra.algebraMapSubmonoid A M)) x → Membe …
    -/
  · rintro ⟨a, ⟨r, hr, rfl⟩, rfl⟩
    /-
      case h.mp.intro.intro.intro.intro
      R : Type u
      A : Type v
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      M : Submonoid R
      B : Type w
      inst✝¹ : CommRing B
      inst✝ : Algebra R B
      f : AlgHom R A B
      r : R
      hr : Membership.mem (↑M) r
      ⊢ Membership.mem (Algebra.algebraMapSubmonoid B M) (f ((algebraMap R A) r))
    -/
    simp only [AlgHom.commutes]
    /-
      case h.mp.intro.intro.intro.intro
      R : Type u
      A : Type v
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      M : Submonoid R
      B : Type w
      inst✝¹ : CommRing B
      inst✝ : Algebra R B
      f : AlgHom R A B
      r : R
      hr : Membership.mem (↑M) r
      ⊢ Membership.mem (Algebra.algebraMapSubmonoid B M) ((algebraMap R B) r)
    -/
    use r
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u
      A : Type v
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      M : Submonoid R
      B : Type w
      inst✝¹ : CommRing B
      inst✝ : Algebra R B
      f : AlgHom R A B
      x : B
      ⊢ Membership.mem (Algebra.algebraMapSubmonoid B M) x → Membership.mem (Submono …
    -/
  · rintro ⟨r, hr, rfl⟩
    /-
      case h.mpr.intro.intro
      R : Type u
      A : Type v
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      M : Submonoid R
      B : Type w
      inst✝¹ : CommRing B
      inst✝ : Algebra R B
      f : AlgHom R A B
      r : R
      hr : Membership.mem (↑M) r
      ⊢ Membership.mem (Submonoid.map f (Algebra.algebraMapSubmonoid A M)) ((algebra …
    -/
    simp only [Submonoid.mem_map]
    /-
      case h.mpr.intro.intro
      R : Type u
      A : Type v
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      M : Submonoid R
      B : Type w
      inst✝¹ : CommRing B
      inst✝ : Algebra R B
      f : AlgHom R A B
      r : R
      hr : Membership.mem (↑M) r
      ⊢ Exists fun x => And (Membership.mem (Algebra.algebraMapSubmonoid A M) x) (Eq …
    -/
    use (algebraMap R A r)
    /-
      case h
      R : Type u
      A : Type v
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      M : Submonoid R
      B : Type w
      inst✝¹ : CommRing B
      inst✝ : Algebra R B
      f : AlgHom R A B
      r : R
      hr : Membership.mem (↑M) r
      ⊢ And (Membership.mem (Algebra.algebraMapSubmonoid A M) ((algebraMap R A) r))  …
    -/
    simp only [AlgHom.commutes, and_true]
    /-
      case h
      R : Type u
      A : Type v
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      M : Submonoid R
      B : Type w
      inst✝¹ : CommRing B
      inst✝ : Algebra R B
      f : AlgHom R A B
      r : R
      hr : Membership.mem (↑M) r
      ⊢ Membership.mem (Algebra.algebraMapSubmonoid A M) ((algebraMap R A) r)
    -/
    use r
    /-
      🎉 no goals
    -/


lemma algebraMapSubmonoid_le_comap (f : A →ₐ[R] B) :
    algebraMapSubmonoid A M ≤ (algebraMapSubmonoid B M).comap f.toRingHom := by
  /-
    R : Type u
    A : Type v
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    M : Submonoid R
    B : Type w
    inst✝¹ : CommRing B
    inst✝ : Algebra R B
    f : AlgHom R A B
    ⊢ LE.le (Algebra.algebraMapSubmonoid A M) (Submonoid.comap f.toRingHom (Algebr …
  -/
  rw [← algebraMapSubmonoid_map_eq M f]
  /-
    R : Type u
    A : Type v
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    M : Submonoid R
    B : Type w
    inst✝¹ : CommRing B
    inst✝ : Algebra R B
    f : AlgHom R A B
    ⊢ LE.le (Algebra.algebraMapSubmonoid A M) (Submonoid.comap f.toRingHom (Submon …
  -/
  exact Submonoid.le_comap_map (Algebra.algebraMapSubmonoid A M)
  /-
    🎉 no goals
  -/


/-- Each element of the monoid defines an algebra homomorphism.

This is a stronger version of `MulSemiringAction.toRingHom` and
`DistribMulAction.toLinearMap`. -/
@[simps]
def toAlgHom (m : M) : A →ₐ[R] A :=
  { MulSemiringAction.toRingHom _ _ m with
    toFun := fun a => m • a
    commutes' := smul_algebraMap _ }


theorem toAlgHom_injective [FaithfulSMul M A] :
    Function.Injective (MulSemiringAction.toAlgHom R A : M → A →ₐ[R] A) := fun _m₁ _m₂ h =>
  eq_of_smul_eq_smul fun r => AlgHom.ext_iff.1 h r


