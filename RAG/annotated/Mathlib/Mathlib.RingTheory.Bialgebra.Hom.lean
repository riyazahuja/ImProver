/-- Given `R`-algebras `A, B` with comultiplication maps `Δ_A, Δ_B` and counit maps
`ε_A, ε_B`, an `R`-bialgebra homomorphism `A →ₐc[R] B` is an `R`-algebra map `f` such that
`ε_B ∘ f = ε_A` and `(f ⊗ f) ∘ Δ_A = Δ_B ∘ f`. -/
structure BialgHom (R A B : Type*) [CommSemiring R]
    [Semiring A] [Algebra R A] [Semiring B] [Algebra R B]
    [CoalgebraStruct R A] [CoalgebraStruct R B] extends A →ₗc[R] B, A →* B


@[inherit_doc BialgHom]
infixr:25 " →ₐc " => BialgHom _


@[inherit_doc]
notation:25 A " →ₐc[" R "] " B => BialgHom R A B


/-- `BialgHomClass F R A B` asserts `F` is a type of bundled bialgebra homomorphisms
from `A` to `B`. -/
class BialgHomClass (F : Type*) (R A B : outParam Type*)
    [CommSemiring R] [Semiring A] [Algebra R A] [Semiring B] [Algebra R B]
    [CoalgebraStruct R A] [CoalgebraStruct R B] [FunLike F A B]
    extends CoalgHomClass F R A B, MonoidHomClass F A B : Prop


instance (priority := 100) toAlgHomClass : AlgHomClass F R A B where
  map_mul := map_mul
  map_one := map_one
  map_add := map_add
  map_zero := map_zero
  commutes := fun c r => by
    /-
      R : Type u_1
      A : Type u_2
      B : Type u_3
      F : Type u_4
      inst✝⁸ : CommSemiring R
      inst✝⁷ : Semiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Semiring B
      inst✝⁴ : Algebra R B
      inst✝³ : CoalgebraStruct R A
      inst✝² : CoalgebraStruct R B
      inst✝¹ : FunLike F A B
      inst✝ : BialgHomClass F R A B
      c : F
      r : R
      ⊢ Eq (c ((algebraMap R A) r)) ((algebraMap R B) r)
    -/
    simp only [Algebra.algebraMap_eq_smul_one, map_smul, map_one]
    /-
      🎉 no goals
    -/


/-- Turn an element of a type `F` satisfying `BialgHomClass F R A B` into an actual
`BialgHom`. This is declared as the default coercion from `F` to `A →ₐc[R] B`. -/
@[coe]
def toBialgHom (f : F) : A →ₐc[R] B :=
  { CoalgHomClass.toCoalgHom f, AlgHomClass.toAlgHom f with
    toFun := f }


instance instCoeToBialgHom :
    CoeHead F (A →ₐc[R] B) :=
  ⟨BialgHomClass.toBialgHom⟩


@[simp]
theorem counitAlgHom_comp (f : F) :
    (counitAlgHom R B).comp (f : A →ₐ[R] B) = counitAlgHom R A :=
  AlgHom.toLinearMap_injective (CoalgHomClass.counit_comp f)


@[simp]
theorem map_comp_comulAlgHom (f : F) :
    (Algebra.TensorProduct.map f f).comp (comulAlgHom R A) = (comulAlgHom R B).comp f :=
  AlgHom.toLinearMap_injective (CoalgHomClass.map_comp_comul f)


instance funLike : FunLike (A →ₐc[R] B) A B where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      R : Type u_1
      A : Type u_2
      B : Type u_3
      C : Type u_4
      D : Type u_5
      inst✝¹² : CommSemiring R
      inst✝¹¹ : Semiring A
      inst✝¹⁰ : Algebra R A
      inst✝⁹ : Semiring B
      inst✝⁸ : Algebra R B
      inst✝⁷ : Semiring C
      inst✝⁶ : Algebra R C
      inst✝⁵ : Semiring D
      inst✝⁴ : Algebra R D
      inst✝³ : CoalgebraStruct R A
      inst✝² : CoalgebraStruct R B
      inst✝¹ : CoalgebraStruct R C
      inst✝ : CoalgebraStruct R D
      f g : BialgHom R A B
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    rcases f with ⟨_, _⟩
    /-
      case mk
      R : Type u_1
      A : Type u_2
      B : Type u_3
      C : Type u_4
      D : Type u_5
      inst✝¹² : CommSemiring R
      inst✝¹¹ : Semiring A
      inst✝¹⁰ : Algebra R A
      inst✝⁹ : Semiring B
      inst✝⁸ : Algebra R B
      inst✝⁷ : Semiring C
      inst✝⁶ : Algebra R C
      inst✝⁵ : Semiring D
      inst✝⁴ : Algebra R D
      inst✝³ : CoalgebraStruct R A
      inst✝² : CoalgebraStruct R B
      inst✝¹ : CoalgebraStruct R C
      inst✝ : CoalgebraStruct R D
      g : BialgHom R A B
      toCoalgHom✝ : CoalgHom R A B
      map_one'✝ : Eq (toCoalgHom✝.toFun 1) 1
      map_mul'✝ : ∀ (x y : A), Eq (toCoalgHom✝.toFun (HMul.hMul x y)) (HMul.hMul (to …
      h : Eq ((fun f => f.toFun) { toCoalgHom := toCoalgHom✝, map_one' := map_one'✝, …
      ⊢ Eq { toCoalgHom := toCoalgHom✝, map_one' := map_one'✝, map_mul' := map_mul'✝ …
    -/
    rcases g with ⟨_, _⟩
    /-
      case mk.mk
      R : Type u_1
      A : Type u_2
      B : Type u_3
      C : Type u_4
      D : Type u_5
      inst✝¹² : CommSemiring R
      inst✝¹¹ : Semiring A
      inst✝¹⁰ : Algebra R A
      inst✝⁹ : Semiring B
      inst✝⁸ : Algebra R B
      inst✝⁷ : Semiring C
      inst✝⁶ : Algebra R C
      inst✝⁵ : Semiring D
      inst✝⁴ : Algebra R D
      inst✝³ : CoalgebraStruct R A
      inst✝² : CoalgebraStruct R B
      inst✝¹ : CoalgebraStruct R C
      inst✝ : CoalgebraStruct R D
      toCoalgHom✝¹ : CoalgHom R A B
      map_one'✝¹ : Eq (toCoalgHom✝¹.toFun 1) 1
      map_mul'✝¹ : ∀ (x y : A), Eq (toCoalgHom✝¹.toFun (HMul.hMul x y)) (HMul.hMul ( …
      toCoalgHom✝ : CoalgHom R A B
      map_one'✝ : Eq (toCoalgHom✝.toFun 1) 1
      map_mul'✝ : ∀ (x y : A), Eq (toCoalgHom✝.toFun (HMul.hMul x y)) (HMul.hMul (to …
      h : Eq ((fun f => f.toFun) { toCoalgHom := toCoalgHom✝¹, map_one' := map_one'✝ …
      ⊢ Eq { toCoalgHom := toCoalgHom✝¹, map_one' := map_one'✝¹, map_mul' := map_mul …
    -/
    simp_all
    /-
      🎉 no goals
    -/


instance bialgHomClass : BialgHomClass (A →ₐc[R] B) R A B where
  map_add := fun f => f.map_add'
  map_smulₛₗ := fun f => f.map_smul'
  counit_comp := fun f => f.counit_comp
  map_comp_comul := fun f => f.map_comp_comul
  map_mul := fun f => f.map_mul'
  map_one := fun f => f.map_one'


/-- See Note [custom simps projection] -/
def Simps.apply {R α β : Type*} [CommSemiring R]
    [Semiring α] [Algebra R α] [Semiring β]
    [Algebra R β] [CoalgebraStruct R α] [CoalgebraStruct R β]
    (f : α →ₐc[R] β) : α → β := f


@[simp]
protected theorem coe_coe {F : Type*} [FunLike F A B] [BialgHomClass F R A B] (f : F) :
    ⇑(f : A →ₐc[R] B) = f :=
  rfl


@[simp]
theorem coe_mk {f : A →ₗc[R] B} (h h₁) : ((⟨f, h, h₁⟩ : A →ₐc[R] B) : A → B) = f :=
  rfl


@[norm_cast]
theorem coe_mks {f : A → B} (h₀ h₁ h₂ h₃ h₄ h₅) :
    ⇑(⟨⟨⟨⟨f, h₀⟩, h₁⟩, h₂, h₃⟩, h₄, h₅⟩ : A →ₐc[R] B) = f :=
  rfl


@[simp, norm_cast]
theorem coe_coalgHom_mk {f : A →ₗc[R] B} (h h₁) :
    ((⟨f, h, h₁⟩ : A →ₐc[R] B) : A →ₗc[R] B) = f := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    inst✝³ : Semiring B
    inst✝² : Algebra R B
    inst✝¹ : CoalgebraStruct R A
    inst✝ : CoalgebraStruct R B
    f : CoalgHom R A B
    h : Eq (f.toFun 1) 1
    h₁ : ∀ (x y : A), Eq (f.toFun (HMul.hMul x y)) (HMul.hMul (f.toFun x) (f.toFun …
    ⊢ Eq (↑{ toCoalgHom := f, map_one' := h, map_mul' := h₁ }) f
  -/
  rfl
  /-
    🎉 no goals
  -/


@[norm_cast]
theorem coe_toCoalgHom (f : A →ₐc[R] B) : ⇑(f : A →ₗc[R] B) = f :=
  rfl


@[simp, norm_cast]
theorem coe_toLinearMap (f : A →ₐc[R] B) : ⇑(f : A →ₗ[R] B) = f :=
  rfl


@[norm_cast]
theorem coe_toAlgHom (f : A →ₐc[R] B) : ⇑(f : A →ₐ[R] B) = f :=
  rfl


theorem toAlgHom_toLinearMap (f : A →ₐc[R] B) :
    ((f : A →ₐ[R] B) : A →ₗ[R] B) = f := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    inst✝³ : Semiring B
    inst✝² : Algebra R B
    inst✝¹ : CoalgebraStruct R A
    inst✝ : CoalgebraStruct R B
    f : BialgHom R A B
    ⊢ Eq ↑↑f ↑f
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem coe_fn_injective : @Function.Injective (A →ₐc[R] B) (A → B) (↑) :=
  DFunLike.coe_injective


theorem coe_fn_inj {φ₁ φ₂ : A →ₐc[R] B} : (φ₁ : A → B) = φ₂ ↔ φ₁ = φ₂ :=
  DFunLike.coe_fn_eq


theorem coe_coalgHom_injective : Function.Injective ((↑) : (A →ₐc[R] B) → A →ₗc[R] B) :=
  fun φ₁ φ₂ H => coe_fn_injective <|
    show ((φ₁ : A →ₗc[R] B) : A → B) = ((φ₂ : A →ₗc[R] B) : A → B) from congr_arg _ H


theorem coe_algHom_injective : Function.Injective ((↑) : (A →ₐc[R] B) → A →ₐ[R] B) :=
  fun φ₁ φ₂ H => coe_fn_injective <|
    show ((φ₁ : A →ₐ[R] B) : A → B) = ((φ₂ : A →ₐ[R] B) : A → B) from congr_arg _ H


theorem coe_linearMap_injective : Function.Injective ((↑) : (A →ₐc[R] B) → A →ₗ[R] B) :=
  CoalgHom.coe_linearMap_injective.comp coe_coalgHom_injective


protected theorem congr_fun {φ₁ φ₂ : A →ₐc[R] B} (H : φ₁ = φ₂) (x : A) : φ₁ x = φ₂ x :=
  DFunLike.congr_fun H x


protected theorem congr_arg (φ : A →ₐc[R] B) {x y : A} (h : x = y) : φ x = φ y :=
  DFunLike.congr_arg φ h


@[ext]
theorem ext {φ₁ φ₂ : A →ₐc[R] B} (H : ∀ x, φ₁ x = φ₂ x) : φ₁ = φ₂ :=
  DFunLike.ext _ _ H


@[ext high]
theorem ext_of_ring {f g : R →ₐc[R] A} (h : f 1 = g 1) : f = g :=
                              /-
                                R : Type u_1
                                A : Type u_2
                                inst✝³ : CommSemiring R
                                inst✝² : Semiring A
                                inst✝¹ : Algebra R A
                                inst✝ : CoalgebraStruct R A
                                f g : BialgHom R R A
                                h : Eq (f 1) (g 1)
                                ⊢ Eq ((fun x => ↑x) f) ((fun x => ↑x) g)
                              -/
  coe_linearMap_injective (by ext; assumption)
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem mk_coe {f : A →ₐc[R] B} (h₀ h₁ h₂ h₃ h₄ h₅) :
    (⟨⟨⟨⟨f, h₀⟩, h₁⟩, h₂, h₃⟩, h₄, h₅⟩ : A →ₐc[R] B) = f :=
  rfl


/-- Copy of a `BialgHom` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (f : A →ₐc[R] B) (f' : A → B) (h : f' = ⇑f) : A →ₐc[R] B :=
  { toCoalgHom := (f : A →ₗc[R] B).copy f' h
                   /-
                     R : Type u_1
                     A : Type u_2
                     B : Type u_3
                     C : Type u_4
                     D : Type u_5
                     inst✝¹² : CommSemiring R
                     inst✝¹¹ : Semiring A
                     inst✝¹⁰ : Algebra R A
                     inst✝⁹ : Semiring B
                     inst✝⁸ : Algebra R B
                     inst✝⁷ : Semiring C
                     inst✝⁶ : Algebra R C
                     inst✝⁵ : Semiring D
                     inst✝⁴ : Algebra R D
                     inst✝³ : CoalgebraStruct R A
                     inst✝² : CoalgebraStruct R B
                     inst✝¹ : CoalgebraStruct R C
                     inst✝ : CoalgebraStruct R D
                     φ f : BialgHom R A B
                     f' : A → B
                     h : Eq f' ⇑f
                     ⊢ Eq (((↑f).copy f' h).toFun 1) 1
                   -/
    map_one' := by simp_all
                   /-
                     🎉 no goals
                   -/
                   /-
                     R : Type u_1
                     A : Type u_2
                     B : Type u_3
                     C : Type u_4
                     D : Type u_5
                     inst✝¹² : CommSemiring R
                     inst✝¹¹ : Semiring A
                     inst✝¹⁰ : Algebra R A
                     inst✝⁹ : Semiring B
                     inst✝⁸ : Algebra R B
                     inst✝⁷ : Semiring C
                     inst✝⁶ : Algebra R C
                     inst✝⁵ : Semiring D
                     inst✝⁴ : Algebra R D
                     inst✝³ : CoalgebraStruct R A
                     inst✝² : CoalgebraStruct R B
                     inst✝¹ : CoalgebraStruct R C
                     inst✝ : CoalgebraStruct R D
                     φ f : BialgHom R A B
                     f' : A → B
                     h : Eq f' ⇑f
                     ⊢ ∀ (x y : A), Eq (((↑f).copy f' h).toFun (HMul.hMul x y)) (HMul.hMul (((↑f).c …
                   -/
    map_mul' := by intros; simp_all }
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem coe_copy (f : A →ₗc[R] B) (f' : A → B) (h : f' = ⇑f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : A →ₗc[R] B) (f' : A → B) (h : f' = ⇑f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- Identity map as a `BialgHom`. -/
@[simps!] protected def id : A →ₐc[R] A :=
  { CoalgHom.id R A, AlgHom.id R A with }


@[simp]
theorem coe_id : ⇑(BialgHom.id R A) = id :=
  rfl


@[simp]
theorem id_toCoalgHom : BialgHom.id R A = CoalgHom.id R A :=
  rfl


@[simp]
theorem id_toAlgHom : BialgHom.id R A = AlgHom.id R A :=
  rfl


/-- Composition of bialgebra homomorphisms. -/
@[simps!] def comp (φ₁ : B →ₐc[R] C) (φ₂ : A →ₐc[R] B) : A →ₐc[R] C :=
  { (φ₁ : B →ₗc[R] C).comp (φ₂ : A →ₗc[R] B), (φ₁ : B →ₐ[R] C).comp (φ₂ : A →ₐ[R] B) with }


@[simp]
theorem coe_comp (φ₁ : B →ₐc[R] C) (φ₂ : A →ₐc[R] B) : ⇑(φ₁.comp φ₂) = φ₁ ∘ φ₂ :=
  rfl


@[simp]
theorem comp_toCoalgHom (φ₁ : B →ₐc[R] C) (φ₂ : A →ₐc[R] B) :
    φ₁.comp φ₂ = (φ₁ : B →ₗc[R] C).comp (φ₂ : A →ₗc[R] B) :=
  rfl


@[simp]
theorem comp_toAlgHom (φ₁ : B →ₐc[R] C) (φ₂ : A →ₐc[R] B) :
    φ₁.comp φ₂ = (φ₁ : B →ₐ[R] C).comp (φ₂ : A →ₐ[R] B) :=
  rfl


@[simp]
theorem comp_id : φ.comp (BialgHom.id R A) = φ :=
  ext fun _x => rfl


@[simp]
theorem id_comp : (BialgHom.id R B).comp φ = φ :=
  ext fun _x => rfl


theorem comp_assoc (φ₁ : C →ₐc[R] D) (φ₂ : B →ₐc[R] C) (φ₃ : A →ₐc[R] B) :
    (φ₁.comp φ₂).comp φ₃ = φ₁.comp (φ₂.comp φ₃) :=
  ext fun _x => rfl


theorem map_smul_of_tower {R'} [SMul R' A] [SMul R' B] [LinearMap.CompatibleSMul A B R' R] (r : R')
    (x : A) : φ (r • x) = r • φ x :=
  φ.toLinearMap.map_smul_of_tower r x


@[simps (config := .lemmasOnly) toSemigroup_toMul_mul toOne_one]
instance End : Monoid (A →ₐc[R] A) where
  mul := comp
  mul_assoc _ _ _ := rfl
  one := BialgHom.id R A
  one_mul _ := ext fun _ => rfl
  mul_one _ := ext fun _ => rfl


@[simp]
theorem one_apply (x : A) : (1 : A →ₐc[R] A) x = x :=
  rfl


@[simp]
theorem mul_apply (φ ψ : A →ₐc[R] A) (x : A) : (φ * ψ) x = φ (ψ x) :=
  rfl


/-- The counit of a bialgebra as a `BialgHom`. -/
def counitBialgHom : A →ₐc[R] R :=
  { Coalgebra.counitCoalgHom R A, counitAlgHom R A with }


@[simp]
theorem counitBialgHom_apply (x : A) :
    counitBialgHom R A x = Coalgebra.counit x := rfl


@[simp]
theorem counitBialgHom_toCoalgHom :
    counitBialgHom R A = Coalgebra.counitCoalgHom R A := rfl


instance subsingleton_to_ring : Subsingleton (A →ₐc[R] R) :=
  ⟨fun _ _ => BialgHom.coe_coalgHom_injective (Subsingleton.elim _ _)⟩


@[ext high]
theorem ext_to_ring (f g : A →ₐc[R] R) : f = g := Subsingleton.elim _ _


