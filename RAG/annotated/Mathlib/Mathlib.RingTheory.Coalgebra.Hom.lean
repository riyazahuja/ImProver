/-- Given `R`-modules `A, B` with comultiplication maps `Δ_A, Δ_B` and counit maps
`ε_A, ε_B`, an `R`-coalgebra homomorphism `A →ₗc[R] B` is an `R`-linear map `f` such that
`ε_B ∘ f = ε_A` and `(f ⊗ f) ∘ Δ_A = Δ_B ∘ f`. -/
structure CoalgHom (R A B : Type*) [CommSemiring R]
    [AddCommMonoid A] [Module R A] [AddCommMonoid B] [Module R B]
    [CoalgebraStruct R A] [CoalgebraStruct R B] extends A →ₗ[R] B where
  counit_comp : counit ∘ₗ toLinearMap = counit
  map_comp_comul : TensorProduct.map toLinearMap toLinearMap ∘ₗ comul = comul ∘ₗ toLinearMap


@[inherit_doc CoalgHom]
infixr:25 " →ₗc " => CoalgHom _


@[inherit_doc]
notation:25 A " →ₗc[" R "] " B => CoalgHom R A B


/-- `CoalgHomClass F R A B` asserts `F` is a type of bundled coalgebra homomorphisms
from `A` to `B`. -/
class CoalgHomClass (F : Type*) (R A B : outParam Type*)
    [CommSemiring R] [AddCommMonoid A] [Module R A] [AddCommMonoid B] [Module R B]
    [CoalgebraStruct R A] [CoalgebraStruct R B] [FunLike F A B]
    extends SemilinearMapClass F (RingHom.id R) A B : Prop where
  counit_comp : ∀ f : F, counit ∘ₗ (f : A →ₗ[R] B) = counit
  map_comp_comul : ∀ f : F, TensorProduct.map (f : A →ₗ[R] B)
    (f : A →ₗ[R] B) ∘ₗ comul = comul ∘ₗ (f : A →ₗ[R] B)


/-- Turn an element of a type `F` satisfying `CoalgHomClass F R A B` into an actual
`CoalgHom`. This is declared as the default coercion from `F` to `A →ₗc[R] B`. -/
@[coe]
def toCoalgHom (f : F) : A →ₗc[R] B :=
  { (f : A →ₗ[R] B) with
    toFun := f
    counit_comp := CoalgHomClass.counit_comp f
    map_comp_comul := CoalgHomClass.map_comp_comul f }


instance instCoeToCoalgHom : CoeHead F (A →ₗc[R] B) :=
  ⟨CoalgHomClass.toCoalgHom⟩


@[simp]
theorem counit_comp_apply (f : F) (x : A) : counit (f x) = counit (R := R) x :=
  LinearMap.congr_fun (counit_comp f) _


@[simp]
theorem map_comp_comul_apply (f : F) (x : A) :
    TensorProduct.map f f (comul x) = comul (R := R) (f x) :=
  LinearMap.congr_fun (map_comp_comul f) _


instance funLike : FunLike (A →ₗc[R] B) A B where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      R : Type u_1
      A : Type u_2
      B : Type u_3
      C : Type u_4
      D : Type u_5
      inst✝¹² : CommSemiring R
      inst✝¹¹ : AddCommMonoid A
      inst✝¹⁰ : Module R A
      inst✝⁹ : AddCommMonoid B
      inst✝⁸ : Module R B
      inst✝⁷ : AddCommMonoid C
      inst✝⁶ : Module R C
      inst✝⁵ : AddCommMonoid D
      inst✝⁴ : Module R D
      inst✝³ : CoalgebraStruct R A
      inst✝² : CoalgebraStruct R B
      inst✝¹ : CoalgebraStruct R C
      inst✝ : CoalgebraStruct R D
      f g : CoalgHom R A B
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    rcases f with ⟨⟨⟨_, _⟩, _⟩, _, _⟩
    /-
      case mk.mk.mk
      R : Type u_1
      A : Type u_2
      B : Type u_3
      C : Type u_4
      D : Type u_5
      inst✝¹² : CommSemiring R
      inst✝¹¹ : AddCommMonoid A
      inst✝¹⁰ : Module R A
      inst✝⁹ : AddCommMonoid B
      inst✝⁸ : Module R B
      inst✝⁷ : AddCommMonoid C
      inst✝⁶ : Module R C
      inst✝⁵ : AddCommMonoid D
      inst✝⁴ : Module R D
      inst✝³ : CoalgebraStruct R A
      inst✝² : CoalgebraStruct R B
      inst✝¹ : CoalgebraStruct R C
      inst✝ : CoalgebraStruct R D
      g : CoalgHom R A B
      toFun✝ : A → B
      map_add'✝ : ∀ (x y : A), Eq (toFun✝ (HAdd.hAdd x y)) (HAdd.hAdd (toFun✝ x) (to …
      map_smul'✝ : ∀ (m : R) (x : A), Eq ({ toFun := toFun✝, map_add' := map_add'✝ } …
      counit_comp✝ : Eq (CoalgebraStruct.counit.comp { toFun := toFun✝, map_add' :=  …
      map_comp_comul✝ : Eq ((TensorProduct.map { toFun := toFun✝, map_add' := map_ad …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝, map_add' := map_add'✝, map_smul' …
      ⊢ Eq { toFun := toFun✝, map_add' := map_add'✝, map_smul' := map_smul'✝, counit …
    -/
    rcases g with ⟨⟨⟨_, _⟩, _⟩, _, _⟩
    /-
      case mk.mk.mk.mk.mk.mk
      R : Type u_1
      A : Type u_2
      B : Type u_3
      C : Type u_4
      D : Type u_5
      inst✝¹² : CommSemiring R
      inst✝¹¹ : AddCommMonoid A
      inst✝¹⁰ : Module R A
      inst✝⁹ : AddCommMonoid B
      inst✝⁸ : Module R B
      inst✝⁷ : AddCommMonoid C
      inst✝⁶ : Module R C
      inst✝⁵ : AddCommMonoid D
      inst✝⁴ : Module R D
      inst✝³ : CoalgebraStruct R A
      inst✝² : CoalgebraStruct R B
      inst✝¹ : CoalgebraStruct R C
      inst✝ : CoalgebraStruct R D
      toFun✝¹ : A → B
      map_add'✝¹ : ∀ (x y : A), Eq (toFun✝¹ (HAdd.hAdd x y)) (HAdd.hAdd (toFun✝¹ x)  …
      map_smul'✝¹ : ∀ (m : R) (x : A), Eq ({ toFun := toFun✝¹, map_add' := map_add'✝ …
      counit_comp✝¹ : Eq (CoalgebraStruct.counit.comp { toFun := toFun✝¹, map_add' : …
      map_comp_comul✝¹ : Eq ((TensorProduct.map { toFun := toFun✝¹, map_add' := map_ …
      toFun✝ : A → B
      map_add'✝ : ∀ (x y : A), Eq (toFun✝ (HAdd.hAdd x y)) (HAdd.hAdd (toFun✝ x) (to …
      map_smul'✝ : ∀ (m : R) (x : A), Eq ({ toFun := toFun✝, map_add' := map_add'✝ } …
      counit_comp✝ : Eq (CoalgebraStruct.counit.comp { toFun := toFun✝, map_add' :=  …
      map_comp_comul✝ : Eq ((TensorProduct.map { toFun := toFun✝, map_add' := map_ad …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝¹, map_add' := map_add'✝¹, map_smu …
      ⊢ Eq { toFun := toFun✝¹, map_add' := map_add'✝¹, map_smul' := map_smul'✝¹, cou …
    -/
    congr
    /-
      🎉 no goals
    -/


instance coalgHomClass : CoalgHomClass (A →ₗc[R] B) R A B where
  map_add := fun f => f.map_add'
  map_smulₛₗ := fun f => f.map_smul'
  counit_comp := fun f => f.counit_comp
  map_comp_comul := fun f => f.map_comp_comul


/-- See Note [custom simps projection] -/
def Simps.apply {R α β : Type*} [CommSemiring R]
    [AddCommMonoid α] [Module R α] [AddCommMonoid β]
    [Module R β] [CoalgebraStruct R α] [CoalgebraStruct R β]
    (f : α →ₗc[R] β) : α → β := f


@[simp]
protected theorem coe_coe {F : Type*} [FunLike F A B] [CoalgHomClass F R A B] (f : F) :
    ⇑(f : A →ₗc[R] B) = f :=
  rfl


@[simp]
theorem coe_mk {f : A →ₗ[R] B} (h h₁) : ((⟨f, h, h₁⟩ : A →ₗc[R] B) : A → B) = f :=
  rfl


@[norm_cast]
theorem coe_mks {f : A → B} (h₁ h₂ h₃ h₄) : ⇑(⟨⟨⟨f, h₁⟩, h₂⟩, h₃, h₄⟩ : A →ₗc[R] B) = f :=
  rfl


@[simp, norm_cast]
theorem coe_linearMap_mk {f : A →ₗ[R] B} (h h₁) : ((⟨f, h, h₁⟩ : A →ₗc[R] B) : A →ₗ[R] B) = f :=
  rfl


@[simp]
theorem toLinearMap_eq_coe (f : A →ₗc[R] B) : f.toLinearMap = f :=
  rfl


@[simp, norm_cast]
theorem coe_toLinearMap (f : A →ₗc[R] B) : ⇑(f : A →ₗ[R] B) = f :=
  rfl


@[norm_cast]
theorem coe_toAddMonoidHom (f : A →ₗc[R] B) : ⇑(f : A →+ B) = f :=
  rfl


theorem coe_fn_injective : @Function.Injective (A →ₗc[R] B) (A → B) (↑) :=
  DFunLike.coe_injective


theorem coe_fn_inj {φ₁ φ₂ : A →ₗc[R] B} : (φ₁ : A → B) = φ₂ ↔ φ₁ = φ₂ :=
  DFunLike.coe_fn_eq


theorem coe_linearMap_injective : Function.Injective ((↑) : (A →ₗc[R] B) → A →ₗ[R] B) :=
  fun φ₁ φ₂ H => coe_fn_injective <|
    show ((φ₁ : A →ₗ[R] B) : A → B) = ((φ₂ : A →ₗ[R] B) : A → B) from congr_arg _ H


theorem coe_addMonoidHom_injective : Function.Injective ((↑) : (A →ₗc[R] B) → A →+ B) :=
  LinearMap.toAddMonoidHom_injective.comp coe_linearMap_injective


protected theorem congr_fun {φ₁ φ₂ : A →ₗc[R] B} (H : φ₁ = φ₂) (x : A) : φ₁ x = φ₂ x :=
  DFunLike.congr_fun H x


protected theorem congr_arg (φ : A →ₗc[R] B) {x y : A} (h : x = y) : φ x = φ y :=
  DFunLike.congr_arg φ h


@[ext]
theorem ext {φ₁ φ₂ : A →ₗc[R] B} (H : ∀ x, φ₁ x = φ₂ x) : φ₁ = φ₂ :=
  DFunLike.ext _ _ H


@[ext high]
theorem ext_of_ring {f g : R →ₗc[R] A} (h : f 1 = g 1) : f = g :=
                              /-
                                R : Type u_1
                                A : Type u_2
                                inst✝³ : CommSemiring R
                                inst✝² : AddCommMonoid A
                                inst✝¹ : Module R A
                                inst✝ : CoalgebraStruct R A
                                f g : CoalgHom R R A
                                h : Eq (f 1) (g 1)
                                ⊢ Eq ((fun x => ↑x) f) ((fun x => ↑x) g)
                              -/
  coe_linearMap_injective (by ext; assumption)
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem mk_coe {f : A →ₗc[R] B} (h₁ h₂ h₃ h₄) : (⟨⟨⟨f, h₁⟩, h₂⟩, h₃, h₄⟩ : A →ₗc[R] B) = f :=
  ext fun _ => rfl


/-- Copy of a `CoalgHom` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (f : A →ₗc[R] B) (f' : A → B) (h : f' = ⇑f) : A →ₗc[R] B :=
  { toLinearMap := (f : A →ₗ[R] B).copy f' h
                      /-
                        R : Type u_1
                        A : Type u_2
                        B : Type u_3
                        C : Type u_4
                        D : Type u_5
                        inst✝¹² : CommSemiring R
                        inst✝¹¹ : AddCommMonoid A
                        inst✝¹⁰ : Module R A
                        inst✝⁹ : AddCommMonoid B
                        inst✝⁸ : Module R B
                        inst✝⁷ : AddCommMonoid C
                        inst✝⁶ : Module R C
                        inst✝⁵ : AddCommMonoid D
                        inst✝⁴ : Module R D
                        inst✝³ : CoalgebraStruct R A
                        inst✝² : CoalgebraStruct R B
                        inst✝¹ : CoalgebraStruct R C
                        inst✝ : CoalgebraStruct R D
                        f : CoalgHom R A B
                        f' : A → B
                        h : Eq f' ⇑f
                        ⊢ Eq (CoalgebraStruct.counit.comp ((↑f).copy f' h)) CoalgebraStruct.counit
                      -/
    counit_comp := by ext; simp_all
                           /-
                             🎉 no goals
                           -/
    map_comp_comul := by simp only [(f : A →ₗ[R] B).copy_eq f' h,
      CoalgHomClass.map_comp_comul] }


@[simp]
theorem coe_copy (f : A →ₗc[R] B) (f' : A → B) (h : f' = ⇑f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : A →ₗc[R] B) (f' : A → B) (h : f' = ⇑f) : f.copy f' h = f :=
  DFunLike.ext' h


/-- Identity map as a `CoalgHom`. -/
@[simps!] protected def id : A →ₗc[R] A :=
  { LinearMap.id with
                      /-
                        R : Type u_1
                        A : Type u_2
                        B : Type u_3
                        C : Type u_4
                        D : Type u_5
                        inst✝¹² : CommSemiring R
                        inst✝¹¹ : AddCommMonoid A
                        inst✝¹⁰ : Module R A
                        inst✝⁹ : AddCommMonoid B
                        inst✝⁸ : Module R B
                        inst✝⁷ : AddCommMonoid C
                        inst✝⁶ : Module R C
                        inst✝⁵ : AddCommMonoid D
                        inst✝⁴ : Module R D
                        inst✝³ : CoalgebraStruct R A
                        inst✝² : CoalgebraStruct R B
                        inst✝¹ : CoalgebraStruct R C
                        inst✝ : CoalgebraStruct R D
                        ⊢ Eq (CoalgebraStruct.counit.comp __src✝) CoalgebraStruct.counit
                      -/
    counit_comp := by ext; rfl
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
                           inst✝¹¹ : AddCommMonoid A
                           inst✝¹⁰ : Module R A
                           inst✝⁹ : AddCommMonoid B
                           inst✝⁸ : Module R B
                           inst✝⁷ : AddCommMonoid C
                           inst✝⁶ : Module R C
                           inst✝⁵ : AddCommMonoid D
                           inst✝⁴ : Module R D
                           inst✝³ : CoalgebraStruct R A
                           inst✝² : CoalgebraStruct R B
                           inst✝¹ : CoalgebraStruct R C
                           inst✝ : CoalgebraStruct R D
                           ⊢ Eq ((TensorProduct.map __src✝ __src✝).comp CoalgebraStruct.comul) (Coalgebra …
                         -/
    map_comp_comul := by simp only [map_id, LinearMap.id_comp, LinearMap.comp_id] }
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem coe_id : ⇑(CoalgHom.id R A) = id :=
  rfl


@[simp]
theorem id_toLinearMap : (CoalgHom.id R A : A →ₗ[R] A) = LinearMap.id := rfl


/-- Composition of coalgebra homomorphisms. -/
@[simps!] def comp (φ₁ : B →ₗc[R] C) (φ₂ : A →ₗc[R] B) : A →ₗc[R] C :=
  { (φ₁ : B →ₗ[R] C) ∘ₗ (φ₂ : A →ₗ[R] B) with
                      /-
                        R : Type u_1
                        A : Type u_2
                        B : Type u_3
                        C : Type u_4
                        D : Type u_5
                        inst✝¹² : CommSemiring R
                        inst✝¹¹ : AddCommMonoid A
                        inst✝¹⁰ : Module R A
                        inst✝⁹ : AddCommMonoid B
                        inst✝⁸ : Module R B
                        inst✝⁷ : AddCommMonoid C
                        inst✝⁶ : Module R C
                        inst✝⁵ : AddCommMonoid D
                        inst✝⁴ : Module R D
                        inst✝³ : CoalgebraStruct R A
                        inst✝² : CoalgebraStruct R B
                        inst✝¹ : CoalgebraStruct R C
                        inst✝ : CoalgebraStruct R D
                        φ₁ : CoalgHom R B C
                        φ₂ : CoalgHom R A B
                        ⊢ Eq (CoalgebraStruct.counit.comp __src✝) CoalgebraStruct.counit
                      -/
    counit_comp := by ext; simp
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
                           inst✝¹¹ : AddCommMonoid A
                           inst✝¹⁰ : Module R A
                           inst✝⁹ : AddCommMonoid B
                           inst✝⁸ : Module R B
                           inst✝⁷ : AddCommMonoid C
                           inst✝⁶ : Module R C
                           inst✝⁵ : AddCommMonoid D
                           inst✝⁴ : Module R D
                           inst✝³ : CoalgebraStruct R A
                           inst✝² : CoalgebraStruct R B
                           inst✝¹ : CoalgebraStruct R C
                           inst✝ : CoalgebraStruct R D
                           φ₁ : CoalgHom R B C
                           φ₂ : CoalgHom R A B
                           ⊢ Eq ((TensorProduct.map __src✝ __src✝).comp CoalgebraStruct.comul) (Coalgebra …
                         -/
    map_comp_comul := by ext; simp [map_comp] }
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem coe_comp (φ₁ : B →ₗc[R] C) (φ₂ : A →ₗc[R] B) : ⇑(φ₁.comp φ₂) = φ₁ ∘ φ₂ := rfl


@[simp]
theorem comp_toLinearMap (φ₁ : B →ₗc[R] C) (φ₂ : A →ₗc[R] B) :
    φ₁.comp φ₂ = (φ₁ : B →ₗ[R] C) ∘ₗ (φ₂ : A →ₗ[R] B) := rfl


@[simp]
theorem comp_id : φ.comp (CoalgHom.id R A) = φ :=
  ext fun _x => rfl


@[simp]
theorem id_comp : (CoalgHom.id R B).comp φ = φ :=
  ext fun _x => rfl


theorem comp_assoc (φ₁ : C →ₗc[R] D) (φ₂ : B →ₗc[R] C) (φ₃ : A →ₗc[R] B) :
    (φ₁.comp φ₂).comp φ₃ = φ₁.comp (φ₂.comp φ₃) :=
  ext fun _x => rfl


theorem map_smul_of_tower {R'} [SMul R' A] [SMul R' B] [LinearMap.CompatibleSMul A B R' R] (r : R')
    (x : A) : φ (r • x) = r • φ x :=
  φ.toLinearMap.map_smul_of_tower r x


@[simps (config := .lemmasOnly) toSemigroup_toMul_mul toOne_one]
instance End : Monoid (A →ₗc[R] A) where
  mul := comp
  mul_assoc _ _ _ := rfl
  one := CoalgHom.id R A
  one_mul _ := ext fun _ => rfl
  mul_one _ := ext fun _ => rfl


@[simp]
theorem one_apply (x : A) : (1 : A →ₗc[R] A) x = x :=
  rfl


@[simp]
theorem mul_apply (φ ψ : A →ₗc[R] A) (x : A) : (φ * ψ) x = φ (ψ x) :=
  rfl


/-- The counit of a coalgebra as a `CoalgHom`. -/
def counitCoalgHom : A →ₗc[R] R :=
  { counit with
                      /-
                        R : Type u
                        A : Type v
                        B : Type w
                        inst✝⁶ : CommSemiring R
                        inst✝⁵ : AddCommMonoid A
                        inst✝⁴ : AddCommMonoid B
                        inst✝³ : Module R A
                        inst✝² : Module R B
                        inst✝¹ : Coalgebra R A
                        inst✝ : Coalgebra R B
                        ⊢ Eq (CoalgebraStruct.counit.comp __src✝) CoalgebraStruct.counit
                      -/
    counit_comp := by ext; simp
                           /-
                             🎉 no goals
                           -/
    map_comp_comul := by
      /-
        R : Type u
        A : Type v
        B : Type w
        inst✝⁶ : CommSemiring R
        inst✝⁵ : AddCommMonoid A
        inst✝⁴ : AddCommMonoid B
        inst✝³ : Module R A
        inst✝² : Module R B
        inst✝¹ : Coalgebra R A
        inst✝ : Coalgebra R B
        ⊢ Eq ((TensorProduct.map __src✝ __src✝).comp CoalgebraStruct.comul) (Coalgebra …
      -/
      ext
      simp only [LinearMap.coe_comp, Function.comp_apply, CommSemiring.comul_apply,
        ← LinearMap.lTensor_comp_rTensor, rTensor_counit_comul, LinearMap.lTensor_tmul] }


@[simp]
theorem counitCoalgHom_apply (x : A) :
    counitCoalgHom R A x = counit x := rfl


@[simp]
theorem counitCoalgHom_toLinearMap :
    counitCoalgHom R A = counit (R := R) (A := A) := rfl


instance subsingleton_to_ring : Subsingleton (A →ₗc[R] R) :=
  ⟨fun f g => CoalgHom.ext fun x => by
    /-
      R : Type u
      A : Type v
      B : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid A
      inst✝⁴ : AddCommMonoid B
      inst✝³ : Module R A
      inst✝² : Module R B
      inst✝¹ : Coalgebra R A
      inst✝ : Coalgebra R B
      f g : CoalgHom R A R
      x : A
      ⊢ Eq (f x) (g x)
    -/
    have hf := CoalgHomClass.counit_comp_apply f x
    /-
      R : Type u
      A : Type v
      B : Type w
      inst✝⁶ : CommSemiring R
      inst✝⁵ : AddCommMonoid A
      inst✝⁴ : AddCommMonoid B
      inst✝³ : Module R A
      inst✝² : Module R B
      inst✝¹ : Coalgebra R A
      inst✝ : Coalgebra R B
      f g : CoalgHom R A R
      x : A
      hf : Eq (CoalgebraStruct.counit (f x)) (CoalgebraStruct.counit x)
      ⊢ Eq (f x) (g x)
    -/
    have hg := CoalgHomClass.counit_comp_apply g x
    simp_all only [CoalgHom.toLinearMap_eq_coe, LinearMap.coe_comp, CoalgHom.coe_toLinearMap,
      Function.comp_apply, CommSemiring.counit_apply]⟩


@[ext high]
theorem ext_to_ring (f g : A →ₗc[R] R) : f = g := Subsingleton.elim _ _


/--
If `φ : A → B` is a coalgebra map and `a = ∑ xᵢ ⊗ yᵢ`, then `φ a = ∑ φ xᵢ ⊗ φ yᵢ`
-/
@[simps]
def Repr.induced {a : A} (repr : Repr R a)
    {F : Type*} [FunLike F A B] [CoalgHomClass F R A B]
    (φ : F) : Repr R (φ a) where
  index := repr.index
  left := φ ∘ repr.left
  right := φ ∘ repr.right
  eq := (congr($((CoalgHomClass.map_comp_comul φ).symm) a).trans <|
         /-
           R : Type u
           A : Type v
           B : Type w
           inst✝⁸ : CommSemiring R
           inst✝⁷ : AddCommMonoid A
           inst✝⁶ : AddCommMonoid B
           inst✝⁵ : Module R A
           inst✝⁴ : Module R B
           inst✝³ : Coalgebra R A
           inst✝² : Coalgebra R B
           a : A
           repr : Coalgebra.Repr R a
           F : Type u_1
           inst✝¹ : FunLike F A B
           inst✝ : CoalgHomClass F R A B
           φ : F
           ⊢ Eq (((TensorProduct.map ↑φ ↑φ).comp CoalgebraStruct.comul) a) (repr.index.su …
         -/
      by rw [LinearMap.comp_apply, ← repr.eq, map_sum]; rfl).symm
                                                        /-
                                                          🎉 no goals
                                                        -/


