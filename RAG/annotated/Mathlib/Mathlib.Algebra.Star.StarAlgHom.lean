/-- A *non-unital ⋆-algebra homomorphism* is a non-unital algebra homomorphism between
non-unital `R`-algebras `A` and `B` equipped with a `star` operation, and this homomorphism is
also `star`-preserving. -/
structure NonUnitalStarAlgHom (R A B : Type*) [Monoid R] [NonUnitalNonAssocSemiring A]
  [DistribMulAction R A] [Star A] [NonUnitalNonAssocSemiring B] [DistribMulAction R B]
  [Star B] extends A →ₙₐ[R] B where
  /-- By definition, a non-unital ⋆-algebra homomorphism preserves the `star` operation. -/
  map_star' : ∀ a : A, toFun (star a) = star (toFun a)


@[inherit_doc NonUnitalStarAlgHom] infixr:25 " →⋆ₙₐ " => NonUnitalStarAlgHom _


@[inherit_doc] notation:25 A " →⋆ₙₐ[" R "] " B => NonUnitalStarAlgHom R A B


/-- `NonUnitalStarAlgHomClass F R A B` asserts `F` is a type of bundled non-unital ⋆-algebra
homomorphisms from `A` to `B`. -/
@[deprecated StarHomClass (since := "2024-09-08")]
class NonUnitalStarAlgHomClass (F : Type*) (R A B : outParam Type*)
  [Monoid R] [Star A] [Star B] [NonUnitalNonAssocSemiring A] [NonUnitalNonAssocSemiring B]
  [DistribMulAction R A] [DistribMulAction R B] [FunLike F A B] [NonUnitalAlgHomClass F R A B]
  extends StarHomClass F A B : Prop


/-- Turn an element of a type `F` satisfying `NonUnitalAlgHomClass F R A B` and `StarHomClass F A B`
into an actual `NonUnitalStarAlgHom`. This is declared as the default coercion from `F` to
`A →⋆ₙₐ[R] B`. -/
@[coe]
def toNonUnitalStarAlgHom [StarHomClass F A B] (f : F) : A →⋆ₙₐ[R] B :=
  { (f : A →ₙₐ[R] B) with
    map_star' := map_star f }


instance [StarHomClass F A B] : CoeTC F (A →⋆ₙₐ[R] B) :=
  ⟨toNonUnitalStarAlgHom⟩


instance [StarHomClass F A B] : NonUnitalStarRingHomClass F A B :=
  NonUnitalStarRingHomClass.mk


instance : FunLike (A →⋆ₙₐ[R] B) A B where
  coe f := f.toFun
                       /-
                         R : Type u_1
                         A : Type u_2
                         B : Type u_3
                         C : Type u_4
                         D : Type u_5
                         inst✝¹² : Monoid R
                         inst✝¹¹ : NonUnitalNonAssocSemiring A
                         inst✝¹⁰ : DistribMulAction R A
                         inst✝⁹ : Star A
                         inst✝⁸ : NonUnitalNonAssocSemiring B
                         inst✝⁷ : DistribMulAction R B
                         inst✝⁶ : Star B
                         inst✝⁵ : NonUnitalNonAssocSemiring C
                         inst✝⁴ : DistribMulAction R C
                         inst✝³ : Star C
                         inst✝² : NonUnitalNonAssocSemiring D
                         inst✝¹ : DistribMulAction R D
                         inst✝ : Star D
                         ⊢ Function.Injective fun f => f.toFun
                       -/
  coe_injective' := by rintro ⟨⟨⟨⟨f, _⟩, _⟩, _⟩, _⟩ ⟨⟨⟨⟨g, _⟩, _⟩, _⟩, _⟩ h; congr
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


instance : NonUnitalAlgHomClass (A →⋆ₙₐ[R] B) R A B where
  map_smulₛₗ f := f.map_smul'
  map_add f := f.map_add'
  map_zero f := f.map_zero'
  map_mul f := f.map_mul'


instance : StarHomClass (A →⋆ₙₐ[R] B) A B where
  map_star f := f.map_star'

-- Porting note: in mathlib3 we didn't need the `Simps.apply` hint.

/-- See Note [custom simps projection] -/
def Simps.apply (f : A →⋆ₙₐ[R] B) : A → B := f


@[simp]
protected theorem coe_coe {F : Type*} [FunLike F A B] [NonUnitalAlgHomClass F R A B]
    [StarHomClass F A B] (f : F) :
    ⇑(f : A →⋆ₙₐ[R] B) = f := rfl


@[simp]
theorem coe_toNonUnitalAlgHom {f : A →⋆ₙₐ[R] B} : (f.toNonUnitalAlgHom : A → B) = f :=
  rfl


@[ext]
theorem ext {f g : A →⋆ₙₐ[R] B} (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext _ _ h


/-- Copy of a `NonUnitalStarAlgHom` with a new `toFun` equal to the old one. Useful
to fix definitional equalities. -/
protected def copy (f : A →⋆ₙₐ[R] B) (f' : A → B) (h : f' = f) : A →⋆ₙₐ[R] B where
  toFun := f'
  map_smul' := h.symm ▸ map_smul f
  map_zero' := h.symm ▸ map_zero f
  map_add' := h.symm ▸ map_add f
  map_mul' := h.symm ▸ map_mul f
  map_star' := h.symm ▸ map_star f


@[simp]
theorem coe_copy (f : A →⋆ₙₐ[R] B) (f' : A → B) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : A →⋆ₙₐ[R] B) (f' : A → B) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


@[simp]
theorem coe_mk (f : A → B) (h₁ h₂ h₃ h₄ h₅) :
    ((⟨⟨⟨⟨f, h₁⟩, h₂, h₃⟩, h₄⟩, h₅⟩ : A →⋆ₙₐ[R] B) : A → B) = f :=
  rfl

-- this is probably the more useful lemma for Lean 4 and should likely replace `coe_mk` above

@[simp]
theorem coe_mk' (f : A →ₙₐ[R] B) (h) :
    ((⟨f, h⟩ : A →⋆ₙₐ[R] B) : A → B) = f :=
  rfl


@[simp]
theorem mk_coe (f : A →⋆ₙₐ[R] B) (h₁ h₂ h₃ h₄ h₅) :
    (⟨⟨⟨⟨f, h₁⟩, h₂, h₃⟩, h₄⟩, h₅⟩ : A →⋆ₙₐ[R] B) = f := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁶ : Monoid R
    inst✝⁵ : NonUnitalNonAssocSemiring A
    inst✝⁴ : DistribMulAction R A
    inst✝³ : Star A
    inst✝² : NonUnitalNonAssocSemiring B
    inst✝¹ : DistribMulAction R B
    inst✝ : Star B
    f : NonUnitalStarAlgHom R A B
    h₁ : ∀ (m : R) (x : A), Eq (f (HSMul.hSMul m x)) (HSMul.hSMul ((MonoidHom.id R …
    h₂ : Eq ({ toFun := ⇑f, map_smul' := h₁ }.toFun 0) 0
    h₃ : ∀ (x y : A), Eq ({ toFun := ⇑f, map_smul' := h₁ }.toFun (HAdd.hAdd x y))  …
    h₄ : ∀ (x y : A), Eq ({ toFun := ⇑f, map_smul' := h₁, map_zero' := h₂, map_add …
    h₅ : ∀ (a : A), Eq ({ toFun := ⇑f, map_smul' := h₁, map_zero' := h₂, map_add'  …
    ⊢ Eq { toFun := ⇑f, map_smul' := h₁, map_zero' := h₂, map_add' := h₃, map_mul' …
  -/
  ext
  /-
    case h
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁶ : Monoid R
    inst✝⁵ : NonUnitalNonAssocSemiring A
    inst✝⁴ : DistribMulAction R A
    inst✝³ : Star A
    inst✝² : NonUnitalNonAssocSemiring B
    inst✝¹ : DistribMulAction R B
    inst✝ : Star B
    f : NonUnitalStarAlgHom R A B
    h₁ : ∀ (m : R) (x : A), Eq (f (HSMul.hSMul m x)) (HSMul.hSMul ((MonoidHom.id R …
    h₂ : Eq ({ toFun := ⇑f, map_smul' := h₁ }.toFun 0) 0
    h₃ : ∀ (x y : A), Eq ({ toFun := ⇑f, map_smul' := h₁ }.toFun (HAdd.hAdd x y))  …
    h₄ : ∀ (x y : A), Eq ({ toFun := ⇑f, map_smul' := h₁, map_zero' := h₂, map_add …
    h₅ : ∀ (a : A), Eq ({ toFun := ⇑f, map_smul' := h₁, map_zero' := h₂, map_add'  …
    x✝ : A
    ⊢ Eq ({ toFun := ⇑f, map_smul' := h₁, map_zero' := h₂, map_add' := h₃, map_mul …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The identity as a non-unital ⋆-algebra homomorphism. -/
protected def id : A →⋆ₙₐ[R] A :=
  { (1 : A →ₙₐ[R] A) with map_star' := fun _ => rfl }


@[simp]
theorem coe_id : ⇑(NonUnitalStarAlgHom.id R A) = id :=
  rfl


/-- The composition of non-unital ⋆-algebra homomorphisms, as a non-unital ⋆-algebra
homomorphism. -/
def comp (f : B →⋆ₙₐ[R] C) (g : A →⋆ₙₐ[R] B) : A →⋆ₙₐ[R] C :=
  { f.toNonUnitalAlgHom.comp g.toNonUnitalAlgHom with
    map_star' := by
      simp only [map_star, NonUnitalAlgHom.toFun_eq_coe, eq_self_iff_true, NonUnitalAlgHom.coe_comp,
        coe_toNonUnitalAlgHom, Function.comp_apply, forall_const] }


@[simp]
theorem coe_comp (f : B →⋆ₙₐ[R] C) (g : A →⋆ₙₐ[R] B) : ⇑(comp f g) = f ∘ g :=
  rfl


@[simp]
theorem comp_apply (f : B →⋆ₙₐ[R] C) (g : A →⋆ₙₐ[R] B) (a : A) : comp f g a = f (g a) :=
  rfl


@[simp]
theorem comp_assoc (f : C →⋆ₙₐ[R] D) (g : B →⋆ₙₐ[R] C) (h : A →⋆ₙₐ[R] B) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp]
theorem id_comp (f : A →⋆ₙₐ[R] B) : (NonUnitalStarAlgHom.id _ _).comp f = f :=
  ext fun _ => rfl


@[simp]
theorem comp_id (f : A →⋆ₙₐ[R] B) : f.comp (NonUnitalStarAlgHom.id _ _) = f :=
  ext fun _ => rfl


instance : Monoid (A →⋆ₙₐ[R] A) where
  mul := comp
  mul_assoc := comp_assoc
  one := NonUnitalStarAlgHom.id R A
  one_mul := id_comp
  mul_one := comp_id


@[simp]
theorem coe_one : ((1 : A →⋆ₙₐ[R] A) : A → A) = id :=
  rfl


theorem one_apply (a : A) : (1 : A →⋆ₙₐ[R] A) a = a :=
  rfl


instance : Zero (A →⋆ₙₐ[R] B) :=
                                                                     /-
                                                                       R : Type u_1
                                                                       A : Type u_2
                                                                       B : Type u_3
                                                                       C : Type u_4
                                                                       D : Type u_5
                                                                       inst✝⁶ : Monoid R
                                                                       inst✝⁵ : NonUnitalNonAssocSemiring A
                                                                       inst✝⁴ : DistribMulAction R A
                                                                       inst✝³ : StarAddMonoid A
                                                                       inst✝² : NonUnitalNonAssocSemiring B
                                                                       inst✝¹ : DistribMulAction R B
                                                                       inst✝ : StarAddMonoid B
                                                                       ⊢ ∀ (a : A), Eq (__src✝.toFun (Star.star a)) (Star.star (__src✝.toFun a))
                                                                     -/
  ⟨{ (0 : NonUnitalAlgHom (MonoidHom.id R) A B) with map_star' := by simp }⟩
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


instance : Inhabited (A →⋆ₙₐ[R] B) :=
  ⟨0⟩


instance : MonoidWithZero (A →⋆ₙₐ[R] A) :=
  { inferInstanceAs (Monoid (A →⋆ₙₐ[R] A)),
    inferInstanceAs (Zero (A →⋆ₙₐ[R] A)) with
    zero_mul := fun _ => ext fun _ => rfl
    mul_zero := fun f => ext fun _ => map_zero f }


@[simp]
theorem coe_zero : ((0 : A →⋆ₙₐ[R] B) : A → B) = 0 :=
  rfl


theorem zero_apply (a : A) : (0 : A →⋆ₙₐ[R] B) a = 0 :=
  rfl


/-- If a monoid `R` acts on another monoid `S`, then a non-unital star algebra homomorphism
over `S` can be viewed as a non-unital star algebra homomorphism over `R`. -/
def restrictScalars (f : A →⋆ₙₐ[S] B) : A →⋆ₙₐ[R] B :=
  { (f : A →ₙₐ[S] B).restrictScalars R with
    map_star' := map_star f }


@[simp]
lemma restrictScalars_apply (f : A →⋆ₙₐ[S] B) (x : A) : f.restrictScalars R x = f x := rfl


lemma coe_restrictScalars (f : A →⋆ₙₐ[S] B) : (f.restrictScalars R : A →ₙ+* B) = f := rfl


lemma coe_restrictScalars' (f : A →⋆ₙₐ[S] B) : (f.restrictScalars R : A → B) = f := rfl


theorem restrictScalars_injective :
    Function.Injective (restrictScalars R : (A →⋆ₙₐ[S] B) → A →⋆ₙₐ[R] B) :=
  fun _ _ h ↦ ext (DFunLike.congr_fun h : _)


/-- A *⋆-algebra homomorphism* is an algebra homomorphism between `R`-algebras `A` and `B`
equipped with a `star` operation, and this homomorphism is also `star`-preserving. -/
structure StarAlgHom (R A B : Type*) [CommSemiring R] [Semiring A] [Algebra R A] [Star A]
  [Semiring B] [Algebra R B] [Star B] extends AlgHom R A B where
  /-- By definition, a ⋆-algebra homomorphism preserves the `star` operation. -/
  map_star' : ∀ x : A, toFun (star x) = star (toFun x)


@[inherit_doc StarAlgHom] infixr:25 " →⋆ₐ " => StarAlgHom _


@[inherit_doc] notation:25 A " →⋆ₐ[" R "] " B => StarAlgHom R A B


/-- `StarAlgHomClass F R A B` states that `F` is a type of ⋆-algebra homomorphisms.
You should also extend this typeclass when you extend `StarAlgHom`. -/
@[deprecated StarHomClass (since := "2024-09-08")]
class StarAlgHomClass (F : Type*) (R A B : outParam Type*)
    [CommSemiring R] [Semiring A] [Algebra R A] [Star A] [Semiring B] [Algebra R B] [Star B]
    [FunLike F A B] [AlgHomClass F R A B] extends StarHomClass F A B : Prop

/-- Turn an element of a type `F` satisfying `AlgHomClass F R A B` and `StarHomClass F A B` into an
actual `StarAlgHom`. This is declared as the default coercion from `F` to `A →⋆ₐ[R] B`. -/
@[coe]
def toStarAlgHom (f : F) : A →⋆ₐ[R] B :=
  { (f : A →ₐ[R] B) with
    map_star' := map_star f }


instance : CoeTC F (A →⋆ₐ[R] B) :=
  ⟨toStarAlgHom⟩


instance : FunLike (A →⋆ₐ[R] B) A B where
  coe f := f.toFun
                       /-
                         F : Type u_1
                         R : Type u_2
                         A : Type u_3
                         B : Type u_4
                         C : Type u_5
                         D : Type u_6
                         inst✝¹² : CommSemiring R
                         inst✝¹¹ : Semiring A
                         inst✝¹⁰ : Algebra R A
                         inst✝⁹ : Star A
                         inst✝⁸ : Semiring B
                         inst✝⁷ : Algebra R B
                         inst✝⁶ : Star B
                         inst✝⁵ : Semiring C
                         inst✝⁴ : Algebra R C
                         inst✝³ : Star C
                         inst✝² : Semiring D
                         inst✝¹ : Algebra R D
                         inst✝ : Star D
                         ⊢ Function.Injective fun f => (↑↑f.toRingHom).toFun
                       -/
  coe_injective' := by rintro ⟨⟨⟨⟨⟨f, _⟩, _⟩, _⟩, _⟩, _⟩ ⟨⟨⟨⟨⟨g, _⟩, _⟩, _⟩, _⟩, _⟩ h; congr
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


instance : AlgHomClass (A →⋆ₐ[R] B) R A B where
  map_mul f := f.map_mul'
  map_one f := f.map_one'
  map_add f := f.map_add'
  map_zero f := f.map_zero'
  commutes f := f.commutes'


instance : StarHomClass (A →⋆ₐ[R] B) A B where
  map_star f := f.map_star'


@[simp]
protected theorem coe_coe {F : Type*} [FunLike F A B] [AlgHomClass F R A B]
    [StarHomClass F A B] (f : F) :
    ⇑(f : A →⋆ₐ[R] B) = f :=
  rfl

-- Porting note: in mathlib3 we didn't need the `Simps.apply` hint.

/-- See Note [custom simps projection] -/
def Simps.apply (f : A →⋆ₐ[R] B) : A → B := f


@[simp]
theorem coe_toAlgHom {f : A →⋆ₐ[R] B} : (f.toAlgHom : A → B) = f :=
  rfl


@[ext]
theorem ext {f g : A →⋆ₐ[R] B} (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext _ _ h


/-- Copy of a `StarAlgHom` with a new `toFun` equal to the old one. Useful
to fix definitional equalities. -/
protected def copy (f : A →⋆ₐ[R] B) (f' : A → B) (h : f' = f) : A →⋆ₐ[R] B where
  toFun := f'
  map_one' := h.symm ▸ map_one f
  map_mul' := h.symm ▸ map_mul f
  map_zero' := h.symm ▸ map_zero f
  map_add' := h.symm ▸ map_add f
  commutes' := h.symm ▸ AlgHomClass.commutes f
  map_star' := h.symm ▸ map_star f


@[simp]
theorem coe_copy (f : A →⋆ₐ[R] B) (f' : A → B) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : A →⋆ₐ[R] B) (f' : A → B) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


@[simp]
theorem coe_mk (f : A → B) (h₁ h₂ h₃ h₄ h₅ h₆) :
    ((⟨⟨⟨⟨⟨f, h₁⟩, h₂⟩, h₃, h₄⟩, h₅⟩, h₆⟩ : A →⋆ₐ[R] B) : A → B) = f :=
  rfl

-- this is probably the more useful lemma for Lean 4 and should likely replace `coe_mk` above

@[simp]
theorem coe_mk' (f : A →ₐ[R] B) (h) :
    ((⟨f, h⟩ : A →⋆ₐ[R] B) : A → B) = f :=
  rfl


@[simp]
theorem mk_coe (f : A →⋆ₐ[R] B) (h₁ h₂ h₃ h₄ h₅ h₆) :
    (⟨⟨⟨⟨⟨f, h₁⟩, h₂⟩, h₃, h₄⟩, h₅⟩, h₆⟩ : A →⋆ₐ[R] B) = f := by
  /-
    R : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    inst✝³ : Star A
    inst✝² : Semiring B
    inst✝¹ : Algebra R B
    inst✝ : Star B
    f : StarAlgHom R A B
    h₁ : Eq (f 1) 1
    h₂ : ∀ (x y : A), Eq ({ toFun := ⇑f, map_one' := h₁ }.toFun (HMul.hMul x y)) ( …
    h₃ : Eq ((↑{ toFun := ⇑f, map_one' := h₁, map_mul' := h₂ }).toFun 0) 0
    h₄ : ∀ (x y : A), Eq ((↑{ toFun := ⇑f, map_one' := h₁, map_mul' := h₂ }).toFun …
    h₅ : ∀ (r : R), Eq ((↑↑{ toFun := ⇑f, map_one' := h₁, map_mul' := h₂, map_zero …
    h₆ : ∀ (x : A), Eq ((↑↑{ toFun := ⇑f, map_one' := h₁, map_mul' := h₂, map_zero …
    ⊢ Eq { toFun := ⇑f, map_one' := h₁, map_mul' := h₂, map_zero' := h₃, map_add'  …
  -/
  ext
  /-
    case h
    R : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    inst✝³ : Star A
    inst✝² : Semiring B
    inst✝¹ : Algebra R B
    inst✝ : Star B
    f : StarAlgHom R A B
    h₁ : Eq (f 1) 1
    h₂ : ∀ (x y : A), Eq ({ toFun := ⇑f, map_one' := h₁ }.toFun (HMul.hMul x y)) ( …
    h₃ : Eq ((↑{ toFun := ⇑f, map_one' := h₁, map_mul' := h₂ }).toFun 0) 0
    h₄ : ∀ (x y : A), Eq ((↑{ toFun := ⇑f, map_one' := h₁, map_mul' := h₂ }).toFun …
    h₅ : ∀ (r : R), Eq ((↑↑{ toFun := ⇑f, map_one' := h₁, map_mul' := h₂, map_zero …
    h₆ : ∀ (x : A), Eq ((↑↑{ toFun := ⇑f, map_one' := h₁, map_mul' := h₂, map_zero …
    x✝ : A
    ⊢ Eq ({ toFun := ⇑f, map_one' := h₁, map_mul' := h₂, map_zero' := h₃, map_add' …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The identity as a `StarAlgHom`. -/
protected def id : A →⋆ₐ[R] A :=
  { AlgHom.id _ _ with map_star' := fun _ => rfl }


@[simp]
theorem coe_id : ⇑(StarAlgHom.id R A) = id :=
  rfl


/-- `algebraMap R A` as a `StarAlgHom` when `A` is a star algebra over `R`. -/
@[simps]
def ofId (R A : Type*) [CommSemiring R] [StarRing R] [Semiring A] [StarMul A]
    [Algebra R A] [StarModule R A] : R →⋆ₐ[R] A :=
  { Algebra.ofId R A with
    toFun := algebraMap R A
                    /-
                      F : Type u_1
                      R✝ : Type u_2
                      A✝ : Type u_3
                      B : Type u_4
                      C : Type u_5
                      D : Type u_6
                      inst✝¹⁸ : CommSemiring R✝
                      inst✝¹⁷ : Semiring A✝
                      inst✝¹⁶ : Algebra R✝ A✝
                      inst✝¹⁵ : Star A✝
                      inst✝¹⁴ : Semiring B
                      inst✝¹³ : Algebra R✝ B
                      inst✝¹² : Star B
                      inst✝¹¹ : Semiring C
                      inst✝¹⁰ : Algebra R✝ C
                      inst✝⁹ : Star C
                      inst✝⁸ : Semiring D
                      inst✝⁷ : Algebra R✝ D
                      inst✝⁶ : Star D
                      R : Type u_7
                      A : Type u_8
                      inst✝⁵ : CommSemiring R
                      inst✝⁴ : StarRing R
                      inst✝³ : Semiring A
                      inst✝² : StarMul A
                      inst✝¹ : Algebra R A
                      inst✝ : StarModule R A
                      ⊢ ∀ (x : R), Eq ((↑↑{ toFun := ⇑(algebraMap R A), map_one' := ⋯, map_mul' := ⋯ …
                    -/
    map_star' := by simp [Algebra.algebraMap_eq_smul_one] }
                    /-
                      🎉 no goals
                    -/


instance : Inhabited (A →⋆ₐ[R] A) :=
  ⟨StarAlgHom.id R A⟩


/-- The composition of ⋆-algebra homomorphisms, as a ⋆-algebra homomorphism. -/
def comp (f : B →⋆ₐ[R] C) (g : A →⋆ₐ[R] B) : A →⋆ₐ[R] C :=
  { f.toAlgHom.comp g.toAlgHom with
    map_star' := by
      simp only [map_star, AlgHom.toFun_eq_coe, AlgHom.coe_comp, coe_toAlgHom,
        Function.comp_apply, eq_self_iff_true, forall_const] }


@[simp]
theorem coe_comp (f : B →⋆ₐ[R] C) (g : A →⋆ₐ[R] B) : ⇑(comp f g) = f ∘ g :=
  rfl


@[simp]
theorem comp_apply (f : B →⋆ₐ[R] C) (g : A →⋆ₐ[R] B) (a : A) : comp f g a = f (g a) :=
  rfl


@[simp]
theorem comp_assoc (f : C →⋆ₐ[R] D) (g : B →⋆ₐ[R] C) (h : A →⋆ₐ[R] B) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp]
theorem id_comp (f : A →⋆ₐ[R] B) : (StarAlgHom.id _ _).comp f = f :=
  ext fun _ => rfl


@[simp]
theorem comp_id (f : A →⋆ₐ[R] B) : f.comp (StarAlgHom.id _ _) = f :=
  ext fun _ => rfl


instance : Monoid (A →⋆ₐ[R] A) where
  mul := comp
  mul_assoc := comp_assoc
  one := StarAlgHom.id R A
  one_mul := id_comp
  mul_one := comp_id


/-- A unital morphism of ⋆-algebras is a `NonUnitalStarAlgHom`. -/
def toNonUnitalStarAlgHom (f : A →⋆ₐ[R] B) : A →⋆ₙₐ[R] B :=
  { f with map_smul' := map_smul f }


@[simp]
theorem coe_toNonUnitalStarAlgHom (f : A →⋆ₐ[R] B) : (f.toNonUnitalStarAlgHom : A → B) = f :=
  rfl


/-- The first projection of a product is a non-unital ⋆-algebra homomorphism. -/
@[simps!]
def fst : A × B →⋆ₙₐ[R] A :=
  { NonUnitalAlgHom.fst R A B with map_star' := fun _ => rfl }


/-- The second projection of a product is a non-unital ⋆-algebra homomorphism. -/
@[simps!]
def snd : A × B →⋆ₙₐ[R] B :=
  { NonUnitalAlgHom.snd R A B with map_star' := fun _ => rfl }


/-- The `Pi.prod` of two morphisms is a morphism. -/
@[simps!]
def prod (f : A →⋆ₙₐ[R] B) (g : A →⋆ₙₐ[R] C) : A →⋆ₙₐ[R] B × C :=
  { f.toNonUnitalAlgHom.prod g.toNonUnitalAlgHom with
                             /-
                               R : Type u_1
                               A : Type u_2
                               B : Type u_3
                               C : Type u_4
                               inst✝⁹ : Monoid R
                               inst✝⁸ : NonUnitalNonAssocSemiring A
                               inst✝⁷ : DistribMulAction R A
                               inst✝⁶ : Star A
                               inst✝⁵ : NonUnitalNonAssocSemiring B
                               inst✝⁴ : DistribMulAction R B
                               inst✝³ : Star B
                               inst✝² : NonUnitalNonAssocSemiring C
                               inst✝¹ : DistribMulAction R C
                               inst✝ : Star C
                               f : NonUnitalStarAlgHom R A B
                               g : NonUnitalStarAlgHom R A C
                               x : A
                               ⊢ Eq (__src✝.toFun (Star.star x)) (Star.star (__src✝.toFun x))
                             -/
    map_star' := fun x => by simp [map_star, Prod.star_def] }
                             /-
                               🎉 no goals
                             -/


theorem coe_prod (f : A →⋆ₙₐ[R] B) (g : A →⋆ₙₐ[R] C) : ⇑(f.prod g) = Pi.prod f g :=
  rfl


@[simp]
theorem fst_prod (f : A →⋆ₙₐ[R] B) (g : A →⋆ₙₐ[R] C) : (fst R B C).comp (prod f g) = f := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    C : Type u_4
    inst✝⁹ : Monoid R
    inst✝⁸ : NonUnitalNonAssocSemiring A
    inst✝⁷ : DistribMulAction R A
    inst✝⁶ : Star A
    inst✝⁵ : NonUnitalNonAssocSemiring B
    inst✝⁴ : DistribMulAction R B
    inst✝³ : Star B
    inst✝² : NonUnitalNonAssocSemiring C
    inst✝¹ : DistribMulAction R C
    inst✝ : Star C
    f : NonUnitalStarAlgHom R A B
    g : NonUnitalStarAlgHom R A C
    ⊢ Eq ((NonUnitalStarAlgHom.fst R B C).comp (f.prod g)) f
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


@[simp]
theorem snd_prod (f : A →⋆ₙₐ[R] B) (g : A →⋆ₙₐ[R] C) : (snd R B C).comp (prod f g) = g := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    C : Type u_4
    inst✝⁹ : Monoid R
    inst✝⁸ : NonUnitalNonAssocSemiring A
    inst✝⁷ : DistribMulAction R A
    inst✝⁶ : Star A
    inst✝⁵ : NonUnitalNonAssocSemiring B
    inst✝⁴ : DistribMulAction R B
    inst✝³ : Star B
    inst✝² : NonUnitalNonAssocSemiring C
    inst✝¹ : DistribMulAction R C
    inst✝ : Star C
    f : NonUnitalStarAlgHom R A B
    g : NonUnitalStarAlgHom R A C
    ⊢ Eq ((NonUnitalStarAlgHom.snd R B C).comp (f.prod g)) g
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


@[simp]
theorem prod_fst_snd : prod (fst R A B) (snd R A B) = 1 :=
  DFunLike.coe_injective Pi.prod_fst_snd


/-- Taking the product of two maps with the same domain is equivalent to taking the product of
their codomains. -/
@[simps]
def prodEquiv : (A →⋆ₙₐ[R] B) × (A →⋆ₙₐ[R] C) ≃ (A →⋆ₙₐ[R] B × C) where
  toFun f := f.1.prod f.2
  invFun f := ((fst _ _ _).comp f, (snd _ _ _).comp f)
                   /-
                     R : Type u_1
                     A : Type u_2
                     B : Type u_3
                     C : Type u_4
                     inst✝⁹ : Monoid R
                     inst✝⁸ : NonUnitalNonAssocSemiring A
                     inst✝⁷ : DistribMulAction R A
                     inst✝⁶ : Star A
                     inst✝⁵ : NonUnitalNonAssocSemiring B
                     inst✝⁴ : DistribMulAction R B
                     inst✝³ : Star B
                     inst✝² : NonUnitalNonAssocSemiring C
                     inst✝¹ : DistribMulAction R C
                     inst✝ : Star C
                     f : Prod (NonUnitalStarAlgHom R A B) (NonUnitalStarAlgHom R A C)
                     ⊢ Eq ((fun f => { fst := (NonUnitalStarAlgHom.fst R B C).comp f, snd := (NonUn …
                   -/
                           /-
                             🎉 no goals
                           -/
  left_inv f := by ext <;> rfl
                           /-
                             🎉 no goals
                           -/
                    /-
                      R : Type u_1
                      A : Type u_2
                      B : Type u_3
                      C : Type u_4
                      inst✝⁹ : Monoid R
                      inst✝⁸ : NonUnitalNonAssocSemiring A
                      inst✝⁷ : DistribMulAction R A
                      inst✝⁶ : Star A
                      inst✝⁵ : NonUnitalNonAssocSemiring B
                      inst✝⁴ : DistribMulAction R B
                      inst✝³ : Star B
                      inst✝² : NonUnitalNonAssocSemiring C
                      inst✝¹ : DistribMulAction R C
                      inst✝ : Star C
                      f : NonUnitalStarAlgHom R A (Prod B C)
                      ⊢ Eq ((fun f => f.1.prod f.2) ((fun f => { fst := (NonUnitalStarAlgHom.fst R B …
                    -/
                            /-
                              🎉 no goals
                            -/
  right_inv f := by ext <;> rfl
                            /-
                              🎉 no goals
                            -/


/-- The left injection into a product is a non-unital algebra homomorphism. -/
def inl : A →⋆ₙₐ[R] A × B :=
  prod 1 0


/-- The right injection into a product is a non-unital algebra homomorphism. -/
def inr : B →⋆ₙₐ[R] A × B :=
  prod 0 1


@[simp]
theorem coe_inl : (inl R A B : A → A × B) = fun x => (x, 0) :=
  rfl


theorem inl_apply (x : A) : inl R A B x = (x, 0) :=
  rfl


@[simp]
theorem coe_inr : (inr R A B : B → A × B) = Prod.mk 0 :=
  rfl


theorem inr_apply (x : B) : inr R A B x = (0, x) :=
  rfl


/-- The first projection of a product is a ⋆-algebra homomorphism. -/
@[simps!]
def fst : A × B →⋆ₐ[R] A :=
  { AlgHom.fst R A B with map_star' := fun _ => rfl }


/-- The second projection of a product is a ⋆-algebra homomorphism. -/
@[simps!]
def snd : A × B →⋆ₐ[R] B :=
  { AlgHom.snd R A B with map_star' := fun _ => rfl }


/-- The `Pi.prod` of two morphisms is a morphism. -/
@[simps!]
def prod (f : A →⋆ₐ[R] B) (g : A →⋆ₐ[R] C) : A →⋆ₐ[R] B × C :=
                                                             /-
                                                               R : Type u_1
                                                               A : Type u_2
                                                               B : Type u_3
                                                               C : Type u_4
                                                               inst✝⁹ : CommSemiring R
                                                               inst✝⁸ : Semiring A
                                                               inst✝⁷ : Algebra R A
                                                               inst✝⁶ : Star A
                                                               inst✝⁵ : Semiring B
                                                               inst✝⁴ : Algebra R B
                                                               inst✝³ : Star B
                                                               inst✝² : Semiring C
                                                               inst✝¹ : Algebra R C
                                                               inst✝ : Star C
                                                               f : StarAlgHom R A B
                                                               g : StarAlgHom R A C
                                                               x : A
                                                               ⊢ Eq ((↑↑__src✝.toRingHom).toFun (Star.star x)) (Star.star ((↑↑__src✝.toRingHo …
                                                             -/
  { f.toAlgHom.prod g.toAlgHom with map_star' := fun x => by simp [Prod.star_def, map_star] }
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem coe_prod (f : A →⋆ₐ[R] B) (g : A →⋆ₐ[R] C) : ⇑(f.prod g) = Pi.prod f g :=
  rfl


@[simp]
theorem fst_prod (f : A →⋆ₐ[R] B) (g : A →⋆ₐ[R] C) : (fst R B C).comp (prod f g) = f := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    C : Type u_4
    inst✝⁹ : CommSemiring R
    inst✝⁸ : Semiring A
    inst✝⁷ : Algebra R A
    inst✝⁶ : Star A
    inst✝⁵ : Semiring B
    inst✝⁴ : Algebra R B
    inst✝³ : Star B
    inst✝² : Semiring C
    inst✝¹ : Algebra R C
    inst✝ : Star C
    f : StarAlgHom R A B
    g : StarAlgHom R A C
    ⊢ Eq ((StarAlgHom.fst R B C).comp (f.prod g)) f
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


@[simp]
theorem snd_prod (f : A →⋆ₐ[R] B) (g : A →⋆ₐ[R] C) : (snd R B C).comp (prod f g) = g := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    C : Type u_4
    inst✝⁹ : CommSemiring R
    inst✝⁸ : Semiring A
    inst✝⁷ : Algebra R A
    inst✝⁶ : Star A
    inst✝⁵ : Semiring B
    inst✝⁴ : Algebra R B
    inst✝³ : Star B
    inst✝² : Semiring C
    inst✝¹ : Algebra R C
    inst✝ : Star C
    f : StarAlgHom R A B
    g : StarAlgHom R A C
    ⊢ Eq ((StarAlgHom.snd R B C).comp (f.prod g)) g
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


/-- Taking the product of two maps with the same domain is equivalent to taking the product of
their codomains. -/
@[simps]
def prodEquiv : (A →⋆ₐ[R] B) × (A →⋆ₐ[R] C) ≃ (A →⋆ₐ[R] B × C) where
  toFun f := f.1.prod f.2
  invFun f := ((fst _ _ _).comp f, (snd _ _ _).comp f)
                   /-
                     R : Type u_1
                     A : Type u_2
                     B : Type u_3
                     C : Type u_4
                     inst✝⁹ : CommSemiring R
                     inst✝⁸ : Semiring A
                     inst✝⁷ : Algebra R A
                     inst✝⁶ : Star A
                     inst✝⁵ : Semiring B
                     inst✝⁴ : Algebra R B
                     inst✝³ : Star B
                     inst✝² : Semiring C
                     inst✝¹ : Algebra R C
                     inst✝ : Star C
                     f : Prod (StarAlgHom R A B) (StarAlgHom R A C)
                     ⊢ Eq ((fun f => { fst := (StarAlgHom.fst R B C).comp f, snd := (StarAlgHom.snd …
                   -/
                           /-
                             🎉 no goals
                           -/
  left_inv f := by ext <;> rfl
                           /-
                             🎉 no goals
                           -/
                    /-
                      R : Type u_1
                      A : Type u_2
                      B : Type u_3
                      C : Type u_4
                      inst✝⁹ : CommSemiring R
                      inst✝⁸ : Semiring A
                      inst✝⁷ : Algebra R A
                      inst✝⁶ : Star A
                      inst✝⁵ : Semiring B
                      inst✝⁴ : Algebra R B
                      inst✝³ : Star B
                      inst✝² : Semiring C
                      inst✝¹ : Algebra R C
                      inst✝ : Star C
                      f : StarAlgHom R A (Prod B C)
                      ⊢ Eq ((fun f => f.1.prod f.2) ((fun f => { fst := (StarAlgHom.fst R B C).comp  …
                    -/
                            /-
                              🎉 no goals
                            -/
  right_inv f := by ext <;> rfl
                            /-
                              🎉 no goals
                            -/


/-- A *⋆-algebra* equivalence is an equivalence preserving addition, multiplication, scalar
multiplication and the star operation, which allows for considering both unital and non-unital
equivalences with a single structure. Currently, `AlgEquiv` requires unital algebras, which is
why this structure does not extend it. -/
structure StarAlgEquiv (R A B : Type*) [Add A] [Add B] [Mul A] [Mul B] [SMul R A] [SMul R B]
  [Star A] [Star B] extends A ≃+* B where
  /-- By definition, a ⋆-algebra equivalence preserves the `star` operation. -/
  map_star' : ∀ a : A, toFun (star a) = star (toFun a)
  /-- By definition, a ⋆-algebra equivalence commutes with the action of scalars. -/
  map_smul' : ∀ (r : R) (a : A), toFun (r • a) = r • toFun a


@[inherit_doc StarAlgEquiv] infixr:25 " ≃⋆ₐ " => StarAlgEquiv _


@[inherit_doc] notation:25 A " ≃⋆ₐ[" R "] " B => StarAlgEquiv R A B


/-- The class that directly extends `RingEquivClass` and `SMulHomClass`.

Mostly an implementation detail for `StarAlgEquivClass`.
-/
class NonUnitalAlgEquivClass (F : Type*) (R A B : outParam Type*)
  [Add A] [Mul A] [SMul R A] [Add B] [Mul B] [SMul R B] [EquivLike F A B]
  extends RingEquivClass F A B, MulActionSemiHomClass F (@id R) A B : Prop where


/-- `StarAlgEquivClass F R A B` asserts `F` is a type of bundled ⋆-algebra equivalences between
`A` and `B`.
You should also extend this typeclass when you extend `StarAlgEquiv`. -/
@[deprecated StarHomClass (since := "2024-09-08")]
class StarAlgEquivClass (F : Type*) (R A B : outParam Type*)
  [Add A] [Mul A] [SMul R A] [Star A] [Add B] [Mul B] [SMul R B]
  [Star B] [EquivLike F A B] [NonUnitalAlgEquivClass F R A B] : Prop where
  /-- By definition, a ⋆-algebra equivalence preserves the `star` operation. -/
  protected map_star : ∀ (f : F) (a : A), f (star a) = star (f a)


instance (priority := 100) {F R A B : Type*} [Monoid R] [NonUnitalNonAssocSemiring A]
    [DistribMulAction R A] [NonUnitalNonAssocSemiring B] [DistribMulAction R B] [EquivLike F A B]
    [NonUnitalAlgEquivClass F R A B] :
    NonUnitalAlgHomClass F R A B :=
  { }

-- See note [lower instance priority]

instance (priority := 100) instAlgHomClass (F R A B : Type*) [CommSemiring R] [Semiring A]
    [Algebra R A] [Semiring B] [Algebra R B] [EquivLike F A B] [NonUnitalAlgEquivClass F R A B] :
    AlgEquivClass F R A B :=
                              /-
                                F : Type u_1
                                R : Type u_2
                                A : Type u_3
                                B : Type u_4
                                inst✝⁶ : CommSemiring R
                                inst✝⁵ : Semiring A
                                inst✝⁴ : Algebra R A
                                inst✝³ : Semiring B
                                inst✝² : Algebra R B
                                inst✝¹ : EquivLike F A B
                                inst✝ : NonUnitalAlgEquivClass F R A B
                                f : F
                                r : R
                                ⊢ Eq (f ((algebraMap R A) r)) ((algebraMap R B) r)
                              -/
  { commutes := fun f r => by simp only [Algebra.algebraMap_eq_smul_one, map_smul, map_one] }
                              /-
                                🎉 no goals
                              -/


/-- Turn an element of a type `F` satisfying `AlgEquivClass F R A B` and `StarHomClass F A B` into
an actual `StarAlgEquiv`. This is declared as the default coercion from `F` to `A ≃⋆ₐ[R] B`. -/
@[coe]
def toStarAlgEquiv {F R A B : Type*} [Add A] [Mul A] [SMul R A] [Star A] [Add B] [Mul B] [SMul R B]
    [Star B] [EquivLike F A B] [NonUnitalAlgEquivClass F R A B] [StarHomClass F A B]
    (f : F) : A ≃⋆ₐ[R] B :=
  { (f : A ≃+* B) with
    map_star' := map_star f
    map_smul' := map_smul f}


/-- Any type satisfying `AlgEquivClass` and `StarHomClass` can be cast into `StarAlgEquiv` via
`StarAlgEquivClass.toStarAlgEquiv`. -/
instance instCoeHead {F R A B : Type*} [Add A] [Mul A] [SMul R A] [Star A] [Add B] [Mul B]
    [SMul R B] [Star B] [EquivLike F A B] [NonUnitalAlgEquivClass F R A B] [StarHomClass F A B] :
    CoeHead F (A ≃⋆ₐ[R] B) :=
  ⟨toStarAlgEquiv⟩


instance : EquivLike (A ≃⋆ₐ[R] B) A B where
  coe f := f.toFun
  inv f := f.invFun
  left_inv f := f.left_inv
  right_inv f := f.right_inv
  coe_injective' f g h₁ h₂ := by
    /-
      F : Type u_1
      R : Type u_2
      A : Type u_3
      B : Type u_4
      C : Type u_5
      inst✝¹¹ : Add A
      inst✝¹⁰ : Add B
      inst✝⁹ : Mul A
      inst✝⁸ : Mul B
      inst✝⁷ : SMul R A
      inst✝⁶ : SMul R B
      inst✝⁵ : Star A
      inst✝⁴ : Star B
      inst✝³ : Add C
      inst✝² : Mul C
      inst✝¹ : SMul R C
      inst✝ : Star C
      f g : StarAlgEquiv R A B
      h₁ : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      h₂ : Eq ((fun f => f.invFun) f) ((fun f => f.invFun) g)
      ⊢ Eq f g
    -/
    rcases f with ⟨⟨⟨_, _, _⟩, _⟩, _⟩
    /-
      case mk.mk.mk
      F : Type u_1
      R : Type u_2
      A : Type u_3
      B : Type u_4
      C : Type u_5
      inst✝¹¹ : Add A
      inst✝¹⁰ : Add B
      inst✝⁹ : Mul A
      inst✝⁸ : Mul B
      inst✝⁷ : SMul R A
      inst✝⁶ : SMul R B
      inst✝⁵ : Star A
      inst✝⁴ : Star B
      inst✝³ : Add C
      inst✝² : Mul C
      inst✝¹ : SMul R C
      inst✝ : Star C
      g : StarAlgEquiv R A B
      toFun✝ : A → B
      invFun✝ : B → A
      left_inv✝ : Function.LeftInverse invFun✝ toFun✝
      right_inv✝ : Function.RightInverse invFun✝ toFun✝
      map_mul'✝ : ∀ (x y : A), Eq ({ toFun := toFun✝, invFun := invFun✝, left_inv := …
      map_add'✝ : ∀ (x y : A), Eq ({ toFun := toFun✝, invFun := invFun✝, left_inv := …
      map_star'✝ : ∀ (a : A), Eq ({ toFun := toFun✝, invFun := invFun✝, left_inv :=  …
      map_smul'✝ : ∀ (r : R) (a : A), Eq ({ toFun := toFun✝, invFun := invFun✝, left …
      h₁ : Eq ((fun f => f.toFun) { toFun := toFun✝, invFun := invFun✝, left_inv :=  …
      h₂ : Eq ((fun f => f.invFun) { toFun := toFun✝, invFun := invFun✝, left_inv := …
      ⊢ Eq { toFun := toFun✝, invFun := invFun✝, left_inv := left_inv✝, right_inv := …
    -/
    rcases g with ⟨⟨⟨_, _, _⟩, _⟩, _⟩
    /-
      case mk.mk.mk.mk.mk.mk
      F : Type u_1
      R : Type u_2
      A : Type u_3
      B : Type u_4
      C : Type u_5
      inst✝¹¹ : Add A
      inst✝¹⁰ : Add B
      inst✝⁹ : Mul A
      inst✝⁸ : Mul B
      inst✝⁷ : SMul R A
      inst✝⁶ : SMul R B
      inst✝⁵ : Star A
      inst✝⁴ : Star B
      inst✝³ : Add C
      inst✝² : Mul C
      inst✝¹ : SMul R C
      inst✝ : Star C
      toFun✝¹ : A → B
      invFun✝¹ : B → A
      left_inv✝¹ : Function.LeftInverse invFun✝¹ toFun✝¹
      right_inv✝¹ : Function.RightInverse invFun✝¹ toFun✝¹
      map_mul'✝¹ : ∀ (x y : A), Eq ({ toFun := toFun✝¹, invFun := invFun✝¹, left_inv …
      map_add'✝¹ : ∀ (x y : A), Eq ({ toFun := toFun✝¹, invFun := invFun✝¹, left_inv …
      map_star'✝¹ : ∀ (a : A), Eq ({ toFun := toFun✝¹, invFun := invFun✝¹, left_inv  …
      map_smul'✝¹ : ∀ (r : R) (a : A), Eq ({ toFun := toFun✝¹, invFun := invFun✝¹, l …
      toFun✝ : A → B
      invFun✝ : B → A
      left_inv✝ : Function.LeftInverse invFun✝ toFun✝
      right_inv✝ : Function.RightInverse invFun✝ toFun✝
      map_mul'✝ : ∀ (x y : A), Eq ({ toFun := toFun✝, invFun := invFun✝, left_inv := …
      map_add'✝ : ∀ (x y : A), Eq ({ toFun := toFun✝, invFun := invFun✝, left_inv := …
      map_star'✝ : ∀ (a : A), Eq ({ toFun := toFun✝, invFun := invFun✝, left_inv :=  …
      map_smul'✝ : ∀ (r : R) (a : A), Eq ({ toFun := toFun✝, invFun := invFun✝, left …
      h₁ : Eq ((fun f => f.toFun) { toFun := toFun✝¹, invFun := invFun✝¹, left_inv : …
      h₂ : Eq ((fun f => f.invFun) { toFun := toFun✝¹, invFun := invFun✝¹, left_inv  …
      ⊢ Eq { toFun := toFun✝¹, invFun := invFun✝¹, left_inv := left_inv✝¹, right_inv …
    -/
    congr
    /-
      🎉 no goals
    -/


instance : NonUnitalAlgEquivClass (A ≃⋆ₐ[R] B) R A B where
  map_mul f := f.map_mul'
  map_add f := f.map_add'
  map_smulₛₗ := map_smul'


instance : StarHomClass (A ≃⋆ₐ[R] B) A B where
  map_star := map_star'


/-- Helper instance for cases where the inference via `EquivLike` is too hard. -/
instance : FunLike (A ≃⋆ₐ[R] B) A B where
  coe f := f.toFun
  coe_injective' := DFunLike.coe_injective


@[simp]
theorem toRingEquiv_eq_coe (e : A ≃⋆ₐ[R] B) : e.toRingEquiv = e :=
  rfl


@[ext]
theorem ext {f g : A ≃⋆ₐ[R] B} (h : ∀ a, f a = g a) : f = g :=
  DFunLike.ext f g h


/-- The identity map is a star algebra isomorphism. -/
@[refl]
def refl : A ≃⋆ₐ[R] A :=
  { RingEquiv.refl A with
    map_smul' := fun _ _ => rfl
    map_star' := fun _ => rfl }


instance : Inhabited (A ≃⋆ₐ[R] A) :=
  ⟨refl⟩


@[simp]
theorem coe_refl : ⇑(refl : A ≃⋆ₐ[R] A) = id :=
  rfl

-- Porting note: changed proof a bit by using `EquivLike` to avoid lots of coercions

/-- The inverse of a star algebra isomorphism is a star algebra isomorphism. -/
@[symm]
nonrec def symm (e : A ≃⋆ₐ[R] B) : B ≃⋆ₐ[R] A :=
  { e.symm with
    map_star' := fun b => by
      simpa only [apply_inv_apply, inv_apply_apply] using
        congr_arg (inv e) (map_star e (inv e b)).symm
    map_smul' := fun r b => by
      simpa only [apply_inv_apply, inv_apply_apply] using
        congr_arg (inv e) (map_smul e r (inv e b)).symm }

-- Porting note: in mathlib3 we didn't need the `Simps.apply` hint.

/-- See Note [custom simps projection] -/
def Simps.apply (e : A ≃⋆ₐ[R] B) : A → B := e


/-- See Note [custom simps projection] -/
def Simps.symm_apply (e : A ≃⋆ₐ[R] B) : B → A :=
  e.symm


@[simp]
theorem invFun_eq_symm {e : A ≃⋆ₐ[R] B} : EquivLike.inv e = e.symm :=
  rfl


@[simp]
theorem symm_symm (e : A ≃⋆ₐ[R] B) : e.symm.symm = e := rfl


theorem symm_bijective : Function.Bijective (symm : (A ≃⋆ₐ[R] B) → B ≃⋆ₐ[R] A) :=
  Function.bijective_iff_has_inverse.mpr ⟨_, symm_symm, symm_symm⟩


@[simp]
theorem coe_mk (e h₁ h₂) : ⇑(⟨e, h₁, h₂⟩ : A ≃⋆ₐ[R] B) = e := rfl


@[simp]
theorem mk_coe (e : A ≃⋆ₐ[R] B) (e' h₁ h₂ h₃ h₄ h₅ h₆) :
    (⟨⟨⟨e, e', h₁, h₂⟩, h₃, h₄⟩, h₅, h₆⟩ : A ≃⋆ₐ[R] B) = e := ext fun _ => rfl


/-- Auxiliary definition to avoid looping in `dsimp` with `StarAlgEquiv.symm_mk`. -/
protected def symm_mk.aux (f f') (h₁ h₂ h₃ h₄ h₅ h₆) :=
  (⟨⟨⟨f, f', h₁, h₂⟩, h₃, h₄⟩, h₅, h₆⟩ : A ≃⋆ₐ[R] B).symm


@[simp]
theorem symm_mk (f f') (h₁ h₂ h₃ h₄ h₅ h₆) :
    (⟨⟨⟨f, f', h₁, h₂⟩, h₃, h₄⟩, h₅, h₆⟩ : A ≃⋆ₐ[R] B).symm =
      { symm_mk.aux f f' h₁ h₂ h₃ h₄ h₅ h₆ with
        toFun := f'
        invFun := f } :=
  rfl


@[simp]
theorem refl_symm : (StarAlgEquiv.refl : A ≃⋆ₐ[R] A).symm = StarAlgEquiv.refl :=
  rfl

-- should be a `simp` lemma, but causes a linter timeout

theorem to_ringEquiv_symm (f : A ≃⋆ₐ[R] B) : (f : A ≃+* B).symm = f.symm :=
  rfl


@[simp]
theorem symm_to_ringEquiv (e : A ≃⋆ₐ[R] B) : (e.symm : B ≃+* A) = (e : A ≃+* B).symm :=
  rfl


/-- Transitivity of `StarAlgEquiv`. -/
@[trans]
def trans (e₁ : A ≃⋆ₐ[R] B) (e₂ : B ≃⋆ₐ[R] C) : A ≃⋆ₐ[R] C :=
  { e₁.toRingEquiv.trans
      e₂.toRingEquiv with
    map_smul' := fun r a =>
      show e₂.toFun (e₁.toFun (r • a)) = r • e₂.toFun (e₁.toFun a) by
        /-
          F : Type u_1
          R : Type u_2
          A : Type u_3
          B : Type u_4
          C : Type u_5
          inst✝¹¹ : Add A
          inst✝¹⁰ : Add B
          inst✝⁹ : Mul A
          inst✝⁸ : Mul B
          inst✝⁷ : SMul R A
          inst✝⁶ : SMul R B
          inst✝⁵ : Star A
          inst✝⁴ : Star B
          inst✝³ : Add C
          inst✝² : Mul C
          inst✝¹ : SMul R C
          inst✝ : Star C
          e₁ : StarAlgEquiv R A B
          e₂ : StarAlgEquiv R B C
          r : R
          a : A
          ⊢ Eq (e₂.toFun (e₁.toFun (HSMul.hSMul r a))) (HSMul.hSMul r (e₂.toFun (e₁.toFu …
        -/
        rw [e₁.map_smul', e₂.map_smul']
        /-
          🎉 no goals
        -/
        /-
          F : Type u_1
          R : Type u_2
          A : Type u_3
          B : Type u_4
          C : Type u_5
          inst✝¹¹ : Add A
          inst✝¹⁰ : Add B
          inst✝⁹ : Mul A
          inst✝⁸ : Mul B
          inst✝⁷ : SMul R A
          inst✝⁶ : SMul R B
          inst✝⁵ : Star A
          inst✝⁴ : Star B
          inst✝³ : Add C
          inst✝² : Mul C
          inst✝¹ : SMul R C
          inst✝ : Star C
          e₁ : StarAlgEquiv R A B
          e₂ : StarAlgEquiv R B C
          a : A
          ⊢ Eq (e₂.toFun (e₁.toFun (Star.star a))) (Star.star (e₂.toFun (e₁.toFun a)))
        -/
    map_star' := fun a =>
        /-
          🎉 no goals
        -/
      show e₂.toFun (e₁.toFun (star a)) = star (e₂.toFun (e₁.toFun a)) by
        rw [e₁.map_star', e₂.map_star'] }


@[simp]
theorem apply_symm_apply (e : A ≃⋆ₐ[R] B) : ∀ x, e (e.symm x) = x :=
  e.toRingEquiv.apply_symm_apply


@[simp]
theorem symm_apply_apply (e : A ≃⋆ₐ[R] B) : ∀ x, e.symm (e x) = x :=
  e.toRingEquiv.symm_apply_apply


@[simp]
theorem symm_trans_apply (e₁ : A ≃⋆ₐ[R] B) (e₂ : B ≃⋆ₐ[R] C) (x : C) :
    (e₁.trans e₂).symm x = e₁.symm (e₂.symm x) :=
  rfl


@[simp]
theorem coe_trans (e₁ : A ≃⋆ₐ[R] B) (e₂ : B ≃⋆ₐ[R] C) : ⇑(e₁.trans e₂) = e₂ ∘ e₁ :=
  rfl


@[simp]
theorem trans_apply (e₁ : A ≃⋆ₐ[R] B) (e₂ : B ≃⋆ₐ[R] C) (x : A) : (e₁.trans e₂) x = e₂ (e₁ x) :=
  rfl


theorem leftInverse_symm (e : A ≃⋆ₐ[R] B) : Function.LeftInverse e.symm e :=
  e.left_inv


theorem rightInverse_symm (e : A ≃⋆ₐ[R] B) : Function.RightInverse e.symm e :=
  e.right_inv


/-- If a (unital or non-unital) star algebra morphism has an inverse, it is an isomorphism of
star algebras. -/
@[simps]
def ofStarAlgHom (f : F) (g : G) (h₁ : ∀ x, g (f x) = x) (h₂ : ∀ x, f (g x) = x) : A ≃⋆ₐ[R] B where
  toFun := f
  invFun := g
  left_inv := h₁
  right_inv := h₂
  map_add' := map_add f
  map_mul' := map_mul f
  map_smul' := map_smul f
  map_star' := map_star f


/-- Promote a bijective star algebra homomorphism to a star algebra equivalence. -/
noncomputable def ofBijective (f : F) (hf : Function.Bijective f) : A ≃⋆ₐ[R] B :=
  {
    RingEquiv.ofBijective f
      (hf : Function.Bijective (f : A → B)) with
    toFun := f
    map_star' := map_star f
    map_smul' := map_smul f }


@[simp]
theorem coe_ofBijective {f : F} (hf : Function.Bijective f) :
    (StarAlgEquiv.ofBijective f hf : A → B) = f :=
  rfl


theorem ofBijective_apply {f : F} (hf : Function.Bijective f) (a : A) :
    (StarAlgEquiv.ofBijective f hf) a = f a :=
  rfl


