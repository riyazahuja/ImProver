instance : CoeSort (Under R) (Type u) where
  coe A := A.right


instance (A : Under R) : Algebra R A := RingHom.toAlgebra A.hom.hom


/-- Turn a morphism in `Under R` into an algebra homomorphism. -/
def toAlgHom {A B : Under R} (f : A ⟶ B) : A →ₐ[R] B where
  __ := f.right.hom
  commutes' a := by
    /-
      R S : CommRingCat
      A B : CategoryTheory.Under R
      f : Quiver.Hom A B
      a : ↑R
      ⊢ Eq ((↑↑__spread✝⁻⁰).toFun ((algebraMap ↑R ↑A.right) a)) ((algebraMap ↑R ↑B.r …
    -/
    have : (A.hom ≫ f.right) a = B.hom a := by simp
    /-
      R S : CommRingCat
      A B : CategoryTheory.Under R
      f : Quiver.Hom A B
      a : ↑R
      this : Eq ((CategoryTheory.CategoryStruct.comp A.hom f.right).hom a) (B.hom.ho …
      ⊢ Eq ((↑↑__spread✝⁻⁰).toFun ((algebraMap ↑R ↑A.right) a)) ((algebraMap ↑R ↑B.r …
    -/
    simpa only [Functor.const_obj_obj, Functor.id_obj, CommRingCat.comp_apply] using this
    /-
      🎉 no goals
    -/


@[simp]
lemma toAlgHom_id (A : Under R) : toAlgHom (𝟙 A) = AlgHom.id R A := rfl


@[simp]
lemma toAlgHom_comp {A B C : Under R} (f : A ⟶ B) (g : B ⟶ C) :
    toAlgHom (f ≫ g) = (toAlgHom g).comp (toAlgHom f) := rfl


@[simp]
lemma toAlgHom_apply {A B : Under R} (f : A ⟶ B) (a : A) :
    toAlgHom f a = f.right a :=
  rfl


variable (R) in
/-- Make an object of `Under R` from an `R`-algebra. -/
@[simps! hom, simps! (config := .lemmasOnly) right]
def mkUnder (A : Type u) [CommRing A] [Algebra R A] : Under R :=
  Under.mk (CommRingCat.ofHom <| algebraMap R A)


@[ext]
lemma mkUnder_ext {A : Type u} [CommRing A] [Algebra R A] {B : Under R}
    {f g : mkUnder R A ⟶ B} (h : ∀ a : A, f.right a = g.right a) :
    f = g := by
  /-
    R : CommRingCat
    A : Type u
    inst✝¹ : CommRing A
    inst✝ : Algebra (↑R) A
    B : CategoryTheory.Under R
    f g : Quiver.Hom (R.mkUnder A) B
    h : ∀ (a : A), Eq (f.right.hom a) (g.right.hom a)
    ⊢ Eq f g
  -/
  ext x
  /-
    case h.hf.a
    R : CommRingCat
    A : Type u
    inst✝¹ : CommRing A
    inst✝ : Algebra (↑R) A
    B : CategoryTheory.Under R
    f g : Quiver.Hom (R.mkUnder A) B
    h : ∀ (a : A), Eq (f.right.hom a) (g.right.hom a)
    x : ↑(R.mkUnder A).right
    ⊢ Eq (f.right.hom x) (g.right.hom x)
  -/
  exact h x
  /-
    🎉 no goals
  -/


/-- Make a morphism in `Under R` from an algebra map. -/
def toUnder {A B : Type u} [CommRing A] [CommRing B] [Algebra R A] [Algebra R B]
    (f : A →ₐ[R] B) : CommRingCat.mkUnder R A ⟶ CommRingCat.mkUnder R B :=
  Under.homMk (CommRingCat.ofHom f.toRingHom) <| by
    /-
      R S : CommRingCat
      A B : Type u
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra (↑R) A
      inst✝ : Algebra (↑R) B
      f : AlgHom (↑R) A B
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (R.mkUnder A).hom (CommRingCat.ofHom  …
    -/
    ext a
    /-
      case hf.a
      R S : CommRingCat
      A B : Type u
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra (↑R) A
      inst✝ : Algebra (↑R) B
      f : AlgHom (↑R) A B
      a : ↑((CategoryTheory.Functor.fromPUnit R).obj (R.mkUnder A).left)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (R.mkUnder A).hom (CommRingCat.ofHom …
    -/
    exact f.commutes' a
    /-
      🎉 no goals
    -/


@[simp]
lemma toUnder_right {A B : Type u} [CommRing A] [CommRing B] [Algebra R A]
    [Algebra R B] (f : A →ₐ[R] B) (a : A) :
    f.toUnder.right a = f a :=
  rfl


@[simp]
lemma toUnder_comp {A B C : Type u} [CommRing A] [CommRing B] [CommRing C]
    [Algebra R A] [Algebra R B] [Algebra R C] (f : A →ₐ[R] B) (g : B →ₐ[R] C) :
    (g.comp f).toUnder = f.toUnder ≫ g.toUnder :=
  rfl


/-- Make an isomorphism in `Under R` from an algebra isomorphism. -/
def toUnder {A B : Type u} [CommRing A] [CommRing B] [Algebra R A] [Algebra R B]
    (f : A ≃ₐ[R] B) :
    CommRingCat.mkUnder R A ≅ CommRingCat.mkUnder R B where
  hom := f.toAlgHom.toUnder
  inv := f.symm.toAlgHom.toUnder
  hom_inv_id := by
    /-
      R S : CommRingCat
      A B : Type u
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra (↑R) A
      inst✝ : Algebra (↑R) B
      f : AlgEquiv (↑R) A B
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (↑f).toUnder (↑f.symm).toUnder) (Cate …
    -/
    ext (a : (CommRingCat.mkUnder R A).right)
    /-
      case h
      R S : CommRingCat
      A B : Type u
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra (↑R) A
      inst✝ : Algebra (↑R) B
      f : AlgEquiv (↑R) A B
      a : ↑(R.mkUnder A).right
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (↑f).toUnder (↑f.symm).toUnder).righ …
    -/
    simp
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      R S : CommRingCat
      A B : Type u
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra (↑R) A
      inst✝ : Algebra (↑R) B
      f : AlgEquiv (↑R) A B
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (↑f.symm).toUnder (↑f).toUnder) (Cate …
    -/
    ext a
    /-
      case h
      R S : CommRingCat
      A B : Type u
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra (↑R) A
      inst✝ : Algebra (↑R) B
      f : AlgEquiv (↑R) A B
      a : B
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (↑f.symm).toUnder (↑f).toUnder).righ …
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
lemma toUnder_hom_right_apply {A B : Type u} [CommRing A] [CommRing B] [Algebra R A]
    [Algebra R B] (f : A ≃ₐ[R] B) (a : A) :
    f.toUnder.hom.right a = f a := rfl


@[simp]
lemma toUnder_inv_right_apply {A B : Type u} [CommRing A] [CommRing B] [Algebra R A]
    [Algebra R B] (f : A ≃ₐ[R] B) (b : B) :
    f.toUnder.inv.right b = f.symm b := rfl


@[simp]
lemma toUnder_trans {A B C : Type u} [CommRing A] [CommRing B] [CommRing C]
    [Algebra R A] [Algebra R B] [Algebra R C] (f : A ≃ₐ[R] B) (g : B ≃ₐ[R] C) :
    (f.trans g).toUnder = f.toUnder ≪≫ g.toUnder :=
  rfl


variable (R S) in
/-- The base change functor `A ↦ S ⊗[R] A`. -/
@[simps! map_right]
def tensorProd : Under R ⥤ Under S where
  obj A := mkUnder S (S ⊗[R] A)
  map f := Algebra.TensorProduct.map (AlgHom.id S S) (toAlgHom f) |>.toUnder
                             /-
                               R S : CommRingCat
                               inst✝ : Algebra ↑R ↑S
                               X Y Z : CategoryTheory.Under R
                               f : Quiver.Hom X Y
                               g : Quiver.Hom Y Z
                               ⊢ Eq ({ obj := fun A => S.mkUnder (TensorProduct ↑R ↑S ↑A.right), map := fun { …
                             -/
  map_comp {X Y Z} f g := by simp [Algebra.TensorProduct.map_id_comp]
                             /-
                               🎉 no goals
                             -/


variable (S) in
/-- The natural isomorphism `S ⊗[R] A ≅ pushout A.hom (algebraMap R S)` in `Under S`. -/
def tensorProdObjIsoPushoutObj (A : Under R) :
    mkUnder S (S ⊗[R] A) ≅ (Under.pushout (ofHom <| algebraMap R S)).obj A :=
  Under.isoMk (CommRingCat.isPushout_tensorProduct R S A).flip.isoPushout <| by
    simp only [Functor.const_obj_obj, Under.pushout_obj, Functor.id_obj, Under.mk_right,
      mkUnder_hom, AlgHom.toRingHom_eq_coe, IsPushout.inr_isoPushout_hom, Under.mk_hom]
    /-
      R S : CommRingCat
      inst✝ : Algebra ↑R ↑S
      A : CategoryTheory.Under R
      ⊢ Eq (CategoryTheory.Limits.pushout.inr (CommRingCat.ofHom (algebraMap ↑R ↑A.r …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
lemma pushout_inl_tensorProdObjIsoPushoutObj_inv_right (A : Under R) :
    pushout.inl A.hom (ofHom <| algebraMap R S) ≫ (tensorProdObjIsoPushoutObj S A).inv.right =
      (ofHom <| Algebra.TensorProduct.includeRight.toRingHom) := by
  /-
    R S : CommRingCat
    inst✝ : Algebra ↑R ↑S
    A : CategoryTheory.Under R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl A. …
  -/
  simp [tensorProdObjIsoPushoutObj]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma pushout_inr_tensorProdObjIsoPushoutObj_inv_right (A : Under R) :
    pushout.inr A.hom (ofHom <| algebraMap R S) ≫
      (tensorProdObjIsoPushoutObj S A).inv.right =
      (CommRingCat.ofHom <| Algebra.TensorProduct.includeLeftRingHom) := by
  /-
    R S : CommRingCat
    inst✝ : Algebra ↑R ↑S
    A : CategoryTheory.Under R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr A. …
  -/
  simp [tensorProdObjIsoPushoutObj]
  /-
    🎉 no goals
  -/


variable (R S) in
/-- `A ↦ S ⊗[R] A` is naturally isomorphic to `A ↦ pushout A.hom (algebraMap R S)`. -/
def tensorProdIsoPushout : tensorProd R S ≅ Under.pushout (ofHom <| algebraMap R S) :=
  NatIso.ofComponents (fun A ↦ tensorProdObjIsoPushoutObj S A) <| by
    /-
      R S : CommRingCat
      inst✝ : Algebra ↑R ↑S
      ⊢ ∀ {X Y : CategoryTheory.Under R} (f : Quiver.Hom X Y), Eq (CategoryTheory.Ca …
    -/
    intro A B f
    /-
      R S : CommRingCat
      inst✝ : Algebra ↑R ↑S
      A B : CategoryTheory.Under R
      f : Quiver.Hom A B
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((R.tensorProd S).map f) ((fun A => C …
    -/
    dsimp
    /-
      R S : CommRingCat
      inst✝ : Algebra ↑R ↑S
      A B : CategoryTheory.Under R
      f : Quiver.Hom A B
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((R.tensorProd S).map f) (CommRingCat …
    -/
    rw [← cancel_epi (tensorProdObjIsoPushoutObj S A).inv]
    /-
      R S : CommRingCat
      inst✝ : Algebra ↑R ↑S
      A B : CategoryTheory.Under R
      f : Quiver.Hom A B
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.tensorProdObjIsoPushoutO …
    -/
    ext : 1
    /-
      case h
      R S : CommRingCat
      inst✝ : Algebra ↑R ↑S
      A B : CategoryTheory.Under R
      f : Quiver.Hom A B
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.tensorProdObjIsoPushoutO …
    -/
    apply pushout.hom_ext
      /-
        case h.h₀
        R S : CommRingCat
        inst✝ : Algebra ↑R ↑S
        A B : CategoryTheory.Under R
        f : Quiver.Hom A B
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl A. …
      -/
    · rw [← cancel_mono (tensorProdObjIsoPushoutObj S B).inv.right]
      /-
        case h.h₀
        R S : CommRingCat
        inst✝ : Algebra ↑R ↑S
        A B : CategoryTheory.Under R
        f : Quiver.Hom A B
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      ext x
      /-
        case h.h₀.hf.a
        R S : CommRingCat
        inst✝ : Algebra ↑R ↑S
        A B : CategoryTheory.Under R
        f : Quiver.Hom A B
        x : ↑((CategoryTheory.Functor.id CommRingCat).obj A.right)
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
      -/
      simp [mkUnder_right]
      /-
        🎉 no goals
      -/
      /-
        case h.h₁
        R S : CommRingCat
        inst✝ : Algebra ↑R ↑S
        A B : CategoryTheory.Under R
        f : Quiver.Hom A B
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr A. …
      -/
    · rw [← cancel_mono (tensorProdObjIsoPushoutObj S B).inv.right]
      /-
        case h.h₁
        R S : CommRingCat
        inst✝ : Algebra ↑R ↑S
        A B : CategoryTheory.Under R
        f : Quiver.Hom A B
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      ext (x : S)
      /-
        case h.h₁.hf.a
        R S : CommRingCat
        inst✝ : Algebra ↑R ↑S
        A B : CategoryTheory.Under R
        f : Quiver.Hom A B
        x : ↑S
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
      -/
      simp [mkUnder_right]
      /-
        🎉 no goals
      -/


@[simp]
lemma tensorProdIsoPushout_app (A : Under R) :
    (tensorProdIsoPushout R S).app A = tensorProdObjIsoPushoutObj S A :=
  rfl


