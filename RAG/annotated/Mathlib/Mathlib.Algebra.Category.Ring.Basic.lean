/-- The category of semirings. -/
structure SemiRingCat where
  private mk ::
  /-- The underlying type. -/
  carrier : Type u
  [semiring : Semiring carrier]


instance : CoeSort (SemiRingCat) (Type u) :=
  ⟨SemiRingCat.carrier⟩


/-- The object in the category of R-algebras associated to a type equipped with the appropriate
typeclasses. This is the preferred way to construct a term of `SemiRingCat`. -/
abbrev of (R : Type u) [Semiring R] : SemiRingCat :=
  ⟨R⟩


lemma coe_of (R : Type u) [Semiring R] : (of R : Type u) = R :=
  rfl


lemma of_carrier (R : SemiRingCat.{u}) : of R = R := rfl


variable {R} in
/-- The type of morphisms in `SemiRingCat`. -/
@[ext]
structure Hom (R S : SemiRingCat) where
  private mk ::
  /-- The underlying ring hom. -/
  hom : R →+* S


instance : Category SemiRingCat where
  Hom R S := Hom R S
  id R := ⟨RingHom.id R⟩
  comp f g := ⟨g.hom.comp f.hom⟩


instance {R S : SemiRingCat.{u}} : CoeFun (R ⟶ S) (fun _ ↦ R → S) where
  coe f := f.hom


@[simp]
lemma hom_id {R : SemiRingCat} : (𝟙 R : R ⟶ R).hom = RingHom.id R := rfl

/- Provided for rewriting. -/

lemma id_apply (R : SemiRingCat) (r : R) :
                              /-
                                R : SemiRingCat
                                r : ↑R
                                ⊢ Eq ((CategoryTheory.CategoryStruct.id R).hom r) r
                              -/
    (𝟙 R : R ⟶ R) r = r := by simp
                              /-
                                🎉 no goals
                              -/


@[simp]
lemma hom_comp {R S T : SemiRingCat} (f : R ⟶ S) (g : S ⟶ T) :
    (f ≫ g).hom = g.hom.comp f.hom := rfl

/- Provided for rewriting. -/

lemma comp_apply {R S T : SemiRingCat} (f : R ⟶ S) (g : S ⟶ T) (r : R) :
                              /-
                                R S T : SemiRingCat
                                f : Quiver.Hom R S
                                g : Quiver.Hom S T
                                r : ↑R
                                ⊢ Eq ((CategoryTheory.CategoryStruct.comp f g).hom r) (g.hom (f.hom r))
                              -/
    (f ≫ g) r = g (f r) := by simp
                              /-
                                🎉 no goals
                              -/


@[ext]
lemma hom_ext {R S : SemiRingCat} {f g : R ⟶ S} (hf : f.hom = g.hom) : f = g :=
  Hom.ext hf


/-- Typecheck a `RingHom` as a morphism in `SemiRingCat`. -/
abbrev ofHom {R S : Type u} [Semiring R] [Semiring S] (f : R →+* S) : of R ⟶ of S :=
  ⟨f⟩


lemma hom_ofHom {R S : Type u} [Semiring R] [Semiring S] (f : R →+* S) : (ofHom f).hom = f := rfl


@[simp]
lemma ofHom_hom {R S : SemiRingCat} (f : R ⟶ S) :
    ofHom (Hom.hom f) = f := rfl


@[simp]
lemma ofHom_id {R : Type u} [Semiring R] : ofHom (RingHom.id R) = 𝟙 (of R) := rfl


@[simp]
lemma ofHom_comp {R S T : Type u} [Semiring R] [Semiring S] [Semiring T]
    (f : R →+* S) (g : S →+* T) :
    ofHom (g.comp f) = ofHom f ≫ ofHom g :=
  rfl


lemma ofHom_apply {R S : Type u} [Semiring R] [Semiring S]
    (f : R →+* S) (r : R) : ofHom f r = f r := rfl


@[simp]
lemma inv_hom_apply {R S : SemiRingCat} (e : R ≅ S) (r : R) : e.inv (e.hom r) = r := by
  /-
    R S : SemiRingCat
    e : CategoryTheory.Iso R S
    r : ↑R
    ⊢ Eq (e.inv.hom (e.hom.hom r)) r
  -/
  rw [← comp_apply]
  /-
    R S : SemiRingCat
    e : CategoryTheory.Iso R S
    r : ↑R
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp e.hom e.inv).hom r) r
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma hom_inv_apply {R S : SemiRingCat} (e : R ≅ S) (s : S) : e.hom (e.inv s) = s := by
  /-
    R S : SemiRingCat
    e : CategoryTheory.Iso R S
    s : ↑S
    ⊢ Eq (e.hom.hom (e.inv.hom s)) s
  -/
  rw [← comp_apply]
  /-
    R S : SemiRingCat
    e : CategoryTheory.Iso R S
    s : ↑S
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp e.inv e.hom).hom s) s
  -/
  simp
  /-
    🎉 no goals
  -/


instance : Inhabited SemiRingCat :=
  ⟨of PUnit⟩


instance : ConcreteCategory.{u} SemiRingCat where
  forget :=
    { obj := fun R => R
      map := fun f => f.hom }
                                  /-
                                    X✝ Y✝ : SemiRingCat
                                    a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
                                    h : Eq ({ obj := fun R => ↑R, map := fun {X Y} f => ⇑f.hom, map_id := ⋯, map_c …
                                    ⊢ Eq a₁✝ a₂✝
                                  -/
  forget_faithful := ⟨fun h => by ext x; simpa using congrFun h x⟩
                                         /-
                                           🎉 no goals
                                         -/


lemma forget_obj {R : SemiRingCat} : (forget SemiRingCat).obj R = R := rfl


lemma forget_map {R S : SemiRingCat} (f : R ⟶ S) :
    (forget SemiRingCat).map f = f :=
  rfl


instance {R : SemiRingCat} : Semiring ((forget SemiRingCat).obj R) :=
  (inferInstance : Semiring R.carrier)


instance hasForgetToMonCat : HasForget₂ SemiRingCat MonCat where
  forget₂ :=
    { obj := fun R ↦ MonCat.of R
      map := fun f ↦ f.hom.toMonoidHom }


instance hasForgetToAddCommMonCat : HasForget₂ SemiRingCat AddCommMonCat where
  forget₂ :=
    { obj := fun R ↦ AddCommMonCat.of R
      map := fun f ↦ f.hom.toAddMonoidHom }


/-- Ring equivalence are isomorphisms in category of semirings -/
@[simps]
def _root_.RingEquiv.toSemiRingCatIso {R S : Type u} [Semiring R] [Semiring S] (e : R ≃+* S) :
    of R ≅ of S where
  hom := ⟨e⟩
  inv := ⟨e.symm⟩


instance forgetReflectIsos : (forget SemiRingCat).ReflectsIsomorphisms where
  reflects {X Y} f _ := by
    /-
      X Y : SemiRingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget SemiRingCat).map f)
      ⊢ CategoryTheory.IsIso f
    -/
    let i := asIso ((forget SemiRingCat).map f)
    /-
      X Y : SemiRingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget SemiRingCat).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget SemiRingCat).obj X) ((CategoryT …
      ⊢ CategoryTheory.IsIso f
    -/
    let ff : X →+* Y := f.hom
    /-
      X Y : SemiRingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget SemiRingCat).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget SemiRingCat).obj X) ((CategoryT …
      ff : RingHom ↑X ↑Y := f.hom
      ⊢ CategoryTheory.IsIso f
    -/
    let e : X ≃+* Y := { ff, i.toEquiv with }
    /-
      X Y : SemiRingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget SemiRingCat).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget SemiRingCat).obj X) ((CategoryT …
      ff : RingHom ↑X ↑Y := f.hom
      e : RingEquiv ↑X ↑Y :=
        let __src := i.toEquiv;
        { toFun := (↑↑ff).toFun, invFun := __src.invFun, left_inv := ⋯, right_inv := …
      ⊢ CategoryTheory.IsIso f
    -/
    exact e.toSemiRingCatIso.isIso_hom
    /-
      🎉 no goals
    -/


/-- The category of semirings. -/
structure RingCat where
  private mk ::
  /-- The underlying type. -/
  carrier : Type u
  [ring : Ring carrier]


instance : CoeSort (RingCat) (Type u) :=
  ⟨RingCat.carrier⟩


/-- The object in the category of R-algebras associated to a type equipped with the appropriate
typeclasses. This is the preferred way to construct a term of `RingCat`. -/
abbrev of (R : Type u) [Ring R] : RingCat :=
  ⟨R⟩


lemma coe_of (R : Type u) [Ring R] : (of R : Type u) = R :=
  rfl


lemma of_carrier (R : RingCat.{u}) : of R = R := rfl


variable {R} in
/-- The type of morphisms in `RingCat`. -/
@[ext]
structure Hom (R S : RingCat) where
  private mk ::
  /-- The underlying ring hom. -/
  hom : R →+* S


instance : Category RingCat where
  Hom R S := Hom R S
  id R := ⟨RingHom.id R⟩
  comp f g := ⟨g.hom.comp f.hom⟩


instance {R S : RingCat.{u}} : CoeFun (R ⟶ S) (fun _ ↦ R → S) where
  coe f := f.hom


@[simp]
lemma hom_id {R : RingCat} : (𝟙 R : R ⟶ R).hom = RingHom.id R := rfl

/- Provided for rewriting. -/

lemma id_apply (R : RingCat) (r : R) :
                              /-
                                R : RingCat
                                r : ↑R
                                ⊢ Eq ((CategoryTheory.CategoryStruct.id R).hom r) r
                              -/
    (𝟙 R : R ⟶ R) r = r := by simp
                              /-
                                🎉 no goals
                              -/


@[simp]
lemma hom_comp {R S T : RingCat} (f : R ⟶ S) (g : S ⟶ T) :
    (f ≫ g).hom = g.hom.comp f.hom := rfl

/- Provided for rewriting. -/

lemma comp_apply {R S T : RingCat} (f : R ⟶ S) (g : S ⟶ T) (r : R) :
                              /-
                                R S T : RingCat
                                f : Quiver.Hom R S
                                g : Quiver.Hom S T
                                r : ↑R
                                ⊢ Eq ((CategoryTheory.CategoryStruct.comp f g).hom r) (g.hom (f.hom r))
                              -/
    (f ≫ g) r = g (f r) := by simp
                              /-
                                🎉 no goals
                              -/


@[ext]
lemma hom_ext {R S : RingCat} {f g : R ⟶ S} (hf : f.hom = g.hom) : f = g :=
  Hom.ext hf


/-- Typecheck a `RingHom` as a morphism in `RingCat`. -/
abbrev ofHom {R S : Type u} [Ring R] [Ring S] (f : R →+* S) : of R ⟶ of S :=
  ⟨f⟩


lemma hom_ofHom {R S : Type u} [Ring R] [Ring S] (f : R →+* S) : (ofHom f).hom = f := rfl


@[simp]
lemma ofHom_hom {R S : RingCat} (f : R ⟶ S) :
    ofHom (Hom.hom f) = f := rfl


@[simp]
lemma ofHom_id {R : Type u} [Ring R] : ofHom (RingHom.id R) = 𝟙 (of R) := rfl


@[simp]
lemma ofHom_comp {R S T : Type u} [Ring R] [Ring S] [Ring T]
    (f : R →+* S) (g : S →+* T) :
    ofHom (g.comp f) = ofHom f ≫ ofHom g :=
  rfl


lemma ofHom_apply {R S : Type u} [Ring R] [Ring S]
    (f : R →+* S) (r : R) : ofHom f r = f r := rfl


@[simp]
lemma inv_hom_apply {R S : RingCat} (e : R ≅ S) (r : R) : e.inv (e.hom r) = r := by
  /-
    R S : RingCat
    e : CategoryTheory.Iso R S
    r : ↑R
    ⊢ Eq (e.inv.hom (e.hom.hom r)) r
  -/
  rw [← comp_apply]
  /-
    R S : RingCat
    e : CategoryTheory.Iso R S
    r : ↑R
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp e.hom e.inv).hom r) r
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma hom_inv_apply {R S : RingCat} (e : R ≅ S) (s : S) : e.hom (e.inv s) = s := by
  /-
    R S : RingCat
    e : CategoryTheory.Iso R S
    s : ↑S
    ⊢ Eq (e.hom.hom (e.inv.hom s)) s
  -/
  rw [← comp_apply]
  /-
    R S : RingCat
    e : CategoryTheory.Iso R S
    s : ↑S
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp e.inv e.hom).hom s) s
  -/
  simp
  /-
    🎉 no goals
  -/


instance : Inhabited RingCat :=
  ⟨of PUnit⟩


instance : ConcreteCategory.{u} RingCat where
  forget :=
    { obj := fun R => R
      map := fun f => f.hom }
                                  /-
                                    X✝ Y✝ : RingCat
                                    a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
                                    h : Eq ({ obj := fun R => ↑R, map := fun {X Y} f => ⇑f.hom, map_id := ⋯, map_c …
                                    ⊢ Eq a₁✝ a₂✝
                                  -/
  forget_faithful := ⟨fun h => by ext x; simpa using congrFun h x⟩
                                         /-
                                           🎉 no goals
                                         -/


lemma forget_obj {R : RingCat} : (forget RingCat).obj R = R := rfl


lemma forget_map {R S : RingCat} (f : R ⟶ S) :
    (forget RingCat).map f = f :=
  rfl


instance {R : RingCat} : Ring ((forget RingCat).obj R) :=
  (inferInstance : Ring R.carrier)


instance hasForgetToSemiRingCat : HasForget₂ RingCat SemiRingCat where
  forget₂ :=
    { obj := fun R ↦ SemiRingCat.of R
      map := fun f ↦ SemiRingCat.ofHom f.hom }


instance hasForgetToAddCommGrp : HasForget₂ RingCat AddCommGrp where
  forget₂ :=
    { obj := fun R ↦ AddCommGrp.of R
      map := fun f ↦ f.hom.toAddMonoidHom }


/-- Ring equivalence are isomorphisms in category of semirings -/
@[simps]
def _root_.RingEquiv.toRingCatIso {R S : Type u} [Ring R] [Ring S] (e : R ≃+* S) :
    of R ≅ of S where
  hom := ⟨e⟩
  inv := ⟨e.symm⟩


instance forgetReflectIsos : (forget RingCat).ReflectsIsomorphisms where
  reflects {X Y} f _ := by
    /-
      X Y : RingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget RingCat).map f)
      ⊢ CategoryTheory.IsIso f
    -/
    let i := asIso ((forget RingCat).map f)
    /-
      X Y : RingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget RingCat).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget RingCat).obj X) ((CategoryTheor …
      ⊢ CategoryTheory.IsIso f
    -/
    let ff : X →+* Y := f.hom
    /-
      X Y : RingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget RingCat).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget RingCat).obj X) ((CategoryTheor …
      ff : RingHom ↑X ↑Y := f.hom
      ⊢ CategoryTheory.IsIso f
    -/
    let e : X ≃+* Y := { ff, i.toEquiv with }
    /-
      X Y : RingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget RingCat).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget RingCat).obj X) ((CategoryTheor …
      ff : RingHom ↑X ↑Y := f.hom
      e : RingEquiv ↑X ↑Y :=
        let __src := i.toEquiv;
        { toFun := (↑↑ff).toFun, invFun := __src.invFun, left_inv := ⋯, right_inv := …
      ⊢ CategoryTheory.IsIso f
    -/
    exact e.toRingCatIso.isIso_hom
    /-
      🎉 no goals
    -/


/-- The category of semirings. -/
structure CommSemiRingCat where
  private mk ::
  /-- The underlying type. -/
  carrier : Type u
  [commSemiring : CommSemiring carrier]


instance : CoeSort (CommSemiRingCat) (Type u) :=
  ⟨CommSemiRingCat.carrier⟩


/-- The object in the category of R-algebras associated to a type equipped with the appropriate
typeclasses. This is the preferred way to construct a term of `CommSemiRingCat`. -/
abbrev of (R : Type u) [CommSemiring R] : CommSemiRingCat :=
  ⟨R⟩


lemma coe_of (R : Type u) [CommSemiring R] : (of R : Type u) = R :=
  rfl


lemma of_carrier (R : CommSemiRingCat.{u}) : of R = R := rfl


variable {R} in
/-- The type of morphisms in `CommSemiRingCat`. -/
@[ext]
structure Hom (R S : CommSemiRingCat) where
  private mk ::
  /-- The underlying ring hom. -/
  hom : R →+* S


instance : Category CommSemiRingCat where
  Hom R S := Hom R S
  id R := ⟨RingHom.id R⟩
  comp f g := ⟨g.hom.comp f.hom⟩


instance {R S : CommSemiRingCat.{u}} : CoeFun (R ⟶ S) (fun _ ↦ R → S) where
  coe f := f.hom


@[simp]
lemma hom_id {R : CommSemiRingCat} : (𝟙 R : R ⟶ R).hom = RingHom.id R := rfl

/- Provided for rewriting. -/

lemma id_apply (R : CommSemiRingCat) (r : R) :
                              /-
                                R : CommSemiRingCat
                                r : ↑R
                                ⊢ Eq ((CategoryTheory.CategoryStruct.id R).hom r) r
                              -/
    (𝟙 R : R ⟶ R) r = r := by simp
                              /-
                                🎉 no goals
                              -/


@[simp]
lemma hom_comp {R S T : CommSemiRingCat} (f : R ⟶ S) (g : S ⟶ T) :
    (f ≫ g).hom = g.hom.comp f.hom := rfl

/- Provided for rewriting. -/

lemma comp_apply {R S T : CommSemiRingCat} (f : R ⟶ S) (g : S ⟶ T) (r : R) :
                              /-
                                R S T : CommSemiRingCat
                                f : Quiver.Hom R S
                                g : Quiver.Hom S T
                                r : ↑R
                                ⊢ Eq ((CategoryTheory.CategoryStruct.comp f g).hom r) (g.hom (f.hom r))
                              -/
    (f ≫ g) r = g (f r) := by simp
                              /-
                                🎉 no goals
                              -/


@[ext]
lemma hom_ext {R S : CommSemiRingCat} {f g : R ⟶ S} (hf : f.hom = g.hom) : f = g :=
  Hom.ext hf


/-- Typecheck a `RingHom` as a morphism in `CommSemiRingCat`. -/
abbrev ofHom {R S : Type u} [CommSemiring R] [CommSemiring S] (f : R →+* S) : of R ⟶ of S :=
  ⟨f⟩


lemma hom_ofHom {R S : Type u} [CommSemiring R] [CommSemiring S] (f : R →+* S) :
  (ofHom f).hom = f := rfl


@[simp]
lemma ofHom_hom {R S : CommSemiRingCat} (f : R ⟶ S) :
    ofHom (Hom.hom f) = f := rfl


@[simp]
lemma ofHom_id {R : Type u} [CommSemiring R] : ofHom (RingHom.id R) = 𝟙 (of R) := rfl


@[simp]
lemma ofHom_comp {R S T : Type u} [CommSemiring R] [CommSemiring S] [CommSemiring T]
    (f : R →+* S) (g : S →+* T) :
    ofHom (g.comp f) = ofHom f ≫ ofHom g :=
  rfl


lemma ofHom_apply {R S : Type u} [CommSemiring R] [CommSemiring S]
    (f : R →+* S) (r : R) : ofHom f r = f r := rfl


@[simp]
lemma inv_hom_apply {R S : CommSemiRingCat} (e : R ≅ S) (r : R) : e.inv (e.hom r) = r := by
  /-
    R S : CommSemiRingCat
    e : CategoryTheory.Iso R S
    r : ↑R
    ⊢ Eq (e.inv.hom (e.hom.hom r)) r
  -/
  rw [← comp_apply]
  /-
    R S : CommSemiRingCat
    e : CategoryTheory.Iso R S
    r : ↑R
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp e.hom e.inv).hom r) r
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma hom_inv_apply {R S : CommSemiRingCat} (e : R ≅ S) (s : S) : e.hom (e.inv s) = s := by
  /-
    R S : CommSemiRingCat
    e : CategoryTheory.Iso R S
    s : ↑S
    ⊢ Eq (e.hom.hom (e.inv.hom s)) s
  -/
  rw [← comp_apply]
  /-
    R S : CommSemiRingCat
    e : CategoryTheory.Iso R S
    s : ↑S
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp e.inv e.hom).hom s) s
  -/
  simp
  /-
    🎉 no goals
  -/


instance : Inhabited CommSemiRingCat :=
  ⟨of PUnit⟩


instance : ConcreteCategory.{u} CommSemiRingCat where
  forget :=
    { obj := fun R => R
      map := fun f => f.hom }
                                  /-
                                    X✝ Y✝ : CommSemiRingCat
                                    a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
                                    h : Eq ({ obj := fun R => ↑R, map := fun {X Y} f => ⇑f.hom, map_id := ⋯, map_c …
                                    ⊢ Eq a₁✝ a₂✝
                                  -/
  forget_faithful := ⟨fun h => by ext x; simpa using congrFun h x⟩
                                         /-
                                           🎉 no goals
                                         -/


lemma forget_obj {R : CommSemiRingCat} : (forget CommSemiRingCat).obj R = R := rfl


lemma forget_map {R S : CommSemiRingCat} (f : R ⟶ S) :
    (forget CommSemiRingCat).map f = f :=
  rfl


instance {R : CommSemiRingCat} : CommSemiring ((forget CommSemiRingCat).obj R) :=
  (inferInstance : CommSemiring R.carrier)


instance hasForgetToSemiRingCat : HasForget₂ CommSemiRingCat SemiRingCat where
  forget₂ :=
    { obj := fun R ↦ ⟨R⟩
      map := fun f ↦ ⟨f.hom⟩ }


/-- The forgetful functor from commutative rings to (multiplicative) commutative monoids. -/
instance hasForgetToCommMonCat : HasForget₂ CommSemiRingCat CommMonCat where
  forget₂ :=
    { obj := fun R ↦ CommMonCat.of R
      map := fun f ↦ f.hom.toMonoidHom }


/-- Ring equivalence are isomorphisms in category of semirings -/
@[simps]
def _root_.RingEquiv.toCommSemiRingCatIso
    {R S : Type u} [CommSemiring R] [CommSemiring S] (e : R ≃+* S) :
    of R ≅ of S where
  hom := ⟨e⟩
  inv := ⟨e.symm⟩


instance forgetReflectIsos : (forget CommSemiRingCat).ReflectsIsomorphisms where
  reflects {X Y} f _ := by
    /-
      X Y : CommSemiRingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget CommSemiRingCat).map f)
      ⊢ CategoryTheory.IsIso f
    -/
    let i := asIso ((forget CommSemiRingCat).map f)
    /-
      X Y : CommSemiRingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget CommSemiRingCat).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget CommSemiRingCat).obj X) ((Categ …
      ⊢ CategoryTheory.IsIso f
    -/
    let ff : X →+* Y := f.hom
    /-
      X Y : CommSemiRingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget CommSemiRingCat).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget CommSemiRingCat).obj X) ((Categ …
      ff : RingHom ↑X ↑Y := f.hom
      ⊢ CategoryTheory.IsIso f
    -/
    let e : X ≃+* Y := { ff, i.toEquiv with }
    /-
      X Y : CommSemiRingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget CommSemiRingCat).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget CommSemiRingCat).obj X) ((Categ …
      ff : RingHom ↑X ↑Y := f.hom
      e : RingEquiv ↑X ↑Y :=
        let __src := i.toEquiv;
        { toFun := (↑↑ff).toFun, invFun := __src.invFun, left_inv := ⋯, right_inv := …
      ⊢ CategoryTheory.IsIso f
    -/
    exact e.toCommSemiRingCatIso.isIso_hom
    /-
      🎉 no goals
    -/


/-- The category of semirings. -/
structure CommRingCat where
  private mk ::
  /-- The underlying type. -/
  carrier : Type u
  [commRing : CommRing carrier]


instance : CoeSort (CommRingCat) (Type u) :=
  ⟨CommRingCat.carrier⟩


/-- The object in the category of R-algebras associated to a type equipped with the appropriate
typeclasses. This is the preferred way to construct a term of `CommRingCat`. -/
abbrev of (R : Type u) [CommRing R] : CommRingCat :=
  ⟨R⟩


lemma coe_of (R : Type u) [CommRing R] : (of R : Type u) = R :=
  rfl


lemma of_carrier (R : CommRingCat.{u}) : of R = R := rfl


variable {R} in
/-- The type of morphisms in `CommRingCat`. -/
@[ext]
structure Hom (R S : CommRingCat) where
  private mk ::
  /-- The underlying ring hom. -/
  hom : R →+* S


instance : Category CommRingCat where
  Hom R S := Hom R S
  id R := ⟨RingHom.id R⟩
  comp f g := ⟨g.hom.comp f.hom⟩


instance {R S : CommRingCat.{u}} : CoeFun (R ⟶ S) (fun _ ↦ R → S) where
  coe f := f.hom


@[simp]
lemma hom_id {R : CommRingCat} : (𝟙 R : R ⟶ R).hom = RingHom.id R := rfl

/- Provided for rewriting. -/

lemma id_apply (R : CommRingCat) (r : R) :
                              /-
                                R : CommRingCat
                                r : ↑R
                                ⊢ Eq ((CategoryTheory.CategoryStruct.id R).hom r) r
                              -/
    (𝟙 R : R ⟶ R) r = r := by simp
                              /-
                                🎉 no goals
                              -/


@[simp]
lemma hom_comp {R S T : CommRingCat} (f : R ⟶ S) (g : S ⟶ T) :
    (f ≫ g).hom = g.hom.comp f.hom := rfl

/- Provided for rewriting. -/

lemma comp_apply {R S T : CommRingCat} (f : R ⟶ S) (g : S ⟶ T) (r : R) :
                              /-
                                R S T : CommRingCat
                                f : Quiver.Hom R S
                                g : Quiver.Hom S T
                                r : ↑R
                                ⊢ Eq ((CategoryTheory.CategoryStruct.comp f g).hom r) (g.hom (f.hom r))
                              -/
    (f ≫ g) r = g (f r) := by simp
                              /-
                                🎉 no goals
                              -/


@[ext]
lemma hom_ext {R S : CommRingCat} {f g : R ⟶ S} (hf : f.hom = g.hom) : f = g :=
  Hom.ext hf


/-- Typecheck a `RingHom` as a morphism in `CommRingCat`. -/
abbrev ofHom {R S : Type u} [CommRing R] [CommRing S] (f : R →+* S) : of R ⟶ of S :=
  ⟨f⟩


lemma hom_ofHom {R S : Type u} [CommRing R] [CommRing S] (f : R →+* S) :
  (ofHom f).hom = f := rfl


@[simp]
lemma ofHom_hom {R S : CommRingCat} (f : R ⟶ S) :
    ofHom (Hom.hom f) = f := rfl


@[simp]
lemma ofHom_id {R : Type u} [CommRing R] : ofHom (RingHom.id R) = 𝟙 (of R) := rfl


@[simp]
lemma ofHom_comp {R S T : Type u} [CommRing R] [CommRing S] [CommRing T]
    (f : R →+* S) (g : S →+* T) :
    ofHom (g.comp f) = ofHom f ≫ ofHom g :=
  rfl


lemma ofHom_apply {R S : Type u} [CommRing R] [CommRing S]
    (f : R →+* S) (r : R) : ofHom f r = f r := rfl


@[simp]
lemma inv_hom_apply {R S : CommRingCat} (e : R ≅ S) (r : R) : e.inv (e.hom r) = r := by
  /-
    R S : CommRingCat
    e : CategoryTheory.Iso R S
    r : ↑R
    ⊢ Eq (e.inv.hom (e.hom.hom r)) r
  -/
  rw [← comp_apply]
  /-
    R S : CommRingCat
    e : CategoryTheory.Iso R S
    r : ↑R
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp e.hom e.inv).hom r) r
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma hom_inv_apply {R S : CommRingCat} (e : R ≅ S) (s : S) : e.hom (e.inv s) = s := by
  /-
    R S : CommRingCat
    e : CategoryTheory.Iso R S
    s : ↑S
    ⊢ Eq (e.hom.hom (e.inv.hom s)) s
  -/
  rw [← comp_apply]
  /-
    R S : CommRingCat
    e : CategoryTheory.Iso R S
    s : ↑S
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp e.inv e.hom).hom s) s
  -/
  simp
  /-
    🎉 no goals
  -/


instance : Inhabited CommRingCat :=
  ⟨of PUnit⟩


instance : ConcreteCategory.{u} CommRingCat where
  forget :=
    { obj := fun R => R
      map := fun f => f.hom }
                                  /-
                                    X✝ Y✝ : CommRingCat
                                    a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
                                    h : Eq ({ obj := fun R => ↑R, map := fun {X Y} f => ⇑f.hom, map_id := ⋯, map_c …
                                    ⊢ Eq a₁✝ a₂✝
                                  -/
  forget_faithful := ⟨fun h => by ext x; simpa using congrFun h x⟩
                                         /-
                                           🎉 no goals
                                         -/


lemma forget_obj {R : CommRingCat} : (forget CommRingCat).obj R = R := rfl


lemma forget_map {R S : CommRingCat} (f : R ⟶ S) :
    (forget CommRingCat).map f = f :=
  rfl


instance {R : CommRingCat} : CommRing ((forget CommRingCat).obj R) :=
  (inferInstance : CommRing R.carrier)


instance hasForgetToRingCat : HasForget₂ CommRingCat RingCat where
  forget₂ :=
    { obj := fun R ↦ RingCat.of R
      map := fun f ↦ RingCat.ofHom f.hom }


@[simp] lemma forgetToRingCat_map_hom {R S : CommRingCat} (f : R ⟶ S) :
    ((forget₂ CommRingCat RingCat).map f).hom = f.hom :=
  rfl


@[simp] lemma forgetToRingCat_obj {R : CommRingCat} :
    (((forget₂ CommRingCat RingCat).obj R) : Type u) = R :=
  rfl


instance hasForgetToAddCommMonCat : HasForget₂ CommRingCat CommSemiRingCat where
  forget₂ :=
    { obj := fun R ↦ CommSemiRingCat.of R
      map := fun f ↦ CommSemiRingCat.ofHom f.hom }


/-- Ring equivalence are isomorphisms in category of semirings -/
@[simps]
def _root_.RingEquiv.toCommRingCatIso
    {R S : Type u} [CommRing R] [CommRing S] (e : R ≃+* S) :
    of R ≅ of S where
  hom := ⟨e⟩
  inv := ⟨e.symm⟩


instance forgetReflectIsos : (forget CommRingCat).ReflectsIsomorphisms where
  reflects {X Y} f _ := by
    /-
      X Y : CommRingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget CommRingCat).map f)
      ⊢ CategoryTheory.IsIso f
    -/
    let i := asIso ((forget CommRingCat).map f)
    /-
      X Y : CommRingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget CommRingCat).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget CommRingCat).obj X) ((CategoryT …
      ⊢ CategoryTheory.IsIso f
    -/
    let ff : X →+* Y := f.hom
    /-
      X Y : CommRingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget CommRingCat).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget CommRingCat).obj X) ((CategoryT …
      ff : RingHom ↑X ↑Y := f.hom
      ⊢ CategoryTheory.IsIso f
    -/
    let e : X ≃+* Y := { ff, i.toEquiv with }
    /-
      X Y : CommRingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget CommRingCat).map f)
      i : CategoryTheory.Iso ((CategoryTheory.forget CommRingCat).obj X) ((CategoryT …
      ff : RingHom ↑X ↑Y := f.hom
      e : RingEquiv ↑X ↑Y :=
        let __src := i.toEquiv;
        { toFun := (↑↑ff).toFun, invFun := __src.invFun, left_inv := ⋯, right_inv := …
      ⊢ CategoryTheory.IsIso f
    -/
    exact e.toCommRingCatIso.isIso_hom
    /-
      🎉 no goals
    -/


/-- Build a `RingEquiv` from an isomorphism in the category `SemiRingCat`. -/
def semiRingCatIsoToRingEquiv {R S : SemiRingCat.{u}} (e : R ≅ S) : R ≃+* S :=
                                             /-
                                               R S : SemiRingCat
                                               e : CategoryTheory.Iso R S
                                               ⊢ Eq ((↑e.inv.hom).comp ↑e.hom.hom) (RingHom.id ↑R)
                                             -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  RingEquiv.ofHomInv e.hom.hom e.inv.hom (by ext; simp) (by ext; simp)
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- Build a `RingEquiv` from an isomorphism in the category `RingCat`. -/
def ringCatIsoToRingEquiv {R S : RingCat.{u}} (e : R ≅ S) : R ≃+* S :=
                                             /-
                                               R S : RingCat
                                               e : CategoryTheory.Iso R S
                                               ⊢ Eq ((↑e.inv.hom).comp ↑e.hom.hom) (RingHom.id ↑R)
                                             -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  RingEquiv.ofHomInv e.hom.hom e.inv.hom (by ext; simp) (by ext; simp)
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- Build a `RingEquiv` from an isomorphism in the category `CommSemiRingCat`. -/
def commSemiRingCatIsoToRingEquiv {R S : CommSemiRingCat.{u}} (e : R ≅ S) : R ≃+* S :=
                                             /-
                                               R S : CommSemiRingCat
                                               e : CategoryTheory.Iso R S
                                               ⊢ Eq ((↑e.inv.hom).comp ↑e.hom.hom) (RingHom.id ↑R)
                                             -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  RingEquiv.ofHomInv e.hom.hom e.inv.hom (by ext; simp) (by ext; simp)
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- Build a `RingEquiv` from an isomorphism in the category `CommRingCat`. -/
def commRingCatIsoToRingEquiv {R S : CommRingCat.{u}} (e : R ≅ S) : R ≃+* S :=
                                             /-
                                               R S : CommRingCat
                                               e : CategoryTheory.Iso R S
                                               ⊢ Eq ((↑e.inv.hom).comp ↑e.hom.hom) (RingHom.id ↑R)
                                             -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  RingEquiv.ofHomInv e.hom.hom e.inv.hom (by ext; simp) (by ext; simp)
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp] lemma semiRingCatIsoToRingEquiv_toRingHom {R S : SemiRingCat.{u}} (e : R ≅ S) :
  (e.semiRingCatIsoToRingEquiv : R →+* S) = e.hom.hom := rfl


@[simp] lemma ringCatIsoToRingEquiv_toRingHom {R S : RingCat.{u}} (e : R ≅ S) :
  (e.ringCatIsoToRingEquiv : R →+* S) = e.hom.hom := rfl


@[simp] lemma commSemiRingCatIsoToRingEquiv_toRingHom {R S : CommSemiRingCat.{u}} (e : R ≅ S) :
  (e.commSemiRingCatIsoToRingEquiv : R →+* S) = e.hom.hom := rfl


@[simp] lemma commRingCatIsoToRingEquiv_toRingHom {R S : CommRingCat.{u}} (e : R ≅ S) :
  (e.commRingCatIsoToRingEquiv : R →+* S) = e.hom.hom := rfl


/-- An alias for `SemiringCat.{max u v}`, to deal around unification issues. -/
@[nolint checkUnivs]
abbrev SemiRingCatMax.{u1, u2} := SemiRingCat.{max u1 u2}


/-- An alias for `RingCat.{max u v}`, to deal around unification issues. -/
@[nolint checkUnivs]
abbrev RingCatMax.{u1, u2} := RingCat.{max u1 u2}


/-- An alias for `CommSemiRingCat.{max u v}`, to deal around unification issues. -/
@[nolint checkUnivs]
abbrev CommSemiRingCatMax.{u1, u2} := CommSemiRingCat.{max u1 u2}


/-- An alias for `CommRingCat.{max u v}`, to deal around unification issues. -/
@[nolint checkUnivs]
abbrev CommRingCatMax.{u1, u2} := CommRingCat.{max u1 u2}


lemma RingCat.forget_map_apply {R S : RingCat} (f : R ⟶ S)
    (x : (CategoryTheory.forget RingCat).obj R) :
    @DFunLike.coe _ _ _ ConcreteCategory.instFunLike f x = f x :=
  rfl


lemma CommRingCat.forget_map_apply {R S : CommRingCat} (f : R ⟶ S)
    (x : (CategoryTheory.forget CommRingCat).obj R) :
    @DFunLike.coe _ _ _ ConcreteCategory.instFunLike f x = f x :=
  rfl

