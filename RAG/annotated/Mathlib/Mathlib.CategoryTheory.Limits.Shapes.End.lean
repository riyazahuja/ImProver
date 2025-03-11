/-- Given `F : Jᵒᵖ ⥤ J ⥤ C`, this is the multicospan index which shall be used
to define the end of `F`. -/
@[simps]
def multicospanIndexEnd : MulticospanIndex C where
  L := J
  R := Arrow J
  fstTo f := f.left
  sndTo f := f.right
  left j := (F.obj (op j)).obj j
  right f := (F.obj (op f.left)).obj f.right
  fst f := (F.obj (op f.left)).map f.hom
  snd f := (F.map f.hom.op).app f.right


/-- Given `F : Jᵒᵖ ⥤ J ⥤ C`, a wedge for `F` is a type of cones (specifically
the type of multiforks for `multicospanIndexEnd F`):
the point of universal of these wedges shall be the end of `F`. -/
abbrev Wedge := Multifork (multicospanIndexEnd F)


/-- Constructor for wedges. -/
@[simps! pt]
abbrev mk : Wedge F :=
  Multifork.ofι _ pt π (fun f ↦ hπ f.hom)


@[simp]
lemma mk_ι (j : J) : (mk pt π hπ).ι j = π j := rfl


@[reassoc]
lemma condition (c : Wedge F) {i j : J} (f : i ⟶ j) :
    c.ι i ≫ (F.obj (op i)).map f = c.ι j ≫ (F.map f.op).app j :=
  Multifork.condition c (Arrow.mk f)


lemma hom_ext (hc : IsLimit c) {X : C} {f g : X ⟶ c.pt} (h : ∀ j, f ≫ c.ι j = g ≫ c.ι j) :
    f = g :=
  Multifork.IsLimit.hom_ext hc h


/-- Construct a morphism to the end from its universal property. -/
def lift (hc : IsLimit c) {X : C} (f : ∀ j, X ⟶ (F.obj (op j)).obj j)
    (hf : ∀ ⦃i j : J⦄ (g : i ⟶ j), f i ≫ (F.obj (op i)).map g = f j ≫ (F.map g.op).app j) :
    X ⟶ c.pt :=
  Multifork.IsLimit.lift hc f (fun _ ↦ hf _)


@[reassoc (attr := simp)]
lemma lift_ι (hc : IsLimit c) {X : C} (f : ∀ j, X ⟶ (F.obj (op j)).obj j)
    (hf : ∀ ⦃i j : J⦄ (g : i ⟶ j), f i ≫ (F.obj (op i)).map g = f j ≫ (F.map g.op).app j) (j : J) :
    lift hc f hf ≫ c.ι j = f j := by
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} J
    C : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} C
    F : CategoryTheory.Functor (Opposite J) (CategoryTheory.Functor J C)
    c : CategoryTheory.Limits.Wedge F
    hc : CategoryTheory.Limits.IsLimit c
    X : C
    f : (j : J) → Quiver.Hom X ((F.obj { unop := j }).obj j)
    hf : ∀ ⦃i j : J⦄ (g : Quiver.Hom i j), Eq (CategoryTheory.CategoryStruct.comp  …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Wedge.IsLimit. …
  -/
  apply IsLimit.fac
  /-
    🎉 no goals
  -/



/-- Given `F : Jᵒᵖ ⥤ J ⥤ C`, this property asserts the existence of the end of `F`. -/
abbrev HasEnd := HasMultiequalizer (multicospanIndexEnd F)


/-- The end of a functor `F : Jᵒᵖ ⥤ J ⥤ C`. -/
noncomputable def end_ : C := multiequalizer (multicospanIndexEnd F)


/-- Given `F : Jᵒᵖ ⥤ J ⥤ C`, this is the projection `end_ F ⟶ (F.obj (op j)).obj j`
for any `j : J`. -/
noncomputable def end_.π (j : J) : end_ F ⟶ (F.obj (op j)).obj j := Multiequalizer.ι _ _


@[reassoc]
lemma end_.condition {i j : J} (f : i ⟶ j) :
    π F i ≫ (F.obj (op i)).map f = π F j ≫ (F.map f.op).app j := by
  /-
    J : Type u
    inst✝² : CategoryTheory.Category.{v, u} J
    C : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} C
    F : CategoryTheory.Functor (Opposite J) (CategoryTheory.Functor J C)
    inst✝ : CategoryTheory.Limits.HasEnd F
    i j : J
    f : Quiver.Hom i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.end_.π F i) (( …
  -/
  apply Wedge.condition
  /-
    🎉 no goals
  -/


@[ext]
lemma hom_ext {X : C} {f g : X ⟶ end_ F} (h : ∀ j, f ≫ end_.π F j = g ≫ end_.π F j) :
    f = g :=
  Multiequalizer.hom_ext _ _ _ (fun _ ↦ h _)


/-- Constructor for morphisms to the end of a functor. -/
noncomputable def end_.lift : X ⟶ end_ F :=
  Wedge.IsLimit.lift (limit.isLimit _) f hf


@[reassoc (attr := simp)]
lemma end_.lift_π (j : J) : lift f hf ≫ π F j = f j := by
  /-
    J : Type u
    inst✝² : CategoryTheory.Category.{v, u} J
    C : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} C
    F : CategoryTheory.Functor (Opposite J) (CategoryTheory.Functor J C)
    inst✝ : CategoryTheory.Limits.HasEnd F
    X : C
    f : (j : J) → Quiver.Hom X ((F.obj { unop := j }).obj j)
    hf : ∀ ⦃i j : J⦄ (g : Quiver.Hom i j), Eq (CategoryTheory.CategoryStruct.comp  …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.end_.lift f hf …
  -/
  apply IsLimit.fac
  /-
    🎉 no goals
  -/


