/--
The equivalence between natural transformations from the yoneda embedding (to the sheaf category)
and elements of `F.val.obj X`.
-/
def yonedaEquiv {X : C} {F : Sheaf J (Type v)} : (J.yoneda.obj X ⟶ F) ≃ F.val.obj (op X) :=
  (fullyFaithfulSheafToPresheaf _ _).homEquiv.trans CategoryTheory.yonedaEquiv


theorem yonedaEquiv_apply {X : C} {F : Sheaf J (Type v)} (f : J.yoneda.obj X ⟶ F) :
    yonedaEquiv J f = f.val.app (op X) (𝟙 X) :=
  rfl


@[simp]
theorem yonedaEquiv_symm_app_apply {X : C} {F : Sheaf J (Type v)} (x : F.val.obj (op X)) (Y : Cᵒᵖ)
    (f : Y.unop ⟶ X) : (J.yonedaEquiv.symm x).val.app Y f = F.val.map f.op x :=
  rfl


/-- See also `yonedaEquiv_naturality'` for a more general version. -/
lemma yonedaEquiv_naturality {X Y : C} {F : Sheaf J (Type v)} (f : J.yoneda.obj X ⟶ F)
    (g : Y ⟶ X) : F.val.map g.op (J.yonedaEquiv f) = J.yonedaEquiv (J.yoneda.map g ≫ f) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X Y : C
    F : CategoryTheory.Sheaf J (Type v)
    f : Quiver.Hom (J.yoneda.obj X) F
    g : Quiver.Hom Y X
    ⊢ Eq (F.val.map g.op (J.yonedaEquiv f)) (J.yonedaEquiv (CategoryTheory.Categor …
  -/
  simp [yonedaEquiv, CategoryTheory.yonedaEquiv_naturality]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X Y : C
    F : CategoryTheory.Sheaf J (Type v)
    f : Quiver.Hom (J.yoneda.obj X) F
    g : Quiver.Hom Y X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map g) ((Categ …
  -/
  rfl
  /-
    🎉 no goals
  -/


/--
Variant of `yonedaEquiv_naturality` with general `g`. This is technically strictly more general
than `yonedaEquiv_naturality`, but `yonedaEquiv_naturality` is sometimes preferable because it
can avoid the "motive is not type correct" error.
-/
lemma yonedaEquiv_naturality' {X Y : Cᵒᵖ} {F : Sheaf J (Type v)} (f : J.yoneda.obj (unop X) ⟶ F)
    (g : X ⟶ Y) : F.val.map g (J.yonedaEquiv f) = J.yonedaEquiv (J.yoneda.map g.unop ≫ f) :=
  J.yonedaEquiv_naturality _ _


lemma yonedaEquiv_comp {X : C} {F G : Sheaf J (Type v)} (α : J.yoneda.obj X ⟶ F) (β : F ⟶ G) :
    J.yonedaEquiv (α ≫ β) = β.val.app _ (J.yonedaEquiv α) :=
  rfl


lemma yonedaEquiv_yoneda_map {X Y : C} (f : X ⟶ Y) : J.yonedaEquiv (J.yoneda.map f) = f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (J.yonedaEquiv (J.yoneda.map f)) f
  -/
  rw [yonedaEquiv_apply]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq ((J.yoneda.map f).val.app { unop := X } (CategoryTheory.CategoryStruct.id …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma yonedaEquiv_symm_naturality_left {X X' : C} (f : X' ⟶ X) (F : Sheaf J (Type v))
    (x : F.val.obj ⟨X⟩) : J.yoneda.map f ≫ J.yonedaEquiv.symm x = J.yonedaEquiv.symm
      ((F.val.map f.op) x) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X X' : C
    f : Quiver.Hom X' X
    F : CategoryTheory.Sheaf J (Type v)
    x : F.val.obj { unop := X }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.yoneda.map f) (J.yonedaEquiv.symm  …
  -/
  apply J.yonedaEquiv.injective
  /-
    case a
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X X' : C
    f : Quiver.Hom X' X
    F : CategoryTheory.Sheaf J (Type v)
    x : F.val.obj { unop := X }
    ⊢ Eq (J.yonedaEquiv (CategoryTheory.CategoryStruct.comp (J.yoneda.map f) (J.yo …
  -/
  simp only [yonedaEquiv_comp, yoneda_obj_obj, yonedaEquiv_symm_app_apply, Equiv.apply_symm_apply]
  /-
    case a
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X X' : C
    f : Quiver.Hom X' X
    F : CategoryTheory.Sheaf J (Type v)
    x : F.val.obj { unop := X }
    ⊢ Eq (F.val.map (Quiver.Hom.op (J.yonedaEquiv (J.yoneda.map f))) x) (F.val.map …
  -/
  rw [yonedaEquiv_yoneda_map]
  /-
    🎉 no goals
  -/


lemma yonedaEquiv_symm_naturality_right (X : C) {F F' : Sheaf J (Type v)} (f : F ⟶ F')
    (x : F.val.obj ⟨X⟩) : J.yonedaEquiv.symm x ≫ f = J.yonedaEquiv.symm (f.val.app ⟨X⟩ x) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X : C
    F F' : CategoryTheory.Sheaf J (Type v)
    f : Quiver.Hom F F'
    x : F.val.obj { unop := X }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.yonedaEquiv.symm x) f) (J.yonedaEq …
  -/
  apply J.yonedaEquiv.injective
  /-
    case a
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X : C
    F F' : CategoryTheory.Sheaf J (Type v)
    f : Quiver.Hom F F'
    x : F.val.obj { unop := X }
    ⊢ Eq (J.yonedaEquiv (CategoryTheory.CategoryStruct.comp (J.yonedaEquiv.symm x) …
  -/
  simp [yonedaEquiv_comp]
  /-
    🎉 no goals
  -/


/-- See also `map_yonedaEquiv'` for a more general version. -/
lemma map_yonedaEquiv {X Y : C} {F : Sheaf J (Type v)} (f : J.yoneda.obj X ⟶ F)
    (g : Y ⟶ X) : F.val.map g.op (J.yonedaEquiv f) = f.val.app (op Y) g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X Y : C
    F : CategoryTheory.Sheaf J (Type v)
    f : Quiver.Hom (J.yoneda.obj X) F
    g : Quiver.Hom Y X
    ⊢ Eq (F.val.map g.op (J.yonedaEquiv f)) (f.val.app { unop := Y } g)
  -/
  rw [yonedaEquiv_naturality, yonedaEquiv_comp, yonedaEquiv_yoneda_map]
  /-
    🎉 no goals
  -/


/--
Variant of `map_yonedaEquiv` with general `g`. This is technically strictly more general
than `map_yonedaEquiv`, but `map_yonedaEquiv` is sometimes preferable because it
can avoid the "motive is not type correct" error.
-/
lemma map_yonedaEquiv' {X Y : Cᵒᵖ} {F : Sheaf J (Type v)} (f : J.yoneda.obj (unop X) ⟶ F)
    (g : X ⟶ Y) : F.val.map g (J.yonedaEquiv f) = f.val.app Y g.unop := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X Y : Opposite C
    F : CategoryTheory.Sheaf J (Type v)
    f : Quiver.Hom (J.yoneda.obj (Opposite.unop X)) F
    g : Quiver.Hom X Y
    ⊢ Eq (F.val.map g (J.yonedaEquiv f)) (f.val.app Y g.unop)
  -/
  rw [yonedaEquiv_naturality', yonedaEquiv_comp, yonedaEquiv_yoneda_map]
  /-
    🎉 no goals
  -/


lemma yonedaEquiv_symm_map {X Y : Cᵒᵖ} (f : X ⟶ Y) {F : Sheaf J (Type v)} (t : F.val.obj X) :
    J.yonedaEquiv.symm (F.val.map f t) = J.yoneda.map f.unop ≫ J.yonedaEquiv.symm t := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X Y : Opposite C
    f : Quiver.Hom X Y
    F : CategoryTheory.Sheaf J (Type v)
    t : F.val.obj X
    ⊢ Eq (J.yonedaEquiv.symm (F.val.map f t)) (CategoryTheory.CategoryStruct.comp  …
  -/
  obtain ⟨u, rfl⟩ := J.yonedaEquiv.surjective t
  /-
    case intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X Y : Opposite C
    f : Quiver.Hom X Y
    F : CategoryTheory.Sheaf J (Type v)
    u : Quiver.Hom (J.yoneda.obj (Opposite.unop X)) F
    ⊢ Eq (J.yonedaEquiv.symm (F.val.map f (J.yonedaEquiv u))) (CategoryTheory.Cate …
  -/
  rw [yonedaEquiv_naturality', Equiv.symm_apply_apply, Equiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


/--
Two morphisms of sheaves of types `P ⟶ Q` coincide if the precompositions with morphisms
`yoneda.obj X ⟶ P` agree.
-/
lemma hom_ext_yoneda {P Q : Sheaf J (Type v)} {f g : P ⟶ Q}
    (h : ∀ (X : C) (p : J.yoneda.obj X ⟶ P), p ≫ f = p ≫ g) :
    f = g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    P Q : CategoryTheory.Sheaf J (Type v)
    f g : Quiver.Hom P Q
    h : ∀ (X : C) (p : Quiver.Hom (J.yoneda.obj X) P), Eq (CategoryTheory.Category …
    ⊢ Eq f g
  -/
  ext X x
  simpa only [yonedaEquiv_comp, Equiv.apply_symm_apply]
    using congr_arg (J.yonedaEquiv) (h _ (J.yonedaEquiv.symm x))


/--
The Yoneda embedding into a category of sheaves taking values in sets possibly larger than the
morphisms in the defining site.
-/
@[pp_with_univ]
def yonedaULift : C ⥤ Sheaf J (Type (max v v')) := J.yoneda ⋙ sheafCompose J uliftFunctor.{v'}


/-- A version of `yonedaEquiv` for `yonedaULift`. -/
def yonedaULiftEquiv {X : C} {F : Sheaf J (Type (max v v'))} :
    ((yonedaULift.{v'} J).obj X ⟶ F) ≃ F.val.obj (op X) :=
  (fullyFaithfulSheafToPresheaf _ _).homEquiv.trans (yonedaCompUliftFunctorEquiv _ _)


theorem yonedaULiftEquiv_apply {X : C} {F : Sheaf J (Type (max v v'))}
    (f : J.yonedaULift.obj X ⟶ F) : yonedaULiftEquiv.{v'} J f = f.val.app (op X) ⟨𝟙 X⟩ :=
  rfl


@[simp]
theorem yonedaULiftEquiv_symm_app_apply {X : C} {F : Sheaf J (Type (max v v'))}
    (x : F.val.obj (op X)) (Y : Cᵒᵖ) (f : Y.unop ⟶ X) :
      (J.yonedaULiftEquiv.symm x).val.app Y ⟨f⟩ = F.val.map f.op x :=
  rfl


/-- See also `yonedaEquiv_naturality'` for a more general version. -/
lemma yonedaULiftEquiv_naturality {X Y : C} {F : Sheaf J (Type (max v v'))}
    (f : J.yonedaULift.obj X ⟶ F) (g : Y ⟶ X) :
      F.val.map g.op (J.yonedaULiftEquiv f) = J.yonedaULiftEquiv (J.yonedaULift.map g ≫ f) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X Y : C
    F : CategoryTheory.Sheaf J (Type (max v v'))
    f : Quiver.Hom ((CategoryTheory.GrothendieckTopology.yonedaULift.{v', v, u} J) …
    g : Quiver.Hom Y X
    ⊢ Eq (F.val.map g.op (J.yonedaULiftEquiv f)) (J.yonedaULiftEquiv (CategoryTheo …
  -/
  change (f.val.app (op X) ≫ F.val.map g.op) ⟨𝟙 X⟩ = f.val.app (op Y) ⟨𝟙 Y ≫ g⟩
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X Y : C
    F : CategoryTheory.Sheaf J (Type (max v v'))
    f : Quiver.Hom ((CategoryTheory.GrothendieckTopology.yonedaULift.{v', v, u} J) …
    g : Quiver.Hom Y X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.val.app { unop := X }) (F.val.map  …
  -/
  rw [← f.val.naturality]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X Y : C
    F : CategoryTheory.Sheaf J (Type (max v v'))
    f : Quiver.Hom ((CategoryTheory.GrothendieckTopology.yonedaULift.{v', v, u} J) …
    g : Quiver.Hom Y X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.GrothendieckTopolog …
  -/
  simp [yonedaULift]
  /-
    🎉 no goals
  -/


/-- Variant of `yonedaEquiv_naturality` with general `g`. This is technically strictly more general
    than `yonedaEquiv_naturality`, but `yonedaEquiv_naturality` is sometimes preferable because it
    can avoid the "motive is not type correct" error. -/
lemma yonedaULiftEquiv_naturality' {X Y : Cᵒᵖ} {F : Sheaf J (Type (max v v'))}
    (f : J.yonedaULift.obj (unop X) ⟶ F) (g : X ⟶ Y) :
      F.val.map g (J.yonedaULiftEquiv f) = J.yonedaULiftEquiv (J.yonedaULift.map g.unop ≫ f) :=
  J.yonedaULiftEquiv_naturality _ _


lemma yonedaULiftEquiv_comp {X : C} {F G : Sheaf J (Type (max v v'))} (α : J.yonedaULift.obj X ⟶ F)
    (β : F ⟶ G) : J.yonedaULiftEquiv (α ≫ β) = β.val.app _ (J.yonedaULiftEquiv α) :=
  rfl


lemma yonedaULiftEquiv_yonedaULift_map {X Y : C} (f : X ⟶ Y) :
    (yonedaULiftEquiv.{v'} J) (J.yonedaULift.map f) = ⟨f⟩ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (J.yonedaULiftEquiv ((CategoryTheory.GrothendieckTopology.yonedaULift.{v' …
  -/
  rw [yonedaULiftEquiv_apply]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (((CategoryTheory.GrothendieckTopology.yonedaULift.{v', v, u} J).map f).v …
  -/
  simp [yonedaULift]
  /-
    🎉 no goals
  -/


lemma yonedaULiftEquiv_symm_naturality_left {X X' : C} (f : X' ⟶ X) (F : Sheaf J (Type (max v v')))
    (x : F.val.obj ⟨X⟩) : J.yonedaULift.map f ≫ J.yonedaULiftEquiv.symm x = J.yonedaULiftEquiv.symm
      ((F.val.map f.op) x) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X X' : C
    f : Quiver.Hom X' X
    F : CategoryTheory.Sheaf J (Type (max v v'))
    x : F.val.obj { unop := X }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.GrothendieckTopology …
  -/
  apply J.yonedaULiftEquiv.injective
  /-
    case a
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X X' : C
    f : Quiver.Hom X' X
    F : CategoryTheory.Sheaf J (Type (max v v'))
    x : F.val.obj { unop := X }
    ⊢ Eq (J.yonedaULiftEquiv (CategoryTheory.CategoryStruct.comp ((CategoryTheory. …
  -/
  simp only [yonedaULiftEquiv_comp, Equiv.apply_symm_apply]
  /-
    case a
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X X' : C
    f : Quiver.Hom X' X
    F : CategoryTheory.Sheaf J (Type (max v v'))
    x : F.val.obj { unop := X }
    ⊢ Eq ((J.yonedaULiftEquiv.symm x).val.app { unop := X' } (J.yonedaULiftEquiv ( …
  -/
  rw [yonedaULiftEquiv_yonedaULift_map]
  /-
    case a
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X X' : C
    f : Quiver.Hom X' X
    F : CategoryTheory.Sheaf J (Type (max v v'))
    x : F.val.obj { unop := X }
    ⊢ Eq ((J.yonedaULiftEquiv.symm x).val.app { unop := X' } { down := f }) (F.val …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma yonedaULiftEquiv_symm_naturality_right (X : C) {F F' : Sheaf J (Type (max v v'))}
    (f : F ⟶ F') (x : F.val.obj ⟨X⟩) :
      J.yonedaULiftEquiv.symm x ≫ f = J.yonedaULiftEquiv.symm (f.val.app ⟨X⟩ x) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X : C
    F F' : CategoryTheory.Sheaf J (Type (max v v'))
    f : Quiver.Hom F F'
    x : F.val.obj { unop := X }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (J.yonedaULiftEquiv.symm x) f) (J.yon …
  -/
  apply J.yonedaULiftEquiv.injective
  /-
    case a
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X : C
    F F' : CategoryTheory.Sheaf J (Type (max v v'))
    f : Quiver.Hom F F'
    x : F.val.obj { unop := X }
    ⊢ Eq (J.yonedaULiftEquiv (CategoryTheory.CategoryStruct.comp (J.yonedaULiftEqu …
  -/
  simp [yonedaULiftEquiv_comp]
  /-
    🎉 no goals
  -/


/-- See also `map_yonedaEquiv'` for a more general version. -/
lemma map_yonedaULiftEquiv {X Y : C} {F : Sheaf J (Type (max v v'))}
    (f : J.yonedaULift.obj X ⟶ F) (g : Y ⟶ X) :
      F.val.map g.op (J.yonedaULiftEquiv f) = f.val.app (op Y) ⟨g⟩ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X Y : C
    F : CategoryTheory.Sheaf J (Type (max v v'))
    f : Quiver.Hom ((CategoryTheory.GrothendieckTopology.yonedaULift.{v', v, u} J) …
    g : Quiver.Hom Y X
    ⊢ Eq (F.val.map g.op (J.yonedaULiftEquiv f)) (f.val.app { unop := Y } { down : …
  -/
  rw [yonedaULiftEquiv_naturality, yonedaULiftEquiv_comp, yonedaULiftEquiv_yonedaULift_map]
  /-
    🎉 no goals
  -/


/-- Variant of `map_yonedaEquiv` with general `g`. This is technically strictly more general
    than `map_yonedaEquiv`, but `map_yonedaEquiv` is sometimes preferable because it
    can avoid the "motive is not type correct" error. -/
lemma map_yonedaULiftEquiv' {X Y : Cᵒᵖ} {F : Sheaf J (Type (max v v'))}
    (f : J.yonedaULift.obj (unop X) ⟶ F)
    (g : X ⟶ Y) : F.val.map g (J.yonedaULiftEquiv f) = f.val.app Y ⟨g.unop⟩ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X Y : Opposite C
    F : CategoryTheory.Sheaf J (Type (max v v'))
    f : Quiver.Hom ((CategoryTheory.GrothendieckTopology.yonedaULift.{v', v, u} J) …
    g : Quiver.Hom X Y
    ⊢ Eq (F.val.map g (J.yonedaULiftEquiv f)) (f.val.app Y { down := g.unop })
  -/
  rw [yonedaULiftEquiv_naturality', yonedaULiftEquiv_comp, yonedaULiftEquiv_yonedaULift_map]
  /-
    🎉 no goals
  -/


lemma yonedaULeftEquiv_symm_map {X Y : Cᵒᵖ} (f : X ⟶ Y) {F : Sheaf J (Type (max v v'))}
    (t : F.val.obj X) : J.yonedaULiftEquiv.symm (F.val.map f t) =
      J.yonedaULift.map f.unop ≫ J.yonedaULiftEquiv.symm t := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X Y : Opposite C
    f : Quiver.Hom X Y
    F : CategoryTheory.Sheaf J (Type (max v v'))
    t : F.val.obj X
    ⊢ Eq (J.yonedaULiftEquiv.symm (F.val.map f t)) (CategoryTheory.CategoryStruct. …
  -/
  obtain ⟨u, rfl⟩ := J.yonedaULiftEquiv.surjective t
  /-
    case intro
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    X Y : Opposite C
    f : Quiver.Hom X Y
    F : CategoryTheory.Sheaf J (Type (max v v'))
    u : Quiver.Hom ((CategoryTheory.GrothendieckTopology.yonedaULift.{v', v, u} J) …
    ⊢ Eq (J.yonedaULiftEquiv.symm (F.val.map f (J.yonedaULiftEquiv u))) (CategoryT …
  -/
  rw [yonedaULiftEquiv_naturality', Equiv.symm_apply_apply, Equiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


/-- Two morphisms of sheaves of types `P ⟶ Q` coincide if the precompositions
with morphisms `yoneda.obj X ⟶ P` agree. -/
lemma hom_ext_yonedaULift {P Q : Sheaf J (Type (max v v'))} {f g : P ⟶ Q}
    (h : ∀ (X : C) (p : J.yonedaULift.obj X ⟶ P), p ≫ f = p ≫ g) :
    f = g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    inst✝ : J.Subcanonical
    P Q : CategoryTheory.Sheaf J (Type (max v v'))
    f g : Quiver.Hom P Q
    h : ∀ (X : C) (p : Quiver.Hom ((CategoryTheory.GrothendieckTopology.yonedaULif …
    ⊢ Eq f g
  -/
  ext X x
  simpa only [yonedaULiftEquiv_comp, Equiv.apply_symm_apply]
    using congr_arg (J.yonedaULiftEquiv) (h _ (J.yonedaULiftEquiv.symm x))


