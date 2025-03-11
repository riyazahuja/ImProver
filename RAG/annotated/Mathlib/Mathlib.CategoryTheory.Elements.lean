/-- The type of objects for the category of elements of a functor `F : C ⥤ Type`
is a pair `(X : C, x : F.obj X)`.
-/
def Functor.Elements (F : C ⥤ Type w) :=
  Σc : C, F.obj c


/-- Constructor for the type `F.Elements` when `F` is a functor to types. -/
abbrev Functor.elementsMk (F : C ⥤ Type w) (X : C) (x : F.obj X) : F.Elements := ⟨X, x⟩


lemma Functor.Elements.ext {F : C ⥤ Type w} (x y : F.Elements) (h₁ : x.fst = y.fst)
    (h₂ : F.map (eqToHom h₁) x.snd = y.snd) : x = y := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C (Type w)
    x y : F.Elements
    h₁ : Eq x.fst y.fst
    h₂ : Eq (F.map (CategoryTheory.eqToHom h₁) x.snd) y.snd
    ⊢ Eq x y
  -/
  cases x
  /-
    case mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C (Type w)
    y : F.Elements
    fst✝ : C
    snd✝ : F.obj fst✝
    h₁ : Eq ⟨fst✝, snd✝⟩.fst y.fst
    h₂ : Eq (F.map (CategoryTheory.eqToHom h₁) ⟨fst✝, snd✝⟩.snd) y.snd
    ⊢ Eq ⟨fst✝, snd✝⟩ y
  -/
  cases y
  /-
    case mk.mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C (Type w)
    fst✝¹ : C
    snd✝¹ : F.obj fst✝¹
    fst✝ : C
    snd✝ : F.obj fst✝
    h₁ : Eq ⟨fst✝¹, snd✝¹⟩.fst ⟨fst✝, snd✝⟩.fst
    h₂ : Eq (F.map (CategoryTheory.eqToHom h₁) ⟨fst✝¹, snd✝¹⟩.snd) ⟨fst✝, snd✝⟩.snd
    ⊢ Eq ⟨fst✝¹, snd✝¹⟩ ⟨fst✝, snd✝⟩
  -/
  cases h₁
  /-
    case mk.mk.refl
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C (Type w)
    fst✝ : C
    snd✝¹ snd✝ : F.obj fst✝
    h₂ : Eq (F.map (CategoryTheory.eqToHom ⋯) ⟨fst✝, snd✝¹⟩.snd) ⟨fst✝, snd✝⟩.snd
    ⊢ Eq ⟨fst✝, snd✝¹⟩ ⟨fst✝, snd✝⟩
  -/
  simp only [eqToHom_refl, FunctorToTypes.map_id_apply] at h₂
  /-
    case mk.mk.refl
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor C (Type w)
    fst✝ : C
    snd✝¹ snd✝ : F.obj fst✝
    h₂ : Eq snd✝¹ snd✝
    ⊢ Eq ⟨fst✝, snd✝¹⟩ ⟨fst✝, snd✝⟩
  -/
  simp [h₂]
  /-
    🎉 no goals
  -/


/-- The category structure on `F.Elements`, for `F : C ⥤ Type`.
    A morphism `(X, x) ⟶ (Y, y)` is a morphism `f : X ⟶ Y` in `C`, so `F.map f` takes `x` to `y`.
 -/
instance categoryOfElements (F : C ⥤ Type w) : Category.{v} F.Elements where
  Hom p q := { f : p.1 ⟶ q.1 // (F.map f) p.2 = q.2 }
                     /-
                       C : Type u
                       inst✝ : CategoryTheory.Category.{v, u} C
                       F : CategoryTheory.Functor C (Type w)
                       p : F.Elements
                       ⊢ Eq (F.map (CategoryTheory.CategoryStruct.id p.fst) p.snd) p.snd
                     -/
  id p := ⟨𝟙 p.1, by aesop_cat⟩
                     /-
                       🎉 no goals
                     -/
                                         /-
                                           C : Type u
                                           inst✝ : CategoryTheory.Category.{v, u} C
                                           F : CategoryTheory.Functor C (Type w)
                                           X Y Z : F.Elements
                                           f : Quiver.Hom X Y
                                           g : Quiver.Hom Y Z
                                           ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp ↑f ↑g) X.snd) Z.snd
                                         -/
  comp {X Y Z} f g := ⟨f.val ≫ g.val, by simp [f.2, g.2]⟩
                                         /-
                                           🎉 no goals
                                         -/


/-- Natural transformations are mapped to functors between category of elements -/
@[simps]
def NatTrans.mapElements {F G : C ⥤ Type w} (φ : F ⟶ G) : F.Elements ⥤ G.Elements where
  obj := fun ⟨X, x⟩ ↦ ⟨_, φ.app X x⟩
                                   /-
                                     C : Type u
                                     inst✝ : CategoryTheory.Category.{v, u} C
                                     F G : CategoryTheory.Functor C (Type w)
                                     φ : Quiver.Hom F G
                                     p q : F.Elements
                                     x✝ : Quiver.Hom p q
                                     f : Quiver.Hom p.fst q.fst
                                     h : Eq (F.map f p.snd) q.snd
                                     ⊢ Eq (G.map f ((fun x => CategoryTheory.NatTrans.mapElements.match_1 (fun x => …
                                   -/
  map {p q} := fun ⟨f, h⟩ ↦ ⟨f, by have hb := congrFun (φ.naturality f) p.2; aesop_cat⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- The functor mapping functors `C ⥤ Type w` to their category of elements -/
@[simps]
def Functor.elementsFunctor : (C ⥤ Type w) ⥤ Cat where
  obj F := Cat.of F.Elements
  map n := NatTrans.mapElements n


/-- Constructor for morphisms in the category of elements of a functor to types. -/
@[simps]
def homMk {F : C ⥤ Type w} (x y : F.Elements) (f : x.1 ⟶ y.1) (hf : F.map f x.snd = y.snd) :
    x ⟶ y :=
  ⟨f, hf⟩


@[ext]
theorem ext (F : C ⥤ Type w) {x y : F.Elements} (f g : x ⟶ y) (w : f.val = g.val) : f = g :=
  Subtype.ext_val w


@[simp]
theorem comp_val {F : C ⥤ Type w} {p q r : F.Elements} {f : p ⟶ q} {g : q ⟶ r} :
    (f ≫ g).val = f.val ≫ g.val :=
  rfl


@[simp]
theorem id_val {F : C ⥤ Type w} {p : F.Elements} : (𝟙 p : p ⟶ p).val = 𝟙 p.1 :=
  rfl


@[simp]
theorem map_snd {F : C ⥤ Type w} {p q : F.Elements} (f : p ⟶ q) : (F.map f.val) p.2 = q.2 :=
  f.property


/-- Constructor for isomorphisms in the category of elements of a functor to types. -/
@[simps]
def isoMk {F : C ⥤ Type w} (x y : F.Elements) (e : x.1 ≅ y.1) (he : F.map e.hom x.snd = y.snd) :
    x ≅ y where
  hom := homMk x y e.hom he
                             /-
                               C : Type u
                               inst✝ : CategoryTheory.Category.{v, u} C
                               F : CategoryTheory.Functor C (Type w)
                               x y : F.Elements
                               e : CategoryTheory.Iso x.fst y.fst
                               he : Eq (F.map e.hom x.snd) y.snd
                               ⊢ Eq (F.map e.inv y.snd) x.snd
                             -/
  inv := homMk y x e.inv (by rw [← he, FunctorToTypes.map_inv_map_hom_apply])
                             /-
                               🎉 no goals
                             -/


instance groupoidOfElements {G : Type u} [Groupoid.{v} G] (F : G ⥤ Type w) :
    Groupoid F.Elements where
  inv {p q} f :=
    ⟨Groupoid.inv f.val,
      calc
                                                                                            /-
                                                                                              C : Type u
                                                                                              inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                              G : Type u
                                                                                              inst✝ : CategoryTheory.Groupoid G
                                                                                              F : CategoryTheory.Functor G (Type w)
                                                                                              p q : F.Elements
                                                                                              f : Quiver.Hom p q
                                                                                              ⊢ Eq (F.map (CategoryTheory.Groupoid.inv ↑f) q.snd) (F.map (CategoryTheory.Gro …
                                                                                            -/
        F.map (Groupoid.inv f.val) q.2 = F.map (Groupoid.inv f.val) (F.map f.val p.2) := by rw [f.2]
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/
        _ = (F.map f.val ≫ F.map (Groupoid.inv f.val)) p.2 := rfl
        _ = p.2 := by
          /-
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            G : Type u
            inst✝ : CategoryTheory.Groupoid G
            F : CategoryTheory.Functor G (Type w)
            p q : F.Elements
            f : Quiver.Hom p q
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ↑f) (F.map (CategoryTheory.Gro …
          -/
          rw [← F.map_comp]
          /-
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            G : Type u
            inst✝ : CategoryTheory.Groupoid G
            F : CategoryTheory.Functor G (Type w)
            p q : F.Elements
            f : Quiver.Hom p q
            ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (↑f) (CategoryTheory.Groupoid. …
          -/
          simp
          /-
            🎉 no goals
          -/
        ⟩
  inv_comp _ := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      G : Type u
      inst✝ : CategoryTheory.Groupoid G
      F : CategoryTheory.Functor G (Type w)
      X✝ Y✝ : F.Elements
      x✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun {p q} f => ⟨CategoryTheory.Grou …
    -/
    ext
    /-
      case w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      G : Type u
      inst✝ : CategoryTheory.Groupoid G
      F : CategoryTheory.Functor G (Type w)
      X✝ Y✝ : F.Elements
      x✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq ↑(CategoryTheory.CategoryStruct.comp ((fun {p q} f => ⟨CategoryTheory.Gro …
    -/
    simp
    /-
      🎉 no goals
    -/
  comp_inv _ := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      G : Type u
      inst✝ : CategoryTheory.Groupoid G
      F : CategoryTheory.Functor G (Type w)
      X✝ Y✝ : F.Elements
      x✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp x✝ ((fun {p q} f => ⟨CategoryTheory.G …
    -/
    ext
    /-
      case w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      G : Type u
      inst✝ : CategoryTheory.Groupoid G
      F : CategoryTheory.Functor G (Type w)
      X✝ Y✝ : F.Elements
      x✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq ↑(CategoryTheory.CategoryStruct.comp x✝ ((fun {p q} f => ⟨CategoryTheory. …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The functor out of the category of elements which forgets the element. -/
@[simps]
def π : F.Elements ⥤ C where
  obj X := X.1
  map f := f.val


instance : (π F).Faithful where


instance : (π F).ReflectsIsomorphisms where
  reflects {X Y} f h := ⟨⟨⟨inv ((π F).map f),
       /-
         C : Type u
         inst✝ : CategoryTheory.Category.{v, u} C
         F : CategoryTheory.Functor C (Type w)
         X Y : F.Elements
         f : Quiver.Hom X Y
         h : CategoryTheory.IsIso ((CategoryTheory.CategoryOfElements.π F).map f)
         ⊢ Eq (F.map (CategoryTheory.inv ((CategoryTheory.CategoryOfElements.π F).map f …
       -/
                                                          /-
                                                            🎉 no goals
                                                          -/
    by rw [← map_snd f, ← FunctorToTypes.map_comp_apply]; simp⟩, by aesop_cat⟩⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- A natural transformation between functors induces a functor between the categories of elements.
-/
@[simps]
def map {F₁ F₂ : C ⥤ Type w} (α : F₁ ⟶ F₂) : F₁.Elements ⥤ F₂.Elements where
  obj t := ⟨t.1, α.app t.1 t.2⟩
                            /-
                              C : Type u
                              inst✝ : CategoryTheory.Category.{v, u} C
                              F F₁ F₂ : CategoryTheory.Functor C (Type w)
                              α : Quiver.Hom F₁ F₂
                              t₁ t₂ : F₁.Elements
                              k : Quiver.Hom t₁ t₂
                              ⊢ Eq (F₂.map (↑k) ((fun t => ⟨t.fst, α.app t.fst t.snd⟩) t₁).snd) ((fun t => ⟨ …
                            -/
  map {t₁ t₂} k := ⟨k.1, by simpa [map_snd] using (FunctorToTypes.naturality _ _ α k.1 t₁.2).symm⟩
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem map_π {F₁ F₂ : C ⥤ Type w} (α : F₁ ⟶ F₂) : map α ⋙ π F₂ = π F₁ :=
  rfl


/-- The forward direction of the equivalence `F.Elements ≅ (*, F)`. -/
def toStructuredArrow : F.Elements ⥤ StructuredArrow PUnit F where
  obj X := StructuredArrow.mk fun _ => X.2
                                                 /-
                                                   C : Type u
                                                   inst✝ : CategoryTheory.Category.{v, u} C
                                                   F : CategoryTheory.Functor C (Type w)
                                                   X Y : F.Elements
                                                   f : Quiver.Hom X Y
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X => CategoryTheory.StructuredA …
                                                 -/
  map {X Y} f := StructuredArrow.homMk f.val (by funext; simp [f.2])
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
theorem toStructuredArrow_obj (X) :
    (toStructuredArrow F).obj X =
      { left := ⟨⟨⟩⟩
        right := X.1
        hom := fun _ => X.2 } :=
  rfl


@[simp]
theorem to_comma_map_right {X Y} (f : X ⟶ Y) : ((toStructuredArrow F).map f).right = f.val :=
  rfl


/-- The reverse direction of the equivalence `F.Elements ≅ (*, F)`. -/
def fromStructuredArrow : StructuredArrow PUnit F ⥤ F.Elements where
  obj X := ⟨X.right, X.hom PUnit.unit⟩
  map f := ⟨f.right, congr_fun f.w.symm PUnit.unit⟩


@[simp]
theorem fromStructuredArrow_obj (X) : (fromStructuredArrow F).obj X = ⟨X.right, X.hom PUnit.unit⟩ :=
  rfl


@[simp]
theorem fromStructuredArrow_map {X Y} (f : X ⟶ Y) :
    (fromStructuredArrow F).map f = ⟨f.right, congr_fun f.w.symm PUnit.unit⟩ :=
  rfl


/-- The equivalence between the category of elements `F.Elements`
    and the comma category `(*, F)`. -/
@[simps]
def structuredArrowEquivalence : F.Elements ≌ StructuredArrow PUnit F where
  functor := toStructuredArrow F
  inverse := fromStructuredArrow F
  unitIso := Iso.refl _
  counitIso := Iso.refl _


/-- The forward direction of the equivalence `F.Elementsᵒᵖ ≅ (yoneda, F)`,
given by `CategoryTheory.yonedaEquiv`.
-/
@[simps]
def toCostructuredArrow (F : Cᵒᵖ ⥤ Type v) : F.Elementsᵒᵖ ⥤ CostructuredArrow yoneda F where
  obj X := CostructuredArrow.mk (yonedaEquiv.symm (unop X).2)
  map f := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F✝ : CategoryTheory.Functor C (Type w)
      F : CategoryTheory.Functor (Opposite C) (Type v)
      X✝ Y✝ : Opposite F.Elements
      f : Quiver.Hom X✝ Y✝
      ⊢ Quiver.Hom ((fun X => CategoryTheory.CostructuredArrow.mk (CategoryTheory.yo …
    -/
    fapply CostructuredArrow.homMk
      /-
        case g
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F✝ : CategoryTheory.Functor C (Type w)
        F : CategoryTheory.Functor (Opposite C) (Type v)
        X✝ Y✝ : Opposite F.Elements
        f : Quiver.Hom X✝ Y✝
        ⊢ Quiver.Hom ((fun X => CategoryTheory.CostructuredArrow.mk (CategoryTheory.yo …
      -/
    · exact f.unop.val.unop
      /-
        🎉 no goals
      -/
      /-
        case w
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F✝ : CategoryTheory.Functor C (Type w)
        F : CategoryTheory.Functor (Opposite C) (Type v)
        X✝ Y✝ : Opposite F.Elements
        f : Quiver.Hom X✝ Y✝
        ⊢ autoParam (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map …
      -/
    · ext Z y
      /-
        case w.w.h.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F✝ : CategoryTheory.Functor C (Type w)
        F : CategoryTheory.Functor (Opposite C) (Type v)
        X✝ Y✝ : Opposite F.Elements
        f : Quiver.Hom X✝ Y✝
        Z : Opposite C
        y : (CategoryTheory.yoneda.obj ((fun X => CategoryTheory.CostructuredArrow.mk  …
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map (↑f.unop) …
      -/
      dsimp [yonedaEquiv]
      /-
        case w.w.h.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F✝ : CategoryTheory.Functor C (Type w)
        F : CategoryTheory.Functor (Opposite C) (Type v)
        X✝ Y✝ : Opposite F.Elements
        f : Quiver.Hom X✝ Y✝
        Z : Opposite C
        y : (CategoryTheory.yoneda.obj ((fun X => CategoryTheory.CostructuredArrow.mk  …
        ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (↑f.unop) (Quiver.Hom.op y)) ( …
      -/
      simp only [FunctorToTypes.map_comp_apply, ← f.unop.2]
      /-
        🎉 no goals
      -/


/-- The reverse direction of the equivalence `F.Elementsᵒᵖ ≅ (yoneda, F)`,
given by `CategoryTheory.yonedaEquiv`.
-/
@[simps]
def fromCostructuredArrow (F : Cᵒᵖ ⥤ Type v) : (CostructuredArrow yoneda F)ᵒᵖ ⥤ F.Elements where
  obj X := ⟨op (unop X).1, yonedaEquiv.1 (unop X).3⟩
  map {X Y} f :=
    ⟨f.unop.1.op, by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F✝ : CategoryTheory.Functor C (Type w)
        F : CategoryTheory.Functor (Opposite C) (Type v)
        X Y : Opposite (CategoryTheory.CostructuredArrow CategoryTheory.yoneda F)
        f : Quiver.Hom X Y
        ⊢ Eq (F.map f.unop.left.op ((fun X => ⟨{ unop := (Opposite.unop X).left }, Cat …
      -/
      convert (congr_fun ((unop X).hom.naturality f.unop.left.op) (𝟙 _)).symm
      simp only [Equiv.toFun_as_coe, Quiver.Hom.unop_op, yonedaEquiv_apply, types_comp_apply,
        Category.comp_id, yoneda_obj_map]
      have : yoneda.map f.unop.left ≫ (unop X).hom = (unop Y).hom := by
        convert f.unop.3
      /-
        case h.e'_3.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F✝ : CategoryTheory.Functor C (Type w)
        F : CategoryTheory.Functor (Opposite C) (Type v)
        X Y : Opposite (CategoryTheory.CostructuredArrow CategoryTheory.yoneda F)
        f : Quiver.Hom X Y
        e_1✝ : Eq (F.obj ((fun X => ⟨{ unop := (Opposite.unop X).left }, CategoryTheor …
        this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map f.uno …
        ⊢ Eq ((Opposite.unop Y).hom.app { unop := (Opposite.unop Y).left } (CategoryTh …
      -/
      rw [← this]
      /-
        case h.e'_3.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F✝ : CategoryTheory.Functor C (Type w)
        F : CategoryTheory.Functor (Opposite C) (Type v)
        X Y : Opposite (CategoryTheory.CostructuredArrow CategoryTheory.yoneda F)
        f : Quiver.Hom X Y
        e_1✝ : Eq (F.obj ((fun X => ⟨{ unop := (Opposite.unop X).left }, CategoryTheor …
        this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map f.uno …
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map f.unop.le …
      -/
      simp only [yoneda_map_app, FunctorToTypes.comp]
      /-
        case h.e'_3.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        F✝ : CategoryTheory.Functor C (Type w)
        F : CategoryTheory.Functor (Opposite C) (Type v)
        X Y : Opposite (CategoryTheory.CostructuredArrow CategoryTheory.yoneda F)
        f : Quiver.Hom X Y
        e_1✝ : Eq (F.obj ((fun X => ⟨{ unop := (Opposite.unop X).left }, CategoryTheor …
        this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map f.uno …
        ⊢ Eq ((Opposite.unop X).hom.app { unop := (Opposite.unop Y).left } (CategoryTh …
      -/
      rw [Category.id_comp]⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem fromCostructuredArrow_obj_mk (F : Cᵒᵖ ⥤ Type v) {X : C} (f : yoneda.obj X ⟶ F) :
    (fromCostructuredArrow F).obj (op (CostructuredArrow.mk f)) = ⟨op X, yonedaEquiv.1 f⟩ :=
  rfl


/-- The equivalence `F.Elementsᵒᵖ ≅ (yoneda, F)` given by yoneda lemma. -/
@[simps]
def costructuredArrowYonedaEquivalence (F : Cᵒᵖ ⥤ Type v) :
    F.Elementsᵒᵖ ≌ CostructuredArrow yoneda F where
  functor := toCostructuredArrow F
  inverse := (fromCostructuredArrow F).rightOp
  unitIso :=
    NatIso.ofComponents
                                                                     /-
                                                                       C : Type u
                                                                       inst✝ : CategoryTheory.Category.{v, u} C
                                                                       F✝ : CategoryTheory.Functor C (Type w)
                                                                       F : CategoryTheory.Functor (Opposite C) (Type v)
                                                                       X : Opposite F.Elements
                                                                       ⊢ Eq (F.map (CategoryTheory.Iso.refl ((CategoryTheory.CategoryOfElements.fromC …
                                                                     -/
      (fun X ↦ Iso.op (CategoryOfElements.isoMk _ _ (Iso.refl _) (by simp))) (by
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
        /-
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          F✝ : CategoryTheory.Functor C (Type w)
          F : CategoryTheory.Functor (Opposite C) (Type v)
          ⊢ ∀ {X Y : Opposite F.Elements} (f : Quiver.Hom X Y), Eq (CategoryTheory.Categ …
        -/
        rintro ⟨x⟩ ⟨y⟩ ⟨f : y ⟶ x⟩
        /-
          case op.op.op
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          F✝ : CategoryTheory.Functor C (Type w)
          F : CategoryTheory.Functor (Opposite C) (Type v)
          x y : F.Elements
          f : Quiver.Hom y x
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Opposite …
        -/
        exact Quiver.Hom.unop_inj (by ext; simp))
        /-
          🎉 no goals
        -/
                                            /-
                                              C : Type u
                                              inst✝ : CategoryTheory.Category.{v, u} C
                                              F✝ : CategoryTheory.Functor C (Type w)
                                              F : CategoryTheory.Functor (Opposite C) (Type v)
                                              X : CategoryTheory.CostructuredArrow CategoryTheory.yoneda F
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda.map (CategoryT …
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  counitIso := NatIso.ofComponents (fun X ↦ CostructuredArrow.isoMk (Iso.refl _))
               /-
                 🎉 no goals
               -/


/-- The equivalence `(-.Elements)ᵒᵖ ≅ (yoneda, -)` of is actually a natural isomorphism of functors.
-/
theorem costructuredArrow_yoneda_equivalence_naturality {F₁ F₂ : Cᵒᵖ ⥤ Type v} (α : F₁ ⟶ F₂) :
    (map α).op ⋙ toCostructuredArrow F₂ = toCostructuredArrow F₁ ⋙ CostructuredArrow.map α := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F₁ F₂ : CategoryTheory.Functor (Opposite C) (Type v)
    α : Quiver.Hom F₁ F₂
    ⊢ Eq ((CategoryTheory.CategoryOfElements.map α).op.comp (CategoryTheory.Catego …
  -/
  fapply Functor.ext
    /-
      case h_obj
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) (Type v)
      α : Quiver.Hom F₁ F₂
      ⊢ ∀ (X : Opposite F₁.Elements), Eq (((CategoryTheory.CategoryOfElements.map α) …
    -/
  · intro X
    simp only [CostructuredArrow.map_mk, toCostructuredArrow_obj, Functor.op_obj,
      Functor.comp_obj]
    /-
      case h_obj
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) (Type v)
      α : Quiver.Hom F₁ F₂
      X : Opposite F₁.Elements
      ⊢ Eq (CategoryTheory.CostructuredArrow.mk (CategoryTheory.yonedaEquiv.symm ((C …
    -/
    congr
    /-
      case h_obj.e_f
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) (Type v)
      α : Quiver.Hom F₁ F₂
      X : Opposite F₁.Elements
      ⊢ Eq (CategoryTheory.yonedaEquiv.symm ((CategoryTheory.CategoryOfElements.map  …
    -/
    ext _ f
    /-
      case h_obj.e_f.w.h.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) (Type v)
      α : Quiver.Hom F₁ F₂
      X : Opposite F₁.Elements
      x✝ : Opposite C
      f : (CategoryTheory.yoneda.obj (Opposite.unop ((CategoryTheory.CategoryOfEleme …
      ⊢ Eq ((CategoryTheory.yonedaEquiv.symm ((CategoryTheory.CategoryOfElements.map …
    -/
    simpa using congr_fun (α.naturality f.op).symm (unop X).snd
    /-
      🎉 no goals
    -/
    /-
      case h_map
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F₁ F₂ : CategoryTheory.Functor (Opposite C) (Type v)
      α : Quiver.Hom F₁ F₂
      ⊢ autoParam (∀ (X Y : Opposite F₁.Elements) (f : Quiver.Hom X Y), Eq (((Catego …
    -/
  · simp
    /-
      🎉 no goals
    -/


/-- The equivalence `F.elementsᵒᵖ ≌ (yoneda, F)` is compatible with the forgetful functors. -/
@[simps!]
def costructuredArrowYonedaEquivalenceFunctorProj (F : Cᵒᵖ ⥤ Type v) :
    (costructuredArrowYonedaEquivalence F).functor ⋙ CostructuredArrow.proj _ _ ≅ (π F).leftOp :=
  Iso.refl _


/-- The equivalence `F.elementsᵒᵖ ≌ (yoneda, F)` is compatible with the forgetful functors. -/
@[simps!]
def costructuredArrowYonedaEquivalenceInverseπ (F : Cᵒᵖ ⥤ Type v) :
    (costructuredArrowYonedaEquivalence F).inverse ⋙ (π F).leftOp ≅ CostructuredArrow.proj _ _ :=
  Iso.refl _


/--
The initial object in the category of elements for a representable functor. In `isInitial` it is
shown that this is initial.
-/
def Elements.initial (A : C) : (yoneda.obj A).Elements :=
  ⟨Opposite.op A, 𝟙 _⟩


/-- Show that `Elements.initial A` is initial in the category of elements for the `yoneda` functor.
-/
def Elements.isInitial (A : C) : Limits.IsInitial (Elements.initial A) where
  desc s := ⟨s.pt.2.op, Category.comp_id _⟩
  uniq s m _ := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      A : C
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Functor.empty (CategoryTheory …
      m : Quiver.Hom (CategoryTheory.Limits.asEmptyCocone (CategoryTheory.Functor.El …
      x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategorySt …
      ⊢ Eq m ((fun s => ⟨Quiver.Hom.op s.pt.snd, ⋯⟩) s)
    -/
    simp_rw [← m.2]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      A : C
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Functor.empty (CategoryTheory …
      m : Quiver.Hom (CategoryTheory.Limits.asEmptyCocone (CategoryTheory.Functor.El …
      x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategorySt …
      ⊢ Eq m ⟨Quiver.Hom.op ((CategoryTheory.yoneda.obj A).map (↑m) (CategoryTheory. …
    -/
            /-
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              A : C
              ⊢ ∀ (s : CategoryTheory.Limits.Cocone (CategoryTheory.Functor.empty (CategoryT …
            -/
    dsimp [Elements.initial]
            /-
              🎉 no goals
            -/
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      A : C
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Functor.empty (CategoryTheory …
      m : Quiver.Hom (CategoryTheory.Limits.asEmptyCocone (CategoryTheory.Functor.El …
      x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategorySt …
      ⊢ Eq m ⟨CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id { …
    -/
    simp
    /-
      🎉 no goals
    -/
  fac := by rintro s ⟨⟨⟩⟩


