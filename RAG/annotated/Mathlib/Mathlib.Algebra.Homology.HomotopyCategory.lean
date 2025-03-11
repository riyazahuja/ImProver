/-- The congruence on `HomologicalComplex V c` given by the existence of a homotopy.
-/
def homotopic : HomRel (HomologicalComplex V c) := fun _ _ f g => Nonempty (Homotopy f g)


instance homotopy_congruence : Congruence (homotopic V c) where
  equivalence :=
    { refl := fun C => ⟨Homotopy.refl C⟩
      symm := fun ⟨w⟩ => ⟨w.symm⟩
      trans := fun ⟨w₁⟩ ⟨w₂⟩ => ⟨w₁.trans w₂⟩ }
  compLeft := fun _ _ _ ⟨i⟩ => ⟨i.compLeft _⟩
  compRight := fun _ ⟨i⟩ => ⟨i.compRight _⟩


/-- `HomotopyCategory V c` is the category of chain complexes of shape `c` in `V`,
with chain maps identified when they are homotopic. -/
def HomotopyCategory :=
  CategoryTheory.Quotient (homotopic V c)


instance : Category (HomotopyCategory V c) := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    ι : Type u_2
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    ⊢ CategoryTheory.Category.{?u.1736, max (max u_2 u) v} (HomotopyCategory V c)
  -/
  dsimp only [HomotopyCategory]
  /-
    R : Type u_1
    inst✝² : Semiring R
    ι : Type u_2
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    ⊢ CategoryTheory.Category.{?u.1736, max (max u_2 u) v} (CategoryTheory.Quotien …
  -/
  infer_instance
  /-
    🎉 no goals
  -/

-- TODO the homotopy_category is preadditive

instance : Preadditive (HomotopyCategory V c) := Quotient.preadditive _ (by
  /-
    R : Type u_1
    inst✝² : Semiring R
    ι : Type u_2
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    ⊢ ∀ ⦃X Y : HomologicalComplex V c⦄ (f₁ f₂ g₁ g₂ : Quiver.Hom X Y), homotopic V …
  -/
  rintro _ _ _ _ _ _ ⟨h⟩ ⟨h'⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝² : Semiring R
    ι : Type u_2
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    X✝ Y✝ : HomologicalComplex V c
    f₁✝ f₂✝ g₁✝ g₂✝ : Quiver.Hom X✝ Y✝
    h : Homotopy f₁✝ f₂✝
    h' : Homotopy g₁✝ g₂✝
    ⊢ homotopic V c (HAdd.hAdd f₁✝ g₁✝) (HAdd.hAdd f₂✝ g₂✝)
  -/
  exact ⟨Homotopy.add h h'⟩)
  /-
    🎉 no goals
  -/


/-- The quotient functor from complexes to the homotopy category. -/
def quotient : HomologicalComplex V c ⥤ HomotopyCategory V c :=
  CategoryTheory.Quotient.functor _


instance : (quotient V c).Full := Quotient.full_functor _


instance : (quotient V c).EssSurj := Quotient.essSurj_functor _


instance : (quotient V c).Additive where


instance : Preadditive (CategoryTheory.Quotient (homotopic V c)) :=
  (inferInstance : Preadditive (HomotopyCategory V c))


instance : Functor.Additive (Quotient.functor (homotopic V c)) where


instance [Linear R V] : Linear R (HomotopyCategory V c) :=
  Quotient.linear R (homotopic V c) (fun _ _ _ _ _ h => ⟨h.some.smul _⟩)


instance [Linear R V] : Functor.Linear R (HomotopyCategory.quotient V c) :=
  Quotient.linear_functor _ _ _


instance [HasZeroObject V] : Inhabited (HomotopyCategory V c) :=
  ⟨(quotient V c).obj 0⟩


instance [HasZeroObject V] : HasZeroObject (HomotopyCategory V c) :=
  ⟨(quotient V c).obj 0, by
    /-
      R : Type u_1
      inst✝³ : Semiring R
      ι : Type u_2
      V : Type u
      inst✝² : CategoryTheory.Category.{v, u} V
      inst✝¹ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroObject V
      ⊢ CategoryTheory.Limits.IsZero ((HomotopyCategory.quotient V c).obj 0)
    -/
    rw [IsZero.iff_id_eq_zero, ← (quotient V c).map_id, id_zero, Functor.map_zero]⟩
    /-
      🎉 no goals
    -/


instance {D : Type*} [Category D] : ((whiskeringLeft _ _ D).obj (quotient V c)).Full :=
  Quotient.full_whiskeringLeft_functor _ _


instance {D : Type*} [Category D] : ((whiskeringLeft _ _ D).obj (quotient V c)).Faithful :=
  Quotient.faithful_whiskeringLeft_functor _ _


theorem quotient_obj_as (C : HomologicalComplex V c) : ((quotient V c).obj C).as = C :=
  rfl


@[simp]
theorem quotient_map_out {C D : HomotopyCategory V c} (f : C ⟶ D) : (quotient V c).map f.out = f :=
  Quot.out_eq _

-- Porting note: added to ease the port

theorem quot_mk_eq_quotient_map {C D : HomologicalComplex V c} (f : C ⟶ D) :
    Quot.mk _ f = (quotient V c).map f := rfl


theorem eq_of_homotopy {C D : HomologicalComplex V c} (f g : C ⟶ D) (h : Homotopy f g) :
    (quotient V c).map f = (quotient V c).map g :=
  CategoryTheory.Quotient.sound _ ⟨h⟩


/-- If two chain maps become equal in the homotopy category, then they are homotopic. -/
def homotopyOfEq {C D : HomologicalComplex V c} (f g : C ⟶ D)
    (w : (quotient V c).map f = (quotient V c).map g) : Homotopy f g :=
  ((Quotient.functor_map_eq_iff _ _ _).mp w).some


/-- An arbitrarily chosen representation of the image of a chain map in the homotopy category
is homotopic to the original chain map.
-/
def homotopyOutMap {C D : HomologicalComplex V c} (f : C ⟶ D) :
    Homotopy ((quotient V c).map f).out f := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    ι : Type u_2
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    f : Quiver.Hom C D
    ⊢ Homotopy (Quot.out ((HomotopyCategory.quotient V c).map f)) f
  -/
  apply homotopyOfEq
  /-
    case w
    R : Type u_1
    inst✝² : Semiring R
    ι : Type u_2
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    f : Quiver.Hom C D
    ⊢ Eq ((HomotopyCategory.quotient V c).map (Quot.out ((HomotopyCategory.quotien …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp 1100]
theorem quotient_map_out_comp_out {C D E : HomotopyCategory V c} (f : C ⟶ D) (g : D ⟶ E) :
                                                               /-
                                                                 ι : Type u_2
                                                                 V : Type u
                                                                 inst✝¹ : CategoryTheory.Category.{v, u} V
                                                                 inst✝ : CategoryTheory.Preadditive V
                                                                 c : ComplexShape ι
                                                                 C D E : HomotopyCategory V c
                                                                 f : Quiver.Hom C D
                                                                 g : Quiver.Hom D E
                                                                 ⊢ Eq ((HomotopyCategory.quotient V c).map (CategoryTheory.CategoryStruct.comp  …
                                                               -/
    (quotient V c).map (Quot.out f ≫ Quot.out g) = f ≫ g := by simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- Homotopy equivalent complexes become isomorphic in the homotopy category. -/
@[simps]
def isoOfHomotopyEquiv {C D : HomologicalComplex V c} (f : HomotopyEquiv C D) :
    (quotient V c).obj C ≅ (quotient V c).obj D where
  hom := (quotient V c).map f.hom
  inv := (quotient V c).map f.inv
  hom_inv_id := by
    /-
      R : Type u_1
      inst✝² : Semiring R
      ι : Type u_2
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D : HomologicalComplex V c
      f : HomotopyEquiv C D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.quotient V c).map  …
    -/
    rw [← (quotient V c).map_comp, ← (quotient V c).map_id]
    /-
      R : Type u_1
      inst✝² : Semiring R
      ι : Type u_2
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D : HomologicalComplex V c
      f : HomotopyEquiv C D
      ⊢ Eq ((HomotopyCategory.quotient V c).map (CategoryTheory.CategoryStruct.comp  …
    -/
    exact eq_of_homotopy _ _ f.homotopyHomInvId
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      R : Type u_1
      inst✝² : Semiring R
      ι : Type u_2
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D : HomologicalComplex V c
      f : HomotopyEquiv C D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.quotient V c).map  …
    -/
    rw [← (quotient V c).map_comp, ← (quotient V c).map_id]
    /-
      R : Type u_1
      inst✝² : Semiring R
      ι : Type u_2
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D : HomologicalComplex V c
      f : HomotopyEquiv C D
      ⊢ Eq ((HomotopyCategory.quotient V c).map (CategoryTheory.CategoryStruct.comp  …
    -/
    exact eq_of_homotopy _ _ f.homotopyInvHomId
    /-
      🎉 no goals
    -/


/-- If two complexes become isomorphic in the homotopy category,
  then they were homotopy equivalent. -/
def homotopyEquivOfIso {C D : HomologicalComplex V c}
    (i : (quotient V c).obj C ≅ (quotient V c).obj D) : HomotopyEquiv C D where
  hom := Quot.out i.hom
  inv := Quot.out i.inv
  homotopyHomInvId :=
    homotopyOfEq _ _
          /-
            R : Type u_1
            inst✝² : Semiring R
            ι : Type u_2
            V : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} V
            inst✝ : CategoryTheory.Preadditive V
            c : ComplexShape ι
            C D : HomologicalComplex V c
            i : CategoryTheory.Iso ((HomotopyCategory.quotient V c).obj C) ((HomotopyCateg …
            ⊢ Eq ((HomotopyCategory.quotient V c).map (CategoryTheory.CategoryStruct.comp  …
          -/
      (by rw [quotient_map_out_comp_out, i.hom_inv_id, (quotient V c).map_id])
          /-
            🎉 no goals
          -/
  homotopyInvHomId :=
    homotopyOfEq _ _
          /-
            R : Type u_1
            inst✝² : Semiring R
            ι : Type u_2
            V : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} V
            inst✝ : CategoryTheory.Preadditive V
            c : ComplexShape ι
            C D : HomologicalComplex V c
            i : CategoryTheory.Iso ((HomotopyCategory.quotient V c).obj C) ((HomotopyCateg …
            ⊢ Eq ((HomotopyCategory.quotient V c).map (CategoryTheory.CategoryStruct.comp  …
          -/
      (by rw [quotient_map_out_comp_out, i.inv_hom_id, (quotient V c).map_id])
          /-
            🎉 no goals
          -/


variable (V c) in
lemma quotient_inverts_homotopyEquivalences :
    (HomologicalComplex.homotopyEquivalences V c).IsInvertedBy (quotient V c) := by
  /-
    ι : Type u_2
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    ⊢ (HomologicalComplex.homotopyEquivalences V c).IsInvertedBy (HomotopyCategory …
  -/
  rintro K L _ ⟨e, rfl⟩
  /-
    case intro
    ι : Type u_2
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    K L : HomologicalComplex V c
    e : HomotopyEquiv K L
    ⊢ CategoryTheory.IsIso ((HomotopyCategory.quotient V c).map e.hom)
  -/
  change IsIso (isoOfHomotopyEquiv e).hom
  /-
    case intro
    ι : Type u_2
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    K L : HomologicalComplex V c
    e : HomotopyEquiv K L
    ⊢ CategoryTheory.IsIso (HomotopyCategory.isoOfHomotopyEquiv e).hom
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma isZero_quotient_obj_iff (C : HomologicalComplex V c) :
    IsZero ((quotient _ _).obj C) ↔ Nonempty (Homotopy (𝟙 C) 0) := by
  /-
    ι : Type u_2
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C : HomologicalComplex V c
    ⊢ Iff (CategoryTheory.Limits.IsZero ((HomotopyCategory.quotient V c).obj C)) ( …
  -/
  rw [IsZero.iff_id_eq_zero]
  /-
    ι : Type u_2
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C : HomologicalComplex V c
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.id ((HomotopyCategory.quotient V c).o …
  -/
  constructor
    /-
      case mp
      ι : Type u_2
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C : HomologicalComplex V c
      ⊢ Eq (CategoryTheory.CategoryStruct.id ((HomotopyCategory.quotient V c).obj C) …
    -/
  · intro h
    /-
      case mp
      ι : Type u_2
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C : HomologicalComplex V c
      h : Eq (CategoryTheory.CategoryStruct.id ((HomotopyCategory.quotient V c).obj  …
      ⊢ Nonempty (Homotopy (CategoryTheory.CategoryStruct.id C) 0)
    -/
    exact ⟨(homotopyOfEq _ _ (by simp [h]))⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_2
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C : HomologicalComplex V c
      ⊢ Nonempty (Homotopy (CategoryTheory.CategoryStruct.id C) 0) → Eq (CategoryThe …
    -/
  · rintro ⟨h⟩
    /-
      case mpr.intro
      ι : Type u_2
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C : HomologicalComplex V c
      h : Homotopy (CategoryTheory.CategoryStruct.id C) 0
      ⊢ Eq (CategoryTheory.CategoryStruct.id ((HomotopyCategory.quotient V c).obj C) …
    -/
    simpa using (eq_of_homotopy _ _ h)
    /-
      🎉 no goals
    -/


open Classical in
/-- The `i`-th homology, as a functor from the homotopy category. -/
noncomputable def homologyFunctor (i : ι) : HomotopyCategory V c ⥤ V :=
  CategoryTheory.Quotient.lift _ (HomologicalComplex.homologyFunctor V c i) (by
    /-
      R : Type u_1
      inst✝³ : Semiring R
      ι : Type u_2
      V : Type u
      inst✝² : CategoryTheory.Category.{v, u} V
      inst✝¹ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      inst✝ : CategoryTheory.CategoryWithHomology V
      i : ι
      ⊢ ∀ (x y : HomologicalComplex V c) (f₁ f₂ : Quiver.Hom x y), homotopic V c f₁  …
    -/
    rintro K L f g ⟨h⟩
    /-
      case intro
      R : Type u_1
      inst✝³ : Semiring R
      ι : Type u_2
      V : Type u
      inst✝² : CategoryTheory.Category.{v, u} V
      inst✝¹ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      inst✝ : CategoryTheory.CategoryWithHomology V
      i : ι
      K L : HomologicalComplex V c
      f g : Quiver.Hom K L
      h : Homotopy f g
      ⊢ Eq ((HomologicalComplex.homologyFunctor V c i).map f) ((HomologicalComplex.h …
    -/
    exact h.homologyMap_eq i)
    /-
      🎉 no goals
    -/


/-- The homology functor on the homotopy category is induced by
the homology functor on homological complexes. -/
noncomputable def homologyFunctorFactors (i : ι) :
    quotient V c ⋙ homologyFunctor V c i ≅
      HomologicalComplex.homologyFunctor V c i :=
  Quotient.lift.isLift _ _ _

-- this is to prevent any abuse of defeq

instance (i : ι) : (homologyFunctor V c i).Additive := by
  /-
    R : Type u_1
    inst✝³ : Semiring R
    ι : Type u_2
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    inst✝ : CategoryTheory.CategoryWithHomology V
    i : ι
    ⊢ (HomotopyCategory.homologyFunctor V c i).Additive
  -/
  have := Functor.additive_of_iso (homologyFunctorFactors V c i).symm
  /-
    R : Type u_1
    inst✝³ : Semiring R
    ι : Type u_2
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    inst✝ : CategoryTheory.CategoryWithHomology V
    i : ι
    this : ((HomotopyCategory.quotient V c).comp (HomotopyCategory.homologyFunctor …
    ⊢ (HomotopyCategory.homologyFunctor V c i).Additive
  -/
  exact Functor.additive_of_full_essSurj_comp (quotient V c) _
  /-
    🎉 no goals
  -/


/-- An additive functor induces a functor between homotopy categories. -/
@[simps! obj]
def Functor.mapHomotopyCategory (F : V ⥤ W) [F.Additive] (c : ComplexShape ι) :
    HomotopyCategory V c ⥤ HomotopyCategory W c :=
  CategoryTheory.Quotient.lift _ (F.mapHomologicalComplex c ⋙ HomotopyCategory.quotient W c)
    (fun _ _ _ _ ⟨h⟩ => HomotopyCategory.eq_of_homotopy _ _ (F.mapHomotopy h))


@[simp]
lemma Functor.mapHomotopyCategory_map (F : V ⥤ W) [F.Additive] {c : ComplexShape ι}
    {K L : HomologicalComplex V c} (f : K ⟶ L) :
    (F.mapHomotopyCategory c).map ((HomotopyCategory.quotient V c).map f) =
      (HomotopyCategory.quotient W c).map ((F.mapHomologicalComplex c).map f) :=
  rfl


/-- The obvious isomorphism between
`HomotopyCategory.quotient V c ⋙ F.mapHomotopyCategory c` and
`F.mapHomologicalComplex c ⋙ HomotopyCategory.quotient W c` when `F : V ⥤ W` is
an additive functor. -/
def Functor.mapHomotopyCategoryFactors (F : V ⥤ W) [F.Additive] (c : ComplexShape ι) :
    HomotopyCategory.quotient V c ⋙ F.mapHomotopyCategory c ≅
      F.mapHomologicalComplex c ⋙ HomotopyCategory.quotient W c :=
  CategoryTheory.Quotient.lift.isLift _ _ _

-- TODO `F.mapHomotopyCategory c` is additive (and linear when `F` is linear).
-- TODO develop lifting of natural transformations for general quotient categories so that
-- `NatTrans.mapHomotopyCategory` become a particular case of it

/-- A natural transformation induces a natural transformation between
  the induced functors on the homotopy category. -/
@[simps]
def NatTrans.mapHomotopyCategory {F G : V ⥤ W} [F.Additive] [G.Additive] (α : F ⟶ G)
    (c : ComplexShape ι) : F.mapHomotopyCategory c ⟶ G.mapHomotopyCategory c where
  app C := (HomotopyCategory.quotient W c).map ((NatTrans.mapHomologicalComplex α c).app C.as)
  naturality := by
    /-
      R : Type u_1
      inst✝⁶ : Semiring R
      ι : Type u_2
      V : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} V
      inst✝⁴ : CategoryTheory.Preadditive V
      c✝ : ComplexShape ι
      W : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.37757, u_3} W
      inst✝² : CategoryTheory.Preadditive W
      F G : CategoryTheory.Functor V W
      inst✝¹ : F.Additive
      inst✝ : G.Additive
      α : Quiver.Hom F G
      c : ComplexShape ι
      ⊢ ∀ ⦃X Y : HomotopyCategory V c⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.Cate …
    -/
    rintro ⟨C⟩ ⟨D⟩ ⟨f : C ⟶ D⟩
    simp only [HomotopyCategory.quot_mk_eq_quotient_map, Functor.mapHomotopyCategory_map,
      ← Functor.map_comp, NatTrans.naturality]


@[simp]
theorem NatTrans.mapHomotopyCategory_id (c : ComplexShape ι) (F : V ⥤ W) [F.Additive] :
                                                                             /-
                                                                               ι : Type u_2
                                                                               V : Type u
                                                                               inst✝⁴ : CategoryTheory.Category.{v, u} V
                                                                               inst✝³ : CategoryTheory.Preadditive V
                                                                               W : Type u_3
                                                                               inst✝² : CategoryTheory.Category.{u_4, u_3} W
                                                                               inst✝¹ : CategoryTheory.Preadditive W
                                                                               c : ComplexShape ι
                                                                               F : CategoryTheory.Functor V W
                                                                               inst✝ : F.Additive
                                                                               ⊢ Eq (CategoryTheory.NatTrans.mapHomotopyCategory (CategoryTheory.CategoryStru …
                                                                             -/
    NatTrans.mapHomotopyCategory (𝟙 F) c = 𝟙 (F.mapHomotopyCategory c) := by aesop_cat
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[simp]
theorem NatTrans.mapHomotopyCategory_comp (c : ComplexShape ι) {F G H : V ⥤ W} [F.Additive]
    [G.Additive] [H.Additive] (α : F ⟶ G) (β : G ⟶ H) :
    NatTrans.mapHomotopyCategory (α ≫ β) c =
                                                                                /-
                                                                                  ι : Type u_2
                                                                                  V : Type u
                                                                                  inst✝⁶ : CategoryTheory.Category.{v, u} V
                                                                                  inst✝⁵ : CategoryTheory.Preadditive V
                                                                                  W : Type u_3
                                                                                  inst✝⁴ : CategoryTheory.Category.{u_4, u_3} W
                                                                                  inst✝³ : CategoryTheory.Preadditive W
                                                                                  c : ComplexShape ι
                                                                                  F G H : CategoryTheory.Functor V W
                                                                                  inst✝² : F.Additive
                                                                                  inst✝¹ : G.Additive
                                                                                  inst✝ : H.Additive
                                                                                  α : Quiver.Hom F G
                                                                                  β : Quiver.Hom G H
                                                                                  ⊢ Eq (CategoryTheory.NatTrans.mapHomotopyCategory (CategoryTheory.CategoryStru …
                                                                                -/
      NatTrans.mapHomotopyCategory α c ≫ NatTrans.mapHomotopyCategory β c := by aesop_cat
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


