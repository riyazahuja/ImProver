/-- An object `X` in a category is a *zero object* if for every object `Y`
there is a unique morphism `to : X → Y` and a unique morphism `from : Y → X`.

This is a characteristic predicate for `has_zero_object`. -/
structure IsZero (X : C) : Prop where
  /-- there are unique morphisms to the object -/
  unique_to : ∀ Y, Nonempty (Unique (X ⟶ Y))
  /-- there are unique morphisms from the object -/
  unique_from : ∀ Y, Nonempty (Unique (Y ⟶ X))


/-- If `h : IsZero X`, then `h.to_ Y` is a choice of unique morphism `X → Y`. -/
protected def to_ (h : IsZero X) (Y : C) : X ⟶ Y :=
  @default _ <| (h.unique_to Y).some.toInhabited


theorem eq_to (h : IsZero X) (f : X ⟶ Y) : f = h.to_ Y :=
  @Unique.eq_default _ (id _) _


theorem to_eq (h : IsZero X) (f : X ⟶ Y) : h.to_ Y = f :=
  (h.eq_to f).symm

-- Porting note: `from` is a reserved word, it was replaced by `from_`

/-- If `h : is_zero X`, then `h.from_ Y` is a choice of unique morphism `Y → X`. -/
protected def from_ (h : IsZero X) (Y : C) : Y ⟶ X :=
  @default _ <| (h.unique_from Y).some.toInhabited


theorem eq_from (h : IsZero X) (f : Y ⟶ X) : f = h.from_ Y :=
  @Unique.eq_default _ (id _) _


theorem from_eq (h : IsZero X) (f : Y ⟶ X) : h.from_ Y = f :=
  (h.eq_from f).symm


theorem eq_of_src (hX : IsZero X) (f g : X ⟶ Y) : f = g :=
  (hX.eq_to f).trans (hX.eq_to g).symm


theorem eq_of_tgt (hX : IsZero X) (f g : Y ⟶ X) : f = g :=
  (hX.eq_from f).trans (hX.eq_from g).symm


/-- Any two zero objects are isomorphic. -/
def iso (hX : IsZero X) (hY : IsZero Y) : X ≅ Y where
  hom := hX.to_ Y
  inv := hX.from_ Y
  hom_inv_id := hX.eq_of_src _ _
  inv_hom_id := hY.eq_of_src _ _


/-- A zero object is in particular initial. -/
protected def isInitial (hX : IsZero X) : IsInitial X :=
  @IsInitial.ofUnique _ _ X fun Y => (hX.unique_to Y).some


/-- A zero object is in particular terminal. -/
protected def isTerminal (hX : IsZero X) : IsTerminal X :=
  @IsTerminal.ofUnique _ _ X fun Y => (hX.unique_from Y).some


/-- The (unique) isomorphism between any initial object and the zero object. -/
def isoIsInitial (hX : IsZero X) (hY : IsInitial Y) : X ≅ Y :=
  IsInitial.uniqueUpToIso hX.isInitial hY


/-- The (unique) isomorphism between any terminal object and the zero object. -/
def isoIsTerminal (hX : IsZero X) (hY : IsTerminal Y) : X ≅ Y :=
  IsTerminal.uniqueUpToIso hX.isTerminal hY


theorem of_iso (hY : IsZero Y) (e : X ≅ Y) : IsZero X := by
  refine ⟨fun Z => ⟨⟨⟨e.hom ≫ hY.to_ Z⟩, fun f => ?_⟩⟩,
    fun Z => ⟨⟨⟨hY.from_ Z ≫ e.inv⟩, fun f => ?_⟩⟩⟩
    /-
      case refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      hY : CategoryTheory.Limits.IsZero Y
      e : CategoryTheory.Iso X Y
      Z : C
      f : Quiver.Hom X Z
      ⊢ Eq f Inhabited.default
    -/
  · rw [← cancel_epi e.inv]
    /-
      case refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      hY : CategoryTheory.Limits.IsZero Y
      e : CategoryTheory.Iso X Y
      Z : C
      f : Quiver.Hom X Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp e.inv f) (CategoryTheory.CategoryStru …
    -/
    apply hY.eq_of_src
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      hY : CategoryTheory.Limits.IsZero Y
      e : CategoryTheory.Iso X Y
      Z : C
      f : Quiver.Hom Z X
      ⊢ Eq f Inhabited.default
    -/
  · rw [← cancel_mono e.hom]
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      hY : CategoryTheory.Limits.IsZero Y
      e : CategoryTheory.Iso X Y
      Z : C
      f : Quiver.Hom Z X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f e.hom) (CategoryTheory.CategoryStru …
    -/
    apply hY.eq_of_tgt
    /-
      🎉 no goals
    -/


theorem op (h : IsZero X) : IsZero (Opposite.op X) :=
  ⟨fun Y => ⟨⟨⟨(h.from_ (Opposite.unop Y)).op⟩, fun _ => Quiver.Hom.unop_inj (h.eq_of_tgt _ _)⟩⟩,
    fun Y => ⟨⟨⟨(h.to_ (Opposite.unop Y)).op⟩, fun _ => Quiver.Hom.unop_inj (h.eq_of_src _ _)⟩⟩⟩


theorem unop {X : Cᵒᵖ} (h : IsZero X) : IsZero (Opposite.unop X) :=
  ⟨fun Y => ⟨⟨⟨(h.from_ (Opposite.op Y)).unop⟩, fun _ => Quiver.Hom.op_inj (h.eq_of_tgt _ _)⟩⟩,
    fun Y => ⟨⟨⟨(h.to_ (Opposite.op Y)).unop⟩, fun _ => Quiver.Hom.op_inj (h.eq_of_src _ _)⟩⟩⟩


theorem Iso.isZero_iff {X Y : C} (e : X ≅ Y) : IsZero X ↔ IsZero Y :=
  ⟨fun h => h.of_iso e.symm, fun h => h.of_iso e⟩


theorem Functor.isZero (F : C ⥤ D) (hF : ∀ X, IsZero (F.obj X)) : IsZero F := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    hF : ∀ (X : C), CategoryTheory.Limits.IsZero (F.obj X)
    ⊢ CategoryTheory.Limits.IsZero F
  -/
  constructor <;> intro G <;> refine ⟨⟨⟨?_⟩, ?_⟩⟩
  · refine
      { app := fun X => (hF _).to_ _
        naturality := ?_ }
    /-
      case unique_to.refine_1
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      hF : ∀ (X : C), CategoryTheory.Limits.IsZero (F.obj X)
      G : CategoryTheory.Functor C D
      ⊢ ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (F. …
    -/
    intros
    /-
      case unique_to.refine_1
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      hF : ∀ (X : C), CategoryTheory.Limits.IsZero (F.obj X)
      G : CategoryTheory.Functor C D
      X✝ Y✝ : C
      f✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f✝) ((fun X => ⋯.to_ (G.obj X) …
    -/
    exact (hF _).eq_of_src _ _
    /-
      🎉 no goals
    -/
    /-
      case unique_to.refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      hF : ∀ (X : C), CategoryTheory.Limits.IsZero (F.obj X)
      G : CategoryTheory.Functor C D
      ⊢ ∀ (a : Quiver.Hom F G), Eq a Inhabited.default
    -/
  · intro f
    /-
      case unique_to.refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      hF : ∀ (X : C), CategoryTheory.Limits.IsZero (F.obj X)
      G : CategoryTheory.Functor C D
      f : Quiver.Hom F G
      ⊢ Eq f Inhabited.default
    -/
    ext
    /-
      case unique_to.refine_2.w.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      hF : ∀ (X : C), CategoryTheory.Limits.IsZero (F.obj X)
      G : CategoryTheory.Functor C D
      f : Quiver.Hom F G
      x✝ : C
      ⊢ Eq (f.app x✝) (Inhabited.default.app x✝)
    -/
    apply (hF _).eq_of_src _ _
    /-
      🎉 no goals
    -/
  · refine
      { app := fun X => (hF _).from_ _
        naturality := ?_ }
    /-
      case unique_from.refine_1
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      hF : ∀ (X : C), CategoryTheory.Limits.IsZero (F.obj X)
      G : CategoryTheory.Functor C D
      ⊢ ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (G. …
    -/
    intros
    /-
      case unique_from.refine_1
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      hF : ∀ (X : C), CategoryTheory.Limits.IsZero (F.obj X)
      G : CategoryTheory.Functor C D
      X✝ Y✝ : C
      f✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f✝) ((fun X => ⋯.from_ (G.obj  …
    -/
    exact (hF _).eq_of_tgt _ _
    /-
      🎉 no goals
    -/
    /-
      case unique_from.refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      hF : ∀ (X : C), CategoryTheory.Limits.IsZero (F.obj X)
      G : CategoryTheory.Functor C D
      ⊢ ∀ (a : Quiver.Hom G F), Eq a Inhabited.default
    -/
  · intro f
    /-
      case unique_from.refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      hF : ∀ (X : C), CategoryTheory.Limits.IsZero (F.obj X)
      G : CategoryTheory.Functor C D
      f : Quiver.Hom G F
      ⊢ Eq f Inhabited.default
    -/
    ext
    /-
      case unique_from.refine_2.w.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      hF : ∀ (X : C), CategoryTheory.Limits.IsZero (F.obj X)
      G : CategoryTheory.Functor C D
      f : Quiver.Hom G F
      x✝ : C
      ⊢ Eq (f.app x✝) (Inhabited.default.app x✝)
    -/
    apply (hF _).eq_of_tgt _ _
    /-
      🎉 no goals
    -/


/-- A category "has a zero object" if it has an object which is both initial and terminal. -/
class HasZeroObject : Prop where
  /-- there exists a zero object -/
  zero : ∃ X : C, IsZero X


instance hasZeroObject_pUnit : HasZeroObject (Discrete PUnit) where zero :=
  ⟨⟨⟨⟩⟩,
    { unique_to := fun ⟨⟨⟩⟩ =>
      ⟨{ default := 𝟙 _,
                     /-
                       C : Type u
                       inst✝¹ : CategoryTheory.Category.{v, u} C
                       D : Type u'
                       inst✝ : CategoryTheory.Category.{v', u'} D
                       x✝ : CategoryTheory.Discrete PUnit.{u_1 + 1}
                       ⊢ ∀ (a : Quiver.Hom { as := PUnit.unit } { as := PUnit.unit }), Eq a Inhabited …
                     -/
          uniq := by subsingleton }⟩
                     /-
                       🎉 no goals
                     -/
      unique_from := fun ⟨⟨⟩⟩ =>
      ⟨{ default := 𝟙 _,
                     /-
                       C : Type u
                       inst✝¹ : CategoryTheory.Category.{v, u} C
                       D : Type u'
                       inst✝ : CategoryTheory.Category.{v', u'} D
                       x✝ : CategoryTheory.Discrete PUnit.{u_1 + 1}
                       ⊢ ∀ (a : Quiver.Hom { as := PUnit.unit } { as := PUnit.unit }), Eq a Inhabited …
                     -/
          uniq := by subsingleton }⟩}⟩
                     /-
                       🎉 no goals
                     -/


/-- Construct a `Zero C` for a category with a zero object.
This can not be a global instance as it will trigger for every `Zero C` typeclass search.
-/
protected def HasZeroObject.zero' : Zero C where zero := HasZeroObject.zero.choose


theorem isZero_zero : IsZero (0 : C) :=
  HasZeroObject.zero.choose_spec


instance hasZeroObject_op : HasZeroObject Cᵒᵖ :=
  ⟨⟨Opposite.op 0, IsZero.op (isZero_zero C)⟩⟩


theorem hasZeroObject_unop [HasZeroObject Cᵒᵖ] : HasZeroObject C :=
  ⟨⟨Opposite.unop 0, IsZero.unop (isZero_zero Cᵒᵖ)⟩⟩


theorem IsZero.hasZeroObject {X : C} (hX : IsZero X) : HasZeroObject C :=
  ⟨⟨X, hX⟩⟩


/-- Every zero object is isomorphic to *the* zero object. -/
def IsZero.isoZero [HasZeroObject C] {X : C} (hX : IsZero X) : X ≅ 0 :=
  hX.iso (isZero_zero C)


theorem IsZero.obj [HasZeroObject D] {F : C ⥤ D} (hF : IsZero F) (X : C) : IsZero (F.obj X) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.Limits.HasZeroObject D
    F : CategoryTheory.Functor C D
    hF : CategoryTheory.Limits.IsZero F
    X : C
    ⊢ CategoryTheory.Limits.IsZero (F.obj X)
  -/
  let G : C ⥤ D := (CategoryTheory.Functor.const C).obj 0
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.Limits.HasZeroObject D
    F : CategoryTheory.Functor C D
    hF : CategoryTheory.Limits.IsZero F
    X : C
    G : CategoryTheory.Functor C D := (CategoryTheory.Functor.const C).obj 0
    ⊢ CategoryTheory.Limits.IsZero (F.obj X)
  -/
  have hG : IsZero G := Functor.isZero _ fun _ => isZero_zero _
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.Limits.HasZeroObject D
    F : CategoryTheory.Functor C D
    hF : CategoryTheory.Limits.IsZero F
    X : C
    G : CategoryTheory.Functor C D := (CategoryTheory.Functor.const C).obj 0
    hG : CategoryTheory.Limits.IsZero G
    ⊢ CategoryTheory.Limits.IsZero (F.obj X)
  -/
  let e : F ≅ G := hF.iso hG
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.Limits.HasZeroObject D
    F : CategoryTheory.Functor C D
    hF : CategoryTheory.Limits.IsZero F
    X : C
    G : CategoryTheory.Functor C D := (CategoryTheory.Functor.const C).obj 0
    hG : CategoryTheory.Limits.IsZero G
    e : CategoryTheory.Iso F G := hF.iso hG
    ⊢ CategoryTheory.Limits.IsZero (F.obj X)
  -/
  exact (isZero_zero _).of_iso (e.app X)
  /-
    🎉 no goals
  -/


/-- There is a unique morphism from the zero object to any object `X`. -/
protected def uniqueTo (X : C) : Unique (0 ⟶ X) :=
  ((isZero_zero C).unique_to X).some


/-- There is a unique morphism from any object `X` to the zero object. -/
protected def uniqueFrom (X : C) : Unique (X ⟶ 0) :=
  ((isZero_zero C).unique_from X).some


@[ext]
theorem to_zero_ext {X : C} (f g : X ⟶ 0) : f = g :=
  (isZero_zero C).eq_of_tgt _ _


@[ext]
theorem from_zero_ext {X : C} (f g : 0 ⟶ X) : f = g :=
  (isZero_zero C).eq_of_src _ _


                                                          /-
                                                            C : Type u
                                                            inst✝² : CategoryTheory.Category.{v, u} C
                                                            D : Type u'
                                                            inst✝¹ : CategoryTheory.Category.{v', u'} D
                                                            inst✝ : CategoryTheory.Limits.HasZeroObject C
                                                            X : C
                                                            f g : CategoryTheory.Iso X 0
                                                            ⊢ Eq f g
                                                          -/
instance (X : C) : Subsingleton (X ≅ 0) := ⟨fun f g => by ext⟩
                                                          /-
                                                            🎉 no goals
                                                          -/


                                                                           /-
                                                                             C : Type u
                                                                             inst✝² : CategoryTheory.Category.{v, u} C
                                                                             D : Type u'
                                                                             inst✝¹ : CategoryTheory.Category.{v', u'} D
                                                                             inst✝ : CategoryTheory.Limits.HasZeroObject C
                                                                             X : C
                                                                             f : Quiver.Hom 0 X
                                                                             Z✝ : C
                                                                             g h : Quiver.Hom Z✝ 0
                                                                             x✝ : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.CategoryStruc …
                                                                             ⊢ Eq g h
                                                                           -/
instance {X : C} (f : 0 ⟶ X) : Mono f where right_cancellation g h _ := by ext
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


                                                                         /-
                                                                           C : Type u
                                                                           inst✝² : CategoryTheory.Category.{v, u} C
                                                                           D : Type u'
                                                                           inst✝¹ : CategoryTheory.Category.{v', u'} D
                                                                           inst✝ : CategoryTheory.Limits.HasZeroObject C
                                                                           X : C
                                                                           f : Quiver.Hom X 0
                                                                           Z✝ : C
                                                                           g h : Quiver.Hom 0 Z✝
                                                                           x✝ : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruc …
                                                                           ⊢ Eq g h
                                                                         -/
instance {X : C} (f : X ⟶ 0) : Epi f where left_cancellation g h _ := by ext
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


instance zero_to_zero_isIso (f : (0 : C) ⟶ 0) : IsIso f := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    f : Quiver.Hom 0 0
    ⊢ CategoryTheory.IsIso f
  -/
  convert show IsIso (𝟙 (0 : C)) by infer_instance
  /-
    case h.e'_5
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} D
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    f : Quiver.Hom 0 0
    ⊢ Eq f (CategoryTheory.CategoryStruct.id 0)
  -/
  subsingleton
  /-
    🎉 no goals
  -/


/-- A zero object is in particular initial. -/
def zeroIsInitial : IsInitial (0 : C) :=
  (isZero_zero C).isInitial


/-- A zero object is in particular terminal. -/
def zeroIsTerminal : IsTerminal (0 : C) :=
  (isZero_zero C).isTerminal


/-- A zero object is in particular initial. -/
instance (priority := 10) hasInitial : HasInitial C :=
  hasInitial_of_unique 0


/-- A zero object is in particular terminal. -/
instance (priority := 10) hasTerminal : HasTerminal C :=
  hasTerminal_of_unique 0


/-- The (unique) isomorphism between any initial object and the zero object. -/
def zeroIsoIsInitial {X : C} (t : IsInitial X) : 0 ≅ X :=
  zeroIsInitial.uniqueUpToIso t


/-- The (unique) isomorphism between any terminal object and the zero object. -/
def zeroIsoIsTerminal {X : C} (t : IsTerminal X) : 0 ≅ X :=
  zeroIsTerminal.uniqueUpToIso t


/-- The (unique) isomorphism between the chosen initial object and the chosen zero object. -/
def zeroIsoInitial [HasInitial C] : 0 ≅ ⊥_ C :=
  zeroIsInitial.uniqueUpToIso initialIsInitial


/-- The (unique) isomorphism between the chosen terminal object and the chosen zero object. -/
def zeroIsoTerminal [HasTerminal C] : 0 ≅ ⊤_ C :=
  zeroIsTerminal.uniqueUpToIso terminalIsTerminal


instance (priority := 100) initialMonoClass : InitialMonoClass C :=
                                                          /-
                                                            C : Type u
                                                            inst✝² : CategoryTheory.Category.{v, u} C
                                                            D : Type u'
                                                            inst✝¹ : CategoryTheory.Category.{v', u'} D
                                                            inst✝ : CategoryTheory.Limits.HasZeroObject C
                                                            X : C
                                                            ⊢ CategoryTheory.Mono (CategoryTheory.Limits.HasZeroObject.zeroIsInitial.to X)
                                                          -/
  InitialMonoClass.of_isInitial zeroIsInitial fun X => by infer_instance
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem Functor.isZero_iff [HasZeroObject D] (F : C ⥤ D) : IsZero F ↔ ∀ X, IsZero (F.obj X) :=
  ⟨fun hF X => hF.obj X, Functor.isZero _⟩


