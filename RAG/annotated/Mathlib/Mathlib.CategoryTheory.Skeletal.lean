/-- A category is skeletal if isomorphic objects are equal. -/
def Skeletal : Prop :=
  ∀ ⦃X Y : C⦄, IsIsomorphic X Y → X = Y


/-- `IsSkeletonOf C D F` says that `F : D ⥤ C` exhibits `D` as a skeletal full subcategory of `C`,
in particular `F` is a (strong) equivalence and `D` is skeletal.
-/
structure IsSkeletonOf (F : D ⥤ C) : Prop where
  /-- The category `D` has isomorphic objects equal -/
  skel : Skeletal D
  /-- The functor `F` is an equivalence -/
  eqv : F.IsEquivalence := by infer_instance


/-- If `C` is thin and skeletal, then any naturally isomorphic functors to `C` are equal. -/
theorem Functor.eq_of_iso {F₁ F₂ : D ⥤ C} [Quiver.IsThin C] (hC : Skeletal C) (hF : F₁ ≅ F₂) :
    F₁ = F₂ :=
  Functor.ext (fun X => hC ⟨hF.app X⟩) fun _ _ _ => Subsingleton.elim _ _


/-- If `C` is thin and skeletal, `D ⥤ C` is skeletal.
`CategoryTheory.functor_thin` shows it is thin also.
-/
theorem functor_skeletal [Quiver.IsThin C] (hC : Skeletal C) : Skeletal (D ⥤ C) := fun _ _ h =>
  h.elim (Functor.eq_of_iso hC)


/-- Construct the skeleton category as the induced category on the isomorphism classes, and derive
its category structure.
-/
def Skeleton : Type u₁ := InducedCategory (C := Quotient (isIsomorphicSetoid C)) C Quotient.out


instance [Inhabited C] : Inhabited (Skeleton C) :=
  ⟨⟦default⟧⟩

-- Porting note: previously `Skeleton` used `deriving Category`

noncomputable instance : Category (Skeleton C) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    ⊢ CategoryTheory.Category.{?u.1548, u₁} (CategoryTheory.Skeleton C)
  -/
  apply InducedCategory.category
  /-
    🎉 no goals
  -/


noncomputable instance {α} [CoeSort C α] : CoeSort (Skeleton C) α :=
  inferInstanceAs (CoeSort (InducedCategory _ _) _)


/-- The functor from the skeleton of `C` to `C`. -/
@[simps!]
noncomputable def fromSkeleton : Skeleton C ⥤ C :=
  inducedFunctor _

-- Porting note: previously `fromSkeleton` used `deriving Faithful, Full`

noncomputable instance : (fromSkeleton C).Full := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    ⊢ (CategoryTheory.fromSkeleton C).Full
  -/
  apply InducedCategory.full
  /-
    🎉 no goals
  -/

noncomputable instance : (fromSkeleton C).Faithful := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    ⊢ (CategoryTheory.fromSkeleton C).Faithful
  -/
  apply InducedCategory.faithful
  /-
    🎉 no goals
  -/


instance : (fromSkeleton C).EssSurj where mem_essImage X := ⟨Quotient.mk' X, Quotient.mk_out X⟩

-- Porting note: named this instance

noncomputable instance fromSkeleton.isEquivalence : (fromSkeleton C).IsEquivalence where


/-- The class of an object in the skeleton. -/
abbrev toSkeleton (X : C) : Skeleton C := ⟦X⟧


/-- The isomorphism between `⟦X⟧.out` and `X`. -/
noncomputable def preCounitIso (X : C) : (fromSkeleton C).obj (toSkeleton X) ≅ X :=
  Nonempty.some (Quotient.mk_out X)


/-- An inverse to `fromSkeleton C` that forms an equivalence with it. -/
@[simps] noncomputable def toSkeletonFunctor : C ⥤ Skeleton C where
  obj := toSkeleton
                    /-
                      C : Type u₁
                      inst✝² : CategoryTheory.Category.{v₁, u₁} C
                      D : Type u₂
                      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                      E : Type u₃
                      inst✝ : CategoryTheory.Category.{v₃, u₃} E
                      X Y : C
                      f : Quiver.Hom X Y
                      ⊢ Quiver.Hom (CategoryTheory.toSkeleton X) (CategoryTheory.toSkeleton Y)
                    -/
  map {X Y} f := by apply (preCounitIso X).hom ≫ f ≫ (preCounitIso Y).inv
                    /-
                      🎉 no goals
                    -/
                 /-
                   C : Type u₁
                   inst✝² : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                   E : Type u₃
                   inst✝ : CategoryTheory.Category.{v₃, u₃} E
                   x✝ : C
                   ⊢ Eq ({ obj := CategoryTheory.toSkeleton, map := fun {X Y} f => CategoryTheory …
                 -/
  map_id _ := by aesop
                 /-
                   🎉 no goals
                 -/
                     /-
                       C : Type u₁
                       inst✝² : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                       E : Type u₃
                       inst✝ : CategoryTheory.Category.{v₃, u₃} E
                       X✝ Y✝ Z✝ : C
                       x✝¹ : Quiver.Hom X✝ Y✝
                       x✝ : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := CategoryTheory.toSkeleton, map := fun {X Y} f => CategoryTheory …
                     -/
  map_comp _ _ := by change _ = CategoryStruct.comp (obj := C) _ _; simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- The equivalence between the skeleton and the category itself. -/
@[simps] noncomputable def skeletonEquivalence : Skeleton C ≌ C where
  functor := fromSkeleton C
  inverse := toSkeletonFunctor C
  unitIso := NatIso.ofComponents
    (fun X ↦ InducedCategory.isoMk (Nonempty.some <| Quotient.mk_out X.out).symm)
    fun _ ↦ .symm <| Iso.inv_hom_id_assoc _ _
               /-
                 C : Type u₁
                 inst✝² : CategoryTheory.Category.{v₁, u₁} C
                 D : Type u₂
                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                 E : Type u₃
                 inst✝ : CategoryTheory.Category.{v₃, u₃} E
                 ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
               -/
  counitIso := NatIso.ofComponents preCounitIso
               /-
                 🎉 no goals
               -/
  functor_unitIso_comp _ := Iso.inv_hom_id _


theorem skeleton_skeletal : Skeletal (Skeleton C) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    ⊢ CategoryTheory.Skeletal (CategoryTheory.Skeleton C)
  -/
  rintro X Y ⟨h⟩
  /-
    case intro
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : CategoryTheory.Skeleton C
    h : CategoryTheory.Iso X Y
    ⊢ Eq X Y
  -/
  have : X.out ≈ Y.out := ⟨(fromSkeleton C).mapIso h⟩
  /-
    case intro
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : CategoryTheory.Skeleton C
    h : CategoryTheory.Iso X Y
    this : HasEquiv.Equiv (Quotient.out X) (Quotient.out Y)
    ⊢ Eq X Y
  -/
  simpa using Quotient.sound this
  /-
    🎉 no goals
  -/


/-- The `skeleton` of `C` given by choice is a skeleton of `C`. -/
lemma skeleton_isSkeleton : IsSkeletonOf C (Skeleton C) (fromSkeleton C) where
  skel := skeleton_skeletal C
  eqv := fromSkeleton.isEquivalence C


/-- Two categories which are categorically equivalent have skeletons with equivalent objects.
-/
noncomputable def Equivalence.skeletonEquiv (e : C ≌ D) : Skeleton C ≃ Skeleton D :=
  let f := ((skeletonEquivalence C).trans e).trans (skeletonEquivalence D).symm
  { toFun := f.functor.obj
    invFun := f.inverse.obj
    left_inv := fun X => skeleton_skeletal C ⟨(f.unitIso.app X).symm⟩
    right_inv := fun Y => skeleton_skeletal D ⟨f.counitIso.app Y⟩ }


/-- Construct the skeleton category by taking the quotient of objects. This construction gives a
preorder with nice definitional properties, but is only really appropriate for thin categories.
If your original category is not thin, you probably want to be using `skeleton` instead of this.
-/
def ThinSkeleton : Type u₁ :=
  Quotient (isIsomorphicSetoid C)


instance inhabitedThinSkeleton [Inhabited C] : Inhabited (ThinSkeleton C) :=
  ⟨@Quotient.mk' C (isIsomorphicSetoid C) default⟩


instance ThinSkeleton.preorder : Preorder (ThinSkeleton C) where
  le :=
    @Quotient.lift₂ C C _ (isIsomorphicSetoid C) (isIsomorphicSetoid C)
      (fun X Y => Nonempty (X ⟶ Y))
        (by
          /-
            C : Type u₁
            inst✝² : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
            E : Type u₃
            inst✝ : CategoryTheory.Category.{v₃, u₃} E
            ⊢ ∀ (a₁ b₁ a₂ b₂ : C), HasEquiv.Equiv a₁ a₂ → HasEquiv.Equiv b₁ b₂ → Eq ((fun  …
          -/
          rintro _ _ _ _ ⟨i₁⟩ ⟨i₂⟩
          exact
            propext
              ⟨Nonempty.map fun f => i₁.inv ≫ f ≫ i₂.hom,
                Nonempty.map fun f => i₁.hom ≫ f ≫ i₂.inv⟩)
  le_refl := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      ⊢ ∀ (a : CategoryTheory.ThinSkeleton C), LE.le a a
    -/
    refine Quotient.ind fun a => ?_
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      a : C
      ⊢ LE.le (Quotient.mk (CategoryTheory.isIsomorphicSetoid C) a) (Quotient.mk (Ca …
    -/
    exact ⟨𝟙 _⟩
    /-
      🎉 no goals
    -/
  le_trans a b c := Quotient.inductionOn₃ a b c fun _ _ _ => Nonempty.map2 (· ≫ ·)


/-- The functor from a category to its thin skeleton. -/
@[simps]
def toThinSkeleton : C ⥤ ThinSkeleton C where
  obj := @Quotient.mk' C _
  map f := homOfLE (Nonempty.intro f)


/-- The thin skeleton is thin. -/
instance thin : Quiver.IsThin (ThinSkeleton C) := fun _ _ =>
  ⟨by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      x✝¹ x✝ : CategoryTheory.ThinSkeleton C
      ⊢ ∀ (a b : Quiver.Hom x✝¹ x✝), Eq a b
    -/
    rintro ⟨⟨f₁⟩⟩ ⟨⟨_⟩⟩
    /-
      case up.up.up.up
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      x✝¹ x✝ : CategoryTheory.ThinSkeleton C
      f₁ down✝ : LE.le x✝¹ x✝
      ⊢ Eq { down := { down := f₁ } } { down := { down := down✝ } }
    -/
    rfl⟩
    /-
      🎉 no goals
    -/


/-- A functor `C ⥤ D` computably lowers to a functor `ThinSkeleton C ⥤ ThinSkeleton D`. -/
@[simps]
def map (F : C ⥤ D) : ThinSkeleton C ⥤ ThinSkeleton D where
  obj := Quotient.map F.obj fun _ _ ⟨hX⟩ => ⟨F.mapIso hX⟩
  map {X} {Y} := Quotient.recOnSubsingleton₂ X Y fun _ _ k => homOfLE (k.le.elim fun t => ⟨F.map t⟩)


theorem comp_toThinSkeleton (F : C ⥤ D) : F ⋙ toThinSkeleton D = toThinSkeleton C ⋙ map F :=
  rfl


/-- Given a natural transformation `F₁ ⟶ F₂`, induce a natural transformation `map F₁ ⟶ map F₂`. -/
def mapNatTrans {F₁ F₂ : C ⥤ D} (k : F₁ ⟶ F₂) : map F₁ ⟶ map F₂ where
  app X := Quotient.recOnSubsingleton X fun x => ⟨⟨⟨k.app x⟩⟩⟩

/- Porting note: `map₂ObjMap`, `map₂Functor`, and `map₂NatTrans` were all extracted
from the original `map₂` proof. Lean needed an extensive amount explicit type
annotations to figure things out. This also translated into repeated deterministic
timeouts. The extracted defs allow for explicit motives for the multiple
descents to the quotients.

It would be better to prove that
`ThinSkeleton (C × D) ≌ ThinSkeleton C × ThinSkeleton D`
which is more immediate from comparing the preorders. Then one could get
`map₂` by currying.
-/

/-- Given a bifunctor, we descend to a function on objects of `ThinSkeleton` -/
def map₂ObjMap (F : C ⥤ D ⥤ E) : ThinSkeleton C → ThinSkeleton D → ThinSkeleton E :=
  fun x y =>
    @Quotient.map₂ C D (isIsomorphicSetoid C) (isIsomorphicSetoid D) E (isIsomorphicSetoid E)
      (fun X Y => (F.obj X).obj Y)
          (fun X₁ _ ⟨hX⟩ _ Y₂ ⟨hY⟩ => ⟨(F.obj X₁).mapIso hY ≪≫ (F.mapIso hX).app Y₂⟩) x y


/-- For each `x : ThinSkeleton C`, we promote `map₂ObjMap F x` to a functor -/
def map₂Functor (F : C ⥤ D ⥤ E) : ThinSkeleton C → ThinSkeleton D ⥤ ThinSkeleton E :=
  fun x =>
    { obj := fun y => map₂ObjMap F x y
      map := fun {y₁} {y₂} => @Quotient.recOnSubsingleton C (isIsomorphicSetoid C)
        (fun x => (y₁ ⟶ y₂) → (map₂ObjMap F x y₁ ⟶ map₂ObjMap F x y₂)) _ x fun X
          => Quotient.recOnSubsingleton₂ y₁ y₂ fun _ _ hY =>
            homOfLE (hY.le.elim fun g => ⟨(F.obj X).map g⟩) }


/-- This provides natural transformations `map₂Functor F x₁ ⟶ map₂Functor F x₂` given
`x₁ ⟶ x₂` -/
def map₂NatTrans (F : C ⥤ D ⥤ E) : {x₁ x₂ : ThinSkeleton C} → (x₁ ⟶ x₂) →
    (map₂Functor F x₁ ⟶ map₂Functor F x₂) := fun {x₁} {x₂} =>
  @Quotient.recOnSubsingleton₂ C C (isIsomorphicSetoid C) (isIsomorphicSetoid C)
    (fun x x' : ThinSkeleton C => (x ⟶ x') → (map₂Functor F x ⟶ map₂Functor F x')) _ x₁ x₂
    (fun X₁ X₂ f => { app := fun y =>
      Quotient.recOnSubsingleton y fun Y => homOfLE (f.le.elim fun f' => ⟨(F.map f').app Y⟩) })

-- TODO: state the lemmas about what happens when you compose with `toThinSkeleton`

/-- A functor `C ⥤ D ⥤ E` computably lowers to a functor
`ThinSkeleton C ⥤ ThinSkeleton D ⥤ ThinSkeleton E` -/
@[simps]
def map₂ (F : C ⥤ D ⥤ E) : ThinSkeleton C ⥤ ThinSkeleton D ⥤ ThinSkeleton E where
  obj := map₂Functor F
  map := map₂NatTrans F


instance toThinSkeleton_faithful : (toThinSkeleton C).Faithful where


/-- Use `Quotient.out` to create a functor out of the thin skeleton. -/
@[simps]
noncomputable def fromThinSkeleton : ThinSkeleton C ⥤ C where
  obj := Quotient.out
  map {x} {y} :=
    Quotient.recOnSubsingleton₂ x y fun X Y f =>
      (Nonempty.some (Quotient.mk_out X)).hom ≫ f.le.some ≫ (Nonempty.some (Quotient.mk_out Y)).inv


/-- The equivalence between the thin skeleton and the category itself. -/
noncomputable def equivalence : ThinSkeleton C ≌ C where
  functor := fromThinSkeleton C
  inverse := toThinSkeleton C
               /-
                 C : Type u₁
                 inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                 D : Type u₂
                 inst✝² : CategoryTheory.Category.{v₂, u₂} D
                 E : Type u₃
                 inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
                 inst✝ : Quiver.IsThin C
                 ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
               -/
             /-
               C : Type u₁
               inst✝³ : CategoryTheory.Category.{v₁, u₁} C
               D : Type u₂
               inst✝² : CategoryTheory.Category.{v₂, u₂} D
               E : Type u₃
               inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
               inst✝ : Quiver.IsThin C
               ⊢ ∀ {X Y : CategoryTheory.ThinSkeleton C} (f : Quiver.Hom X Y), Eq (CategoryTh …
             -/
  counitIso := NatIso.ofComponents fun X => Nonempty.some (Quotient.mk_out X)
             /-
               🎉 no goals
             -/
               /-
                 🎉 no goals
               -/
  unitIso := NatIso.ofComponents fun x => Quotient.recOnSubsingleton x fun X =>
    eqToIso (Quotient.sound ⟨(Nonempty.some (Quotient.mk_out X)).symm⟩)


noncomputable instance fromThinSkeleton_isEquivalence : (fromThinSkeleton C).IsEquivalence :=
  (equivalence C).isEquivalence_functor


theorem equiv_of_both_ways {X Y : C} (f : X ⟶ Y) (g : Y ⟶ X) : X ≈ Y :=
  ⟨iso_of_both_ways f g⟩


instance thinSkeletonPartialOrder : PartialOrder (ThinSkeleton C) :=
  { CategoryTheory.ThinSkeleton.preorder C with
    le_antisymm :=
      Quotient.ind₂
        (by
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝² : CategoryTheory.Category.{v₂, u₂} D
            E : Type u₃
            inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
            inst✝ : Quiver.IsThin C
            ⊢ ∀ (a b : C), LE.le (Quotient.mk (CategoryTheory.isIsomorphicSetoid C) a) (Qu …
          -/
          rintro _ _ ⟨f⟩ ⟨g⟩
          /-
            case intro.intro
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝² : CategoryTheory.Category.{v₂, u₂} D
            E : Type u₃
            inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
            inst✝ : Quiver.IsThin C
            a✝ b✝ : C
            f : Quiver.Hom a✝ b✝
            g : Quiver.Hom b✝ a✝
            ⊢ Eq (Quotient.mk (CategoryTheory.isIsomorphicSetoid C) a✝) (Quotient.mk (Cate …
          -/
          apply Quotient.sound (equiv_of_both_ways f g)) }
          /-
            🎉 no goals
          -/


theorem skeletal : Skeletal (ThinSkeleton C) := fun X Y =>
  Quotient.inductionOn₂ X Y fun _ _ h => h.elim fun i => i.1.le.antisymm i.2.le


theorem map_comp_eq (F : E ⥤ D) (G : D ⥤ C) : map (F ⋙ G) = map F ⋙ map G :=
  Functor.eq_of_iso skeletal <|
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} E
      inst✝ : Quiver.IsThin C
      F : CategoryTheory.Functor E D
      G : CategoryTheory.Functor D C
      ⊢ ∀ {X Y : CategoryTheory.ThinSkeleton E} (f : Quiver.Hom X Y), Eq (CategoryTh …
    -/
    NatIso.ofComponents fun X => Quotient.recOnSubsingleton X fun _ => Iso.refl _
    /-
      🎉 no goals
    -/


theorem map_id_eq : map (𝟭 C) = 𝟭 (ThinSkeleton C) :=
  Functor.eq_of_iso skeletal <|
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : Quiver.IsThin C
      ⊢ ∀ {X Y : CategoryTheory.ThinSkeleton C} (f : Quiver.Hom X Y), Eq (CategoryTh …
    -/
    NatIso.ofComponents fun X => Quotient.recOnSubsingleton X fun _ => Iso.refl _
    /-
      🎉 no goals
    -/


theorem map_iso_eq {F₁ F₂ : D ⥤ C} (h : F₁ ≅ F₂) : map F₁ = map F₂ :=
  Functor.eq_of_iso skeletal
    { hom := mapNatTrans h.hom
      inv := mapNatTrans h.inv }


/-- `fromThinSkeleton C` exhibits the thin skeleton as a skeleton. -/
lemma thinSkeleton_isSkeleton : IsSkeletonOf C (ThinSkeleton C) (fromThinSkeleton C) where
  skel := skeletal


instance isSkeletonOfInhabited :
    Inhabited (IsSkeletonOf C (ThinSkeleton C) (fromThinSkeleton C)) :=
  ⟨thinSkeleton_isSkeleton⟩


/-- An adjunction between thin categories gives an adjunction between their thin skeletons. -/
def lowerAdjunction (R : D ⥤ C) (L : C ⥤ D) (h : L ⊣ R) :
    ThinSkeleton.map L ⊣ ThinSkeleton.map R where
  unit :=
    { app := fun X => by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          E : Type u₃
          inst✝ : CategoryTheory.Category.{v₃, u₃} E
          R : CategoryTheory.Functor D C
          L : CategoryTheory.Functor C D
          h : CategoryTheory.Adjunction L R
          X : CategoryTheory.ThinSkeleton C
          ⊢ Quiver.Hom ((CategoryTheory.Functor.id (CategoryTheory.ThinSkeleton C)).obj  …
        -/
        letI := isIsomorphicSetoid C
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          E : Type u₃
          inst✝ : CategoryTheory.Category.{v₃, u₃} E
          R : CategoryTheory.Functor D C
          L : CategoryTheory.Functor C D
          h : CategoryTheory.Adjunction L R
          X : CategoryTheory.ThinSkeleton C
          this : Setoid C := CategoryTheory.isIsomorphicSetoid C
          ⊢ Quiver.Hom ((CategoryTheory.Functor.id (CategoryTheory.ThinSkeleton C)).obj  …
        -/
        exact Quotient.recOnSubsingleton X fun x => homOfLE ⟨h.unit.app x⟩ }
        /-
          🎉 no goals
        -/
      -- TODO: make quotient.rec_on_subsingleton' so the letI isn't needed
  counit :=
    { app := fun X => by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          E : Type u₃
          inst✝ : CategoryTheory.Category.{v₃, u₃} E
          R : CategoryTheory.Functor D C
          L : CategoryTheory.Functor C D
          h : CategoryTheory.Adjunction L R
          X : CategoryTheory.ThinSkeleton D
          ⊢ Quiver.Hom (((CategoryTheory.ThinSkeleton.map R).comp (CategoryTheory.ThinSk …
        -/
        letI := isIsomorphicSetoid D
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          E : Type u₃
          inst✝ : CategoryTheory.Category.{v₃, u₃} E
          R : CategoryTheory.Functor D C
          L : CategoryTheory.Functor C D
          h : CategoryTheory.Adjunction L R
          X : CategoryTheory.ThinSkeleton D
          this : Setoid D := CategoryTheory.isIsomorphicSetoid D
          ⊢ Quiver.Hom (((CategoryTheory.ThinSkeleton.map R).comp (CategoryTheory.ThinSk …
        -/
        exact Quotient.recOnSubsingleton X fun x => homOfLE ⟨h.counit.app x⟩ }
        /-
          🎉 no goals
        -/


/--
When `e : C ≌ α` is a categorical equivalence from a thin category `C` to some partial order `α`,
the `ThinSkeleton C` is order isomorphic to `α`.
-/
noncomputable def Equivalence.thinSkeletonOrderIso [Quiver.IsThin C] (e : C ≌ α) :
    ThinSkeleton C ≃o α :=
  ((ThinSkeleton.equivalence C).trans e).toOrderIso


