/-- Given two classes of morphisms `W₁` and `W₂` on a category `C`, this is
the data of the factorization of a morphism `f : X ⟶ Y` as `i ≫ p` with
`W₁ i` and `W₂ p`. -/
structure MapFactorizationData {X Y : C} (f : X ⟶ Y) where
  /-- the intermediate object in the factorization -/
  Z : C
  /-- the first morphism in the factorization -/
  i : X ⟶ Z
  /-- the second morphism in the factorization -/
  p : Z ⟶ Y
  fac : i ≫ p = f := by aesop_cat
  hi : W₁ i
  hp : W₂ p


attribute [reassoc (attr := simp)] MapFactorizationData.fac


/-- The data of a term in `MapFactorizationData W₁ W₂ f` for any morphism `f`. -/
abbrev FactorizationData := ∀ {X Y : C} (f : X ⟶ Y), MapFactorizationData W₁ W₂ f


/-- The factorization axiom for two classes of morphisms `W₁` and `W₂` in a category `C`. It
asserts that any morphism can be factored as a morphism in `W₁` followed by a morphism
in `W₂`. -/
class HasFactorization : Prop where
  nonempty_mapFactorizationData {X Y : C} (f : X ⟶ Y) : Nonempty (MapFactorizationData W₁ W₂ f)


/-- A chosen term in `FactorizationData W₁ W₂` when `HasFactorization W₁ W₂` holds. -/
noncomputable def factorizationData [HasFactorization W₁ W₂] : FactorizationData W₁ W₂ :=
  fun _ => Nonempty.some (HasFactorization.nonempty_mapFactorizationData _)


/-- The class of morphisms that are of the form `i ≫ p` with `W₁ i` and `W₂ p`. -/
def comp : MorphismProperty C := fun _ _ f => Nonempty (MapFactorizationData W₁ W₂ f)


lemma comp_eq_top_iff : W₁.comp W₂ = ⊤ ↔ HasFactorization W₁ W₂ := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    W₁ W₂ : CategoryTheory.MorphismProperty C
    ⊢ Iff (Eq (W₁.comp W₂) Top.top) (W₁.HasFactorization W₂)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      W₁ W₂ : CategoryTheory.MorphismProperty C
      ⊢ Eq (W₁.comp W₂) Top.top → W₁.HasFactorization W₂
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      W₁ W₂ : CategoryTheory.MorphismProperty C
      h : Eq (W₁.comp W₂) Top.top
      ⊢ W₁.HasFactorization W₂
    -/
    refine ⟨fun f => ?_⟩
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      W₁ W₂ : CategoryTheory.MorphismProperty C
      h : Eq (W₁.comp W₂) Top.top
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      ⊢ Nonempty (W₁.MapFactorizationData W₂ f)
    -/
    have : W₁.comp W₂ f := by simp only [h, top_apply]
    /-
      case mp
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      W₁ W₂ : CategoryTheory.MorphismProperty C
      h : Eq (W₁.comp W₂) Top.top
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      this : W₁.comp W₂ f
      ⊢ Nonempty (W₁.MapFactorizationData W₂ f)
    -/
    exact ⟨this.some⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      W₁ W₂ : CategoryTheory.MorphismProperty C
      ⊢ W₁.HasFactorization W₂ → Eq (W₁.comp W₂) Top.top
    -/
  · intro
    /-
      case mpr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      W₁ W₂ : CategoryTheory.MorphismProperty C
      a✝ : W₁.HasFactorization W₂
      ⊢ Eq (W₁.comp W₂) Top.top
    -/
    ext X Y f
    /-
      case mpr.h
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      W₁ W₂ : CategoryTheory.MorphismProperty C
      a✝ : W₁.HasFactorization W₂
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Iff (W₁.comp W₂ f) (Top.top f)
    -/
    simp only [top_apply, iff_true]
    /-
      case mpr.h
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      W₁ W₂ : CategoryTheory.MorphismProperty C
      a✝ : W₁.HasFactorization W₂
      X Y : C
      f : Quiver.Hom X Y
      ⊢ W₁.comp W₂ f
    -/
    exact ⟨factorizationData W₁ W₂ f⟩
    /-
      🎉 no goals
    -/


/-- The data of a functorial factorization of any morphism in `C` as a morphism in `W₁`
followed by a morphism in `W₂`. -/
structure FunctorialFactorizationData where
  /-- the intermediate objects in the factorizations -/
  Z : Arrow C ⥤ C
  /-- the first morphism in the factorizations -/
  i : Arrow.leftFunc ⟶ Z
  /-- the second morphism in the factorizations -/
  p : Z ⟶ Arrow.rightFunc
  fac : i ≫ p = Arrow.leftToRight := by aesop_cat
  hi (f : Arrow C) : W₁ (i.app f)
  hp (f : Arrow C) : W₂ (p.app f)


attribute [reassoc (attr := simp)] fac


@[reassoc (attr := simp)]
lemma fac_app {f : Arrow C} : data.i.app f ≫ data.p.app f = f.hom := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    W₁ W₂ : CategoryTheory.MorphismProperty C
    data : W₁.FunctorialFactorizationData W₂
    f : CategoryTheory.Arrow C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (data.i.app f) (data.p.app f)) f.hom
  -/
  rw [← NatTrans.comp_app, fac,Arrow.leftToRight_app]
  /-
    🎉 no goals
  -/


/-- If `W₁ ≤ W₁'` and `W₂ ≤ W₂'`, then a functorial factorization for `W₁` and `W₂` induces
a functorial factorization for `W₁'` and `W₂'`. -/
def ofLE {W₁' W₂' : MorphismProperty C} (le₁ : W₁ ≤ W₁') (le₂ : W₂ ≤ W₂') :
    FunctorialFactorizationData W₁' W₂' where
  Z := data.Z
  i := data.i
  p := data.p
  hi f := le₁ _ (data.hi f)
  hp f := le₂ _ (data.hp f)


/-- The term in `FactorizationData W₁ W₂` that is deduced from a functorial factorization. -/
def factorizationData : FactorizationData W₁ W₂ := fun f =>
  { i := data.i.app (Arrow.mk f)
    p := data.p.app (Arrow.mk f)
    hi := data.hi _
    hp := data.hp _ }


/-- When `data : FunctorialFactorizationData W₁ W₂`, this is the
morphism `(data.factorizationData f).Z ⟶ (data.factorizationData g).Z` expressing the
functoriality of the intermediate objects of the factorizations
for `φ : Arrow.mk f ⟶ Arrow.mk g`. -/
def mapZ : (data.factorizationData f).Z ⟶ (data.factorizationData g).Z := data.Z.map φ


@[reassoc (attr := simp)]
lemma i_mapZ :
    (data.factorizationData f).i ≫ data.mapZ φ = φ.left ≫ (data.factorizationData g).i :=
  (data.i.naturality φ).symm


@[reassoc (attr := simp)]
lemma mapZ_p :
    data.mapZ φ ≫ (data.factorizationData g).p = (data.factorizationData f).p ≫ φ.right :=
  data.p.naturality φ


variable (f) in
@[simp]
lemma mapZ_id : data.mapZ (𝟙 (Arrow.mk f)) = 𝟙 _ :=
  data.Z.map_id _


@[reassoc, simp]
lemma mapZ_comp {X'' Y'' : C} {h : X'' ⟶ Y''} (ψ : Arrow.mk g ⟶ Arrow.mk h) :
    data.mapZ (φ ≫ ψ) = data.mapZ φ ≫ data.mapZ ψ :=
  data.Z.map_comp _ _


/-- Auxiliary definition for `FunctorialFactorizationData.functorCategory`. -/
@[simps]
def functorCategory.Z : Arrow (J ⥤ C) ⥤ J ⥤ C where
  obj f :=
    { obj := fun j => (data.factorizationData (f.hom.app j)).Z
      map := fun φ => data.mapZ
        { left := f.left.map φ
          right := f.right.map φ }
      map_id := fun j => by
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
          W₁ W₂ : CategoryTheory.MorphismProperty C
          data : W₁.FunctorialFactorizationData W₂
          J : Type u_2
          inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
          f : CategoryTheory.Arrow (CategoryTheory.Functor J C)
          j : J
          ⊢ Eq ({ obj := fun j => (data.factorizationData (f.hom.app j)).Z, map := fun { …
        -/
        dsimp
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
          W₁ W₂ : CategoryTheory.MorphismProperty C
          data : W₁.FunctorialFactorizationData W₂
          J : Type u_2
          inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
          f : CategoryTheory.Arrow (CategoryTheory.Functor J C)
          j : J
          ⊢ Eq (data.mapZ { left := f.left.map (CategoryTheory.CategoryStruct.id j), rig …
        -/
        rw [← data.mapZ_id (f.hom.app j)]
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
          W₁ W₂ : CategoryTheory.MorphismProperty C
          data : W₁.FunctorialFactorizationData W₂
          J : Type u_2
          inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
          f : CategoryTheory.Arrow (CategoryTheory.Functor J C)
          j : J
          ⊢ Eq (data.mapZ { left := f.left.map (CategoryTheory.CategoryStruct.id j), rig …
        -/
                  /-
                    🎉 no goals
                  -/
        congr <;> simp
                  /-
                    🎉 no goals
                  -/
      map_comp := fun _ _ => by
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
          W₁ W₂ : CategoryTheory.MorphismProperty C
          data : W₁.FunctorialFactorizationData W₂
          J : Type u_2
          inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
          f : CategoryTheory.Arrow (CategoryTheory.Functor J C)
          X✝ Y✝ Z✝ : J
          x✝¹ : Quiver.Hom X✝ Y✝
          x✝ : Quiver.Hom Y✝ Z✝
          ⊢ Eq ({ obj := fun j => (data.factorizationData (f.hom.app j)).Z, map := fun { …
        -/
        dsimp
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
          W₁ W₂ : CategoryTheory.MorphismProperty C
          data : W₁.FunctorialFactorizationData W₂
          J : Type u_2
          inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
          f : CategoryTheory.Arrow (CategoryTheory.Functor J C)
          X✝ Y✝ Z✝ : J
          x✝¹ : Quiver.Hom X✝ Y✝
          x✝ : Quiver.Hom Y✝ Z✝
          ⊢ Eq (data.mapZ { left := f.left.map (CategoryTheory.CategoryStruct.comp x✝¹ x …
        -/
        rw [← data.mapZ_comp]
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
          W₁ W₂ : CategoryTheory.MorphismProperty C
          data : W₁.FunctorialFactorizationData W₂
          J : Type u_2
          inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
          f : CategoryTheory.Arrow (CategoryTheory.Functor J C)
          X✝ Y✝ Z✝ : J
          x✝¹ : Quiver.Hom X✝ Y✝
          x✝ : Quiver.Hom Y✝ Z✝
          ⊢ Eq (data.mapZ { left := f.left.map (CategoryTheory.CategoryStruct.comp x✝¹ x …
        -/
                  /-
                    🎉 no goals
                  -/
        congr <;> simp }
                  /-
                    🎉 no goals
                  -/
  map τ :=
    { app := fun j => data.mapZ
        { left := τ.left.app j
          right := τ.right.app j
          w := congr_app τ.w j }
      naturality := fun _ _ α => by
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
          W₁ W₂ : CategoryTheory.MorphismProperty C
          data : W₁.FunctorialFactorizationData W₂
          J : Type u_2
          inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
          X✝ Y✝ : CategoryTheory.Arrow (CategoryTheory.Functor J C)
          τ : Quiver.Hom X✝ Y✝
          x✝¹ x✝ : J
          α : Quiver.Hom x✝¹ x✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun f => { obj := fun j => (data.f …
        -/
        dsimp
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
          W₁ W₂ : CategoryTheory.MorphismProperty C
          data : W₁.FunctorialFactorizationData W₂
          J : Type u_2
          inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
          X✝ Y✝ : CategoryTheory.Arrow (CategoryTheory.Functor J C)
          τ : Quiver.Hom X✝ Y✝
          x✝¹ x✝ : J
          α : Quiver.Hom x✝¹ x✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (data.mapZ { left := X✝.left.map α, r …
        -/
        rw [← data.mapZ_comp, ← data.mapZ_comp]
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
          W₁ W₂ : CategoryTheory.MorphismProperty C
          data : W₁.FunctorialFactorizationData W₂
          J : Type u_2
          inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
          X✝ Y✝ : CategoryTheory.Arrow (CategoryTheory.Functor J C)
          τ : Quiver.Hom X✝ Y✝
          x✝¹ x✝ : J
          α : Quiver.Hom x✝¹ x✝
          ⊢ Eq (data.mapZ (CategoryTheory.CategoryStruct.comp { left := X✝.left.map α, r …
        -/
        congr 1
        /-
          case e_φ
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
          W₁ W₂ : CategoryTheory.MorphismProperty C
          data : W₁.FunctorialFactorizationData W₂
          J : Type u_2
          inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
          X✝ Y✝ : CategoryTheory.Arrow (CategoryTheory.Functor J C)
          τ : Quiver.Hom X✝ Y✝
          x✝¹ x✝ : J
          α : Quiver.Hom x✝¹ x✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp { left := X✝.left.map α, right := X✝. …
        -/
                /-
                  🎉 no goals
                -/
        ext <;> simp }
                /-
                  🎉 no goals
                -/
  map_id f := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
      W₁ W₂ : CategoryTheory.MorphismProperty C
      data : W₁.FunctorialFactorizationData W₂
      J : Type u_2
      inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
      f : CategoryTheory.Arrow (CategoryTheory.Functor J C)
      ⊢ Eq ({ obj := fun f => { obj := fun j => (data.factorizationData (f.hom.app j …
    -/
    ext j
    /-
      case w.h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
      W₁ W₂ : CategoryTheory.MorphismProperty C
      data : W₁.FunctorialFactorizationData W₂
      J : Type u_2
      inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
      f : CategoryTheory.Arrow (CategoryTheory.Functor J C)
      j : J
      ⊢ Eq (({ obj := fun f => { obj := fun j => (data.factorizationData (f.hom.app  …
    -/
    dsimp
    /-
      case w.h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
      W₁ W₂ : CategoryTheory.MorphismProperty C
      data : W₁.FunctorialFactorizationData W₂
      J : Type u_2
      inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
      f : CategoryTheory.Arrow (CategoryTheory.Functor J C)
      j : J
      ⊢ Eq (data.mapZ { left := CategoryTheory.CategoryStruct.id (f.left.obj j), rig …
    -/
    rw [← data.mapZ_id]
    /-
      case w.h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
      W₁ W₂ : CategoryTheory.MorphismProperty C
      data : W₁.FunctorialFactorizationData W₂
      J : Type u_2
      inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
      f : CategoryTheory.Arrow (CategoryTheory.Functor J C)
      j : J
      ⊢ Eq (data.mapZ { left := CategoryTheory.CategoryStruct.id (f.left.obj j), rig …
    -/
    congr 1
    /-
      🎉 no goals
    -/
  map_comp f g := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
      W₁ W₂ : CategoryTheory.MorphismProperty C
      data : W₁.FunctorialFactorizationData W₂
      J : Type u_2
      inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
      X✝ Y✝ Z✝ : CategoryTheory.Arrow (CategoryTheory.Functor J C)
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun f => { obj := fun j => (data.factorizationData (f.hom.app j …
    -/
    ext j
    /-
      case w.h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
      W₁ W₂ : CategoryTheory.MorphismProperty C
      data : W₁.FunctorialFactorizationData W₂
      J : Type u_2
      inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
      X✝ Y✝ Z✝ : CategoryTheory.Arrow (CategoryTheory.Functor J C)
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      j : J
      ⊢ Eq (({ obj := fun f => { obj := fun j => (data.factorizationData (f.hom.app  …
    -/
    dsimp
    /-
      case w.h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
      W₁ W₂ : CategoryTheory.MorphismProperty C
      data : W₁.FunctorialFactorizationData W₂
      J : Type u_2
      inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
      X✝ Y✝ Z✝ : CategoryTheory.Arrow (CategoryTheory.Functor J C)
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      j : J
      ⊢ Eq (data.mapZ { left := CategoryTheory.CategoryStruct.comp (f.left.app j) (g …
    -/
    rw [← data.mapZ_comp]
    /-
      case w.h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13741, u_1} C
      W₁ W₂ : CategoryTheory.MorphismProperty C
      data : W₁.FunctorialFactorizationData W₂
      J : Type u_2
      inst✝ : CategoryTheory.Category.{?u.13827, u_2} J
      X✝ Y✝ Z✝ : CategoryTheory.Arrow (CategoryTheory.Functor J C)
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      j : J
      ⊢ Eq (data.mapZ { left := CategoryTheory.CategoryStruct.comp (f.left.app j) (g …
    -/
    congr 1
    /-
      🎉 no goals
    -/


/-- A functorial factorization in the category `C` extends to the functor category `J ⥤ C`. -/
def functorCategory :
    FunctorialFactorizationData (W₁.functorCategory J) (W₂.functorCategory J) where
  Z := functorCategory.Z data J
  i := { app := fun f => { app := fun j => (data.factorizationData (f.hom.app j)).i } }
  p := { app := fun f => { app := fun j => (data.factorizationData (f.hom.app j)).p } }
  hi _ _ := data.hi _
  hp _ _ := data.hp _


/-- The functorial factorization axiom for two classes of morphisms `W₁` and `W₂` in a
category `C`. It asserts that any morphism can be factored in a functorial manner
as a morphism in `W₁` followed by a morphism in `W₂`. -/
class HasFunctorialFactorization : Prop where
  nonempty_functorialFactorizationData : Nonempty (FunctorialFactorizationData W₁ W₂)


/-- A chosen term in `FunctorialFactorizationData W₁ W₂` when the functorial factorization
axiom `HasFunctorialFactorization W₁ W₂` holds. -/
noncomputable def functorialFactorizationData [HasFunctorialFactorization W₁ W₂] :
    FunctorialFactorizationData W₁ W₂ :=
  Nonempty.some (HasFunctorialFactorization.nonempty_functorialFactorizationData)


instance [HasFunctorialFactorization W₁ W₂] : HasFactorization W₁ W₂ where
  nonempty_mapFactorizationData f := ⟨(functorialFactorizationData W₁ W₂).factorizationData f⟩


instance [HasFunctorialFactorization W₁ W₂] (J : Type*) [Category J] :
    HasFunctorialFactorization (W₁.functorCategory J) (W₂.functorCategory J) :=
  ⟨⟨(functorialFactorizationData W₁ W₂).functorCategory J⟩⟩


