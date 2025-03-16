instance {X : C} : Top (MonoOver X) where top := mk' (𝟙 _)


instance {X : C} : Inhabited (MonoOver X) :=
  ⟨⊤⟩


/-- The morphism to the top object in `MonoOver X`. -/
def leTop (f : MonoOver X) : f ⟶ ⊤ :=
  homMk f.arrow (comp_id _)


@[simp]
theorem top_left (X : C) : ((⊤ : MonoOver X) : C) = X :=
  rfl


@[simp]
theorem top_arrow (X : C) : (⊤ : MonoOver X).arrow = 𝟙 X :=
  rfl


/-- `map f` sends `⊤ : MonoOver X` to `⟨X, f⟩ : MonoOver Y`. -/
def mapTop (f : X ⟶ Y) [Mono f] : (map f).obj ⊤ ≅ mk' f :=
                                                      /-
                                                        C : Type u₁
                                                        inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                        X Y Z : C
                                                        D : Type u₂
                                                        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                        f : Quiver.Hom X Y
                                                        inst✝ : CategoryTheory.Mono f
                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Ca …
                                                      -/
  iso_of_both_ways (homMk (𝟙 _) rfl) (homMk (𝟙 _) (by simp [id_comp f]))
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- The pullback of the top object in `MonoOver Y`
is (isomorphic to) the top object in `MonoOver X`. -/
def pullbackTop (f : X ⟶ Y) : (pullback f).obj ⊤ ≅ ⊤ :=
  iso_of_both_ways (leTop _)
                                      /-
                                        C : Type u₁
                                        inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                        X Y Z : C
                                        D : Type u₂
                                        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                        inst✝ : CategoryTheory.Limits.HasPullbacks C
                                        f : Quiver.Hom X Y
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp f ((CategoryTheory.MonoOver.forget Y) …
                                      -/
    (homMk (pullback.lift f (𝟙 _) (by aesop_cat)) (pullback.lift_snd _ _ _))
                                      /-
                                        🎉 no goals
                                      -/


/-- There is a morphism from `⊤ : MonoOver A` to the pullback of a monomorphism along itself;
as the category is thin this is an isomorphism. -/
def topLEPullbackSelf {A B : C} (f : A ⟶ B) [Mono f] :
    (⊤ : MonoOver A) ⟶ (pullback f).obj (mk' f) :=
  homMk _ (pullback.lift_snd _ _ rfl)


/-- The pullback of a monomorphism along itself is isomorphic to the top object. -/
def pullbackSelf {A B : C} (f : A ⟶ B) [Mono f] : (pullback f).obj (mk' f) ≅ ⊤ :=
  iso_of_both_ways (leTop _) (topLEPullbackSelf _)


instance {X : C} : Bot (MonoOver X) where bot := mk' (initial.to X)


@[simp]
theorem bot_left (X : C) : ((⊥ : MonoOver X) : C) = ⊥_ C :=
  rfl


@[simp]
theorem bot_arrow {X : C} : (⊥ : MonoOver X).arrow = initial.to X :=
  rfl


/-- The (unique) morphism from `⊥ : MonoOver X` to any other `f : MonoOver X`. -/
def botLE {X : C} (f : MonoOver X) : ⊥ ⟶ f :=
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    X✝ Y Z : C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : CategoryTheory.Limits.InitialMonoClass C
    X : C
    f : CategoryTheory.MonoOver X
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.initial.to f.o …
  -/
  homMk (initial.to _)
  /-
    🎉 no goals
  -/


/-- `map f` sends `⊥ : MonoOver X` to `⊥ : MonoOver Y`. -/
def mapBot (f : X ⟶ Y) [Mono f] : (map f).obj ⊥ ≅ ⊥ :=
                    /-
                      C : Type u₁
                      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                      X Y Z : C
                      D : Type u₂
                      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                      inst✝² : CategoryTheory.Limits.HasInitial C
                      inst✝¹ : CategoryTheory.Limits.InitialMonoClass C
                      f : Quiver.Hom X Y
                      inst✝ : CategoryTheory.Mono f
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.initial.to Bot …
                    -/
                    /-
                      🎉 no goals
                    -/
  iso_of_both_ways (homMk (initial.to _)) (homMk (𝟙 _))
                                           /-
                                             🎉 no goals
                                           -/


/-- The object underlying `⊥ : Subobject B` is (up to isomorphism) the zero object. -/
def botCoeIsoZero {B : C} : ((⊥ : MonoOver B) : C) ≅ 0 :=
  initialIsInitial.uniqueUpToIso HasZeroObject.zeroIsInitial

-- Porting note: removed @[simp] as the LHS simplifies

theorem bot_arrow_eq_zero [HasZeroMorphisms C] {B : C} : (⊥ : MonoOver B).arrow = 0 :=
  zero_of_source_iso_zero _ botCoeIsoZero


/-- When `[HasPullbacks C]`, `MonoOver A` has "intersections", functorial in both arguments.

As `MonoOver A` is only a preorder, this doesn't satisfy the axioms of `SemilatticeInf`,
but we reuse all the names from `SemilatticeInf` because they will be used to construct
`SemilatticeInf (subobject A)` shortly.
-/
@[simps]
def inf {A : C} : MonoOver A ⥤ MonoOver A ⥤ MonoOver A where
  obj f := pullback f.arrow ⋙ map f.arrow
  map k :=
    { app := fun g => by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          X Y Z : C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          inst✝ : CategoryTheory.Limits.HasPullbacks C
          A : C
          X✝ Y✝ : CategoryTheory.MonoOver A
          k : Quiver.Hom X✝ Y✝
          g : CategoryTheory.MonoOver A
          ⊢ Quiver.Hom (((fun f => (CategoryTheory.MonoOver.pullback f.arrow).comp (Cate …
        -/
        apply homMk _ _
          /-
            C : Type u₁
            inst✝² : CategoryTheory.Category.{v₁, u₁} C
            X Y Z : C
            D : Type u₂
            inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
            inst✝ : CategoryTheory.Limits.HasPullbacks C
            A : C
            X✝ Y✝ : CategoryTheory.MonoOver A
            k : Quiver.Hom X✝ Y✝
            g : CategoryTheory.MonoOver A
            ⊢ Quiver.Hom (((fun f => (CategoryTheory.MonoOver.pullback f.arrow).comp (Cate …
          -/
        · apply pullback.lift (pullback.fst _ _) (pullback.snd _ _ ≫ k.left) _
          /-
            C : Type u₁
            inst✝² : CategoryTheory.Category.{v₁, u₁} C
            X Y Z : C
            D : Type u₂
            inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
            inst✝ : CategoryTheory.Limits.HasPullbacks C
            A : C
            X✝ Y✝ : CategoryTheory.MonoOver A
            k : Quiver.Hom X✝ Y✝
            g : CategoryTheory.MonoOver A
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
          -/
          rw [pullback.condition, assoc, w k]
          /-
            🎉 no goals
          -/
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          X Y Z : C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          inst✝ : CategoryTheory.Limits.HasPullbacks C
          A : C
          X✝ Y✝ : CategoryTheory.MonoOver A
          k : Quiver.Hom X✝ Y✝
          g : CategoryTheory.MonoOver A
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          X Y Z : C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          inst✝ : CategoryTheory.Limits.HasPullbacks C
          A : C
          X✝ Y✝ : CategoryTheory.MonoOver A
          k : Quiver.Hom X✝ Y✝
          g : CategoryTheory.MonoOver A
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
        -/
        rw [pullback.lift_snd_assoc, assoc, w k] }
        /-
          🎉 no goals
        -/


/-- A morphism from the "infimum" of two objects in `MonoOver A` to the first object. -/
def infLELeft {A : C} (f g : MonoOver A) : (inf.obj f).obj g ⟶ f :=
  homMk _ rfl


/-- A morphism from the "infimum" of two objects in `MonoOver A` to the second object. -/
def infLERight {A : C} (f g : MonoOver A) : (inf.obj f).obj g ⟶ g :=
  homMk _ pullback.condition


/-- A morphism version of the `le_inf` axiom. -/
def leInf {A : C} (f g h : MonoOver A) : (h ⟶ f) → (h ⟶ g) → (h ⟶ (inf.obj f).obj g) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    A : C
    f g h : CategoryTheory.MonoOver A
    ⊢ Quiver.Hom h f → Quiver.Hom h g → Quiver.Hom h ((CategoryTheory.MonoOver.inf …
  -/
  intro k₁ k₂
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    A : C
    f g h : CategoryTheory.MonoOver A
    k₁ : Quiver.Hom h f
    k₂ : Quiver.Hom h g
    ⊢ Quiver.Hom h ((CategoryTheory.MonoOver.inf.obj f).obj g)
  -/
  refine homMk (pullback.lift k₂.left k₁.left ?_) ?_
    /-
      case refine_1
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      A : C
      f g h : CategoryTheory.MonoOver A
      k₁ : Quiver.Hom h f
      k₂ : Quiver.Hom h g
      ⊢ Eq (CategoryTheory.CategoryStruct.comp k₂.left ((CategoryTheory.MonoOver.for …
    -/
  · rw [w k₁, w k₂]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      A : C
      f g h : CategoryTheory.MonoOver A
      k₁ : Quiver.Hom h f
      k₂ : Quiver.Hom h g
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
    -/
  · erw [pullback.lift_snd_assoc, w k₁]
    /-
      🎉 no goals
    -/


/-- When `[HasImages C] [HasBinaryCoproducts C]`, `MonoOver A` has a `sup` construction,
which is functorial in both arguments,
and which on `Subobject A` will induce a `SemilatticeSup`. -/
def sup {A : C} : MonoOver A ⥤ MonoOver A ⥤ MonoOver A :=
  curryObj ((forget A).prod (forget A) ⋙ uncurry.obj Over.coprod ⋙ image)


/-- A morphism version of `le_sup_left`. -/
def leSupLeft {A : C} (f g : MonoOver A) : f ⟶ (sup.obj f).obj g := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasImages C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
    A : C
    f g : CategoryTheory.MonoOver A
    ⊢ Quiver.Hom f ((CategoryTheory.MonoOver.sup.obj f).obj g)
  -/
  refine homMk (coprod.inl ≫ factorThruImage _) ?_
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasImages C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
    A : C
    f g : CategoryTheory.MonoOver A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp C …
  -/
  erw [Category.assoc, image.fac, coprod.inl_desc]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasImages C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
    A : C
    f g : CategoryTheory.MonoOver A
    ⊢ Eq (((CategoryTheory.MonoOver.forget A).prod (CategoryTheory.MonoOver.forget …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A morphism version of `le_sup_right`. -/
def leSupRight {A : C} (f g : MonoOver A) : g ⟶ (sup.obj f).obj g := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasImages C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
    A : C
    f g : CategoryTheory.MonoOver A
    ⊢ Quiver.Hom g ((CategoryTheory.MonoOver.sup.obj f).obj g)
  -/
  refine homMk (coprod.inr ≫ factorThruImage _) ?_
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasImages C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
    A : C
    f g : CategoryTheory.MonoOver A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp C …
  -/
  erw [Category.assoc, image.fac, coprod.inr_desc]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasImages C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
    A : C
    f g : CategoryTheory.MonoOver A
    ⊢ Eq (((CategoryTheory.MonoOver.forget A).prod (CategoryTheory.MonoOver.forget …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A morphism version of `sup_le`. -/
def supLe {A : C} (f g h : MonoOver A) : (f ⟶ h) → (g ⟶ h) → ((sup.obj f).obj g ⟶ h) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasImages C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
    A : C
    f g h : CategoryTheory.MonoOver A
    ⊢ Quiver.Hom f h → Quiver.Hom g h → Quiver.Hom ((CategoryTheory.MonoOver.sup.o …
  -/
  intro k₁ k₂
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasImages C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
    A : C
    f g h : CategoryTheory.MonoOver A
    k₁ : Quiver.Hom f h
    k₂ : Quiver.Hom g h
    ⊢ Quiver.Hom ((CategoryTheory.MonoOver.sup.obj f).obj g) h
  -/
  refine homMk ?_ ?_
    /-
      case refine_1
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Limits.HasImages C
      inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
      A : C
      f g h : CategoryTheory.MonoOver A
      k₁ : Quiver.Hom f h
      k₂ : Quiver.Hom g h
      ⊢ Quiver.Hom ((CategoryTheory.MonoOver.sup.obj f).obj g).obj.left h.obj.left
    -/
  · apply image.lift ⟨_, h.arrow, coprod.desc k₁.left k₂.left, _⟩
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Limits.HasImages C
      inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
      A : C
      f g h : CategoryTheory.MonoOver A
      k₁ : Quiver.Hom f h
      k₂ : Quiver.Hom g h
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.desc k₁ …
    -/
    ext
      /-
        case h₁
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        inst✝¹ : CategoryTheory.Limits.HasImages C
        inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
        A : C
        f g h : CategoryTheory.MonoOver A
        k₁ : Quiver.Hom f h
        k₂ : Quiver.Hom g h
        ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inl (Cat …
      -/
    · simp [w k₁]
      /-
        🎉 no goals
      -/
      /-
        case h₂
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        inst✝¹ : CategoryTheory.Limits.HasImages C
        inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
        A : C
        f g h : CategoryTheory.MonoOver A
        k₁ : Quiver.Hom f h
        k₂ : Quiver.Hom g h
        ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inr (Cat …
      -/
    · simp [w k₂]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      X Y Z : C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Limits.HasImages C
      inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
      A : C
      f g h : CategoryTheory.MonoOver A
      k₁ : Quiver.Hom f h
      k₂ : Quiver.Hom g h
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.lift (Ca …
    -/
  · apply image.lift_fac
    /-
      🎉 no goals
    -/


instance orderTop {X : C} : OrderTop (Subobject X) where
  top := Quotient.mk'' ⊤
  le_top := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X✝ Y Z : C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X : C
      ⊢ ∀ (a : CategoryTheory.Subobject X), LE.le a Top.top
    -/
    refine Quotient.ind' fun f => ?_
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X✝ Y Z : C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X : C
      f : CategoryTheory.MonoOver X
      ⊢ LE.le (Quotient.mk'' f) Top.top
    -/
    exact ⟨MonoOver.leTop f⟩
    /-
      🎉 no goals
    -/


instance {X : C} : Inhabited (Subobject X) :=
  ⟨⊤⟩


theorem top_eq_id (B : C) : (⊤ : Subobject B) = Subobject.mk (𝟙 B) :=
  rfl


theorem underlyingIso_top_hom {B : C} : (underlyingIso (𝟙 B)).hom = (⊤ : Subobject B).arrow := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    B : C
    ⊢ Eq (CategoryTheory.Subobject.underlyingIso (CategoryTheory.CategoryStruct.id …
  -/
  convert underlyingIso_hom_comp_eq_mk (𝟙 B)
  /-
    case h.e'_2
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    B : C
    ⊢ Eq (CategoryTheory.Subobject.underlyingIso (CategoryTheory.CategoryStruct.id …
  -/
  simp only [comp_id]
  /-
    🎉 no goals
  -/


instance top_arrow_isIso {B : C} : IsIso (⊤ : Subobject B).arrow := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    B : C
    ⊢ CategoryTheory.IsIso Top.top.arrow
  -/
  rw [← underlyingIso_top_hom]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    B : C
    ⊢ CategoryTheory.IsIso (CategoryTheory.Subobject.underlyingIso (CategoryTheory …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem underlyingIso_inv_top_arrow {B : C} :
    (underlyingIso _).inv ≫ (⊤ : Subobject B).arrow = 𝟙 B :=
  underlyingIso_arrow _


@[simp]
theorem map_top (f : X ⟶ Y) [Mono f] : (map f).obj ⊤ = Subobject.mk f :=
  Quotient.sound' ⟨MonoOver.mapTop f⟩


theorem top_factors {A B : C} (f : A ⟶ B) : (⊤ : Subobject B).Factors f :=
  ⟨f, comp_id _⟩


theorem isIso_iff_mk_eq_top {X Y : C} (f : X ⟶ Y) [Mono f] : IsIso f ↔ mk f = ⊤ :=
  ⟨fun _ => mk_eq_mk_of_comm _ _ (asIso f) (Category.comp_id _), fun h => by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono f
      h : Eq (CategoryTheory.Subobject.mk f) Top.top
      ⊢ CategoryTheory.IsIso f
    -/
    rw [← ofMkLEMk_comp h.le, Category.comp_id]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.Mono f
      h : Eq (CategoryTheory.Subobject.mk f) Top.top
      ⊢ CategoryTheory.IsIso (CategoryTheory.Subobject.ofMkLEMk f (CategoryTheory.Ca …
    -/
    exact (isoOfMkEqMk _ _ h).isIso_hom⟩
    /-
      🎉 no goals
    -/


theorem isIso_arrow_iff_eq_top {Y : C} (P : Subobject Y) : IsIso P.arrow ↔ P = ⊤ := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    Y : C
    P : CategoryTheory.Subobject Y
    ⊢ Iff (CategoryTheory.IsIso P.arrow) (Eq P Top.top)
  -/
  rw [isIso_iff_mk_eq_top, mk_arrow]
  /-
    🎉 no goals
  -/


                                                                       /-
                                                                         C : Type u₁
                                                                         inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                         X Y✝ Z : C
                                                                         D : Type u₂
                                                                         inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                         Y : C
                                                                         ⊢ CategoryTheory.IsIso Top.top.arrow
                                                                       -/
instance isIso_top_arrow {Y : C} : IsIso (⊤ : Subobject Y).arrow := by rw [isIso_arrow_iff_eq_top]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem mk_eq_top_of_isIso {X Y : C} (f : X ⟶ Y) [IsIso f] : mk f = ⊤ :=
  (isIso_iff_mk_eq_top f).mp inferInstance


theorem eq_top_of_isIso_arrow {Y : C} (P : Subobject Y) [IsIso P.arrow] : P = ⊤ :=
  (isIso_arrow_iff_eq_top P).mp inferInstance


theorem pullback_top (f : X ⟶ Y) : (pullback f).obj ⊤ = ⊤ :=
  Quotient.sound' ⟨MonoOver.pullbackTop f⟩


theorem pullback_self {A B : C} (f : A ⟶ B) [Mono f] : (pullback f).obj (mk f) = ⊤ :=
  Quotient.sound' ⟨MonoOver.pullbackSelf f⟩


instance orderBot {X : C} : OrderBot (Subobject X) where
  bot := Quotient.mk'' ⊥
  bot_le := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      X✝ Y Z : C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Limits.HasInitial C
      inst✝ : CategoryTheory.Limits.InitialMonoClass C
      X : C
      ⊢ ∀ (a : CategoryTheory.Subobject X), LE.le Bot.bot a
    -/
    refine Quotient.ind' fun f => ?_
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      X✝ Y Z : C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Limits.HasInitial C
      inst✝ : CategoryTheory.Limits.InitialMonoClass C
      X : C
      f : CategoryTheory.MonoOver X
      ⊢ LE.le Bot.bot (Quotient.mk'' f)
    -/
    exact ⟨MonoOver.botLE f⟩
    /-
      🎉 no goals
    -/


theorem bot_eq_initial_to {B : C} : (⊥ : Subobject B) = Subobject.mk (initial.to B) :=
  rfl


/-- The object underlying `⊥ : Subobject B` is (up to isomorphism) the initial object. -/
def botCoeIsoInitial {B : C} : ((⊥ : Subobject B) : C) ≅ ⊥_ C :=
  underlyingIso _


theorem map_bot (f : X ⟶ Y) [Mono f] : (map f).obj ⊥ = ⊥ :=
  Quotient.sound' ⟨MonoOver.mapBot f⟩


/-- The object underlying `⊥ : Subobject B` is (up to isomorphism) the zero object. -/
def botCoeIsoZero {B : C} : ((⊥ : Subobject B) : C) ≅ 0 :=
  botCoeIsoInitial ≪≫ initialIsInitial.uniqueUpToIso HasZeroObject.zeroIsInitial


theorem bot_eq_zero {B : C} : (⊥ : Subobject B) = Subobject.mk (0 : 0 ⟶ B) :=
  mk_eq_mk_of_comm _ _ (initialIsInitial.uniqueUpToIso HasZeroObject.zeroIsInitial)
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          B : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.initialIsIniti …
        -/
    (by simp [eq_iff_true_of_subsingleton])
        /-
          🎉 no goals
        -/


@[simp]
theorem bot_arrow {B : C} : (⊥ : Subobject B).arrow = 0 :=
  zero_of_source_iso_zero _ botCoeIsoZero


theorem bot_factors_iff_zero {A B : C} (f : A ⟶ B) : (⊥ : Subobject B).Factors f ↔ f = 0 :=
  ⟨by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      A B : C
      f : Quiver.Hom A B
      ⊢ Bot.bot.Factors f → Eq f 0
    -/
    rintro ⟨h, rfl⟩
    simp only [MonoOver.bot_arrow_eq_zero, Functor.id_obj, Functor.const_obj_obj,
      MonoOver.bot_left, comp_zero],
   by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      A B : C
      f : Quiver.Hom A B
      ⊢ Eq f 0 → Bot.bot.Factors f
    -/
    rintro rfl
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      A B : C
      ⊢ Bot.bot.Factors 0
    -/
    exact ⟨0, by simp⟩⟩
    /-
      🎉 no goals
    -/


theorem mk_eq_bot_iff_zero {f : X ⟶ Y} [Mono f] : Subobject.mk f = ⊥ ↔ f = 0 :=
               /-
                 C : Type u₁
                 inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                 X Y : C
                 inst✝² : CategoryTheory.Limits.HasZeroObject C
                 inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                 f : Quiver.Hom X Y
                 inst✝ : CategoryTheory.Mono f
                 h : Eq (CategoryTheory.Subobject.mk f) Bot.bot
                 ⊢ Eq f 0
               -/
  ⟨fun h => by simpa [h, bot_factors_iff_zero] using mk_factors_self f, fun h =>
               /-
                 🎉 no goals
               -/
                                                                                          /-
                                                                                            C : Type u₁
                                                                                            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                                                                            X Y : C
                                                                                            inst✝² : CategoryTheory.Limits.HasZeroObject C
                                                                                            inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                            f : Quiver.Hom X Y
                                                                                            inst✝ : CategoryTheory.Mono f
                                                                                            h : Eq f 0
                                                                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.isoZeroOfMono …
                                                                                          -/
    mk_eq_mk_of_comm _ _ ((isoZeroOfMonoEqZero h).trans HasZeroObject.zeroIsoInitial) (by simp [h])⟩
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


/-- Sending `X : C` to `Subobject X` is a contravariant functor `Cᵒᵖ ⥤ Type`. -/
@[simps]
def functor [HasPullbacks C] : Cᵒᵖ ⥤ Type max u₁ v₁ where
  obj X := Subobject X.unop
  map f := (pullback f.unop).obj
  map_id _ := funext pullback_id
  map_comp _ _ := funext (pullback_comp _ _)


/-- The functorial infimum on `MonoOver A` descends to an infimum on `Subobject A`. -/
def inf {A : C} : Subobject A ⥤ Subobject A ⥤ Subobject A :=
  ThinSkeleton.map₂ MonoOver.inf


theorem inf_le_left {A : C} (f g : Subobject A) : (inf.obj f).obj g ≤ f :=
  Quotient.inductionOn₂' f g fun _ _ => ⟨MonoOver.infLELeft _ _⟩


theorem inf_le_right {A : C} (f g : Subobject A) : (inf.obj f).obj g ≤ g :=
  Quotient.inductionOn₂' f g fun _ _ => ⟨MonoOver.infLERight _ _⟩


theorem le_inf {A : C} (h f g : Subobject A) : h ≤ f → h ≤ g → h ≤ (inf.obj f).obj g :=
  Quotient.inductionOn₃' h f g
    (by
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        A : C
        h f g : CategoryTheory.Subobject A
        ⊢ ∀ (a₁ a₂ a₃ : CategoryTheory.MonoOver A), LE.le (Quotient.mk'' a₁) (Quotient …
      -/
      rintro f g h ⟨k⟩ ⟨l⟩
      /-
        case intro.intro
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        A : C
        h✝ f✝ g✝ : CategoryTheory.Subobject A
        f g h : CategoryTheory.MonoOver A
        k : Quiver.Hom f g
        l : Quiver.Hom f h
        ⊢ LE.le (Quotient.mk'' f) ((CategoryTheory.Subobject.inf.obj (Quotient.mk'' g) …
      -/
      exact ⟨MonoOver.leInf _ _ _ k l⟩)
      /-
        🎉 no goals
      -/


instance semilatticeInf {B : C} : SemilatticeInf (Subobject B) where
  inf := fun m n => (inf.obj m).obj n
  inf_le_left := inf_le_left
  inf_le_right := inf_le_right
  le_inf := le_inf


theorem factors_left_of_inf_factors {A B : C} {X Y : Subobject B} {f : A ⟶ B}
    (h : (X ⊓ Y).Factors f) : X.Factors f :=
  factors_of_le _ (inf_le_left _ _) h


theorem factors_right_of_inf_factors {A B : C} {X Y : Subobject B} {f : A ⟶ B}
    (h : (X ⊓ Y).Factors f) : Y.Factors f :=
  factors_of_le _ (inf_le_right _ _) h


@[simp]
theorem inf_factors {A B : C} {X Y : Subobject B} (f : A ⟶ B) :
    (X ⊓ Y).Factors f ↔ X.Factors f ∧ Y.Factors f :=
  ⟨fun h => ⟨factors_left_of_inf_factors h, factors_right_of_inf_factors h⟩, by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      A B : C
      X Y : CategoryTheory.Subobject B
      f : Quiver.Hom A B
      ⊢ And (X.Factors f) (Y.Factors f) → (Min.min X Y).Factors f
    -/
    revert X Y
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      A B : C
      f : Quiver.Hom A B
      ⊢ ∀ {X Y : CategoryTheory.Subobject B}, And (X.Factors f) (Y.Factors f) → (Min …
    -/
    apply Quotient.ind₂'
    /-
      case h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      A B : C
      f : Quiver.Hom A B
      ⊢ ∀ (a₁ a₂ : CategoryTheory.MonoOver B), And (CategoryTheory.Subobject.Factors …
    -/
    rintro X Y ⟨⟨g₁, rfl⟩, ⟨g₂, hg₂⟩⟩
    /-
      case h.intro.intro.intro
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      A B : C
      X Y : CategoryTheory.MonoOver B
      g₁ : Quiver.Hom A X.obj.left
      g₂ : Quiver.Hom A Y.obj.left
      hg₂ : Eq (CategoryTheory.CategoryStruct.comp g₂ Y.arrow) (CategoryTheory.Categ …
      ⊢ (Min.min (Quotient.mk'' X) (Quotient.mk'' Y)).Factors (CategoryTheory.Catego …
    -/
    exact ⟨_, pullback.lift_snd_assoc _ _ hg₂ _⟩⟩
    /-
      🎉 no goals
    -/


theorem inf_arrow_factors_left {B : C} (X Y : Subobject B) : X.Factors (X ⊓ Y).arrow :=
                                                              /-
                                                                C : Type u₁
                                                                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                inst✝ : CategoryTheory.Limits.HasPullbacks C
                                                                B : C
                                                                X Y : CategoryTheory.Subobject B
                                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Min.min X Y).ofLE X ⋯) (CategoryThe …
                                                              -/
  (factors_iff _ _).mpr ⟨ofLE (X ⊓ Y) X (inf_le_left X Y), by simp⟩
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem inf_arrow_factors_right {B : C} (X Y : Subobject B) : Y.Factors (X ⊓ Y).arrow :=
                                                               /-
                                                                 C : Type u₁
                                                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                 inst✝ : CategoryTheory.Limits.HasPullbacks C
                                                                 B : C
                                                                 X Y : CategoryTheory.Subobject B
                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Min.min X Y).ofLE Y ⋯) (CategoryThe …
                                                               -/
  (factors_iff _ _).mpr ⟨ofLE (X ⊓ Y) Y (inf_le_right X Y), by simp⟩
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
theorem finset_inf_factors {I : Type*} {A B : C} {s : Finset I} {P : I → Subobject B} (f : A ⟶ B) :
    (s.inf P).Factors f ↔ ∀ i ∈ s, (P i).Factors f := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [top_factors]
  | insert _ ih => simp [ih]

-- `i` is explicit here because often we'd like to defer a proof of `m`

theorem finset_inf_arrow_factors {I : Type*} {B : C} (s : Finset I) (P : I → Subobject B) (i : I)
    (m : i ∈ s) : (P i).Factors (s.inf P).arrow := by
  classical
  revert i m
  induction s using Finset.induction_on with
  | empty => rintro _ ⟨⟩
  | insert _ ih =>
    intro _ m
    rw [Finset.inf_insert]
    simp only [Finset.mem_insert] at m
    rcases m with (rfl | m)
    · rw [← factorThru_arrow _ _ (inf_arrow_factors_left _ _)]
      exact factors_comp_arrow _
    · rw [← factorThru_arrow _ _ (inf_arrow_factors_right _ _)]
      apply factors_of_factors_right
      exact ih _ m


theorem inf_eq_map_pullback' {A : C} (f₁ : MonoOver A) (f₂ : Subobject A) :
    (Subobject.inf.obj (Quotient.mk'' f₁)).obj f₂ =
      (Subobject.map f₁.arrow).obj ((Subobject.pullback f₁.arrow).obj f₂) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    A : C
    f₁ : CategoryTheory.MonoOver A
    f₂ : CategoryTheory.Subobject A
    ⊢ Eq ((CategoryTheory.Subobject.inf.obj (Quotient.mk'' f₁)).obj f₂) ((Category …
  -/
  induction' f₂ using Quotient.inductionOn' with f₂
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    A : C
    f₁ f₂ : CategoryTheory.MonoOver A
    ⊢ Eq ((CategoryTheory.Subobject.inf.obj (Quotient.mk'' f₁)).obj (Quotient.mk'' …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem inf_eq_map_pullback {A : C} (f₁ : MonoOver A) (f₂ : Subobject A) :
    (Quotient.mk'' f₁ ⊓ f₂ : Subobject A) = (map f₁.arrow).obj ((pullback f₁.arrow).obj f₂) :=
  inf_eq_map_pullback' f₁ f₂


theorem prod_eq_inf {A : C} {f₁ f₂ : Subobject A} [HasBinaryProduct f₁ f₂] :
    (f₁ ⨯ f₂) = f₁ ⊓ f₂ := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    A : C
    f₁ f₂ : CategoryTheory.Subobject A
    inst✝ : CategoryTheory.Limits.HasBinaryProduct f₁ f₂
    ⊢ Eq (CategoryTheory.Limits.prod f₁ f₂) (Min.min f₁ f₂)
  -/
  apply le_antisymm
    /-
      case a
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      A : C
      f₁ f₂ : CategoryTheory.Subobject A
      inst✝ : CategoryTheory.Limits.HasBinaryProduct f₁ f₂
      ⊢ LE.le (CategoryTheory.Limits.prod f₁ f₂) (Min.min f₁ f₂)
    -/
  · refine le_inf _ _ _ (Limits.prod.fst.le) (Limits.prod.snd.le)
    /-
      🎉 no goals
    -/
    /-
      case a
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      A : C
      f₁ f₂ : CategoryTheory.Subobject A
      inst✝ : CategoryTheory.Limits.HasBinaryProduct f₁ f₂
      ⊢ LE.le (Min.min f₁ f₂) (CategoryTheory.Limits.prod f₁ f₂)
    -/
  · apply leOfHom
    /-
      case a.h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      A : C
      f₁ f₂ : CategoryTheory.Subobject A
      inst✝ : CategoryTheory.Limits.HasBinaryProduct f₁ f₂
      ⊢ Quiver.Hom (Min.min f₁ f₂) (CategoryTheory.Limits.prod f₁ f₂)
    -/
    exact prod.lift (inf_le_left _ _).hom (inf_le_right _ _).hom
    /-
      🎉 no goals
    -/


theorem inf_def {B : C} (m m' : Subobject B) : m ⊓ m' = (inf.obj m).obj m' :=
  rfl


/-- `⊓` commutes with pullback. -/
theorem inf_pullback {X Y : C} (g : X ⟶ Y) (f₁ f₂) :
    (pullback g).obj (f₁ ⊓ f₂) = (pullback g).obj f₁ ⊓ (pullback g).obj f₂ := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y : C
    g : Quiver.Hom X Y
    f₁ f₂ : CategoryTheory.Subobject Y
    ⊢ Eq ((CategoryTheory.Subobject.pullback g).obj (Min.min f₁ f₂)) (Min.min ((Ca …
  -/
  revert f₁
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y : C
    g : Quiver.Hom X Y
    f₂ : CategoryTheory.Subobject Y
    ⊢ ∀ (f₁ : CategoryTheory.Subobject Y), Eq ((CategoryTheory.Subobject.pullback  …
  -/
  apply Quotient.ind'
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y : C
    g : Quiver.Hom X Y
    f₂ : CategoryTheory.Subobject Y
    ⊢ ∀ (a : CategoryTheory.MonoOver Y), Eq ((CategoryTheory.Subobject.pullback g) …
  -/
  intro f₁
  erw [inf_def, inf_def, inf_eq_map_pullback', inf_eq_map_pullback', ← pullback_comp, ←
    map_pullback pullback.condition (pullbackIsPullback f₁.arrow g), ← pullback_comp,
    pullback.condition]
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y : C
    g : Quiver.Hom X Y
    f₂ : CategoryTheory.Subobject Y
    f₁ : CategoryTheory.MonoOver Y
    ⊢ Eq ((CategoryTheory.Subobject.map (CategoryTheory.Limits.pullback.snd f₁.arr …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `⊓` commutes with map. -/
theorem inf_map {X Y : C} (g : Y ⟶ X) [Mono g] (f₁ f₂) :
    (map g).obj (f₁ ⊓ f₂) = (map g).obj f₁ ⊓ (map g).obj f₂ := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    X Y : C
    g : Quiver.Hom Y X
    inst✝ : CategoryTheory.Mono g
    f₁ f₂ : CategoryTheory.Subobject Y
    ⊢ Eq ((CategoryTheory.Subobject.map g).obj (Min.min f₁ f₂)) (Min.min ((Categor …
  -/
  revert f₁
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    X Y : C
    g : Quiver.Hom Y X
    inst✝ : CategoryTheory.Mono g
    f₂ : CategoryTheory.Subobject Y
    ⊢ ∀ (f₁ : CategoryTheory.Subobject Y), Eq ((CategoryTheory.Subobject.map g).ob …
  -/
  apply Quotient.ind'
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    X Y : C
    g : Quiver.Hom Y X
    inst✝ : CategoryTheory.Mono g
    f₂ : CategoryTheory.Subobject Y
    ⊢ ∀ (a : CategoryTheory.MonoOver Y), Eq ((CategoryTheory.Subobject.map g).obj  …
  -/
  intro f₁
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    X Y : C
    g : Quiver.Hom Y X
    inst✝ : CategoryTheory.Mono g
    f₂ : CategoryTheory.Subobject Y
    f₁ : CategoryTheory.MonoOver Y
    ⊢ Eq ((CategoryTheory.Subobject.map g).obj (Min.min (Quotient.mk'' f₁) f₂)) (M …
  -/
  erw [inf_def, inf_def, inf_eq_map_pullback', inf_eq_map_pullback', ← map_comp]
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    X Y : C
    g : Quiver.Hom Y X
    inst✝ : CategoryTheory.Mono g
    f₂ : CategoryTheory.Subobject Y
    f₁ : CategoryTheory.MonoOver Y
    ⊢ Eq ((CategoryTheory.Subobject.map (CategoryTheory.CategoryStruct.comp f₁.arr …
  -/
  dsimp
  /-
    case h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    X Y : C
    g : Quiver.Hom Y X
    inst✝ : CategoryTheory.Mono g
    f₂ : CategoryTheory.Subobject Y
    f₁ : CategoryTheory.MonoOver Y
    ⊢ Eq ((CategoryTheory.Subobject.map (CategoryTheory.CategoryStruct.comp f₁.arr …
  -/
  rw [pullback_comp, pullback_map_self]
  /-
    🎉 no goals
  -/


/-- The functorial supremum on `MonoOver A` descends to a supremum on `Subobject A`. -/
def sup {A : C} : Subobject A ⥤ Subobject A ⥤ Subobject A :=
  ThinSkeleton.map₂ MonoOver.sup


instance semilatticeSup {B : C} : SemilatticeSup (Subobject B) where
  sup := fun m n => (sup.obj m).obj n
  le_sup_left := fun m n => Quotient.inductionOn₂' m n fun _ _ => ⟨MonoOver.leSupLeft _ _⟩
  le_sup_right := fun m n => Quotient.inductionOn₂' m n fun _ _ => ⟨MonoOver.leSupRight _ _⟩
  sup_le := fun m n k =>
    Quotient.inductionOn₃' m n k fun _ _ _ ⟨i⟩ ⟨j⟩ => ⟨MonoOver.supLe _ _ _ i j⟩


theorem sup_factors_of_factors_left {A B : C} {X Y : Subobject B} {f : A ⟶ B} (P : X.Factors f) :
    (X ⊔ Y).Factors f :=
  factors_of_le f le_sup_left P


theorem sup_factors_of_factors_right {A B : C} {X Y : Subobject B} {f : A ⟶ B} (P : Y.Factors f) :
    (X ⊔ Y).Factors f :=
  factors_of_le f le_sup_right P


theorem finset_sup_factors {I : Type*} {A B : C} {s : Finset I} {P : I → Subobject B} {f : A ⟶ B}
    (h : ∃ i ∈ s, (P i).Factors f) : (s.sup P).Factors f := by
  classical
  revert h
  induction s using Finset.induction_on with
  | empty => rintro ⟨_, ⟨⟨⟩, _⟩⟩
  | insert _ ih =>
    rintro ⟨j, ⟨m, h⟩⟩
    simp only [Finset.sup_insert]
    simp only [Finset.mem_insert] at m
    rcases m with (rfl | m)
    · exact sup_factors_of_factors_left h
    · exact sup_factors_of_factors_right (ih ⟨j, ⟨m, h⟩⟩)


instance boundedOrder [HasInitial C] [InitialMonoClass C] {B : C} : BoundedOrder (Subobject B) :=
  { Subobject.orderTop, Subobject.orderBot with }


instance {B : C} : Lattice (Subobject B) :=
  { Subobject.semilatticeInf, Subobject.semilatticeSup with }


/-- The "wide cospan" diagram, with a small indexing type, constructed from a set of subobjects.
(This is just the diagram of all the subobjects pasted together, but using `WellPowered C`
to make the diagram small.)
-/
def wideCospan {A : C} (s : Set (Subobject A)) : WidePullbackShape (equivShrink _ '' s) ⥤ C :=
  WidePullbackShape.wideCospan A
    (fun j : equivShrink _ '' s => ((equivShrink (Subobject A)).symm j : C)) fun j =>
    ((equivShrink (Subobject A)).symm j).arrow


@[simp]
theorem wideCospan_map_term {A : C} (s : Set (Subobject A)) (j) :
    (wideCospan s).map (WidePullbackShape.Hom.term j) =
      ((equivShrink (Subobject A)).symm j).arrow :=
  rfl


/-- Auxiliary construction of a cone for `le_inf`. -/
def leInfCone {A : C} (s : Set (Subobject A)) (f : Subobject A) (k : ∀ g ∈ s, f ≤ g) :
    Cone (wideCospan s) :=
  WidePullbackShape.mkCone f.arrow
    (fun j =>
      underlying.map
        (homOfLE
          (k _
            (by
              /-
                C : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                X Y Z : C
                D : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} D
                inst✝¹ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
                inst✝ : CategoryTheory.WellPowered.{w, v₁, u₁} C
                A : C
                s : Set (CategoryTheory.Subobject A)
                f : CategoryTheory.Subobject A
                k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le f g
                j : ↑(Set.image (⇑(equivShrink (CategoryTheory.Subobject A))) s)
                ⊢ Membership.mem s ((equivShrink (CategoryTheory.Subobject A)).symm ↑j)
              -/
              rcases j with ⟨-, ⟨g, ⟨m, rfl⟩⟩⟩
              /-
                case mk.intro.intro
                C : Type u₁
                inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                X Y Z : C
                D : Type u₂
                inst✝² : CategoryTheory.Category.{v₂, u₂} D
                inst✝¹ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
                inst✝ : CategoryTheory.WellPowered.{w, v₁, u₁} C
                A : C
                s : Set (CategoryTheory.Subobject A)
                f : CategoryTheory.Subobject A
                k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le f g
                g : CategoryTheory.Subobject A
                m : Membership.mem s g
                ⊢ Membership.mem s ((equivShrink (CategoryTheory.Subobject A)).symm ↑⟨(equivSh …
              -/
              simpa using m))))
              /-
                🎉 no goals
              -/
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          X Y Z : C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          inst✝¹ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
          inst✝ : CategoryTheory.WellPowered.{w, v₁, u₁} C
          A : C
          s : Set (CategoryTheory.Subobject A)
          f : CategoryTheory.Subobject A
          k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le f g
          ⊢ ∀ (j : ↑(Set.image (⇑(equivShrink (CategoryTheory.Subobject A))) s)), Eq (Ca …
        -/
    (by aesop_cat)
        /-
          🎉 no goals
        -/


@[simp]
theorem leInfCone_π_app_none {A : C} (s : Set (Subobject A)) (f : Subobject A)
    (k : ∀ g ∈ s, f ≤ g) : (leInfCone s f k).π.app none = f.arrow :=
  rfl


/-- The limit of `wideCospan s`. (This will be the supremum of the set of subobjects.)
-/
def widePullback {A : C} (s : Set (Subobject A)) : C :=
  Limits.limit (wideCospan s)


/-- The inclusion map from `widePullback s` to `A`
-/
def widePullbackι {A : C} (s : Set (Subobject A)) : widePullback s ⟶ A :=
  Limits.limit.π (wideCospan s) none


instance widePullbackι_mono {A : C} (s : Set (Subobject A)) : Mono (widePullbackι s) :=
  ⟨fun u v h =>
    limit.hom_ext fun j => by
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        X Y Z : C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        inst✝² : CategoryTheory.LocallySmall.{w, v₁, u₁} C
        inst✝¹ : CategoryTheory.WellPowered.{w, v₁, u₁} C
        inst✝ : CategoryTheory.Limits.HasWidePullbacks C
        A : C
        s : Set (CategoryTheory.Subobject A)
        Z✝ : C
        u v : Quiver.Hom Z✝ (CategoryTheory.Subobject.widePullback s)
        h : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Subobject.widePul …
        j : CategoryTheory.Limits.WidePullbackShape ↑(Set.image (⇑(equivShrink (Catego …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Limits.limit.π (Cat …
      -/
      cases j
        /-
          case none
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          X Y Z : C
          D : Type u₂
          inst✝³ : CategoryTheory.Category.{v₂, u₂} D
          inst✝² : CategoryTheory.LocallySmall.{w, v₁, u₁} C
          inst✝¹ : CategoryTheory.WellPowered.{w, v₁, u₁} C
          inst✝ : CategoryTheory.Limits.HasWidePullbacks C
          A : C
          s : Set (CategoryTheory.Subobject A)
          Z✝ : C
          u v : Quiver.Hom Z✝ (CategoryTheory.Subobject.widePullback s)
          h : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Subobject.widePul …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Limits.limit.π (Cat …
        -/
      · exact h
        /-
          🎉 no goals
        -/
        /-
          case some
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          X Y Z : C
          D : Type u₂
          inst✝³ : CategoryTheory.Category.{v₂, u₂} D
          inst✝² : CategoryTheory.LocallySmall.{w, v₁, u₁} C
          inst✝¹ : CategoryTheory.WellPowered.{w, v₁, u₁} C
          inst✝ : CategoryTheory.Limits.HasWidePullbacks C
          A : C
          s : Set (CategoryTheory.Subobject A)
          Z✝ : C
          u v : Quiver.Hom Z✝ (CategoryTheory.Subobject.widePullback s)
          h : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Subobject.widePul …
          val✝ : ↑(Set.image (⇑(equivShrink (CategoryTheory.Subobject A))) s)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Limits.limit.π (Cat …
        -/
      · apply (cancel_mono ((equivShrink (Subobject A)).symm _).arrow).1
        /-
          case some
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          X Y Z : C
          D : Type u₂
          inst✝³ : CategoryTheory.Category.{v₂, u₂} D
          inst✝² : CategoryTheory.LocallySmall.{w, v₁, u₁} C
          inst✝¹ : CategoryTheory.WellPowered.{w, v₁, u₁} C
          inst✝ : CategoryTheory.Limits.HasWidePullbacks C
          A : C
          s : Set (CategoryTheory.Subobject A)
          Z✝ : C
          u v : Quiver.Hom Z✝ (CategoryTheory.Subobject.widePullback s)
          h : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Subobject.widePul …
          val✝ : ↑(Set.image (⇑(equivShrink (CategoryTheory.Subobject A))) s)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp u …
        -/
        rw [assoc, assoc]
        /-
          case some
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          X Y Z : C
          D : Type u₂
          inst✝³ : CategoryTheory.Category.{v₂, u₂} D
          inst✝² : CategoryTheory.LocallySmall.{w, v₁, u₁} C
          inst✝¹ : CategoryTheory.WellPowered.{w, v₁, u₁} C
          inst✝ : CategoryTheory.Limits.HasWidePullbacks C
          A : C
          s : Set (CategoryTheory.Subobject A)
          Z✝ : C
          u v : Quiver.Hom Z✝ (CategoryTheory.Subobject.widePullback s)
          h : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Subobject.widePul …
          val✝ : ↑(Set.image (⇑(equivShrink (CategoryTheory.Subobject A))) s)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.CategoryStruct.comp …
        -/
        erw [limit.w (wideCospan s) (WidePullbackShape.Hom.term _)]
        /-
          case some
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          X Y Z : C
          D : Type u₂
          inst✝³ : CategoryTheory.Category.{v₂, u₂} D
          inst✝² : CategoryTheory.LocallySmall.{w, v₁, u₁} C
          inst✝¹ : CategoryTheory.WellPowered.{w, v₁, u₁} C
          inst✝ : CategoryTheory.Limits.HasWidePullbacks C
          A : C
          s : Set (CategoryTheory.Subobject A)
          Z✝ : C
          u v : Quiver.Hom Z✝ (CategoryTheory.Subobject.widePullback s)
          h : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Subobject.widePul …
          val✝ : ↑(Set.image (⇑(equivShrink (CategoryTheory.Subobject A))) s)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Limits.limit.π (Cat …
        -/
        exact h⟩
        /-
          🎉 no goals
        -/


/-- When `[WellPowered C]` and `[HasWidePullbacks C]`, `Subobject A` has arbitrary infimums.
-/
def sInf {A : C} (s : Set (Subobject A)) : Subobject A :=
  Subobject.mk (widePullbackι s)


theorem sInf_le {A : C} (s : Set (Subobject A)) (f) (hf : f ∈ s) : sInf s ≤ f := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.LocallySmall.{w, v₁, u₁} C
    inst✝¹ : CategoryTheory.WellPowered.{w, v₁, u₁} C
    inst✝ : CategoryTheory.Limits.HasWidePullbacks C
    A : C
    s : Set (CategoryTheory.Subobject A)
    f : CategoryTheory.Subobject A
    hf : Membership.mem s f
    ⊢ LE.le (CategoryTheory.Subobject.sInf s) f
  -/
  fapply le_of_comm
  · exact (underlyingIso _).hom ≫
      Limits.limit.π (wideCospan s)
        (some ⟨equivShrink (Subobject A) f,
          Set.mem_image_of_mem (equivShrink (Subobject A)) hf⟩) ≫
      eqToHom (congr_arg (fun X : Subobject A => (X : C)) (Equiv.symm_apply_apply _ _))
    /-
      case w
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.LocallySmall.{w, v₁, u₁} C
      inst✝¹ : CategoryTheory.WellPowered.{w, v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasWidePullbacks C
      A : C
      s : Set (CategoryTheory.Subobject A)
      f : CategoryTheory.Subobject A
      hf : Membership.mem s f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · dsimp [sInf]
    simp only [Category.comp_id, Category.assoc, ← underlyingIso_hom_comp_eq_mk,
      Subobject.arrow_congr, congrArg_mpr_hom_left, Iso.cancel_iso_hom_left]
    /-
      case w
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.LocallySmall.{w, v₁, u₁} C
      inst✝¹ : CategoryTheory.WellPowered.{w, v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasWidePullbacks C
      A : C
      s : Set (CategoryTheory.Subobject A)
      f : CategoryTheory.Subobject A
      hf : Membership.mem s f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.π (Categ …
    -/
    convert limit.w (wideCospan s) (WidePullbackShape.Hom.term _)
    /-
      case h.e'_2.h.h.e'_7.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.LocallySmall.{w, v₁, u₁} C
      inst✝¹ : CategoryTheory.WellPowered.{w, v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasWidePullbacks C
      A : C
      s : Set (CategoryTheory.Subobject A)
      f : CategoryTheory.Subobject A
      hf : Membership.mem s f
      e_1✝ : Eq (Quiver.Hom (CategoryTheory.Subobject.widePullback s) A) (Quiver.Hom …
      e_5✝ : Eq A ((CategoryTheory.Subobject.wideCospan s).obj Option.none)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) f.arrow) ( …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/


theorem le_sInf {A : C} (s : Set (Subobject A)) (f : Subobject A) (k : ∀ g ∈ s, f ≤ g) :
    f ≤ sInf s := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.LocallySmall.{w, v₁, u₁} C
    inst✝¹ : CategoryTheory.WellPowered.{w, v₁, u₁} C
    inst✝ : CategoryTheory.Limits.HasWidePullbacks C
    A : C
    s : Set (CategoryTheory.Subobject A)
    f : CategoryTheory.Subobject A
    k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le f g
    ⊢ LE.le f (CategoryTheory.Subobject.sInf s)
  -/
  fapply le_of_comm
    /-
      case f
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.LocallySmall.{w, v₁, u₁} C
      inst✝¹ : CategoryTheory.WellPowered.{w, v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasWidePullbacks C
      A : C
      s : Set (CategoryTheory.Subobject A)
      f : CategoryTheory.Subobject A
      k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le f g
      ⊢ Quiver.Hom (CategoryTheory.Subobject.underlying.obj f) (CategoryTheory.Subob …
    -/
  · exact Limits.limit.lift _ (leInfCone s f k) ≫ (underlyingIso _).inv
    /-
      🎉 no goals
    -/
    /-
      case w
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.LocallySmall.{w, v₁, u₁} C
      inst✝¹ : CategoryTheory.WellPowered.{w, v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasWidePullbacks C
      A : C
      s : Set (CategoryTheory.Subobject A)
      f : CategoryTheory.Subobject A
      k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le f g
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · dsimp [sInf]
    /-
      case w
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.LocallySmall.{w, v₁, u₁} C
      inst✝¹ : CategoryTheory.WellPowered.{w, v₁, u₁} C
      inst✝ : CategoryTheory.Limits.HasWidePullbacks C
      A : C
      s : Set (CategoryTheory.Subobject A)
      f : CategoryTheory.Subobject A
      k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le f g
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [assoc, underlyingIso_arrow, widePullbackι, limit.lift_π, leInfCone_π_app_none]
    /-
      🎉 no goals
    -/


instance completeSemilatticeInf {B : C} : CompleteSemilatticeInf (Subobject B) where
  sInf := sInf
  sInf_le := sInf_le
  le_sInf := le_sInf


/-- The universal morphism out of the coproduct of a set of subobjects,
after using `[WellPowered C]` to reindex by a small type.
-/
def smallCoproductDesc {A : C} (s : Set (Subobject A)) :=
  Limits.Sigma.desc fun j : equivShrink _ '' s => ((equivShrink (Subobject A)).symm j).arrow


/-- When `[WellPowered C] [HasImages C] [HasCoproducts C]`,
`Subobject A` has arbitrary supremums. -/
def sSup {A : C} (s : Set (Subobject A)) : Subobject A :=
  Subobject.mk (image.ι (smallCoproductDesc s))


theorem le_sSup {A : C} (s : Set (Subobject A)) (f) (hf : f ∈ s) : f ≤ sSup s := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
    inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
    inst✝¹ : CategoryTheory.Limits.HasCoproducts C
    inst✝ : CategoryTheory.Limits.HasImages C
    A : C
    s : Set (CategoryTheory.Subobject A)
    f : CategoryTheory.Subobject A
    hf : Membership.mem s f
    ⊢ LE.le f (CategoryTheory.Subobject.sSup s)
  -/
  fapply le_of_comm
  · refine eqToHom ?_ ≫ Sigma.ι _ ⟨equivShrink (Subobject A) f, by simpa [Set.mem_image] using hf⟩
      ≫ factorThruImage _ ≫ (underlyingIso _).inv
    /-
      case f
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
      inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasCoproducts C
      inst✝ : CategoryTheory.Limits.HasImages C
      A : C
      s : Set (CategoryTheory.Subobject A)
      f : CategoryTheory.Subobject A
      hf : Membership.mem s f
      ⊢ Eq (CategoryTheory.Subobject.underlying.obj f) (CategoryTheory.Subobject.und …
    -/
    exact (congr_arg (fun X : Subobject A => (X : C)) (Equiv.symm_apply_apply _ _).symm)
    /-
      🎉 no goals
    -/
    /-
      case w
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
      inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasCoproducts C
      inst✝ : CategoryTheory.Limits.HasImages C
      A : C
      s : Set (CategoryTheory.Subobject A)
      f : CategoryTheory.Subobject A
      hf : Membership.mem s f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp [sSup, smallCoproductDesc]
    /-
      🎉 no goals
    -/


theorem symm_apply_mem_iff_mem_image {α β : Type*} (e : α ≃ β) (s : Set α) (x : β) :
    e.symm x ∈ s ↔ x ∈ e '' s :=
                             /-
                               α : Type u_1
                               β : Type u_2
                               e : Equiv α β
                               s : Set α
                               x : β
                               h : Membership.mem s (e.symm x)
                               ⊢ Eq (e (e.symm x)) x
                             -/
  ⟨fun h => ⟨e.symm x, h, by simp⟩, by
                             /-
                               🎉 no goals
                             -/
    /-
      α : Type u_1
      β : Type u_2
      e : Equiv α β
      s : Set α
      x : β
      ⊢ Membership.mem (Set.image (⇑e) s) x → Membership.mem s (e.symm x)
    -/
    rintro ⟨a, m, rfl⟩
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      e : Equiv α β
      s : Set α
      a : α
      m : Membership.mem s a
      ⊢ Membership.mem s (e.symm (e a))
    -/
    simpa using m⟩
    /-
      🎉 no goals
    -/


theorem sSup_le {A : C} (s : Set (Subobject A)) (f : Subobject A) (k : ∀ g ∈ s, g ≤ f) :
    sSup s ≤ f := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
    inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
    inst✝¹ : CategoryTheory.Limits.HasCoproducts C
    inst✝ : CategoryTheory.Limits.HasImages C
    A : C
    s : Set (CategoryTheory.Subobject A)
    f : CategoryTheory.Subobject A
    k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le g f
    ⊢ LE.le (CategoryTheory.Subobject.sSup s) f
  -/
  fapply le_of_comm
    /-
      case f
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
      inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasCoproducts C
      inst✝ : CategoryTheory.Limits.HasImages C
      A : C
      s : Set (CategoryTheory.Subobject A)
      f : CategoryTheory.Subobject A
      k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le g f
      ⊢ Quiver.Hom (CategoryTheory.Subobject.underlying.obj (CategoryTheory.Subobjec …
    -/
  · refine(underlyingIso _).hom ≫ image.lift ⟨_, f.arrow, ?_, ?_⟩
      /-
        case f.refine_1
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
        inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
        inst✝¹ : CategoryTheory.Limits.HasCoproducts C
        inst✝ : CategoryTheory.Limits.HasImages C
        A : C
        s : Set (CategoryTheory.Subobject A)
        f : CategoryTheory.Subobject A
        k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le g f
        ⊢ Quiver.Hom (CategoryTheory.Limits.sigmaObj fun j => CategoryTheory.Subobject …
      -/
    · refine Sigma.desc ?_
      /-
        case f.refine_1
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
        inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
        inst✝¹ : CategoryTheory.Limits.HasCoproducts C
        inst✝ : CategoryTheory.Limits.HasImages C
        A : C
        s : Set (CategoryTheory.Subobject A)
        f : CategoryTheory.Subobject A
        k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le g f
        ⊢ (b : ↑(Set.image (⇑(equivShrink (CategoryTheory.Subobject A))) s)) → Quiver. …
      -/
      rintro ⟨g, m⟩
      /-
        case f.refine_1.mk
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
        inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
        inst✝¹ : CategoryTheory.Limits.HasCoproducts C
        inst✝ : CategoryTheory.Limits.HasImages C
        A : C
        s : Set (CategoryTheory.Subobject A)
        f : CategoryTheory.Subobject A
        k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le g f
        g : Shrink.{w, max u₁ v₁} (CategoryTheory.Subobject A)
        m : Membership.mem (Set.image (⇑(equivShrink (CategoryTheory.Subobject A))) s) g
        ⊢ Quiver.Hom (CategoryTheory.Subobject.underlying.obj ((equivShrink (CategoryT …
      -/
      refine underlying.map (homOfLE (k _ ?_))
      /-
        case f.refine_1.mk
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
        inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
        inst✝¹ : CategoryTheory.Limits.HasCoproducts C
        inst✝ : CategoryTheory.Limits.HasImages C
        A : C
        s : Set (CategoryTheory.Subobject A)
        f : CategoryTheory.Subobject A
        k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le g f
        g : Shrink.{w, max u₁ v₁} (CategoryTheory.Subobject A)
        m : Membership.mem (Set.image (⇑(equivShrink (CategoryTheory.Subobject A))) s) g
        ⊢ Membership.mem s ((equivShrink (CategoryTheory.Subobject A)).symm ↑⟨g, m⟩)
      -/
      simpa using m
      /-
        🎉 no goals
      -/
      /-
        case f.refine_2
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
        inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
        inst✝¹ : CategoryTheory.Limits.HasCoproducts C
        inst✝ : CategoryTheory.Limits.HasImages C
        A : C
        s : Set (CategoryTheory.Subobject A)
        f : CategoryTheory.Subobject A
        k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le g f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.desc fun …
      -/
    · ext
      /-
        case f.refine_2.h
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
        inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
        inst✝¹ : CategoryTheory.Limits.HasCoproducts C
        inst✝ : CategoryTheory.Limits.HasImages C
        A : C
        s : Set (CategoryTheory.Subobject A)
        f : CategoryTheory.Subobject A
        k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le g f
        b✝ : ↑(Set.image (⇑(equivShrink (CategoryTheory.Subobject A))) s)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (fun j …
      -/
      dsimp [smallCoproductDesc]
      /-
        case f.refine_2.h
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
        inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
        inst✝¹ : CategoryTheory.Limits.HasCoproducts C
        inst✝ : CategoryTheory.Limits.HasImages C
        A : C
        s : Set (CategoryTheory.Subobject A)
        f : CategoryTheory.Subobject A
        k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le g f
        b✝ : ↑(Set.image (⇑(equivShrink (CategoryTheory.Subobject A))) s)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (fun j …
      -/
      simp
      /-
        🎉 no goals
      -/
    /-
      case w
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
      inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasCoproducts C
      inst✝ : CategoryTheory.Limits.HasImages C
      A : C
      s : Set (CategoryTheory.Subobject A)
      f : CategoryTheory.Subobject A
      k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le g f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · dsimp [sSup]
    /-
      case w
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.LocallySmall.{w, v₁, u₁} C
      inst✝² : CategoryTheory.WellPowered.{w, v₁, u₁} C
      inst✝¹ : CategoryTheory.Limits.HasCoproducts C
      inst✝ : CategoryTheory.Limits.HasImages C
      A : C
      s : Set (CategoryTheory.Subobject A)
      f : CategoryTheory.Subobject A
      k : ∀ (g : CategoryTheory.Subobject A), Membership.mem s g → LE.le g f
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [assoc, image.lift_fac, underlyingIso_hom_comp_eq_mk]
    /-
      🎉 no goals
    -/


instance completeSemilatticeSup {B : C} : CompleteSemilatticeSup (Subobject B) where
  sSup := sSup
  le_sSup := le_sSup
  sSup_le := sSup_le


instance {B : C} : CompleteLattice (Subobject B) :=
  { Subobject.semilatticeInf, Subobject.semilatticeSup, Subobject.boundedOrder,
    Subobject.completeSemilatticeInf, Subobject.completeSemilatticeSup with }


/-- A nonzero object has nontrivial subobject lattice. -/
theorem nontrivial_of_not_isZero {X : C} (h : ¬IsZero X) : Nontrivial (Subobject X) :=
  ⟨⟨mk (0 : 0 ⟶ X), mk (𝟙 X), fun w => h (IsZero.of_iso (isZero_zero C) (isoOfMkEqMk _ _ w).symm)⟩⟩


/-- The subobject lattice of a subobject `Y` is order isomorphic to the interval `Set.Iic Y`. -/
def subobjectOrderIso {X : C} (Y : Subobject X) : Subobject (Y : C) ≃o Set.Iic Y where
  toFun Z :=
    ⟨Subobject.mk (Z.arrow ≫ Y.arrow),
                                                                        /-
                                                                          C : Type u₁
                                                                          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                          X✝ Y✝ Z✝ : C
                                                                          D : Type u₂
                                                                          inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                          X : C
                                                                          Y : CategoryTheory.Subobject X
                                                                          Z : CategoryTheory.Subobject (CategoryTheory.Subobject.underlying.obj Y)
                                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                                                        -/
      Set.mem_Iic.mpr (le_of_comm ((underlyingIso _).hom ≫ Z.arrow) (by simp))⟩
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
  invFun Z := Subobject.mk (ofLE _ _ Z.2)
                                                      /-
                                                        C : Type u₁
                                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                        X✝ Y✝ Z✝ : C
                                                        D : Type u₂
                                                        inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                        X : C
                                                        Y : CategoryTheory.Subobject X
                                                        Z : CategoryTheory.Subobject (CategoryTheory.Subobject.underlying.obj Y)
                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subobject.underlyingI …
                                                      -/
  left_inv Z := mk_eq_of_comm _ (underlyingIso _) (by aesop_cat)
                                                      /-
                                                        🎉 no goals
                                                      -/
  right_inv Z := Subtype.ext (mk_eq_of_comm _ (underlyingIso _) (by
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            X✝ Y✝ Z✝ : C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            X : C
            Y : CategoryTheory.Subobject X
            Z : ↑(Set.Iic Y)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subobject.underlyingI …
          -/
          dsimp
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            X✝ Y✝ Z✝ : C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            X : C
            Y : CategoryTheory.Subobject X
            Z : ↑(Set.Iic Y)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subobject.underlyingI …
          -/
          simp [← Iso.eq_inv_comp]))
          /-
            🎉 no goals
          -/
  map_rel_iff' {W Z} := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X✝ Y✝ Z✝ : C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X : C
      Y : CategoryTheory.Subobject X
      W Z : CategoryTheory.Subobject (CategoryTheory.Subobject.underlying.obj Y)
      ⊢ Iff (LE.le ({ toFun := fun Z => ⟨CategoryTheory.Subobject.mk (CategoryTheory …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X✝ Y✝ Z✝ : C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      X : C
      Y : CategoryTheory.Subobject X
      W Z : CategoryTheory.Subobject (CategoryTheory.Subobject.underlying.obj Y)
      ⊢ Iff (LE.le ⟨CategoryTheory.Subobject.mk (CategoryTheory.CategoryStruct.comp  …
    -/
    constructor
      /-
        case mp
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X✝ Y✝ Z✝ : C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        X : C
        Y : CategoryTheory.Subobject X
        W Z : CategoryTheory.Subobject (CategoryTheory.Subobject.underlying.obj Y)
        ⊢ LE.le ⟨CategoryTheory.Subobject.mk (CategoryTheory.CategoryStruct.comp W.arr …
      -/
    · intro h
      exact le_of_comm (((underlyingIso _).inv ≫ ofLE _ _ (Subtype.mk_le_mk.mp h) ≫
        (underlyingIso _).hom)) (by aesop_cat)
      /-
        case mpr
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        X✝ Y✝ Z✝ : C
        D : Type u₂
        inst✝ : CategoryTheory.Category.{v₂, u₂} D
        X : C
        Y : CategoryTheory.Subobject X
        W Z : CategoryTheory.Subobject (CategoryTheory.Subobject.underlying.obj Y)
        ⊢ LE.le W Z → LE.le ⟨CategoryTheory.Subobject.mk (CategoryTheory.CategoryStruc …
      -/
    · intro h
      exact Subtype.mk_le_mk.mpr (le_of_comm
        ((underlyingIso _).hom ≫ ofLE _ _ h ≫ (underlyingIso _).inv) (by simp))


