/-- An auxiliary structure that witnesses the fact that `f` factors through an image object of `G`.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed `@[nolint has_nonempty_instance]`
structure Presieve.CoverByImageStructure (G : C ⥤ D) {V U : D} (f : V ⟶ U) where
  obj : C
  lift : V ⟶ G.obj obj
  map : G.obj obj ⟶ U
  fac : lift ≫ map = f := by aesop_cat

attribute [reassoc (attr := simp)] Presieve.CoverByImageStructure.fac


/-- For a functor `G : C ⥤ D`, and an object `U : D`, `Presieve.coverByImage G U` is the presieve
of `U` consisting of those arrows that factor through images of `G`.
-/
def Presieve.coverByImage (G : C ⥤ D) (U : D) : Presieve U := fun _ f =>
  Nonempty (Presieve.CoverByImageStructure G f)


/-- For a functor `G : C ⥤ D`, and an object `U : D`, `Sieve.coverByImage G U` is the sieve of `U`
consisting of those arrows that factor through images of `G`.
-/
def Sieve.coverByImage (G : C ⥤ D) (U : D) : Sieve U :=
  ⟨Presieve.coverByImage G U, fun ⟨⟨Z, f₁, f₂, (e : _ = _)⟩⟩ g =>
                                                   /-
                                                     C : Type u_1
                                                     inst✝² : CategoryTheory.Category.{?u.2621, u_1} C
                                                     D : Type u_2
                                                     inst✝¹ : CategoryTheory.Category.{?u.2628, u_2} D
                                                     E : Type u_3
                                                     inst✝ : CategoryTheory.Category.{?u.2635, u_3} E
                                                     J : CategoryTheory.GrothendieckTopology C
                                                     K : CategoryTheory.GrothendieckTopology D
                                                     L : CategoryTheory.GrothendieckTopology E
                                                     G : CategoryTheory.Functor C D
                                                     U Y✝ Z✝ : D
                                                     f✝ : Quiver.Hom Y✝ U
                                                     x✝ : CategoryTheory.Presieve.coverByImage G U f✝
                                                     g : Quiver.Hom Z✝ Y✝
                                                     Z : C
                                                     f₁ : Quiver.Hom Y✝ (G.obj Z)
                                                     f₂ : Quiver.Hom (G.obj Z) U
                                                     e : Eq (CategoryTheory.CategoryStruct.comp f₁ f₂) f✝
                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp g …
                                                   -/
    ⟨⟨Z, g ≫ f₁, f₂, show (g ≫ f₁) ≫ f₂ = g ≫ _ by rw [Category.assoc, ← e]⟩⟩⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem Presieve.in_coverByImage (G : C ⥤ D) {X : D} {Y : C} (f : G.obj Y ⟶ X) :
    Presieve.coverByImage G X f :=
                  /-
                    C : Type u_1
                    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
                    D : Type u_2
                    inst✝ : CategoryTheory.Category.{u_5, u_2} D
                    G : CategoryTheory.Functor C D
                    X : D
                    Y : C
                    f : Quiver.Hom (G.obj Y) X
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (G. …
                  -/
  ⟨⟨Y, 𝟙 _, f, by simp⟩⟩
                  /-
                    🎉 no goals
                  -/


/-- A functor `G : (C, J) ⥤ (D, K)` is cover dense if for each object in `D`,
  there exists a covering sieve in `D` that factors through images of `G`.

This definition can be found in https://ncatlab.org/nlab/show/dense+sub-site Definition 2.2.
-/
class Functor.IsCoverDense (G : C ⥤ D) (K : GrothendieckTopology D) : Prop where
  is_cover : ∀ U : D, Sieve.coverByImage G U ∈ K U


lemma Functor.is_cover_of_isCoverDense (G : C ⥤ D) (K : GrothendieckTopology D)
    [G.IsCoverDense K] (U : D) : Sieve.coverByImage G U ∈ K U := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_5, u_2} D
    G : CategoryTheory.Functor C D
    K : CategoryTheory.GrothendieckTopology D
    inst✝ : G.IsCoverDense K
    U : D
    ⊢ Membership.mem (K U) (CategoryTheory.Sieve.coverByImage G U)
  -/
  apply Functor.IsCoverDense.is_cover
  /-
    🎉 no goals
  -/


lemma Functor.isCoverDense_of_generate_singleton_functor_π_mem (G : C ⥤ D)
    (K : GrothendieckTopology D)
    (h : ∀ B, ∃ (X : C) (f : G.obj X ⟶ B), Sieve.generate (Presieve.singleton f) ∈ K B) :
    G.IsCoverDense K where
  is_cover B := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{u_5, u_2} D
      G : CategoryTheory.Functor C D
      K : CategoryTheory.GrothendieckTopology D
      h : ∀ (B : D), Exists fun X => Exists fun f => Membership.mem (K B) (CategoryT …
      B : D
      ⊢ Membership.mem (K B) (CategoryTheory.Sieve.coverByImage G B)
    -/
    obtain ⟨X, f, h⟩ := h B
    /-
      case intro.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{u_5, u_2} D
      G : CategoryTheory.Functor C D
      K : CategoryTheory.GrothendieckTopology D
      h✝ : ∀ (B : D), Exists fun X => Exists fun f => Membership.mem (K B) (Category …
      B : D
      X : C
      f : Quiver.Hom (G.obj X) B
      h : Membership.mem (K B) (CategoryTheory.Sieve.generate (CategoryTheory.Presie …
      ⊢ Membership.mem (K B) (CategoryTheory.Sieve.coverByImage G B)
    -/
    refine K.superset_covering ?_ h
    /-
      case intro.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{u_5, u_2} D
      G : CategoryTheory.Functor C D
      K : CategoryTheory.GrothendieckTopology D
      h✝ : ∀ (B : D), Exists fun X => Exists fun f => Membership.mem (K B) (Category …
      B : D
      X : C
      f : Quiver.Hom (G.obj X) B
      h : Membership.mem (K B) (CategoryTheory.Sieve.generate (CategoryTheory.Presie …
      ⊢ LE.le (CategoryTheory.Sieve.generate (CategoryTheory.Presieve.singleton f))  …
    -/
    intro Y f ⟨Z, g, _, h, w⟩
    /-
      case intro.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{u_5, u_2} D
      G : CategoryTheory.Functor C D
      K : CategoryTheory.GrothendieckTopology D
      h✝¹ : ∀ (B : D), Exists fun X => Exists fun f => Membership.mem (K B) (Categor …
      B : D
      X : C
      f✝ : Quiver.Hom (G.obj X) B
      h✝ : Membership.mem (K B) (CategoryTheory.Sieve.generate (CategoryTheory.Presi …
      Y : D
      f : Quiver.Hom Y B
      Z : D
      g : Quiver.Hom Y Z
      w✝ : Quiver.Hom Z B
      h : CategoryTheory.Presieve.singleton f✝ w✝
      w : Eq (CategoryTheory.CategoryStruct.comp g w✝) f
      ⊢ (CategoryTheory.Sieve.coverByImage G B).arrows f
    -/
    cases h
    /-
      case intro.intro.mk
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝ : CategoryTheory.Category.{u_5, u_2} D
      G : CategoryTheory.Functor C D
      K : CategoryTheory.GrothendieckTopology D
      h✝ : ∀ (B : D), Exists fun X => Exists fun f => Membership.mem (K B) (Category …
      B : D
      X : C
      f✝ : Quiver.Hom (G.obj X) B
      h : Membership.mem (K B) (CategoryTheory.Sieve.generate (CategoryTheory.Presie …
      Y : D
      f : Quiver.Hom Y B
      g : Quiver.Hom Y (G.obj X)
      w : Eq (CategoryTheory.CategoryStruct.comp g f✝) f
      ⊢ (CategoryTheory.Sieve.coverByImage G B).arrows f
    -/
    exact ⟨⟨_, g, _, w⟩⟩
    /-
      🎉 no goals
    -/


theorem ext [G.IsCoverDense K] (ℱ : Sheaf K (Type _)) (X : D) {s t : ℱ.val.obj (op X)}
    (h : ∀ ⦃Y : C⦄ (f : G.obj Y ⟶ X), ℱ.val.map f.op s = ℱ.val.map f.op t) : s = t := by
  apply ((isSheaf_iff_isSheaf_of_type _ _ ).1 ℱ.cond
    (Sieve.coverByImage G X) (G.is_cover_of_isCoverDense K X)).isSeparatedFor.ext
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝ : G.IsCoverDense K
    ℱ : CategoryTheory.Sheaf K (Type u_7)
    X : D
    s t : ℱ.val.obj { unop := X }
    h : ∀ ⦃Y : C⦄ (f : Quiver.Hom (G.obj Y) X), Eq (ℱ.val.map f.op s) (ℱ.val.map f …
    ⊢ ∀ ⦃Y : D⦄ ⦃f : Quiver.Hom Y X⦄, (CategoryTheory.Sieve.coverByImage G X).arro …
  -/
  rintro Y _ ⟨Z, f₁, f₂, ⟨rfl⟩⟩
  /-
    case intro.mk.refl
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝ : G.IsCoverDense K
    ℱ : CategoryTheory.Sheaf K (Type u_7)
    X : D
    s t : ℱ.val.obj { unop := X }
    h : ∀ ⦃Y : C⦄ (f : Quiver.Hom (G.obj Y) X), Eq (ℱ.val.map f.op s) (ℱ.val.map f …
    Y : D
    Z : C
    f₁ : Quiver.Hom Y (G.obj Z)
    f₂ : Quiver.Hom (G.obj Z) X
    ⊢ Eq (ℱ.val.map (CategoryTheory.CategoryStruct.comp f₁ f₂).op s) (ℱ.val.map (C …
  -/
  simp [h f₂]
  /-
    🎉 no goals
  -/


theorem functorPullback_pushforward_covering [G.IsCoverDense K] [G.IsLocallyFull K] {X : C}
    (T : K (G.obj X)) : (T.val.functorPullback G).functorPushforward G ∈ K (G.obj X) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    X : C
    T : ↑(K (G.obj X))
    ⊢ Membership.mem (K (G.obj X)) (CategoryTheory.Sieve.functorPushforward G (Cat …
  -/
  refine K.transitive T.2 _ fun Y iYX hiYX ↦ ?_
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    X : C
    T : ↑(K (G.obj X))
    Y : D
    iYX : Quiver.Hom Y (G.obj X)
    hiYX : (↑T).arrows iYX
    ⊢ Membership.mem (K Y) (CategoryTheory.Sieve.pullback iYX (CategoryTheory.Siev …
  -/
  apply K.transitive (G.is_cover_of_isCoverDense _ _) _
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    X : C
    T : ↑(K (G.obj X))
    Y : D
    iYX : Quiver.Hom Y (G.obj X)
    hiYX : (↑T).arrows iYX
    ⊢ ∀ ⦃Y_1 : D⦄ ⦃f : Quiver.Hom Y_1 Y⦄, (CategoryTheory.Sieve.coverByImage G Y). …
  -/
  rintro W _ ⟨Z, iWZ, iZY, rfl⟩
  /-
    case intro.mk
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    X : C
    T : ↑(K (G.obj X))
    Y : D
    iYX : Quiver.Hom Y (G.obj X)
    hiYX : (↑T).arrows iYX
    W : D
    Z : C
    iWZ : Quiver.Hom W (G.obj Z)
    iZY : Quiver.Hom (G.obj Z) Y
    ⊢ Membership.mem (K W) (CategoryTheory.Sieve.pullback (CategoryTheory.Category …
  -/
  rw [Sieve.pullback_comp]; apply K.pullback_stable; clear W iWZ
  /-
    case intro.mk.hS
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    X : C
    T : ↑(K (G.obj X))
    Y : D
    iYX : Quiver.Hom Y (G.obj X)
    hiYX : (↑T).arrows iYX
    Z : C
    iZY : Quiver.Hom (G.obj Z) Y
    ⊢ Membership.mem (K (G.obj Z)) (CategoryTheory.Sieve.pullback iZY (CategoryThe …
  -/
  apply K.superset_covering ?_ (G.functorPushforward_imageSieve_mem _ (iZY ≫ iYX))
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    X : C
    T : ↑(K (G.obj X))
    Y : D
    iYX : Quiver.Hom Y (G.obj X)
    hiYX : (↑T).arrows iYX
    Z : C
    iZY : Quiver.Hom (G.obj Z) Y
    ⊢ LE.le (CategoryTheory.Sieve.functorPushforward G (G.imageSieve (CategoryTheo …
  -/
  rintro W _ ⟨V, iVZ, iWV, ⟨iVX, e⟩, rfl⟩
  /-
    case intro.intro.intro.intro.intro
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    X : C
    T : ↑(K (G.obj X))
    Y : D
    iYX : Quiver.Hom Y (G.obj X)
    hiYX : (↑T).arrows iYX
    Z : C
    iZY : Quiver.Hom (G.obj Z) Y
    W : D
    V : C
    iVZ : Quiver.Hom V Z
    iWV : Quiver.Hom W (G.obj V)
    iVX : Quiver.Hom V X
    e : Eq (G.map iVX) (CategoryTheory.CategoryStruct.comp (G.map iVZ) (CategoryTh …
    ⊢ (CategoryTheory.Sieve.pullback iZY (CategoryTheory.Sieve.pullback iYX (Categ …
  -/
  exact ⟨_, iVX, iWV, by simpa [e] using T.1.downward_closed hiYX (G.map iVZ ≫ iZY), by simp [e]⟩
  /-
    🎉 no goals
  -/


/-- (Implementation). Given a hom between the pullbacks of two sheaves, we can whisker it with
`coyoneda` to obtain a hom between the pullbacks of the sheaves of maps from `X`.
-/
@[simps!]
def homOver {ℱ : Dᵒᵖ ⥤ A} {ℱ' : Sheaf K A} (α : G.op ⋙ ℱ ⟶ G.op ⋙ ℱ'.val) (X : A) :
    G.op ⋙ ℱ ⋙ coyoneda.obj (op X) ⟶ G.op ⋙ (sheafOver ℱ' X).val :=
  whiskerRight α (coyoneda.obj (op X))


/-- (Implementation). Given an iso between the pullbacks of two sheaves, we can whisker it with
`coyoneda` to obtain an iso between the pullbacks of the sheaves of maps from `X`.
-/
@[simps!]
def isoOver {ℱ ℱ' : Sheaf K A} (α : G.op ⋙ ℱ.val ≅ G.op ⋙ ℱ'.val) (X : A) :
    G.op ⋙ (sheafOver ℱ X).val ≅ G.op ⋙ (sheafOver ℱ' X).val :=
  isoWhiskerRight α (coyoneda.obj (op X))


theorem sheaf_eq_amalgamation (ℱ : Sheaf K A) {X : A} {U : D} {T : Sieve U} (hT)
    (x : FamilyOfElements _ T) (hx) (t) (h : x.IsAmalgamation t) :
    t = (ℱ.cond X T hT).amalgamate x hx :=
  (ℱ.cond X T hT).isSeparatedFor x t _ h ((ℱ.cond X T hT).isAmalgamation hx)


theorem naturality_apply [G.IsLocallyFull K] {X Y : C} (i : G.obj X ⟶ G.obj Y) (x) :
    ℱ'.1.map i.op (α.app _ x) = α.app _ (ℱ.map i.op x) := by
  have {X Y} (i : X ⟶ Y) (x) :
      ℱ'.1.map (G.map i).op (α.app _ x) = α.app _ (ℱ.map (G.map i).op x) := by
    exact congr_fun (α.naturality i.op).symm x
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    ℱ : CategoryTheory.Functor (Opposite D) (Type v)
    ℱ' : CategoryTheory.Sheaf K (Type v)
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    inst✝ : G.IsLocallyFull K
    X Y : C
    i : Quiver.Hom (G.obj X) (G.obj Y)
    x : (G.op.comp ℱ).obj { unop := Y }
    this : ∀ {X Y : C} (i : Quiver.Hom X Y) (x : (G.op.comp ℱ).obj { unop := Y }), …
    ⊢ Eq (ℱ'.val.map i.op (α.app { unop := Y } x)) (α.app { unop := X } (ℱ.map i.o …
  -/
  refine IsLocallyFull.ext G _ i fun V iVX iVY e ↦ ?_
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    ℱ : CategoryTheory.Functor (Opposite D) (Type v)
    ℱ' : CategoryTheory.Sheaf K (Type v)
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    inst✝ : G.IsLocallyFull K
    X Y : C
    i : Quiver.Hom (G.obj X) (G.obj Y)
    x : (G.op.comp ℱ).obj { unop := Y }
    this : ∀ {X Y : C} (i : Quiver.Hom X Y) (x : (G.op.comp ℱ).obj { unop := Y }), …
    V : C
    iVX : Quiver.Hom V X
    iVY : Quiver.Hom V Y
    e : Eq (G.map iVY) (CategoryTheory.CategoryStruct.comp (G.map iVX) i)
    ⊢ Eq (ℱ'.val.map (G.map iVX).op (ℱ'.val.map i.op (α.app { unop := Y } x))) (ℱ' …
  -/
  simp only [comp_obj, types_comp_apply, ← FunctorToTypes.map_comp_apply, ← op_comp, ← e, this]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem naturality [G.IsLocallyFull K] {X Y : C} (i : G.obj X ⟶ G.obj Y) :
    α.app _ ≫ ℱ'.1.map i.op = ℱ.map i.op ≫ α.app _ := types_ext _ _ (naturality_apply α i)


/--
(Implementation). Given a section of `ℱ` on `X`, we can obtain a family of elements valued in `ℱ'`
that is defined on a cover generated by the images of `G`. -/
noncomputable def pushforwardFamily {X} (x : ℱ.obj (op X)) :
    FamilyOfElements ℱ'.val (coverByImage G X) := fun _ _ hf =>
  ℱ'.val.map hf.some.lift.op <| α.app (op _) (ℱ.map hf.some.map.op x : _)


@[simp] theorem pushforwardFamily_def {X} (x : ℱ.obj (op X)) :
    pushforwardFamily α x = fun _ _ hf =>
  ℱ'.val.map hf.some.lift.op <| α.app (op _) (ℱ.map hf.some.map.op x : _) := rfl


@[simp]
theorem pushforwardFamily_apply [G.IsLocallyFull K]
    {X} (x : ℱ.obj (op X)) {Y : C} (f : G.obj Y ⟶ X) :
    pushforwardFamily α x f (Presieve.in_coverByImage G f) = α.app (op Y) (ℱ.map f.op x) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    ℱ : CategoryTheory.Functor (Opposite D) (Type v)
    ℱ' : CategoryTheory.Sheaf K (Type v)
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    inst✝ : G.IsLocallyFull K
    X : D
    x : ℱ.obj { unop := X }
    Y : C
    f : Quiver.Hom (G.obj Y) X
    ⊢ Eq (CategoryTheory.Functor.IsCoverDense.Types.pushforwardFamily α x f ⋯) (α. …
  -/
  simp only [pushforwardFamily_def, op_obj]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    ℱ : CategoryTheory.Functor (Opposite D) (Type v)
    ℱ' : CategoryTheory.Sheaf K (Type v)
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    inst✝ : G.IsLocallyFull K
    X : D
    x : ℱ.obj { unop := X }
    Y : C
    f : Quiver.Hom (G.obj Y) X
    ⊢ Eq (ℱ'.val.map (Nonempty.some ⋯).lift.op (α.app { unop := (Nonempty.some ⋯). …
  -/
  generalize Nonempty.some (Presieve.in_coverByImage G f) = l
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    ℱ : CategoryTheory.Functor (Opposite D) (Type v)
    ℱ' : CategoryTheory.Sheaf K (Type v)
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    inst✝ : G.IsLocallyFull K
    X : D
    x : ℱ.obj { unop := X }
    Y : C
    f : Quiver.Hom (G.obj Y) X
    l : CategoryTheory.Presieve.CoverByImageStructure G f
    ⊢ Eq (ℱ'.val.map l.lift.op (α.app { unop := l.1 } (ℱ.map l.map.op x))) (α.app  …
  -/
  obtain ⟨W, iYW, iWX, rfl⟩ := l
  /-
    case mk
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    ℱ : CategoryTheory.Functor (Opposite D) (Type v)
    ℱ' : CategoryTheory.Sheaf K (Type v)
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    inst✝ : G.IsLocallyFull K
    X : D
    x : ℱ.obj { unop := X }
    Y W : C
    iYW : Quiver.Hom (G.obj Y) (G.obj W)
    iWX : Quiver.Hom (G.obj W) X
    ⊢ Eq (ℱ'.val.map { obj := W, lift := iYW, map := iWX, fac := ⋯ }.lift.op (α.ap …
  -/
  simp only [← op_comp, ← FunctorToTypes.map_comp_apply, naturality_apply]
  /-
    🎉 no goals
  -/


/-- (Implementation). The `pushforwardFamily` defined is compatible. -/
theorem pushforwardFamily_compatible {X} (x : ℱ.obj (op X)) :
    (pushforwardFamily α x).Compatible := by
  suffices ∀ {Z W₁ W₂} (iWX₁ : G.obj W₁ ⟶ X) (iWX₂ : G.obj W₂ ⟶ X) (iZW₁ : Z ⟶ G.obj W₁)
      (iZW₂ : Z ⟶ G.obj W₂), iZW₁ ≫ iWX₁ = iZW₂ ≫ iWX₂ →
      ℱ'.1.map iZW₁.op (α.app _ (ℱ.map iWX₁.op x)) = ℱ'.1.map iZW₂.op (α.app _ (ℱ.map iWX₂.op x)) by
    rintro Y₁ Y₂ Z iZY₁ iZY₂ f₁ f₂ h₁ h₂ e
    simp only [pushforwardFamily, ← FunctorToTypes.map_comp_apply, ← op_comp]
    generalize Nonempty.some h₁ = l₁
    generalize Nonempty.some h₂ = l₂
    obtain ⟨W₁, iYW₁, iWX₁, rfl⟩ := l₁
    obtain ⟨W₂, iYW₂, iWX₂, rfl⟩ := l₂
    exact this _ _ _ _ (by simpa only [Category.assoc] using e)
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    ℱ : CategoryTheory.Functor (Opposite D) (Type v)
    ℱ' : CategoryTheory.Sheaf K (Type v)
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    X : D
    x : ℱ.obj { unop := X }
    ⊢ ∀ {Z : D} {W₁ W₂ : C} (iWX₁ : Quiver.Hom (G.obj W₁) X) (iWX₂ : Quiver.Hom (G …
  -/
  introv e
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    ℱ : CategoryTheory.Functor (Opposite D) (Type v)
    ℱ' : CategoryTheory.Sheaf K (Type v)
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    X : D
    x : ℱ.obj { unop := X }
    Z : D
    W₁ W₂ : C
    iWX₁ : Quiver.Hom (G.obj W₁) X
    iWX₂ : Quiver.Hom (G.obj W₂) X
    iZW₁ : Quiver.Hom Z (G.obj W₁)
    iZW₂ : Quiver.Hom Z (G.obj W₂)
    e : Eq (CategoryTheory.CategoryStruct.comp iZW₁ iWX₁) (CategoryTheory.Category …
    ⊢ Eq (ℱ'.val.map iZW₁.op (α.app { unop := W₁ } (ℱ.map iWX₁.op x))) (ℱ'.val.map …
  -/
  refine ext G _ _ fun V iVZ ↦ ?_
  simp only [← op_comp, ← FunctorToTypes.map_comp_apply, ← Functor.map_comp, naturality_apply,
    Category.assoc, e]


/-- (Implementation). The morphism `ℱ(X) ⟶ ℱ'(X)` given by gluing the `pushforwardFamily`. -/
noncomputable def appHom (X : D) : ℱ.obj (op X) ⟶ ℱ'.val.obj (op X) := fun x =>
  ((isSheaf_iff_isSheaf_of_type _ _ ).1 ℱ'.cond _
    (G.is_cover_of_isCoverDense _ X)).amalgamate (pushforwardFamily α x)
      (pushforwardFamily_compatible α x)


@[simp]
theorem appHom_restrict {X : D} {Y : C} (f : op X ⟶ op (G.obj Y)) (x) :
    ℱ'.val.map f (appHom α X x) = α.app (op Y) (ℱ.map f x) :=
  (((isSheaf_iff_isSheaf_of_type _ _ ).1 ℱ'.cond _ (G.is_cover_of_isCoverDense _ X)).valid_glue
      (pushforwardFamily_compatible α x) f.unop
          (Presieve.in_coverByImage G f.unop)).trans (pushforwardFamily_apply _ _ _)


@[simp]
theorem appHom_valid_glue {X : D} {Y : C} (f : op X ⟶ op (G.obj Y)) :
    appHom α X ≫ ℱ'.val.map f = ℱ.map f ≫ α.app (op Y) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    ℱ : CategoryTheory.Functor (Opposite D) (Type v)
    ℱ' : CategoryTheory.Sheaf K (Type v)
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    X : D
    Y : C
    f : Quiver.Hom { unop := X } { unop := G.obj Y }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCoverDense. …
  -/
  ext
  /-
    case h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    ℱ : CategoryTheory.Functor (Opposite D) (Type v)
    ℱ' : CategoryTheory.Sheaf K (Type v)
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    X : D
    Y : C
    f : Quiver.Hom { unop := X } { unop := G.obj Y }
    a✝ : ℱ.obj { unop := X }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCoverDense. …
  -/
  apply appHom_restrict
  /-
    🎉 no goals
  -/


/--
(Implementation). The maps given in `appIso` is inverse to each other and gives a `ℱ(X) ≅ ℱ'(X)`.
-/
@[simps]
noncomputable def appIso {ℱ ℱ' : Sheaf K (Type v)} (i : G.op ⋙ ℱ.val ≅ G.op ⋙ ℱ'.val)
    (X : D) : ℱ.val.obj (op X) ≅ ℱ'.val.obj (op X) where
  hom := appHom i.hom X
  inv := appHom i.inv X
  hom_inv_id := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.53687, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.53694, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.53701, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.53756, u_4} A
      G : CategoryTheory.Functor C D
      ℱ✝ : CategoryTheory.Functor (Opposite D) (Type v)
      ℱ'✝ : CategoryTheory.Sheaf K (Type v)
      α : Quiver.Hom (G.op.comp ℱ✝) (G.op.comp ℱ'✝.val)
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ ℱ' : CategoryTheory.Sheaf K (Type v)
      i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
      X : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCoverDense. …
    -/
    ext x
    /-
      case h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.53687, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.53694, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.53701, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.53756, u_4} A
      G : CategoryTheory.Functor C D
      ℱ✝ : CategoryTheory.Functor (Opposite D) (Type v)
      ℱ'✝ : CategoryTheory.Sheaf K (Type v)
      α : Quiver.Hom (G.op.comp ℱ✝) (G.op.comp ℱ'✝.val)
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ ℱ' : CategoryTheory.Sheaf K (Type v)
      i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
      X : D
      x : ℱ.val.obj { unop := X }
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCoverDense. …
    -/
    apply Functor.IsCoverDense.ext G
    /-
      case h.h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.53687, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.53694, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.53701, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.53756, u_4} A
      G : CategoryTheory.Functor C D
      ℱ✝ : CategoryTheory.Functor (Opposite D) (Type v)
      ℱ'✝ : CategoryTheory.Sheaf K (Type v)
      α : Quiver.Hom (G.op.comp ℱ✝) (G.op.comp ℱ'✝.val)
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ ℱ' : CategoryTheory.Sheaf K (Type v)
      i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
      X : D
      x : ℱ.val.obj { unop := X }
      ⊢ ∀ ⦃Y : C⦄ (f : Quiver.Hom (G.obj Y) X), Eq (ℱ.val.map f.op (CategoryTheory.C …
    -/
    intro Y f
    /-
      case h.h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.53687, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.53694, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.53701, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.53756, u_4} A
      G : CategoryTheory.Functor C D
      ℱ✝ : CategoryTheory.Functor (Opposite D) (Type v)
      ℱ'✝ : CategoryTheory.Sheaf K (Type v)
      α : Quiver.Hom (G.op.comp ℱ✝) (G.op.comp ℱ'✝.val)
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ ℱ' : CategoryTheory.Sheaf K (Type v)
      i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
      X : D
      x : ℱ.val.obj { unop := X }
      Y : C
      f : Quiver.Hom (G.obj Y) X
      ⊢ Eq (ℱ.val.map f.op (CategoryTheory.CategoryStruct.comp (CategoryTheory.Funct …
    -/
    simp
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.53687, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.53694, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.53701, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.53756, u_4} A
      G : CategoryTheory.Functor C D
      ℱ✝ : CategoryTheory.Functor (Opposite D) (Type v)
      ℱ'✝ : CategoryTheory.Sheaf K (Type v)
      α : Quiver.Hom (G.op.comp ℱ✝) (G.op.comp ℱ'✝.val)
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ ℱ' : CategoryTheory.Sheaf K (Type v)
      i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
      X : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCoverDense. …
    -/
    ext x
    /-
      case h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.53687, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.53694, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.53701, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.53756, u_4} A
      G : CategoryTheory.Functor C D
      ℱ✝ : CategoryTheory.Functor (Opposite D) (Type v)
      ℱ'✝ : CategoryTheory.Sheaf K (Type v)
      α : Quiver.Hom (G.op.comp ℱ✝) (G.op.comp ℱ'✝.val)
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ ℱ' : CategoryTheory.Sheaf K (Type v)
      i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
      X : D
      x : ℱ'.val.obj { unop := X }
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCoverDense. …
    -/
    apply Functor.IsCoverDense.ext G
    /-
      case h.h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.53687, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.53694, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.53701, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.53756, u_4} A
      G : CategoryTheory.Functor C D
      ℱ✝ : CategoryTheory.Functor (Opposite D) (Type v)
      ℱ'✝ : CategoryTheory.Sheaf K (Type v)
      α : Quiver.Hom (G.op.comp ℱ✝) (G.op.comp ℱ'✝.val)
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ ℱ' : CategoryTheory.Sheaf K (Type v)
      i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
      X : D
      x : ℱ'.val.obj { unop := X }
      ⊢ ∀ ⦃Y : C⦄ (f : Quiver.Hom (G.obj Y) X), Eq (ℱ'.val.map f.op (CategoryTheory. …
    -/
    intro Y f
    /-
      case h.h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.53687, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.53694, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.53701, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.53756, u_4} A
      G : CategoryTheory.Functor C D
      ℱ✝ : CategoryTheory.Functor (Opposite D) (Type v)
      ℱ'✝ : CategoryTheory.Sheaf K (Type v)
      α : Quiver.Hom (G.op.comp ℱ✝) (G.op.comp ℱ'✝.val)
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ ℱ' : CategoryTheory.Sheaf K (Type v)
      i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
      X : D
      x : ℱ'.val.obj { unop := X }
      Y : C
      f : Quiver.Hom (G.obj Y) X
      ⊢ Eq (ℱ'.val.map f.op (CategoryTheory.CategoryStruct.comp (CategoryTheory.Func …
    -/
    simp
    /-
      🎉 no goals
    -/


/--
Given a natural transformation `G ⋙ ℱ ⟶ G ⋙ ℱ'` between presheaves of types,
where `G` is locally-full and cover-dense, and `ℱ'` is a sheaf,
we may obtain a natural transformation between sheaves.
-/
@[simps]
noncomputable def presheafHom (α : G.op ⋙ ℱ ⟶ G.op ⋙ ℱ'.val) : ℱ ⟶ ℱ'.val where
  app X := appHom α (unop X)
  naturality X Y f := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.58770, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.58777, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.58784, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.58839, u_4} A
      G : CategoryTheory.Functor C D
      ℱ : CategoryTheory.Functor (Opposite D) (Type v)
      ℱ' : CategoryTheory.Sheaf K (Type v)
      α✝ : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      X Y : Opposite D
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (ℱ.map f) ((fun X => CategoryTheory.F …
    -/
    ext x
    /-
      case h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.58770, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.58777, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.58784, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.58839, u_4} A
      G : CategoryTheory.Functor C D
      ℱ : CategoryTheory.Functor (Opposite D) (Type v)
      ℱ' : CategoryTheory.Sheaf K (Type v)
      α✝ : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      X Y : Opposite D
      f : Quiver.Hom X Y
      x : ℱ.obj X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (ℱ.map f) ((fun X => CategoryTheory.F …
    -/
    apply Functor.IsCoverDense.ext G
    /-
      case h.h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.58770, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.58777, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.58784, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.58839, u_4} A
      G : CategoryTheory.Functor C D
      ℱ : CategoryTheory.Functor (Opposite D) (Type v)
      ℱ' : CategoryTheory.Sheaf K (Type v)
      α✝ : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      X Y : Opposite D
      f : Quiver.Hom X Y
      x : ℱ.obj X
      ⊢ ∀ ⦃Y_1 : C⦄ (f_1 : Quiver.Hom (G.obj Y_1) (Opposite.unop Y)), Eq (ℱ'.val.map …
    -/
    intro Y' f'
    /-
      case h.h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.58770, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.58777, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.58784, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.58839, u_4} A
      G : CategoryTheory.Functor C D
      ℱ : CategoryTheory.Functor (Opposite D) (Type v)
      ℱ' : CategoryTheory.Sheaf K (Type v)
      α✝ : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      X Y : Opposite D
      f : Quiver.Hom X Y
      x : ℱ.obj X
      Y' : C
      f' : Quiver.Hom (G.obj Y') (Opposite.unop Y)
      ⊢ Eq (ℱ'.val.map f'.op (CategoryTheory.CategoryStruct.comp (ℱ.map f) ((fun X = …
    -/
    simp only [appHom_restrict, types_comp_apply, ← FunctorToTypes.map_comp_apply]
    /-
      🎉 no goals
    -/


/--
Given a natural isomorphism `G ⋙ ℱ ≅ G ⋙ ℱ'` between presheaves of types,
where `G` is locally-full and cover-dense, and `ℱ, ℱ'` are sheaves,
we may obtain a natural isomorphism between presheaves.
-/
@[simps!]
noncomputable def presheafIso {ℱ ℱ' : Sheaf K (Type v)} (i : G.op ⋙ ℱ.val ≅ G.op ⋙ ℱ'.val) :
    ℱ.val ≅ ℱ'.val :=
  NatIso.ofComponents (fun X => appIso i (unop X)) @(presheafHom i.hom).naturality


/--
Given a natural isomorphism `G ⋙ ℱ ≅ G ⋙ ℱ'` between presheaves of types,
where `G` is locally-full and cover-dense, and `ℱ, ℱ'` are sheaves,
we may obtain a natural isomorphism between sheaves.
-/
@[simps]
noncomputable def sheafIso {ℱ ℱ' : Sheaf K (Type v)} (i : G.op ⋙ ℱ.val ≅ G.op ⋙ ℱ'.val) :
    ℱ ≅ ℱ' where
  hom := ⟨(presheafIso i).hom⟩
  inv := ⟨(presheafIso i).inv⟩
  hom_inv_id := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.66522, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.66529, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.66536, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.66591, u_4} A
      G : CategoryTheory.Functor C D
      ℱ✝ : CategoryTheory.Functor (Opposite D) (Type v)
      ℱ'✝ : CategoryTheory.Sheaf K (Type v)
      α : Quiver.Hom (G.op.comp ℱ✝) (G.op.comp ℱ'✝.val)
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ ℱ' : CategoryTheory.Sheaf K (Type v)
      i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { val := (CategoryTheory.Functor.IsCo …
    -/
    ext1
    /-
      case h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.66522, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.66529, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.66536, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.66591, u_4} A
      G : CategoryTheory.Functor C D
      ℱ✝ : CategoryTheory.Functor (Opposite D) (Type v)
      ℱ'✝ : CategoryTheory.Sheaf K (Type v)
      α : Quiver.Hom (G.op.comp ℱ✝) (G.op.comp ℱ'✝.val)
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ ℱ' : CategoryTheory.Sheaf K (Type v)
      i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { val := (CategoryTheory.Functor.IsCo …
    -/
    apply (presheafIso i).hom_inv_id
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.66522, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.66529, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.66536, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.66591, u_4} A
      G : CategoryTheory.Functor C D
      ℱ✝ : CategoryTheory.Functor (Opposite D) (Type v)
      ℱ'✝ : CategoryTheory.Sheaf K (Type v)
      α : Quiver.Hom (G.op.comp ℱ✝) (G.op.comp ℱ'✝.val)
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ ℱ' : CategoryTheory.Sheaf K (Type v)
      i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { val := (CategoryTheory.Functor.IsCo …
    -/
    ext1
    /-
      case h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.66522, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.66529, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.66536, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.66591, u_4} A
      G : CategoryTheory.Functor C D
      ℱ✝ : CategoryTheory.Functor (Opposite D) (Type v)
      ℱ'✝ : CategoryTheory.Sheaf K (Type v)
      α : Quiver.Hom (G.op.comp ℱ✝) (G.op.comp ℱ'✝.val)
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ ℱ' : CategoryTheory.Sheaf K (Type v)
      i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { val := (CategoryTheory.Functor.IsCo …
    -/
    apply (presheafIso i).inv_hom_id
    /-
      🎉 no goals
    -/


/-- (Implementation). The sheaf map given in `types.sheaf_hom` is natural in terms of `X`. -/
@[simps]
noncomputable def sheafCoyonedaHom (α : G.op ⋙ ℱ ⟶ G.op ⋙ ℱ'.val) :
    coyoneda ⋙ (whiskeringLeft Dᵒᵖ A (Type _)).obj ℱ ⟶
      coyoneda ⋙ (whiskeringLeft Dᵒᵖ A (Type _)).obj ℱ'.val where
  app X := presheafHom (homOver α (unop X))
  naturality X Y f := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.71342, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.71349, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.71356, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.74559, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ : CategoryTheory.Functor (Opposite D) A
      ℱ' : CategoryTheory.Sheaf K A
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      X Y : Opposite A
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.coyoneda.comp ((Cate …
    -/
    ext U x
    change
      appHom (homOver α (unop Y)) (unop U) (f.unop ≫ x) =
        f.unop ≫ appHom (homOver α (unop X)) (unop U) x
    /-
      case w.h.h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.71342, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.71349, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.71356, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.74559, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ : CategoryTheory.Functor (Opposite D) A
      ℱ' : CategoryTheory.Sheaf K A
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      X Y : Opposite A
      f : Quiver.Hom X Y
      U : Opposite D
      x : ((CategoryTheory.coyoneda.comp ((CategoryTheory.whiskeringLeft (Opposite D …
      ⊢ Eq (CategoryTheory.Functor.IsCoverDense.Types.appHom (CategoryTheory.Functor …
    -/
    symm
    /-
      case w.h.h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.71342, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.71349, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.71356, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.74559, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ : CategoryTheory.Functor (Opposite D) A
      ℱ' : CategoryTheory.Sheaf K A
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      X Y : Opposite A
      f : Quiver.Hom X Y
      U : Opposite D
      x : ((CategoryTheory.coyoneda.comp ((CategoryTheory.whiskeringLeft (Opposite D …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f.unop (CategoryTheory.Functor.IsCove …
    -/
    apply sheaf_eq_amalgamation
      /-
        case w.h.h.hT
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{?u.71342, u_1} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{?u.71349, u_2} D
        E : Type u_3
        inst✝³ : CategoryTheory.Category.{?u.71356, u_3} E
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        L : CategoryTheory.GrothendieckTopology E
        A : Type u_4
        inst✝² : CategoryTheory.Category.{?u.74559, u_4} A
        G : CategoryTheory.Functor C D
        inst✝¹ : G.IsCoverDense K
        inst✝ : G.IsLocallyFull K
        ℱ : CategoryTheory.Functor (Opposite D) A
        ℱ' : CategoryTheory.Sheaf K A
        α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
        X Y : Opposite A
        f : Quiver.Hom X Y
        U : Opposite D
        x : ((CategoryTheory.coyoneda.comp ((CategoryTheory.whiskeringLeft (Opposite D …
        ⊢ Membership.mem (K (Opposite.unop U)) (CategoryTheory.Sieve.coverByImage G (O …
      -/
    · apply G.is_cover_of_isCoverDense
      /-
        🎉 no goals
      -/
    -- Porting note: the following line closes a goal which didn't exist before reenableeta
      /-
        case w.h.h.hx
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{?u.71342, u_1} C
        D : Type u_2
        inst✝⁴ : CategoryTheory.Category.{?u.71349, u_2} D
        E : Type u_3
        inst✝³ : CategoryTheory.Category.{?u.71356, u_3} E
        J : CategoryTheory.GrothendieckTopology C
        K : CategoryTheory.GrothendieckTopology D
        L : CategoryTheory.GrothendieckTopology E
        A : Type u_4
        inst✝² : CategoryTheory.Category.{?u.74559, u_4} A
        G : CategoryTheory.Functor C D
        inst✝¹ : G.IsCoverDense K
        inst✝ : G.IsLocallyFull K
        ℱ : CategoryTheory.Functor (Opposite D) A
        ℱ' : CategoryTheory.Sheaf K A
        α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
        X Y : Opposite A
        f : Quiver.Hom X Y
        U : Opposite D
        x : ((CategoryTheory.coyoneda.comp ((CategoryTheory.whiskeringLeft (Opposite D …
        ⊢ (CategoryTheory.Functor.IsCoverDense.Types.pushforwardFamily (CategoryTheory …
      -/
    · exact pushforwardFamily_compatible (homOver α Y.unop) (f.unop ≫ x)
      /-
        🎉 no goals
      -/
    /-
      case w.h.h.h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.71342, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.71349, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.71356, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.74559, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ : CategoryTheory.Functor (Opposite D) A
      ℱ' : CategoryTheory.Sheaf K A
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      X Y : Opposite A
      f : Quiver.Hom X Y
      U : Opposite D
      x : ((CategoryTheory.coyoneda.comp ((CategoryTheory.whiskeringLeft (Opposite D …
      ⊢ (CategoryTheory.Functor.IsCoverDense.Types.pushforwardFamily (CategoryTheory …
    -/
    intro Y' f' hf'
    /-
      case w.h.h.h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.71342, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.71349, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.71356, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.74559, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ : CategoryTheory.Functor (Opposite D) A
      ℱ' : CategoryTheory.Sheaf K A
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      X Y : Opposite A
      f : Quiver.Hom X Y
      U : Opposite D
      x : ((CategoryTheory.coyoneda.comp ((CategoryTheory.whiskeringLeft (Opposite D …
      Y' : D
      f' : Quiver.Hom Y' (Opposite.unop U)
      hf' : (CategoryTheory.Sieve.coverByImage G (Opposite.unop U)).arrows f'
      ⊢ Eq ((ℱ'.val.comp (CategoryTheory.coyoneda.obj { unop := Opposite.unop Y })). …
    -/
    change unop X ⟶ ℱ.obj (op (unop _)) at x
    /-
      case w.h.h.h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.71342, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.71349, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.71356, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.74559, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ : CategoryTheory.Functor (Opposite D) A
      ℱ' : CategoryTheory.Sheaf K A
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      X Y : Opposite A
      f : Quiver.Hom X Y
      U : Opposite D
      Y' : D
      f' : Quiver.Hom Y' (Opposite.unop U)
      hf' : (CategoryTheory.Sieve.coverByImage G (Opposite.unop U)).arrows f'
      x : Quiver.Hom (Opposite.unop X) (ℱ.obj { unop := Opposite.unop U })
      ⊢ Eq ((ℱ'.val.comp (CategoryTheory.coyoneda.obj { unop := Opposite.unop Y })). …
    -/
    dsimp
    /-
      case w.h.h.h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.71342, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.71349, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.71356, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.74559, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ : CategoryTheory.Functor (Opposite D) A
      ℱ' : CategoryTheory.Sheaf K A
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      X Y : Opposite A
      f : Quiver.Hom X Y
      U : Opposite D
      Y' : D
      f' : Quiver.Hom Y' (Opposite.unop U)
      hf' : (CategoryTheory.Sieve.coverByImage G (Opposite.unop U)).arrows f'
      x : Quiver.Hom (Opposite.unop X) (ℱ.obj { unop := Opposite.unop U })
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    simp only [pushforwardFamily, Functor.comp_map, coyoneda_obj_map, homOver_app, Category.assoc]
    /-
      case w.h.h.h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.71342, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.71349, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.71356, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.74559, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ : CategoryTheory.Functor (Opposite D) A
      ℱ' : CategoryTheory.Sheaf K A
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      X Y : Opposite A
      f : Quiver.Hom X Y
      U : Opposite D
      Y' : D
      f' : Quiver.Hom Y' (Opposite.unop U)
      hf' : (CategoryTheory.Sieve.coverByImage G (Opposite.unop U)).arrows f'
      x : Quiver.Hom (Opposite.unop X) (ℱ.obj { unop := Opposite.unop U })
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f.unop (CategoryTheory.CategoryStruct …
    -/
    congr 1
    /-
      case w.h.h.h.e_a
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.71342, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.71349, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.71356, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.74559, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ : CategoryTheory.Functor (Opposite D) A
      ℱ' : CategoryTheory.Sheaf K A
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      X Y : Opposite A
      f : Quiver.Hom X Y
      U : Opposite D
      Y' : D
      f' : Quiver.Hom Y' (Opposite.unop U)
      hf' : (CategoryTheory.Sieve.coverByImage G (Opposite.unop U)).arrows f'
      x : Quiver.Hom (Opposite.unop X) (ℱ.obj { unop := Opposite.unop U })
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCoverDense. …
    -/
    conv_lhs => rw [← hf'.some.fac]
    /-
      case w.h.h.h.e_a
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.71342, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.71349, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.71356, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.74559, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ : CategoryTheory.Functor (Opposite D) A
      ℱ' : CategoryTheory.Sheaf K A
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      X Y : Opposite A
      f : Quiver.Hom X Y
      U : Opposite D
      Y' : D
      f' : Quiver.Hom Y' (Opposite.unop U)
      hf' : (CategoryTheory.Sieve.coverByImage G (Opposite.unop U)).arrows f'
      x : Quiver.Hom (Opposite.unop X) (ℱ.obj { unop := Opposite.unop U })
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCoverDense. …
    -/
    simp only [← Category.assoc, op_comp, Functor.map_comp]
    /-
      case w.h.h.h.e_a
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.71342, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.71349, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.71356, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.74559, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ : CategoryTheory.Functor (Opposite D) A
      ℱ' : CategoryTheory.Sheaf K A
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      X Y : Opposite A
      f : Quiver.Hom X Y
      U : Opposite D
      Y' : D
      f' : Quiver.Hom Y' (Opposite.unop U)
      hf' : (CategoryTheory.Sieve.coverByImage G (Opposite.unop U)).arrows f'
      x : Quiver.Hom (Opposite.unop X) (ℱ.obj { unop := Opposite.unop U })
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    congr 1
    /-
      case w.h.h.h.e_a.e_a
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.71342, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.71349, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.71356, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.74559, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ : CategoryTheory.Functor (Opposite D) A
      ℱ' : CategoryTheory.Sheaf K A
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      X Y : Opposite A
      f : Quiver.Hom X Y
      U : Opposite D
      Y' : D
      f' : Quiver.Hom Y' (Opposite.unop U)
      hf' : (CategoryTheory.Sieve.coverByImage G (Opposite.unop U)).arrows f'
      x : Quiver.Hom (Opposite.unop X) (ℱ.obj { unop := Opposite.unop U })
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.IsCoverDense. …
    -/
    exact (appHom_restrict (homOver α (unop X)) hf'.some.map.op x).trans (by simp)
    /-
      🎉 no goals
    -/


/--
(Implementation). `sheafCoyonedaHom` but the order of the arguments of the functor are swapped.
-/
noncomputable def sheafYonedaHom (α : G.op ⋙ ℱ ⟶ G.op ⋙ ℱ'.val) :
    ℱ ⋙ yoneda ⟶ ℱ'.val ⋙ yoneda where
  app U :=
    let α := (sheafCoyonedaHom α)
    { app := fun X => (α.app X).app U
                                    /-
                                      C : Type u_1
                                      inst✝⁵ : CategoryTheory.Category.{?u.80790, u_1} C
                                      D : Type u_2
                                      inst✝⁴ : CategoryTheory.Category.{?u.80797, u_2} D
                                      E : Type u_3
                                      inst✝³ : CategoryTheory.Category.{?u.80804, u_3} E
                                      J : CategoryTheory.GrothendieckTopology C
                                      K : CategoryTheory.GrothendieckTopology D
                                      L : CategoryTheory.GrothendieckTopology E
                                      A : Type u_4
                                      inst✝² : CategoryTheory.Category.{?u.80859, u_4} A
                                      G : CategoryTheory.Functor C D
                                      inst✝¹ : G.IsCoverDense K
                                      inst✝ : G.IsLocallyFull K
                                      ℱ : CategoryTheory.Functor (Opposite D) A
                                      ℱ' : CategoryTheory.Sheaf K A
                                      α✝ : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
                                      U : Opposite D
                                      α : Quiver.Hom (CategoryTheory.coyoneda.comp ((CategoryTheory.whiskeringLeft ( …
                                      X Y : Opposite A
                                      f : Quiver.Hom X Y
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((ℱ.comp CategoryTheory.yoneda).obj  …
                                    -/
      naturality := fun X Y f => by simpa using congr_app (α.naturality f) U }
                                    /-
                                      🎉 no goals
                                    -/
  naturality U V i := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.80790, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.80797, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.80804, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.80859, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ : CategoryTheory.Functor (Opposite D) A
      ℱ' : CategoryTheory.Sheaf K A
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      U V : Opposite D
      i : Quiver.Hom U V
      ⊢ Eq
          (CategoryTheory.CategoryStruct.comp ((ℱ.comp CategoryTheory.yoneda).map i)
            ((fun U =>
                let α := CategoryTheory.Functor.IsCoverDense.sheafCoyonedaHom α;
                { app := fun X => (α.app X).app U, naturality := ⋯ })
              V))
          (CategoryTheory.CategoryStruct.comp
            ((fun U =>
                let α := CategoryTheory.Functor.IsCoverDense.sheafCoyonedaHom α;
                { app := fun X => (α.app X).app U, naturality := ⋯ })
              U)
            ((ℱ'.val.comp CategoryTheory.yoneda).map i))
    -/
    ext X x
    /-
      case w.h.h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.80790, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.80797, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.80804, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.80859, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ : CategoryTheory.Functor (Opposite D) A
      ℱ' : CategoryTheory.Sheaf K A
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      U V : Opposite D
      i : Quiver.Hom U V
      X : Opposite A
      x : ((ℱ.comp CategoryTheory.yoneda).obj U).obj X
      ⊢ Eq
          ((CategoryTheory.CategoryStruct.comp ((ℱ.comp CategoryTheory.yoneda).map i)
                ((fun U =>
                    let α := CategoryTheory.Functor.IsCoverDense.sheafCoyonedaHom α;
                    { app := fun X => (α.app X).app U, naturality := ⋯ })
                  V)).app
            X x)
          ((CategoryTheory.CategoryStruct.comp
                ((fun U =>
                    let α := CategoryTheory.Functor.IsCoverDense.sheafCoyonedaHom α;
                    { app := fun X => (α.app X).app U, naturality := ⋯ })
                  U)
                ((ℱ'.val.comp CategoryTheory.yoneda).map i)).app
            X x)
    -/
    exact congr_fun (((sheafCoyonedaHom α).app X).naturality i) x
    /-
      🎉 no goals
    -/


/--
Given a natural transformation `G ⋙ ℱ ⟶ G ⋙ ℱ'` between presheaves of arbitrary category,
where `G` is locally-full and cover-dense, and `ℱ'` is a sheaf, we may obtain a natural
transformation between presheaves.
-/
noncomputable def sheafHom (α : G.op ⋙ ℱ ⟶ G.op ⋙ ℱ'.val) : ℱ ⟶ ℱ'.val :=
  let α' := sheafYonedaHom α
  { app := fun X => yoneda.preimage (α'.app X)
                                                        /-
                                                          C : Type u_1
                                                          inst✝⁵ : CategoryTheory.Category.{?u.90041, u_1} C
                                                          D : Type u_2
                                                          inst✝⁴ : CategoryTheory.Category.{?u.90048, u_2} D
                                                          E : Type u_3
                                                          inst✝³ : CategoryTheory.Category.{?u.90055, u_3} E
                                                          J : CategoryTheory.GrothendieckTopology C
                                                          K : CategoryTheory.GrothendieckTopology D
                                                          L : CategoryTheory.GrothendieckTopology E
                                                          A : Type u_4
                                                          inst✝² : CategoryTheory.Category.{?u.90110, u_4} A
                                                          G : CategoryTheory.Functor C D
                                                          inst✝¹ : G.IsCoverDense K
                                                          inst✝ : G.IsLocallyFull K
                                                          ℱ : CategoryTheory.Functor (Opposite D) A
                                                          ℱ' : CategoryTheory.Sheaf K A
                                                          α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
                                                          α' : Quiver.Hom (ℱ.comp CategoryTheory.yoneda) (ℱ'.val.comp CategoryTheory.yon …
                                                          X Y : Opposite D
                                                          f : Quiver.Hom X Y
                                                          ⊢ Eq (CategoryTheory.yoneda.map (CategoryTheory.CategoryStruct.comp (ℱ.map f)  …
                                                        -/
    naturality := fun X Y f => yoneda.map_injective (by simpa using α'.naturality f) }
                                                        /-
                                                          🎉 no goals
                                                        -/


/--
Given a natural isomorphism `G ⋙ ℱ ≅ G ⋙ ℱ'` between presheaves of arbitrary category,
where `G` is locally-full and cover-dense, and `ℱ', ℱ` are sheaves,
we may obtain a natural isomorphism between presheaves.
-/
@[simps!]
noncomputable def presheafIso {ℱ ℱ' : Sheaf K A} (i : G.op ⋙ ℱ.val ≅ G.op ⋙ ℱ'.val) :
    ℱ.val ≅ ℱ'.val := by
  have : ∀ X : Dᵒᵖ, IsIso ((sheafHom i.hom).app X) := by
    intro X
    rw [← isIso_iff_of_reflects_iso _ yoneda]
    use (sheafYonedaHom i.inv).app X
    constructor <;> ext x : 2 <;>
      simp only [sheafHom, NatTrans.comp_app, NatTrans.id_app, Functor.map_preimage]
    · exact ((Types.presheafIso (isoOver i (unop x))).app X).hom_inv_id
    · exact ((Types.presheafIso (isoOver i (unop x))).app X).inv_hom_id
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.97177, u_1} C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{?u.97184, u_2} D
    E : Type u_3
    inst✝³ : CategoryTheory.Category.{?u.97191, u_3} E
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    L : CategoryTheory.GrothendieckTopology E
    A : Type u_4
    inst✝² : CategoryTheory.Category.{?u.97246, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ✝ : CategoryTheory.Functor (Opposite D) A
    ℱ'✝ ℱ ℱ' : CategoryTheory.Sheaf K A
    i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
    this : ∀ (X : Opposite D), CategoryTheory.IsIso ((CategoryTheory.Functor.IsCov …
    ⊢ CategoryTheory.Iso ℱ.val ℱ'.val
  -/
  haveI : IsIso (sheafHom i.hom) := by apply NatIso.isIso_of_isIso_app
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.97177, u_1} C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{?u.97184, u_2} D
    E : Type u_3
    inst✝³ : CategoryTheory.Category.{?u.97191, u_3} E
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    L : CategoryTheory.GrothendieckTopology E
    A : Type u_4
    inst✝² : CategoryTheory.Category.{?u.97246, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ✝ : CategoryTheory.Functor (Opposite D) A
    ℱ'✝ ℱ ℱ' : CategoryTheory.Sheaf K A
    i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
    this✝ : ∀ (X : Opposite D), CategoryTheory.IsIso ((CategoryTheory.Functor.IsCo …
    this : CategoryTheory.IsIso (CategoryTheory.Functor.IsCoverDense.sheafHom i.hom)
    ⊢ CategoryTheory.Iso ℱ.val ℱ'.val
  -/
  apply asIso (sheafHom i.hom)
  /-
    🎉 no goals
  -/


/--
Given a natural isomorphism `G ⋙ ℱ ≅ G ⋙ ℱ'` between presheaves of arbitrary category,
where `G` is locally-full and cover-dense, and `ℱ', ℱ` are sheaves,
we may obtain a natural isomorphism between presheaves.
-/
@[simps]
noncomputable def sheafIso {ℱ ℱ' : Sheaf K A} (i : G.op ⋙ ℱ.val ≅ G.op ⋙ ℱ'.val) : ℱ ≅ ℱ' where
  hom := ⟨(presheafIso i).hom⟩
  inv := ⟨(presheafIso i).inv⟩
  hom_inv_id := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.104830, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.104837, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.104844, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.104899, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ✝ : CategoryTheory.Functor (Opposite D) A
      ℱ'✝ ℱ ℱ' : CategoryTheory.Sheaf K A
      i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { val := (CategoryTheory.Functor.IsCo …
    -/
    ext1
    /-
      case h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.104830, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.104837, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.104844, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.104899, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ✝ : CategoryTheory.Functor (Opposite D) A
      ℱ'✝ ℱ ℱ' : CategoryTheory.Sheaf K A
      i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { val := (CategoryTheory.Functor.IsCo …
    -/
    apply (presheafIso i).hom_inv_id
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.104830, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.104837, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.104844, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.104899, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ✝ : CategoryTheory.Functor (Opposite D) A
      ℱ'✝ ℱ ℱ' : CategoryTheory.Sheaf K A
      i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { val := (CategoryTheory.Functor.IsCo …
    -/
    ext1
    /-
      case h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.104830, u_1} C
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.104837, u_2} D
      E : Type u_3
      inst✝³ : CategoryTheory.Category.{?u.104844, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝² : CategoryTheory.Category.{?u.104899, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ✝ : CategoryTheory.Functor (Opposite D) A
      ℱ'✝ ℱ ℱ' : CategoryTheory.Sheaf K A
      i : CategoryTheory.Iso (G.op.comp ℱ.val) (G.op.comp ℱ'.val)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { val := (CategoryTheory.Functor.IsCo …
    -/
    apply (presheafIso i).inv_hom_id
    /-
      🎉 no goals
    -/


/-- The constructed `sheafHom α` is equal to `α` when restricted onto `C`. -/
theorem sheafHom_restrict_eq (α : G.op ⋙ ℱ ⟶ G.op ⋙ ℱ'.val) :
    whiskerLeft G.op (sheafHom α) = α := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    ⊢ Eq (CategoryTheory.whiskerLeft G.op (CategoryTheory.Functor.IsCoverDense.she …
  -/
  ext X
  /-
    case w.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    X : Opposite C
    ⊢ Eq ((CategoryTheory.whiskerLeft G.op (CategoryTheory.Functor.IsCoverDense.sh …
  -/
  apply yoneda.map_injective
  /-
    case w.h.a
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    X : Opposite C
    ⊢ Eq (CategoryTheory.yoneda.map ((CategoryTheory.whiskerLeft G.op (CategoryThe …
  -/
  ext U
  /-
    case w.h.a.w.h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    X : Opposite C
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj ((G.op.comp ℱ).obj X)).obj U
    ⊢ Eq ((CategoryTheory.yoneda.map ((CategoryTheory.whiskerLeft G.op (CategoryTh …
  -/
  erw [yoneda.map_preimage]
  /-
    case w.h.a.w.h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    X : Opposite C
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj ((G.op.comp ℱ).obj X)).obj U
    ⊢ Eq (((CategoryTheory.Functor.IsCoverDense.sheafYonedaHom α).app (G.op.obj X) …
  -/
  symm
  /-
    case w.h.a.w.h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    X : Opposite C
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj ((G.op.comp ℱ).obj X)).obj U
    ⊢ Eq ((CategoryTheory.yoneda.map (α.app X)).app U a✝) (((CategoryTheory.Functo …
  -/
  change (show (ℱ'.val ⋙ coyoneda.obj (op (unop U))).obj (op (G.obj (unop X))) from _) = _
  /-
    case w.h.a.w.h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    X : Opposite C
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj ((G.op.comp ℱ).obj X)).obj U
    ⊢ Eq (letFun ((CategoryTheory.yoneda.map (α.app X)).app U a✝) fun this => this …
  -/
  apply sheaf_eq_amalgamation ℱ' (G.is_cover_of_isCoverDense _ _)
  -- Porting note: next line was not needed in mathlib3
    /-
      case w.h.a.w.h.h.hx
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_7, u_2} D
      K : CategoryTheory.GrothendieckTopology D
      A : Type u_4
      inst✝² : CategoryTheory.Category.{u_5, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ : CategoryTheory.Functor (Opposite D) A
      ℱ' : CategoryTheory.Sheaf K A
      α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
      X : Opposite C
      U : Opposite A
      a✝ : (CategoryTheory.yoneda.obj ((G.op.comp ℱ).obj X)).obj U
      ⊢ (CategoryTheory.Functor.IsCoverDense.Types.pushforwardFamily (CategoryTheory …
    -/
  · exact (pushforwardFamily_compatible _ _)
    /-
      🎉 no goals
    -/
  /-
    case w.h.a.w.h.h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    X : Opposite C
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj ((G.op.comp ℱ).obj X)).obj U
    ⊢ (CategoryTheory.Functor.IsCoverDense.Types.pushforwardFamily (CategoryTheory …
  -/
  intro Y f hf
  /-
    case w.h.a.w.h.h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    X : Opposite C
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj ((G.op.comp ℱ).obj X)).obj U
    Y : D
    f : Quiver.Hom Y (G.obj (Opposite.unop X))
    hf : (CategoryTheory.Sieve.coverByImage G (G.obj (Opposite.unop X))).arrows f
    ⊢ Eq ((ℱ'.val.comp (CategoryTheory.coyoneda.obj { unop := Opposite.unop U })). …
  -/
  conv_lhs => rw [← hf.some.fac]
  simp only [pushforwardFamily, Functor.comp_map, yoneda_map_app, coyoneda_obj_map, op_comp,
    FunctorToTypes.map_comp_apply, homOver_app, ← Category.assoc]
  /-
    case w.h.a.w.h.h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    X : Opposite C
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj ((G.op.comp ℱ).obj X)).obj U
    Y : D
    f : Quiver.Hom Y (G.obj (Opposite.unop X))
    hf : (CategoryTheory.Sieve.coverByImage G (G.obj (Opposite.unop X))).arrows f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  congr 1
  /-
    case w.h.a.w.h.h.h.e_a
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    X : Opposite C
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj ((G.op.comp ℱ).obj X)).obj U
    Y : D
    f : Quiver.Hom Y (G.obj (Opposite.unop X))
    hf : (CategoryTheory.Sieve.coverByImage G (G.obj (Opposite.unop X))).arrows f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp a …
  -/
  simp only [Category.assoc]
  /-
    case w.h.a.w.h.h.h.e_a
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    X : Opposite C
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj ((G.op.comp ℱ).obj X)).obj U
    Y : D
    f : Quiver.Hom Y (G.obj (Opposite.unop X))
    hf : (CategoryTheory.Sieve.coverByImage G (G.obj (Opposite.unop X))).arrows f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp a✝ (CategoryTheory.CategoryStruct.com …
  -/
  congr 1
  have := naturality_apply (G := G) (ℱ := ℱ ⋙ coyoneda.obj (op <| (G.op ⋙ ℱ).obj X))
    (ℱ' := ⟨_, Presheaf.isSheaf_comp_of_isSheaf K ℱ'.val
      (coyoneda.obj (op ((G.op ⋙ ℱ).obj X))) ℱ'.cond⟩)
    (whiskerRight α (coyoneda.obj _)) hf.some.map (𝟙 _)
  /-
    case w.h.a.w.h.h.h.e_a.e_a
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_6, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_7, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom (G.op.comp ℱ) (G.op.comp ℱ'.val)
    X : Opposite C
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj ((G.op.comp ℱ).obj X)).obj U
    Y : D
    f : Quiver.Hom Y (G.obj (Opposite.unop X))
    hf : (CategoryTheory.Sieve.coverByImage G (G.obj (Opposite.unop X))).arrows f
    this : Eq ({ val := ℱ'.val.comp (CategoryTheory.coyoneda.obj { unop := (G.op.c …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.app X) (ℱ'.val.map (Nonempty.some  …
  -/
  simpa using this
  /-
    🎉 no goals
  -/


/--
If the pullback map is obtained via whiskering,
then the result `sheaf_hom (whisker_left G.op α)` is equal to `α`.
-/
theorem sheafHom_eq (α : ℱ ⟶ ℱ'.val) : sheafHom (whiskerLeft G.op α) = α := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom ℱ ℱ'.val
    ⊢ Eq (CategoryTheory.Functor.IsCoverDense.sheafHom (CategoryTheory.whiskerLeft …
  -/
  ext X
  /-
    case w.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom ℱ ℱ'.val
    X : Opposite D
    ⊢ Eq ((CategoryTheory.Functor.IsCoverDense.sheafHom (CategoryTheory.whiskerLef …
  -/
  apply yoneda.map_injective
  /-
    case w.h.a
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom ℱ ℱ'.val
    X : Opposite D
    ⊢ Eq (CategoryTheory.yoneda.map ((CategoryTheory.Functor.IsCoverDense.sheafHom …
  -/
  ext U
  /-
    case w.h.a.w.h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom ℱ ℱ'.val
    X : Opposite D
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj (ℱ.obj X)).obj U
    ⊢ Eq ((CategoryTheory.yoneda.map ((CategoryTheory.Functor.IsCoverDense.sheafHo …
  -/
  erw [yoneda.map_preimage]
  /-
    case w.h.a.w.h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom ℱ ℱ'.val
    X : Opposite D
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj (ℱ.obj X)).obj U
    ⊢ Eq (((CategoryTheory.Functor.IsCoverDense.sheafYonedaHom (CategoryTheory.whi …
  -/
  symm
  /-
    case w.h.a.w.h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom ℱ ℱ'.val
    X : Opposite D
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj (ℱ.obj X)).obj U
    ⊢ Eq ((CategoryTheory.yoneda.map (α.app X)).app U a✝) (((CategoryTheory.Functo …
  -/
  change (show (ℱ'.val ⋙ coyoneda.obj (op (unop U))).obj (op (unop X)) from _) = _
  /-
    case w.h.a.w.h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom ℱ ℱ'.val
    X : Opposite D
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj (ℱ.obj X)).obj U
    ⊢ Eq (letFun ((CategoryTheory.yoneda.map (α.app X)).app U a✝) fun this => this …
  -/
  apply sheaf_eq_amalgamation ℱ' (G.is_cover_of_isCoverDense _ _)
  -- Porting note: next line was not needed in mathlib3
    /-
      case w.h.a.w.h.h.hx
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_6, u_2} D
      K : CategoryTheory.GrothendieckTopology D
      A : Type u_4
      inst✝² : CategoryTheory.Category.{u_5, u_4} A
      G : CategoryTheory.Functor C D
      inst✝¹ : G.IsCoverDense K
      inst✝ : G.IsLocallyFull K
      ℱ : CategoryTheory.Functor (Opposite D) A
      ℱ' : CategoryTheory.Sheaf K A
      α : Quiver.Hom ℱ ℱ'.val
      X : Opposite D
      U : Opposite A
      a✝ : (CategoryTheory.yoneda.obj (ℱ.obj X)).obj U
      ⊢ (CategoryTheory.Functor.IsCoverDense.Types.pushforwardFamily (CategoryTheory …
    -/
  · exact (pushforwardFamily_compatible _ _)
    /-
      🎉 no goals
    -/
  /-
    case w.h.a.w.h.h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom ℱ ℱ'.val
    X : Opposite D
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj (ℱ.obj X)).obj U
    ⊢ (CategoryTheory.Functor.IsCoverDense.Types.pushforwardFamily (CategoryTheory …
  -/
  intro Y f hf
  /-
    case w.h.a.w.h.h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom ℱ ℱ'.val
    X : Opposite D
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj (ℱ.obj X)).obj U
    Y : D
    f : Quiver.Hom Y (Opposite.unop X)
    hf : (CategoryTheory.Sieve.coverByImage G (Opposite.unop X)).arrows f
    ⊢ Eq ((ℱ'.val.comp (CategoryTheory.coyoneda.obj { unop := Opposite.unop U })). …
  -/
  conv_lhs => rw [← hf.some.fac]
  /-
    case w.h.a.w.h.h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom ℱ ℱ'.val
    X : Opposite D
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj (ℱ.obj X)).obj U
    Y : D
    f : Quiver.Hom Y (Opposite.unop X)
    hf : (CategoryTheory.Sieve.coverByImage G (Opposite.unop X)).arrows f
    ⊢ Eq ((ℱ'.val.comp (CategoryTheory.coyoneda.obj { unop := Opposite.unop U })). …
  -/
  dsimp
  /-
    case w.h.a.w.h.h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Functor (Opposite D) A
    ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom ℱ ℱ'.val
    X : Opposite D
    U : Opposite A
    a✝ : (CategoryTheory.yoneda.obj (ℱ.obj X)).obj U
    Y : D
    f : Quiver.Hom Y (Opposite.unop X)
    hf : (CategoryTheory.Sieve.coverByImage G (Opposite.unop X)).arrows f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp a …
  -/
  simp
  /-
    🎉 no goals
  -/


/--
A locally-full and cover-dense functor `G` induces an equivalence between morphisms into a sheaf and
morphisms over the restrictions via `G`.
-/
noncomputable def restrictHomEquivHom : (G.op ⋙ ℱ ⟶ G.op ⋙ ℱ'.val) ≃ (ℱ ⟶ ℱ'.val) where
  toFun := sheafHom
  invFun := whiskerLeft G.op
  left_inv := sheafHom_restrict_eq
  right_inv := sheafHom_eq _


/-- Given a locally-full and cover-dense functor `G` and a natural transformation of sheaves
`α : ℱ ⟶ ℱ'`, if the pullback of `α` along `G` is iso, then `α` is also iso.
-/
theorem iso_of_restrict_iso {ℱ ℱ' : Sheaf K A} (α : ℱ ⟶ ℱ') (i : IsIso (whiskerLeft G.op α.val)) :
    IsIso α := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_6, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom ℱ ℱ'
    i : CategoryTheory.IsIso (CategoryTheory.whiskerLeft G.op α.val)
    ⊢ CategoryTheory.IsIso α
  -/
  convert (sheafIso (asIso (whiskerLeft G.op α.val))).isIso_hom using 1
  /-
    case h.e'_5
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_6, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom ℱ ℱ'
    i : CategoryTheory.IsIso (CategoryTheory.whiskerLeft G.op α.val)
    ⊢ Eq α (CategoryTheory.Functor.IsCoverDense.sheafIso (CategoryTheory.asIso (Ca …
  -/
  ext1
  /-
    case h.e'_5.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    A : Type u_4
    inst✝² : CategoryTheory.Category.{u_6, u_4} A
    G : CategoryTheory.Functor C D
    inst✝¹ : G.IsCoverDense K
    inst✝ : G.IsLocallyFull K
    ℱ ℱ' : CategoryTheory.Sheaf K A
    α : Quiver.Hom ℱ ℱ'
    i : CategoryTheory.IsIso (CategoryTheory.whiskerLeft G.op α.val)
    ⊢ Eq α.val (CategoryTheory.Functor.IsCoverDense.sheafIso (CategoryTheory.asIso …
  -/
  apply (sheafHom_eq _ _).symm
  /-
    🎉 no goals
  -/


/-- A locally-fully-faithful and cover-dense functor preserves compatible families. -/
lemma compatiblePreserving [G.IsLocallyFaithful K] : CompatiblePreserving K G := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝² : G.IsCoverDense K
    inst✝¹ : G.IsLocallyFull K
    inst✝ : G.IsLocallyFaithful K
    ⊢ CategoryTheory.CompatiblePreserving K G
  -/
  constructor
  /-
    case compatible
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝² : G.IsCoverDense K
    inst✝¹ : G.IsLocallyFull K
    inst✝ : G.IsLocallyFaithful K
    ⊢ ∀ (ℱ : CategoryTheory.Sheaf K (Type u_7)) {Z : C} {T : CategoryTheory.Presie …
  -/
  intro ℱ Z T x hx Y₁ Y₂ X f₁ f₂ g₁ g₂ hg₁ hg₂ eq
  /-
    case compatible
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝² : G.IsCoverDense K
    inst✝¹ : G.IsLocallyFull K
    inst✝ : G.IsLocallyFaithful K
    ℱ : CategoryTheory.Sheaf K (Type u_7)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (G.obj Y₁)
    f₂ : Quiver.Hom X (G.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    eq : Eq (CategoryTheory.CategoryStruct.comp f₁ (G.map g₁)) (CategoryTheory.Cat …
    ⊢ Eq (ℱ.val.map f₁.op (x g₁ hg₁)) (ℱ.val.map f₂.op (x g₂ hg₂))
  -/
  apply Functor.IsCoverDense.ext G
  /-
    case compatible.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝² : G.IsCoverDense K
    inst✝¹ : G.IsLocallyFull K
    inst✝ : G.IsLocallyFaithful K
    ℱ : CategoryTheory.Sheaf K (Type u_7)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (G.obj Y₁)
    f₂ : Quiver.Hom X (G.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    eq : Eq (CategoryTheory.CategoryStruct.comp f₁ (G.map g₁)) (CategoryTheory.Cat …
    ⊢ ∀ ⦃Y : C⦄ (f : Quiver.Hom (G.obj Y) X), Eq (ℱ.val.map f.op (ℱ.val.map f₁.op  …
  -/
  intro W i
  /-
    case compatible.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝² : G.IsCoverDense K
    inst✝¹ : G.IsLocallyFull K
    inst✝ : G.IsLocallyFaithful K
    ℱ : CategoryTheory.Sheaf K (Type u_7)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (G.obj Y₁)
    f₂ : Quiver.Hom X (G.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    eq : Eq (CategoryTheory.CategoryStruct.comp f₁ (G.map g₁)) (CategoryTheory.Cat …
    W : C
    i : Quiver.Hom (G.obj W) X
    ⊢ Eq (ℱ.val.map i.op (ℱ.val.map f₁.op (x g₁ hg₁))) (ℱ.val.map i.op (ℱ.val.map  …
  -/
  refine IsLocallyFull.ext G _ (i ≫ f₁) fun V₁ iVW iV₁Y₁ e₁ ↦ ?_
  /-
    case compatible.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝² : G.IsCoverDense K
    inst✝¹ : G.IsLocallyFull K
    inst✝ : G.IsLocallyFaithful K
    ℱ : CategoryTheory.Sheaf K (Type u_7)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (G.obj Y₁)
    f₂ : Quiver.Hom X (G.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    eq : Eq (CategoryTheory.CategoryStruct.comp f₁ (G.map g₁)) (CategoryTheory.Cat …
    W : C
    i : Quiver.Hom (G.obj W) X
    V₁ : C
    iVW : Quiver.Hom V₁ W
    iV₁Y₁ : Quiver.Hom V₁ Y₁
    e₁ : Eq (G.map iV₁Y₁) (CategoryTheory.CategoryStruct.comp (G.map iVW) (Categor …
    ⊢ Eq (ℱ.val.map (G.map iVW).op (ℱ.val.map i.op (ℱ.val.map f₁.op (x g₁ hg₁))))  …
  -/
  refine IsLocallyFull.ext G _ (G.map iVW ≫ i ≫ f₂) fun V₂ iV₂V₁ iV₂Y₂ e₂ ↦ ?_
  /-
    case compatible.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝² : G.IsCoverDense K
    inst✝¹ : G.IsLocallyFull K
    inst✝ : G.IsLocallyFaithful K
    ℱ : CategoryTheory.Sheaf K (Type u_7)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (G.obj Y₁)
    f₂ : Quiver.Hom X (G.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    eq : Eq (CategoryTheory.CategoryStruct.comp f₁ (G.map g₁)) (CategoryTheory.Cat …
    W : C
    i : Quiver.Hom (G.obj W) X
    V₁ : C
    iVW : Quiver.Hom V₁ W
    iV₁Y₁ : Quiver.Hom V₁ Y₁
    e₁ : Eq (G.map iV₁Y₁) (CategoryTheory.CategoryStruct.comp (G.map iVW) (Categor …
    V₂ : C
    iV₂V₁ : Quiver.Hom V₂ V₁
    iV₂Y₂ : Quiver.Hom V₂ Y₂
    e₂ : Eq (G.map iV₂Y₂) (CategoryTheory.CategoryStruct.comp (G.map iV₂V₁) (Categ …
    ⊢ Eq (ℱ.val.map (G.map iV₂V₁).op (ℱ.val.map (G.map iVW).op (ℱ.val.map i.op (ℱ. …
  -/
  refine IsLocallyFaithful.ext G _ (iV₂V₁ ≫ iV₁Y₁ ≫ g₁) (iV₂Y₂ ≫ g₂) (by simp [e₁, e₂, eq]) ?_
  /-
    case compatible.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝² : G.IsCoverDense K
    inst✝¹ : G.IsLocallyFull K
    inst✝ : G.IsLocallyFaithful K
    ℱ : CategoryTheory.Sheaf K (Type u_7)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (G.obj Y₁)
    f₂ : Quiver.Hom X (G.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    eq : Eq (CategoryTheory.CategoryStruct.comp f₁ (G.map g₁)) (CategoryTheory.Cat …
    W : C
    i : Quiver.Hom (G.obj W) X
    V₁ : C
    iVW : Quiver.Hom V₁ W
    iV₁Y₁ : Quiver.Hom V₁ Y₁
    e₁ : Eq (G.map iV₁Y₁) (CategoryTheory.CategoryStruct.comp (G.map iVW) (Categor …
    V₂ : C
    iV₂V₁ : Quiver.Hom V₂ V₁
    iV₂Y₂ : Quiver.Hom V₂ Y₂
    e₂ : Eq (G.map iV₂Y₂) (CategoryTheory.CategoryStruct.comp (G.map iV₂V₁) (Categ …
    ⊢ ∀ ⦃Z_1 : C⦄ (j : Quiver.Hom Z_1 V₂), Eq (CategoryTheory.CategoryStruct.comp  …
  -/
  intro V₃ iV₃ e₄
  /-
    case compatible.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝² : G.IsCoverDense K
    inst✝¹ : G.IsLocallyFull K
    inst✝ : G.IsLocallyFaithful K
    ℱ : CategoryTheory.Sheaf K (Type u_7)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (G.obj Y₁)
    f₂ : Quiver.Hom X (G.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    eq : Eq (CategoryTheory.CategoryStruct.comp f₁ (G.map g₁)) (CategoryTheory.Cat …
    W : C
    i : Quiver.Hom (G.obj W) X
    V₁ : C
    iVW : Quiver.Hom V₁ W
    iV₁Y₁ : Quiver.Hom V₁ Y₁
    e₁ : Eq (G.map iV₁Y₁) (CategoryTheory.CategoryStruct.comp (G.map iVW) (Categor …
    V₂ : C
    iV₂V₁ : Quiver.Hom V₂ V₁
    iV₂Y₂ : Quiver.Hom V₂ Y₂
    e₂ : Eq (G.map iV₂Y₂) (CategoryTheory.CategoryStruct.comp (G.map iV₂V₁) (Categ …
    V₃ : C
    iV₃ : Quiver.Hom V₃ V₂
    e₄ : Eq (CategoryTheory.CategoryStruct.comp iV₃ (CategoryTheory.CategoryStruct …
    ⊢ Eq (ℱ.val.map (G.map iV₃).op (ℱ.val.map (G.map iV₂V₁).op (ℱ.val.map (G.map i …
  -/
  simp only [← op_comp, ← FunctorToTypes.map_comp_apply, ← e₁, ← e₂, ← Functor.map_comp]
  /-
    case compatible.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝² : G.IsCoverDense K
    inst✝¹ : G.IsLocallyFull K
    inst✝ : G.IsLocallyFaithful K
    ℱ : CategoryTheory.Sheaf K (Type u_7)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (G.obj Y₁)
    f₂ : Quiver.Hom X (G.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    eq : Eq (CategoryTheory.CategoryStruct.comp f₁ (G.map g₁)) (CategoryTheory.Cat …
    W : C
    i : Quiver.Hom (G.obj W) X
    V₁ : C
    iVW : Quiver.Hom V₁ W
    iV₁Y₁ : Quiver.Hom V₁ Y₁
    e₁ : Eq (G.map iV₁Y₁) (CategoryTheory.CategoryStruct.comp (G.map iVW) (Categor …
    V₂ : C
    iV₂V₁ : Quiver.Hom V₂ V₁
    iV₂Y₂ : Quiver.Hom V₂ Y₂
    e₂ : Eq (G.map iV₂Y₂) (CategoryTheory.CategoryStruct.comp (G.map iV₂V₁) (Categ …
    V₃ : C
    iV₃ : Quiver.Hom V₃ V₂
    e₄ : Eq (CategoryTheory.CategoryStruct.comp iV₃ (CategoryTheory.CategoryStruct …
    ⊢ Eq (ℱ.val.map (G.map (CategoryTheory.CategoryStruct.comp iV₃ (CategoryTheory …
  -/
  apply hx
  /-
    case compatible.h.a
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_2} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝² : G.IsCoverDense K
    inst✝¹ : G.IsLocallyFull K
    inst✝ : G.IsLocallyFaithful K
    ℱ : CategoryTheory.Sheaf K (Type u_7)
    Z : C
    T : CategoryTheory.Presieve Z
    x : CategoryTheory.Presieve.FamilyOfElements (G.op.comp ℱ.val) T
    hx : x.Compatible
    Y₁ Y₂ : C
    X : D
    f₁ : Quiver.Hom X (G.obj Y₁)
    f₂ : Quiver.Hom X (G.obj Y₂)
    g₁ : Quiver.Hom Y₁ Z
    g₂ : Quiver.Hom Y₂ Z
    hg₁ : T g₁
    hg₂ : T g₂
    eq : Eq (CategoryTheory.CategoryStruct.comp f₁ (G.map g₁)) (CategoryTheory.Cat …
    W : C
    i : Quiver.Hom (G.obj W) X
    V₁ : C
    iVW : Quiver.Hom V₁ W
    iV₁Y₁ : Quiver.Hom V₁ Y₁
    e₁ : Eq (G.map iV₁Y₁) (CategoryTheory.CategoryStruct.comp (G.map iVW) (Categor …
    V₂ : C
    iV₂V₁ : Quiver.Hom V₂ V₁
    iV₂Y₂ : Quiver.Hom V₂ Y₂
    e₂ : Eq (G.map iV₂Y₂) (CategoryTheory.CategoryStruct.comp (G.map iV₂V₁) (Categ …
    V₃ : C
    iV₃ : Quiver.Hom V₃ V₂
    e₄ : Eq (CategoryTheory.CategoryStruct.comp iV₃ (CategoryTheory.CategoryStruct …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp i …
  -/
  simpa using e₄
  /-
    🎉 no goals
  -/


lemma isContinuous [G.IsLocallyFaithful K] (Hp : CoverPreserving J K G) : G.IsContinuous J K :=
  isContinuous_of_coverPreserving (compatiblePreserving K G) Hp


instance full_sheafPushforwardContinuous [G.IsContinuous J K] :
    Full (G.sheafPushforwardContinuous A J K) where
  map_surjective α := ⟨⟨sheafHom α.val⟩, Sheaf.Hom.ext <| sheafHom_restrict_eq α.val⟩


instance faithful_sheafPushforwardContinuous [G.IsContinuous J K] :
    Faithful (G.sheafPushforwardContinuous A J K) where
  map_injective := by
    /-
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_6, u_1} C
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{u_7, u_2} D
      E : Type u_3
      inst✝⁴ : CategoryTheory.Category.{?u.147271, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝³ : CategoryTheory.Category.{u_5, u_4} A
      G : CategoryTheory.Functor C D
      inst✝² : G.IsCoverDense K
      inst✝¹ : G.IsLocallyFull K
      ℱ : CategoryTheory.Functor (Opposite D) A
      ℱ' : CategoryTheory.Sheaf K A
      inst✝ : G.IsContinuous J K
      ⊢ ∀ {X Y : CategoryTheory.Sheaf K A}, Function.Injective (G.sheafPushforwardCo …
    -/
    intro ℱ ℱ' α β e
    /-
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_6, u_1} C
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{u_7, u_2} D
      E : Type u_3
      inst✝⁴ : CategoryTheory.Category.{?u.147271, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝³ : CategoryTheory.Category.{u_5, u_4} A
      G : CategoryTheory.Functor C D
      inst✝² : G.IsCoverDense K
      inst✝¹ : G.IsLocallyFull K
      ℱ✝ : CategoryTheory.Functor (Opposite D) A
      ℱ'✝ : CategoryTheory.Sheaf K A
      inst✝ : G.IsContinuous J K
      ℱ ℱ' : CategoryTheory.Sheaf K A
      α β : Quiver.Hom ℱ ℱ'
      e : Eq ((G.sheafPushforwardContinuous A J K).map α) ((G.sheafPushforwardContin …
      ⊢ Eq α β
    -/
    ext1
    /-
      case h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_6, u_1} C
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{u_7, u_2} D
      E : Type u_3
      inst✝⁴ : CategoryTheory.Category.{?u.147271, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝³ : CategoryTheory.Category.{u_5, u_4} A
      G : CategoryTheory.Functor C D
      inst✝² : G.IsCoverDense K
      inst✝¹ : G.IsLocallyFull K
      ℱ✝ : CategoryTheory.Functor (Opposite D) A
      ℱ'✝ : CategoryTheory.Sheaf K A
      inst✝ : G.IsContinuous J K
      ℱ ℱ' : CategoryTheory.Sheaf K A
      α β : Quiver.Hom ℱ ℱ'
      e : Eq ((G.sheafPushforwardContinuous A J K).map α) ((G.sheafPushforwardContin …
      ⊢ Eq α.val β.val
    -/
    apply_fun fun e => e.val at e
    /-
      case h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_6, u_1} C
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{u_7, u_2} D
      E : Type u_3
      inst✝⁴ : CategoryTheory.Category.{?u.147271, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝³ : CategoryTheory.Category.{u_5, u_4} A
      G : CategoryTheory.Functor C D
      inst✝² : G.IsCoverDense K
      inst✝¹ : G.IsLocallyFull K
      ℱ✝ : CategoryTheory.Functor (Opposite D) A
      ℱ'✝ : CategoryTheory.Sheaf K A
      inst✝ : G.IsContinuous J K
      ℱ ℱ' : CategoryTheory.Sheaf K A
      α β : Quiver.Hom ℱ ℱ'
      e : Eq ((G.sheafPushforwardContinuous A J K).map α).val ((G.sheafPushforwardCo …
      ⊢ Eq α.val β.val
    -/
    dsimp [sheafPushforwardContinuous] at e
    /-
      case h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{u_6, u_1} C
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{u_7, u_2} D
      E : Type u_3
      inst✝⁴ : CategoryTheory.Category.{?u.147271, u_3} E
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      L : CategoryTheory.GrothendieckTopology E
      A : Type u_4
      inst✝³ : CategoryTheory.Category.{u_5, u_4} A
      G : CategoryTheory.Functor C D
      inst✝² : G.IsCoverDense K
      inst✝¹ : G.IsLocallyFull K
      ℱ✝ : CategoryTheory.Functor (Opposite D) A
      ℱ'✝ : CategoryTheory.Sheaf K A
      inst✝ : G.IsContinuous J K
      ℱ ℱ' : CategoryTheory.Sheaf K A
      α β : Quiver.Hom ℱ ℱ'
      e : Eq (CategoryTheory.whiskerLeft G.op α.val) (CategoryTheory.whiskerLeft G.o …
      ⊢ Eq α.val β.val
    -/
    rw [← sheafHom_eq G α.val, ← sheafHom_eq G β.val, e]
    /-
      🎉 no goals
    -/


/-- If `G : C ⥤ D` is cover dense and full, then the
map `(P ⟶ Q) → (G.op ⋙ P ⟶ G.op ⋙ Q)` is bijective when `Q` is a sheaf`. -/
lemma whiskerLeft_obj_map_bijective_of_isCoverDense (G : C ⥤ D)
    [G.IsCoverDense K] [G.IsLocallyFull K] {A : Type*} [Category A]
    (P Q : Dᵒᵖ ⥤ A) (hQ : Presheaf.IsSheaf K Q) :
    Function.Bijective (((whiskeringLeft Cᵒᵖ Dᵒᵖ A).obj G.op).map : (P ⟶ Q) → _) :=
  (IsCoverDense.restrictHomEquivHom (ℱ' := ⟨Q, hQ⟩)).symm.bijective


/-- The functor `G : C ⥤ D` exhibits `(C, J)` as a dense subsite of `(D, K)`
if `G` is cover-dense, locally fully-faithful,
and `S` is a cover of `C` if and only if the image of `S` in `D` is a cover. -/
class IsDenseSubsite : Prop where
  isCoverDense' : G.IsCoverDense K := by infer_instance
  isLocallyFull' : G.IsLocallyFull K := by infer_instance
  isLocallyFaithful' : G.IsLocallyFaithful K := by infer_instance
  functorPushforward_mem_iff : ∀ {X : C} {S : Sieve X}, S.functorPushforward G ∈ K _ ↔ S ∈ J _


lemma functorPushforward_mem_iff {X : C} {S : Sieve X} [G.IsDenseSubsite J K]:
    S.functorPushforward G ∈ K _ ↔ S ∈ J _ := IsDenseSubsite.functorPushforward_mem_iff


lemma isCoverDense : G.IsCoverDense K := isCoverDense' J

lemma isLocallyFull : G.IsLocallyFull K := isLocallyFull' J

lemma isLocallyFaithful : G.IsLocallyFaithful K := isLocallyFaithful' J


lemma coverPreserving : CoverPreserving J K G :=
  ⟨functorPushforward_mem_iff.mpr⟩


instance (priority := 900) : G.IsContinuous J K :=
  letI := IsDenseSubsite.isCoverDense J K G
  letI := IsDenseSubsite.isLocallyFull J K G
  letI := IsDenseSubsite.isLocallyFaithful J K G
  IsCoverDense.isContinuous J K G (IsDenseSubsite.coverPreserving J K G)


instance (priority := 900) : G.IsCocontinuous J K where
  cover_lift hS :=
    letI := IsDenseSubsite.isCoverDense J K G
    letI := IsDenseSubsite.isLocallyFull J K G
    IsDenseSubsite.functorPushforward_mem_iff.mp
      (IsCoverDense.functorPullback_pushforward_covering ⟨_, hS⟩)


instance full_sheafPushforwardContinuous :
    Full (G.sheafPushforwardContinuous A J K) :=
  letI := IsDenseSubsite.isCoverDense J K G
  letI := IsDenseSubsite.isLocallyFull J K G
  inferInstance


instance faithful_sheafPushforwardContinuous :
    Faithful (G.sheafPushforwardContinuous A J K) :=
  letI := IsDenseSubsite.isCoverDense J K G
  letI := IsDenseSubsite.isLocallyFull J K G
  inferInstance


lemma imageSieve_mem {U V} (f : G.obj U ⟶ G.obj V) :
    G.imageSieve f ∈ J _ :=
  letI := IsDenseSubsite.isLocallyFull J K G
  IsDenseSubsite.functorPushforward_mem_iff.mp (G.functorPushforward_imageSieve_mem K f)


lemma equalizer_mem {U V} (f₁ f₂ : U ⟶ V) (e : G.map f₁ = G.map f₂) :
    Sieve.equalizer f₁ f₂ ∈ J _ :=
  letI := IsDenseSubsite.isLocallyFaithful J K G
  IsDenseSubsite.functorPushforward_mem_iff.mp (G.functorPushforward_equalizer_mem K f₁ f₂ e)


