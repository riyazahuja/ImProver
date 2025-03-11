/-- The type of objects for the diagram indexing a wide (co)equalizer. -/
inductive WalkingParallelFamily (J : Type w) : Type w
  | zero : WalkingParallelFamily J
  | one : WalkingParallelFamily J


instance : DecidableEq (WalkingParallelFamily J)
  | zero, zero => isTrue rfl
  | zero, one => isFalse fun t => WalkingParallelFamily.noConfusion t
  | one, zero => isFalse fun t => WalkingParallelFamily.noConfusion t
  | one, one => isTrue rfl


instance : Inhabited (WalkingParallelFamily J) :=
  ⟨zero⟩


/-- The type family of morphisms for the diagram indexing a wide (co)equalizer. -/
inductive WalkingParallelFamily.Hom (J : Type w) :
  WalkingParallelFamily J → WalkingParallelFamily J → Type w
  | id : ∀ X : WalkingParallelFamily.{w} J, WalkingParallelFamily.Hom J X X
  | line : J → WalkingParallelFamily.Hom J zero one
  deriving DecidableEq


/-- Satisfying the inhabited linter -/
instance (J : Type v) : Inhabited (WalkingParallelFamily.Hom J zero zero) where default := Hom.id _


/-- Composition of morphisms in the indexing diagram for wide (co)equalizers. -/
def WalkingParallelFamily.Hom.comp :
    ∀ {X Y Z : WalkingParallelFamily J} (_ : WalkingParallelFamily.Hom J X Y)
      (_ : WalkingParallelFamily.Hom J Y Z), WalkingParallelFamily.Hom J X Z
  | _, _, _, id _, h => h
  | _, _, _, line j, id one => line j

-- attribute [local tidy] tactic.case_bash Porting note: no tidy, no local


instance WalkingParallelFamily.category : SmallCategory (WalkingParallelFamily J) where
  Hom := WalkingParallelFamily.Hom J
  id := WalkingParallelFamily.Hom.id
  comp := WalkingParallelFamily.Hom.comp
                    /-
                      J : Type w
                      W✝ X✝ Y✝ Z✝ : CategoryTheory.Limits.WalkingParallelFamily J
                      f : Quiver.Hom W✝ X✝
                      g : Quiver.Hom X✝ Y✝
                      h : Quiver.Hom Y✝ Z✝
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
                    -/
                  /-
                    J : Type w
                    X✝ Y✝ : CategoryTheory.Limits.WalkingParallelFamily J
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Y …
                  -/
                              /-
                                🎉 no goals
                              -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                              /-
                                🎉 no goals
                              -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  assoc f g h := by cases f <;> cases g <;> cases h <;> aesop_cat
                                                        /-
                                                          🎉 no goals
                                                        -/
  comp_id f := by cases f <;> aesop_cat


@[simp]
theorem WalkingParallelFamily.hom_id (X : WalkingParallelFamily J) :
    WalkingParallelFamily.Hom.id X = 𝟙 X :=
  rfl


/-- `parallelFamily f` is the diagram in `C` consisting of the given family of morphisms, each with
common domain and codomain.
-/
def parallelFamily : WalkingParallelFamily J ⥤ C where
  obj x := WalkingParallelFamily.casesOn x X Y
  map {x y} h :=
    match x, y, h with
    | _, _, Hom.id _ => 𝟙 _
    | _, _, line j => f j
  map_comp := by
    /-
      J : Type w
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : J → Quiver.Hom X Y
      ⊢ ∀ {X_1 Y_1 Z : CategoryTheory.Limits.WalkingParallelFamily J} (f_1 : Quiver. …
    -/
    rintro _ _ _ ⟨⟩ ⟨⟩ <;>
        /-
          case id.id
          J : Type w
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : C
          f : J → Quiver.Hom X Y
          X✝ : CategoryTheory.Limits.WalkingParallelFamily J
          ⊢ Eq ({ obj := fun x => CategoryTheory.Limits.WalkingParallelFamily.casesOn x  …
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
      · aesop_cat
        /-
          🎉 no goals
        -/


@[simp]
theorem parallelFamily_obj_zero : (parallelFamily f).obj zero = X :=
  rfl


@[simp]
theorem parallelFamily_obj_one : (parallelFamily f).obj one = Y :=
  rfl


@[simp]
theorem parallelFamily_map_left {j : J} : (parallelFamily f).map (line j) = f j :=
  rfl


/-- Every functor indexing a wide (co)equalizer is naturally isomorphic (actually, equal) to a
    `parallelFamily` -/
@[simps!]
def diagramIsoParallelFamily (F : WalkingParallelFamily J ⥤ C) :
    F ≅ parallelFamily fun j => F.map (line j) :=
                                              /-
                                                J : Type w
                                                C : Type u
                                                inst✝ : CategoryTheory.Category.{v, u} C
                                                X Y : C
                                                f : J → Quiver.Hom X Y
                                                F : CategoryTheory.Functor (CategoryTheory.Limits.WalkingParallelFamily J) C
                                                j : CategoryTheory.Limits.WalkingParallelFamily J
                                                ⊢ Eq (F.obj j) ((CategoryTheory.Limits.parallelFamily fun j => F.map (Category …
                                              -/
                                                          /-
                                                            🎉 no goals
                                                          -/
  NatIso.ofComponents (fun j => eqToIso <| by cases j <;> aesop_cat) <| by
                                                          /-
                                                            🎉 no goals
                                                          -/
    /-
      J : Type w
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : J → Quiver.Hom X Y
      F : CategoryTheory.Functor (CategoryTheory.Limits.WalkingParallelFamily J) C
      ⊢ ∀ {X Y : CategoryTheory.Limits.WalkingParallelFamily J} (f : Quiver.Hom X Y) …
    -/
                         /-
                           🎉 no goals
                         -/
    rintro _ _ (_|_) <;> aesop_cat
                         /-
                           🎉 no goals
                         -/


/-- `WalkingParallelPair` as a category is equivalent to a special case of
`WalkingParallelFamily`. -/
@[simps!]
def walkingParallelFamilyEquivWalkingParallelPair :
    WalkingParallelFamily.{w} (ULift Bool) ≌ WalkingParallelPair where
  functor :=
    parallelFamily fun p => cond p.down WalkingParallelPairHom.left WalkingParallelPairHom.right
  inverse := parallelPair (line (ULift.up true)) (line (ULift.up false))
                                                       /-
                                                         J : Type w
                                                         C : Type u
                                                         inst✝ : CategoryTheory.Category.{v, u} C
                                                         X✝ Y : C
                                                         f : J → Quiver.Hom X✝ Y
                                                         X : CategoryTheory.Limits.WalkingParallelFamily (ULift.{w, 0} Bool)
                                                         ⊢ Eq ((CategoryTheory.Functor.id (CategoryTheory.Limits.WalkingParallelFamily  …
                                                       -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  unitIso := NatIso.ofComponents (fun X => eqToIso (by cases X <;> rfl)) (by
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
    /-
      J : Type w
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : J → Quiver.Hom X Y
      ⊢ ∀ {X Y : CategoryTheory.Limits.WalkingParallelFamily (ULift.{w, 0} Bool)} (f …
    -/
                             /-
                               🎉 no goals
                             -/
                             /-
                               🎉 no goals
                             -/
    rintro _ _ (_|⟨_|_⟩) <;> aesop_cat)
                             /-
                               🎉 no goals
                             -/
                                                         /-
                                                           J : Type w
                                                           C : Type u
                                                           inst✝ : CategoryTheory.Category.{v, u} C
                                                           X✝ Y : C
                                                           f : J → Quiver.Hom X✝ Y
                                                           X : CategoryTheory.Limits.WalkingParallelPair
                                                           ⊢ Eq (((CategoryTheory.Limits.parallelPair (CategoryTheory.Limits.WalkingParal …
                                                         -/
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  counitIso := NatIso.ofComponents (fun X => eqToIso (by cases X <;> rfl)) (by
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    /-
      J : Type w
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : J → Quiver.Hom X Y
      ⊢ ∀ {X Y : CategoryTheory.Limits.WalkingParallelPair} (f : Quiver.Hom X Y), Eq …
    -/
                           /-
                             🎉 no goals
                           -/
                           /-
                             🎉 no goals
                           -/
    rintro _ _ (_|_|_) <;> aesop_cat)
                           /-
                             🎉 no goals
                           -/
                             /-
                               J : Type w
                               C : Type u
                               inst✝ : CategoryTheory.Category.{v, u} C
                               X Y : C
                               f : J → Quiver.Hom X Y
                               ⊢ ∀ (X : CategoryTheory.Limits.WalkingParallelFamily (ULift.{w, 0} Bool)), Eq  …
                             -/
                                              /-
                                                🎉 no goals
                                              -/
  functor_unitIso_comp := by rintro (_|_) <;> aesop_cat
                                              /-
                                                🎉 no goals
                                              -/


/-- A trident on `f` is just a `Cone (parallelFamily f)`. -/
abbrev Trident :=
  Cone (parallelFamily f)


/-- A cotrident on `f` and `g` is just a `Cocone (parallelFamily f)`. -/
abbrev Cotrident :=
  Cocone (parallelFamily f)


/-- A trident `t` on the parallel family `f : J → (X ⟶ Y)` consists of two morphisms
    `t.π.app zero : t.X ⟶ X` and `t.π.app one : t.X ⟶ Y`. Of these, only the first one is
    interesting, and we give it the shorter name `Trident.ι t`. -/
abbrev Trident.ι (t : Trident f) :=
  t.π.app zero


/-- A cotrident `t` on the parallel family `f : J → (X ⟶ Y)` consists of two morphisms
    `t.ι.app zero : X ⟶ t.X` and `t.ι.app one : Y ⟶ t.X`. Of these, only the second one is
    interesting, and we give it the shorter name `Cotrident.π t`. -/
abbrev Cotrident.π (t : Cotrident f) :=
  t.ι.app one


@[simp]
theorem Trident.ι_eq_app_zero (t : Trident f) : t.ι = t.π.app zero :=
  rfl


@[simp]
theorem Cotrident.π_eq_app_one (t : Cotrident f) : t.π = t.ι.app one :=
  rfl


@[reassoc (attr := simp)]
theorem Trident.app_zero (s : Trident f) (j : J) : s.π.app zero ≫ f j = s.π.app one := by
  /-
    J : Type w
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : J → Quiver.Hom X Y
    s : CategoryTheory.Limits.Trident f
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app CategoryTheory.Limits.Walkin …
  -/
  rw [← s.w (line j), parallelFamily_map_left]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem Cotrident.app_one (s : Cotrident f) (j : J) : f j ≫ s.ι.app one = s.ι.app zero := by
  /-
    J : Type w
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : J → Quiver.Hom X Y
    s : CategoryTheory.Limits.Cotrident f
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f j) (s.ι.app CategoryTheory.Limits. …
  -/
  rw [← s.w (line j), parallelFamily_map_left]
  /-
    🎉 no goals
  -/


/-- A trident on `f : J → (X ⟶ Y)` is determined by the morphism `ι : P ⟶ X` satisfying
`∀ j₁ j₂, ι ≫ f j₁ = ι ≫ f j₂`.
-/
@[simps]
def Trident.ofι [Nonempty J] {P : C} (ι : P ⟶ X) (w : ∀ j₁ j₂, ι ≫ f j₁ = ι ≫ f j₂) :
    Trident f where
  pt := P
  π :=
    { app := fun X => WalkingParallelFamily.casesOn X ι (ι ≫ f (Classical.arbitrary J))
      naturality := fun i j f => by
        /-
          J : Type w
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X Y : C
          f✝ : J → Quiver.Hom X Y
          inst✝ : Nonempty J
          P : C
          ι : Quiver.Hom P X
          w : ∀ (j₁ j₂ : J), Eq (CategoryTheory.CategoryStruct.comp ι (f✝ j₁)) (Category …
          i j : CategoryTheory.Limits.WalkingParallelFamily J
          f : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Cate …
        -/
        dsimp
        /-
          J : Type w
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X Y : C
          f✝ : J → Quiver.Hom X Y
          inst✝ : Nonempty J
          P : C
          ι : Quiver.Hom P X
          w : ∀ (j₁ j₂ : J), Eq (CategoryTheory.CategoryStruct.comp ι (f✝ j₁)) (Category …
          i j : CategoryTheory.Limits.WalkingParallelFamily J
          f : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id P)  …
        -/
        cases' f with _ k
          /-
            case id
            J : Type w
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            X Y : C
            f : J → Quiver.Hom X Y
            inst✝ : Nonempty J
            P : C
            ι : Quiver.Hom P X
            w : ∀ (j₁ j₂ : J), Eq (CategoryTheory.CategoryStruct.comp ι (f j₁)) (CategoryT …
            i : CategoryTheory.Limits.WalkingParallelFamily J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id P)  …
          -/
        · simp
          /-
            🎉 no goals
          -/
          /-
            case line
            J : Type w
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            X Y : C
            f : J → Quiver.Hom X Y
            inst✝ : Nonempty J
            P : C
            ι : Quiver.Hom P X
            w : ∀ (j₁ j₂ : J), Eq (CategoryTheory.CategoryStruct.comp ι (f j₁)) (CategoryT …
            k : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id P)  …
          -/
        · simp [w (Classical.arbitrary J) k] }
          /-
            🎉 no goals
          -/


/-- A cotrident on `f : J → (X ⟶ Y)` is determined by the morphism `π : Y ⟶ P` satisfying
`∀ j₁ j₂, f j₁ ≫ π = f j₂ ≫ π`.
-/
@[simps]
def Cotrident.ofπ [Nonempty J] {P : C} (π : Y ⟶ P) (w : ∀ j₁ j₂, f j₁ ≫ π = f j₂ ≫ π) :
    Cotrident f where
  pt := P
  ι :=
    { app := fun X => WalkingParallelFamily.casesOn X (f (Classical.arbitrary J) ≫ π) π
      naturality := fun i j f => by
        /-
          J : Type w
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X Y : C
          f✝ : J → Quiver.Hom X Y
          inst✝ : Nonempty J
          P : C
          π : Quiver.Hom Y P
          w : ∀ (j₁ j₂ : J), Eq (CategoryTheory.CategoryStruct.comp (f✝ j₁) π) (Category …
          i j : CategoryTheory.Limits.WalkingParallelFamily J
          f : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.parallelFamil …
        -/
        dsimp
        /-
          J : Type w
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X Y : C
          f✝ : J → Quiver.Hom X Y
          inst✝ : Nonempty J
          P : C
          π : Quiver.Hom Y P
          w : ∀ (j₁ j₂ : J), Eq (CategoryTheory.CategoryStruct.comp (f✝ j₁) π) (Category …
          i j : CategoryTheory.Limits.WalkingParallelFamily J
          f : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.parallelFamil …
        -/
        cases' f with _ k
          /-
            case id
            J : Type w
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            X Y : C
            f : J → Quiver.Hom X Y
            inst✝ : Nonempty J
            P : C
            π : Quiver.Hom Y P
            w : ∀ (j₁ j₂ : J), Eq (CategoryTheory.CategoryStruct.comp (f j₁) π) (CategoryT …
            i : CategoryTheory.Limits.WalkingParallelFamily J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.parallelFamil …
          -/
        · simp
          /-
            🎉 no goals
          -/
          /-
            case line
            J : Type w
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            X Y : C
            f : J → Quiver.Hom X Y
            inst✝ : Nonempty J
            P : C
            π : Quiver.Hom Y P
            w : ∀ (j₁ j₂ : J), Eq (CategoryTheory.CategoryStruct.comp (f j₁) π) (CategoryT …
            k : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.parallelFamil …
          -/
        · simp [w (Classical.arbitrary J) k] }
          /-
            🎉 no goals
          -/

-- See note [dsimp, simp]

theorem Trident.ι_ofι [Nonempty J] {P : C} (ι : P ⟶ X) (w : ∀ j₁ j₂, ι ≫ f j₁ = ι ≫ f j₂) :
    (Trident.ofι ι w).ι = ι :=
  rfl


theorem Cotrident.π_ofπ [Nonempty J] {P : C} (π : Y ⟶ P) (w : ∀ j₁ j₂, f j₁ ≫ π = f j₂ ≫ π) :
    (Cotrident.ofπ π w).π = π :=
  rfl


@[reassoc]
theorem Trident.condition (j₁ j₂ : J) (t : Trident f) : t.ι ≫ f j₁ = t.ι ≫ f j₂ := by
  /-
    J : Type w
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : J → Quiver.Hom X Y
    j₁ j₂ : J
    t : CategoryTheory.Limits.Trident f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp t.ι (f j₁)) (CategoryTheory.CategoryS …
  -/
  rw [t.app_zero, t.app_zero]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem Cotrident.condition (j₁ j₂ : J) (t : Cotrident f) : f j₁ ≫ t.π = f j₂ ≫ t.π := by
  /-
    J : Type w
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : J → Quiver.Hom X Y
    j₁ j₂ : J
    t : CategoryTheory.Limits.Cotrident f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f j₁) t.π) (CategoryTheory.CategoryS …
  -/
  rw [t.app_one, t.app_one]
  /-
    🎉 no goals
  -/


/-- To check whether two maps are equalized by both maps of a trident, it suffices to check it for
the first map -/
theorem Trident.equalizer_ext [Nonempty J] (s : Trident f) {W : C} {k l : W ⟶ s.pt}
    (h : k ≫ s.ι = l ≫ s.ι) : ∀ j : WalkingParallelFamily J, k ≫ s.π.app j = l ≫ s.π.app j
  | zero => h
              /-
                J : Type w
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                X Y : C
                f : J → Quiver.Hom X Y
                inst✝ : Nonempty J
                s : CategoryTheory.Limits.Trident f
                W : C
                k l : Quiver.Hom W s.pt
                h : Eq (CategoryTheory.CategoryStruct.comp k s.ι) (CategoryTheory.CategoryStru …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp k (s.π.app CategoryTheory.Limits.Walk …
              -/
  | one => by rw [← s.app_zero (Classical.arbitrary J), reassoc_of% h]
              /-
                🎉 no goals
              -/


/-- To check whether two maps are coequalized by both maps of a cotrident, it suffices to check it
for the second map -/
theorem Cotrident.coequalizer_ext [Nonempty J] (s : Cotrident f) {W : C} {k l : s.pt ⟶ W}
    (h : s.π ≫ k = s.π ≫ l) : ∀ j : WalkingParallelFamily J, s.ι.app j ≫ k = s.ι.app j ≫ l
               /-
                 J : Type w
                 C : Type u
                 inst✝¹ : CategoryTheory.Category.{v, u} C
                 X Y : C
                 f : J → Quiver.Hom X Y
                 inst✝ : Nonempty J
                 s : CategoryTheory.Limits.Cotrident f
                 W : C
                 k l : Quiver.Hom s.pt W
                 h : Eq (CategoryTheory.CategoryStruct.comp s.π k) (CategoryTheory.CategoryStru …
                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Walkin …
               -/
  | zero => by rw [← s.app_one (Classical.arbitrary J), Category.assoc, Category.assoc, h]
               /-
                 🎉 no goals
               -/
  | one => h


theorem Trident.IsLimit.hom_ext [Nonempty J] {s : Trident f} (hs : IsLimit s) {W : C}
    {k l : W ⟶ s.pt} (h : k ≫ s.ι = l ≫ s.ι) : k = l :=
  hs.hom_ext <| Trident.equalizer_ext _ h


theorem Cotrident.IsColimit.hom_ext [Nonempty J] {s : Cotrident f} (hs : IsColimit s) {W : C}
    {k l : s.pt ⟶ W} (h : s.π ≫ k = s.π ≫ l) : k = l :=
  hs.hom_ext <| Cotrident.coequalizer_ext _ h


/-- If `s` is a limit trident over `f`, then a morphism `k : W ⟶ X` satisfying
    `∀ j₁ j₂, k ≫ f j₁ = k ≫ f j₂` induces a morphism `l : W ⟶ s.X` such that
    `l ≫ Trident.ι s = k`. -/
def Trident.IsLimit.lift' [Nonempty J] {s : Trident f} (hs : IsLimit s) {W : C} (k : W ⟶ X)
    (h : ∀ j₁ j₂, k ≫ f j₁ = k ≫ f j₂) : { l : W ⟶ s.pt // l ≫ Trident.ι s = k } :=
  ⟨hs.lift <| Trident.ofι _ h, hs.fac _ _⟩


/-- If `s` is a colimit cotrident over `f`, then a morphism `k : Y ⟶ W` satisfying
    `∀ j₁ j₂, f j₁ ≫ k = f j₂ ≫ k` induces a morphism `l : s.X ⟶ W` such that
    `Cotrident.π s ≫ l = k`. -/
def Cotrident.IsColimit.desc' [Nonempty J] {s : Cotrident f} (hs : IsColimit s) {W : C} (k : Y ⟶ W)
    (h : ∀ j₁ j₂, f j₁ ≫ k = f j₂ ≫ k) : { l : s.pt ⟶ W // Cotrident.π s ≫ l = k } :=
  ⟨hs.desc <| Cotrident.ofπ _ h, hs.fac _ _⟩


/-- This is a slightly more convenient method to verify that a trident is a limit cone. It
    only asks for a proof of facts that carry any mathematical content -/
def Trident.IsLimit.mk [Nonempty J] (t : Trident f) (lift : ∀ s : Trident f, s.pt ⟶ t.pt)
    (fac : ∀ s : Trident f, lift s ≫ t.ι = s.ι)
    (uniq :
      ∀ (s : Trident f) (m : s.pt ⟶ t.pt)
        (_ : ∀ j : WalkingParallelFamily J, m ≫ t.π.app j = s.π.app j), m = lift s) :
    IsLimit t :=
  { lift
    fac := fun s j =>
      WalkingParallelFamily.casesOn j (fac s)
            /-
              J : Type w
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              X Y : C
              f : J → Quiver.Hom X Y
              inst✝ : Nonempty J
              t : CategoryTheory.Limits.Trident f
              lift : (s : CategoryTheory.Limits.Trident f) → Quiver.Hom s.pt t.pt
              fac : ∀ (s : CategoryTheory.Limits.Trident f), Eq (CategoryTheory.CategoryStru …
              uniq : ∀ (s : CategoryTheory.Limits.Trident f) (m : Quiver.Hom s.pt t.pt), (∀  …
              s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.parallelFamily f)
              j : CategoryTheory.Limits.WalkingParallelFamily J
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (lift s) (t.π.app CategoryTheory.Limi …
            -/
        (by rw [← t.w (line (Classical.arbitrary J)), reassoc_of% fac, s.w])
            /-
              🎉 no goals
            -/
    uniq := uniq }


/-- This is another convenient method to verify that a trident is a limit cone. It
    only asks for a proof of facts that carry any mathematical content, and allows access to the
    same `s` for all parts. -/
def Trident.IsLimit.mk' [Nonempty J] (t : Trident f)
    (create : ∀ s : Trident f, { l // l ≫ t.ι = s.ι ∧ ∀ {m}, m ≫ t.ι = s.ι → m = l }) :
    IsLimit t :=
  Trident.IsLimit.mk t (fun s => (create s).1) (fun s => (create s).2.1) fun s _ w =>
    (create s).2.2 (w zero)


/-- This is a slightly more convenient method to verify that a cotrident is a colimit cocone. It
    only asks for a proof of facts that carry any mathematical content -/
def Cotrident.IsColimit.mk [Nonempty J] (t : Cotrident f) (desc : ∀ s : Cotrident f, t.pt ⟶ s.pt)
    (fac : ∀ s : Cotrident f, t.π ≫ desc s = s.π)
    (uniq :
      ∀ (s : Cotrident f) (m : t.pt ⟶ s.pt)
        (_ : ∀ j : WalkingParallelFamily J, t.ι.app j ≫ m = s.ι.app j), m = desc s) :
    IsColimit t :=
  { desc
    fac := fun s j =>
                                          /-
                                            J : Type w
                                            C : Type u
                                            inst✝¹ : CategoryTheory.Category.{v, u} C
                                            X Y : C
                                            f : J → Quiver.Hom X Y
                                            inst✝ : Nonempty J
                                            t : CategoryTheory.Limits.Cotrident f
                                            desc : (s : CategoryTheory.Limits.Cotrident f) → Quiver.Hom t.pt s.pt
                                            fac : ∀ (s : CategoryTheory.Limits.Cotrident f), Eq (CategoryTheory.CategorySt …
                                            uniq : ∀ (s : CategoryTheory.Limits.Cotrident f) (m : Quiver.Hom t.pt s.pt), ( …
                                            s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelFamily f)
                                            j : CategoryTheory.Limits.WalkingParallelFamily J
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.ι.app CategoryTheory.Limits.Walkin …
                                          -/
      WalkingParallelFamily.casesOn j (by rw [← t.w_assoc (line (Classical.arbitrary J)), fac, s.w])
                                          /-
                                            🎉 no goals
                                          -/
        (fac s)
    uniq := uniq }


/-- This is another convenient method to verify that a cotrident is a colimit cocone. It
    only asks for a proof of facts that carry any mathematical content, and allows access to the
    same `s` for all parts. -/
def Cotrident.IsColimit.mk' [Nonempty J] (t : Cotrident f)
    (create :
      ∀ s : Cotrident f, { l : t.pt ⟶ s.pt // t.π ≫ l = s.π ∧ ∀ {m}, t.π ≫ m = s.π → m = l }) :
    IsColimit t :=
  Cotrident.IsColimit.mk t (fun s => (create s).1) (fun s => (create s).2.1) fun s _ w =>
    (create s).2.2 (w one)


/--
Given a limit cone for the family `f : J → (X ⟶ Y)`, for any `Z`, morphisms from `Z` to its point
are in bijection with morphisms `h : Z ⟶ X` such that `∀ j₁ j₂, h ≫ f j₁ = h ≫ f j₂`.
Further, this bijection is natural in `Z`: see `Trident.Limits.homIso_natural`.
-/
@[simps]
def Trident.IsLimit.homIso [Nonempty J] {t : Trident f} (ht : IsLimit t) (Z : C) :
    (Z ⟶ t.pt) ≃ { h : Z ⟶ X // ∀ j₁ j₂, h ≫ f j₁ = h ≫ f j₂ } where
                          /-
                            J : Type w
                            C : Type u
                            inst✝¹ : CategoryTheory.Category.{v, u} C
                            X Y : C
                            f : J → Quiver.Hom X Y
                            inst✝ : Nonempty J
                            t : CategoryTheory.Limits.Trident f
                            ht : CategoryTheory.Limits.IsLimit t
                            Z : C
                            k : Quiver.Hom Z t.pt
                            ⊢ ∀ (j₁ j₂ : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Catego …
                          -/
  toFun k := ⟨k ≫ t.ι, by simp⟩
                          /-
                            🎉 no goals
                          -/
  invFun h := (Trident.IsLimit.lift' ht _ h.prop).1
  left_inv _ := Trident.IsLimit.hom_ext ht (Trident.IsLimit.lift' _ _ _).prop
  right_inv _ := Subtype.ext (Trident.IsLimit.lift' ht _ _).prop


/-- The bijection of `Trident.IsLimit.homIso` is natural in `Z`. -/
theorem Trident.IsLimit.homIso_natural [Nonempty J] {t : Trident f} (ht : IsLimit t) {Z Z' : C}
    (q : Z' ⟶ Z) (k : Z ⟶ t.pt) :
    (Trident.IsLimit.homIso ht _ (q ≫ k) : Z' ⟶ X) =
      q ≫ (Trident.IsLimit.homIso ht _ k : Z ⟶ X) :=
  Category.assoc _ _ _


/-- Given a colimit cocone for the family `f : J → (X ⟶ Y)`, for any `Z`, morphisms from the cocone
point to `Z` are in bijection with morphisms `h : Z ⟶ X` such that
`∀ j₁ j₂, f j₁ ≫ h = f j₂ ≫ h`.  Further, this bijection is natural in `Z`: see
`Cotrident.IsColimit.homIso_natural`.
-/
@[simps]
def Cotrident.IsColimit.homIso [Nonempty J] {t : Cotrident f} (ht : IsColimit t) (Z : C) :
    (t.pt ⟶ Z) ≃ { h : Y ⟶ Z // ∀ j₁ j₂, f j₁ ≫ h = f j₂ ≫ h } where
                          /-
                            J : Type w
                            C : Type u
                            inst✝¹ : CategoryTheory.Category.{v, u} C
                            X Y : C
                            f : J → Quiver.Hom X Y
                            inst✝ : Nonempty J
                            t : CategoryTheory.Limits.Cotrident f
                            ht : CategoryTheory.Limits.IsColimit t
                            Z : C
                            k : Quiver.Hom t.pt Z
                            ⊢ ∀ (j₁ j₂ : J), Eq (CategoryTheory.CategoryStruct.comp (f j₁) (CategoryTheory …
                          -/
  toFun k := ⟨t.π ≫ k, by simp⟩
                          /-
                            🎉 no goals
                          -/
  invFun h := (Cotrident.IsColimit.desc' ht _ h.prop).1
  left_inv _ := Cotrident.IsColimit.hom_ext ht (Cotrident.IsColimit.desc' _ _ _).prop
  right_inv _ := Subtype.ext (Cotrident.IsColimit.desc' ht _ _).prop


/-- The bijection of `Cotrident.IsColimit.homIso` is natural in `Z`. -/
theorem Cotrident.IsColimit.homIso_natural [Nonempty J] {t : Cotrident f} {Z Z' : C} (q : Z ⟶ Z')
    (ht : IsColimit t) (k : t.pt ⟶ Z) :
    (Cotrident.IsColimit.homIso ht _ (k ≫ q) : Y ⟶ Z') =
      (Cotrident.IsColimit.homIso ht _ k : Y ⟶ Z) ≫ q :=
  (Category.assoc _ _ _).symm


/-- This is a helper construction that can be useful when verifying that a category has certain wide
    equalizers. Given `F : WalkingParallelFamily ⥤ C`, which is really the same as
    `parallelFamily (fun j ↦ F.map (line j))`, and a trident on `fun j ↦ F.map (line j)`,
    we get a cone on `F`.

    If you're thinking about using this, have a look at
    `hasWideEqualizers_of_hasLimit_parallelFamily`, which you may find to be an easier way of
    achieving your goal. -/
def Cone.ofTrident {F : WalkingParallelFamily J ⥤ C} (t : Trident fun j => F.map (line j)) :
    Cone F where
  pt := t.pt
  π :=
                                              /-
                                                J : Type w
                                                C : Type u
                                                inst✝ : CategoryTheory.Category.{v, u} C
                                                X✝ Y : C
                                                f : J → Quiver.Hom X✝ Y
                                                F : CategoryTheory.Functor (CategoryTheory.Limits.WalkingParallelFamily J) C
                                                t : CategoryTheory.Limits.Trident fun j => F.map (CategoryTheory.Limits.Walkin …
                                                X : CategoryTheory.Limits.WalkingParallelFamily J
                                                ⊢ Eq ((CategoryTheory.Limits.parallelFamily fun j => F.map (CategoryTheory.Lim …
                                              -/
                                                          /-
                                                            🎉 no goals
                                                          -/
    { app := fun X => t.π.app X ≫ eqToHom (by cases X <;> aesop_cat)
                                                          /-
                                                            🎉 no goals
                                                          -/
                                     /-
                                       J : Type w
                                       C : Type u
                                       inst✝ : CategoryTheory.Category.{v, u} C
                                       X Y : C
                                       f : J → Quiver.Hom X Y
                                       F : CategoryTheory.Functor (CategoryTheory.Limits.WalkingParallelFamily J) C
                                       t : CategoryTheory.Limits.Trident fun j => F.map (CategoryTheory.Limits.Walkin …
                                       j j' : CategoryTheory.Limits.WalkingParallelFamily J
                                       g : Quiver.Hom j j'
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Cate …
                                     -/
                                                 /-
                                                   🎉 no goals
                                                 -/
      naturality := fun j j' g => by cases g <;> aesop_cat }
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- This is a helper construction that can be useful when verifying that a category has all
    coequalizers. Given `F : WalkingParallelFamily ⥤ C`, which is really the same as
    `parallelFamily (fun j ↦ F.map (line j))`, and a cotrident on `fun j ↦ F.map (line j)` we get a
    cocone on `F`.

    If you're thinking about using this, have a look at
    `hasWideCoequalizers_of_hasColimit_parallelFamily`, which you may find to be an easier way
    of achieving your goal. -/
def Cocone.ofCotrident {F : WalkingParallelFamily J ⥤ C} (t : Cotrident fun j => F.map (line j)) :
    Cocone F where
  pt := t.pt
  ι :=
                                  /-
                                    J : Type w
                                    C : Type u
                                    inst✝ : CategoryTheory.Category.{v, u} C
                                    X✝ Y : C
                                    f : J → Quiver.Hom X✝ Y
                                    F : CategoryTheory.Functor (CategoryTheory.Limits.WalkingParallelFamily J) C
                                    t : CategoryTheory.Limits.Cotrident fun j => F.map (CategoryTheory.Limits.Walk …
                                    X : CategoryTheory.Limits.WalkingParallelFamily J
                                    ⊢ Eq (F.obj X) ((CategoryTheory.Limits.parallelFamily fun j => F.map (Category …
                                  -/
                                              /-
                                                🎉 no goals
                                              -/
    { app := fun X => eqToHom (by cases X <;> aesop_cat) ≫ t.ι.app X
                                              /-
                                                🎉 no goals
                                              -/
                                     /-
                                       J : Type w
                                       C : Type u
                                       inst✝ : CategoryTheory.Category.{v, u} C
                                       X Y : C
                                       f : J → Quiver.Hom X Y
                                       F : CategoryTheory.Functor (CategoryTheory.Limits.WalkingParallelFamily J) C
                                       t : CategoryTheory.Limits.Cotrident fun j => F.map (CategoryTheory.Limits.Walk …
                                       j j' : CategoryTheory.Limits.WalkingParallelFamily J
                                       g : Quiver.Hom j j'
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map g) ((fun X => CategoryTheory.C …
                                     -/
                                                           /-
                                                             🎉 no goals
                                                           -/
      naturality := fun j j' g => by cases g <;> dsimp <;> simp [Cotrident.app_one t] }
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
theorem Cone.ofTrident_π {F : WalkingParallelFamily J ⥤ C} (t : Trident fun j => F.map (line j))
                                                               /-
                                                                 J : Type w
                                                                 C : Type u
                                                                 inst✝ : CategoryTheory.Category.{v, u} C
                                                                 X Y : C
                                                                 f : J → Quiver.Hom X Y
                                                                 F : CategoryTheory.Functor (CategoryTheory.Limits.WalkingParallelFamily J) C
                                                                 t : CategoryTheory.Limits.Trident fun j => F.map (CategoryTheory.Limits.Walkin …
                                                                 j : CategoryTheory.Limits.WalkingParallelFamily J
                                                                 ⊢ Eq ((CategoryTheory.Limits.parallelFamily fun j => F.map (CategoryTheory.Lim …
                                                               -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
    (j) : (Cone.ofTrident t).π.app j = t.π.app j ≫ eqToHom (by cases j <;> aesop_cat) :=
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  rfl


@[simp]
theorem Cocone.ofCotrident_ι {F : WalkingParallelFamily J ⥤ C}
    (t : Cotrident fun j => F.map (line j)) (j) :
                                                 /-
                                                   J : Type w
                                                   C : Type u
                                                   inst✝ : CategoryTheory.Category.{v, u} C
                                                   X Y : C
                                                   f : J → Quiver.Hom X Y
                                                   F : CategoryTheory.Functor (CategoryTheory.Limits.WalkingParallelFamily J) C
                                                   t : CategoryTheory.Limits.Cotrident fun j => F.map (CategoryTheory.Limits.Walk …
                                                   j : CategoryTheory.Limits.WalkingParallelFamily J
                                                   ⊢ Eq (F.obj j) ((CategoryTheory.Limits.parallelFamily fun j => F.map (Category …
                                                 -/
                                                             /-
                                                               🎉 no goals
                                                             -/
    (Cocone.ofCotrident t).ι.app j = eqToHom (by cases j <;> aesop_cat) ≫ t.ι.app j :=
                                                             /-
                                                               🎉 no goals
                                                             -/
  rfl


/-- Given `F : WalkingParallelFamily ⥤ C`, which is really the same as
    `parallelFamily (fun j ↦ F.map (line j))` and a cone on `F`, we get a trident on
    `fun j ↦ F.map (line j)`. -/
def Trident.ofCone {F : WalkingParallelFamily J ⥤ C} (t : Cone F) :
    Trident fun j => F.map (line j) where
  pt := t.pt
  π :=
                                              /-
                                                J : Type w
                                                C : Type u
                                                inst✝ : CategoryTheory.Category.{v, u} C
                                                X✝ Y : C
                                                f : J → Quiver.Hom X✝ Y
                                                F : CategoryTheory.Functor (CategoryTheory.Limits.WalkingParallelFamily J) C
                                                t : CategoryTheory.Limits.Cone F
                                                X : CategoryTheory.Limits.WalkingParallelFamily J
                                                ⊢ Eq (F.obj X) ((CategoryTheory.Limits.parallelFamily fun j => F.map (Category …
                                              -/
                                                          /-
                                                            🎉 no goals
                                                          -/
    { app := fun X => t.π.app X ≫ eqToHom (by cases X <;> aesop_cat)
                                                          /-
                                                            🎉 no goals
                                                          -/
                       /-
                         J : Type w
                         C : Type u
                         inst✝ : CategoryTheory.Category.{v, u} C
                         X Y : C
                         f : J → Quiver.Hom X Y
                         F : CategoryTheory.Functor (CategoryTheory.Limits.WalkingParallelFamily J) C
                         t : CategoryTheory.Limits.Cone F
                         ⊢ ∀ ⦃X Y : CategoryTheory.Limits.WalkingParallelFamily J⦄ (f : Quiver.Hom X Y) …
                       -/
                                            /-
                                              🎉 no goals
                                            -/
      naturality := by rintro _ _ (_|_) <;> aesop_cat }
                                            /-
                                              🎉 no goals
                                            -/


/-- Given `F : WalkingParallelFamily ⥤ C`, which is really the same as
    `parallelFamily (F.map left) (F.map right)` and a cocone on `F`, we get a cotrident on
    `fun j ↦ F.map (line j)`. -/
def Cotrident.ofCocone {F : WalkingParallelFamily J ⥤ C} (t : Cocone F) :
    Cotrident fun j => F.map (line j) where
  pt := t.pt
  ι :=
                                  /-
                                    J : Type w
                                    C : Type u
                                    inst✝ : CategoryTheory.Category.{v, u} C
                                    X✝ Y : C
                                    f : J → Quiver.Hom X✝ Y
                                    F : CategoryTheory.Functor (CategoryTheory.Limits.WalkingParallelFamily J) C
                                    t : CategoryTheory.Limits.Cocone F
                                    X : CategoryTheory.Limits.WalkingParallelFamily J
                                    ⊢ Eq ((CategoryTheory.Limits.parallelFamily fun j => F.map (CategoryTheory.Lim …
                                  -/
                                              /-
                                                🎉 no goals
                                              -/
    { app := fun X => eqToHom (by cases X <;> aesop_cat) ≫ t.ι.app X
                                              /-
                                                🎉 no goals
                                              -/
                       /-
                         J : Type w
                         C : Type u
                         inst✝ : CategoryTheory.Category.{v, u} C
                         X Y : C
                         f : J → Quiver.Hom X Y
                         F : CategoryTheory.Functor (CategoryTheory.Limits.WalkingParallelFamily J) C
                         t : CategoryTheory.Limits.Cocone F
                         ⊢ ∀ ⦃X Y : CategoryTheory.Limits.WalkingParallelFamily J⦄ (f : Quiver.Hom X Y) …
                       -/
                                            /-
                                              🎉 no goals
                                            -/
      naturality := by rintro _ _ (_|_) <;> aesop_cat }
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem Trident.ofCone_π {F : WalkingParallelFamily J ⥤ C} (t : Cone F) (j) :
                                                         /-
                                                           J : Type w
                                                           C : Type u
                                                           inst✝ : CategoryTheory.Category.{v, u} C
                                                           X Y : C
                                                           f : J → Quiver.Hom X Y
                                                           F : CategoryTheory.Functor (CategoryTheory.Limits.WalkingParallelFamily J) C
                                                           t : CategoryTheory.Limits.Cone F
                                                           j : CategoryTheory.Limits.WalkingParallelFamily J
                                                           ⊢ Eq (F.obj j) ((CategoryTheory.Limits.parallelFamily fun j => F.map (Category …
                                                         -/
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    (Trident.ofCone t).π.app j = t.π.app j ≫ eqToHom (by cases j <;> aesop_cat) :=
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  rfl


@[simp]
theorem Cotrident.ofCocone_ι {F : WalkingParallelFamily J ⥤ C} (t : Cocone F) (j) :
                                                 /-
                                                   J : Type w
                                                   C : Type u
                                                   inst✝ : CategoryTheory.Category.{v, u} C
                                                   X Y : C
                                                   f : J → Quiver.Hom X Y
                                                   F : CategoryTheory.Functor (CategoryTheory.Limits.WalkingParallelFamily J) C
                                                   t : CategoryTheory.Limits.Cocone F
                                                   j : CategoryTheory.Limits.WalkingParallelFamily J
                                                   ⊢ Eq ((CategoryTheory.Limits.parallelFamily fun j => F.map (CategoryTheory.Lim …
                                                 -/
                                                             /-
                                                               🎉 no goals
                                                             -/
    (Cotrident.ofCocone t).ι.app j = eqToHom (by cases j <;> aesop_cat) ≫ t.ι.app j :=
                                                             /-
                                                               🎉 no goals
                                                             -/
  rfl


/-- Helper function for constructing morphisms between wide equalizer tridents.
-/
@[simps]
def Trident.mkHom [Nonempty J] {s t : Trident f} (k : s.pt ⟶ t.pt)
    (w : k ≫ t.ι = s.ι := by aesop_cat) : s ⟶ t where
  hom := k
  w := by
    /-
      J : Type w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : J → Quiver.Hom X Y
      inst✝ : Nonempty J
      s t : CategoryTheory.Limits.Trident f
      k : Quiver.Hom s.pt t.pt
      w : autoParam (Eq (CategoryTheory.CategoryStruct.comp k t.ι) s.ι) _auto✝
      ⊢ ∀ (j : CategoryTheory.Limits.WalkingParallelFamily J), Eq (CategoryTheory.Ca …
    -/
    rintro ⟨_ | _⟩
      /-
        case zero
        J : Type w
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X Y : C
        f : J → Quiver.Hom X Y
        inst✝ : Nonempty J
        s t : CategoryTheory.Limits.Trident f
        k : Quiver.Hom s.pt t.pt
        w : autoParam (Eq (CategoryTheory.CategoryStruct.comp k t.ι) s.ι) _auto✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp k (t.π.app CategoryTheory.Limits.Walk …
      -/
    · exact w
      /-
        🎉 no goals
      -/
      /-
        case one
        J : Type w
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X Y : C
        f : J → Quiver.Hom X Y
        inst✝ : Nonempty J
        s t : CategoryTheory.Limits.Trident f
        k : Quiver.Hom s.pt t.pt
        w : autoParam (Eq (CategoryTheory.CategoryStruct.comp k t.ι) s.ι) _auto✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp k (t.π.app CategoryTheory.Limits.Walk …
      -/
    · simpa using w =≫ f (Classical.arbitrary J)
      /-
        🎉 no goals
      -/


/-- To construct an isomorphism between tridents,
it suffices to give an isomorphism between the cone points
and check that it commutes with the `ι` morphisms.
-/
@[simps]
def Trident.ext [Nonempty J] {s t : Trident f} (i : s.pt ≅ t.pt)
    (w : i.hom ≫ t.ι = s.ι := by aesop_cat) : s ≅ t where
  hom := Trident.mkHom i.hom w
                                 /-
                                   J : Type w
                                   C : Type u
                                   inst✝¹ : CategoryTheory.Category.{v, u} C
                                   X Y : C
                                   f : J → Quiver.Hom X Y
                                   inst✝ : Nonempty J
                                   s t : CategoryTheory.Limits.Trident f
                                   i : CategoryTheory.Iso s.pt t.pt
                                   w : autoParam (Eq (CategoryTheory.CategoryStruct.comp i.hom t.ι) s.ι) _auto✝
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp i.inv s.ι) t.ι
                                 -/
  inv := Trident.mkHom i.inv (by rw [← w, Iso.inv_hom_id_assoc])
                                 /-
                                   🎉 no goals
                                 -/


/-- Helper function for constructing morphisms between coequalizer cotridents.
-/
@[simps]
def Cotrident.mkHom [Nonempty J] {s t : Cotrident f} (k : s.pt ⟶ t.pt)
    (w : s.π ≫ k = t.π := by aesop_cat) : s ⟶ t where
  hom := k
  w := by
    /-
      J : Type w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y : C
      f : J → Quiver.Hom X Y
      inst✝ : Nonempty J
      s t : CategoryTheory.Limits.Cotrident f
      k : Quiver.Hom s.pt t.pt
      w : autoParam (Eq (CategoryTheory.CategoryStruct.comp s.π k) t.π) _auto✝
      ⊢ ∀ (j : CategoryTheory.Limits.WalkingParallelFamily J), Eq (CategoryTheory.Ca …
    -/
    rintro ⟨_ | _⟩
      /-
        case zero
        J : Type w
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X Y : C
        f : J → Quiver.Hom X Y
        inst✝ : Nonempty J
        s t : CategoryTheory.Limits.Cotrident f
        k : Quiver.Hom s.pt t.pt
        w : autoParam (Eq (CategoryTheory.CategoryStruct.comp s.π k) t.π) _auto✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Walkin …
      -/
    · simpa using f (Classical.arbitrary J) ≫= w
      /-
        🎉 no goals
      -/
      /-
        case one
        J : Type w
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X Y : C
        f : J → Quiver.Hom X Y
        inst✝ : Nonempty J
        s t : CategoryTheory.Limits.Cotrident f
        k : Quiver.Hom s.pt t.pt
        w : autoParam (Eq (CategoryTheory.CategoryStruct.comp s.π k) t.π) _auto✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Walkin …
      -/
    · exact w
      /-
        🎉 no goals
      -/


/-- To construct an isomorphism between cotridents,
it suffices to give an isomorphism between the cocone points
and check that it commutes with the `π` morphisms.
-/
def Cotrident.ext [Nonempty J] {s t : Cotrident f} (i : s.pt ≅ t.pt)
    (w : s.π ≫ i.hom = t.π := by aesop_cat) : s ≅ t where
  hom := Cotrident.mkHom i.hom w
                                   /-
                                     J : Type w
                                     C : Type u
                                     inst✝¹ : CategoryTheory.Category.{v, u} C
                                     X Y : C
                                     f : J → Quiver.Hom X Y
                                     inst✝ : Nonempty J
                                     s t : CategoryTheory.Limits.Cotrident f
                                     i : CategoryTheory.Iso s.pt t.pt
                                     w : autoParam (Eq (CategoryTheory.CategoryStruct.comp s.π i.hom) t.π) _auto✝
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp t.π i.inv) s.π
                                   -/
  inv := Cotrident.mkHom i.inv (by rw [Iso.comp_inv_eq, w])
                                   /-
                                     🎉 no goals
                                   -/


/--
`HasWideEqualizer f` represents a particular choice of limiting cone for the parallel family of
morphisms `f`.
-/
abbrev HasWideEqualizer :=
  HasLimit (parallelFamily f)


/-- If a wide equalizer of `f` exists, we can access an arbitrary choice of such by
    saying `wideEqualizer f`. -/
abbrev wideEqualizer : C :=
  limit (parallelFamily f)


/-- If a wide equalizer of `f` exists, we can access the inclusion `wideEqualizer f ⟶ X` by
    saying `wideEqualizer.ι f`. -/
abbrev wideEqualizer.ι : wideEqualizer f ⟶ X :=
  limit.π (parallelFamily f) zero


/-- A wide equalizer cone for a parallel family `f`.
-/
abbrev wideEqualizer.trident : Trident f :=
  limit.cone (parallelFamily f)


@[simp]
theorem wideEqualizer.trident_ι : (wideEqualizer.trident f).ι = wideEqualizer.ι f :=
  rfl


@[simp 1100]
theorem wideEqualizer.trident_π_app_zero :
    (wideEqualizer.trident f).π.app zero = wideEqualizer.ι f :=
  rfl


@[reassoc]
theorem wideEqualizer.condition (j₁ j₂ : J) : wideEqualizer.ι f ≫ f j₁ = wideEqualizer.ι f ≫ f j₂ :=
  Trident.condition j₁ j₂ <| limit.cone <| parallelFamily f


/-- The wideEqualizer built from `wideEqualizer.ι f` is limiting. -/
def wideEqualizerIsWideEqualizer [Nonempty J] :
    IsLimit (Trident.ofι (wideEqualizer.ι f) (wideEqualizer.condition f)) :=
                                        /-
                                          J : Type w
                                          C : Type u
                                          inst✝² : CategoryTheory.Category.{v, u} C
                                          X Y : C
                                          f : J → Quiver.Hom X Y
                                          inst✝¹ : CategoryTheory.Limits.HasWideEqualizer f
                                          inst✝ : Nonempty J
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl (CategoryThe …
                                        -/
  IsLimit.ofIsoLimit (limit.isLimit _) (Trident.ext (Iso.refl _))
                                        /-
                                          🎉 no goals
                                        -/


/-- A morphism `k : W ⟶ X` satisfying `∀ j₁ j₂, k ≫ f j₁ = k ≫ f j₂` factors through the
    wide equalizer of `f` via `wideEqualizer.lift : W ⟶ wideEqualizer f`. -/
abbrev wideEqualizer.lift [Nonempty J] {W : C} (k : W ⟶ X) (h : ∀ j₁ j₂, k ≫ f j₁ = k ≫ f j₂) :
    W ⟶ wideEqualizer f :=
  limit.lift (parallelFamily f) (Trident.ofι k h)


@[reassoc (attr := simp 1100)]
theorem wideEqualizer.lift_ι [Nonempty J] {W : C} (k : W ⟶ X)
    (h : ∀ j₁ j₂, k ≫ f j₁ = k ≫ f j₂) :
    wideEqualizer.lift k h ≫ wideEqualizer.ι f = k :=
  limit.lift_π _ _


/-- A morphism `k : W ⟶ X` satisfying `∀ j₁ j₂, k ≫ f j₁ = k ≫ f j₂` induces a morphism
    `l : W ⟶ wideEqualizer f` satisfying `l ≫ wideEqualizer.ι f = k`. -/
def wideEqualizer.lift' [Nonempty J] {W : C} (k : W ⟶ X) (h : ∀ j₁ j₂, k ≫ f j₁ = k ≫ f j₂) :
    { l : W ⟶ wideEqualizer f // l ≫ wideEqualizer.ι f = k } :=
  ⟨wideEqualizer.lift k h, wideEqualizer.lift_ι _ _⟩


/-- Two maps into a wide equalizer are equal if they are equal when composed with the wide
    equalizer map. -/
@[ext]
theorem wideEqualizer.hom_ext [Nonempty J] {W : C} {k l : W ⟶ wideEqualizer f}
    (h : k ≫ wideEqualizer.ι f = l ≫ wideEqualizer.ι f) : k = l :=
  Trident.IsLimit.hom_ext (limit.isLimit _) h


/-- A wide equalizer morphism is a monomorphism -/
instance wideEqualizer.ι_mono [Nonempty J] : Mono (wideEqualizer.ι f) where
  right_cancellation _ _ w := wideEqualizer.hom_ext w


/-- The wide equalizer morphism in any limit cone is a monomorphism. -/
theorem mono_of_isLimit_parallelFamily [Nonempty J] {c : Cone (parallelFamily f)} (i : IsLimit c) :
    Mono (Trident.ι c) where
  right_cancellation _ _ w := Trident.IsLimit.hom_ext i w


/-- `HasWideCoequalizer f g` represents a particular choice of colimiting cocone
for the parallel family of morphisms `f`.
-/
abbrev HasWideCoequalizer :=
  HasColimit (parallelFamily f)


/-- If a wide coequalizer of `f`, we can access an arbitrary choice of such by
    saying `wideCoequalizer f`. -/
abbrev wideCoequalizer : C :=
  colimit (parallelFamily f)


/-- If a wideCoequalizer of `f` exists, we can access the corresponding projection by
    saying `wideCoequalizer.π f`. -/
abbrev wideCoequalizer.π : Y ⟶ wideCoequalizer f :=
  colimit.ι (parallelFamily f) one


/-- An arbitrary choice of coequalizer cocone for a parallel family `f`.
-/
abbrev wideCoequalizer.cotrident : Cotrident f :=
  colimit.cocone (parallelFamily f)


@[simp]
theorem wideCoequalizer.cotrident_π : (wideCoequalizer.cotrident f).π = wideCoequalizer.π f :=
  rfl


@[simp 1100]
theorem wideCoequalizer.cotrident_ι_app_one :
    (wideCoequalizer.cotrident f).ι.app one = wideCoequalizer.π f :=
  rfl


@[reassoc]
theorem wideCoequalizer.condition (j₁ j₂ : J) :
    f j₁ ≫ wideCoequalizer.π f = f j₂ ≫ wideCoequalizer.π f :=
  Cotrident.condition j₁ j₂ <| colimit.cocone <| parallelFamily f


/-- The cotrident built from `wideCoequalizer.π f` is colimiting. -/
def wideCoequalizerIsWideCoequalizer [Nonempty J] :
    IsColimit (Cotrident.ofπ (wideCoequalizer.π f) (wideCoequalizer.condition f)) :=
                                                /-
                                                  J : Type w
                                                  C : Type u
                                                  inst✝² : CategoryTheory.Category.{v, u} C
                                                  X Y : C
                                                  f : J → Quiver.Hom X Y
                                                  inst✝¹ : CategoryTheory.Limits.HasWideCoequalizer f
                                                  inst✝ : Nonempty J
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cotrident.π (C …
                                                -/
  IsColimit.ofIsoColimit (colimit.isColimit _) (Cotrident.ext (Iso.refl _))
                                                /-
                                                  🎉 no goals
                                                -/


/-- Any morphism `k : Y ⟶ W` satisfying `∀ j₁ j₂, f j₁ ≫ k = f j₂ ≫ k` factors through the
    wide coequalizer of `f` via `wideCoequalizer.desc : wideCoequalizer f ⟶ W`. -/
abbrev wideCoequalizer.desc [Nonempty J] {W : C} (k : Y ⟶ W) (h : ∀ j₁ j₂, f j₁ ≫ k = f j₂ ≫ k) :
    wideCoequalizer f ⟶ W :=
  colimit.desc (parallelFamily f) (Cotrident.ofπ k h)


@[reassoc (attr := simp 1100)]
theorem wideCoequalizer.π_desc [Nonempty J] {W : C} (k : Y ⟶ W)
    (h : ∀ j₁ j₂, f j₁ ≫ k = f j₂ ≫ k) :
    wideCoequalizer.π f ≫ wideCoequalizer.desc k h = k :=
  colimit.ι_desc _ _


/-- Any morphism `k : Y ⟶ W` satisfying `∀ j₁ j₂, f j₁ ≫ k = f j₂ ≫ k` induces a morphism
    `l : wideCoequalizer f ⟶ W` satisfying `wideCoequalizer.π ≫ g = l`. -/
def wideCoequalizer.desc' [Nonempty J] {W : C} (k : Y ⟶ W) (h : ∀ j₁ j₂, f j₁ ≫ k = f j₂ ≫ k) :
    { l : wideCoequalizer f ⟶ W // wideCoequalizer.π f ≫ l = k } :=
  ⟨wideCoequalizer.desc k h, wideCoequalizer.π_desc _ _⟩


/-- Two maps from a wide coequalizer are equal if they are equal when composed with the wide
    coequalizer map -/
@[ext]
theorem wideCoequalizer.hom_ext [Nonempty J] {W : C} {k l : wideCoequalizer f ⟶ W}
    (h : wideCoequalizer.π f ≫ k = wideCoequalizer.π f ≫ l) : k = l :=
  Cotrident.IsColimit.hom_ext (colimit.isColimit _) h


/-- A wide coequalizer morphism is an epimorphism -/
instance wideCoequalizer.π_epi [Nonempty J] : Epi (wideCoequalizer.π f) where
  left_cancellation _ _ w := wideCoequalizer.hom_ext w


/-- The wide coequalizer morphism in any colimit cocone is an epimorphism. -/
theorem epi_of_isColimit_parallelFamily [Nonempty J] {c : Cocone (parallelFamily f)}
    (i : IsColimit c) : Epi (c.ι.app one) where
  left_cancellation _ _ w := Cotrident.IsColimit.hom_ext i w


/-- `HasWideEqualizers` represents a choice of wide equalizer for every family of morphisms -/
abbrev HasWideEqualizers :=
  ∀ J, HasLimitsOfShape (WalkingParallelFamily.{w} J) C


/-- `HasWideCoequalizers` represents a choice of wide coequalizer for every family of morphisms -/
abbrev HasWideCoequalizers :=
  ∀ J, HasColimitsOfShape (WalkingParallelFamily.{w} J) C


/-- If `C` has all limits of diagrams `parallelFamily f`, then it has all wide equalizers -/
theorem hasWideEqualizers_of_hasLimit_parallelFamily
    [∀ {J : Type w} {X Y : C} {f : J → (X ⟶ Y)}, HasLimit (parallelFamily f)] :
    HasWideEqualizers.{w} C := fun _ =>
  { has_limit := fun F => hasLimitOfIso (diagramIsoParallelFamily F).symm }


/-- If `C` has all colimits of diagrams `parallelFamily f`, then it has all wide coequalizers -/
theorem hasWideCoequalizers_of_hasColimit_parallelFamily
    [∀ {J : Type w} {X Y : C} {f : J → (X ⟶ Y)}, HasColimit (parallelFamily f)] :
    HasWideCoequalizers.{w} C := fun _ =>
  { has_colimit := fun F => hasColimitOfIso (diagramIsoParallelFamily F) }


instance (priority := 10) hasEqualizers_of_hasWideEqualizers [HasWideEqualizers.{w} C] :
    HasEqualizers C :=
  hasLimitsOfShape_of_equivalence.{w} walkingParallelFamilyEquivWalkingParallelPair


instance (priority := 10) hasCoequalizers_of_hasWideCoequalizers [HasWideCoequalizers.{w} C] :
    HasCoequalizers C :=
  hasColimitsOfShape_of_equivalence.{w} walkingParallelFamilyEquivWalkingParallelPair


