/-- `sum C D` gives the direct sum of two categories.
-/
instance sum : Category.{v₁} (C ⊕ D) where
  Hom X Y :=
    match X, Y with
    | inl X, inl Y => X ⟶ Y
    | inl _, inr _ => PEmpty
    | inr _, inl _ => PEmpty
    | inr X, inr Y => X ⟶ Y
  id X :=
    match X with
    | inl X => 𝟙 X
    | inr X => 𝟙 X
  comp {X Y Z} f g :=
    match X, Y, Z, f, g with
    | inl _, inl _, inl _, f, g => f ≫ g
    | inr _, inr _, inr _, f, g => f ≫ g
  assoc {W X Y Z} f g h :=
    match X, Y, Z, W with
    | inl _, inl _, inl _, inl _ => Category.assoc f g h
    | inr _, inr _, inr _, inr _ => Category.assoc f g h


@[aesop norm -10 destruct (rule_sets := [CategoryTheory])]
theorem hom_inl_inr_false {X : C} {Y : D} (f : Sum.inl X ⟶ Sum.inr Y) : False := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} D
    X : C
    Y : D
    f : Quiver.Hom (Sum.inl X) (Sum.inr Y)
    ⊢ False
  -/
  cases f
  /-
    🎉 no goals
  -/


@[aesop norm -10 destruct (rule_sets := [CategoryTheory])]
theorem hom_inr_inl_false {X : C} {Y : D} (f : Sum.inr X ⟶ Sum.inl Y) : False := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} D
    X : C
    Y : D
    f : Quiver.Hom (Sum.inr X) (Sum.inl Y)
    ⊢ False
  -/
  cases f
  /-
    🎉 no goals
  -/


theorem sum_comp_inl {P Q R : C} (f : (inl P : C ⊕ D) ⟶ inl Q) (g : (inl Q : C ⊕ D) ⟶ inl R) :
    @CategoryStruct.comp _ _ P Q R (f : P ⟶ Q) (g : Q ⟶ R) =
      @CategoryStruct.comp _ _ (inl P) (inl Q) (inl R) (f : P ⟶ Q) (g : Q ⟶ R) :=
  rfl


theorem sum_comp_inr {P Q R : D} (f : (inr P : C ⊕ D) ⟶ inr Q) (g : (inr Q : C ⊕ D) ⟶ inr R) :
    @CategoryStruct.comp _ _ P Q R (f : P ⟶ Q) (g : Q ⟶ R) =
      @CategoryStruct.comp _ _ (inr P) (inr Q) (inr R) (f : P ⟶ Q) (g : Q ⟶ R) :=
  rfl


/-- `inl_` is the functor `X ↦ inl X`. -/
@[simps]
def inl_ : C ⥤ C ⊕ D where
  obj X := inl X
  map {_ _} f := f


/-- `inr_` is the functor `X ↦ inr X`. -/
@[simps]
def inr_ : D ⥤ C ⊕ D where
  obj X := inr X
  map {_ _} f := f

/- Porting note: `aesop_cat` not firing on `map_comp` where autotac in Lean 3 did
but `map_id` was ok. -/


/-- The functor exchanging two direct summand categories. -/
def swap : C ⊕ D ⥤ D ⊕ C where
  obj X :=
    match X with
    | inl X => inr X
    | inr X => inl X
  map := @fun X Y f =>
    match X, Y, f with
    | inl _, inl _, f => f
    | inr _, inr _, f => f
  map_comp := fun {X} {Y} {Z} _ _ =>
    match X, Y, Z with
                                /-
                                  C : Type u₁
                                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                  D : Type u₁
                                  inst✝ : CategoryTheory.Category.{v₁, u₁} D
                                  X✝ Y✝ Z✝ : Sum C D
                                  X Y Z : C
                                  x✝¹ : Quiver.Hom (Sum.inl X) (Sum.inl Y)
                                  x✝ : Quiver.Hom (Sum.inl Y) (Sum.inl Z)
                                  ⊢ Eq ({ obj := fun X => CategoryTheory.sum.match_2 C D (fun X => Sum D C) X (f …
                                -/
    | inl X, inl Y, inl Z => by rfl
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  C : Type u₁
                                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                  D : Type u₁
                                  inst✝ : CategoryTheory.Category.{v₁, u₁} D
                                  X✝ Y✝ Z✝ : Sum C D
                                  X Y Z : D
                                  x✝¹ : Quiver.Hom (Sum.inr X) (Sum.inr Y)
                                  x✝ : Quiver.Hom (Sum.inr Y) (Sum.inr Z)
                                  ⊢ Eq ({ obj := fun X => CategoryTheory.sum.match_2 C D (fun X => Sum D C) X (f …
                                -/
    | inr X, inr Y, inr Z => by rfl
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem swap_obj_inl (X : C) : (swap C D).obj (inl X) = inr X :=
  rfl


@[simp]
theorem swap_obj_inr (X : D) : (swap C D).obj (inr X) = inl X :=
  rfl


@[simp]
theorem swap_map_inl {X Y : C} {f : inl X ⟶ inl Y} : (swap C D).map f = f :=
  rfl


@[simp]
theorem swap_map_inr {X Y : D} {f : inr X ⟶ inr Y} : (swap C D).map f = f :=
  rfl


/-- `swap` gives an equivalence between `C ⊕ D` and `D ⊕ C`. -/
@[simps functor inverse]
def equivalence : C ⊕ D ≌ D ⊕ C where
  functor := swap C D
  inverse := swap D C
                                     /-
                                       C : Type u₁
                                       inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                       D : Type u₁
                                       inst✝ : CategoryTheory.Category.{v₁, u₁} D
                                       ⊢ (X : Sum C D) → CategoryTheory.Iso ((CategoryTheory.Functor.id (Sum C D)).ob …
                                     -/
                                                      /-
                                                        🎉 no goals
                                                      -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  unitIso := NatIso.ofComponents (by rintro (_|_) <;> exact Iso.refl _)
             /-
               🎉 no goals
             -/
                                       /-
                                         C : Type u₁
                                         inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                         D : Type u₁
                                         inst✝ : CategoryTheory.Category.{v₁, u₁} D
                                         ⊢ (X : Sum D C) → CategoryTheory.Iso (((CategoryTheory.Sum.swap D C).comp (Cat …
                                       -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  counitIso := NatIso.ofComponents (by rintro (_|_) <;> exact Iso.refl _)
               /-
                 🎉 no goals
               -/


instance isEquivalence : (swap C D).IsEquivalence :=
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} D
        ⊢ (CategoryTheory.Sum.Swap.equivalence C D).functor.IsEquivalence
      -/
  (by infer_instance : (equivalence C D).functor.IsEquivalence)
      /-
        🎉 no goals
      -/


/-- The double swap on `C ⊕ D` is naturally isomorphic to the identity functor. -/
def symmetry : swap C D ⋙ swap D C ≅ 𝟭 (C ⊕ D) :=
  (equivalence C D).unitIso.symm


/-- The sum of two functors. -/
def sum (F : A ⥤ B) (G : C ⥤ D) : A ⊕ C ⥤ B ⊕ D where
  obj X :=
    match X with
    | inl X => inl (F.obj X)
    | inr X => inr (G.obj X)
  map {X Y} f :=
    match X, Y, f with
    | inl _, inl _, f => F.map f
    | inr _, inr _, f => G.map f
                   /-
                     A : Type u₁
                     inst✝³ : CategoryTheory.Category.{v₁, u₁} A
                     B : Type u₁
                     inst✝² : CategoryTheory.Category.{v₁, u₁} B
                     C : Type u₁
                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                     D : Type u₁
                     inst✝ : CategoryTheory.Category.{v₁, u₁} D
                     F : CategoryTheory.Functor A B
                     G : CategoryTheory.Functor C D
                     X : Sum A C
                     ⊢ Eq ({ obj := fun X => CategoryTheory.sum.match_2 A C (fun X => Sum B D) X (f …
                   -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  map_id {X} := by cases X <;> (erw [Functor.map_id]; rfl)
                                                      /-
                                                        🎉 no goals
                                                      -/
  map_comp {X Y Z} f g :=
    match X, Y, Z, f, g with
                                      /-
                                        A : Type u₁
                                        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
                                        B : Type u₁
                                        inst✝² : CategoryTheory.Category.{v₁, u₁} B
                                        C : Type u₁
                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                        D : Type u₁
                                        inst✝ : CategoryTheory.Category.{v₁, u₁} D
                                        F : CategoryTheory.Functor A B
                                        G : CategoryTheory.Functor C D
                                        X✝ Y✝ Z✝ : Sum A C
                                        f✝ : Quiver.Hom X✝ Y✝
                                        g✝ : Quiver.Hom Y✝ Z✝
                                        X Y Z : A
                                        f : Quiver.Hom (Sum.inl X) (Sum.inl Y)
                                        g : Quiver.Hom (Sum.inl Y) (Sum.inl Z)
                                        ⊢ Eq ({ obj := fun X => CategoryTheory.sum.match_2 A C (fun X => Sum B D) X (f …
                                      -/
    | inl X, inl Y, inl Z, f, g => by erw [F.map_comp]; rfl
                                                        /-
                                                          🎉 no goals
                                                        -/
                                      /-
                                        A : Type u₁
                                        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
                                        B : Type u₁
                                        inst✝² : CategoryTheory.Category.{v₁, u₁} B
                                        C : Type u₁
                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                        D : Type u₁
                                        inst✝ : CategoryTheory.Category.{v₁, u₁} D
                                        F : CategoryTheory.Functor A B
                                        G : CategoryTheory.Functor C D
                                        X✝ Y✝ Z✝ : Sum A C
                                        f✝ : Quiver.Hom X✝ Y✝
                                        g✝ : Quiver.Hom Y✝ Z✝
                                        X Y Z : C
                                        f : Quiver.Hom (Sum.inr X) (Sum.inr Y)
                                        g : Quiver.Hom (Sum.inr Y) (Sum.inr Z)
                                        ⊢ Eq ({ obj := fun X => CategoryTheory.sum.match_2 A C (fun X => Sum B D) X (f …
                                      -/
    | inr X, inr Y, inr Z, f, g => by erw [G.map_comp]; rfl
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- Similar to `sum`, but both functors land in the same category `C` -/
def sum' (F : A ⥤ C) (G : B ⥤ C) : A ⊕ B ⥤ C where
  obj X :=
    match X with
    | inl X => F.obj X
    | inr X => G.obj X
  map {X Y} f :=
    match X, Y, f with
    | inl _, inl _, f => F.map f
    | inr _, inr _, f => G.map f
                   /-
                     A : Type u₁
                     inst✝³ : CategoryTheory.Category.{v₁, u₁} A
                     B : Type u₁
                     inst✝² : CategoryTheory.Category.{v₁, u₁} B
                     C : Type u₁
                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                     D : Type u₁
                     inst✝ : CategoryTheory.Category.{v₁, u₁} D
                     F : CategoryTheory.Functor A C
                     G : CategoryTheory.Functor B C
                     X : Sum A B
                     ⊢ Eq ({ obj := fun X => CategoryTheory.sum.match_2 A B (fun X => C) X (fun X = …
                   -/
                               /-
                                 🎉 no goals
                               -/
  map_id {X} := by cases X <;> erw [Functor.map_id]
                               /-
                                 🎉 no goals
                               -/
  map_comp {X Y Z} f g :=
    match X, Y, Z, f, g with
                                      /-
                                        A : Type u₁
                                        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
                                        B : Type u₁
                                        inst✝² : CategoryTheory.Category.{v₁, u₁} B
                                        C : Type u₁
                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                        D : Type u₁
                                        inst✝ : CategoryTheory.Category.{v₁, u₁} D
                                        F : CategoryTheory.Functor A C
                                        G : CategoryTheory.Functor B C
                                        X Y Z : Sum A B
                                        f✝ : Quiver.Hom X Y
                                        g✝ : Quiver.Hom Y Z
                                        val✝² val✝¹ val✝ : A
                                        f : Quiver.Hom (Sum.inl val✝²) (Sum.inl val✝¹)
                                        g : Quiver.Hom (Sum.inl val✝¹) (Sum.inl val✝)
                                        ⊢ Eq ({ obj := fun X => CategoryTheory.sum.match_2 A B (fun X => C) X (fun X = …
                                      -/
    | inl _, inl _, inl _, f, g => by erw [F.map_comp]
                                      /-
                                        🎉 no goals
                                      -/
                                      /-
                                        A : Type u₁
                                        inst✝³ : CategoryTheory.Category.{v₁, u₁} A
                                        B : Type u₁
                                        inst✝² : CategoryTheory.Category.{v₁, u₁} B
                                        C : Type u₁
                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                        D : Type u₁
                                        inst✝ : CategoryTheory.Category.{v₁, u₁} D
                                        F : CategoryTheory.Functor A C
                                        G : CategoryTheory.Functor B C
                                        X Y Z : Sum A B
                                        f✝ : Quiver.Hom X Y
                                        g✝ : Quiver.Hom Y Z
                                        val✝² val✝¹ val✝ : B
                                        f : Quiver.Hom (Sum.inr val✝²) (Sum.inr val✝¹)
                                        g : Quiver.Hom (Sum.inr val✝¹) (Sum.inr val✝)
                                        ⊢ Eq ({ obj := fun X => CategoryTheory.sum.match_2 A B (fun X => C) X (fun X = …
                                      -/
    | inr _, inr _, inr _, f, g => by erw [G.map_comp]
                                      /-
                                        🎉 no goals
                                      -/


/-- The sum `F.sum' G` precomposed with the left inclusion functor is isomorphic to `F` -/
@[simps!]
def inlCompSum' (F : A ⥤ C) (G : B ⥤ C) : Sum.inl_ A B ⋙ F.sum' G ≅ F :=
  /-
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} B
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} D
    F : CategoryTheory.Functor A C
    G : CategoryTheory.Functor B C
    ⊢ ∀ {X Y : A} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


/-- The sum `F.sum' G` precomposed with the right inclusion functor is isomorphic to `G` -/
@[simps!]
def inrCompSum' (F : A ⥤ C) (G : B ⥤ C) : Sum.inr_ A B ⋙ F.sum' G ≅ G :=
  /-
    A : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} A
    B : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} B
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} D
    F : CategoryTheory.Functor A C
    G : CategoryTheory.Functor B C
    ⊢ ∀ {X Y : B} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


@[simp]
theorem sum_obj_inl (F : A ⥤ B) (G : C ⥤ D) (a : A) : (F.sum G).obj (inl a) = inl (F.obj a) :=
  rfl


@[simp]
theorem sum_obj_inr (F : A ⥤ B) (G : C ⥤ D) (c : C) : (F.sum G).obj (inr c) = inr (G.obj c) :=
  rfl


@[simp]
theorem sum_map_inl (F : A ⥤ B) (G : C ⥤ D) {a a' : A} (f : inl a ⟶ inl a') :
    (F.sum G).map f = F.map f :=
  rfl


@[simp]
theorem sum_map_inr (F : A ⥤ B) (G : C ⥤ D) {c c' : C} (f : inr c ⟶ inr c') :
    (F.sum G).map f = G.map f :=
  rfl


/-- The sum of two natural transformations. -/
def sum {F G : A ⥤ B} {H I : C ⥤ D} (α : F ⟶ G) (β : H ⟶ I) : F.sum H ⟶ G.sum I where
  app X :=
    match X with
    | inl X => α.app X
    | inr X => β.app X
  naturality X Y f :=
    match X, Y, f with
                            /-
                              A : Type u₁
                              inst✝³ : CategoryTheory.Category.{v₁, u₁} A
                              B : Type u₁
                              inst✝² : CategoryTheory.Category.{v₁, u₁} B
                              C : Type u₁
                              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                              D : Type u₁
                              inst✝ : CategoryTheory.Category.{v₁, u₁} D
                              F G : CategoryTheory.Functor A B
                              H I : CategoryTheory.Functor C D
                              α : Quiver.Hom F G
                              β : Quiver.Hom H I
                              X✝ Y✝ : Sum A C
                              f✝ : Quiver.Hom X✝ Y✝
                              X Y : A
                              f : Quiver.Hom (Sum.inl X) (Sum.inl Y)
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.sum H).map f) ((fun X => Category …
                            -/
    | inl X, inl Y, f => by erw [α.naturality]; rfl
                                                /-
                                                  🎉 no goals
                                                -/
                            /-
                              A : Type u₁
                              inst✝³ : CategoryTheory.Category.{v₁, u₁} A
                              B : Type u₁
                              inst✝² : CategoryTheory.Category.{v₁, u₁} B
                              C : Type u₁
                              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                              D : Type u₁
                              inst✝ : CategoryTheory.Category.{v₁, u₁} D
                              F G : CategoryTheory.Functor A B
                              H I : CategoryTheory.Functor C D
                              α : Quiver.Hom F G
                              β : Quiver.Hom H I
                              X✝ Y✝ : Sum A C
                              f✝ : Quiver.Hom X✝ Y✝
                              X Y : C
                              f : Quiver.Hom (Sum.inr X) (Sum.inr Y)
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.sum H).map f) ((fun X => Category …
                            -/
    | inr X, inr Y, f => by erw [β.naturality]; rfl
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem sum_app_inl {F G : A ⥤ B} {H I : C ⥤ D} (α : F ⟶ G) (β : H ⟶ I) (a : A) :
    (sum α β).app (inl a) = α.app a :=
  rfl


@[simp]
theorem sum_app_inr {F G : A ⥤ B} {H I : C ⥤ D} (α : F ⟶ G) (β : H ⟶ I) (c : C) :
    (sum α β).app (inr c) = β.app c :=
  rfl


