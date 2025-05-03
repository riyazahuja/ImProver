/-- `F ⊣ G` represents the data of an adjunction between two functors
`F : C ⥤ D` and `G : D ⥤ C`. `F` is the left adjoint and `G` is the right adjoint.

We use the unit-counit definition of an adjunction. There is a constructor `Adjunction.mk'`
which constructs an adjunction from the data of a hom set equivalence, a unit, and a counit,
together with proofs of the equalities `homEquiv_unit` and `homEquiv_counit` relating them to each
other.

There is also a constructor `Adjunction.mkOfHomEquiv` which constructs an adjunction from a natural
hom set equivalence.

To construct adjoints to a given functor, there are constructors `leftAdjointOfEquiv` and
`adjunctionOfEquivLeft` (as well as their duals).

See <https://stacks.math.columbia.edu/tag/0037>.
-/
structure Adjunction (F : C ⥤ D) (G : D ⥤ C) where
  /-- The unit of an adjunction -/
  unit : 𝟭 C ⟶ F.comp G
  /-- The counit of an adjunction -/
  counit : G.comp F ⟶ 𝟭 D
  /-- Equality of the composition of the unit and counit with the identity `F ⟶ FGF ⟶ F = 𝟙` -/
  left_triangle_components (X : C) :
      F.map (unit.app X) ≫ counit.app (F.obj X) = 𝟙 (F.obj X) := by aesop_cat
  /-- Equality of the composition of the unit and counit with the identity `G ⟶ GFG ⟶ G = 𝟙` -/
  right_triangle_components (Y : D) :
      unit.app (G.obj Y) ≫ G.map (counit.app Y) = 𝟙 (G.obj Y) := by aesop_cat


/-- The notation `F ⊣ G` stands for `Adjunction F G` representing that `F` is left adjoint to `G` -/
infixl:15 " ⊣ " => Adjunction


/-- A class asserting the existence of a right adjoint. -/
class IsLeftAdjoint (left : C ⥤ D) : Prop where
  exists_rightAdjoint : ∃ (right : D ⥤ C), Nonempty (left ⊣ right)


/-- A class asserting the existence of a left adjoint. -/
class IsRightAdjoint (right : D ⥤ C) : Prop where
  exists_leftAdjoint : ∃ (left : C ⥤ D), Nonempty (left ⊣ right)


/-- A chosen left adjoint to a functor that is a right adjoint. -/
noncomputable def leftAdjoint (R : D ⥤ C) [IsRightAdjoint R] : C ⥤ D :=
  (IsRightAdjoint.exists_leftAdjoint (right := R)).choose


/-- A chosen right adjoint to a functor that is a left adjoint. -/
noncomputable def rightAdjoint (L : C ⥤ D) [IsLeftAdjoint L] : D ⥤ C :=
  (IsLeftAdjoint.exists_rightAdjoint (left := L)).choose


/-- The adjunction associated to a functor known to be a left adjoint. -/
noncomputable def Adjunction.ofIsLeftAdjoint (left : C ⥤ D) [left.IsLeftAdjoint] :
    left ⊣ left.rightAdjoint :=
  Functor.IsLeftAdjoint.exists_rightAdjoint.choose_spec.some


/-- The adjunction associated to a functor known to be a right adjoint. -/
noncomputable def Adjunction.ofIsRightAdjoint (right : C ⥤ D) [right.IsRightAdjoint] :
    right.leftAdjoint ⊣ right :=
  Functor.IsRightAdjoint.exists_leftAdjoint.choose_spec.some


attribute [reassoc (attr := simp)] left_triangle_components right_triangle_components


/-- The hom set equivalence associated to an adjunction. -/
@[simps (config := .lemmasOnly)]
def homEquiv {F : C ⥤ D} {G : D ⥤ C} (adj : F ⊣ G) (X : C) (Y : D) :
    (F.obj X ⟶ Y) ≃ (X ⟶ G.obj Y) where
  toFun := fun f => adj.unit.app X ≫ G.map f
  invFun := fun g => F.map g ≫ adj.counit.app Y
  left_inv := fun f => by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      X : C
      Y : D
      f : Quiver.Hom (F.obj X) Y
      ⊢ Eq ((fun g => CategoryTheory.CategoryStruct.comp (F.map g) (adj.counit.app Y …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      X : C
      Y : D
      f : Quiver.Hom (F.obj X) Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
    -/
    rw [F.map_comp, assoc, ← Functor.comp_map, adj.counit.naturality, ← assoc]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      X : C
      Y : D
      f : Quiver.Hom (F.obj X) Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp
    /-
      🎉 no goals
    -/
  right_inv := fun g => by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      X : C
      Y : D
      g : Quiver.Hom X (G.obj Y)
      ⊢ Eq ((fun f => CategoryTheory.CategoryStruct.comp (adj.unit.app X) (G.map f)) …
    -/
    simp only [Functor.comp_obj, Functor.map_comp]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      X : C
      Y : D
      g : Quiver.Hom X (G.obj Y)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app X) (CategoryTheory.Cate …
    -/
    rw [← assoc, ← Functor.comp_map, ← adj.unit.naturality]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      X : C
      Y : D
      g : Quiver.Hom X (G.obj Y)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp
    /-
      🎉 no goals
    -/


alias homEquiv_unit := homEquiv_apply

alias homEquiv_counit := homEquiv_symm_apply


@[ext]
lemma ext {F : C ⥤ D} {G : D ⥤ C} {adj adj' : F ⊣ G}
    (h : adj.unit = adj'.unit) : adj = adj' := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj adj' : CategoryTheory.Adjunction F G
    h : Eq adj.unit adj'.unit
    ⊢ Eq adj adj'
  -/
  suffices h' : adj.counit = adj'.counit by cases adj; cases adj'; aesop
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj adj' : CategoryTheory.Adjunction F G
    h : Eq adj.unit adj'.unit
    ⊢ Eq adj.counit adj'.counit
  -/
  ext X
  /-
    case w.h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj adj' : CategoryTheory.Adjunction F G
    h : Eq adj.unit adj'.unit
    X : D
    ⊢ Eq (adj.counit.app X) (adj'.counit.app X)
  -/
  apply (adj.homEquiv _ _).injective
  rw [Adjunction.homEquiv_unit, Adjunction.homEquiv_unit,
    Adjunction.right_triangle_components, h, Adjunction.right_triangle_components]


lemma isLeftAdjoint (adj : F ⊣ G) : F.IsLeftAdjoint := ⟨_, ⟨adj⟩⟩


lemma isRightAdjoint (adj : F ⊣ G) : G.IsRightAdjoint := ⟨_, ⟨adj⟩⟩


instance (R : D ⥤ C) [R.IsRightAdjoint] : R.leftAdjoint.IsLeftAdjoint :=
  (ofIsRightAdjoint R).isLeftAdjoint


instance (L : C ⥤ D) [L.IsLeftAdjoint] : L.rightAdjoint.IsRightAdjoint :=
  (ofIsLeftAdjoint L).isRightAdjoint


                                                                            /-
                                                                              C : Type u₁
                                                                              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                              D : Type u₂
                                                                              inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                              F : CategoryTheory.Functor C D
                                                                              G : CategoryTheory.Functor D C
                                                                              adj : CategoryTheory.Adjunction F G
                                                                              X : C
                                                                              ⊢ Eq ((adj.homEquiv X (F.obj X)) (CategoryTheory.CategoryStruct.id (F.obj X))) …
                                                                            -/
theorem homEquiv_id (X : C) : adj.homEquiv X _ (𝟙 _) = adj.unit.app X := by simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


                                                                                          /-
                                                                                            C : Type u₁
                                                                                            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                            D : Type u₂
                                                                                            inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                                            F : CategoryTheory.Functor C D
                                                                                            G : CategoryTheory.Functor D C
                                                                                            adj : CategoryTheory.Adjunction F G
                                                                                            X : D
                                                                                            ⊢ Eq ((adj.homEquiv (G.obj X) X).symm (CategoryTheory.CategoryStruct.id (G.obj …
                                                                                          -/
theorem homEquiv_symm_id (X : D) : (adj.homEquiv _ X).symm (𝟙 _) = adj.counit.app X := by simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


theorem homEquiv_naturality_left_symm (f : X' ⟶ X) (g : X ⟶ G.obj Y) :
    (adj.homEquiv X' Y).symm (f ≫ g) = F.map f ≫ (adj.homEquiv X Y).symm g := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    X' X : C
    Y : D
    f : Quiver.Hom X' X
    g : Quiver.Hom X (G.obj Y)
    ⊢ Eq ((adj.homEquiv X' Y).symm (CategoryTheory.CategoryStruct.comp f g)) (Cate …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem homEquiv_naturality_left (f : X' ⟶ X) (g : F.obj X ⟶ Y) :
    (adj.homEquiv X' Y) (F.map f ≫ g) = f ≫ (adj.homEquiv X Y) g := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    X' X : C
    Y : D
    f : Quiver.Hom X' X
    g : Quiver.Hom (F.obj X) Y
    ⊢ Eq ((adj.homEquiv X' Y) (CategoryTheory.CategoryStruct.comp (F.map f) g)) (C …
  -/
  rw [← Equiv.eq_symm_apply]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    X' X : C
    Y : D
    f : Quiver.Hom X' X
    g : Quiver.Hom (F.obj X) Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) g) ((adj.homEquiv X' Y).sym …
  -/
  simp only [Equiv.symm_apply_apply, eq_self_iff_true, homEquiv_naturality_left_symm]
  /-
    🎉 no goals
  -/


theorem homEquiv_naturality_right (f : F.obj X ⟶ Y) (g : Y ⟶ Y') :
    (adj.homEquiv X Y') (f ≫ g) = (adj.homEquiv X Y) f ≫ G.map g := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    X : C
    Y Y' : D
    f : Quiver.Hom (F.obj X) Y
    g : Quiver.Hom Y Y'
    ⊢ Eq ((adj.homEquiv X Y') (CategoryTheory.CategoryStruct.comp f g)) (CategoryT …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem homEquiv_naturality_right_symm (f : X ⟶ G.obj Y) (g : Y ⟶ Y') :
    (adj.homEquiv X Y').symm (f ≫ G.map g) = (adj.homEquiv X Y).symm f ≫ g := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    X : C
    Y Y' : D
    f : Quiver.Hom X (G.obj Y)
    g : Quiver.Hom Y Y'
    ⊢ Eq ((adj.homEquiv X Y').symm (CategoryTheory.CategoryStruct.comp f (G.map g) …
  -/
  rw [Equiv.symm_apply_eq]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    X : C
    Y Y' : D
    f : Quiver.Hom X (G.obj Y)
    g : Quiver.Hom Y Y'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (G.map g)) ((adj.homEquiv X Y') (Ca …
  -/
  simp only [homEquiv_naturality_right, eq_self_iff_true, Equiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem homEquiv_naturality_left_square (f : X' ⟶ X) (g : F.obj X ⟶ Y')
    (h : F.obj X' ⟶ Y) (k : Y ⟶ Y') (w : F.map f ≫ g = h ≫ k) :
    f ≫ (adj.homEquiv X Y') g = (adj.homEquiv X' Y) h ≫ G.map k := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    X' X : C
    Y Y' : D
    f : Quiver.Hom X' X
    g : Quiver.Hom (F.obj X) Y'
    h : Quiver.Hom (F.obj X') Y
    k : Quiver.Hom Y Y'
    w : Eq (CategoryTheory.CategoryStruct.comp (F.map f) g) (CategoryTheory.Catego …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f ((adj.homEquiv X Y') g)) (CategoryT …
  -/
  rw [← homEquiv_naturality_left, ← homEquiv_naturality_right, w]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem homEquiv_naturality_right_square (f : X' ⟶ X) (g : X ⟶ G.obj Y')
    (h : X' ⟶ G.obj Y) (k : Y ⟶ Y') (w : f ≫ g = h ≫ G.map k) :
    F.map f ≫ (adj.homEquiv X Y').symm g = (adj.homEquiv X' Y).symm h ≫ k := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    X' X : C
    Y Y' : D
    f : Quiver.Hom X' X
    g : Quiver.Hom X (G.obj Y')
    h : Quiver.Hom X' (G.obj Y)
    k : Quiver.Hom Y Y'
    w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((adj.homEquiv X Y').symm g …
  -/
  rw [← homEquiv_naturality_left_symm, ← homEquiv_naturality_right_symm, w]
  /-
    🎉 no goals
  -/


theorem homEquiv_naturality_left_square_iff (f : X' ⟶ X) (g : F.obj X ⟶ Y')
    (h : F.obj X' ⟶ Y) (k : Y ⟶ Y') :
    (f ≫ (adj.homEquiv X Y') g = (adj.homEquiv X' Y) h ≫ G.map k) ↔
      (F.map f ≫ g = h ≫ k) :=
  ⟨fun w ↦ by simpa only [Equiv.symm_apply_apply]
      using homEquiv_naturality_right_square adj _ _ _ _ w,
    homEquiv_naturality_left_square adj f g h k⟩


theorem homEquiv_naturality_right_square_iff (f : X' ⟶ X) (g : X ⟶ G.obj Y')
    (h : X' ⟶ G.obj Y) (k : Y ⟶ Y') :
    (F.map f ≫ (adj.homEquiv X Y').symm g = (adj.homEquiv X' Y).symm h ≫ k) ↔
      (f ≫ g = h ≫ G.map k) :=
  ⟨fun w ↦ by simpa only [Equiv.apply_symm_apply]
      using homEquiv_naturality_left_square adj _ _ _ _ w,
    homEquiv_naturality_right_square adj f g h k⟩


@[simp]
theorem left_triangle : whiskerRight adj.unit F ≫ whiskerLeft F adj.counit = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight adj.unit …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp]
theorem right_triangle : whiskerLeft G adj.unit ≫ whiskerRight adj.counit G = 𝟙 _ := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction F G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft G adj.uni …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[reassoc (attr := simp)]
theorem counit_naturality {X Y : D} (f : X ⟶ Y) :
    F.map (G.map f) ≫ adj.counit.app Y = adj.counit.app X ≫ f :=
  adj.counit.naturality f


@[reassoc (attr := simp)]
theorem unit_naturality {X Y : C} (f : X ⟶ Y) :
    adj.unit.app X ≫ G.map (F.map f) = f ≫ adj.unit.app Y :=
  (adj.unit.naturality f).symm


lemma unit_comp_map_eq_iff {A : C} {B : D} (f : F.obj A ⟶ B) (g : A ⟶ G.obj B) :
    adj.unit.app A ≫ G.map f = g ↔ f = F.map g ≫ adj.counit.app B :=
               /-
                 C : Type u₁
                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                 D : Type u₂
                 inst✝ : CategoryTheory.Category.{v₂, u₂} D
                 F : CategoryTheory.Functor C D
                 G : CategoryTheory.Functor D C
                 adj : CategoryTheory.Adjunction F G
                 A : C
                 B : D
                 f : Quiver.Hom (F.obj A) B
                 g : Quiver.Hom A (G.obj B)
                 h : Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app A) (G.map f)) g
                 ⊢ Eq f (CategoryTheory.CategoryStruct.comp (F.map g) (adj.counit.app B))
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun h => by simp [← h], fun h => by simp [h]⟩
                                       /-
                                         🎉 no goals
                                       -/


lemma eq_unit_comp_map_iff {A : C} {B : D} (f : F.obj A ⟶ B) (g : A ⟶ G.obj B) :
    g = adj.unit.app A ≫ G.map f ↔ F.map g ≫ adj.counit.app B = f :=
               /-
                 C : Type u₁
                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                 D : Type u₂
                 inst✝ : CategoryTheory.Category.{v₂, u₂} D
                 F : CategoryTheory.Functor C D
                 G : CategoryTheory.Functor D C
                 adj : CategoryTheory.Adjunction F G
                 A : C
                 B : D
                 f : Quiver.Hom (F.obj A) B
                 g : Quiver.Hom A (G.obj B)
                 h : Eq g (CategoryTheory.CategoryStruct.comp (adj.unit.app A) (G.map f))
                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map g) (adj.counit.app B)) f
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun h => by simp [h], fun h => by simp [← h]⟩
                                     /-
                                       🎉 no goals
                                     -/


theorem homEquiv_apply_eq {A : C} {B : D} (f : F.obj A ⟶ B) (g : A ⟶ G.obj B) :
    adj.homEquiv A B f = g ↔ f = (adj.homEquiv A B).symm g :=
  unit_comp_map_eq_iff adj f g


theorem eq_homEquiv_apply {A : C} {B : D} (f : F.obj A ⟶ B) (g : A ⟶ G.obj B) :
    g = adj.homEquiv A B f ↔ (adj.homEquiv A B).symm g = f :=
  eq_unit_comp_map_iff adj f g


/--  If `adj : F ⊣ G`, and `X : C`, then `F.obj X` corepresents `Y ↦ (X ⟶ G.obj Y)`-/
@[simps]
def corepresentableBy (X : C) :
    (G ⋙ coyoneda.obj (Opposite.op X)).CorepresentableBy (F.obj X) where
  homEquiv := adj.homEquiv _ _
                      /-
                        C : Type u₁
                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                        D : Type u₂
                        inst✝ : CategoryTheory.Category.{v₂, u₂} D
                        F : CategoryTheory.Functor C D
                        G : CategoryTheory.Functor D C
                        adj : CategoryTheory.Adjunction F G
                        X' X✝ : C
                        Y Y' : D
                        X : C
                        ⊢ ∀ {Y Y' : D} (g : Quiver.Hom Y Y') (f : Quiver.Hom (F.obj X) Y), Eq ((fun {Y …
                      -/
  homEquiv_comp := by aesop_cat
                      /-
                        🎉 no goals
                      -/


/--  If `adj : F ⊣ G`, and `Y : D`, then `G.obj Y` represents `X ↦ (F.obj X ⟶ Y)`-/
@[simps]
def representableBy (Y : D) :
    (F.op ⋙ yoneda.obj Y).RepresentableBy (G.obj Y) where
  homEquiv := (adj.homEquiv _ _).symm
                      /-
                        C : Type u₁
                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                        D : Type u₂
                        inst✝ : CategoryTheory.Category.{v₂, u₂} D
                        F : CategoryTheory.Functor C D
                        G : CategoryTheory.Functor D C
                        adj : CategoryTheory.Adjunction F G
                        X' X : C
                        Y✝ Y' Y : D
                        ⊢ ∀ {X X' : C} (f : Quiver.Hom X X') (g : Quiver.Hom X' (G.obj Y)), Eq ((fun { …
                      -/
  homEquiv_comp := by aesop_cat
                      /-
                        🎉 no goals
                      -/


/--
This is an auxiliary data structure useful for constructing adjunctions.
See `Adjunction.mk'`. This structure won't typically be used anywhere else.
-/
structure CoreHomEquivUnitCounit (F : C ⥤ D) (G : D ⥤ C) where
  /-- The equivalence between `Hom (F X) Y` and `Hom X (G Y)` coming from an adjunction -/
  homEquiv : ∀ X Y, (F.obj X ⟶ Y) ≃ (X ⟶ G.obj Y)
  /-- The unit of an adjunction -/
  unit : 𝟭 C ⟶ F ⋙ G
  /-- The counit of an adjunction -/
  counit : G ⋙ F ⟶ 𝟭 D
  /-- The relationship between the unit and hom set equivalence of an adjunction -/
  homEquiv_unit : ∀ {X Y f}, (homEquiv X Y) f = unit.app X ≫ G.map f := by aesop_cat
  /-- The relationship between the counit and hom set equivalence of an adjunction -/
  homEquiv_counit : ∀ {X Y g}, (homEquiv X Y).symm g = F.map g ≫ counit.app Y := by aesop_cat


/-- This is an auxiliary data structure useful for constructing adjunctions.
See `Adjunction.mkOfHomEquiv`.
This structure won't typically be used anywhere else.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): `has_nonempty_instance` linter not ported yet
-- @[nolint has_nonempty_instance]
structure CoreHomEquiv (F : C ⥤ D) (G : D ⥤ C) where
  /-- The equivalence between `Hom (F X) Y` and `Hom X (G Y)` -/
  homEquiv : ∀ X Y, (F.obj X ⟶ Y) ≃ (X ⟶ G.obj Y)
  /-- The property that describes how `homEquiv.symm` transforms compositions `X' ⟶ X ⟶ G Y` -/
  homEquiv_naturality_left_symm :
    ∀ {X' X Y} (f : X' ⟶ X) (g : X ⟶ G.obj Y),
      (homEquiv X' Y).symm (f ≫ g) = F.map f ≫ (homEquiv X Y).symm g := by
    aesop_cat
  /-- The property that describes how `homEquiv` transforms compositions `F X ⟶ Y ⟶ Y'` -/
  homEquiv_naturality_right :
    ∀ {X Y Y'} (f : F.obj X ⟶ Y) (g : Y ⟶ Y'),
      (homEquiv X Y') (f ≫ g) = (homEquiv X Y) f ≫ G.map g := by
    aesop_cat


theorem homEquiv_naturality_left (f : X' ⟶ X) (g : F.obj X ⟶ Y) :
    (adj.homEquiv X' Y) (F.map f ≫ g) = f ≫ (adj.homEquiv X Y) g := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction.CoreHomEquiv F G
    X' X : C
    Y : D
    f : Quiver.Hom X' X
    g : Quiver.Hom (F.obj X) Y
    ⊢ Eq ((adj.homEquiv X' Y) (CategoryTheory.CategoryStruct.comp (F.map f) g)) (C …
  -/
  rw [← Equiv.eq_symm_apply]; simp
                              /-
                                🎉 no goals
                              -/


theorem homEquiv_naturality_right_symm (f : X ⟶ G.obj Y) (g : Y ⟶ Y') :
    (adj.homEquiv X Y').symm (f ≫ G.map g) = (adj.homEquiv X Y).symm f ≫ g := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction.CoreHomEquiv F G
    X : C
    Y Y' : D
    f : Quiver.Hom X (G.obj Y)
    g : Quiver.Hom Y Y'
    ⊢ Eq ((adj.homEquiv X Y').symm (CategoryTheory.CategoryStruct.comp f (G.map g) …
  -/
  rw [Equiv.symm_apply_eq]; simp
                            /-
                              🎉 no goals
                            -/


/-- This is an auxiliary data structure useful for constructing adjunctions.
See `Adjunction.mkOfUnitCounit`.
This structure won't typically be used anywhere else.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): `has_nonempty_instance` linter not ported yet
-- @[nolint has_nonempty_instance]
structure CoreUnitCounit (F : C ⥤ D) (G : D ⥤ C) where
  /-- The unit of an adjunction between `F` and `G` -/
  unit : 𝟭 C ⟶ F.comp G
  /-- The counit of an adjunction between `F` and `G`s -/
  counit : G.comp F ⟶ 𝟭 D
  /-- Equality of the composition of the unit, associator, and counit with the identity
  `F ⟶ (F G) F ⟶ F (G F) ⟶ F = NatTrans.id F` -/
  left_triangle :
    whiskerRight unit F ≫ (Functor.associator F G F).hom ≫ whiskerLeft F counit =
      NatTrans.id (𝟭 C ⋙ F) := by
    aesop_cat
  /-- Equality of the composition of the unit, associator, and counit with the identity
  `G ⟶ G (F G) ⟶ (F G) F ⟶ G = NatTrans.id G` -/
  right_triangle :
    whiskerLeft G unit ≫ (Functor.associator G F G).inv ≫ whiskerRight counit G =
      NatTrans.id (G ⋙ 𝟭 C) := by
    aesop_cat


/--
Construct an adjunction from the data of a `CoreHomEquivUnitCounit`, i.e. a hom set
equivalence, unit and counit natural transformations together with proofs of the equalities
`homEquiv_unit` and `homEquiv_counit` relating them to each other.
-/
@[simps]
def mk' (adj : CoreHomEquivUnitCounit F G) : F ⊣ G where
  unit := adj.unit
  counit := adj.counit
  left_triangle_components X := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction.CoreHomEquivUnitCounit F G
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (adj.unit.app X)) (adj.counit. …
    -/
    rw [← adj.homEquiv_counit, (adj.homEquiv _ _).symm_apply_eq, adj.homEquiv_unit]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction.CoreHomEquivUnitCounit F G
      X : C
      ⊢ Eq (adj.unit.app X) (CategoryTheory.CategoryStruct.comp (adj.unit.app ((Cate …
    -/
    simp
    /-
      🎉 no goals
    -/
  right_triangle_components Y := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction.CoreHomEquivUnitCounit F G
      Y : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (G.obj Y)) (G.map (adj. …
    -/
    rw [← adj.homEquiv_unit, ← (adj.homEquiv _ _).eq_symm_apply, adj.homEquiv_counit]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction.CoreHomEquivUnitCounit F G
      Y : D
      ⊢ Eq (adj.counit.app Y) (CategoryTheory.CategoryStruct.comp (F.map (CategoryTh …
    -/
    simp
    /-
      🎉 no goals
    -/


lemma mk'_homEquiv (adj : CoreHomEquivUnitCounit F G) : (mk' adj).homEquiv = adj.homEquiv := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction.CoreHomEquivUnitCounit F G
    ⊢ Eq (CategoryTheory.Adjunction.mk' adj).homEquiv adj.homEquiv
  -/
  ext
  /-
    case h.h.H
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction.CoreHomEquivUnitCounit F G
    x✝² : C
    x✝¹ : D
    x✝ : Quiver.Hom (F.obj x✝²) x✝¹
    ⊢ Eq (((CategoryTheory.Adjunction.mk' adj).homEquiv x✝² x✝¹) x✝) ((adj.homEqui …
  -/
  rw [homEquiv_unit, adj.homEquiv_unit, mk'_unit]
  /-
    🎉 no goals
  -/


/-- Construct an adjunction between `F` and `G` out of a natural bijection between each
`F.obj X ⟶ Y` and `X ⟶ G.obj Y`. -/
@[simps!]
def mkOfHomEquiv (adj : CoreHomEquiv F G) : F ⊣ G :=
  mk' {
    unit :=
      { app := fun X => (adj.homEquiv X (F.obj X)) (𝟙 (F.obj X))
        naturality := by
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            F : CategoryTheory.Functor C D
            G : CategoryTheory.Functor D C
            adj : CategoryTheory.Adjunction.CoreHomEquiv F G
            ⊢ ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((C …
          -/
          intros
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            F : CategoryTheory.Functor C D
            G : CategoryTheory.Functor D C
            adj : CategoryTheory.Adjunction.CoreHomEquiv F G
            X✝ Y✝ : C
            f✝ : Quiver.Hom X✝ Y✝
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).map f✝ …
          -/
          simp [← adj.homEquiv_naturality_left, ← adj.homEquiv_naturality_right] }
          /-
            🎉 no goals
          -/
    counit :=
      { app := fun Y => (adj.homEquiv _ _).invFun (𝟙 (G.obj Y))
        naturality := by
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            F : CategoryTheory.Functor C D
            G : CategoryTheory.Functor D C
            adj : CategoryTheory.Adjunction.CoreHomEquiv F G
            ⊢ ∀ ⦃X Y : D⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((G …
          -/
          intros
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} D
            F : CategoryTheory.Functor C D
            G : CategoryTheory.Functor D C
            adj : CategoryTheory.Adjunction.CoreHomEquiv F G
            X✝ Y✝ : D
            f✝ : Quiver.Hom X✝ Y✝
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.comp F).map f✝) ((fun Y => (adj.h …
          -/
          simp [← adj.homEquiv_naturality_left_symm, ← adj.homEquiv_naturality_right_symm] }
          /-
            🎉 no goals
          -/
    homEquiv := adj.homEquiv
                                       /-
                                         C : Type u₁
                                         inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                         D : Type u₂
                                         inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                         F : CategoryTheory.Functor C D
                                         G : CategoryTheory.Functor D C
                                         adj : CategoryTheory.Adjunction.CoreHomEquiv F G
                                         X : C
                                         Y : D
                                         f : Quiver.Hom (F.obj X) Y
                                         ⊢ Eq ((adj.homEquiv X Y) f) (CategoryTheory.CategoryStruct.comp ({ app := fun  …
                                       -/
    homEquiv_unit := fun {X Y f} => by simp [← adj.homEquiv_naturality_right]
                                       /-
                                         🎉 no goals
                                       -/
                                         /-
                                           C : Type u₁
                                           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                           D : Type u₂
                                           inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                           F : CategoryTheory.Functor C D
                                           G : CategoryTheory.Functor D C
                                           adj : CategoryTheory.Adjunction.CoreHomEquiv F G
                                           X : C
                                           Y : D
                                           f : Quiver.Hom X (G.obj Y)
                                           ⊢ Eq ((adj.homEquiv X Y).symm f) (CategoryTheory.CategoryStruct.comp (F.map f) …
                                         -/
    homEquiv_counit := fun {X Y f} => by simp [← adj.homEquiv_naturality_left_symm] }
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
lemma mkOfHomEquiv_homEquiv (adj : CoreHomEquiv F G) :
    (mkOfHomEquiv adj).homEquiv = adj.homEquiv := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction.CoreHomEquiv F G
    ⊢ Eq (CategoryTheory.Adjunction.mkOfHomEquiv adj).homEquiv adj.homEquiv
  -/
  ext X Y g
  /-
    case h.h.H
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction.CoreHomEquiv F G
    X : C
    Y : D
    g : Quiver.Hom (F.obj X) Y
    ⊢ Eq (((CategoryTheory.Adjunction.mkOfHomEquiv adj).homEquiv X Y) g) ((adj.hom …
  -/
  simp [mkOfHomEquiv, ← adj.homEquiv_naturality_right (𝟙 _) g]
  /-
    🎉 no goals
  -/


/-- Construct an adjunction between functors `F` and `G` given a unit and counit for the adjunction
satisfying the triangle identities. -/
@[simps!]
def mkOfUnitCounit (adj : CoreUnitCounit F G) : F ⊣ G where
  unit := adj.unit
  counit := adj.counit
  left_triangle_components X := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction.CoreUnitCounit F G
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (adj.unit.app X)) (adj.counit. …
    -/
    have := adj.left_triangle
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction.CoreUnitCounit F G
      X : C
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight adj …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (adj.unit.app X)) (adj.counit. …
    -/
    rw [NatTrans.ext_iff, funext_iff] at this
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction.CoreUnitCounit F G
      X : C
      this : ∀ (x : C), Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whis …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (adj.unit.app X)) (adj.counit. …
    -/
    simpa [-CoreUnitCounit.left_triangle] using this X
    /-
      🎉 no goals
    -/
  right_triangle_components Y := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction.CoreUnitCounit F G
      Y : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (G.obj Y)) (G.map (adj. …
    -/
    have := adj.right_triangle
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction.CoreUnitCounit F G
      Y : D
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft G ad …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (G.obj Y)) (G.map (adj. …
    -/
    rw [NatTrans.ext_iff, funext_iff] at this
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction.CoreUnitCounit F G
      Y : D
      this : ∀ (x : D), Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whis …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (G.obj Y)) (G.map (adj. …
    -/
    simpa [-CoreUnitCounit.right_triangle] using this Y
    /-
      🎉 no goals
    -/


/-- The adjunction between the identity functor on a category and itself. -/
def id : 𝟭 C ⊣ 𝟭 C where
  unit := 𝟙 _
  counit := 𝟙 _

-- Satisfy the inhabited linter.

instance : Inhabited (Adjunction (𝟭 C) (𝟭 C)) :=
  ⟨id⟩


/-- If F and G are naturally isomorphic functors, establish an equivalence of hom-sets. -/
@[simps]
def equivHomsetLeftOfNatIso {F F' : C ⥤ D} (iso : F ≅ F') {X : C} {Y : D} :
    (F.obj X ⟶ Y) ≃ (F'.obj X ⟶ Y) where
  toFun f := iso.inv.app _ ≫ f
  invFun g := iso.hom.app _ ≫ g
                   /-
                     C : Type u₁
                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                     D : Type u₂
                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                     F✝ : CategoryTheory.Functor C D
                     G : CategoryTheory.Functor D C
                     F F' : CategoryTheory.Functor C D
                     iso : CategoryTheory.Iso F F'
                     X : C
                     Y : D
                     f : Quiver.Hom (F.obj X) Y
                     ⊢ Eq ((fun g => CategoryTheory.CategoryStruct.comp (iso.hom.app X) g) ((fun f  …
                   -/
  left_inv f := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      C : Type u₁
                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                      D : Type u₂
                      inst✝ : CategoryTheory.Category.{v₂, u₂} D
                      F✝ : CategoryTheory.Functor C D
                      G : CategoryTheory.Functor D C
                      F F' : CategoryTheory.Functor C D
                      iso : CategoryTheory.Iso F F'
                      X : C
                      Y : D
                      g : Quiver.Hom (F'.obj X) Y
                      ⊢ Eq ((fun f => CategoryTheory.CategoryStruct.comp (iso.inv.app X) f) ((fun g  …
                    -/
  right_inv g := by simp
                    /-
                      🎉 no goals
                    -/


/-- If G and H are naturally isomorphic functors, establish an equivalence of hom-sets. -/
@[simps]
def equivHomsetRightOfNatIso {G G' : D ⥤ C} (iso : G ≅ G') {X : C} {Y : D} :
    (X ⟶ G.obj Y) ≃ (X ⟶ G'.obj Y) where
  toFun f := f ≫ iso.hom.app _
  invFun g := g ≫ iso.inv.app _
                   /-
                     C : Type u₁
                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                     D : Type u₂
                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                     F : CategoryTheory.Functor C D
                     G✝ G G' : CategoryTheory.Functor D C
                     iso : CategoryTheory.Iso G G'
                     X : C
                     Y : D
                     f : Quiver.Hom X (G.obj Y)
                     ⊢ Eq ((fun g => CategoryTheory.CategoryStruct.comp g (iso.inv.app Y)) ((fun f  …
                   -/
  left_inv f := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      C : Type u₁
                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                      D : Type u₂
                      inst✝ : CategoryTheory.Category.{v₂, u₂} D
                      F : CategoryTheory.Functor C D
                      G✝ G G' : CategoryTheory.Functor D C
                      iso : CategoryTheory.Iso G G'
                      X : C
                      Y : D
                      g : Quiver.Hom X (G'.obj Y)
                      ⊢ Eq ((fun f => CategoryTheory.CategoryStruct.comp f (iso.hom.app Y)) ((fun g  …
                    -/
  right_inv g := by simp
                    /-
                      🎉 no goals
                    -/


/-- Transport an adjunction along a natural isomorphism on the left. -/
def ofNatIsoLeft {F G : C ⥤ D} {H : D ⥤ C} (adj : F ⊣ H) (iso : F ≅ G) : G ⊣ H :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun X Y => (equivHomsetLeftOfNatIso iso.symm).trans (adj.homEquiv X Y) }


/-- Transport an adjunction along a natural isomorphism on the right. -/
def ofNatIsoRight {F : C ⥤ D} {G H : D ⥤ C} (adj : F ⊣ G) (iso : G ≅ H) : F ⊣ H :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun X Y => (adj.homEquiv X Y).trans (equivHomsetRightOfNatIso iso) }


/-- The isomorpism which an adjunction `F ⊣ G` induces on `G ⋙ yoneda`. This states that
`Adjunction.homEquiv` is natural in both arguments. -/
@[simps!]
def compYonedaIso {C : Type u₁} [Category.{v₁} C] {D : Type u₂} [Category.{v₁} D]
    {F : C ⥤ D} {G : D ⥤ C} (adj : F ⊣ G) :
    G ⋙ yoneda ≅ yoneda ⋙ (whiskeringLeft _ _ _).obj F.op :=
                               /-
                                 C✝ : Type u₁
                                 inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
                                 D✝ : Type u₂
                                 inst✝² : CategoryTheory.Category.{v₂, u₂} D✝
                                 F✝ : CategoryTheory.Functor C✝ D✝
                                 G✝ : CategoryTheory.Functor D✝ C✝
                                 C : Type u₁
                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                 D : Type u₂
                                 inst✝ : CategoryTheory.Category.{v₁, u₂} D
                                 F : CategoryTheory.Functor C D
                                 G : CategoryTheory.Functor D C
                                 adj : CategoryTheory.Adjunction F G
                                 X : D
                                 ⊢ ∀ {X_1 Y : Opposite C} (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.CategorySt …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun X => NatIso.ofComponents fun Y => (adj.homEquiv Y.unop X).toIso.symm
  /-
    🎉 no goals
  -/


/-- The isomorpism which an adjunction `F ⊣ G` induces on `F.op ⋙ coyoneda`. This states that
`Adjunction.homEquiv` is natural in both arguments. -/
@[simps!]
def compCoyonedaIso {C : Type u₁} [Category.{v₁} C] {D : Type u₂} [Category.{v₁} D]
    {F : C ⥤ D} {G : D ⥤ C} (adj : F ⊣ G) :
    F.op ⋙ coyoneda ≅ coyoneda ⋙ (whiskeringLeft _ _ _).obj G :=
                               /-
                                 C✝ : Type u₁
                                 inst✝³ : CategoryTheory.Category.{v₁, u₁} C✝
                                 D✝ : Type u₂
                                 inst✝² : CategoryTheory.Category.{v₂, u₂} D✝
                                 F✝ : CategoryTheory.Functor C✝ D✝
                                 G✝ : CategoryTheory.Functor D✝ C✝
                                 C : Type u₁
                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                 D : Type u₂
                                 inst✝ : CategoryTheory.Category.{v₁, u₂} D
                                 F : CategoryTheory.Functor C D
                                 G : CategoryTheory.Functor D C
                                 adj : CategoryTheory.Adjunction F G
                                 X : Opposite C
                                 ⊢ ∀ {X_1 Y : D} (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.CategoryStruct.comp …
                               -/
                               /-
                                 🎉 no goals
                               -/
  NatIso.ofComponents fun X => NatIso.ofComponents fun Y => (adj.homEquiv X.unop Y).toIso
  /-
    🎉 no goals
  -/


/-- Composition of adjunctions.

See <https://stacks.math.columbia.edu/tag/0DV0>.
-/
def comp : F ⋙ H ⊣ I ⋙ G :=
  mk' {
    homEquiv := fun _ _ ↦ Equiv.trans (adj₂.homEquiv _ _) (adj₁.homEquiv _ _)
    unit := adj₁.unit ≫ (whiskerLeft F <| whiskerRight adj₂.unit G) ≫ (Functor.associator _ _ _).inv
    counit :=
      (Functor.associator _ _ _).hom ≫ (whiskerLeft I <| whiskerRight adj₁.counit H) ≫ adj₂.counit }


@[simp, reassoc]
lemma comp_unit_app (X : C) :
    (adj₁.comp adj₂).unit.app X = adj₁.unit.app X ≫ G.map (adj₂.unit.app (F.obj X)) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    E : Type u₃
    ℰ : CategoryTheory.Category.{v₃, u₃} E
    H : CategoryTheory.Functor D E
    I : CategoryTheory.Functor E D
    adj₁ : CategoryTheory.Adjunction F G
    adj₂ : CategoryTheory.Adjunction H I
    X : C
    ⊢ Eq ((adj₁.comp adj₂).unit.app X) (CategoryTheory.CategoryStruct.comp (adj₁.u …
  -/
  simp [Adjunction.comp]
  /-
    🎉 no goals
  -/


@[simp, reassoc]
lemma comp_counit_app (X : E) :
    (adj₁.comp adj₂).counit.app X = H.map (adj₁.counit.app (I.obj X)) ≫ adj₂.counit.app X := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    E : Type u₃
    ℰ : CategoryTheory.Category.{v₃, u₃} E
    H : CategoryTheory.Functor D E
    I : CategoryTheory.Functor E D
    adj₁ : CategoryTheory.Adjunction F G
    adj₂ : CategoryTheory.Adjunction H I
    X : E
    ⊢ Eq ((adj₁.comp adj₂).counit.app X) (CategoryTheory.CategoryStruct.comp (H.ma …
  -/
  simp [Adjunction.comp]
  /-
    🎉 no goals
  -/


lemma comp_homEquiv :  (adj₁.comp adj₂).homEquiv =
    fun _ _ ↦ Equiv.trans (adj₂.homEquiv _ _) (adj₁.homEquiv _ _) :=
  mk'_homEquiv _


/-- Construct a left adjoint functor to `G`, given the functor's value on objects `F_obj` and
a bijection `e` between `F_obj X ⟶ Y` and `X ⟶ G.obj Y` satisfying a naturality law
`he : ∀ X Y Y' g h, e X Y' (h ≫ g) = e X Y h ≫ G.map g`.
Dual to `rightAdjointOfEquiv`. -/
@[simps!]
def leftAdjointOfEquiv (he : ∀ X Y Y' g h, e X Y' (h ≫ g) = e X Y h ≫ G.map g) : C ⥤ D where
  obj := F_obj
  map {X} {X'} f := (e X (F_obj X')).symm (f ≫ e X' (F_obj X') (𝟙 _))
  map_comp := fun f f' => by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      F_obj : C → D
      e : (X : C) → (Y : D) → Equiv (Quiver.Hom (F_obj X) Y) (Quiver.Hom X (G.obj Y))
      he : ∀ (X : C) (Y Y' : D) (g : Quiver.Hom Y Y') (h : Quiver.Hom (F_obj X) Y),  …
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      f' : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := F_obj, map := fun {X X'} f => (e X (F_obj X')).symm (CategoryTh …
    -/
    rw [Equiv.symm_apply_eq, he, Equiv.apply_symm_apply]
    conv =>
      rhs
      rw [assoc, ← he, id_comp, Equiv.apply_symm_apply]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      F_obj : C → D
      e : (X : C) → (Y : D) → Equiv (Quiver.Hom (F_obj X) Y) (Quiver.Hom X (G.obj Y))
      he : ∀ (X : C) (Y Y' : D) (g : Quiver.Hom Y Y') (h : Quiver.Hom (F_obj X) Y),  …
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      f' : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Show that the functor given by `leftAdjointOfEquiv` is indeed left adjoint to `G`. Dual
to `adjunctionOfRightEquiv`. -/
@[simps!]
def adjunctionOfEquivLeft : leftAdjointOfEquiv e he ⊣ G :=
  mkOfHomEquiv
    { homEquiv := e
      homEquiv_naturality_left_symm := fun {X'} {X} {Y} f g => by
        have {X : C} {Y Y' : D} (f : X ⟶ G.obj Y) (g : Y ⟶ Y') :
            (e X Y').symm (f ≫ G.map g) = (e X Y).symm f ≫ g := by
          rw [Equiv.symm_apply_eq, he]; simp
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          G : CategoryTheory.Functor D C
          F_obj : C → D
          e : (X : C) → (Y : D) → Equiv (Quiver.Hom (F_obj X) Y) (Quiver.Hom X (G.obj Y))
          he : ∀ (X : C) (Y Y' : D) (g : Quiver.Hom Y Y') (h : Quiver.Hom (F_obj X) Y),  …
          X' X : C
          Y : D
          f : Quiver.Hom X' X
          g : Quiver.Hom X (G.obj Y)
          this : ∀ {X : C} {Y Y' : D} (f : Quiver.Hom X (G.obj Y)) (g : Quiver.Hom Y Y') …
          ⊢ Eq ((e X' Y).symm (CategoryTheory.CategoryStruct.comp f g)) (CategoryTheory. …
        -/
        simp [← this, ← Equiv.apply_eq_iff_eq (e X' Y), ← he] }
        /-
          🎉 no goals
        -/


private theorem he'' (he : ∀ X' X Y f g, e X' Y (F.map f ≫ g) = f ≫ e X Y g)
    {X' X Y} (f g) : F.map f ≫ (e X Y).symm g = (e X' Y).symm (f ≫ g) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G_obj : D → C
    e : (X : C) → (Y : D) → Equiv (Quiver.Hom (F.obj X) Y) (Quiver.Hom X (G_obj Y))
    he : ∀ (X' X : C) (Y : D) (f : Quiver.Hom X' X) (g : Quiver.Hom (F.obj X) Y),  …
    X' X : C
    Y : D
    f : Quiver.Hom X' X
    g : Quiver.Hom X (G_obj Y)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((e X Y).symm g)) ((e X' Y) …
  -/
  rw [Equiv.eq_symm_apply, he]; simp
                                /-
                                  🎉 no goals
                                -/


/-- Construct a right adjoint functor to `F`, given the functor's value on objects `G_obj` and
a bijection `e` between `F.obj X ⟶ Y` and `X ⟶ G_obj Y` satisfying a naturality law
`he : ∀ X Y Y' g h, e X' Y (F.map f ≫ g) = f ≫ e X Y g`.
Dual to `leftAdjointOfEquiv`. -/
@[simps!]
def rightAdjointOfEquiv (he : ∀ X' X Y f g, e X' Y (F.map f ≫ g) = f ≫ e X Y g) : D ⥤ C where
  obj := G_obj
  map {Y} {Y'} g := (e (G_obj Y) Y') ((e (G_obj Y) Y).symm (𝟙 _) ≫ g)
  map_comp := fun {Y} {Y'} {Y''} g g' => by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      G_obj : D → C
      e : (X : C) → (Y : D) → Equiv (Quiver.Hom (F.obj X) Y) (Quiver.Hom X (G_obj Y))
      he : ∀ (X' X : C) (Y : D) (f : Quiver.Hom X' X) (g : Quiver.Hom (F.obj X) Y),  …
      Y Y' Y'' : D
      g : Quiver.Hom Y Y'
      g' : Quiver.Hom Y' Y''
      ⊢ Eq ({ obj := G_obj, map := fun {Y Y'} g => (e (G_obj Y) Y') (CategoryTheory. …
    -/
    rw [← Equiv.eq_symm_apply, ← he'' e he, Equiv.symm_apply_apply]
    conv =>
      rhs
      rw [← assoc, he'' e he, comp_id, Equiv.symm_apply_apply]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      G_obj : D → C
      e : (X : C) → (Y : D) → Equiv (Quiver.Hom (F.obj X) Y) (Quiver.Hom X (G_obj Y))
      he : ∀ (X' X : C) (Y : D) (f : Quiver.Hom X' X) (g : Quiver.Hom (F.obj X) Y),  …
      Y Y' Y'' : D
      g : Quiver.Hom Y Y'
      g' : Quiver.Hom Y' Y''
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((e (G_obj Y) Y).symm (CategoryTheory …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Show that the functor given by `rightAdjointOfEquiv` is indeed right adjoint to `F`. Dual
to `adjunctionOfEquivRight`. -/
@[simps!]
def adjunctionOfEquivRight (he : ∀ X' X Y f g, e X' Y (F.map f ≫ g) = f ≫ e X Y g) :
    F ⊣ (rightAdjointOfEquiv e he) :=
  mkOfHomEquiv
    { homEquiv := e
      homEquiv_naturality_left_symm := by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          G : CategoryTheory.Functor D C
          G_obj : D → C
          e : (X : C) → (Y : D) → Equiv (Quiver.Hom (F.obj X) Y) (Quiver.Hom X (G_obj Y))
          he : ∀ (X' X : C) (Y : D) (f : Quiver.Hom X' X) (g : Quiver.Hom (F.obj X) Y),  …
          ⊢ ∀ {X' X : C} {Y : D} (f : Quiver.Hom X' X) (g : Quiver.Hom X ((CategoryTheor …
        -/
        intro X X' Y f g; rw [Equiv.symm_apply_eq]; simp [he]
                                                    /-
                                                      🎉 no goals
                                                    -/
      homEquiv_naturality_right := by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          G : CategoryTheory.Functor D C
          G_obj : D → C
          e : (X : C) → (Y : D) → Equiv (Quiver.Hom (F.obj X) Y) (Quiver.Hom X (G_obj Y))
          he : ∀ (X' X : C) (Y : D) (f : Quiver.Hom X' X) (g : Quiver.Hom (F.obj X) Y),  …
          ⊢ ∀ {X : C} {Y Y' : D} (f : Quiver.Hom (F.obj X) Y) (g : Quiver.Hom Y Y'), Eq  …
        -/
        intro X Y Y' g h
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          F : CategoryTheory.Functor C D
          G : CategoryTheory.Functor D C
          G_obj : D → C
          e : (X : C) → (Y : D) → Equiv (Quiver.Hom (F.obj X) Y) (Quiver.Hom X (G_obj Y))
          he : ∀ (X' X : C) (Y : D) (f : Quiver.Hom X' X) (g : Quiver.Hom (F.obj X) Y),  …
          X : C
          Y Y' : D
          g : Quiver.Hom (F.obj X) Y
          h : Quiver.Hom Y Y'
          ⊢ Eq ((e X Y') (CategoryTheory.CategoryStruct.comp g h)) (CategoryTheory.Categ …
        -/
        simp [← he, reassoc_of% (he'' e)] }
        /-
          🎉 no goals
        -/


/--
If the unit and counit of a given adjunction are (pointwise) isomorphisms, then we can upgrade the
adjunction to an equivalence.
-/
@[simps!]
noncomputable def toEquivalence (adj : F ⊣ G) [∀ X, IsIso (adj.unit.app X)]
    [∀ Y, IsIso (adj.counit.app Y)] : C ≌ D where
  functor := F
  inverse := G
             /-
               C : Type u₁
               inst✝³ : CategoryTheory.Category.{v₁, u₁} C
               D : Type u₂
               inst✝² : CategoryTheory.Category.{v₂, u₂} D
               F : CategoryTheory.Functor C D
               G : CategoryTheory.Functor D C
               adj : CategoryTheory.Adjunction F G
               inst✝¹ : ∀ (X : C), CategoryTheory.IsIso (adj.unit.app X)
               inst✝ : ∀ (Y : D), CategoryTheory.IsIso (adj.counit.app Y)
               ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((C …
             -/
  unitIso := NatIso.ofComponents fun X => asIso (adj.unit.app X)
             /-
               🎉 no goals
             -/
               /-
                 C : Type u₁
                 inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                 D : Type u₂
                 inst✝² : CategoryTheory.Category.{v₂, u₂} D
                 F : CategoryTheory.Functor C D
                 G : CategoryTheory.Functor D C
                 adj : CategoryTheory.Adjunction F G
                 inst✝¹ : ∀ (X : C), CategoryTheory.IsIso (adj.unit.app X)
                 inst✝ : ∀ (Y : D), CategoryTheory.IsIso (adj.counit.app Y)
                 ⊢ ∀ {X Y : D} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((G …
               -/
  counitIso := NatIso.ofComponents fun Y => asIso (adj.counit.app Y)
               /-
                 🎉 no goals
               -/


/--
If the unit and counit for the adjunction corresponding to a right adjoint functor are (pointwise)
isomorphisms, then the functor is an equivalence of categories.
-/
lemma Functor.isEquivalence_of_isRightAdjoint (G : C ⥤ D) [IsRightAdjoint G]
    [∀ X, IsIso ((Adjunction.ofIsRightAdjoint G).unit.app X)]
    [∀ Y, IsIso ((Adjunction.ofIsRightAdjoint G).counit.app Y)] : G.IsEquivalence :=
  (Adjunction.ofIsRightAdjoint G).toEquivalence.isEquivalence_inverse


/-- The adjunction given by an equivalence of categories. (To obtain the opposite adjunction,
simply use `e.symm.toAdjunction`. -/
@[simps]
def toAdjunction : e.functor ⊣ e.inverse where
  unit := e.unit
  counit := e.counit


lemma isLeftAdjoint_functor : e.functor.IsLeftAdjoint where
  exists_rightAdjoint := ⟨_, ⟨e.toAdjunction⟩⟩


lemma isRightAdjoint_inverse : e.inverse.IsRightAdjoint where
  exists_leftAdjoint := ⟨_, ⟨e.toAdjunction⟩⟩


lemma isLeftAdjoint_inverse : e.inverse.IsLeftAdjoint :=
  e.symm.isLeftAdjoint_functor


lemma isRightAdjoint_functor : e.functor.IsRightAdjoint :=
  e.symm.isRightAdjoint_inverse


lemma trans_toAdjunction {E : Type*} [Category E] (e' : D ≌ E) :
    (e.trans e').toAdjunction = e.toAdjunction.comp e'.toAdjunction := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    E : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} E
    e' : CategoryTheory.Equivalence D E
    ⊢ Eq (e.trans e').toAdjunction (e.toAdjunction.comp e'.toAdjunction)
  -/
  ext
  /-
    case h.w.h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    E : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} E
    e' : CategoryTheory.Equivalence D E
    x✝ : C
    ⊢ Eq ((e.trans e').toAdjunction.unit.app x✝) ((e.toAdjunction.comp e'.toAdjunc …
  -/
  simp [trans]
  /-
    🎉 no goals
  -/


/-- If `F` and `G` are left adjoints then `F ⋙ G` is a left adjoint too. -/
instance isLeftAdjoint_comp {E : Type u₃} [Category.{v₃} E] (F : C ⥤ D) (G : D ⥤ E)
    [F.IsLeftAdjoint] [G.IsLeftAdjoint] : (F ⋙ G).IsLeftAdjoint where
  exists_rightAdjoint :=
    ⟨_, ⟨(Adjunction.ofIsLeftAdjoint F).comp (Adjunction.ofIsLeftAdjoint G)⟩⟩


/-- If `F` and `G` are right adjoints then `F ⋙ G` is a right adjoint too. -/
instance isRightAdjoint_comp {E : Type u₃} [Category.{v₃} E] {F : C ⥤ D} {G : D ⥤ E}
    [IsRightAdjoint F] [IsRightAdjoint G] : IsRightAdjoint (F ⋙ G) where
  exists_leftAdjoint :=
    ⟨_, ⟨(Adjunction.ofIsRightAdjoint G).comp (Adjunction.ofIsRightAdjoint F)⟩⟩


/-- Transport being a right adjoint along a natural isomorphism. -/
lemma isRightAdjoint_of_iso {F G : C ⥤ D} (h : F ≅ G) [F.IsRightAdjoint] :
    IsRightAdjoint G where
  exists_leftAdjoint := ⟨_, ⟨(Adjunction.ofIsRightAdjoint F).ofNatIsoRight h⟩⟩


/-- Transport being a left adjoint along a natural isomorphism. -/
lemma isLeftAdjoint_of_iso {F G : C ⥤ D} (h : F ≅ G) [IsLeftAdjoint F] :
    IsLeftAdjoint G where
  exists_rightAdjoint := ⟨_, ⟨(Adjunction.ofIsLeftAdjoint F).ofNatIsoLeft h⟩⟩



/-- An equivalence `E` is left adjoint to its inverse. -/
noncomputable def adjunction (E : C ⥤ D) [IsEquivalence E] : E ⊣ E.inv :=
  E.asEquivalence.toAdjunction


/-- If `F` is an equivalence, it's a left adjoint. -/
instance (priority := 10) isLeftAdjoint_of_isEquivalence {F : C ⥤ D} [F.IsEquivalence] :
    IsLeftAdjoint F :=
  F.asEquivalence.isLeftAdjoint_functor


/-- If `F` is an equivalence, it's a right adjoint. -/
instance (priority := 10) isRightAdjoint_of_isEquivalence {F : C ⥤ D} [F.IsEquivalence] :
    IsRightAdjoint F :=
  F.asEquivalence.isRightAdjoint_functor


