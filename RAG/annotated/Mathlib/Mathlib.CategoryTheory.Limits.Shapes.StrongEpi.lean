/-- A strong epimorphism `f` is an epimorphism which has the left lifting property
with respect to monomorphisms. -/
class StrongEpi (f : P ⟶ Q) : Prop where
  /-- The epimorphism condition on `f` -/
  epi : Epi f
  /-- The left lifting property with respect to all monomorphism -/
  llp : ∀ ⦃X Y : C⦄ (z : X ⟶ Y) [Mono z], HasLiftingProperty f z



theorem StrongEpi.mk' {f : P ⟶ Q} [Epi f]
    (hf : ∀ (X Y : C) (z : X ⟶ Y)
      (_ : Mono z) (u : P ⟶ X) (v : Q ⟶ Y) (sq : CommSq u f z v), sq.HasLift) :
    StrongEpi f :=
  { epi := inferInstance
    llp := fun {X Y} z hz => ⟨fun {u v} sq => hf X Y z hz u v sq⟩ }


/-- A strong monomorphism `f` is a monomorphism which has the right lifting property
with respect to epimorphisms. -/
class StrongMono (f : P ⟶ Q) : Prop where
  /-- The monomorphism condition on `f` -/
  mono : Mono f
  /-- The right lifting property with respect to all epimorphisms -/
  rlp : ∀ ⦃X Y : C⦄ (z : X ⟶ Y) [Epi z], HasLiftingProperty z f


theorem StrongMono.mk' {f : P ⟶ Q} [Mono f]
    (hf : ∀ (X Y : C) (z : X ⟶ Y) (_ : Epi z) (u : X ⟶ P)
      (v : Y ⟶ Q) (sq : CommSq u z f v), sq.HasLift) : StrongMono f where
  mono := inferInstance
  rlp := fun {X Y} z hz => ⟨fun {u v} sq => hf X Y z hz u v sq⟩


instance (priority := 100) epi_of_strongEpi (f : P ⟶ Q) [StrongEpi f] : Epi f :=
  StrongEpi.epi


instance (priority := 100) mono_of_strongMono (f : P ⟶ Q) [StrongMono f] : Mono f :=
  StrongMono.mono


/-- The composition of two strong epimorphisms is a strong epimorphism. -/
theorem strongEpi_comp [StrongEpi f] [StrongEpi g] : StrongEpi (f ≫ g) :=
  { epi := epi_comp _ _
    llp := by
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        P Q R : C
        f : Quiver.Hom P Q
        g : Quiver.Hom Q R
        inst✝¹ : CategoryTheory.StrongEpi f
        inst✝ : CategoryTheory.StrongEpi g
        ⊢ ∀ ⦃X Y : C⦄ (z : Quiver.Hom X Y) [inst : CategoryTheory.Mono z], CategoryThe …
      -/
      intros
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        P Q R : C
        f : Quiver.Hom P Q
        g : Quiver.Hom Q R
        inst✝² : CategoryTheory.StrongEpi f
        inst✝¹ : CategoryTheory.StrongEpi g
        X✝ Y✝ : C
        z✝ : Quiver.Hom X✝ Y✝
        inst✝ : CategoryTheory.Mono z✝
        ⊢ CategoryTheory.HasLiftingProperty (CategoryTheory.CategoryStruct.comp f g) z✝
      -/
      infer_instance }
      /-
        🎉 no goals
      -/


/-- The composition of two strong monomorphisms is a strong monomorphism. -/
theorem strongMono_comp [StrongMono f] [StrongMono g] : StrongMono (f ≫ g) :=
  { mono := mono_comp _ _
    rlp := by
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        P Q R : C
        f : Quiver.Hom P Q
        g : Quiver.Hom Q R
        inst✝¹ : CategoryTheory.StrongMono f
        inst✝ : CategoryTheory.StrongMono g
        ⊢ ∀ ⦃X Y : C⦄ (z : Quiver.Hom X Y) [inst : CategoryTheory.Epi z], CategoryTheo …
      -/
      intros
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        P Q R : C
        f : Quiver.Hom P Q
        g : Quiver.Hom Q R
        inst✝² : CategoryTheory.StrongMono f
        inst✝¹ : CategoryTheory.StrongMono g
        X✝ Y✝ : C
        z✝ : Quiver.Hom X✝ Y✝
        inst✝ : CategoryTheory.Epi z✝
        ⊢ CategoryTheory.HasLiftingProperty z✝ (CategoryTheory.CategoryStruct.comp f g)
      -/
      infer_instance }
      /-
        🎉 no goals
      -/


/-- If `f ≫ g` is a strong epimorphism, then so is `g`. -/
theorem strongEpi_of_strongEpi [StrongEpi (f ≫ g)] : StrongEpi g :=
  { epi := epi_of_epi f g
    llp := fun {X Y} z _ => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        P Q R : C
        f : Quiver.Hom P Q
        g : Quiver.Hom Q R
        inst✝ : CategoryTheory.StrongEpi (CategoryTheory.CategoryStruct.comp f g)
        X Y : C
        z : Quiver.Hom X Y
        x✝ : CategoryTheory.Mono z
        ⊢ CategoryTheory.HasLiftingProperty g z
      -/
      constructor
      /-
        case sq_hasLift
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        P Q R : C
        f : Quiver.Hom P Q
        g : Quiver.Hom Q R
        inst✝ : CategoryTheory.StrongEpi (CategoryTheory.CategoryStruct.comp f g)
        X Y : C
        z : Quiver.Hom X Y
        x✝ : CategoryTheory.Mono z
        ⊢ ∀ {f : Quiver.Hom Q X} {g_1 : Quiver.Hom R Y} (sq : CategoryTheory.CommSq f  …
      -/
      intro u v sq
      /-
        case sq_hasLift
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        P Q R : C
        f : Quiver.Hom P Q
        g : Quiver.Hom Q R
        inst✝ : CategoryTheory.StrongEpi (CategoryTheory.CategoryStruct.comp f g)
        X Y : C
        z : Quiver.Hom X Y
        x✝ : CategoryTheory.Mono z
        u : Quiver.Hom Q X
        v : Quiver.Hom R Y
        sq : CategoryTheory.CommSq u g z v
        ⊢ sq.HasLift
      -/
      have h₀ : (f ≫ u) ≫ z = (f ≫ g) ≫ v := by simp only [Category.assoc, sq.w]
      exact
        CommSq.HasLift.mk'
          ⟨(CommSq.mk h₀).lift, by
            simp only [← cancel_mono z, Category.assoc, CommSq.fac_right, sq.w], by simp⟩ }


/-- If `f ≫ g` is a strong monomorphism, then so is `f`. -/
theorem strongMono_of_strongMono [StrongMono (f ≫ g)] : StrongMono f :=
  { mono := mono_of_mono f g
    rlp := fun {X Y} z => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        P Q R : C
        f : Quiver.Hom P Q
        g : Quiver.Hom Q R
        inst✝ : CategoryTheory.StrongMono (CategoryTheory.CategoryStruct.comp f g)
        X Y : C
        z : Quiver.Hom X Y
        ⊢ ∀ [inst : CategoryTheory.Epi z], CategoryTheory.HasLiftingProperty z f
      -/
      intros
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        P Q R : C
        f : Quiver.Hom P Q
        g : Quiver.Hom Q R
        inst✝¹ : CategoryTheory.StrongMono (CategoryTheory.CategoryStruct.comp f g)
        X Y : C
        z : Quiver.Hom X Y
        inst✝ : CategoryTheory.Epi z
        ⊢ CategoryTheory.HasLiftingProperty z f
      -/
      constructor
      /-
        case sq_hasLift
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        P Q R : C
        f : Quiver.Hom P Q
        g : Quiver.Hom Q R
        inst✝¹ : CategoryTheory.StrongMono (CategoryTheory.CategoryStruct.comp f g)
        X Y : C
        z : Quiver.Hom X Y
        inst✝ : CategoryTheory.Epi z
        ⊢ ∀ {f_1 : Quiver.Hom X P} {g : Quiver.Hom Y Q} (sq : CategoryTheory.CommSq f_ …
      -/
      intro u v sq
      have h₀ : u ≫ f ≫ g = z ≫ v ≫ g := by
        rw [← Category.assoc, eq_whisker sq.w, Category.assoc]
      /-
        case sq_hasLift
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        P Q R : C
        f : Quiver.Hom P Q
        g : Quiver.Hom Q R
        inst✝¹ : CategoryTheory.StrongMono (CategoryTheory.CategoryStruct.comp f g)
        X Y : C
        z : Quiver.Hom X Y
        inst✝ : CategoryTheory.Epi z
        u : Quiver.Hom X P
        v : Quiver.Hom Y Q
        sq : CategoryTheory.CommSq u z f v
        h₀ : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.CategoryStruct.c …
        ⊢ sq.HasLift
      -/
      exact CommSq.HasLift.mk' ⟨(CommSq.mk h₀).lift, by simp, by simp [← cancel_epi z, sq.w]⟩ }
      /-
        🎉 no goals
      -/


/-- An isomorphism is in particular a strong epimorphism. -/
instance (priority := 100) strongEpi_of_isIso [IsIso f] : StrongEpi f where
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              P Q R : C
              f : Quiver.Hom P Q
              g : Quiver.Hom Q R
              inst✝ : CategoryTheory.IsIso f
              ⊢ CategoryTheory.Epi f
            -/
  epi := by infer_instance
            /-
              🎉 no goals
            -/
  llp {_ _} _ := HasLiftingProperty.of_left_iso _ _


/-- An isomorphism is in particular a strong monomorphism. -/
instance (priority := 100) strongMono_of_isIso [IsIso f] : StrongMono f where
             /-
               C : Type u
               inst✝¹ : CategoryTheory.Category.{v, u} C
               P Q R : C
               f : Quiver.Hom P Q
               g : Quiver.Hom Q R
               inst✝ : CategoryTheory.IsIso f
               ⊢ CategoryTheory.Mono f
             -/
  mono := by infer_instance
             /-
               🎉 no goals
             -/
  rlp {_ _} _ := HasLiftingProperty.of_right_iso _ _


theorem StrongEpi.of_arrow_iso {A B A' B' : C} {f : A ⟶ B} {g : A' ⟶ B'}
    (e : Arrow.mk f ≅ Arrow.mk g) [h : StrongEpi f] : StrongEpi g :=
  { epi := by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        A B A' B' : C
        f : Quiver.Hom A B
        g : Quiver.Hom A' B'
        e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk g)
        h : CategoryTheory.StrongEpi f
        ⊢ CategoryTheory.Epi g
      -/
      rw [Arrow.iso_w' e]
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        A B A' B' : C
        f : Quiver.Hom A B
        g : Quiver.Hom A' B'
        e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk g)
        h : CategoryTheory.StrongEpi f
        ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp e.inv.left (CategoryT …
      -/
      infer_instance
      /-
        🎉 no goals
      -/
    llp := fun {X Y} z => by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        A B A' B' : C
        f : Quiver.Hom A B
        g : Quiver.Hom A' B'
        e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk g)
        h : CategoryTheory.StrongEpi f
        X Y : C
        z : Quiver.Hom X Y
        ⊢ ∀ [inst : CategoryTheory.Mono z], CategoryTheory.HasLiftingProperty g z
      -/
      intro
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        A B A' B' : C
        f : Quiver.Hom A B
        g : Quiver.Hom A' B'
        e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk g)
        h : CategoryTheory.StrongEpi f
        X Y : C
        z : Quiver.Hom X Y
        inst✝ : CategoryTheory.Mono z
        ⊢ CategoryTheory.HasLiftingProperty g z
      -/
      apply HasLiftingProperty.of_arrow_iso_left e z }
      /-
        🎉 no goals
      -/


theorem StrongMono.of_arrow_iso {A B A' B' : C} {f : A ⟶ B} {g : A' ⟶ B'}
    (e : Arrow.mk f ≅ Arrow.mk g) [h : StrongMono f] : StrongMono g :=
  { mono := by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        A B A' B' : C
        f : Quiver.Hom A B
        g : Quiver.Hom A' B'
        e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk g)
        h : CategoryTheory.StrongMono f
        ⊢ CategoryTheory.Mono g
      -/
      rw [Arrow.iso_w' e]
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        A B A' B' : C
        f : Quiver.Hom A B
        g : Quiver.Hom A' B'
        e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk g)
        h : CategoryTheory.StrongMono f
        ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp e.inv.left (Category …
      -/
      infer_instance
      /-
        🎉 no goals
      -/
    rlp := fun {X Y} z => by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        A B A' B' : C
        f : Quiver.Hom A B
        g : Quiver.Hom A' B'
        e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk g)
        h : CategoryTheory.StrongMono f
        X Y : C
        z : Quiver.Hom X Y
        ⊢ ∀ [inst : CategoryTheory.Epi z], CategoryTheory.HasLiftingProperty z g
      -/
      intro
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        A B A' B' : C
        f : Quiver.Hom A B
        g : Quiver.Hom A' B'
        e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk g)
        h : CategoryTheory.StrongMono f
        X Y : C
        z : Quiver.Hom X Y
        inst✝ : CategoryTheory.Epi z
        ⊢ CategoryTheory.HasLiftingProperty z g
      -/
      apply HasLiftingProperty.of_arrow_iso_right z e }
      /-
        🎉 no goals
      -/


theorem StrongEpi.iff_of_arrow_iso {A B A' B' : C} {f : A ⟶ B} {g : A' ⟶ B'}
    (e : Arrow.mk f ≅ Arrow.mk g) : StrongEpi f ↔ StrongEpi g := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A B A' B' : C
    f : Quiver.Hom A B
    g : Quiver.Hom A' B'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk g)
    ⊢ Iff (CategoryTheory.StrongEpi f) (CategoryTheory.StrongEpi g)
  -/
  constructor <;> intro
  /-
    case mp
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A B A' B' : C
    f : Quiver.Hom A B
    g : Quiver.Hom A' B'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk g)
    a✝ : CategoryTheory.StrongEpi f
    ⊢ CategoryTheory.StrongEpi g
  -/
  exacts [StrongEpi.of_arrow_iso e, StrongEpi.of_arrow_iso e.symm]
  /-
    🎉 no goals
  -/


theorem StrongMono.iff_of_arrow_iso {A B A' B' : C} {f : A ⟶ B} {g : A' ⟶ B'}
    (e : Arrow.mk f ≅ Arrow.mk g) : StrongMono f ↔ StrongMono g := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A B A' B' : C
    f : Quiver.Hom A B
    g : Quiver.Hom A' B'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk g)
    ⊢ Iff (CategoryTheory.StrongMono f) (CategoryTheory.StrongMono g)
  -/
  constructor <;> intro
  /-
    case mp
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    A B A' B' : C
    f : Quiver.Hom A B
    g : Quiver.Hom A' B'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk g)
    a✝ : CategoryTheory.StrongMono f
    ⊢ CategoryTheory.StrongMono g
  -/
  exacts [StrongMono.of_arrow_iso e, StrongMono.of_arrow_iso e.symm]
  /-
    🎉 no goals
  -/


/-- A strong epimorphism that is a monomorphism is an isomorphism. -/
theorem isIso_of_mono_of_strongEpi (f : P ⟶ Q) [Mono f] [StrongEpi f] : IsIso f :=
                                          /-
                                            C : Type u
                                            inst✝² : CategoryTheory.Category.{v, u} C
                                            P Q : C
                                            f : Quiver.Hom P Q
                                            inst✝¹ : CategoryTheory.Mono f
                                            inst✝ : CategoryTheory.StrongEpi f
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id P)  …
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  ⟨⟨(CommSq.mk (show 𝟙 P ≫ f = f ≫ 𝟙 Q by simp)).lift, by aesop_cat⟩⟩
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- A strong monomorphism that is an epimorphism is an isomorphism. -/
theorem isIso_of_epi_of_strongMono (f : P ⟶ Q) [Epi f] [StrongMono f] : IsIso f :=
                                          /-
                                            C : Type u
                                            inst✝² : CategoryTheory.Category.{v, u} C
                                            P Q : C
                                            f : Quiver.Hom P Q
                                            inst✝¹ : CategoryTheory.Epi f
                                            inst✝ : CategoryTheory.StrongMono f
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id P)  …
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  ⟨⟨(CommSq.mk (show 𝟙 P ≫ f = f ≫ 𝟙 Q by simp)).lift, by aesop_cat⟩⟩
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- A strong epi category is a category in which every epimorphism is strong. -/
class StrongEpiCategory : Prop where
  /-- A strong epi category is a category in which every epimorphism is strong. -/
  strongEpi_of_epi : ∀ {X Y : C} (f : X ⟶ Y) [Epi f], StrongEpi f


/-- A strong mono category is a category in which every monomorphism is strong. -/
class StrongMonoCategory : Prop where
  /-- A strong mono category is a category in which every monomorphism is strong. -/
  strongMono_of_mono : ∀ {X Y : C} (f : X ⟶ Y) [Mono f], StrongMono f


theorem strongEpi_of_epi [StrongEpiCategory C] (f : P ⟶ Q) [Epi f] : StrongEpi f :=
  StrongEpiCategory.strongEpi_of_epi _


theorem strongMono_of_mono [StrongMonoCategory C] (f : P ⟶ Q) [Mono f] : StrongMono f :=
  StrongMonoCategory.strongMono_of_mono _


instance (priority := 100) balanced_of_strongEpiCategory [StrongEpiCategory C] : Balanced C where
  isIso_of_mono_of_epi _ _ _ := isIso_of_mono_of_strongEpi _


instance (priority := 100) balanced_of_strongMonoCategory [StrongMonoCategory C] : Balanced C where
  isIso_of_mono_of_epi _ _ _ := isIso_of_epi_of_strongMono _


