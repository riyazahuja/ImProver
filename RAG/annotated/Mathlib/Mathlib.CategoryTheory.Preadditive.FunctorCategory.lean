instance {F G : C ⥤ D} : Zero (F ⟶ G) where
  zero := { app := fun _ => 0 }


instance {F G : C ⥤ D} : Add (F ⟶ G) where
  add α β := { app := fun X => α.app X + β.app X }


instance {F G : C ⥤ D} : Neg (F ⟶ G) where
  neg α := { app := fun X => -α.app X }


instance functorCategoryPreadditive : Preadditive (C ⥤ D) where
  homGroup F G :=
    { nsmul := nsmulRec
      zsmul := zsmulRec
      sub := fun α β => { app := fun X => α.app X - β.app X }
      add_assoc := by
        /-
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          ⊢ ∀ (a b c : Quiver.Hom F G), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (H …
        -/
        intros
        /-
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          a✝ b✝ c✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd (HAdd.hAdd a✝ b✝) c✝) (HAdd.hAdd a✝ (HAdd.hAdd b✝ c✝))
        -/
        ext
        /-
          case w.h
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          a✝ b✝ c✝ : Quiver.Hom F G
          x✝ : C
          ⊢ Eq ((HAdd.hAdd (HAdd.hAdd a✝ b✝) c✝).app x✝) ((HAdd.hAdd a✝ (HAdd.hAdd b✝ c✝ …
        -/
        apply add_assoc
        /-
          🎉 no goals
        -/
      zero_add := by
        /-
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          ⊢ ∀ (a : Quiver.Hom F G), Eq (HAdd.hAdd 0 a) a
        -/
        intros
        /-
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          a✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd 0 a✝) a✝
        -/
        dsimp
        /-
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          a✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd 0 a✝) a✝
        -/
        ext
        /-
          case w.h
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          a✝ : Quiver.Hom F G
          x✝ : C
          ⊢ Eq ((HAdd.hAdd 0 a✝).app x✝) (a✝.app x✝)
        -/
        apply zero_add
        /-
          🎉 no goals
        -/
      add_zero := by
        /-
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          ⊢ ∀ (a : Quiver.Hom F G), Eq (HAdd.hAdd a 0) a
        -/
        intros
        /-
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          a✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd a✝ 0) a✝
        -/
        dsimp
        /-
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          a✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd a✝ 0) a✝
        -/
        ext
        /-
          case w.h
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          a✝ : Quiver.Hom F G
          x✝ : C
          ⊢ Eq ((HAdd.hAdd a✝ 0).app x✝) (a✝.app x✝)
        -/
        apply add_zero
        /-
          🎉 no goals
        -/
      add_comm := by
        /-
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          ⊢ ∀ (a b : Quiver.Hom F G), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
        -/
        intros
        /-
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          a✝ b✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd a✝ b✝) (HAdd.hAdd b✝ a✝)
        -/
        dsimp
        /-
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          a✝ b✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd a✝ b✝) (HAdd.hAdd b✝ a✝)
        -/
        /-
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          ⊢ ∀ (a b : Quiver.Hom F G), Eq (HSub.hSub a b) (HAdd.hAdd a (Neg.neg b))
        -/
        ext
        /-
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          a✝ b✝ : Quiver.Hom F G
          ⊢ Eq (HSub.hSub a✝ b✝) (HAdd.hAdd a✝ (Neg.neg b✝))
        -/
        /-
          case w.h
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          a✝ b✝ : Quiver.Hom F G
          x✝ : C
          ⊢ Eq ((HAdd.hAdd a✝ b✝).app x✝) ((HAdd.hAdd b✝ a✝).app x✝)
        -/
        /-
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          a✝ b✝ : Quiver.Hom F G
          ⊢ Eq (HSub.hSub a✝ b✝) (HAdd.hAdd a✝ (Neg.neg b✝))
        -/
        apply add_comm
        /-
          case w.h
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          a✝ b✝ : Quiver.Hom F G
          x✝ : C
          ⊢ Eq ((HSub.hSub a✝ b✝).app x✝) ((HAdd.hAdd a✝ (Neg.neg b✝)).app x✝)
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
      sub_eq_add_neg := by
        /-
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          ⊢ ∀ (a : Quiver.Hom F G), Eq (HAdd.hAdd (Neg.neg a) a) 0
        -/
        intros
        /-
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          a✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd (Neg.neg a✝) a✝) 0
        -/
        dsimp
        /-
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          a✝ : Quiver.Hom F G
          ⊢ Eq (HAdd.hAdd (Neg.neg a✝) a✝) 0
        -/
        ext
        /-
          case w.h
          C : Type u_1
          D : Type u_2
          inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
          inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
          inst✝ : CategoryTheory.Preadditive D
          F G : CategoryTheory.Functor C D
          a✝ : Quiver.Hom F G
          x✝ : C
          ⊢ Eq ((HAdd.hAdd (Neg.neg a✝) a✝).app x✝) (CategoryTheory.NatTrans.app 0 x✝)
        -/
        apply sub_eq_add_neg
        /-
          🎉 no goals
        -/
      neg_add_cancel := by
        intros
        dsimp
        ext
        apply neg_add_cancel }
  add_comp := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
      inst✝ : CategoryTheory.Preadditive D
      ⊢ ∀ (P Q R : CategoryTheory.Functor C D) (f f' : Quiver.Hom P Q) (g : Quiver.H …
    -/
    intros
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
      inst✝ : CategoryTheory.Preadditive D
      P✝ Q✝ R✝ : CategoryTheory.Functor C D
      f✝ f'✝ : Quiver.Hom P✝ Q✝
      g✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd f✝ f'✝) g✝) (HAdd.hAdd (Ca …
    -/
    dsimp
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
      inst✝ : CategoryTheory.Preadditive D
      P✝ Q✝ R✝ : CategoryTheory.Functor C D
      f✝ f'✝ : Quiver.Hom P✝ Q✝
      g✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd f✝ f'✝) g✝) (HAdd.hAdd (Ca …
    -/
    ext
    /-
      case w.h
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
      inst✝ : CategoryTheory.Preadditive D
      P✝ Q✝ R✝ : CategoryTheory.Functor C D
      f✝ f'✝ : Quiver.Hom P✝ Q✝
      g✝ : Quiver.Hom Q✝ R✝
      x✝ : C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (HAdd.hAdd f✝ f'✝) g✝).app x✝) ((HAd …
    -/
    apply add_comp
    /-
      🎉 no goals
    -/
  comp_add := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
      inst✝ : CategoryTheory.Preadditive D
      ⊢ ∀ (P Q R : CategoryTheory.Functor C D) (f : Quiver.Hom P Q) (g g' : Quiver.H …
    -/
    intros
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
      inst✝ : CategoryTheory.Preadditive D
      P✝ Q✝ R✝ : CategoryTheory.Functor C D
      f✝ : Quiver.Hom P✝ Q✝
      g✝ g'✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f✝ (HAdd.hAdd g✝ g'✝)) (HAdd.hAdd (Ca …
    -/
    dsimp
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
      inst✝ : CategoryTheory.Preadditive D
      P✝ Q✝ R✝ : CategoryTheory.Functor C D
      f✝ : Quiver.Hom P✝ Q✝
      g✝ g'✝ : Quiver.Hom Q✝ R✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f✝ (HAdd.hAdd g✝ g'✝)) (HAdd.hAdd (Ca …
    -/
    ext
    /-
      case w.h
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.2848, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.2852, u_2} D
      inst✝ : CategoryTheory.Preadditive D
      P✝ Q✝ R✝ : CategoryTheory.Functor C D
      f✝ : Quiver.Hom P✝ Q✝
      g✝ g'✝ : Quiver.Hom Q✝ R✝
      x✝ : C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp f✝ (HAdd.hAdd g✝ g'✝)).app x✝) ((HAd …
    -/
    apply comp_add
    /-
      🎉 no goals
    -/


/-- Application of a natural transformation at a fixed object,
as group homomorphism -/
@[simps]
def appHom (X : C) : (F ⟶ G) →+ (F.obj X ⟶ G.obj X) where
  toFun α := α.app X
  map_zero' := rfl
  map_add' _ _ := rfl


@[simp]
theorem app_zero (X : C) : (0 : F ⟶ G).app X = 0 :=
  rfl


@[simp]
theorem app_add (X : C) (α β : F ⟶ G) : (α + β).app X = α.app X + β.app X :=
  rfl


@[simp]
theorem app_sub (X : C) (α β : F ⟶ G) : (α - β).app X = α.app X - β.app X :=
  rfl


@[simp]
theorem app_neg (X : C) (α : F ⟶ G) : (-α).app X = -α.app X :=
  rfl


@[simp]
theorem app_nsmul (X : C) (α : F ⟶ G) (n : ℕ) : (n • α).app X = n • α.app X :=
  (appHom X).map_nsmul α n


@[simp]
theorem app_zsmul (X : C) (α : F ⟶ G) (n : ℤ) : (n • α).app X = n • α.app X :=
  (appHom X : (F ⟶ G) →+ (F.obj X ⟶ G.obj X)).map_zsmul α n


@[simp]
theorem app_units_zsmul (X : C) (α : F ⟶ G) (n : ℤˣ) : (n • α).app X = n • α.app X := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_3, u_2} D
    inst✝ : CategoryTheory.Preadditive D
    F G : CategoryTheory.Functor C D
    X : C
    α : Quiver.Hom F G
    n : Units Int
    ⊢ Eq ((HSMul.hSMul n α).app X) (HSMul.hSMul n (α.app X))
  -/
  apply app_zsmul
  /-
    🎉 no goals
  -/


@[simp]
theorem app_sum {ι : Type*} (s : Finset ι) (X : C) (α : ι → (F ⟶ G)) :
    (∑ i ∈ s, α i).app X = ∑ i ∈ s, (α i).app X := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
    inst✝ : CategoryTheory.Preadditive D
    F G : CategoryTheory.Functor C D
    ι : Type u_3
    s : Finset ι
    X : C
    α : ι → Quiver.Hom F G
    ⊢ Eq ((s.sum fun i => α i).app X) (s.sum fun i => (α i).app X)
  -/
  simp only [← appHom_apply, map_sum]
  /-
    🎉 no goals
  -/


