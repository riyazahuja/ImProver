/-- A category "has zero morphisms" if there is a designated "zero morphism" in each morphism space,
and compositions of zero morphisms with anything give the zero morphism. -/
class HasZeroMorphisms where
  /-- Every morphism space has zero -/
  [zero : ∀ X Y : C, Zero (X ⟶ Y)]
  /-- `f` composed with `0` is `0` -/
  comp_zero : ∀ {X Y : C} (f : X ⟶ Y) (Z : C), f ≫ (0 : Y ⟶ Z) = (0 : X ⟶ Z) := by aesop_cat
  /-- `0` composed with `f` is `0` -/
  zero_comp : ∀ (X : C) {Y Z : C} (f : Y ⟶ Z), (0 : X ⟶ Y) ≫ f = (0 : X ⟶ Z) := by aesop_cat


@[simp]
theorem comp_zero [HasZeroMorphisms C] {X Y : C} {f : X ⟶ Y} {Z : C} :
    f ≫ (0 : Y ⟶ Z) = (0 : X ⟶ Z) :=
  HasZeroMorphisms.comp_zero f Z


@[simp]
theorem zero_comp [HasZeroMorphisms C] {X : C} {Y Z : C} {f : Y ⟶ Z} :
    (0 : X ⟶ Y) ≫ f = (0 : X ⟶ Z) :=
  HasZeroMorphisms.zero_comp X f


instance hasZeroMorphismsPEmpty : HasZeroMorphisms (Discrete PEmpty) where
             /-
               C : Type u
               inst✝¹ : CategoryTheory.Category.{v, u} C
               D : Type u'
               inst✝ : CategoryTheory.Category.{v', u'} D
               ⊢ (X Y : CategoryTheory.Discrete PEmpty.{?u.1408 + 1}) → Zero (Quiver.Hom X Y)
             -/
  zero := by aesop_cat
             /-
               🎉 no goals
             -/


instance hasZeroMorphismsPUnit : HasZeroMorphisms (Discrete PUnit) where
                 /-
                   C : Type u
                   inst✝¹ : CategoryTheory.Category.{v, u} C
                   D : Type u'
                   inst✝ : CategoryTheory.Category.{v', u'} D
                   X Y : CategoryTheory.Discrete PUnit.{?u.2380 + 1}
                   ⊢ Zero (Quiver.Hom X Y)
                 -/
  zero X Y := by repeat (constructor)
                 /-
                   🎉 no goals
                 -/


/-- This lemma will be immediately superseded by `ext`, below. -/
private theorem ext_aux (I J : HasZeroMorphisms C)
    (w : ∀ X Y : C, (I.zero X Y).zero = (J.zero X Y).zero) : I = J := by
  have : I.zero = J.zero := by
    funext X Y
    specialize w X Y
    apply congrArg Zero.mk w
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    I J : CategoryTheory.Limits.HasZeroMorphisms C
    w : ∀ (X Y : C), Eq Zero.zero Zero.zero
    this : Eq CategoryTheory.Limits.HasZeroMorphisms.zero CategoryTheory.Limits.Ha …
    ⊢ Eq I J
  -/
  cases I; cases J
  /-
    case mk.mk
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    zero✝¹ : (X Y : C) → Zero (Quiver.Hom X Y)
    comp_zero✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y) (Z : C), Eq (CategoryTheory.Cat …
    zero_comp✝¹ : ∀ (X : C) {Y Z : C} (f : Quiver.Hom Y Z), Eq (CategoryTheory.Cat …
    zero✝ : (X Y : C) → Zero (Quiver.Hom X Y)
    comp_zero✝ : ∀ {X Y : C} (f : Quiver.Hom X Y) (Z : C), Eq (CategoryTheory.Cate …
    zero_comp✝ : ∀ (X : C) {Y Z : C} (f : Quiver.Hom Y Z), Eq (CategoryTheory.Cate …
    w : ∀ (X Y : C), Eq Zero.zero Zero.zero
    this : Eq CategoryTheory.Limits.HasZeroMorphisms.zero CategoryTheory.Limits.Ha …
    ⊢ Eq (CategoryTheory.Limits.HasZeroMorphisms.mk comp_zero✝¹ zero_comp✝¹) (Cate …
  -/
  congr
    /-
      case mk.mk.h.e_4
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      zero✝¹ : (X Y : C) → Zero (Quiver.Hom X Y)
      comp_zero✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y) (Z : C), Eq (CategoryTheory.Cat …
      zero_comp✝¹ : ∀ (X : C) {Y Z : C} (f : Quiver.Hom Y Z), Eq (CategoryTheory.Cat …
      zero✝ : (X Y : C) → Zero (Quiver.Hom X Y)
      comp_zero✝ : ∀ {X Y : C} (f : Quiver.Hom X Y) (Z : C), Eq (CategoryTheory.Cate …
      zero_comp✝ : ∀ (X : C) {Y Z : C} (f : Quiver.Hom Y Z), Eq (CategoryTheory.Cate …
      w : ∀ (X Y : C), Eq Zero.zero Zero.zero
      this : Eq CategoryTheory.Limits.HasZeroMorphisms.zero CategoryTheory.Limits.Ha …
      ⊢ HEq comp_zero✝¹ comp_zero✝
    -/
  · apply proof_irrel_heq
    /-
      🎉 no goals
    -/
    /-
      case mk.mk.h.e_5
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      zero✝¹ : (X Y : C) → Zero (Quiver.Hom X Y)
      comp_zero✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y) (Z : C), Eq (CategoryTheory.Cat …
      zero_comp✝¹ : ∀ (X : C) {Y Z : C} (f : Quiver.Hom Y Z), Eq (CategoryTheory.Cat …
      zero✝ : (X Y : C) → Zero (Quiver.Hom X Y)
      comp_zero✝ : ∀ {X Y : C} (f : Quiver.Hom X Y) (Z : C), Eq (CategoryTheory.Cate …
      zero_comp✝ : ∀ (X : C) {Y Z : C} (f : Quiver.Hom Y Z), Eq (CategoryTheory.Cate …
      w : ∀ (X Y : C), Eq Zero.zero Zero.zero
      this : Eq CategoryTheory.Limits.HasZeroMorphisms.zero CategoryTheory.Limits.Ha …
      ⊢ HEq zero_comp✝¹ zero_comp✝
    -/
  · apply proof_irrel_heq
    /-
      🎉 no goals
    -/


/-- If you're tempted to use this lemma "in the wild", you should probably
carefully consider whether you've made a mistake in allowing two
instances of `HasZeroMorphisms` to exist at all.

See, particularly, the note on `zeroMorphismsOfZeroObject` below.
-/
theorem ext (I J : HasZeroMorphisms C) : I = J := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    I J : CategoryTheory.Limits.HasZeroMorphisms C
    ⊢ Eq I J
  -/
  apply ext_aux
  /-
    case w
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    I J : CategoryTheory.Limits.HasZeroMorphisms C
    ⊢ ∀ (X Y : C), Eq Zero.zero Zero.zero
  -/
  intro X Y
  have : (I.zero X Y).zero ≫ (J.zero Y Y).zero = (I.zero X Y).zero := by
    apply I.zero_comp X (J.zero Y Y).zero
  have that : (I.zero X Y).zero ≫ (J.zero Y Y).zero = (J.zero X Y).zero := by
    apply J.comp_zero (I.zero X Y).zero Y
  /-
    case w
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    I J : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    this : Eq (CategoryTheory.CategoryStruct.comp Zero.zero Zero.zero) Zero.zero
    that : Eq (CategoryTheory.CategoryStruct.comp Zero.zero Zero.zero) Zero.zero
    ⊢ Eq Zero.zero Zero.zero
  -/
  rw [← this, ← that]
  /-
    🎉 no goals
  -/


instance : Subsingleton (HasZeroMorphisms C) :=
  ⟨ext⟩


instance hasZeroMorphismsOpposite [HasZeroMorphisms C] : HasZeroMorphisms Cᵒᵖ where
  zero X Y := ⟨(0 : unop Y ⟶ unop X).op⟩
  comp_zero f Z := congr_arg Quiver.Hom.op (HasZeroMorphisms.zero_comp (unop Z) f.unop)
  zero_comp X {Y Z} (f : Y ⟶ Z) :=
    congrArg Quiver.Hom.op (HasZeroMorphisms.comp_zero f.unop (unop X))


@[simp] lemma op_zero (X Y : C) : (0 : X ⟶ Y).op = 0 := rfl


@[simp] lemma unop_zero (X Y : Cᵒᵖ) : (0 : X ⟶ Y).unop = 0 := rfl


theorem zero_of_comp_mono {X Y Z : C} {f : X ⟶ Y} (g : Y ⟶ Z) [Mono g] (h : f ≫ g = 0) : f = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.Mono g
    h : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq f 0
  -/
  rw [← zero_comp, cancel_mono] at h
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.Mono g
    h✝ : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    h : Eq f 0
    ⊢ Eq f 0
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem zero_of_epi_comp {X Y Z : C} (f : X ⟶ Y) {g : Y ⟶ Z} [Epi f] (h : f ≫ g = 0) : g = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.Epi f
    h : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq g 0
  -/
  rw [← comp_zero, cancel_epi] at h
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.Epi f
    h✝ : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    h : Eq g 0
    ⊢ Eq g 0
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem eq_zero_of_image_eq_zero {X Y : C} {f : X ⟶ Y} [HasImage f] (w : image.ι f = 0) :
                /-
                  C : Type u
                  inst✝² : CategoryTheory.Category.{v, u} C
                  inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                  X Y : C
                  f : Quiver.Hom X Y
                  inst✝ : CategoryTheory.Limits.HasImage f
                  w : Eq (CategoryTheory.Limits.image.ι f) 0
                  ⊢ Eq f 0
                -/
    f = 0 := by rw [← image.fac f, w, HasZeroMorphisms.comp_zero]
                /-
                  🎉 no goals
                -/


theorem nonzero_image_of_nonzero {X Y : C} {f : X ⟶ Y} [HasImage f] (w : f ≠ 0) : image.ι f ≠ 0 :=
  fun h => w (eq_zero_of_image_eq_zero h)


instance : HasZeroMorphisms (C ⥤ D) where
  zero F G := ⟨{ app := fun _ => 0 }⟩
  comp_zero := fun η H => by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms D
      X✝ Y✝ : CategoryTheory.Functor C D
      η : Quiver.Hom X✝ Y✝
      H : CategoryTheory.Functor C D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp η 0) 0
    -/
    ext X; dsimp; apply comp_zero
                  /-
                    🎉 no goals
                  -/
  zero_comp := fun F {G H} η => by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms D
      F G H : CategoryTheory.Functor C D
      η : Quiver.Hom G H
      ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 η) 0
    -/
    ext X; dsimp; apply zero_comp
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem zero_app (F G : C ⥤ D) (j : C) : (0 : F ⟶ G).app j = 0 := rfl


theorem eq_zero_of_src {X Y : C} (o : IsZero X) (f : X ⟶ Y) : f = 0 :=
  o.eq_of_src _ _


theorem eq_zero_of_tgt {X Y : C} (o : IsZero Y) (f : X ⟶ Y) : f = 0 :=
  o.eq_of_tgt _ _


theorem iff_id_eq_zero (X : C) : IsZero X ↔ 𝟙 X = 0 :=
  ⟨fun h => h.eq_of_src _ _, fun h =>
    ⟨fun Y => ⟨⟨⟨0⟩, fun f => by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X : C
          h : Eq (CategoryTheory.CategoryStruct.id X) 0
          Y : C
          f : Quiver.Hom X Y
          ⊢ Eq f Inhabited.default
        -/
        rw [← id_comp f, ← id_comp (0 : X ⟶ Y), h, zero_comp, zero_comp]; simp only⟩⟩,
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    fun Y => ⟨⟨⟨0⟩, fun f => by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X : C
          h : Eq (CategoryTheory.CategoryStruct.id X) 0
          Y : C
          f : Quiver.Hom Y X
          ⊢ Eq f Inhabited.default
        -/
        rw [← comp_id f, ← comp_id (0 : Y ⟶ X), h, comp_zero, comp_zero]; simp only ⟩⟩⟩⟩
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem of_mono_zero (X Y : C) [Mono (0 : X ⟶ Y)] : IsZero X :=
                                                          /-
                                                            C : Type u
                                                            inst✝² : CategoryTheory.Category.{v, u} C
                                                            inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                            X Y : C
                                                            inst✝ : CategoryTheory.Mono 0
                                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
                                                          -/
  (iff_id_eq_zero X).mpr ((cancel_mono (0 : X ⟶ Y)).1 (by simp))
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem of_epi_zero (X Y : C) [Epi (0 : X ⟶ Y)] : IsZero Y :=
                                                         /-
                                                           C : Type u
                                                           inst✝² : CategoryTheory.Category.{v, u} C
                                                           inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                           X Y : C
                                                           inst✝ : CategoryTheory.Epi 0
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 (CategoryTheory.CategoryStruct.id Y …
                                                         -/
  (iff_id_eq_zero Y).mpr ((cancel_epi (0 : X ⟶ Y)).1 (by simp))
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem of_mono_eq_zero {X Y : C} (f : X ⟶ Y) [Mono f] (h : f = 0) : IsZero X := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    h : Eq f 0
    ⊢ CategoryTheory.Limits.IsZero X
  -/
  subst h
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    inst✝ : CategoryTheory.Mono 0
    ⊢ CategoryTheory.Limits.IsZero X
  -/
  apply of_mono_zero X Y
  /-
    🎉 no goals
  -/


theorem of_epi_eq_zero {X Y : C} (f : X ⟶ Y) [Epi f] (h : f = 0) : IsZero Y := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Epi f
    h : Eq f 0
    ⊢ CategoryTheory.Limits.IsZero Y
  -/
  subst h
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    inst✝ : CategoryTheory.Epi 0
    ⊢ CategoryTheory.Limits.IsZero Y
  -/
  apply of_epi_zero X Y
  /-
    🎉 no goals
  -/


theorem iff_isSplitMono_eq_zero {X Y : C} (f : X ⟶ Y) [IsSplitMono f] : IsZero X ↔ f = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsSplitMono f
    ⊢ Iff (CategoryTheory.Limits.IsZero X) (Eq f 0)
  -/
  rw [iff_id_eq_zero]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsSplitMono f
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.id X) 0) (Eq f 0)
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsSplitMono f
      ⊢ Eq (CategoryTheory.CategoryStruct.id X) 0 → Eq f 0
    -/
  · intro h
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsSplitMono f
      h : Eq (CategoryTheory.CategoryStruct.id X) 0
      ⊢ Eq f 0
    -/
    rw [← Category.id_comp f, h, zero_comp]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsSplitMono f
      ⊢ Eq f 0 → Eq (CategoryTheory.CategoryStruct.id X) 0
    -/
  · intro h
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsSplitMono f
      h : Eq f 0
      ⊢ Eq (CategoryTheory.CategoryStruct.id X) 0
    -/
    rw [← IsSplitMono.id f]
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsSplitMono f
      h : Eq f 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.retraction f)) 0
    -/
    simp only [h, zero_comp]
    /-
      🎉 no goals
    -/


theorem iff_isSplitEpi_eq_zero {X Y : C} (f : X ⟶ Y) [IsSplitEpi f] : IsZero Y ↔ f = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsSplitEpi f
    ⊢ Iff (CategoryTheory.Limits.IsZero Y) (Eq f 0)
  -/
  rw [iff_id_eq_zero]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsSplitEpi f
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.id Y) 0) (Eq f 0)
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsSplitEpi f
      ⊢ Eq (CategoryTheory.CategoryStruct.id Y) 0 → Eq f 0
    -/
  · intro h
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsSplitEpi f
      h : Eq (CategoryTheory.CategoryStruct.id Y) 0
      ⊢ Eq f 0
    -/
    rw [← Category.comp_id f, h, comp_zero]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsSplitEpi f
      ⊢ Eq f 0 → Eq (CategoryTheory.CategoryStruct.id Y) 0
    -/
  · intro h
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsSplitEpi f
      h : Eq f 0
      ⊢ Eq (CategoryTheory.CategoryStruct.id Y) 0
    -/
    rw [← IsSplitEpi.id f]
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsSplitEpi f
      h : Eq f 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.section_ f) f) 0
    -/
    simp [h]
    /-
      🎉 no goals
    -/


theorem of_mono {X Y : C} (f : X ⟶ Y) [Mono f] (i : IsZero Y) : IsZero X := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    i : CategoryTheory.Limits.IsZero Y
    ⊢ CategoryTheory.Limits.IsZero X
  -/
  have hf := i.eq_zero_of_tgt f
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    i : CategoryTheory.Limits.IsZero Y
    hf : Eq f 0
    ⊢ CategoryTheory.Limits.IsZero X
  -/
  subst hf
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    i : CategoryTheory.Limits.IsZero Y
    inst✝ : CategoryTheory.Mono 0
    ⊢ CategoryTheory.Limits.IsZero X
  -/
  exact IsZero.of_mono_zero X Y
  /-
    🎉 no goals
  -/


theorem of_epi {X Y : C} (f : X ⟶ Y) [Epi f] (i : IsZero X) : IsZero Y := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Epi f
    i : CategoryTheory.Limits.IsZero X
    ⊢ CategoryTheory.Limits.IsZero Y
  -/
  have hf := i.eq_zero_of_src f
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Epi f
    i : CategoryTheory.Limits.IsZero X
    hf : Eq f 0
    ⊢ CategoryTheory.Limits.IsZero Y
  -/
  subst hf
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    i : CategoryTheory.Limits.IsZero X
    inst✝ : CategoryTheory.Epi 0
    ⊢ CategoryTheory.Limits.IsZero Y
  -/
  exact IsZero.of_epi_zero X Y
  /-
    🎉 no goals
  -/


/-- A category with a zero object has zero morphisms.

    It is rarely a good idea to use this. Many categories that have a zero object have zero
    morphisms for some other reason, for example from additivity. Library code that uses
    `zeroMorphismsOfZeroObject` will then be incompatible with these categories because
    the `HasZeroMorphisms` instances will not be definitionally equal. For this reason library
    code should generally ask for an instance of `HasZeroMorphisms` separately, even if it already
    asks for an instance of `HasZeroObjects`. -/
def IsZero.hasZeroMorphisms {O : C} (hO : IsZero O) : HasZeroMorphisms C where
  zero X Y := { zero := hO.from_ X ≫ hO.to_ Y }
  zero_comp X {Y Z} f := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      O : C
      hO : CategoryTheory.Limits.IsZero O
      X Y Z : C
      f : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 f) 0
    -/
    change (hO.from_ X ≫ hO.to_ Y) ≫ f = hO.from_ X ≫ hO.to_ Z
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      O : C
      hO : CategoryTheory.Limits.IsZero O
      X Y Z : C
      f : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [Category.assoc]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      O : C
      hO : CategoryTheory.Limits.IsZero O
      X Y Z : C
      f : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (hO.from_ X) (CategoryTheory.Category …
    -/
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      O : C
      hO : CategoryTheory.Limits.IsZero O
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f 0) 0
    -/
    congr
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      O : C
      hO : CategoryTheory.Limits.IsZero O
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
    -/
    /-
      case e_a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      O : C
      hO : CategoryTheory.Limits.IsZero O
      X Y Z : C
      f : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (hO.to_ Y) f) (hO.to_ Z)
    -/
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      O : C
      hO : CategoryTheory.Limits.IsZero O
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    apply hO.eq_of_src
    /-
      case e_a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      O : C
      hO : CategoryTheory.Limits.IsZero O
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (hO.from_ Y)) (hO.from_ X)
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  comp_zero {X Y} f Z := by
    change f ≫ (hO.from_ Y ≫ hO.to_ Z) = hO.from_ X ≫ hO.to_ Z
    rw [← Category.assoc]
    congr
    apply hO.eq_of_tgt


/-- A category with a zero object has zero morphisms.

    It is rarely a good idea to use this. Many categories that have a zero object have zero
    morphisms for some other reason, for example from additivity. Library code that uses
    `zeroMorphismsOfZeroObject` will then be incompatible with these categories because
    the `has_zero_morphisms` instances will not be definitionally equal. For this reason library
    code should generally ask for an instance of `HasZeroMorphisms` separately, even if it already
    asks for an instance of `HasZeroObjects`. -/
def zeroMorphismsOfZeroObject : HasZeroMorphisms C where
  zero X _ := { zero := (default : X ⟶ 0) ≫ default }
  zero_comp X {Y Z} f := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y Z : C
      f : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 f) 0
    -/
    change ((default : X ⟶ 0) ≫ default) ≫ f = (default : X ⟶ 0) ≫ default
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y Z : C
      f : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp I …
    -/
    rw [Category.assoc]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y Z : C
      f : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp Inhabited.default (CategoryTheory.Cat …
    -/
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f 0) 0
    -/
    congr
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
    -/
    /-
      case e_a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y Z : C
      f : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp Inhabited.default f) Inhabited.default
    -/
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    simp only [eq_iff_true_of_subsingleton]
    /-
      case e_a
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f Inhabited.default) Inhabited.default
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  comp_zero {X Y} f Z := by
    change f ≫ (default : Y ⟶ 0) ≫ default = (default : X ⟶ 0) ≫ default
    rw [← Category.assoc]
    congr
    simp only [eq_iff_true_of_subsingleton]


@[simp]
                                                                                            /-
                                                                                              C : Type u
                                                                                              inst✝² : CategoryTheory.Category.{v, u} C
                                                                                              inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                                                              inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                              X : C
                                                                                              t : CategoryTheory.Limits.IsInitial X
                                                                                              ⊢ Eq (CategoryTheory.Limits.HasZeroObject.zeroIsoIsInitial t).hom 0
                                                                                            -/
theorem zeroIsoIsInitial_hom {X : C} (t : IsInitial X) : (zeroIsoIsInitial t).hom = 0 := by ext
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


@[simp]
                                                                                            /-
                                                                                              C : Type u
                                                                                              inst✝² : CategoryTheory.Category.{v, u} C
                                                                                              inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                                                              inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                              X : C
                                                                                              t : CategoryTheory.Limits.IsInitial X
                                                                                              ⊢ Eq (CategoryTheory.Limits.HasZeroObject.zeroIsoIsInitial t).inv 0
                                                                                            -/
theorem zeroIsoIsInitial_inv {X : C} (t : IsInitial X) : (zeroIsoIsInitial t).inv = 0 := by ext
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


@[simp]
                                                                                               /-
                                                                                                 C : Type u
                                                                                                 inst✝² : CategoryTheory.Category.{v, u} C
                                                                                                 inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                                                                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                                 X : C
                                                                                                 t : CategoryTheory.Limits.IsTerminal X
                                                                                                 ⊢ Eq (CategoryTheory.Limits.HasZeroObject.zeroIsoIsTerminal t).hom 0
                                                                                               -/
theorem zeroIsoIsTerminal_hom {X : C} (t : IsTerminal X) : (zeroIsoIsTerminal t).hom = 0 := by ext
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


@[simp]
                                                                                               /-
                                                                                                 C : Type u
                                                                                                 inst✝² : CategoryTheory.Category.{v, u} C
                                                                                                 inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                                                                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                                 X : C
                                                                                                 t : CategoryTheory.Limits.IsTerminal X
                                                                                                 ⊢ Eq (CategoryTheory.Limits.HasZeroObject.zeroIsoIsTerminal t).inv 0
                                                                                               -/
theorem zeroIsoIsTerminal_inv {X : C} (t : IsTerminal X) : (zeroIsoIsTerminal t).inv = 0 := by ext
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


@[simp]
                                                                                      /-
                                                                                        C : Type u
                                                                                        inst✝³ : CategoryTheory.Category.{v, u} C
                                                                                        inst✝² : CategoryTheory.Limits.HasZeroObject C
                                                                                        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                        inst✝ : CategoryTheory.Limits.HasInitial C
                                                                                        ⊢ Eq CategoryTheory.Limits.HasZeroObject.zeroIsoInitial.hom 0
                                                                                      -/
theorem zeroIsoInitial_hom [HasInitial C] : zeroIsoInitial.hom = (0 : 0 ⟶ ⊥_ C) := by ext
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


@[simp]
                                                                                      /-
                                                                                        C : Type u
                                                                                        inst✝³ : CategoryTheory.Category.{v, u} C
                                                                                        inst✝² : CategoryTheory.Limits.HasZeroObject C
                                                                                        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                        inst✝ : CategoryTheory.Limits.HasInitial C
                                                                                        ⊢ Eq CategoryTheory.Limits.HasZeroObject.zeroIsoInitial.inv 0
                                                                                      -/
theorem zeroIsoInitial_inv [HasInitial C] : zeroIsoInitial.inv = (0 : ⊥_ C ⟶ 0) := by ext
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


@[simp]
                                                                                         /-
                                                                                           C : Type u
                                                                                           inst✝³ : CategoryTheory.Category.{v, u} C
                                                                                           inst✝² : CategoryTheory.Limits.HasZeroObject C
                                                                                           inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                           inst✝ : CategoryTheory.Limits.HasTerminal C
                                                                                           ⊢ Eq CategoryTheory.Limits.HasZeroObject.zeroIsoTerminal.hom 0
                                                                                         -/
theorem zeroIsoTerminal_hom [HasTerminal C] : zeroIsoTerminal.hom = (0 : 0 ⟶ ⊤_ C) := by ext
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


@[simp]
                                                                                         /-
                                                                                           C : Type u
                                                                                           inst✝³ : CategoryTheory.Category.{v, u} C
                                                                                           inst✝² : CategoryTheory.Limits.HasZeroObject C
                                                                                           inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                           inst✝ : CategoryTheory.Limits.HasTerminal C
                                                                                           ⊢ Eq CategoryTheory.Limits.HasZeroObject.zeroIsoTerminal.inv 0
                                                                                         -/
theorem zeroIsoTerminal_inv [HasTerminal C] : zeroIsoTerminal.inv = (0 : ⊤_ C ⟶ 0) := by ext
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


instance {B : Type*} [Category B] : HasZeroObject (B ⥤ C) :=
  (((CategoryTheory.Functor.const B).obj (0 : C)).isZero fun _ => isZero_zero _).hasZeroObject


@[simp]
theorem IsZero.map [HasZeroObject D] [HasZeroMorphisms D] {F : C ⥤ D} (hF : IsZero F) {X Y : C}
    (f : X ⟶ Y) : F.map f = 0 :=
  (hF.obj _).eq_of_src _ _


@[simp]
theorem _root_.CategoryTheory.Functor.zero_obj [HasZeroObject D] (X : C) :
    IsZero ((0 : C ⥤ D).obj X) :=
  (isZero_zero _).obj _


@[simp]
theorem _root_.CategoryTheory.zero_map [HasZeroObject D] [HasZeroMorphisms D] {X Y : C}
    (f : X ⟶ Y) : (0 : C ⥤ D).map f = 0 :=
  (isZero_zero _).map _


@[simp]
                                                      /-
                                                        C : Type u
                                                        inst✝² : CategoryTheory.Category.{v, u} C
                                                        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                        ⊢ Eq (CategoryTheory.CategoryStruct.id 0) 0
                                                      -/
theorem id_zero : 𝟙 (0 : C) = (0 : (0 : C) ⟶ 0) := by apply HasZeroObject.from_zero_ext
                                                      /-
                                                        🎉 no goals
                                                      -/

-- This can't be a `simp` lemma because the left hand side would be a metavariable.

/-- An arrow ending in the zero object is zero -/
                                                          /-
                                                            C : Type u
                                                            inst✝² : CategoryTheory.Category.{v, u} C
                                                            inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                            inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                            X : C
                                                            f : Quiver.Hom X 0
                                                            ⊢ Eq f 0
                                                          -/
theorem zero_of_to_zero {X : C} (f : X ⟶ 0) : f = 0 := by ext
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem zero_of_target_iso_zero {X Y : C} (f : X ⟶ Y) (i : Y ≅ 0) : f = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    i : CategoryTheory.Iso Y 0
    ⊢ Eq f 0
  -/
  have h : f = f ≫ i.hom ≫ 𝟙 0 ≫ i.inv := by simp only [Iso.hom_inv_id, id_comp, comp_id]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    i : CategoryTheory.Iso Y 0
    h : Eq f (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct. …
    ⊢ Eq f 0
  -/
  simpa using h
  /-
    🎉 no goals
  -/


/-- An arrow starting at the zero object is zero -/
                                                            /-
                                                              C : Type u
                                                              inst✝² : CategoryTheory.Category.{v, u} C
                                                              inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                              inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                              X : C
                                                              f : Quiver.Hom 0 X
                                                              ⊢ Eq f 0
                                                            -/
theorem zero_of_from_zero {X : C} (f : 0 ⟶ X) : f = 0 := by ext
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem zero_of_source_iso_zero {X Y : C} (f : X ⟶ Y) (i : X ≅ 0) : f = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    i : CategoryTheory.Iso X 0
    ⊢ Eq f 0
  -/
  have h : f = i.hom ≫ 𝟙 0 ≫ i.inv ≫ f := by simp only [Iso.hom_inv_id_assoc, id_comp, comp_id]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    i : CategoryTheory.Iso X 0
    h : Eq f (CategoryTheory.CategoryStruct.comp i.hom (CategoryTheory.CategoryStr …
    ⊢ Eq f 0
  -/
  simpa using h
  /-
    🎉 no goals
  -/


theorem zero_of_source_iso_zero' {X Y : C} (f : X ⟶ Y) (i : IsIsomorphic X 0) : f = 0 :=
  zero_of_source_iso_zero f (Nonempty.some i)


theorem zero_of_target_iso_zero' {X Y : C} (f : X ⟶ Y) (i : IsIsomorphic Y 0) : f = 0 :=
  zero_of_target_iso_zero f (Nonempty.some i)


theorem mono_of_source_iso_zero {X Y : C} (f : X ⟶ Y) (i : X ≅ 0) : Mono f :=
                       /-
                         C : Type u
                         inst✝² : CategoryTheory.Category.{v, u} C
                         inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                         inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                         X Y : C
                         f : Quiver.Hom X Y
                         i : CategoryTheory.Iso X 0
                         Z : C
                         g h : Quiver.Hom Z X
                         x✝ : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.CategoryStruc …
                         ⊢ Eq g h
                       -/
  ⟨fun {Z} g h _ => by rw [zero_of_target_iso_zero g i, zero_of_target_iso_zero h i]⟩
                       /-
                         🎉 no goals
                       -/


theorem epi_of_target_iso_zero {X Y : C} (f : X ⟶ Y) (i : Y ≅ 0) : Epi f :=
                       /-
                         C : Type u
                         inst✝² : CategoryTheory.Category.{v, u} C
                         inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                         inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                         X Y : C
                         f : Quiver.Hom X Y
                         i : CategoryTheory.Iso Y 0
                         Z : C
                         g h : Quiver.Hom Y Z
                         x✝ : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruc …
                         ⊢ Eq g h
                       -/
  ⟨fun {Z} g h _ => by rw [zero_of_source_iso_zero g i, zero_of_source_iso_zero h i]⟩
                       /-
                         🎉 no goals
                       -/


/-- An object `X` has `𝟙 X = 0` if and only if it is isomorphic to the zero object.

Because `X ≅ 0` contains data (even if a subsingleton), we express this `↔` as an `≃`.
-/
def idZeroEquivIsoZero (X : C) : 𝟙 X = 0 ≃ (X ≅ 0) where
  toFun h :=
    { hom := 0
      inv := 0 }
  invFun i := zero_of_target_iso_zero (𝟙 X) i
                 /-
                   C : Type u
                   inst✝³ : CategoryTheory.Category.{v, u} C
                   D : Type u'
                   inst✝² : CategoryTheory.Category.{v', u'} D
                   inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                   inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                   X : C
                   ⊢ Function.LeftInverse ⋯ fun h => { hom := 0, inv := 0, hom_inv_id := ⋯, inv_h …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u
                    inst✝³ : CategoryTheory.Category.{v, u} C
                    D : Type u'
                    inst✝² : CategoryTheory.Category.{v', u'} D
                    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                    X : C
                    ⊢ Function.RightInverse ⋯ fun h => { hom := 0, inv := 0, hom_inv_id := ⋯, inv_ …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem idZeroEquivIsoZero_apply_hom (X : C) (h : 𝟙 X = 0) : ((idZeroEquivIsoZero X) h).hom = 0 :=
  rfl


@[simp]
theorem idZeroEquivIsoZero_apply_inv (X : C) (h : 𝟙 X = 0) : ((idZeroEquivIsoZero X) h).inv = 0 :=
  rfl


/-- If `0 : X ⟶ Y` is a monomorphism, then `X ≅ 0`. -/
@[simps]
def isoZeroOfMonoZero {X Y : C} (_ : Mono (0 : X ⟶ Y)) : X ≅ 0 where
  hom := 0
  inv := 0
                                                 /-
                                                   C : Type u
                                                   inst✝³ : CategoryTheory.Category.{v, u} C
                                                   D : Type u'
                                                   inst✝² : CategoryTheory.Category.{v', u'} D
                                                   inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                   inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                   X Y : C
                                                   x✝ : CategoryTheory.Mono 0
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp 0 …
                                                 -/
  hom_inv_id := (cancel_mono (0 : X ⟶ Y)).mp (by simp)
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- If `0 : X ⟶ Y` is an epimorphism, then `Y ≅ 0`. -/
@[simps]
def isoZeroOfEpiZero {X Y : C} (_ : Epi (0 : X ⟶ Y)) : Y ≅ 0 where
  hom := 0
  inv := 0
                                                /-
                                                  C : Type u
                                                  inst✝³ : CategoryTheory.Category.{v, u} C
                                                  D : Type u'
                                                  inst✝² : CategoryTheory.Category.{v', u'} D
                                                  inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                  inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                  X Y : C
                                                  x✝ : CategoryTheory.Epi 0
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 (CategoryTheory.CategoryStruct.comp …
                                                -/
  hom_inv_id := (cancel_epi (0 : X ⟶ Y)).mp (by simp)
                                                /-
                                                  🎉 no goals
                                                -/


/-- If a monomorphism out of `X` is zero, then `X ≅ 0`. -/
def isoZeroOfMonoEqZero {X Y : C} {f : X ⟶ Y} [Mono f] (h : f = 0) : X ≅ 0 := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    h : Eq f 0
    ⊢ CategoryTheory.Iso X 0
  -/
  subst h
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    inst✝ : CategoryTheory.Mono 0
    ⊢ CategoryTheory.Iso X 0
  -/
  apply isoZeroOfMonoZero ‹_›
  /-
    🎉 no goals
  -/


/-- If an epimorphism in to `Y` is zero, then `Y ≅ 0`. -/
def isoZeroOfEpiEqZero {X Y : C} {f : X ⟶ Y} [Epi f] (h : f = 0) : Y ≅ 0 := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Epi f
    h : Eq f 0
    ⊢ CategoryTheory.Iso Y 0
  -/
  subst h
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    inst✝ : CategoryTheory.Epi 0
    ⊢ CategoryTheory.Iso Y 0
  -/
  apply isoZeroOfEpiZero ‹_›
  /-
    🎉 no goals
  -/


/-- If an object `X` is isomorphic to 0, there's no need to use choice to construct
an explicit isomorphism: the zero morphism suffices. -/
def isoOfIsIsomorphicZero {X : C} (P : IsIsomorphic X 0) : X ≅ 0 where
  hom := 0
  inv := 0
  hom_inv_id := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X : C
      P : CategoryTheory.IsIsomorphic X 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 0) (CategoryTheory.CategoryStruct.i …
    -/
    cases' P with P
    /-
      case intro
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X : C
      P : CategoryTheory.Iso X 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 0) (CategoryTheory.CategoryStruct.i …
    -/
    rw [← P.hom_inv_id, ← Category.id_comp P.inv]
    /-
      case intro
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X : C
      P : CategoryTheory.Iso X 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 0) (CategoryTheory.CategoryStruct.c …
    -/
    apply Eq.symm
    /-
      case intro.h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X : C
      P : CategoryTheory.Iso X 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp P.hom (CategoryTheory.CategoryStruct. …
    -/
    simp only [id_comp, Iso.hom_inv_id, comp_zero]
    /-
      case intro.h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X : C
      P : CategoryTheory.Iso X 0
      ⊢ Eq (CategoryTheory.CategoryStruct.id X) 0
    -/
    apply (idZeroEquivIsoZero X).invFun P
    /-
      🎉 no goals
    -/
                   /-
                     C : Type u
                     inst✝³ : CategoryTheory.Category.{v, u} C
                     D : Type u'
                     inst✝² : CategoryTheory.Category.{v', u'} D
                     inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                     X : C
                     P : CategoryTheory.IsIsomorphic X 0
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 0) (CategoryTheory.CategoryStruct.i …
                   -/
  inv_hom_id := by simp
                   /-
                     🎉 no goals
                   -/


/-- A zero morphism `0 : X ⟶ Y` is an isomorphism if and only if
the identities on both `X` and `Y` are zero.
-/
@[simps]
def isIsoZeroEquiv (X Y : C) : IsIso (0 : X ⟶ Y) ≃ 𝟙 X = 0 ∧ 𝟙 Y = 0 where
  toFun := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      ⊢ CategoryTheory.IsIso 0 → And (Eq (CategoryTheory.CategoryStruct.id X) 0) (Eq …
    -/
    intro i
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      i : CategoryTheory.IsIso 0
      ⊢ And (Eq (CategoryTheory.CategoryStruct.id X) 0) (Eq (CategoryTheory.Category …
    -/
    rw [← IsIso.hom_inv_id (0 : X ⟶ Y)]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      i : CategoryTheory.IsIso 0
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp 0 (CategoryTheory.inv 0)) 0) (Eq …
    -/
    rw [← IsIso.inv_hom_id (0 : X ⟶ Y)]
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} D
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      i : CategoryTheory.IsIso 0
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp 0 (CategoryTheory.inv 0)) 0) (Eq …
    -/
    simp only [eq_self_iff_true,comp_zero,and_self,zero_comp]
    /-
      🎉 no goals
    -/
                                /-
                                  C : Type u
                                  inst✝² : CategoryTheory.Category.{v, u} C
                                  D : Type u'
                                  inst✝¹ : CategoryTheory.Category.{v', u'} D
                                  inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                  X Y : C
                                  h : And (Eq (CategoryTheory.CategoryStruct.id X) 0) (Eq (CategoryTheory.Catego …
                                  ⊢ And (Eq (CategoryTheory.CategoryStruct.comp 0 0) (CategoryTheory.CategoryStr …
                                -/
  invFun h := ⟨⟨(0 : Y ⟶ X), by aesop_cat⟩⟩
                                /-
                                  🎉 no goals
                                -/
                 /-
                   C : Type u
                   inst✝² : CategoryTheory.Category.{v, u} C
                   D : Type u'
                   inst✝¹ : CategoryTheory.Category.{v', u'} D
                   inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                   X Y : C
                   ⊢ Function.LeftInverse ⋯ ⋯
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u
                    inst✝² : CategoryTheory.Category.{v, u} C
                    D : Type u'
                    inst✝¹ : CategoryTheory.Category.{v', u'} D
                    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                    X Y : C
                    ⊢ Function.RightInverse ⋯ ⋯
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/

-- Porting note: simp solves these

/-- A zero morphism `0 : X ⟶ X` is an isomorphism if and only if
the identity on `X` is zero.
-/
                                                                   /-
                                                                     C : Type u
                                                                     inst✝² : CategoryTheory.Category.{v, u} C
                                                                     D : Type u'
                                                                     inst✝¹ : CategoryTheory.Category.{v', u'} D
                                                                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                     X : C
                                                                     ⊢ Equiv (CategoryTheory.IsIso 0) (Eq (CategoryTheory.CategoryStruct.id X) 0)
                                                                   -/
def isIsoZeroSelfEquiv (X : C) : IsIso (0 : X ⟶ X) ≃ 𝟙 X = 0 := by simpa using isIsoZeroEquiv X X
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- A zero morphism `0 : X ⟶ Y` is an isomorphism if and only if
`X` and `Y` are isomorphic to the zero object.
-/
def isIsoZeroEquivIsoZero (X Y : C) : IsIso (0 : X ⟶ Y) ≃ (X ≅ 0) × (Y ≅ 0) := by
  -- This is lame, because `Prod` can't cope with `Prop`, so we can't use `Equiv.prodCongr`.
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    ⊢ Equiv (CategoryTheory.IsIso 0) (Prod (CategoryTheory.Iso X 0) (CategoryTheor …
  -/
  refine (isIsoZeroEquiv X Y).trans ?_
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    ⊢ Equiv (And (Eq (CategoryTheory.CategoryStruct.id X) 0) (Eq (CategoryTheory.C …
  -/
  symm
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    ⊢ Equiv (Prod (CategoryTheory.Iso X 0) (CategoryTheory.Iso Y 0)) (And (Eq (Cat …
  -/
  fconstructor
    /-
      case toFun
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : C
      ⊢ Prod (CategoryTheory.Iso X 0) (CategoryTheory.Iso Y 0) → And (Eq (CategoryTh …
    -/
  · rintro ⟨eX, eY⟩
    /-
      case toFun.mk
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : C
      eX : CategoryTheory.Iso X 0
      eY : CategoryTheory.Iso Y 0
      ⊢ And (Eq (CategoryTheory.CategoryStruct.id X) 0) (Eq (CategoryTheory.Category …
    -/
    fconstructor
      /-
        case toFun.mk.left
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝² : CategoryTheory.Category.{v', u'} D
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝ : CategoryTheory.Limits.HasZeroObject C
        X Y : C
        eX : CategoryTheory.Iso X 0
        eY : CategoryTheory.Iso Y 0
        ⊢ Eq (CategoryTheory.CategoryStruct.id X) 0
      -/
    · exact (idZeroEquivIsoZero X).symm eX
      /-
        🎉 no goals
      -/
      /-
        case toFun.mk.right
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝² : CategoryTheory.Category.{v', u'} D
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝ : CategoryTheory.Limits.HasZeroObject C
        X Y : C
        eX : CategoryTheory.Iso X 0
        eY : CategoryTheory.Iso Y 0
        ⊢ Eq (CategoryTheory.CategoryStruct.id Y) 0
      -/
    · exact (idZeroEquivIsoZero Y).symm eY
      /-
        🎉 no goals
      -/
    /-
      case invFun
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : C
      ⊢ And (Eq (CategoryTheory.CategoryStruct.id X) 0) (Eq (CategoryTheory.Category …
    -/
  · rintro ⟨hX, hY⟩
    /-
      case invFun.intro
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : C
      hX : Eq (CategoryTheory.CategoryStruct.id X) 0
      hY : Eq (CategoryTheory.CategoryStruct.id Y) 0
      ⊢ Prod (CategoryTheory.Iso X 0) (CategoryTheory.Iso Y 0)
    -/
    fconstructor
      /-
        case invFun.intro.fst
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝² : CategoryTheory.Category.{v', u'} D
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝ : CategoryTheory.Limits.HasZeroObject C
        X Y : C
        hX : Eq (CategoryTheory.CategoryStruct.id X) 0
        hY : Eq (CategoryTheory.CategoryStruct.id Y) 0
        ⊢ CategoryTheory.Iso X 0
      -/
    · exact (idZeroEquivIsoZero X) hX
      /-
        🎉 no goals
      -/
      /-
        case invFun.intro.snd
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝² : CategoryTheory.Category.{v', u'} D
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        inst✝ : CategoryTheory.Limits.HasZeroObject C
        X Y : C
        hX : Eq (CategoryTheory.CategoryStruct.id X) 0
        hY : Eq (CategoryTheory.CategoryStruct.id Y) 0
        ⊢ CategoryTheory.Iso Y 0
      -/
    · exact (idZeroEquivIsoZero Y) hY
      /-
        🎉 no goals
      -/
    /-
      case left_inv
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : C
      ⊢ Function.LeftInverse (fun a => And.casesOn a fun hX hY => { fst := (Category …
    -/
  · aesop_cat
    /-
      🎉 no goals
    -/
    /-
      case right_inv
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X Y : C
      ⊢ Function.RightInverse (fun a => And.casesOn a fun hX hY => { fst := (Categor …
    -/
  · aesop_cat
    /-
      🎉 no goals
    -/


theorem isIso_of_source_target_iso_zero {X Y : C} (f : X ⟶ Y) (i : X ≅ 0) (j : Y ≅ 0) :
    IsIso f := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    i : CategoryTheory.Iso X 0
    j : CategoryTheory.Iso Y 0
    ⊢ CategoryTheory.IsIso f
  -/
  rw [zero_of_source_iso_zero f i]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    i : CategoryTheory.Iso X 0
    j : CategoryTheory.Iso Y 0
    ⊢ CategoryTheory.IsIso 0
  -/
  exact (isIsoZeroEquivIsoZero _ _).invFun ⟨i, j⟩
  /-
    🎉 no goals
  -/


/-- A zero morphism `0 : X ⟶ X` is an isomorphism if and only if
`X` is isomorphic to the zero object.
-/
def isIsoZeroSelfEquivIsoZero (X : C) : IsIso (0 : X ⟶ X) ≃ (X ≅ 0) :=
  (isIsoZeroEquivIsoZero X X).trans subsingletonProdSelfEquiv


/-- If there are zero morphisms, any initial object is a zero object. -/
theorem hasZeroObject_of_hasInitial_object [HasZeroMorphisms C] [HasInitial C] :
    HasZeroObject C := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.Limits.HasInitial C
    ⊢ CategoryTheory.Limits.HasZeroObject C
  -/
  refine ⟨⟨⊥_ C, fun X => ⟨⟨⟨0⟩, by aesop_cat⟩⟩, fun X => ⟨⟨⟨0⟩, fun f => ?_⟩⟩⟩⟩
  calc
    f = f ≫ 𝟙 _ := (Category.comp_id _).symm
    _ = f ≫ 0 := by congr!; subsingleton
    _ = 0 := HasZeroMorphisms.comp_zero _ _


/-- If there are zero morphisms, any terminal object is a zero object. -/
theorem hasZeroObject_of_hasTerminal_object [HasZeroMorphisms C] [HasTerminal C] :
    HasZeroObject C := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.Limits.HasTerminal C
    ⊢ CategoryTheory.Limits.HasZeroObject C
  -/
  refine ⟨⟨⊤_ C, fun X => ⟨⟨⟨0⟩, fun f => ?_⟩⟩, fun X => ⟨⟨⟨0⟩, by aesop_cat⟩⟩⟩⟩
  calc
    f = 𝟙 _ ≫ f := (Category.id_comp _).symm
    _ = 0 ≫ f := by congr!; subsingleton
    _ = 0 := zero_comp


theorem image_ι_comp_eq_zero {X Y Z : C} {f : X ⟶ Y} {g : Y ⟶ Z} [HasImage f]
    [Epi (factorThruImage f)] (h : f ≫ g = 0) : image.ι f ≫ g = 0 :=
                                             /-
                                               C : Type u
                                               inst✝³ : CategoryTheory.Category.{v, u} C
                                               inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                               X Y Z : C
                                               f : Quiver.Hom X Y
                                               g : Quiver.Hom Y Z
                                               inst✝¹ : CategoryTheory.Limits.HasImage f
                                               inst✝ : CategoryTheory.Epi (CategoryTheory.Limits.factorThruImage f)
                                               h : Eq (CategoryTheory.CategoryStruct.comp f g) 0
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
                                             -/
  zero_of_epi_comp (factorThruImage f) <| by simp [h]
                                             /-
                                               🎉 no goals
                                             -/


theorem comp_factorThruImage_eq_zero {X Y Z : C} {f : X ⟶ Y} {g : Y ⟶ Z} [HasImage g]
    (h : f ≫ g = 0) : f ≫ factorThruImage g = 0 :=
                                      /-
                                        C : Type u
                                        inst✝² : CategoryTheory.Category.{v, u} C
                                        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                        X Y Z : C
                                        f : Quiver.Hom X Y
                                        g : Quiver.Hom Y Z
                                        inst✝ : CategoryTheory.Limits.HasImage g
                                        h : Eq (CategoryTheory.CategoryStruct.comp f g) 0
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
                                      -/
  zero_of_comp_mono (image.ι g) <| by simp [h]
                                      /-
                                        🎉 no goals
                                      -/


/-- The zero morphism has a `MonoFactorisation` through the zero object.
-/
@[simps]
def monoFactorisationZero (X Y : C) : MonoFactorisation (0 : X ⟶ Y) where
  I := 0
  m := 0
  e := 0


/-- The factorisation through the zero object is an image factorisation.
-/
def imageFactorisationZero (X Y : C) : ImageFactorisation (0 : X ⟶ Y) where
  F := monoFactorisationZero X Y
  isImage := { lift := fun _ => 0 }


instance hasImage_zero {X Y : C} : HasImage (0 : X ⟶ Y) :=
  HasImage.mk <| imageFactorisationZero _ _


/-- The image of a zero morphism is the zero object. -/
def imageZero {X Y : C} : image (0 : X ⟶ Y) ≅ 0 :=
  IsImage.isoExt (Image.isImage (0 : X ⟶ Y)) (imageFactorisationZero X Y).isImage


/-- The image of a morphism which is equal to zero is the zero object. -/
def imageZero' {X Y : C} {f : X ⟶ Y} (h : f = 0) [HasImage f] : image f ≅ 0 :=
  image.eqToIso h ≪≫ imageZero


@[simp]
theorem image.ι_zero {X Y : C} [HasImage (0 : X ⟶ Y)] : image.ι (0 : X ⟶ Y) = 0 := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasImage 0
    ⊢ Eq (CategoryTheory.Limits.image.ι 0) 0
  -/
  rw [← image.lift_fac (monoFactorisationZero X Y)]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasImage 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.lift (Ca …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If we know `f = 0`,
it requires a little work to conclude `image.ι f = 0`,
because `f = g` only implies `image f ≅ image g`.
-/
@[simp]
theorem image.ι_zero' [HasEqualizers C] {X Y : C} {f : X ⟶ Y} (h : f = 0) [HasImage f] :
    image.ι f = 0 := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasEqualizers C
    X Y : C
    f : Quiver.Hom X Y
    h : Eq f 0
    inst✝ : CategoryTheory.Limits.HasImage f
    ⊢ Eq (CategoryTheory.Limits.image.ι f) 0
  -/
  rw [image.eq_fac h]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasEqualizers C
    X Y : C
    f : Quiver.Hom X Y
    h : Eq f 0
    inst✝ : CategoryTheory.Limits.HasImage f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.eqToIso  …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- In the presence of zero morphisms, coprojections into a coproduct are (split) monomorphisms. -/
instance isSplitMono_sigma_ι {β : Type u'} [HasZeroMorphisms C] (f : β → C)
    [HasColimit (Discrete.functor f)] (b : β) : IsSplitMono (Sigma.ι f b) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    β : Type u'
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    f : β → C
    inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Discrete.functor f)
    b : β
    ⊢ CategoryTheory.IsSplitMono (CategoryTheory.Limits.Sigma.ι f b)
  -/
  classical exact IsSplitMono.mk' { retraction := Sigma.desc <| Pi.single b (𝟙 _) }
  /-
    🎉 no goals
  -/


/-- In the presence of zero morphisms, projections into a product are (split) epimorphisms. -/
instance isSplitEpi_pi_π {β : Type u'} [HasZeroMorphisms C] (f : β → C)
    [HasLimit (Discrete.functor f)] (b : β) : IsSplitEpi (Pi.π f b) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    β : Type u'
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    f : β → C
    inst✝ : CategoryTheory.Limits.HasLimit (CategoryTheory.Discrete.functor f)
    b : β
    ⊢ CategoryTheory.IsSplitEpi (CategoryTheory.Limits.Pi.π f b)
  -/
  classical exact IsSplitEpi.mk' { section_ := Pi.lift <| Pi.single b (𝟙 _) }
  /-
    🎉 no goals
  -/


/-- In the presence of zero morphisms, coprojections into a coproduct are (split) monomorphisms. -/
instance isSplitMono_coprod_inl [HasZeroMorphisms C] {X Y : C} [HasColimit (pair X Y)] :
    IsSplitMono (coprod.inl : X ⟶ X ⨿ Y) :=
  IsSplitMono.mk' { retraction := coprod.desc (𝟙 X) 0 }


/-- In the presence of zero morphisms, coprojections into a coproduct are (split) monomorphisms. -/
instance isSplitMono_coprod_inr [HasZeroMorphisms C] {X Y : C} [HasColimit (pair X Y)] :
    IsSplitMono (coprod.inr : Y ⟶ X ⨿ Y) :=
  IsSplitMono.mk' { retraction := coprod.desc 0 (𝟙 Y) }


/-- In the presence of zero morphisms, projections into a product are (split) epimorphisms. -/
instance isSplitEpi_prod_fst [HasZeroMorphisms C] {X Y : C} [HasLimit (pair X Y)] :
    IsSplitEpi (prod.fst : X ⨯ Y ⟶ X) :=
  IsSplitEpi.mk' { section_ := prod.lift (𝟙 X) 0 }


/-- In the presence of zero morphisms, projections into a product are (split) epimorphisms. -/
instance isSplitEpi_prod_snd [HasZeroMorphisms C] {X Y : C} [HasLimit (pair X Y)] :
    IsSplitEpi (prod.snd : X ⨯ Y ⟶ Y) :=
  IsSplitEpi.mk' { section_ := prod.lift 0 (𝟙 Y) }



/-- If a functor `F` is zero, then any cone for `F` with a zero point is limit. -/
def IsLimit.ofIsZero (c : Cone F) (hF : IsZero F) (hc : IsZero c.pt) : IsLimit c where
  lift _ := 0
  fac _ j := (F.isZero_iff.1 hF j).eq_of_tgt _ _
  uniq _ _ _ := hc.eq_of_tgt _ _


/-- If a functor `F` is zero, then any cocone for `F` with a zero point is colimit. -/
def IsColimit.ofIsZero (c : Cocone F) (hF : IsZero F) (hc : IsZero c.pt) : IsColimit c where
  desc _ := 0
  fac _ j := (F.isZero_iff.1 hF j).eq_of_src _ _
  uniq _ _ _ := hc.eq_of_src _ _


lemma IsLimit.isZero_pt {c : Cone F} (hc : IsLimit c) (hF : IsZero F) : IsZero c.pt :=
  (isZero_zero C).of_iso (IsLimit.conePointUniqueUpToIso hc
    (IsLimit.ofIsZero (Cone.mk 0 0) hF (isZero_zero C)))


lemma IsColimit.isZero_pt {c : Cocone F} (hc : IsColimit c) (hF : IsZero F) : IsZero c.pt :=
  (isZero_zero C).of_iso (IsColimit.coconePointUniqueUpToIso hc
    (IsColimit.ofIsZero (Cocone.mk 0 0) hF (isZero_zero C)))


lemma IsTerminal.isZero {X : C} (hX : IsTerminal X) : IsZero X := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X : C
    hX : CategoryTheory.Limits.IsTerminal X
    ⊢ CategoryTheory.Limits.IsZero X
  -/
  rw [IsZero.iff_id_eq_zero]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X : C
    hX : CategoryTheory.Limits.IsTerminal X
    ⊢ Eq (CategoryTheory.CategoryStruct.id X) 0
  -/
  apply hX.hom_ext
  /-
    🎉 no goals
  -/


lemma IsInitial.isZero {X : C} (hX : IsInitial X) : IsZero X := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X : C
    hX : CategoryTheory.Limits.IsInitial X
    ⊢ CategoryTheory.Limits.IsZero X
  -/
  rw [IsZero.iff_id_eq_zero]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X : C
    hX : CategoryTheory.Limits.IsInitial X
    ⊢ Eq (CategoryTheory.CategoryStruct.id X) 0
  -/
  apply hX.hom_ext
  /-
    🎉 no goals
  -/


