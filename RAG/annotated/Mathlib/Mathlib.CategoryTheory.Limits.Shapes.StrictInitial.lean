/-- We say `C` has strict initial objects if every initial object is strict, ie given any morphism
`f : A ⟶ I` where `I` is initial, then `f` is an isomorphism.

Strictly speaking, this says that *any* initial object must be strict, rather than that strict
initial objects exist.
-/
class HasStrictInitialObjects : Prop where
  out : ∀ {I A : C} (f : A ⟶ I), IsInitial I → IsIso f


theorem IsInitial.isIso_to (hI : IsInitial I) {A : C} (f : A ⟶ I) : IsIso f :=
  HasStrictInitialObjects.out f hI


theorem IsInitial.strict_hom_ext (hI : IsInitial I) {A : C} (f g : A ⟶ I) : f = g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasStrictInitialObjects C
    I : C
    hI : CategoryTheory.Limits.IsInitial I
    A : C
    f g : Quiver.Hom A I
    ⊢ Eq f g
  -/
  haveI := hI.isIso_to f
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasStrictInitialObjects C
    I : C
    hI : CategoryTheory.Limits.IsInitial I
    A : C
    f g : Quiver.Hom A I
    this : CategoryTheory.IsIso f
    ⊢ Eq f g
  -/
  haveI := hI.isIso_to g
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasStrictInitialObjects C
    I : C
    hI : CategoryTheory.Limits.IsInitial I
    A : C
    f g : Quiver.Hom A I
    this✝ : CategoryTheory.IsIso f
    this : CategoryTheory.IsIso g
    ⊢ Eq f g
  -/
  exact eq_of_inv_eq_inv (hI.hom_ext (inv f) (inv g))
  /-
    🎉 no goals
  -/


theorem IsInitial.subsingleton_to (hI : IsInitial I) {A : C} : Subsingleton (A ⟶ I) :=
  ⟨hI.strict_hom_ext⟩


/-- If `X ⟶ Y` with `Y` being a strict initial object, then `X` is also an initial object. -/
noncomputable
def IsInitial.ofStrict {X Y : C} (f : X ⟶ Y)
    (hY : IsInitial Y) : IsInitial X :=
  letI := hY.isIso_to f
  hY.ofIso (asIso f).symm


instance (priority := 100) initial_mono_of_strict_initial_objects : InitialMonoClass C where
  isInitial_mono_from := fun _ hI => { right_cancellation := fun _ _ _ => hI.strict_hom_ext _ _ }


/-- If `I` is initial, then `X ⨯ I` is isomorphic to it. -/
@[simps! hom]
noncomputable def mulIsInitial (X : C) [HasBinaryProduct X I] (hI : IsInitial I) : X ⨯ I ≅ I := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasStrictInitialObjects C
    I X : C
    inst✝ : CategoryTheory.Limits.HasBinaryProduct X I
    hI : CategoryTheory.Limits.IsInitial I
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.prod X I) I
  -/
  have := hI.isIso_to (prod.snd : X ⨯ I ⟶ I)
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasStrictInitialObjects C
    I X : C
    inst✝ : CategoryTheory.Limits.HasBinaryProduct X I
    hI : CategoryTheory.Limits.IsInitial I
    this : CategoryTheory.IsIso CategoryTheory.Limits.prod.snd
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.prod X I) I
  -/
  exact asIso prod.snd
  /-
    🎉 no goals
  -/


@[simp]
theorem mulIsInitial_inv (X : C) [HasBinaryProduct X I] (hI : IsInitial I) :
    (mulIsInitial X hI).inv = hI.to _ :=
  hI.hom_ext _ _


/-- If `I` is initial, then `I ⨯ X` is isomorphic to it. -/
@[simps! hom]
noncomputable def isInitialMul (X : C) [HasBinaryProduct I X] (hI : IsInitial I) : I ⨯ X ≅ I := by
   /-
     C : Type u
     inst✝² : CategoryTheory.Category.{v, u} C
     inst✝¹ : CategoryTheory.Limits.HasStrictInitialObjects C
     I X : C
     inst✝ : CategoryTheory.Limits.HasBinaryProduct I X
     hI : CategoryTheory.Limits.IsInitial I
     ⊢ CategoryTheory.Iso (CategoryTheory.Limits.prod I X) I
   -/
   have := hI.isIso_to (prod.fst : I ⨯ X ⟶ I)
   /-
     C : Type u
     inst✝² : CategoryTheory.Category.{v, u} C
     inst✝¹ : CategoryTheory.Limits.HasStrictInitialObjects C
     I X : C
     inst✝ : CategoryTheory.Limits.HasBinaryProduct I X
     hI : CategoryTheory.Limits.IsInitial I
     this : CategoryTheory.IsIso CategoryTheory.Limits.prod.fst
     ⊢ CategoryTheory.Iso (CategoryTheory.Limits.prod I X) I
   -/
   exact asIso prod.fst
   /-
     🎉 no goals
   -/


@[simp]
theorem isInitialMul_inv (X : C) [HasBinaryProduct I X] (hI : IsInitial I) :
    (isInitialMul X hI).inv = hI.to _ :=
  hI.hom_ext _ _


instance initial_isIso_to {A : C} (f : A ⟶ ⊥_ C) : IsIso f :=
  initialIsInitial.isIso_to _


@[ext]
theorem initial.strict_hom_ext {A : C} (f g : A ⟶ ⊥_ C) : f = g :=
  initialIsInitial.strict_hom_ext _ _


theorem initial.subsingleton_to {A : C} : Subsingleton (A ⟶ ⊥_ C) :=
  initialIsInitial.subsingleton_to


/-- The product of `X` with an initial object in a category with strict initial objects is itself
initial.
This is the generalisation of the fact that `X × Empty ≃ Empty` for types (or `n * 0 = 0`).
-/
@[simps! hom]
noncomputable def mulInitial (X : C) [HasBinaryProduct X (⊥_ C)] : X ⨯ ⊥_ C ≅ ⊥_ C :=
  mulIsInitial _ initialIsInitial


@[simp]
theorem mulInitial_inv (X : C) [HasBinaryProduct X (⊥_ C)] : (mulInitial X).inv = initial.to _ :=
  Subsingleton.elim _ _


/-- The product of `X` with an initial object in a category with strict initial objects is itself
initial.
This is the generalisation of the fact that `Empty × X ≃ Empty` for types (or `0 * n = 0`).
-/
@[simps! hom]
noncomputable def initialMul (X : C) [HasBinaryProduct (⊥_ C) X] : (⊥_ C) ⨯ X ≅ ⊥_ C :=
  isInitialMul _ initialIsInitial


@[simp]
theorem initialMul_inv (X : C) [HasBinaryProduct (⊥_ C) X] : (initialMul X).inv = initial.to _ :=
  Subsingleton.elim _ _


/-- If `C` has an initial object such that every morphism *to* it is an isomorphism, then `C`
has strict initial objects. -/
theorem hasStrictInitialObjects_of_initial_is_strict [HasInitial C]
    (h : ∀ (A) (f : A ⟶ ⊥_ C), IsIso f) : HasStrictInitialObjects C :=
  { out := fun {I A} f hI =>
      haveI := h A (f ≫ hI.to _)
                                             /-
                                               C : Type u
                                               inst✝¹ : CategoryTheory.Category.{v, u} C
                                               inst✝ : CategoryTheory.Limits.HasInitial C
                                               h : ∀ (A : C) (f : Quiver.Hom A (CategoryTheory.Limits.initial C)), CategoryTh …
                                               I A : C
                                               f : Quiver.Hom A I
                                               hI : CategoryTheory.Limits.IsInitial I
                                               this : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp f (hI.to (Cate …
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
                                             -/
      ⟨⟨hI.to _ ≫ inv (f ≫ hI.to (⊥_ C)), by rw [← assoc, IsIso.hom_inv_id], hI.hom_ext _ _⟩⟩ }
                                             /-
                                               🎉 no goals
                                             -/


/-- We say `C` has strict terminal objects if every terminal object is strict, ie given any morphism
`f : I ⟶ A` where `I` is terminal, then `f` is an isomorphism.

Strictly speaking, this says that *any* terminal object must be strict, rather than that strict
terminal objects exist.
-/
class HasStrictTerminalObjects : Prop where
  out : ∀ {I A : C} (f : I ⟶ A), IsTerminal I → IsIso f


theorem IsTerminal.isIso_from (hI : IsTerminal I) {A : C} (f : I ⟶ A) : IsIso f :=
  HasStrictTerminalObjects.out f hI


theorem IsTerminal.strict_hom_ext (hI : IsTerminal I) {A : C} (f g : I ⟶ A) : f = g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasStrictTerminalObjects C
    I : C
    hI : CategoryTheory.Limits.IsTerminal I
    A : C
    f g : Quiver.Hom I A
    ⊢ Eq f g
  -/
  haveI := hI.isIso_from f
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasStrictTerminalObjects C
    I : C
    hI : CategoryTheory.Limits.IsTerminal I
    A : C
    f g : Quiver.Hom I A
    this : CategoryTheory.IsIso f
    ⊢ Eq f g
  -/
  haveI := hI.isIso_from g
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasStrictTerminalObjects C
    I : C
    hI : CategoryTheory.Limits.IsTerminal I
    A : C
    f g : Quiver.Hom I A
    this✝ : CategoryTheory.IsIso f
    this : CategoryTheory.IsIso g
    ⊢ Eq f g
  -/
  exact eq_of_inv_eq_inv (hI.hom_ext (inv f) (inv g))
  /-
    🎉 no goals
  -/


/-- If `X ⟶ Y` with `Y` being a strict terminal object, then `X` is also an terminal object. -/
noncomputable
def IsTerminal.ofStrict {X Y : C} (f : X ⟶ Y)
    (hY : IsTerminal X) : IsTerminal Y :=
  letI := hY.isIso_from f
  hY.ofIso (asIso f)


theorem IsTerminal.subsingleton_to (hI : IsTerminal I) {A : C} : Subsingleton (I ⟶ A) :=
  ⟨hI.strict_hom_ext⟩


/-- If all but one object in a diagram is strict terminal, then the limit is isomorphic to the
said object via `limit.π`. -/
theorem limit_π_isIso_of_is_strict_terminal (F : J ⥤ C) [HasLimit F] (i : J)
    (H : ∀ (j) (_ : j ≠ i), IsTerminal (F.obj j)) [Subsingleton (i ⟶ i)] : IsIso (limit.π F i) := by
  classical
    refine ⟨⟨limit.lift _ ⟨_, ⟨?_, ?_⟩⟩, ?_, ?_⟩⟩
    · exact fun j =>
        dite (j = i)
          (fun h => eqToHom (by cases h; rfl))
          fun h => (H _ h).from _
    · intro j k f
      split_ifs with h h_1 h_1
      · cases h
        cases h_1
        obtain rfl : f = 𝟙 _ := Subsingleton.elim _ _
        simp
      · cases h
        erw [Category.comp_id]
        haveI : IsIso (F.map f) := (H _ h_1).isIso_from _
        rw [← IsIso.comp_inv_eq]
        apply (H _ h_1).hom_ext
      · cases h_1
        apply (H _ h).hom_ext
      · apply (H _ h).hom_ext
    · ext
      rw [assoc, limit.lift_π]
      dsimp only
      split_ifs with h
      · cases h
        rw [id_comp, eqToHom_refl]
        exact comp_id _
      · apply (H _ h).hom_ext
    · rw [limit.lift_π]
      simp


instance terminal_isIso_from {A : C} (f : ⊤_ C ⟶ A) : IsIso f :=
  terminalIsTerminal.isIso_from _


@[ext]
theorem terminal.strict_hom_ext {A : C} (f g : ⊤_ C ⟶ A) : f = g :=
  terminalIsTerminal.strict_hom_ext _ _


theorem terminal.subsingleton_to {A : C} : Subsingleton (⊤_ C ⟶ A) :=
  terminalIsTerminal.subsingleton_to


/-- If `C` has an object such that every morphism *from* it is an isomorphism, then `C`
has strict terminal objects. -/
theorem hasStrictTerminalObjects_of_terminal_is_strict (I : C) (h : ∀ (A) (f : I ⟶ A), IsIso f) :
    HasStrictTerminalObjects C :=
  { out := fun {I' A} f hI' =>
      haveI := h A (hI'.from _ ≫ f)
                                                               /-
                                                                 C : Type u
                                                                 inst✝ : CategoryTheory.Category.{v, u} C
                                                                 I : C
                                                                 h : ∀ (A : C) (f : Quiver.Hom I A), CategoryTheory.IsIso f
                                                                 I' A : C
                                                                 f : Quiver.Hom I' A
                                                                 hI' : CategoryTheory.Limits.IsTerminal I'
                                                                 this : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (hI'.from I) f)
                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                                               -/
      ⟨⟨inv (hI'.from I ≫ f) ≫ hI'.from I, hI'.hom_ext _ _, by rw [assoc, IsIso.inv_hom_id]⟩⟩ }
                                                               /-
                                                                 🎉 no goals
                                                               -/


