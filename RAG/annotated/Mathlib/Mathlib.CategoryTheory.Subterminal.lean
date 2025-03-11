/-- An object `A` is subterminal iff for any `Z`, there is at most one morphism `Z ⟶ A`. -/
def IsSubterminal (A : C) : Prop :=
  ∀ ⦃Z : C⦄ (f g : Z ⟶ A), f = g


theorem IsSubterminal.def : IsSubterminal A ↔ ∀ ⦃Z : C⦄ (f g : Z ⟶ A), f = g :=
  Iff.rfl


/-- If `A` is subterminal, the unique morphism from it to a terminal object is a monomorphism.
The converse of `isSubterminal_of_mono_isTerminal_from`.
-/
theorem IsSubterminal.mono_isTerminal_from (hA : IsSubterminal A) {T : C} (hT : IsTerminal T) :
    Mono (hT.from A) :=
  { right_cancellation := fun _ _ _ => hA _ _ }


/-- If `A` is subterminal, the unique morphism from it to the terminal object is a monomorphism.
The converse of `isSubterminal_of_mono_terminal_from`.
-/
theorem IsSubterminal.mono_terminal_from [HasTerminal C] (hA : IsSubterminal A) :
    Mono (terminal.from A) :=
  hA.mono_isTerminal_from terminalIsTerminal


/-- If the unique morphism from `A` to a terminal object is a monomorphism, `A` is subterminal.
The converse of `IsSubterminal.mono_isTerminal_from`.
-/
theorem isSubterminal_of_mono_isTerminal_from {T : C} (hT : IsTerminal T) [Mono (hT.from A)] :
    IsSubterminal A := fun Z f g => by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A T : C
    hT : CategoryTheory.Limits.IsTerminal T
    inst✝ : CategoryTheory.Mono (hT.from A)
    Z : C
    f g : Quiver.Hom Z A
    ⊢ Eq f g
  -/
  rw [← cancel_mono (hT.from A)]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A T : C
    hT : CategoryTheory.Limits.IsTerminal T
    inst✝ : CategoryTheory.Mono (hT.from A)
    Z : C
    f g : Quiver.Hom Z A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (hT.from A)) (CategoryTheory.Catego …
  -/
  apply hT.hom_ext
  /-
    🎉 no goals
  -/


/-- If the unique morphism from `A` to the terminal object is a monomorphism, `A` is subterminal.
The converse of `IsSubterminal.mono_terminal_from`.
-/
theorem isSubterminal_of_mono_terminal_from [HasTerminal C] [Mono (terminal.from A)] :
    IsSubterminal A := fun Z f g => by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    A : C
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Mono (CategoryTheory.Limits.terminal.from A)
    Z : C
    f g : Quiver.Hom Z A
    ⊢ Eq f g
  -/
  rw [← cancel_mono (terminal.from A)]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    A : C
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Mono (CategoryTheory.Limits.terminal.from A)
    Z : C
    f g : Quiver.Hom Z A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.terminal.fro …
  -/
  subsingleton
  /-
    🎉 no goals
  -/


theorem isSubterminal_of_isTerminal {T : C} (hT : IsTerminal T) : IsSubterminal T := fun _ _ _ =>
  hT.hom_ext _ _


theorem isSubterminal_of_terminal [HasTerminal C] : IsSubterminal (⊤_ C) := fun _ _ _ => by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Limits.HasTerminal C
    x✝² : C
    x✝¹ x✝ : Quiver.Hom x✝² (CategoryTheory.Limits.terminal C)
    ⊢ Eq x✝¹ x✝
  -/
  subsingleton
  /-
    🎉 no goals
  -/


/-- If `A` is subterminal, its diagonal morphism is an isomorphism.
The converse of `isSubterminal_of_isIso_diag`.
-/
theorem IsSubterminal.isIso_diag (hA : IsSubterminal A) [HasBinaryProduct A A] : IsIso (diag A) :=
  ⟨⟨Limits.prod.fst,
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            A : C
            hA : CategoryTheory.IsSubterminal A
            inst✝ : CategoryTheory.Limits.HasBinaryProduct A A
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.diag A) Catego …
          -/
      ⟨by simp, by
          /-
            🎉 no goals
          -/
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          A : C
          hA : CategoryTheory.IsSubterminal A
          inst✝ : CategoryTheory.Limits.HasBinaryProduct A A
          ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.prod.fst (Categ …
        -/
        rw [IsSubterminal.def] at hA
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          A : C
          hA : ∀ ⦃Z : C⦄ (f g : Quiver.Hom Z A), Eq f g
          inst✝ : CategoryTheory.Limits.HasBinaryProduct A A
          ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.prod.fst (Categ …
        -/
        aesop_cat⟩⟩⟩
        /-
          🎉 no goals
        -/


/-- If the diagonal morphism of `A` is an isomorphism, then it is subterminal.
The converse of `isSubterminal.isIso_diag`.
-/
theorem isSubterminal_of_isIso_diag [HasBinaryProduct A A] [IsIso (diag A)] : IsSubterminal A :=
  fun Z f g => by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    A : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct A A
    inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.diag A)
    Z : C
    f g : Quiver.Hom Z A
    ⊢ Eq f g
  -/
  have : (Limits.prod.fst : A ⨯ A ⟶ _) = Limits.prod.snd := by simp [← cancel_epi (diag A)]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    A : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct A A
    inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.diag A)
    Z : C
    f g : Quiver.Hom Z A
    this : Eq CategoryTheory.Limits.prod.fst CategoryTheory.Limits.prod.snd
    ⊢ Eq f g
  -/
  rw [← prod.lift_fst f g, this, prod.lift_snd]
  /-
    🎉 no goals
  -/


/-- If `A` is subterminal, it is isomorphic to `A ⨯ A`. -/
@[simps!]
def IsSubterminal.isoDiag (hA : IsSubterminal A) [HasBinaryProduct A A] : A ⨯ A ≅ A := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A : C
    hA : CategoryTheory.IsSubterminal A
    inst✝ : CategoryTheory.Limits.HasBinaryProduct A A
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.prod A A) A
  -/
  letI := IsSubterminal.isIso_diag hA
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    A : C
    hA : CategoryTheory.IsSubterminal A
    inst✝ : CategoryTheory.Limits.HasBinaryProduct A A
    this : CategoryTheory.IsIso (CategoryTheory.Limits.diag A) := CategoryTheory.I …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.prod A A) A
  -/
  apply (asIso (diag A)).symm
  /-
    🎉 no goals
  -/


/-- The (full sub)category of subterminal objects.
TODO: If `C` is the category of sheaves on a topological space `X`, this category is equivalent
to the lattice of open subsets of `X`. More generally, if `C` is a topos, this is the lattice of
"external truth values".
-/
def Subterminals (C : Type u₁) [Category.{v₁} C] :=
  FullSubcategory fun A : C => IsSubterminal A


instance (C : Type u₁) [Category.{v₁} C] :
  Category (Subterminals C) := FullSubcategory.category _


instance [HasTerminal C] : Inhabited (Subterminals C) :=
  ⟨⟨⊤_ C, isSubterminal_of_terminal⟩⟩


/-- The inclusion of the subterminal objects into the original category. -/
@[simps!]
def subterminalInclusion : Subterminals C ⥤ C :=
  fullSubcategoryInclusion _


instance (C : Type u₁) [Category.{v₁} C] : (subterminalInclusion C).Full :=
  FullSubcategory.full _


instance (C : Type u₁) [Category.{v₁} C] : (subterminalInclusion C).Faithful :=
  FullSubcategory.faithful _


instance subterminals_thin (X Y : Subterminals C) : Subsingleton (X ⟶ Y) :=
  ⟨fun f g => Y.2 f g⟩


/--
The category of subterminal objects is equivalent to the category of monomorphisms to the terminal
object (which is in turn equivalent to the subobjects of the terminal object).
-/
@[simps]
def subterminalsEquivMonoOverTerminal [HasTerminal C] : Subterminals C ≌ MonoOver (⊤_ C) where
  functor :=
    { obj := fun X => ⟨Over.mk (terminal.from X.1), X.2.mono_terminal_from⟩
                                           /-
                                             C : Type u₁
                                             inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                             A : C
                                             inst✝ : CategoryTheory.Limits.HasTerminal C
                                             X✝ Y✝ : CategoryTheory.Subterminals C
                                             f : Quiver.Hom X✝ Y✝
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.MonoOver.arrow ((fu …
                                           -/
      map := fun f => MonoOver.homMk f (by ext1 ⟨⟨⟩⟩)
                                           /-
                                             🎉 no goals
                                           -/
      map_id := fun _ => rfl
      map_comp := fun _ _ => rfl }
  inverse :=
    { obj := fun X =>
        ⟨X.obj.left, fun Z f g => by
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            A : C
            inst✝ : CategoryTheory.Limits.HasTerminal C
            X : CategoryTheory.MonoOver (CategoryTheory.Limits.terminal C)
            Z : C
            f g : Quiver.Hom Z X.obj.left
            ⊢ Eq f g
          -/
          rw [← cancel_mono X.arrow]
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            A : C
            inst✝ : CategoryTheory.Limits.HasTerminal C
            X : CategoryTheory.MonoOver (CategoryTheory.Limits.terminal C)
            Z : C
            f g : Quiver.Hom Z X.obj.left
            ⊢ Eq (CategoryTheory.CategoryStruct.comp f X.arrow) (CategoryTheory.CategorySt …
          -/
          subsingleton⟩
          /-
            🎉 no goals
          -/
      map := fun f => f.1
      map_id := fun _ => rfl
      map_comp := fun _ _ => rfl }
  -- Porting note: the original definition was triggering a timeout, using `NatIso.ofComponents`
  -- in the definition of the natural isomorphisms makes the situation slightly better
                                                           /-
                                                             C : Type u₁
                                                             inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                             A : C
                                                             inst✝ : CategoryTheory.Limits.HasTerminal C
                                                             ⊢ ∀ {X Y : CategoryTheory.Subterminals C} (f : Quiver.Hom X Y), Eq (CategoryTh …
                                                           -/
  unitIso := NatIso.ofComponents (fun X => Iso.refl X) (by subsingleton)
                                                           /-
                                                             🎉 no goals
                                                           -/
                                             /-
                                               C : Type u₁
                                               inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                               A : C
                                               inst✝ : CategoryTheory.Limits.HasTerminal C
                                               X : CategoryTheory.MonoOver (CategoryTheory.Limits.terminal C)
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl (({ obj := f …
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
  counitIso := NatIso.ofComponents (fun X => MonoOver.isoMk (Iso.refl _)) (by subsingleton)
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
                             /-
                               C : Type u₁
                               inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                               A : C
                               inst✝ : CategoryTheory.Limits.HasTerminal C
                               ⊢ ∀ (X : CategoryTheory.Subterminals C), Eq (CategoryTheory.CategoryStruct.com …
                             -/
  functor_unitIso_comp := by subsingleton
                             /-
                               🎉 no goals
                             -/
  -- With `aesop` filling the auto-params this was taking 20s or so


@[simp]
theorem subterminals_to_monoOver_terminal_comp_forget [HasTerminal C] :
    (subterminalsEquivMonoOverTerminal C).functor ⋙ MonoOver.forget _ ⋙ Over.forget _ =
      subterminalInclusion C :=
  rfl


@[simp]
theorem monoOver_terminal_to_subterminals_comp [HasTerminal C] :
    (subterminalsEquivMonoOverTerminal C).inverse ⋙ subterminalInclusion C =
      MonoOver.forget _ ⋙ Over.forget _ :=
  rfl


