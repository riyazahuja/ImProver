instance pi {J : Type w} [Finite J] {C : J → Type u₁} {D : J → Type u₂}
    [∀ j, Category.{v₁} (C j)] [∀ j, Category.{v₂} (D j)]
    (L : ∀ j, C j ⥤ D j) (W : ∀ j, MorphismProperty (C j))
    [∀ j, (W j).ContainsIdentities] [∀ j, (L j).IsLocalization (W j)] :
    (Functor.pi L).IsLocalization (MorphismProperty.pi W) := by
  /-
    J : Type w
    inst✝⁴ : Finite J
    C : J → Type u₁
    D : J → Type u₂
    inst✝³ : (j : J) → CategoryTheory.Category.{v₁, u₁} (C j)
    inst✝² : (j : J) → CategoryTheory.Category.{v₂, u₂} (D j)
    L : (j : J) → CategoryTheory.Functor (C j) (D j)
    W : (j : J) → CategoryTheory.MorphismProperty (C j)
    inst✝¹ : ∀ (j : J), (W j).ContainsIdentities
    inst✝ : ∀ (j : J), (L j).IsLocalization (W j)
    ⊢ (CategoryTheory.Functor.pi L).IsLocalization (CategoryTheory.MorphismPropert …
  -/
  revert J
  /-
    ⊢ ∀ {J : Type w} [inst : Finite J] {C : J → Type u₁} {D : J → Type u₂} [inst : …
  -/
  apply Finite.induction_empty_option
    /-
      case of_equiv
      ⊢ ∀ {α β : Type w}, Equiv α β → (∀ {C : α → Type u₁} {D : α → Type u₂} [inst : …
    -/
  · intro J₁ J₂ e hJ₁ C₂ D₂ _ _ L₂ W₂ _ _
    /-
      case of_equiv
      J₁ J₂ : Type w
      e : Equiv J₁ J₂
      hJ₁ : ∀ {C : J₁ → Type u₁} {D : J₁ → Type u₂} [inst : (j : J₁) → CategoryTheor …
      C₂ : J₂ → Type u₁
      D₂ : J₂ → Type u₂
      inst✝³ : (j : J₂) → CategoryTheory.Category.{v₁, u₁} (C₂ j)
      inst✝² : (j : J₂) → CategoryTheory.Category.{v₂, u₂} (D₂ j)
      L₂ : (j : J₂) → CategoryTheory.Functor (C₂ j) (D₂ j)
      W₂ : (j : J₂) → CategoryTheory.MorphismProperty (C₂ j)
      inst✝¹ : ∀ (j : J₂), (W₂ j).ContainsIdentities
      inst✝ : ∀ (j : J₂), (L₂ j).IsLocalization (W₂ j)
      ⊢ (CategoryTheory.Functor.pi L₂).IsLocalization (CategoryTheory.MorphismProper …
    -/
    let L₁ := fun j => (L₂ (e j))
    /-
      case of_equiv
      J₁ J₂ : Type w
      e : Equiv J₁ J₂
      hJ₁ : ∀ {C : J₁ → Type u₁} {D : J₁ → Type u₂} [inst : (j : J₁) → CategoryTheor …
      C₂ : J₂ → Type u₁
      D₂ : J₂ → Type u₂
      inst✝³ : (j : J₂) → CategoryTheory.Category.{v₁, u₁} (C₂ j)
      inst✝² : (j : J₂) → CategoryTheory.Category.{v₂, u₂} (D₂ j)
      L₂ : (j : J₂) → CategoryTheory.Functor (C₂ j) (D₂ j)
      W₂ : (j : J₂) → CategoryTheory.MorphismProperty (C₂ j)
      inst✝¹ : ∀ (j : J₂), (W₂ j).ContainsIdentities
      inst✝ : ∀ (j : J₂), (L₂ j).IsLocalization (W₂ j)
      L₁ : (j : J₁) → CategoryTheory.Functor (C₂ (e j)) (D₂ (e j)) := fun j => L₂ (e …
      ⊢ (CategoryTheory.Functor.pi L₂).IsLocalization (CategoryTheory.MorphismProper …
    -/
    let E := Pi.equivalenceOfEquiv C₂ e
    /-
      case of_equiv
      J₁ J₂ : Type w
      e : Equiv J₁ J₂
      hJ₁ : ∀ {C : J₁ → Type u₁} {D : J₁ → Type u₂} [inst : (j : J₁) → CategoryTheor …
      C₂ : J₂ → Type u₁
      D₂ : J₂ → Type u₂
      inst✝³ : (j : J₂) → CategoryTheory.Category.{v₁, u₁} (C₂ j)
      inst✝² : (j : J₂) → CategoryTheory.Category.{v₂, u₂} (D₂ j)
      L₂ : (j : J₂) → CategoryTheory.Functor (C₂ j) (D₂ j)
      W₂ : (j : J₂) → CategoryTheory.MorphismProperty (C₂ j)
      inst✝¹ : ∀ (j : J₂), (W₂ j).ContainsIdentities
      inst✝ : ∀ (j : J₂), (L₂ j).IsLocalization (W₂ j)
      L₁ : (j : J₁) → CategoryTheory.Functor (C₂ (e j)) (D₂ (e j)) := fun j => L₂ (e …
      E : CategoryTheory.Equivalence ((j : J₁) → C₂ (e j)) ((i : J₂) → C₂ i) := Cate …
      ⊢ (CategoryTheory.Functor.pi L₂).IsLocalization (CategoryTheory.MorphismProper …
    -/
    let E' := Pi.equivalenceOfEquiv D₂ e
    haveI : CatCommSq E.functor (Functor.pi L₁) (Functor.pi L₂) E'.functor :=
      (CatCommSq.hInvEquiv E (Functor.pi L₁) (Functor.pi L₂) E').symm ⟨Iso.refl _⟩
    refine IsLocalization.of_equivalences (Functor.pi L₁)
      (MorphismProperty.pi (fun j => (W₂ (e j)))) (Functor.pi L₂)
      (MorphismProperty.pi W₂) E E' ?_
      (MorphismProperty.IsInvertedBy.pi _ _ (fun _ => Localization.inverts _ _))
    /-
      case of_equiv
      J₁ J₂ : Type w
      e : Equiv J₁ J₂
      hJ₁ : ∀ {C : J₁ → Type u₁} {D : J₁ → Type u₂} [inst : (j : J₁) → CategoryTheor …
      C₂ : J₂ → Type u₁
      D₂ : J₂ → Type u₂
      inst✝³ : (j : J₂) → CategoryTheory.Category.{v₁, u₁} (C₂ j)
      inst✝² : (j : J₂) → CategoryTheory.Category.{v₂, u₂} (D₂ j)
      L₂ : (j : J₂) → CategoryTheory.Functor (C₂ j) (D₂ j)
      W₂ : (j : J₂) → CategoryTheory.MorphismProperty (C₂ j)
      inst✝¹ : ∀ (j : J₂), (W₂ j).ContainsIdentities
      inst✝ : ∀ (j : J₂), (L₂ j).IsLocalization (W₂ j)
      L₁ : (j : J₁) → CategoryTheory.Functor (C₂ (e j)) (D₂ (e j)) := fun j => L₂ (e …
      E : CategoryTheory.Equivalence ((j : J₁) → C₂ (e j)) ((i : J₂) → C₂ i) := Cate …
      E' : CategoryTheory.Equivalence ((j : J₁) → D₂ (e j)) ((i : J₂) → D₂ i) := Cat …
      this : CategoryTheory.CatCommSq E.functor (CategoryTheory.Functor.pi L₁) (Cate …
      ⊢ LE.le (CategoryTheory.MorphismProperty.pi fun j => W₂ (e j)) ((CategoryTheor …
    -/
    intro _ _ f hf
    /-
      case of_equiv
      J₁ J₂ : Type w
      e : Equiv J₁ J₂
      hJ₁ : ∀ {C : J₁ → Type u₁} {D : J₁ → Type u₂} [inst : (j : J₁) → CategoryTheor …
      C₂ : J₂ → Type u₁
      D₂ : J₂ → Type u₂
      inst✝³ : (j : J₂) → CategoryTheory.Category.{v₁, u₁} (C₂ j)
      inst✝² : (j : J₂) → CategoryTheory.Category.{v₂, u₂} (D₂ j)
      L₂ : (j : J₂) → CategoryTheory.Functor (C₂ j) (D₂ j)
      W₂ : (j : J₂) → CategoryTheory.MorphismProperty (C₂ j)
      inst✝¹ : ∀ (j : J₂), (W₂ j).ContainsIdentities
      inst✝ : ∀ (j : J₂), (L₂ j).IsLocalization (W₂ j)
      L₁ : (j : J₁) → CategoryTheory.Functor (C₂ (e j)) (D₂ (e j)) := fun j => L₂ (e …
      E : CategoryTheory.Equivalence ((j : J₁) → C₂ (e j)) ((i : J₂) → C₂ i) := Cate …
      E' : CategoryTheory.Equivalence ((j : J₁) → D₂ (e j)) ((i : J₂) → D₂ i) := Cat …
      this : CategoryTheory.CatCommSq E.functor (CategoryTheory.Functor.pi L₁) (Cate …
      X✝ Y✝ : (i : J₁) → C₂ (e i)
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.MorphismProperty.pi (fun j => W₂ (e j)) f
      ⊢ (CategoryTheory.MorphismProperty.pi W₂).isoClosure.inverseImage E.functor f
    -/
    refine ⟨_, _, E.functor.map f, fun i => ?_, ⟨Iso.refl _⟩⟩
    have H : ∀ {j j' : J₂} (h : j = j') {X Y : C₂ j} (g : X ⟶ Y) (_ : W₂ j g),
        W₂ j' ((Pi.eqToEquivalence C₂ h).functor.map g) := by
      rintro j _ rfl _ _ g hg
      exact hg
    /-
      case of_equiv
      J₁ J₂ : Type w
      e : Equiv J₁ J₂
      hJ₁ : ∀ {C : J₁ → Type u₁} {D : J₁ → Type u₂} [inst : (j : J₁) → CategoryTheor …
      C₂ : J₂ → Type u₁
      D₂ : J₂ → Type u₂
      inst✝³ : (j : J₂) → CategoryTheory.Category.{v₁, u₁} (C₂ j)
      inst✝² : (j : J₂) → CategoryTheory.Category.{v₂, u₂} (D₂ j)
      L₂ : (j : J₂) → CategoryTheory.Functor (C₂ j) (D₂ j)
      W₂ : (j : J₂) → CategoryTheory.MorphismProperty (C₂ j)
      inst✝¹ : ∀ (j : J₂), (W₂ j).ContainsIdentities
      inst✝ : ∀ (j : J₂), (L₂ j).IsLocalization (W₂ j)
      L₁ : (j : J₁) → CategoryTheory.Functor (C₂ (e j)) (D₂ (e j)) := fun j => L₂ (e …
      E : CategoryTheory.Equivalence ((j : J₁) → C₂ (e j)) ((i : J₂) → C₂ i) := Cate …
      E' : CategoryTheory.Equivalence ((j : J₁) → D₂ (e j)) ((i : J₂) → D₂ i) := Cat …
      this : CategoryTheory.CatCommSq E.functor (CategoryTheory.Functor.pi L₁) (Cate …
      X✝ Y✝ : (i : J₁) → C₂ (e i)
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.MorphismProperty.pi (fun j => W₂ (e j)) f
      i : J₂
      H : ∀ {j j' : J₂} (h : Eq j j') {X Y : C₂ j} (g : Quiver.Hom X Y), W₂ j g → W₂ …
      ⊢ W₂ i (E.functor.map f i)
    -/
    exact H (e.apply_symm_apply i) _ (hf (e.symm i))
    /-
      🎉 no goals
    -/
    /-
      case h_empty
      ⊢ ∀ {C : PEmpty.{w + 1} → Type u₁} {D : PEmpty.{w + 1} → Type u₂} [inst : (j : …
    -/
  · intro C D _ _ L W _ _
    /-
      case h_empty
      C : PEmpty.{w + 1} → Type u₁
      D : PEmpty.{w + 1} → Type u₂
      inst✝³ : (j : PEmpty.{w + 1}) → CategoryTheory.Category.{v₁, u₁} (C j)
      inst✝² : (j : PEmpty.{w + 1}) → CategoryTheory.Category.{v₂, u₂} (D j)
      L : (j : PEmpty.{w + 1}) → CategoryTheory.Functor (C j) (D j)
      W : (j : PEmpty.{w + 1}) → CategoryTheory.MorphismProperty (C j)
      inst✝¹ : ∀ (j : PEmpty.{w + 1}), (W j).ContainsIdentities
      inst✝ : ∀ (j : PEmpty.{w + 1}), (L j).IsLocalization (W j)
      ⊢ (CategoryTheory.Functor.pi L).IsLocalization (CategoryTheory.MorphismPropert …
    -/
    haveI : ∀ j, IsEquivalence (L j) := by rintro ⟨⟩
    /-
      case h_empty
      C : PEmpty.{w + 1} → Type u₁
      D : PEmpty.{w + 1} → Type u₂
      inst✝³ : (j : PEmpty.{w + 1}) → CategoryTheory.Category.{v₁, u₁} (C j)
      inst✝² : (j : PEmpty.{w + 1}) → CategoryTheory.Category.{v₂, u₂} (D j)
      L : (j : PEmpty.{w + 1}) → CategoryTheory.Functor (C j) (D j)
      W : (j : PEmpty.{w + 1}) → CategoryTheory.MorphismProperty (C j)
      inst✝¹ : ∀ (j : PEmpty.{w + 1}), (W j).ContainsIdentities
      inst✝ : ∀ (j : PEmpty.{w + 1}), (L j).IsLocalization (W j)
      this : ∀ (j : PEmpty.{w + 1}), (L j).IsEquivalence
      ⊢ (CategoryTheory.Functor.pi L).IsLocalization (CategoryTheory.MorphismPropert …
    -/
    refine IsLocalization.of_isEquivalence _ _ (fun _ _ _ _ => ?_)
    /-
      case h_empty
      C : PEmpty.{w + 1} → Type u₁
      D : PEmpty.{w + 1} → Type u₂
      inst✝³ : (j : PEmpty.{w + 1}) → CategoryTheory.Category.{v₁, u₁} (C j)
      inst✝² : (j : PEmpty.{w + 1}) → CategoryTheory.Category.{v₂, u₂} (D j)
      L : (j : PEmpty.{w + 1}) → CategoryTheory.Functor (C j) (D j)
      W : (j : PEmpty.{w + 1}) → CategoryTheory.MorphismProperty (C j)
      inst✝¹ : ∀ (j : PEmpty.{w + 1}), (W j).ContainsIdentities
      inst✝ : ∀ (j : PEmpty.{w + 1}), (L j).IsLocalization (W j)
      this : ∀ (j : PEmpty.{w + 1}), (L j).IsEquivalence
      x✝³ x✝² : (i : PEmpty.{w + 1}) → C i
      x✝¹ : Quiver.Hom x✝³ x✝²
      x✝ : CategoryTheory.MorphismProperty.pi W x✝¹
      ⊢ CategoryTheory.MorphismProperty.isomorphisms ((i : PEmpty.{w + 1}) → C i) x✝¹
    -/
    rw [MorphismProperty.isomorphisms.iff, isIso_pi_iff]
    /-
      case h_empty
      C : PEmpty.{w + 1} → Type u₁
      D : PEmpty.{w + 1} → Type u₂
      inst✝³ : (j : PEmpty.{w + 1}) → CategoryTheory.Category.{v₁, u₁} (C j)
      inst✝² : (j : PEmpty.{w + 1}) → CategoryTheory.Category.{v₂, u₂} (D j)
      L : (j : PEmpty.{w + 1}) → CategoryTheory.Functor (C j) (D j)
      W : (j : PEmpty.{w + 1}) → CategoryTheory.MorphismProperty (C j)
      inst✝¹ : ∀ (j : PEmpty.{w + 1}), (W j).ContainsIdentities
      inst✝ : ∀ (j : PEmpty.{w + 1}), (L j).IsLocalization (W j)
      this : ∀ (j : PEmpty.{w + 1}), (L j).IsEquivalence
      x✝³ x✝² : (i : PEmpty.{w + 1}) → C i
      x✝¹ : Quiver.Hom x✝³ x✝²
      x✝ : CategoryTheory.MorphismProperty.pi W x✝¹
      ⊢ ∀ (i : PEmpty.{w + 1}), CategoryTheory.IsIso (x✝¹ i)
    -/
    rintro ⟨⟩
    /-
      🎉 no goals
    -/
    /-
      case h_option
      ⊢ ∀ {α : Type w} [inst : Fintype α], (∀ {C : α → Type u₁} {D : α → Type u₂} [i …
    -/
  · intro J _ hJ C D _ _ L W _ _
    /-
      case h_option
      J : Type w
      inst✝⁴ : Fintype J
      hJ : ∀ {C : J → Type u₁} {D : J → Type u₂} [inst : (j : J) → CategoryTheory.Ca …
      C : Option J → Type u₁
      D : Option J → Type u₂
      inst✝³ : (j : Option J) → CategoryTheory.Category.{v₁, u₁} (C j)
      inst✝² : (j : Option J) → CategoryTheory.Category.{v₂, u₂} (D j)
      L : (j : Option J) → CategoryTheory.Functor (C j) (D j)
      W : (j : Option J) → CategoryTheory.MorphismProperty (C j)
      inst✝¹ : ∀ (j : Option J), (W j).ContainsIdentities
      inst✝ : ∀ (j : Option J), (L j).IsLocalization (W j)
      ⊢ (CategoryTheory.Functor.pi L).IsLocalization (CategoryTheory.MorphismPropert …
    -/
    let L₁ := (L none).prod (Functor.pi (fun j => L (some j)))
    haveI : CatCommSq (Pi.optionEquivalence C).symm.functor L₁ (Functor.pi L)
      (Pi.optionEquivalence D).symm.functor :=
        ⟨NatIso.pi' (by rintro (_|i) <;> apply Iso.refl)⟩
    refine IsLocalization.of_equivalences L₁
      ((W none).prod (MorphismProperty.pi (fun j => W (some j)))) (Functor.pi L) _
      (Pi.optionEquivalence C).symm (Pi.optionEquivalence D).symm ?_ ?_
      /-
        case h_option.refine_1
        J : Type w
        inst✝⁴ : Fintype J
        hJ : ∀ {C : J → Type u₁} {D : J → Type u₂} [inst : (j : J) → CategoryTheory.Ca …
        C : Option J → Type u₁
        D : Option J → Type u₂
        inst✝³ : (j : Option J) → CategoryTheory.Category.{v₁, u₁} (C j)
        inst✝² : (j : Option J) → CategoryTheory.Category.{v₂, u₂} (D j)
        L : (j : Option J) → CategoryTheory.Functor (C j) (D j)
        W : (j : Option J) → CategoryTheory.MorphismProperty (C j)
        inst✝¹ : ∀ (j : Option J), (W j).ContainsIdentities
        inst✝ : ∀ (j : Option J), (L j).IsLocalization (W j)
        L₁ : CategoryTheory.Functor (Prod (C Option.none) ((i : J) → C (Option.some i) …
        this : CategoryTheory.CatCommSq (CategoryTheory.Pi.optionEquivalence C).symm.f …
        ⊢ LE.le ((W Option.none).prod (CategoryTheory.MorphismProperty.pi fun j => W ( …
      -/
    · intro ⟨X₁, X₂⟩ ⟨Y₁, Y₂⟩ f ⟨hf₁, hf₂⟩
      /-
        case h_option.refine_1
        J : Type w
        inst✝⁴ : Fintype J
        hJ : ∀ {C : J → Type u₁} {D : J → Type u₂} [inst : (j : J) → CategoryTheory.Ca …
        C : Option J → Type u₁
        D : Option J → Type u₂
        inst✝³ : (j : Option J) → CategoryTheory.Category.{v₁, u₁} (C j)
        inst✝² : (j : Option J) → CategoryTheory.Category.{v₂, u₂} (D j)
        L : (j : Option J) → CategoryTheory.Functor (C j) (D j)
        W : (j : Option J) → CategoryTheory.MorphismProperty (C j)
        inst✝¹ : ∀ (j : Option J), (W j).ContainsIdentities
        inst✝ : ∀ (j : Option J), (L j).IsLocalization (W j)
        L₁ : CategoryTheory.Functor (Prod (C Option.none) ((i : J) → C (Option.some i) …
        this : CategoryTheory.CatCommSq (CategoryTheory.Pi.optionEquivalence C).symm.f …
        X₁ : C Option.none
        X₂ : (i : J) → C (Option.some i)
        Y₁ : C Option.none
        Y₂ : (i : J) → C (Option.some i)
        f : Quiver.Hom { fst := X₁, snd := X₂ } { fst := Y₁, snd := Y₂ }
        hf₁ : W Option.none f.1
        hf₂ : CategoryTheory.MorphismProperty.pi (fun j => W (Option.some j)) f.2
        ⊢ (CategoryTheory.MorphismProperty.pi W).isoClosure.inverseImage (CategoryTheo …
      -/
      refine ⟨_, _, (Pi.optionEquivalence C).inverse.map f, ?_, ⟨Iso.refl _⟩⟩
      /-
        case h_option.refine_1
        J : Type w
        inst✝⁴ : Fintype J
        hJ : ∀ {C : J → Type u₁} {D : J → Type u₂} [inst : (j : J) → CategoryTheory.Ca …
        C : Option J → Type u₁
        D : Option J → Type u₂
        inst✝³ : (j : Option J) → CategoryTheory.Category.{v₁, u₁} (C j)
        inst✝² : (j : Option J) → CategoryTheory.Category.{v₂, u₂} (D j)
        L : (j : Option J) → CategoryTheory.Functor (C j) (D j)
        W : (j : Option J) → CategoryTheory.MorphismProperty (C j)
        inst✝¹ : ∀ (j : Option J), (W j).ContainsIdentities
        inst✝ : ∀ (j : Option J), (L j).IsLocalization (W j)
        L₁ : CategoryTheory.Functor (Prod (C Option.none) ((i : J) → C (Option.some i) …
        this : CategoryTheory.CatCommSq (CategoryTheory.Pi.optionEquivalence C).symm.f …
        X₁ : C Option.none
        X₂ : (i : J) → C (Option.some i)
        Y₁ : C Option.none
        Y₂ : (i : J) → C (Option.some i)
        f : Quiver.Hom { fst := X₁, snd := X₂ } { fst := Y₁, snd := Y₂ }
        hf₁ : W Option.none f.1
        hf₂ : CategoryTheory.MorphismProperty.pi (fun j => W (Option.some j)) f.2
        ⊢ CategoryTheory.MorphismProperty.pi W ((CategoryTheory.Pi.optionEquivalence C …
      -/
      rintro (_|i)
        /-
          case h_option.refine_1.none
          J : Type w
          inst✝⁴ : Fintype J
          hJ : ∀ {C : J → Type u₁} {D : J → Type u₂} [inst : (j : J) → CategoryTheory.Ca …
          C : Option J → Type u₁
          D : Option J → Type u₂
          inst✝³ : (j : Option J) → CategoryTheory.Category.{v₁, u₁} (C j)
          inst✝² : (j : Option J) → CategoryTheory.Category.{v₂, u₂} (D j)
          L : (j : Option J) → CategoryTheory.Functor (C j) (D j)
          W : (j : Option J) → CategoryTheory.MorphismProperty (C j)
          inst✝¹ : ∀ (j : Option J), (W j).ContainsIdentities
          inst✝ : ∀ (j : Option J), (L j).IsLocalization (W j)
          L₁ : CategoryTheory.Functor (Prod (C Option.none) ((i : J) → C (Option.some i) …
          this : CategoryTheory.CatCommSq (CategoryTheory.Pi.optionEquivalence C).symm.f …
          X₁ : C Option.none
          X₂ : (i : J) → C (Option.some i)
          Y₁ : C Option.none
          Y₂ : (i : J) → C (Option.some i)
          f : Quiver.Hom { fst := X₁, snd := X₂ } { fst := Y₁, snd := Y₂ }
          hf₁ : W Option.none f.1
          hf₂ : CategoryTheory.MorphismProperty.pi (fun j => W (Option.some j)) f.2
          ⊢ W Option.none ((CategoryTheory.Pi.optionEquivalence C).inverse.map f Option. …
        -/
      · exact hf₁
        /-
          🎉 no goals
        -/
        /-
          case h_option.refine_1.some
          J : Type w
          inst✝⁴ : Fintype J
          hJ : ∀ {C : J → Type u₁} {D : J → Type u₂} [inst : (j : J) → CategoryTheory.Ca …
          C : Option J → Type u₁
          D : Option J → Type u₂
          inst✝³ : (j : Option J) → CategoryTheory.Category.{v₁, u₁} (C j)
          inst✝² : (j : Option J) → CategoryTheory.Category.{v₂, u₂} (D j)
          L : (j : Option J) → CategoryTheory.Functor (C j) (D j)
          W : (j : Option J) → CategoryTheory.MorphismProperty (C j)
          inst✝¹ : ∀ (j : Option J), (W j).ContainsIdentities
          inst✝ : ∀ (j : Option J), (L j).IsLocalization (W j)
          L₁ : CategoryTheory.Functor (Prod (C Option.none) ((i : J) → C (Option.some i) …
          this : CategoryTheory.CatCommSq (CategoryTheory.Pi.optionEquivalence C).symm.f …
          X₁ : C Option.none
          X₂ : (i : J) → C (Option.some i)
          Y₁ : C Option.none
          Y₂ : (i : J) → C (Option.some i)
          f : Quiver.Hom { fst := X₁, snd := X₂ } { fst := Y₁, snd := Y₂ }
          hf₁ : W Option.none f.1
          hf₂ : CategoryTheory.MorphismProperty.pi (fun j => W (Option.some j)) f.2
          i : J
          ⊢ W (Option.some i) ((CategoryTheory.Pi.optionEquivalence C).inverse.map f (Op …
        -/
      · apply hf₂
        /-
          🎉 no goals
        -/
      /-
        case h_option.refine_2
        J : Type w
        inst✝⁴ : Fintype J
        hJ : ∀ {C : J → Type u₁} {D : J → Type u₂} [inst : (j : J) → CategoryTheory.Ca …
        C : Option J → Type u₁
        D : Option J → Type u₂
        inst✝³ : (j : Option J) → CategoryTheory.Category.{v₁, u₁} (C j)
        inst✝² : (j : Option J) → CategoryTheory.Category.{v₂, u₂} (D j)
        L : (j : Option J) → CategoryTheory.Functor (C j) (D j)
        W : (j : Option J) → CategoryTheory.MorphismProperty (C j)
        inst✝¹ : ∀ (j : Option J), (W j).ContainsIdentities
        inst✝ : ∀ (j : Option J), (L j).IsLocalization (W j)
        L₁ : CategoryTheory.Functor (Prod (C Option.none) ((i : J) → C (Option.some i) …
        this : CategoryTheory.CatCommSq (CategoryTheory.Pi.optionEquivalence C).symm.f …
        ⊢ (CategoryTheory.MorphismProperty.pi W).IsInvertedBy (CategoryTheory.Functor. …
      -/
    · apply MorphismProperty.IsInvertedBy.pi
      /-
        case h_option.refine_2.hF
        J : Type w
        inst✝⁴ : Fintype J
        hJ : ∀ {C : J → Type u₁} {D : J → Type u₂} [inst : (j : J) → CategoryTheory.Ca …
        C : Option J → Type u₁
        D : Option J → Type u₂
        inst✝³ : (j : Option J) → CategoryTheory.Category.{v₁, u₁} (C j)
        inst✝² : (j : Option J) → CategoryTheory.Category.{v₂, u₂} (D j)
        L : (j : Option J) → CategoryTheory.Functor (C j) (D j)
        W : (j : Option J) → CategoryTheory.MorphismProperty (C j)
        inst✝¹ : ∀ (j : Option J), (W j).ContainsIdentities
        inst✝ : ∀ (j : Option J), (L j).IsLocalization (W j)
        L₁ : CategoryTheory.Functor (Prod (C Option.none) ((i : J) → C (Option.some i) …
        this : CategoryTheory.CatCommSq (CategoryTheory.Pi.optionEquivalence C).symm.f …
        ⊢ ∀ (j : Option J), (W j).IsInvertedBy (L j)
      -/
                       /-
                         🎉 no goals
                       -/
      rintro (_|i) <;> apply Localization.inverts
                       /-
                         🎉 no goals
                       -/


/-- If `L : C ⥤ D` is a localization functor for `W : MorphismProperty C`, then
the induced functor `(Discrete J ⥤ C) ⥤ (Discrete J ⥤ D)` is also a localization
for `W.functorCategory (Discrete J)` if `W` contains identities. -/
instance {J : Type} [Finite J] {C : Type u₁} {D : Type u₂} [Category.{v₁} C] [Category.{v₂} D]
    (L : C ⥤ D) (W : MorphismProperty C) [W.ContainsIdentities] [L.IsLocalization W]  :
    ((whiskeringRight (Discrete J) C D).obj L).IsLocalization
      (W.functorCategory (Discrete J)) := by
  /-
    J : Type
    inst✝⁴ : Finite J
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : W.ContainsIdentities
    inst✝ : L.IsLocalization W
    ⊢ ((CategoryTheory.whiskeringRight (CategoryTheory.Discrete J) C D).obj L).IsL …
  -/
  let E := piEquivalenceFunctorDiscrete J C
  /-
    J : Type
    inst✝⁴ : Finite J
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : W.ContainsIdentities
    inst✝ : L.IsLocalization W
    E : CategoryTheory.Equivalence (J → C) (CategoryTheory.Functor (CategoryTheory …
    ⊢ ((CategoryTheory.whiskeringRight (CategoryTheory.Discrete J) C D).obj L).IsL …
  -/
  let E' := piEquivalenceFunctorDiscrete J D
  /-
    J : Type
    inst✝⁴ : Finite J
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : W.ContainsIdentities
    inst✝ : L.IsLocalization W
    E : CategoryTheory.Equivalence (J → C) (CategoryTheory.Functor (CategoryTheory …
    E' : CategoryTheory.Equivalence (J → D) (CategoryTheory.Functor (CategoryTheor …
    ⊢ ((CategoryTheory.whiskeringRight (CategoryTheory.Discrete J) C D).obj L).IsL …
  -/
  let L₂ := (whiskeringRight (Discrete J) C D).obj L
  /-
    J : Type
    inst✝⁴ : Finite J
    C : Type u₁
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : W.ContainsIdentities
    inst✝ : L.IsLocalization W
    E : CategoryTheory.Equivalence (J → C) (CategoryTheory.Functor (CategoryTheory …
    E' : CategoryTheory.Equivalence (J → D) (CategoryTheory.Functor (CategoryTheor …
    L₂ : CategoryTheory.Functor (CategoryTheory.Functor (CategoryTheory.Discrete J …
    ⊢ ((CategoryTheory.whiskeringRight (CategoryTheory.Discrete J) C D).obj L).IsL …
  -/
  let L₁ := Functor.pi (fun (_ : J) => L)
  have : CatCommSq E.functor L₁ L₂ E'.functor :=
    ⟨(Functor.rightUnitor _).symm ≪≫ isoWhiskerLeft _ E'.counitIso.symm ≪≫
      Functor.associator _ _ _≪≫ isoWhiskerLeft _ ((Functor.associator _ _ _).symm ≪≫
      isoWhiskerRight (by exact Iso.refl _) _) ≪≫ (Functor.associator _ _ _).symm ≪≫
      isoWhiskerRight ((Functor.associator _ _ _).symm ≪≫
      isoWhiskerRight E.unitIso.symm L₁) _ ≪≫ isoWhiskerRight L₁.leftUnitor _⟩
  refine Functor.IsLocalization.of_equivalences L₁
    (MorphismProperty.pi (fun _ => W)) L₂ _ E E' ?_ ?_
    /-
      case refine_1
      J : Type
      inst✝⁴ : Finite J
      C : Type u₁
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.ContainsIdentities
      inst✝ : L.IsLocalization W
      E : CategoryTheory.Equivalence (J → C) (CategoryTheory.Functor (CategoryTheory …
      E' : CategoryTheory.Equivalence (J → D) (CategoryTheory.Functor (CategoryTheor …
      L₂ : CategoryTheory.Functor (CategoryTheory.Functor (CategoryTheory.Discrete J …
      L₁ : CategoryTheory.Functor (J → C) (J → D) := CategoryTheory.Functor.pi fun x …
      this : CategoryTheory.CatCommSq E.functor L₁ L₂ E'.functor
      ⊢ LE.le (CategoryTheory.MorphismProperty.pi fun x => W) ((W.functorCategory (C …
    -/
  · intro X Y f hf
    /-
      case refine_1
      J : Type
      inst✝⁴ : Finite J
      C : Type u₁
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.ContainsIdentities
      inst✝ : L.IsLocalization W
      E : CategoryTheory.Equivalence (J → C) (CategoryTheory.Functor (CategoryTheory …
      E' : CategoryTheory.Equivalence (J → D) (CategoryTheory.Functor (CategoryTheor …
      L₂ : CategoryTheory.Functor (CategoryTheory.Functor (CategoryTheory.Discrete J …
      L₁ : CategoryTheory.Functor (J → C) (J → D) := CategoryTheory.Functor.pi fun x …
      this : CategoryTheory.CatCommSq E.functor L₁ L₂ E'.functor
      X Y : J → C
      f : Quiver.Hom X Y
      hf : CategoryTheory.MorphismProperty.pi (fun x => W) f
      ⊢ (W.functorCategory (CategoryTheory.Discrete J)).isoClosure.inverseImage E.fu …
    -/
    exact MorphismProperty.le_isoClosure _ _ (fun ⟨j⟩ => hf j)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      J : Type
      inst✝⁴ : Finite J
      C : Type u₁
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.ContainsIdentities
      inst✝ : L.IsLocalization W
      E : CategoryTheory.Equivalence (J → C) (CategoryTheory.Functor (CategoryTheory …
      E' : CategoryTheory.Equivalence (J → D) (CategoryTheory.Functor (CategoryTheor …
      L₂ : CategoryTheory.Functor (CategoryTheory.Functor (CategoryTheory.Discrete J …
      L₁ : CategoryTheory.Functor (J → C) (J → D) := CategoryTheory.Functor.pi fun x …
      this : CategoryTheory.CatCommSq E.functor L₁ L₂ E'.functor
      ⊢ (W.functorCategory (CategoryTheory.Discrete J)).IsInvertedBy L₂
    -/
  · intro X Y f hf
    have : ∀ (j : Discrete J), IsIso ((L₂.map f).app j) :=
      fun j => Localization.inverts L W _ (hf j)
    /-
      case refine_2
      J : Type
      inst✝⁴ : Finite J
      C : Type u₁
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : W.ContainsIdentities
      inst✝ : L.IsLocalization W
      E : CategoryTheory.Equivalence (J → C) (CategoryTheory.Functor (CategoryTheory …
      E' : CategoryTheory.Equivalence (J → D) (CategoryTheory.Functor (CategoryTheor …
      L₂ : CategoryTheory.Functor (CategoryTheory.Functor (CategoryTheory.Discrete J …
      L₁ : CategoryTheory.Functor (J → C) (J → D) := CategoryTheory.Functor.pi fun x …
      this✝ : CategoryTheory.CatCommSq E.functor L₁ L₂ E'.functor
      X Y : CategoryTheory.Functor (CategoryTheory.Discrete J) C
      f : Quiver.Hom X Y
      hf : W.functorCategory (CategoryTheory.Discrete J) f
      this : ∀ (j : CategoryTheory.Discrete J), CategoryTheory.IsIso ((L₂.map f).app …
      ⊢ CategoryTheory.IsIso (L₂.map f)
    -/
    apply NatIso.isIso_of_isIso_app
    /-
      🎉 no goals
    -/


