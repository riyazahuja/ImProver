/-- A presieve is *extensive* if it is finite and its arrows induce an isomorphism from the
coproduct to the target. -/
class Presieve.Extensive {X : C} (R : Presieve X) : Prop where
  /-- `R` consists of a finite collection of arrows that together induce an isomorphism from the
  coproduct of their sources. -/
  arrows_nonempty_isColimit : ∃ (α : Type) (_ : Finite α) (Z : α → C) (π : (a : α) → (Z a ⟶ X)),
    R = Presieve.ofArrows Z π ∧ Nonempty (IsColimit (Cofan.mk X π))


instance {X : C} (S : Presieve X) [S.Extensive] : S.hasPullbacks where
  has_pullbacks := by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.836, u_2} D
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      X : C
      S : CategoryTheory.Presieve X
      inst✝ : S.Extensive
      ⊢ ∀ {Y Z : C} {f : Quiver.Hom Y X}, S f → ∀ {g : Quiver.Hom Z X}, S g → Catego …
    -/
    obtain ⟨_, _, _, _, rfl, ⟨hc⟩⟩ := Presieve.Extensive.arrows_nonempty_isColimit (R := S)
    /-
      case intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.836, u_2} D
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      X : C
      w✝³ : Type
      w✝² : Finite w✝³
      w✝¹ : w✝³ → C
      w✝ : (a : w✝³) → Quiver.Hom (w✝¹ a) X
      inst✝ : (CategoryTheory.Presieve.ofArrows w✝¹ w✝).Extensive
      hc : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk X w✝)
      ⊢ ∀ {Y Z : C} {f : Quiver.Hom Y X}, CategoryTheory.Presieve.ofArrows w✝¹ w✝ f  …
    -/
    intro _ _ _ _ _ hg
    /-
      case intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.836, u_2} D
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      X : C
      w✝³ : Type
      w✝² : Finite w✝³
      w✝¹ : w✝³ → C
      w✝ : (a : w✝³) → Quiver.Hom (w✝¹ a) X
      inst✝ : (CategoryTheory.Presieve.ofArrows w✝¹ w✝).Extensive
      hc : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk X w✝)
      Y✝ Z✝ : C
      f✝ : Quiver.Hom Y✝ X
      x✝ : CategoryTheory.Presieve.ofArrows w✝¹ w✝ f✝
      g✝ : Quiver.Hom Z✝ X
      hg : CategoryTheory.Presieve.ofArrows w✝¹ w✝ g✝
      ⊢ CategoryTheory.Limits.HasPullback f✝ g✝
    -/
    cases hg
    /-
      case intro.intro.intro.intro.intro.intro.mk
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.836, u_2} D
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      X : C
      w✝³ : Type
      w✝² : Finite w✝³
      w✝¹ : w✝³ → C
      w✝ : (a : w✝³) → Quiver.Hom (w✝¹ a) X
      inst✝ : (CategoryTheory.Presieve.ofArrows w✝¹ w✝).Extensive
      hc : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk X w✝)
      Y✝ : C
      f✝ : Quiver.Hom Y✝ X
      x✝ : CategoryTheory.Presieve.ofArrows w✝¹ w✝ f✝
      i✝ : w✝³
      ⊢ CategoryTheory.Limits.HasPullback f✝ (w✝ i✝)
    -/
    apply FinitaryPreExtensive.hasPullbacks_of_is_coproduct hc
    /-
      🎉 no goals
    -/


/--
A finite product preserving presheaf is a sheaf for the extensive topology on a category which is
`FinitaryPreExtensive`.
-/
theorem isSheafFor_extensive_of_preservesFiniteProducts {X : C} (S : Presieve X) [S.Extensive]
    (F : Cᵒᵖ ⥤ Type w) [PreservesFiniteProducts F] : S.IsSheafFor F  := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    X : C
    S : CategoryTheory.Presieve X
    inst✝¹ : S.Extensive
    F : CategoryTheory.Functor (Opposite C) (Type w)
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts F
    ⊢ CategoryTheory.Presieve.IsSheafFor F S
  -/
  obtain ⟨α, _, Z, π, rfl, ⟨hc⟩⟩ := Extensive.arrows_nonempty_isColimit (R := S)
  have : (ofArrows Z (Cofan.mk X π).inj).hasPullbacks :=
    (inferInstance : (ofArrows Z π).hasPullbacks)
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    X : C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    α : Type
    w✝ : Finite α
    Z : α → C
    π : (a : α) → Quiver.Hom (Z a) X
    inst✝ : (CategoryTheory.Presieve.ofArrows Z π).Extensive
    hc : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk X π)
    this : (CategoryTheory.Presieve.ofArrows Z (CategoryTheory.Limits.Cofan.mk X π …
    ⊢ CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Z π)
  -/
  cases nonempty_fintype α
  /-
    case intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.FinitaryPreExtensive C
    X : C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteProducts F
    α : Type
    w✝ : Finite α
    Z : α → C
    π : (a : α) → Quiver.Hom (Z a) X
    inst✝ : (CategoryTheory.Presieve.ofArrows Z π).Extensive
    hc : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofan.mk X π)
    this : (CategoryTheory.Presieve.ofArrows Z (CategoryTheory.Limits.Cofan.mk X π …
    val✝ : Fintype α
    ⊢ CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Z π)
  -/
  exact isSheafFor_of_preservesProduct _ _ hc
  /-
    🎉 no goals
  -/


instance {α : Type} [Finite α] (Z : α → C) : (ofArrows Z (fun i ↦ Sigma.ι Z i)).Extensive :=
  ⟨⟨α, inferInstance, Z, (fun i ↦ Sigma.ι Z i), rfl, ⟨coproductIsCoproduct _⟩⟩⟩


/-- Every Yoneda-presheaf is a sheaf for the extensive topology. -/
theorem extensiveTopology.isSheaf_yoneda_obj (W : C) : Presieve.IsSheaf (extensiveTopology C)
    (yoneda.obj W) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.FinitaryPreExtensive C
    W : C
    ⊢ CategoryTheory.Presieve.IsSheaf (CategoryTheory.extensiveTopology C) (Catego …
  -/
  rw [extensiveTopology, isSheaf_coverage]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.FinitaryPreExtensive C
    W : C
    ⊢ ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheory.e …
  -/
  intro X R ⟨Y, α, Z, π, hR, hi⟩
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.FinitaryPreExtensive C
    W X : C
    R : CategoryTheory.Presieve X
    Y : Type
    α : Finite Y
    Z : Y → C
    π : (a : Y) → Quiver.Hom (Z a) X
    hR : Eq R (CategoryTheory.Presieve.ofArrows Z π)
    hi : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
    ⊢ CategoryTheory.Presieve.IsSheafFor (CategoryTheory.yoneda.obj W) R
  -/
  have : IsIso (Sigma.desc (Cofan.inj (Cofan.mk X π))) := hi
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.FinitaryPreExtensive C
    W X : C
    R : CategoryTheory.Presieve X
    Y : Type
    α : Finite Y
    Z : Y → C
    π : (a : Y) → Quiver.Hom (Z a) X
    hR : Eq R (CategoryTheory.Presieve.ofArrows Z π)
    hi : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
    this : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc (CategoryTheory. …
    ⊢ CategoryTheory.Presieve.IsSheafFor (CategoryTheory.yoneda.obj W) R
  -/
  have : R.Extensive := ⟨Y, α, Z, π, hR, ⟨Cofan.isColimitOfIsIsoSigmaDesc (Cofan.mk X π)⟩⟩
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.FinitaryPreExtensive C
    W X : C
    R : CategoryTheory.Presieve X
    Y : Type
    α : Finite Y
    Z : Y → C
    π : (a : Y) → Quiver.Hom (Z a) X
    hR : Eq R (CategoryTheory.Presieve.ofArrows Z π)
    hi : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
    this✝ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc (CategoryTheory …
    this : R.Extensive
    ⊢ CategoryTheory.Presieve.IsSheafFor (CategoryTheory.yoneda.obj W) R
  -/
  exact isSheafFor_extensive_of_preservesFiniteProducts _ _
  /-
    🎉 no goals
  -/


/-- The extensive topology on a finitary pre-extensive category is subcanonical. -/
instance extensiveTopology.subcanonical : (extensiveTopology C).Subcanonical :=
  GrothendieckTopology.Subcanonical.of_isSheaf_yoneda_obj _ isSheaf_yoneda_obj


/--
A presheaf of sets on a category which is `FinitaryExtensive` is a sheaf iff it preserves finite
products.
-/
theorem Presieve.isSheaf_iff_preservesFiniteProducts (F : Cᵒᵖ ⥤ Type w) :
    Presieve.IsSheaf (extensiveTopology C) F ↔
    Nonempty (PreservesFiniteProducts F) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.FinitaryPreExtensive C
    inst✝ : CategoryTheory.FinitaryExtensive C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    ⊢ Iff (CategoryTheory.Presieve.IsSheaf (CategoryTheory.extensiveTopology C) F) …
  -/
  refine ⟨fun hF ↦ ⟨⟨fun α _ ↦ ⟨fun {K} ↦ ?_⟩⟩⟩, fun hF ↦ ?_⟩
    /-
      case refine_1
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      hF : CategoryTheory.Presieve.IsSheaf (CategoryTheory.extensiveTopology C) F
      α : Type
      x✝ : Fintype α
      K : CategoryTheory.Functor (CategoryTheory.Discrete α) (Opposite C)
      ⊢ CategoryTheory.Limits.PreservesLimit K F
    -/
  · rw [extensiveTopology, isSheaf_coverage] at hF
    /-
      case refine_1
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      hF : ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheor …
      α : Type
      x✝ : Fintype α
      K : CategoryTheory.Functor (CategoryTheory.Discrete α) (Opposite C)
      ⊢ CategoryTheory.Limits.PreservesLimit K F
    -/
    let Z : α → C := fun i ↦ unop (K.obj ⟨i⟩)
    have : (ofArrows Z (Cofan.mk (∐ Z) (Sigma.ι Z)).inj).hasPullbacks :=
      inferInstanceAs (ofArrows Z (Sigma.ι Z)).hasPullbacks
    have : ∀ (i : α), Mono (Cofan.inj (Cofan.mk (∐ Z) (Sigma.ι Z)) i) :=
      inferInstanceAs <| ∀ (i : α), Mono (Sigma.ι Z i)
    /-
      case refine_1
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      hF : ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheor …
      α : Type
      x✝ : Fintype α
      K : CategoryTheory.Functor (CategoryTheory.Discrete α) (Opposite C)
      Z : α → C := fun i => Opposite.unop (K.obj { as := i })
      this✝ : (CategoryTheory.Presieve.ofArrows Z (CategoryTheory.Limits.Cofan.mk (C …
      this : ∀ (i : α), CategoryTheory.Mono ((CategoryTheory.Limits.Cofan.mk (Catego …
      ⊢ CategoryTheory.Limits.PreservesLimit K F
    -/
    let i : K ≅ Discrete.functor (fun i ↦ op (Z i)) := Discrete.natIsoFunctor
    let _ : PreservesLimit (Discrete.functor (fun i ↦ op (Z i))) F :=
        Presieve.preservesProduct_of_isSheafFor F ?_ initialIsInitial _ (coproductIsCoproduct Z)
        (FinitaryExtensive.isPullback_initial_to_sigma_ι Z)
        (hF (Presieve.ofArrows Z (fun i ↦ Sigma.ι Z i)) ?_)
      /-
        case refine_1.refine_3
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_3, u_1} C
        inst✝¹ : CategoryTheory.FinitaryPreExtensive C
        inst✝ : CategoryTheory.FinitaryExtensive C
        F : CategoryTheory.Functor (Opposite C) (Type w)
        hF : ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheor …
        α : Type
        x✝¹ : Fintype α
        K : CategoryTheory.Functor (CategoryTheory.Discrete α) (Opposite C)
        Z : α → C := fun i => Opposite.unop (K.obj { as := i })
        this✝ : (CategoryTheory.Presieve.ofArrows Z (CategoryTheory.Limits.Cofan.mk (C …
        this : ∀ (i : α), CategoryTheory.Mono ((CategoryTheory.Limits.Cofan.mk (Catego …
        i : CategoryTheory.Iso K (CategoryTheory.Discrete.functor fun i => { unop := Z …
        x✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor fun …
        ⊢ CategoryTheory.Limits.PreservesLimit K F
      -/
    · exact preservesLimit_of_iso_diagram F i.symm
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_1
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_3, u_1} C
        inst✝¹ : CategoryTheory.FinitaryPreExtensive C
        inst✝ : CategoryTheory.FinitaryExtensive C
        F : CategoryTheory.Functor (Opposite C) (Type w)
        hF : ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheor …
        α : Type
        x✝ : Fintype α
        K : CategoryTheory.Functor (CategoryTheory.Discrete α) (Opposite C)
        Z : α → C := fun i => Opposite.unop (K.obj { as := i })
        this✝ : (CategoryTheory.Presieve.ofArrows Z (CategoryTheory.Limits.Cofan.mk (C …
        this : ∀ (i : α), CategoryTheory.Mono ((CategoryTheory.Limits.Cofan.mk (Catego …
        i : CategoryTheory.Iso K (CategoryTheory.Discrete.functor fun i => { unop := Z …
        ⊢ CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Empty …
      -/
    · apply hF
      /-
        case refine_1.refine_1.a
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_3, u_1} C
        inst✝¹ : CategoryTheory.FinitaryPreExtensive C
        inst✝ : CategoryTheory.FinitaryExtensive C
        F : CategoryTheory.Functor (Opposite C) (Type w)
        hF : ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheor …
        α : Type
        x✝ : Fintype α
        K : CategoryTheory.Functor (CategoryTheory.Discrete α) (Opposite C)
        Z : α → C := fun i => Opposite.unop (K.obj { as := i })
        this✝ : (CategoryTheory.Presieve.ofArrows Z (CategoryTheory.Limits.Cofan.mk (C …
        this : ∀ (i : α), CategoryTheory.Mono ((CategoryTheory.Limits.Cofan.mk (Catego …
        i : CategoryTheory.Iso K (CategoryTheory.Discrete.functor fun i => { unop := Z …
        ⊢ Membership.mem ((CategoryTheory.extensiveCoverage C).covering (CategoryTheor …
      -/
      refine ⟨Empty, inferInstance, Empty.elim, IsEmpty.elim inferInstance, rfl, ⟨default,?_, ?_⟩⟩
        /-
          case refine_1.refine_1.a.refine_1
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_3, u_1} C
          inst✝¹ : CategoryTheory.FinitaryPreExtensive C
          inst✝ : CategoryTheory.FinitaryExtensive C
          F : CategoryTheory.Functor (Opposite C) (Type w)
          hF : ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheor …
          α : Type
          x✝ : Fintype α
          K : CategoryTheory.Functor (CategoryTheory.Discrete α) (Opposite C)
          Z : α → C := fun i => Opposite.unop (K.obj { as := i })
          this✝ : (CategoryTheory.Presieve.ofArrows Z (CategoryTheory.Limits.Cofan.mk (C …
          this : ∀ (i : α), CategoryTheory.Mono ((CategoryTheory.Limits.Cofan.mk (Catego …
          i : CategoryTheory.Iso K (CategoryTheory.Discrete.functor fun i => { unop := Z …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.desc fun …
        -/
      · ext b
        /-
          case refine_1.refine_1.a.refine_1.h
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_3, u_1} C
          inst✝¹ : CategoryTheory.FinitaryPreExtensive C
          inst✝ : CategoryTheory.FinitaryExtensive C
          F : CategoryTheory.Functor (Opposite C) (Type w)
          hF : ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheor …
          α : Type
          x✝ : Fintype α
          K : CategoryTheory.Functor (CategoryTheory.Discrete α) (Opposite C)
          Z : α → C := fun i => Opposite.unop (K.obj { as := i })
          this✝ : (CategoryTheory.Presieve.ofArrows Z (CategoryTheory.Limits.Cofan.mk (C …
          this : ∀ (i : α), CategoryTheory.Mono ((CategoryTheory.Limits.Cofan.mk (Catego …
          i : CategoryTheory.Iso K (CategoryTheory.Discrete.functor fun i => { unop := Z …
          b : Empty
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι Empty. …
        -/
        cases b
        /-
          🎉 no goals
        -/
        /-
          case refine_1.refine_1.a.refine_2
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_3, u_1} C
          inst✝¹ : CategoryTheory.FinitaryPreExtensive C
          inst✝ : CategoryTheory.FinitaryExtensive C
          F : CategoryTheory.Functor (Opposite C) (Type w)
          hF : ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheor …
          α : Type
          x✝ : Fintype α
          K : CategoryTheory.Functor (CategoryTheory.Discrete α) (Opposite C)
          Z : α → C := fun i => Opposite.unop (K.obj { as := i })
          this✝ : (CategoryTheory.Presieve.ofArrows Z (CategoryTheory.Limits.Cofan.mk (C …
          this : ∀ (i : α), CategoryTheory.Mono ((CategoryTheory.Limits.Cofan.mk (Catego …
          i : CategoryTheory.Iso K (CategoryTheory.Discrete.functor fun i => { unop := Z …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp Inhabited.default (CategoryTheory.Lim …
        -/
      · simp only [eq_iff_true_of_subsingleton]
        /-
          🎉 no goals
        -/
      /-
        case refine_1.refine_2
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_3, u_1} C
        inst✝¹ : CategoryTheory.FinitaryPreExtensive C
        inst✝ : CategoryTheory.FinitaryExtensive C
        F : CategoryTheory.Functor (Opposite C) (Type w)
        hF : ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheor …
        α : Type
        x✝ : Fintype α
        K : CategoryTheory.Functor (CategoryTheory.Discrete α) (Opposite C)
        Z : α → C := fun i => Opposite.unop (K.obj { as := i })
        this✝ : (CategoryTheory.Presieve.ofArrows Z (CategoryTheory.Limits.Cofan.mk (C …
        this : ∀ (i : α), CategoryTheory.Mono ((CategoryTheory.Limits.Cofan.mk (Catego …
        i : CategoryTheory.Iso K (CategoryTheory.Discrete.functor fun i => { unop := Z …
        ⊢ Membership.mem ((CategoryTheory.extensiveCoverage C).covering (CategoryTheor …
      -/
    · refine ⟨α, inferInstance, Z, (fun i ↦ Sigma.ι Z i), rfl, ?_⟩
      /-
        case refine_1.refine_2
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_3, u_1} C
        inst✝¹ : CategoryTheory.FinitaryPreExtensive C
        inst✝ : CategoryTheory.FinitaryExtensive C
        F : CategoryTheory.Functor (Opposite C) (Type w)
        hF : ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheor …
        α : Type
        x✝ : Fintype α
        K : CategoryTheory.Functor (CategoryTheory.Discrete α) (Opposite C)
        Z : α → C := fun i => Opposite.unop (K.obj { as := i })
        this✝ : (CategoryTheory.Presieve.ofArrows Z (CategoryTheory.Limits.Cofan.mk (C …
        this : ∀ (i : α), CategoryTheory.Mono ((CategoryTheory.Limits.Cofan.mk (Catego …
        i : CategoryTheory.Iso K (CategoryTheory.Discrete.functor fun i => { unop := Z …
        ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc fun i => CategoryTheo …
      -/
      suffices Sigma.desc (fun i ↦ Sigma.ι Z i) = 𝟙 _ by rw [this]; infer_instance
      /-
        case refine_1.refine_2
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_3, u_1} C
        inst✝¹ : CategoryTheory.FinitaryPreExtensive C
        inst✝ : CategoryTheory.FinitaryExtensive C
        F : CategoryTheory.Functor (Opposite C) (Type w)
        hF : ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheor …
        α : Type
        x✝ : Fintype α
        K : CategoryTheory.Functor (CategoryTheory.Discrete α) (Opposite C)
        Z : α → C := fun i => Opposite.unop (K.obj { as := i })
        this✝ : (CategoryTheory.Presieve.ofArrows Z (CategoryTheory.Limits.Cofan.mk (C …
        this : ∀ (i : α), CategoryTheory.Mono ((CategoryTheory.Limits.Cofan.mk (Catego …
        i : CategoryTheory.Iso K (CategoryTheory.Discrete.functor fun i => { unop := Z …
        ⊢ Eq (CategoryTheory.Limits.Sigma.desc fun i => CategoryTheory.Limits.Sigma.ι  …
      -/
      ext
      /-
        case refine_1.refine_2.h
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_3, u_1} C
        inst✝¹ : CategoryTheory.FinitaryPreExtensive C
        inst✝ : CategoryTheory.FinitaryExtensive C
        F : CategoryTheory.Functor (Opposite C) (Type w)
        hF : ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheor …
        α : Type
        x✝ : Fintype α
        K : CategoryTheory.Functor (CategoryTheory.Discrete α) (Opposite C)
        Z : α → C := fun i => Opposite.unop (K.obj { as := i })
        this✝ : (CategoryTheory.Presieve.ofArrows Z (CategoryTheory.Limits.Cofan.mk (C …
        this : ∀ (i : α), CategoryTheory.Mono ((CategoryTheory.Limits.Cofan.mk (Catego …
        i : CategoryTheory.Iso K (CategoryTheory.Discrete.functor fun i => { unop := Z …
        b✝ : α
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι Z b✝)  …
      -/
      simp
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      hF : Nonempty (CategoryTheory.Limits.PreservesFiniteProducts F)
      ⊢ CategoryTheory.Presieve.IsSheaf (CategoryTheory.extensiveTopology C) F
    -/
  · let _ := hF.some
    /-
      case refine_2
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      hF : Nonempty (CategoryTheory.Limits.PreservesFiniteProducts F)
      x✝ : CategoryTheory.Limits.PreservesFiniteProducts F := Nonempty.some hF
      ⊢ CategoryTheory.Presieve.IsSheaf (CategoryTheory.extensiveTopology C) F
    -/
    rw [extensiveTopology, Presieve.isSheaf_coverage]
    /-
      case refine_2
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      hF : Nonempty (CategoryTheory.Limits.PreservesFiniteProducts F)
      x✝ : CategoryTheory.Limits.PreservesFiniteProducts F := Nonempty.some hF
      ⊢ ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheory.e …
    -/
    intro X R ⟨Y, α, Z, π, hR, hi⟩
    /-
      case refine_2
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      hF : Nonempty (CategoryTheory.Limits.PreservesFiniteProducts F)
      x✝ : CategoryTheory.Limits.PreservesFiniteProducts F := Nonempty.some hF
      X : C
      R : CategoryTheory.Presieve X
      Y : Type
      α : Finite Y
      Z : Y → C
      π : (a : Y) → Quiver.Hom (Z a) X
      hR : Eq R (CategoryTheory.Presieve.ofArrows Z π)
      hi : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
      ⊢ CategoryTheory.Presieve.IsSheafFor F R
    -/
    have : IsIso (Sigma.desc (Cofan.inj (Cofan.mk X π))) := hi
    /-
      case refine_2
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      hF : Nonempty (CategoryTheory.Limits.PreservesFiniteProducts F)
      x✝ : CategoryTheory.Limits.PreservesFiniteProducts F := Nonempty.some hF
      X : C
      R : CategoryTheory.Presieve X
      Y : Type
      α : Finite Y
      Z : Y → C
      π : (a : Y) → Quiver.Hom (Z a) X
      hR : Eq R (CategoryTheory.Presieve.ofArrows Z π)
      hi : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
      this : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc (CategoryTheory. …
      ⊢ CategoryTheory.Presieve.IsSheafFor F R
    -/
    have : R.Extensive := ⟨Y, α, Z, π, hR, ⟨Cofan.isColimitOfIsIsoSigmaDesc (Cofan.mk X π)⟩⟩
    /-
      case refine_2
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      hF : Nonempty (CategoryTheory.Limits.PreservesFiniteProducts F)
      x✝ : CategoryTheory.Limits.PreservesFiniteProducts F := Nonempty.some hF
      X : C
      R : CategoryTheory.Presieve X
      Y : Type
      α : Finite Y
      Z : Y → C
      π : (a : Y) → Quiver.Hom (Z a) X
      hR : Eq R (CategoryTheory.Presieve.ofArrows Z π)
      hi : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc π)
      this✝ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc (CategoryTheory …
      this : R.Extensive
      ⊢ CategoryTheory.Presieve.IsSheafFor F R
    -/
    exact isSheafFor_extensive_of_preservesFiniteProducts R F
    /-
      🎉 no goals
    -/


/--
A presheaf on a category which is `FinitaryExtensive` is a sheaf iff it preserves finite products.
-/
theorem Presheaf.isSheaf_iff_preservesFiniteProducts (F : Cᵒᵖ ⥤ D) :
    IsSheaf (extensiveTopology C) F ↔ PreservesFiniteProducts F := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    inst✝¹ : CategoryTheory.FinitaryPreExtensive C
    inst✝ : CategoryTheory.FinitaryExtensive C
    F : CategoryTheory.Functor (Opposite C) D
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf (CategoryTheory.extensiveTopology C) F) …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) D
      ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.extensiveTopology C) F → Cat …
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) D
      h : CategoryTheory.Presheaf.IsSheaf (CategoryTheory.extensiveTopology C) F
      ⊢ CategoryTheory.Limits.PreservesFiniteProducts F
    -/
    rw [IsSheaf] at h
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) D
      h : ∀ (E : D), CategoryTheory.Presieve.IsSheaf (CategoryTheory.extensiveTopolo …
      ⊢ CategoryTheory.Limits.PreservesFiniteProducts F
    -/
    refine ⟨fun J _ ↦ ⟨fun {K} ↦ ⟨fun {c} hc ↦ ?_⟩⟩⟩
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) D
      h : ∀ (E : D), CategoryTheory.Presieve.IsSheaf (CategoryTheory.extensiveTopolo …
      J : Type
      x✝ : Fintype J
      K : CategoryTheory.Functor (CategoryTheory.Discrete J) (Opposite C)
      c : CategoryTheory.Limits.Cone K
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (F.mapCone c))
    -/
    constructor
    /-
      case mp.val
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) D
      h : ∀ (E : D), CategoryTheory.Presieve.IsSheaf (CategoryTheory.extensiveTopolo …
      J : Type
      x✝ : Fintype J
      K : CategoryTheory.Functor (CategoryTheory.Discrete J) (Opposite C)
      c : CategoryTheory.Limits.Cone K
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.Limits.IsLimit (F.mapCone c)
    -/
    apply coyonedaJointlyReflectsLimits
    /-
      case mp.val.hc
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) D
      h : ∀ (E : D), CategoryTheory.Presieve.IsSheaf (CategoryTheory.extensiveTopolo …
      J : Type
      x✝ : Fintype J
      K : CategoryTheory.Functor (CategoryTheory.Discrete J) (Opposite C)
      c : CategoryTheory.Limits.Cone K
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ (X : Opposite D) → CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoneda.o …
    -/
    intro ⟨E⟩
    /-
      case mp.val.hc
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) D
      h : ∀ (E : D), CategoryTheory.Presieve.IsSheaf (CategoryTheory.extensiveTopolo …
      J : Type
      x✝ : Fintype J
      K : CategoryTheory.Functor (CategoryTheory.Discrete J) (Opposite C)
      c : CategoryTheory.Limits.Cone K
      hc : CategoryTheory.Limits.IsLimit c
      E : D
      ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoneda.obj { unop := E }).m …
    -/
    specialize h E
    /-
      case mp.val.hc
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) D
      J : Type
      x✝ : Fintype J
      K : CategoryTheory.Functor (CategoryTheory.Discrete J) (Opposite C)
      c : CategoryTheory.Limits.Cone K
      hc : CategoryTheory.Limits.IsLimit c
      E : D
      h : CategoryTheory.Presieve.IsSheaf (CategoryTheory.extensiveTopology C) (F.co …
      ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoneda.obj { unop := E }).m …
    -/
    rw [Presieve.isSheaf_iff_preservesFiniteProducts] at h
    /-
      case mp.val.hc
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) D
      J : Type
      x✝ : Fintype J
      K : CategoryTheory.Functor (CategoryTheory.Discrete J) (Opposite C)
      c : CategoryTheory.Limits.Cone K
      hc : CategoryTheory.Limits.IsLimit c
      E : D
      h : Nonempty (CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryT …
      ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoneda.obj { unop := E }).m …
    -/
    have : PreservesLimit K (F.comp (coyoneda.obj ⟨E⟩)) := (h.some.preserves J).preservesLimit
    /-
      case mp.val.hc
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) D
      J : Type
      x✝ : Fintype J
      K : CategoryTheory.Functor (CategoryTheory.Discrete J) (Opposite C)
      c : CategoryTheory.Limits.Cone K
      hc : CategoryTheory.Limits.IsLimit c
      E : D
      h : Nonempty (CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryT …
      this : CategoryTheory.Limits.PreservesLimit K (F.comp (CategoryTheory.coyoneda …
      ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.coyoneda.obj { unop := E }).m …
    -/
    exact isLimitOfPreserves (F.comp (coyoneda.obj ⟨E⟩)) hc
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) D
      ⊢ CategoryTheory.Limits.PreservesFiniteProducts F → CategoryTheory.Presheaf.Is …
    -/
  · intro _ E
    /-
      case mpr
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) D
      a✝ : CategoryTheory.Limits.PreservesFiniteProducts F
      E : D
      ⊢ CategoryTheory.Presieve.IsSheaf (CategoryTheory.extensiveTopology C) (F.comp …
    -/
    rw [Presieve.isSheaf_iff_preservesFiniteProducts]
    /-
      case mpr
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.FinitaryExtensive C
      F : CategoryTheory.Functor (Opposite C) D
      a✝ : CategoryTheory.Limits.PreservesFiniteProducts F
      E : D
      ⊢ Nonempty (CategoryTheory.Limits.PreservesFiniteProducts (F.comp (CategoryThe …
    -/
    exact ⟨inferInstance⟩
    /-
      🎉 no goals
    -/


instance (F : Sheaf (extensiveTopology C) D) : PreservesFiniteProducts F.val :=
  (Presheaf.isSheaf_iff_preservesFiniteProducts F.val).mp F.cond


