universe w in
lemma isSheaf_coherent (P : Cᵒᵖ ⥤ Type w) :
    Presieve.IsSheaf (coherentTopology C) P ↔
    (∀ (B : C) (α : Type) [Finite α] (X : α → C) (π : (a : α) → (X a ⟶ B)),
      EffectiveEpiFamily X π → (Presieve.ofArrows X π).IsSheafFor P) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Precoherent C
    P : CategoryTheory.Functor (Opposite C) (Type w)
    ⊢ Iff (CategoryTheory.Presieve.IsSheaf (CategoryTheory.coherentTopology C) P)  …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Precoherent C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      ⊢ CategoryTheory.Presieve.IsSheaf (CategoryTheory.coherentTopology C) P → ∀ (B …
    -/
  · intro hP B α _ X π h
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Precoherent C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      hP : CategoryTheory.Presieve.IsSheaf (CategoryTheory.coherentTopology C) P
      B : C
      α : Type
      inst✝ : Finite α
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      h : CategoryTheory.EffectiveEpiFamily X π
      ⊢ CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Presieve.ofArrows X π)
    -/
    simp only [coherentTopology, Presieve.isSheaf_coverage] at hP
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Precoherent C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      α : Type
      inst✝ : Finite α
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      h : CategoryTheory.EffectiveEpiFamily X π
      hP : ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheor …
      ⊢ CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Presieve.ofArrows X π)
    -/
    apply hP
    /-
      case mp.a
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Precoherent C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      B : C
      α : Type
      inst✝ : Finite α
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      h : CategoryTheory.EffectiveEpiFamily X π
      hP : ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheor …
      ⊢ Membership.mem ((CategoryTheory.coherentCoverage C).covering B) (CategoryThe …
    -/
    exact ⟨α, inferInstance, X, π, rfl, h⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Precoherent C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      ⊢ (∀ (B : C) (α : Type) [inst : Finite α] (X : α → C) (π : (a : α) → Quiver.Ho …
    -/
  · intro h
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Precoherent C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      h : ∀ (B : C) (α : Type) [inst : Finite α] (X : α → C) (π : (a : α) → Quiver.H …
      ⊢ CategoryTheory.Presieve.IsSheaf (CategoryTheory.coherentTopology C) P
    -/
    simp only [coherentTopology, Presieve.isSheaf_coverage]
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Precoherent C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      h : ∀ (B : C) (α : Type) [inst : Finite α] (X : α → C) (π : (a : α) → Quiver.H …
      ⊢ ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((CategoryTheory.c …
    -/
    rintro B S ⟨α, _, X, π, rfl, hS⟩
    /-
      case mpr.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Precoherent C
      P : CategoryTheory.Functor (Opposite C) (Type w)
      h : ∀ (B : C) (α : Type) [inst : Finite α] (X : α → C) (π : (a : α) → Quiver.H …
      B : C
      α : Type
      w✝ : Finite α
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      hS : CategoryTheory.EffectiveEpiFamily X π
      ⊢ CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Presieve.ofArrows X π)
    -/
    exact h _ _ _ _ hS
    /-
      🎉 no goals
    -/


/-- Every Yoneda-presheaf is a sheaf for the coherent topology. -/
theorem isSheaf_yoneda_obj (W : C) : Presieve.IsSheaf (coherentTopology C) (yoneda.obj W) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Precoherent C
    W : C
    ⊢ CategoryTheory.Presieve.IsSheaf (CategoryTheory.coherentTopology C) (Categor …
  -/
  rw [isSheaf_coherent]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Precoherent C
    W : C
    ⊢ ∀ (B : C) (α : Type) [inst : Finite α] (X : α → C) (π : (a : α) → Quiver.Hom …
  -/
  intro X α _ Y π H
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Precoherent C
    W X : C
    α : Type
    inst✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    H : CategoryTheory.EffectiveEpiFamily Y π
    ⊢ CategoryTheory.Presieve.IsSheafFor (CategoryTheory.yoneda.obj W) (CategoryTh …
  -/
  have h_colim := isColimitOfEffectiveEpiFamilyStruct Y π H.effectiveEpiFamily.some
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Precoherent C
    W X : C
    α : Type
    inst✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    H : CategoryTheory.EffectiveEpiFamily Y π
    h_colim : CategoryTheory.Limits.IsColimit (CategoryTheory.Sieve.generateFamily …
    ⊢ CategoryTheory.Presieve.IsSheafFor (CategoryTheory.yoneda.obj W) (CategoryTh …
  -/
  rw [← Sieve.generateFamily_eq] at h_colim
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Precoherent C
    W X : C
    α : Type
    inst✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    H : CategoryTheory.EffectiveEpiFamily Y π
    h_colim : CategoryTheory.Limits.IsColimit (CategoryTheory.Sieve.generate (Cate …
    ⊢ CategoryTheory.Presieve.IsSheafFor (CategoryTheory.yoneda.obj W) (CategoryTh …
  -/
  intro x hx
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Precoherent C
    W X : C
    α : Type
    inst✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    H : CategoryTheory.EffectiveEpiFamily Y π
    h_colim : CategoryTheory.Limits.IsColimit (CategoryTheory.Sieve.generate (Cate …
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) (Ca …
    hx : x.Compatible
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  let x_ext := Presieve.FamilyOfElements.sieveExtend x
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Precoherent C
    W X : C
    α : Type
    inst✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    H : CategoryTheory.EffectiveEpiFamily Y π
    h_colim : CategoryTheory.Limits.IsColimit (CategoryTheory.Sieve.generate (Cate …
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) (Ca …
    hx : x.Compatible
    x_ext : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) …
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  have hx_ext := Presieve.FamilyOfElements.Compatible.sieveExtend hx
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Precoherent C
    W X : C
    α : Type
    inst✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    H : CategoryTheory.EffectiveEpiFamily Y π
    h_colim : CategoryTheory.Limits.IsColimit (CategoryTheory.Sieve.generate (Cate …
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) (Ca …
    hx : x.Compatible
    x_ext : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) …
    hx_ext : x.sieveExtend.Compatible
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  let S := Sieve.generate (Presieve.ofArrows Y π)
  obtain ⟨t, t_amalg, t_uniq⟩ : ∃! t, x_ext.IsAmalgamation t :=
    (Sieve.forallYonedaIsSheaf_iff_colimit S).mpr ⟨h_colim⟩ W x_ext hx_ext
  /-
    case intro.intro
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Precoherent C
    W X : C
    α : Type
    inst✝ : Finite α
    Y : α → C
    π : (a : α) → Quiver.Hom (Y a) X
    H : CategoryTheory.EffectiveEpiFamily Y π
    h_colim : CategoryTheory.Limits.IsColimit (CategoryTheory.Sieve.generate (Cate …
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) (Ca …
    hx : x.Compatible
    x_ext : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) …
    hx_ext : x.sieveExtend.Compatible
    S : CategoryTheory.Sieve X := CategoryTheory.Sieve.generate (CategoryTheory.Pr …
    t : (CategoryTheory.yoneda.obj W).obj { unop := X }
    t_amalg : x_ext.IsAmalgamation t
    t_uniq : ∀ (y : (CategoryTheory.yoneda.obj W).obj { unop := X }), (fun t => x_ …
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  refine ⟨t, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Precoherent C
      W X : C
      α : Type
      inst✝ : Finite α
      Y : α → C
      π : (a : α) → Quiver.Hom (Y a) X
      H : CategoryTheory.EffectiveEpiFamily Y π
      h_colim : CategoryTheory.Limits.IsColimit (CategoryTheory.Sieve.generate (Cate …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) (Ca …
      hx : x.Compatible
      x_ext : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) …
      hx_ext : x.sieveExtend.Compatible
      S : CategoryTheory.Sieve X := CategoryTheory.Sieve.generate (CategoryTheory.Pr …
      t : (CategoryTheory.yoneda.obj W).obj { unop := X }
      t_amalg : x_ext.IsAmalgamation t
      t_uniq : ∀ (y : (CategoryTheory.yoneda.obj W).obj { unop := X }), (fun t => x_ …
      ⊢ (fun t => x.IsAmalgamation t) t
    -/
  · convert Presieve.isAmalgamation_restrict (Sieve.le_generate (Presieve.ofArrows Y π)) _ _ t_amalg
    /-
      case h.e.h.e'_6.h.h.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Precoherent C
      W X : C
      α : Type
      inst✝ : Finite α
      Y : α → C
      π : (a : α) → Quiver.Hom (Y a) X
      H : CategoryTheory.EffectiveEpiFamily Y π
      h_colim : CategoryTheory.Limits.IsColimit (CategoryTheory.Sieve.generate (Cate …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) (Ca …
      hx : x.Compatible
      x_ext : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) …
      hx_ext : x.sieveExtend.Compatible
      S : CategoryTheory.Sieve X := CategoryTheory.Sieve.generate (CategoryTheory.Pr …
      t : (CategoryTheory.yoneda.obj W).obj { unop := X }
      t_amalg : x_ext.IsAmalgamation t
      t_uniq : ∀ (y : (CategoryTheory.yoneda.obj W).obj { unop := X }), (fun t => x_ …
      ⊢ Eq x (CategoryTheory.Presieve.FamilyOfElements.restrict ⋯ x_ext)
    -/
    exact (Presieve.restrict_extend hx).symm
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Precoherent C
      W X : C
      α : Type
      inst✝ : Finite α
      Y : α → C
      π : (a : α) → Quiver.Hom (Y a) X
      H : CategoryTheory.EffectiveEpiFamily Y π
      h_colim : CategoryTheory.Limits.IsColimit (CategoryTheory.Sieve.generate (Cate …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) (Ca …
      hx : x.Compatible
      x_ext : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj W) …
      hx_ext : x.sieveExtend.Compatible
      S : CategoryTheory.Sieve X := CategoryTheory.Sieve.generate (CategoryTheory.Pr …
      t : (CategoryTheory.yoneda.obj W).obj { unop := X }
      t_amalg : x_ext.IsAmalgamation t
      t_uniq : ∀ (y : (CategoryTheory.yoneda.obj W).obj { unop := X }), (fun t => x_ …
      ⊢ ∀ (y : (CategoryTheory.yoneda.obj W).obj { unop := X }), (fun t => x.IsAmalg …
    -/
  · exact fun y hy ↦ t_uniq y <| Presieve.isAmalgamation_sieveExtend x y hy
    /-
      🎉 no goals
    -/


variable (C) in
/-- The coherent topology on a precoherent category is subcanonical. -/
instance subcanonical : (coherentTopology C).Subcanonical :=
  GrothendieckTopology.Subcanonical.of_isSheaf_yoneda_obj _ isSheaf_yoneda_obj


