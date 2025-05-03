/-- `Precoherent` is preserved by equivalence of categories. -/
theorem precoherent (e : C ≌ D) : Precoherent D := e.inverse.reflects_precoherent


instance [EssentiallySmall C] :
    Precoherent (SmallModel C) := (equivSmallModel C).precoherent


instance (e : C ≌ D) : haveI := precoherent e
    e.inverse.IsDenseSubsite (coherentTopology D) (coherentTopology C) where
  functorPushforward_mem_iff := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      inst✝ : CategoryTheory.Precoherent C
      e : CategoryTheory.Equivalence C D
      ⊢ ∀ {X : D} {S : CategoryTheory.Sieve X}, Iff (Membership.mem ((CategoryTheory …
    -/
    rw [coherentTopology.eq_induced e.inverse]
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      inst✝ : CategoryTheory.Precoherent C
      e : CategoryTheory.Equivalence C D
      ⊢ ∀ {X : D} {S : CategoryTheory.Sieve X}, Iff (Membership.mem ((CategoryTheory …
    -/
    simp only [Functor.mem_inducedTopology_sieves_iff, implies_true]
    /-
      🎉 no goals
    -/


/--
Equivalent precoherent categories give equivalent coherent toposes.
-/
@[simps!]
def sheafCongrPrecoherent (e : C ≌ D) : haveI := e.precoherent
    Sheaf (coherentTopology C) A ≌ Sheaf (coherentTopology D) A := e.sheafCongr _ _ _


/--
The coherent sheaf condition can be checked after precomposing with the equivalence.
-/
theorem precoherent_isSheaf_iff (e : C ≌ D) (F : Cᵒᵖ ⥤ A) : haveI := e.precoherent
    IsSheaf (coherentTopology C) F ↔ IsSheaf (coherentTopology D) (e.inverse.op ⋙ F) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    inst✝¹ : CategoryTheory.Precoherent C
    A : Type u_3
    inst✝ : CategoryTheory.Category.{u_6, u_3} A
    e : CategoryTheory.Equivalence C D
    F : CategoryTheory.Functor (Opposite C) A
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology C) F)  …
  -/
  refine ⟨fun hF ↦ ((e.sheafCongrPrecoherent A).functor.obj ⟨F, hF⟩).cond, fun hF ↦ ?_⟩
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    inst✝¹ : CategoryTheory.Precoherent C
    A : Type u_3
    inst✝ : CategoryTheory.Category.{u_6, u_3} A
    e : CategoryTheory.Equivalence C D
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology D) (e.in …
    ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology C) F
  -/
  rw [isSheaf_of_iso_iff (P' := e.functor.op ⋙ e.inverse.op ⋙ F)]
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_5, u_2} D
      inst✝¹ : CategoryTheory.Precoherent C
      A : Type u_3
      inst✝ : CategoryTheory.Category.{u_6, u_3} A
      e : CategoryTheory.Equivalence C D
      F : CategoryTheory.Functor (Opposite C) A
      hF : CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology D) (e.in …
      ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology C) (e.funct …
    -/
  · exact (e.sheafCongrPrecoherent A).inverse.obj ⟨e.inverse.op ⋙ F, hF⟩ |>.cond
    /-
      🎉 no goals
    -/
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_5, u_2} D
      inst✝¹ : CategoryTheory.Precoherent C
      A : Type u_3
      inst✝ : CategoryTheory.Category.{u_6, u_3} A
      e : CategoryTheory.Equivalence C D
      F : CategoryTheory.Functor (Opposite C) A
      hF : CategoryTheory.Presheaf.IsSheaf (CategoryTheory.coherentTopology D) (e.in …
      ⊢ CategoryTheory.Iso F (e.functor.op.comp (e.inverse.op.comp F))
    -/
  · exact isoWhiskerRight e.op.unitIso F
    /-
      🎉 no goals
    -/


/--
The coherent sheaf condition on an essentially small site can be checked after precomposing with
the equivalence with a small category.
-/
theorem precoherent_isSheaf_iff_of_essentiallySmall [EssentiallySmall C] (F : Cᵒᵖ ⥤ A) :
    IsSheaf (coherentTopology C) F ↔
      IsSheaf (coherentTopology (SmallModel C)) ((equivSmallModel C).inverse.op ⋙ F) :=
  precoherent_isSheaf_iff _ _ _


/-- `Preregular` is preserved by equivalence of categories. -/
theorem preregular (e : C ≌ D) : Preregular D := e.inverse.reflects_preregular


instance [EssentiallySmall C] :
    Preregular (SmallModel C) := (equivSmallModel C).preregular


instance (e : C ≌ D) : haveI := preregular e
    e.inverse.IsDenseSubsite (regularTopology D) (regularTopology C) where
  functorPushforward_mem_iff := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      inst✝ : CategoryTheory.Preregular C
      e : CategoryTheory.Equivalence C D
      ⊢ ∀ {X : D} {S : CategoryTheory.Sieve X}, Iff (Membership.mem ((CategoryTheory …
    -/
    rw [regularTopology.eq_induced e.inverse]
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
      inst✝ : CategoryTheory.Preregular C
      e : CategoryTheory.Equivalence C D
      ⊢ ∀ {X : D} {S : CategoryTheory.Sieve X}, Iff (Membership.mem ((CategoryTheory …
    -/
    simp only [Functor.mem_inducedTopology_sieves_iff, implies_true]
    /-
      🎉 no goals
    -/


/--
Equivalent preregular categories give equivalent regular toposes.
-/
@[simps!]
def sheafCongrPreregular (e : C ≌ D) : haveI := e.preregular
    Sheaf (regularTopology C) A ≌ Sheaf (regularTopology D) A := e.sheafCongr _ _ _


/--
The regular sheaf condition can be checked after precomposing with the equivalence.
-/
theorem preregular_isSheaf_iff (e : C ≌ D) (F : Cᵒᵖ ⥤ A) : haveI := e.preregular
    IsSheaf (regularTopology C) F ↔ IsSheaf (regularTopology D) (e.inverse.op ⋙ F) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    inst✝¹ : CategoryTheory.Preregular C
    A : Type u_3
    inst✝ : CategoryTheory.Category.{u_6, u_3} A
    e : CategoryTheory.Equivalence C D
    F : CategoryTheory.Functor (Opposite C) A
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf (CategoryTheory.regularTopology C) F) ( …
  -/
  refine ⟨fun hF ↦ ((e.sheafCongrPreregular A).functor.obj ⟨F, hF⟩).cond, fun hF ↦ ?_⟩
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_5, u_2} D
    inst✝¹ : CategoryTheory.Preregular C
    A : Type u_3
    inst✝ : CategoryTheory.Category.{u_6, u_3} A
    e : CategoryTheory.Equivalence C D
    F : CategoryTheory.Functor (Opposite C) A
    hF : CategoryTheory.Presheaf.IsSheaf (CategoryTheory.regularTopology D) (e.inv …
    ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.regularTopology C) F
  -/
  rw [isSheaf_of_iso_iff (P' := e.functor.op ⋙ e.inverse.op ⋙ F)]
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_5, u_2} D
      inst✝¹ : CategoryTheory.Preregular C
      A : Type u_3
      inst✝ : CategoryTheory.Category.{u_6, u_3} A
      e : CategoryTheory.Equivalence C D
      F : CategoryTheory.Functor (Opposite C) A
      hF : CategoryTheory.Presheaf.IsSheaf (CategoryTheory.regularTopology D) (e.inv …
      ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.regularTopology C) (e.functo …
    -/
  · exact (e.sheafCongrPreregular A).inverse.obj ⟨e.inverse.op ⋙ F, hF⟩ |>.cond
    /-
      🎉 no goals
    -/
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_4, u_1} C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_5, u_2} D
      inst✝¹ : CategoryTheory.Preregular C
      A : Type u_3
      inst✝ : CategoryTheory.Category.{u_6, u_3} A
      e : CategoryTheory.Equivalence C D
      F : CategoryTheory.Functor (Opposite C) A
      hF : CategoryTheory.Presheaf.IsSheaf (CategoryTheory.regularTopology D) (e.inv …
      ⊢ CategoryTheory.Iso F (e.functor.op.comp (e.inverse.op.comp F))
    -/
  · exact isoWhiskerRight e.op.unitIso F
    /-
      🎉 no goals
    -/


/--
The regular sheaf condition on an essentially small site can be checked after precomposing with
the equivalence with a small category.
-/
theorem preregular_isSheaf_iff_of_essentiallySmall [EssentiallySmall C] (F : Cᵒᵖ ⥤ A) :
    IsSheaf (regularTopology C) F ↔ IsSheaf (regularTopology (SmallModel C))
    ((equivSmallModel C).inverse.op ⋙ F) := preregular_isSheaf_iff _ _ _


