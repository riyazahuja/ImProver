/-- Given `J : GrothendieckTopology C`, `X : C` and `M : A`, this is the associated
sheaf to the presheaf `Presheaf.freeYoneda X M`. -/
noncomputable def freeYoneda (X : C) (M : A) : Sheaf J A :=
  (presheafToSheaf J A).obj (Presheaf.freeYoneda X M)


variable {J} in
/-- The bijection `(Sheaf.freeYoneda J X M ⟶ F) ≃ (M ⟶ F.val.obj (op X))`
when `F : Sheaf J A`, `X : C` and `M : A`. -/
noncomputable def freeYonedaHomEquiv {X : C} {M : A} {F : Sheaf J A} :
    (freeYoneda J X M ⟶ F) ≃ (M ⟶ F.val.obj (op X)) :=
  ((sheafificationAdjunction J A).homEquiv _ _).trans Presheaf.freeYonedaHomEquiv


lemma isSeparating {ι : Type w} {S : ι → A} (hS : IsSeparating (Set.range S)) :
    IsSeparating (Set.range (fun (⟨X, i⟩ : C × ι) ↦ freeYoneda J X (S i))) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} A
    inst✝¹ : CategoryTheory.Limits.HasCoproducts A
    inst✝ : CategoryTheory.HasWeakSheafify J A
    ι : Type w
    S : ι → A
    hS : CategoryTheory.IsSeparating (Set.range S)
    ⊢ CategoryTheory.IsSeparating (Set.range fun x => CategoryTheory.Sheaf.isSepar …
  -/
  intro F G f g hfg
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} A
    inst✝¹ : CategoryTheory.Limits.HasCoproducts A
    inst✝ : CategoryTheory.HasWeakSheafify J A
    ι : Type w
    S : ι → A
    hS : CategoryTheory.IsSeparating (Set.range S)
    F G : CategoryTheory.Sheaf J A
    f g : Quiver.Hom F G
    hfg : ∀ (G_1 : CategoryTheory.Sheaf J A), Membership.mem (Set.range fun x => C …
    ⊢ Eq f g
  -/
  refine (sheafToPresheaf J A).map_injective (Presheaf.isSeparating C hS _ _ ?_)
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} A
    inst✝¹ : CategoryTheory.Limits.HasCoproducts A
    inst✝ : CategoryTheory.HasWeakSheafify J A
    ι : Type w
    S : ι → A
    hS : CategoryTheory.IsSeparating (Set.range S)
    F G : CategoryTheory.Sheaf J A
    f g : Quiver.Hom F G
    hfg : ∀ (G_1 : CategoryTheory.Sheaf J A), Membership.mem (Set.range fun x => C …
    ⊢ ∀ (G_1 : CategoryTheory.Functor (Opposite C) A), Membership.mem (Set.range f …
  -/
  rintro _ ⟨⟨X, i⟩, rfl⟩ a
  /-
    case intro.mk
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} A
    inst✝¹ : CategoryTheory.Limits.HasCoproducts A
    inst✝ : CategoryTheory.HasWeakSheafify J A
    ι : Type w
    S : ι → A
    hS : CategoryTheory.IsSeparating (Set.range S)
    F G : CategoryTheory.Sheaf J A
    f g : Quiver.Hom F G
    hfg : ∀ (G_1 : CategoryTheory.Sheaf J A), Membership.mem (Set.range fun x => C …
    X : C
    i : ι
    a : Quiver.Hom ((fun x => CategoryTheory.Presheaf.isSeparating.match_1 C (fun  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp a ((CategoryTheory.sheafToPresheaf J  …
  -/
  apply ((sheafificationAdjunction _ _).homEquiv _ _).symm.injective
  simpa only [← Adjunction.homEquiv_naturality_right_symm] using
    hfg _ ⟨⟨X, i⟩, rfl⟩ (((sheafificationAdjunction _ _).homEquiv _ _).symm a)


lemma isSeparator {ι : Type w} {S : ι → A} (hS : IsSeparating (Set.range S))
    [HasCoproduct (fun (⟨X, i⟩ : C × ι) ↦ freeYoneda J X (S i))] [Preadditive A] :
    IsSeparator (∐ (fun (⟨X, i⟩ : C × ι) ↦ freeYoneda J X (S i))) :=
  (isSeparating J hS).isSeparator_coproduct


variable (A) in
instance hasSeparator [HasSeparator A] [Preadditive A] [HasCoproducts.{u} A] :
    HasSeparator (Sheaf J A) where
  hasSeparator := ⟨_, isSeparator J (S := fun (_ : Unit) ↦ separator A)
          /-
            C : Type u
            inst✝⁶ : CategoryTheory.Category.{v, u} C
            J : CategoryTheory.GrothendieckTopology C
            A : Type u'
            inst✝⁵ : CategoryTheory.Category.{v', u'} A
            inst✝⁴ : CategoryTheory.Limits.HasCoproducts A
            inst✝³ : CategoryTheory.HasWeakSheafify J A
            inst✝² : CategoryTheory.HasSeparator A
            inst✝¹ : CategoryTheory.Preadditive A
            inst✝ : CategoryTheory.Limits.HasCoproducts A
            ⊢ CategoryTheory.IsSeparating (Set.range fun x => CategoryTheory.separator A)
          -/
      (by simpa using isSeparator_separator A)⟩
          /-
            🎉 no goals
          -/


