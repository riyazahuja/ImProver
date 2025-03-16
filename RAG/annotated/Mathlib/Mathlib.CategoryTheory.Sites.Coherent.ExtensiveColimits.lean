lemma isSheaf_pointwiseColimit [PreservesFiniteProducts (colim (J := J) (C := A))]
    (G : J ⥤ Sheaf (extensiveTopology C) A) :
    Presheaf.IsSheaf (extensiveTopology C) (pointwiseCocone (G ⋙ sheafToPresheaf _ A)).pt := by
  /-
    A : Type u_1
    C : Type u_2
    J : Type u_3
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} A
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} C
    inst✝³ : CategoryTheory.Category.{u_5, u_3} J
    inst✝² : CategoryTheory.FinitaryExtensive C
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J A
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts CategoryTheory.Limits.co …
    G : CategoryTheory.Functor J (CategoryTheory.Sheaf (CategoryTheory.extensiveTo …
    ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.extensiveTopology C) (Catego …
  -/
  rw [Presheaf.isSheaf_iff_preservesFiniteProducts]
  /-
    A : Type u_1
    C : Type u_2
    J : Type u_3
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} A
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} C
    inst✝³ : CategoryTheory.Category.{u_5, u_3} J
    inst✝² : CategoryTheory.FinitaryExtensive C
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J A
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts CategoryTheory.Limits.co …
    G : CategoryTheory.Functor J (CategoryTheory.Sheaf (CategoryTheory.extensiveTo …
    ⊢ CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.Limits.pointwi …
  -/
  dsimp only [pointwiseCocone_pt]
  /-
    A : Type u_1
    C : Type u_2
    J : Type u_3
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} A
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} C
    inst✝³ : CategoryTheory.Category.{u_5, u_3} J
    inst✝² : CategoryTheory.FinitaryExtensive C
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J A
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts CategoryTheory.Limits.co …
    G : CategoryTheory.Functor J (CategoryTheory.Sheaf (CategoryTheory.extensiveTo …
    ⊢ CategoryTheory.Limits.PreservesFiniteProducts ((G.comp (CategoryTheory.sheaf …
  -/
  apply (config := { allowSynthFailures := true } ) comp_preservesFiniteProducts
  have : ∀ (i : J), PreservesFiniteProducts ((G ⋙ sheafToPresheaf _ A).obj i) := fun i ↦ by
    rw [← Presheaf.isSheaf_iff_preservesFiniteProducts]
    exact Sheaf.cond _
  exact ⟨fun _ ↦ preservesLimitsOfShape_of_evaluation _ _ fun d ↦
    inferInstanceAs (PreservesLimitsOfShape _ ((G ⋙ sheafToPresheaf _ _).obj d))⟩


instance [Preadditive A] : PreservesFiniteProducts (colim (J := J) (C := A)) where
  preserves I _ := by
    apply ( config := {allowSynthFailures := true} )
      preservesProductsOfShape_of_preservesBiproductsOfShape
    /-
      case inst
      A : Type u_1
      C : Type u_2
      J : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} A
      inst✝⁴ : CategoryTheory.Category.{?u.6315, u_2} C
      inst✝³ : CategoryTheory.Category.{u_5, u_3} J
      inst✝² : CategoryTheory.FinitaryExtensive C
      inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J A
      inst✝ : CategoryTheory.Preadditive A
      I : Type
      x✝ : Fintype I
      ⊢ CategoryTheory.Limits.PreservesBiproductsOfShape I CategoryTheory.Limits.colim
    -/
    apply preservesBiproductsOfShape_of_preservesCoproductsOfShape
    /-
      🎉 no goals
    -/


instance [PreservesFiniteProducts (colim (J := J) (C := A))] :
    PreservesColimitsOfShape J (sheafToPresheaf (extensiveTopology C) A) where
  preservesColimit {G} := by
    /-
      A : Type u_1
      C : Type u_2
      J : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} A
      inst✝⁴ : CategoryTheory.Category.{u_6, u_2} C
      inst✝³ : CategoryTheory.Category.{u_5, u_3} J
      inst✝² : CategoryTheory.FinitaryExtensive C
      inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J A
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts CategoryTheory.Limits.co …
      G : CategoryTheory.Functor J (CategoryTheory.Sheaf (CategoryTheory.extensiveTo …
      ⊢ CategoryTheory.Limits.PreservesColimit G (CategoryTheory.sheafToPresheaf (Ca …
    -/
    suffices CreatesColimit G (sheafToPresheaf (extensiveTopology C) A) from inferInstance
    /-
      A : Type u_1
      C : Type u_2
      J : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} A
      inst✝⁴ : CategoryTheory.Category.{u_6, u_2} C
      inst✝³ : CategoryTheory.Category.{u_5, u_3} J
      inst✝² : CategoryTheory.FinitaryExtensive C
      inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J A
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts CategoryTheory.Limits.co …
      G : CategoryTheory.Functor J (CategoryTheory.Sheaf (CategoryTheory.extensiveTo …
      ⊢ CategoryTheory.CreatesColimit G (CategoryTheory.sheafToPresheaf (CategoryThe …
    -/
    refine createsColimitOfIsSheaf _ (fun c hc ↦ ?_)
    let i : c.pt ≅ (G ⋙ sheafToPresheaf _ _).flip ⋙ colim :=
      hc.coconePointUniqueUpToIso (pointwiseIsColimit _)
    /-
      A : Type u_1
      C : Type u_2
      J : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} A
      inst✝⁴ : CategoryTheory.Category.{u_6, u_2} C
      inst✝³ : CategoryTheory.Category.{u_5, u_3} J
      inst✝² : CategoryTheory.FinitaryExtensive C
      inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J A
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts CategoryTheory.Limits.co …
      G : CategoryTheory.Functor J (CategoryTheory.Sheaf (CategoryTheory.extensiveTo …
      c : CategoryTheory.Limits.Cocone (G.comp (CategoryTheory.sheafToPresheaf (Cate …
      hc : CategoryTheory.Limits.IsColimit c
      i : CategoryTheory.Iso c.pt ((G.comp (CategoryTheory.sheafToPresheaf (Category …
      ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.extensiveTopology C) c.pt
    -/
    rw [Presheaf.isSheaf_of_iso_iff i]
    /-
      A : Type u_1
      C : Type u_2
      J : Type u_3
      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} A
      inst✝⁴ : CategoryTheory.Category.{u_6, u_2} C
      inst✝³ : CategoryTheory.Category.{u_5, u_3} J
      inst✝² : CategoryTheory.FinitaryExtensive C
      inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape J A
      inst✝ : CategoryTheory.Limits.PreservesFiniteProducts CategoryTheory.Limits.co …
      G : CategoryTheory.Functor J (CategoryTheory.Sheaf (CategoryTheory.extensiveTo …
      c : CategoryTheory.Limits.Cocone (G.comp (CategoryTheory.sheafToPresheaf (Cate …
      hc : CategoryTheory.Limits.IsColimit c
      i : CategoryTheory.Iso c.pt ((G.comp (CategoryTheory.sheafToPresheaf (Category …
      ⊢ CategoryTheory.Presheaf.IsSheaf (CategoryTheory.extensiveTopology C) ((G.com …
    -/
    exact isSheaf_pointwiseColimit _
    /-
      🎉 no goals
    -/


instance [Preadditive A] [HasFiniteColimits A] :
    PreservesFiniteColimits (sheafToPresheaf (extensiveTopology C) A) where
  preservesFiniteColimits _ := inferInstance


