noncomputable instance : NormalEpiCategory (HomologicalComplex C c) := ⟨fun p _ =>
  NormalEpi.mk _ (kernel.ι p) (kernel.condition _)
    (isColimitOfEval _ _ (fun _ =>
      Abelian.isColimitMapCoconeOfCokernelCoforkOfπ _ _))⟩


noncomputable instance : NormalMonoCategory (HomologicalComplex C c) := ⟨fun p _ =>
  NormalMono.mk _ (cokernel.π p) (cokernel.condition _)
    (isLimitOfEval _ _ (fun _ =>
      Abelian.isLimitMapConeOfKernelForkOfι _ _))⟩


noncomputable instance : Abelian (HomologicalComplex C c) where


lemma exact_of_degreewise_exact (hS : ∀ (i : ι), (S.map (eval C c i)).Exact) :
    S.Exact := by
  /-
    C : Type u_1
    ι : Type u_2
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (HomologicalComplex C c)
    hS : ∀ (i : ι), (S.map (HomologicalComplex.eval C c i)).Exact
    ⊢ S.Exact
  -/
  simp only [ShortComplex.exact_iff_isZero_homology] at hS ⊢
  /-
    C : Type u_1
    ι : Type u_2
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (HomologicalComplex C c)
    hS : ∀ (i : ι), CategoryTheory.Limits.IsZero (S.map (HomologicalComplex.eval C …
    ⊢ CategoryTheory.Limits.IsZero S.homology
  -/
  rw [IsZero.iff_id_eq_zero]
  /-
    C : Type u_1
    ι : Type u_2
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (HomologicalComplex C c)
    hS : ∀ (i : ι), CategoryTheory.Limits.IsZero (S.map (HomologicalComplex.eval C …
    ⊢ Eq (CategoryTheory.CategoryStruct.id S.homology) 0
  -/
  ext i
  /-
    case h
    C : Type u_1
    ι : Type u_2
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (HomologicalComplex C c)
    hS : ∀ (i : ι), CategoryTheory.Limits.IsZero (S.map (HomologicalComplex.eval C …
    i : ι
    ⊢ Eq ((CategoryTheory.CategoryStruct.id S.homology).f i) (HomologicalComplex.H …
  -/
  apply (IsZero.of_iso (hS i) (S.mapHomologyIso (eval C c i)).symm).eq_of_src
  /-
    🎉 no goals
  -/


lemma shortExact_of_degreewise_shortExact
    (hS : ∀ (i : ι), (S.map (eval C c i)).ShortExact) :
    S.ShortExact where
  mono_f := mono_of_mono_f _ (fun i => (hS i).mono_f)
  epi_g := epi_of_epi_f _ (fun i => (hS i).epi_g)
  exact := exact_of_degreewise_exact S (fun i => (hS i).exact)


lemma exact_iff_degreewise_exact :
    S.Exact ↔ ∀ (i : ι), (S.map (eval C c i)).Exact := by
  /-
    C : Type u_1
    ι : Type u_2
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (HomologicalComplex C c)
    ⊢ Iff S.Exact (∀ (i : ι), (S.map (HomologicalComplex.eval C c i)).Exact)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      ι : Type u_2
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      ⊢ S.Exact → ∀ (i : ι), (S.map (HomologicalComplex.eval C c i)).Exact
    -/
  · intro hS i
    /-
      case mp
      C : Type u_1
      ι : Type u_2
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS : S.Exact
      i : ι
      ⊢ (S.map (HomologicalComplex.eval C c i)).Exact
    -/
    exact hS.map (eval C c i)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      ι : Type u_2
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      ⊢ (∀ (i : ι), (S.map (HomologicalComplex.eval C c i)).Exact) → S.Exact
    -/
  · exact exact_of_degreewise_exact S
    /-
      🎉 no goals
    -/


lemma shortExact_iff_degreewise_shortExact :
    S.ShortExact ↔ ∀ (i : ι), (S.map (eval C c i)).ShortExact := by
  /-
    C : Type u_1
    ι : Type u_2
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (HomologicalComplex C c)
    ⊢ Iff S.ShortExact (∀ (i : ι), (S.map (HomologicalComplex.eval C c i)).ShortEx …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      ι : Type u_2
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      ⊢ S.ShortExact → ∀ (i : ι), (S.map (HomologicalComplex.eval C c i)).ShortExact
    -/
  · intro hS i
    /-
      case mp
      C : Type u_1
      ι : Type u_2
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS : S.ShortExact
      i : ι
      ⊢ (S.map (HomologicalComplex.eval C c i)).ShortExact
    -/
    have := hS.mono_f
    /-
      case mp
      C : Type u_1
      ι : Type u_2
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS : S.ShortExact
      i : ι
      this : CategoryTheory.Mono S.f
      ⊢ (S.map (HomologicalComplex.eval C c i)).ShortExact
    -/
    have := hS.epi_g
    /-
      case mp
      C : Type u_1
      ι : Type u_2
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS : S.ShortExact
      i : ι
      this✝ : CategoryTheory.Mono S.f
      this : CategoryTheory.Epi S.g
      ⊢ (S.map (HomologicalComplex.eval C c i)).ShortExact
    -/
    exact hS.map (eval C c i)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      ι : Type u_2
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      ⊢ (∀ (i : ι), (S.map (HomologicalComplex.eval C c i)).ShortExact) → S.ShortExact
    -/
  · exact shortExact_of_degreewise_shortExact S
    /-
      🎉 no goals
    -/


