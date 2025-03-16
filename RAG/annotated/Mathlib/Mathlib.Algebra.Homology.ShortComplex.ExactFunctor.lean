/-- An additive functor which preserves homology preserves finite limits. -/
lemma preservesFiniteLimits_of_preservesHomology
    [HasFiniteProducts C] [HasKernels C] : PreservesFiniteLimits F := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁶ : CategoryTheory.Preadditive C
    inst✝⁵ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁴ : F.Additive
    inst✝³ : F.PreservesHomology
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
    inst✝ : CategoryTheory.Limits.HasKernels C
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits F
  -/
  have := fun {X Y : C} (f : X ⟶ Y) ↦ PreservesHomology.preservesKernel F f
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁶ : CategoryTheory.Preadditive C
    inst✝⁵ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁴ : F.Additive
    inst✝³ : F.PreservesHomology
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
    inst✝ : CategoryTheory.Limits.HasKernels C
    this : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit  …
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits F
  -/
  have : HasBinaryBiproducts C := HasBinaryBiproducts.of_hasBinaryProducts
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁶ : CategoryTheory.Preadditive C
    inst✝⁵ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁴ : F.Additive
    inst✝³ : F.PreservesHomology
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
    inst✝ : CategoryTheory.Limits.HasKernels C
    this✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    this : CategoryTheory.Limits.HasBinaryBiproducts C
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits F
  -/
  have : HasEqualizers C := Preadditive.hasEqualizers_of_hasKernels
  have : HasZeroObject D :=
    ⟨F.obj 0, by rw [IsZero.iff_id_eq_zero, ← F.map_id, id_zero, F.map_zero]⟩
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁶ : CategoryTheory.Preadditive C
    inst✝⁵ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁴ : F.Additive
    inst✝³ : F.PreservesHomology
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasFiniteProducts C
    inst✝ : CategoryTheory.Limits.HasKernels C
    this✝² : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimi …
    this✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    this✝ : CategoryTheory.Limits.HasEqualizers C
    this : CategoryTheory.Limits.HasZeroObject D
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits F
  -/
  exact preservesFiniteLimits_of_preservesKernels F
  /-
    🎉 no goals
  -/


/-- An additive which preserves homology preserves finite colimits. -/
lemma preservesFiniteColimits_of_preservesHomology
    [HasFiniteCoproducts C] [HasCokernels C] : PreservesFiniteColimits F := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁶ : CategoryTheory.Preadditive C
    inst✝⁵ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁴ : F.Additive
    inst✝³ : F.PreservesHomology
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝ : CategoryTheory.Limits.HasCokernels C
    ⊢ CategoryTheory.Limits.PreservesFiniteColimits F
  -/
  have := fun {X Y : C} (f : X ⟶ Y) ↦ PreservesHomology.preservesCokernel F f
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁶ : CategoryTheory.Preadditive C
    inst✝⁵ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁴ : F.Additive
    inst✝³ : F.PreservesHomology
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝ : CategoryTheory.Limits.HasCokernels C
    this : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColimi …
    ⊢ CategoryTheory.Limits.PreservesFiniteColimits F
  -/
  have : HasBinaryBiproducts C := HasBinaryBiproducts.of_hasBinaryCoproducts
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁶ : CategoryTheory.Preadditive C
    inst✝⁵ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁴ : F.Additive
    inst✝³ : F.PreservesHomology
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝ : CategoryTheory.Limits.HasCokernels C
    this✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    this : CategoryTheory.Limits.HasBinaryBiproducts C
    ⊢ CategoryTheory.Limits.PreservesFiniteColimits F
  -/
  have : HasCoequalizers C := Preadditive.hasCoequalizers_of_hasCokernels
  have : HasZeroObject D :=
    ⟨F.obj 0, by rw [IsZero.iff_id_eq_zero, ← F.map_id, id_zero, F.map_zero]⟩
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁶ : CategoryTheory.Preadditive C
    inst✝⁵ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁴ : F.Additive
    inst✝³ : F.PreservesHomology
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝ : CategoryTheory.Limits.HasCokernels C
    this✝² : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColi …
    this✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    this✝ : CategoryTheory.Limits.HasCoequalizers C
    this : CategoryTheory.Limits.HasZeroObject D
    ⊢ CategoryTheory.Limits.PreservesFiniteColimits F
  -/
  exact preservesFiniteColimits_of_preservesCokernels F
  /-
    🎉 no goals
  -/


/--
If a functor `F : C ⥤ D` preserves short exact sequences on the left hand side, (i.e.
if `0 ⟶ A ⟶ B ⟶ C ⟶ 0` is exact then `0 ⟶ F(A) ⟶ F(B) ⟶ F(C)` is exact)
then it preserves monomorphism.
-/
lemma preservesMonomorphisms_of_preserves_shortExact_left
    (h : ∀ (S : ShortComplex C), S.ShortExact → (S.map F).Exact ∧ Mono (F.map S.f)) :
    F.PreservesMonomorphisms where
  preserves f := h _ { exact := exact_cokernel f } |>.2


/--
For an addivite functor `F : C ⥤ D` between abelian categories, the following are equivalent:
- `F` preserves short exact sequences on the left hand side, i.e. if `0 ⟶ A ⟶ B ⟶ C ⟶ 0` is exact
  then `0 ⟶ F(A) ⟶ F(B) ⟶ F(C)` is exact.
- `F` preserves exact sequences on the left hand side, i.e. if `A ⟶ B ⟶ C` is exact where `A ⟶ B`
  is mono, then `F(A) ⟶ F(B) ⟶ F(C)` is exact and `F(A) ⟶ F(B)` is mono as well.
- `F` preserves kernels.
- `F` preserves finite limits.
-/
lemma preservesFiniteLimits_tfae : List.TFAE
    [
      ∀ (S : ShortComplex C), S.ShortExact → (S.map F).Exact ∧ Mono (F.map S.f),
      ∀ (S : ShortComplex C), S.Exact ∧ Mono S.f → (S.map F).Exact ∧ Mono (F.map S.f),
      ∀ ⦃X Y : C⦄ (f : X ⟶ Y), PreservesLimit (parallelPair f 0) F,
      PreservesFiniteLimits F
    ] := by
  tfae_have 1 → 2
  | hF, S, ⟨hS, hf⟩ => by
    have := preservesMonomorphisms_of_preserves_shortExact_left F hF
    refine ⟨?_, inferInstance⟩
    let T := ShortComplex.mk S.f (Abelian.coimage.π S.g) (Abelian.comp_coimage_π_eq_zero S.zero)
    let φ : T.map F ⟶ S.map F :=
      { τ₁ := 𝟙 _
        τ₂ := 𝟙 _
        τ₃ := F.map <| Abelian.factorThruCoimage S.g
        comm₂₃ := show 𝟙 _ ≫ F.map _ = F.map (cokernel.π _) ≫ _ by
          rw [Category.id_comp, ← F.map_comp, cokernel.π_desc] }
    exact (exact_iff_of_epi_of_isIso_of_mono φ).1 (hF T ⟨(S.exact_iff_exact_coimage_π).1 hS⟩).1

  tfae_have 2 → 3
  | hF, X, Y, f => by
    refine preservesLimit_of_preserves_limit_cone (kernelIsKernel f) ?_
    apply (KernelFork.isLimitMapConeEquiv _ F).2
    let S := ShortComplex.mk _ _ (kernel.condition f)
    let hS := hF S ⟨exact_kernel f, inferInstance⟩
    have : Mono (S.map F).f := hS.2
    exact hS.1.fIsKernel

  tfae_have 3 → 4
  | hF => by
    exact preservesFiniteLimits_of_preservesKernels F

  tfae_have 4 → 1
  | ⟨_⟩, S, hS =>
    (S.map F).exact_and_mono_f_iff_f_is_kernel |>.2 ⟨KernelFork.mapIsLimit _ hS.fIsKernel F⟩

  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    tfae_1_to_2 : (∀ (S : CategoryTheory.ShortComplex C), S.ShortExact → And (S.ma …
    tfae_2_to_3 : (∀ (S : CategoryTheory.ShortComplex C), And S.Exact (CategoryThe …
    tfae_3_to_4 : (∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), CategoryTheory.Limits.Preserv …
    tfae_4_to_1 : CategoryTheory.Limits.PreservesFiniteLimits F → ∀ (S : CategoryT …
    ⊢ (List.cons (∀ (S : CategoryTheory.ShortComplex C), S.ShortExact → And (S.map …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


/--
If a functor `F : C ⥤ D` preserves exact sequences on the right hand side (i.e.
if `0 ⟶ A ⟶ B ⟶ C ⟶ 0` is exact then `F(A) ⟶ F(B) ⟶ F(C) ⟶ 0` is exact),
then it preserves epimorphisms.
-/
lemma preservesEpimorphisms_of_preserves_shortExact_right
    (h : ∀ (S : ShortComplex C), S.ShortExact → (S.map F).Exact ∧ Epi (F.map S.g)) :
    F.PreservesEpimorphisms where
  preserves f := h _ { exact := exact_kernel f } |>.2


/--
For an addivite functor `F : C ⥤ D` between abelian categories, the following are equivalent:
- `F` preserves short exact sequences on the right hand side, i.e. if `0 ⟶ A ⟶ B ⟶ C ⟶ 0` is
  exact then `F(A) ⟶ F(B) ⟶ F(C) ⟶ 0` is exact.
- `F` preserves exact sequences on the right hand side, i.e. if `A ⟶ B ⟶ C` is exact where `B ⟶ C`
  is epi, then `F(A) ⟶ F(B) ⟶ F(C) ⟶ 0` is exact and `F(B) ⟶ F(C)` is epi as well.
- `F` preserves cokernels.
- `F` preserves finite colimits.
-/
lemma preservesFiniteColimits_tfae : List.TFAE
    [
      ∀ (S : ShortComplex C), S.ShortExact → (S.map F).Exact ∧ Epi (F.map S.g),
      ∀ (S : ShortComplex C), S.Exact ∧ Epi S.g → (S.map F).Exact ∧ Epi (F.map S.g),
      ∀ ⦃X Y : C⦄ (f : X ⟶ Y), PreservesColimit (parallelPair f 0) F,
      PreservesFiniteColimits F
    ] := by
  tfae_have 1 → 2
  | hF, S, ⟨hS, hf⟩ => by
    have := preservesEpimorphisms_of_preserves_shortExact_right F hF
    refine ⟨?_, inferInstance⟩
    let T := ShortComplex.mk (Abelian.image.ι S.f) S.g (Abelian.image_ι_comp_eq_zero S.zero)
    let φ : S.map F ⟶ T.map F :=
      { τ₁ := F.map <| Abelian.factorThruImage S.f
        τ₂ := 𝟙 _
        τ₃ := 𝟙 _
        comm₁₂ := show _ ≫ F.map (kernel.ι _) = F.map _ ≫ 𝟙 _ by
          rw [← F.map_comp, Abelian.image.fac, Category.comp_id] }
    exact (exact_iff_of_epi_of_isIso_of_mono φ).2 (hF T ⟨(S.exact_iff_exact_image_ι).1 hS⟩).1

  tfae_have 2 → 3
  | hF, X, Y, f => by
    refine preservesColimit_of_preserves_colimit_cocone (cokernelIsCokernel f) ?_
    apply (CokernelCofork.isColimitMapCoconeEquiv _ F).2
    let S := ShortComplex.mk _ _ (cokernel.condition f)
    let hS := hF S ⟨exact_cokernel f, inferInstance⟩
    have : Epi (S.map F).g := hS.2
    exact hS.1.gIsCokernel

  tfae_have 3 → 4
  | hF => by
    exact preservesFiniteColimits_of_preservesCokernels F

  tfae_have 4 → 1
  | ⟨_⟩, S, hS => (S.map F).exact_and_epi_g_iff_g_is_cokernel |>.2
    ⟨CokernelCofork.mapIsColimit _ hS.gIsCokernel F⟩

  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    tfae_1_to_2 : (∀ (S : CategoryTheory.ShortComplex C), S.ShortExact → And (S.ma …
    tfae_2_to_3 : (∀ (S : CategoryTheory.ShortComplex C), And S.Exact (CategoryThe …
    tfae_3_to_4 : (∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), CategoryTheory.Limits.Preserv …
    tfae_4_to_1 : CategoryTheory.Limits.PreservesFiniteColimits F → ∀ (S : Categor …
    ⊢ (List.cons (∀ (S : CategoryTheory.ShortComplex C), S.ShortExact → And (S.map …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


/--
For an additive functor `F : C ⥤ D` between abelian categories, the following are equivalent:
- `F` preserves short exact sequences, i.e. if `0 ⟶ A ⟶ B ⟶ C ⟶ 0` is exact then
  `0 ⟶ F(A) ⟶ F(B) ⟶ F(C) ⟶ 0` is exact.
- `F` preserves exact sequences, i.e. if `A ⟶ B ⟶ C` is exact then `F(A) ⟶ F(B) ⟶ F(C)` is exact.
- `F` preserves homology.
- `F` preserves both finite limits and finite colimits.
-/
lemma exact_tfae : List.TFAE
    [
      ∀ (S : ShortComplex C), S.ShortExact → (S.map F).ShortExact,
      ∀ (S : ShortComplex C), S.Exact → (S.map F).Exact,
      PreservesHomology F,
      PreservesFiniteLimits F ∧ PreservesFiniteColimits F
    ] := by
  tfae_have 1 → 3
  | hF => by
    refine ⟨fun {X Y} f ↦ ?_, fun {X Y} f ↦ ?_⟩
    · have h := (preservesFiniteLimits_tfae F |>.out 0 2 |>.1 fun S hS ↦
        And.intro (hF S hS).exact (hF S hS).mono_f)
      exact h f
    · have h := (preservesFiniteColimits_tfae F |>.out 0 2 |>.1 fun S hS ↦
        And.intro (hF S hS).exact (hF S hS).epi_g)
      exact h f

  tfae_have 2 → 1
  | hF, S, hS => by
    have : Mono (S.map F).f := exact_iff_mono _ (by simp) |>.1 <|
      hF (.mk (0 : 0 ⟶ S.X₁) S.f <| by simp) (exact_iff_mono _ (by simp) |>.2 hS.mono_f)
    have : Epi (S.map F).g := exact_iff_epi _ (by simp) |>.1 <|
      hF (.mk S.g (0 : S.X₃ ⟶ 0) <| by simp) (exact_iff_epi _ (by simp) |>.2 hS.epi_g)
    exact ⟨hF S hS.exact⟩

  tfae_have 3 → 4
  | h => ⟨preservesFiniteLimits_of_preservesHomology F,
      preservesFiniteColimits_of_preservesHomology F⟩

  tfae_have 4 → 2
  | ⟨h1, h2⟩, _, h => h.map F

  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    tfae_1_to_3 : (∀ (S : CategoryTheory.ShortComplex C), S.ShortExact → (S.map F) …
    tfae_2_to_1 : (∀ (S : CategoryTheory.ShortComplex C), S.Exact → (S.map F).Exac …
    tfae_3_to_4 : F.PreservesHomology → And (CategoryTheory.Limits.PreservesFinite …
    tfae_4_to_2 : And (CategoryTheory.Limits.PreservesFiniteLimits F) (CategoryThe …
    ⊢ (List.cons (∀ (S : CategoryTheory.ShortComplex C), S.ShortExact → (S.map F). …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


