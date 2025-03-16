lemma lTensor_shortComplex_exact [Flat R M] (C : ShortComplex <| ModuleCat R) (hC : C.Exact) :
    C.map (tensorLeft M) |>.Exact := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    M : ModuleCat R
    inst✝ : Module.Flat R ↑M
    C : CategoryTheory.ShortComplex (ModuleCat R)
    hC : C.Exact
    ⊢ (C.map (CategoryTheory.MonoidalCategory.tensorLeft M)).Exact
  -/
  rw [moduleCat_exact_iff_function_exact] at hC ⊢
  /-
    R : Type u
    inst✝¹ : CommRing R
    M : ModuleCat R
    inst✝ : Module.Flat R ↑M
    C : CategoryTheory.ShortComplex (ModuleCat R)
    hC : Function.Exact ⇑C.f.hom ⇑C.g.hom
    ⊢ Function.Exact ⇑(C.map (CategoryTheory.MonoidalCategory.tensorLeft M)).f.hom …
  -/
  exact lTensor_exact M hC
  /-
    🎉 no goals
  -/


lemma rTensor_shortComplex_exact [Flat R M] (C : ShortComplex <| ModuleCat R) (hC : C.Exact) :
    C.map (tensorRight M) |>.Exact := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    M : ModuleCat R
    inst✝ : Module.Flat R ↑M
    C : CategoryTheory.ShortComplex (ModuleCat R)
    hC : C.Exact
    ⊢ (C.map (CategoryTheory.MonoidalCategory.tensorRight M)).Exact
  -/
  rw [moduleCat_exact_iff_function_exact] at hC ⊢
  /-
    R : Type u
    inst✝¹ : CommRing R
    M : ModuleCat R
    inst✝ : Module.Flat R ↑M
    C : CategoryTheory.ShortComplex (ModuleCat R)
    hC : Function.Exact ⇑C.f.hom ⇑C.g.hom
    ⊢ Function.Exact ⇑(C.map (CategoryTheory.MonoidalCategory.tensorRight M)).f.ho …
  -/
  exact rTensor_exact M hC
  /-
    🎉 no goals
  -/


lemma iff_lTensor_preserves_shortComplex_exact :
    Flat R M ↔
    ∀ (C : ShortComplex <| ModuleCat R) (_ : C.Exact), (C.map (tensorLeft M) |>.Exact) :=
  ⟨fun _ _ ↦ lTensor_shortComplex_exact _ _, fun H ↦ iff_lTensor_exact.2
    fun _ _ _ _ _ _ _ _ _ f g h ↦
      moduleCat_exact_iff_function_exact _ |>.1 <|
      H (.mk (ModuleCat.ofHom f) (ModuleCat.ofHom g)
        (ModuleCat.hom_ext (DFunLike.ext _ _ h.apply_apply_eq_zero)))
          (moduleCat_exact_iff_function_exact _ |>.2 h)⟩


lemma iff_rTensor_preserves_shortComplex_exact :
    Flat R M ↔
    ∀ (C : ShortComplex <| ModuleCat R) (_ : C.Exact), (C.map (tensorRight M) |>.Exact) :=
  ⟨fun _ _ ↦ rTensor_shortComplex_exact _ _, fun H ↦ iff_rTensor_exact.2
    fun _ _ _ _ _ _ _ _ _ f g h ↦
      moduleCat_exact_iff_function_exact _ |>.1 <|
      H (.mk (ModuleCat.ofHom f) (ModuleCat.ofHom g)
        (ModuleCat.hom_ext (DFunLike.ext _ _ h.apply_apply_eq_zero)))
          (moduleCat_exact_iff_function_exact _ |>.2 h)⟩


