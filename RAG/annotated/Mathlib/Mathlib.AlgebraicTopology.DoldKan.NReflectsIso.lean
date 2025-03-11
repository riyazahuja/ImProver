instance : (N₁ : SimplicialObject C ⥤ Karoubi (ChainComplex C ℕ)).ReflectsIsomorphisms :=
  ⟨fun {X Y} f => by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      ⊢ ∀ [inst : CategoryTheory.IsIso (AlgebraicTopology.DoldKan.N₁.map f)], Catego …
    -/
    intro
    -- restating the result in a way that allows induction on the degree n
    suffices ∀ n : ℕ, IsIso (f.app (op [n])) by
      haveI : ∀ Δ : SimplexCategoryᵒᵖ, IsIso (f.app Δ) := fun Δ => this Δ.unop.len
      apply NatIso.isIso_of_isIso_app
    -- restating the assumption in a more practical form
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsIso (AlgebraicTopology.DoldKan.N₁.map f)
      ⊢ ∀ (n : Nat), CategoryTheory.IsIso (f.app { unop := SimplexCategory.mk n })
    -/
    have h₁ := HomologicalComplex.congr_hom (Karoubi.hom_ext_iff.mp (IsIso.hom_inv_id (N₁.map f)))
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsIso (AlgebraicTopology.DoldKan.N₁.map f)
      h₁ : ∀ (i : Nat), Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicTopology.D …
      ⊢ ∀ (n : Nat), CategoryTheory.IsIso (f.app { unop := SimplexCategory.mk n })
    -/
    have h₂ := HomologicalComplex.congr_hom (Karoubi.hom_ext_iff.mp (IsIso.inv_hom_id (N₁.map f)))
    have h₃ := fun n =>
      Karoubi.HomologicalComplex.p_comm_f_assoc (inv (N₁.map f)) n (f.app (op [n]))
    simp only [N₁_map_f, Karoubi.comp_f, HomologicalComplex.comp_f,
      AlternatingFaceMapComplex.map_f, N₁_obj_p, Karoubi.id_f, assoc] at h₁ h₂ h₃
    -- we have to construct an inverse to f in degree n, by induction on n
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      inst✝ : CategoryTheory.IsIso (AlgebraicTopology.DoldKan.N₁.map f)
      h₂ : ∀ (i : Nat), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.inv  …
      h₃ : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.Do …
      h₁ : ∀ (i : Nat), Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.Do …
      ⊢ ∀ (n : Nat), CategoryTheory.IsIso (f.app { unop := SimplexCategory.mk n })
    -/
    intro n
    induction n with
    -- degree 0
    | zero =>
      use (inv (N₁.map f)).f.f 0
      have h₁₀ := h₁ 0
      have h₂₀ := h₂ 0
      dsimp at h₁₀ h₂₀
      simp only [id_comp, comp_id] at h₁₀ h₂₀
      tauto
    | succ n hn =>
      haveI := hn
      use φ { a := PInfty.f (n + 1) ≫ (inv (N₁.map f)).f.f (n + 1)
              b := fun i => inv (f.app (op [n])) ≫ X.σ i }
      simp only [MorphComponents.id, ← id_φ, ← preComp_φ, preComp, ← postComp_φ, postComp,
        PInfty_f_naturality_assoc, IsIso.hom_inv_id_assoc, assoc, IsIso.inv_hom_id_assoc,
        SimplicialObject.σ_naturality, h₁, h₂, h₃, and_self]⟩


theorem compatibility_N₂_N₁_karoubi :
    N₂ ⋙ (karoubiChainComplexEquivalence C ℕ).functor =
      karoubiFunctorCategoryEmbedding SimplexCategoryᵒᵖ C ⋙
        N₁ ⋙ (karoubiChainComplexEquivalence (Karoubi C) ℕ).functor ⋙
            Functor.mapHomologicalComplex (KaroubiKaroubi.equivalence C).inverse _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    ⊢ Eq (AlgebraicTopology.DoldKan.N₂.comp (CategoryTheory.Idempotents.karoubiCha …
  -/
  refine CategoryTheory.Functor.ext (fun P => ?_) fun P Q f => ?_
    /-
      case refine_1
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      ⊢ Eq ((AlgebraicTopology.DoldKan.N₂.comp (CategoryTheory.Idempotents.karoubiCh …
    -/
  · refine HomologicalComplex.ext ?_ ?_
      /-
        case refine_1.refine_1
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
        ⊢ Eq ((AlgebraicTopology.DoldKan.N₂.comp (CategoryTheory.Idempotents.karoubiCh …
      -/
    · ext n
        /-
          case refine_1.refine_1.h.h_X
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
          n : Nat
          ⊢ Eq (((AlgebraicTopology.DoldKan.N₂.comp (CategoryTheory.Idempotents.karoubiC …
        -/
      · rfl
        /-
          🎉 no goals
        -/
        /-
          case refine_1.refine_1.h.h_p
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
          n : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.N₂.comp  …
        -/
      · dsimp
        /-
          case refine_1.refine_1.h.h_p
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
          n : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        simp only [karoubi_PInfty_f, comp_id, PInfty_f_naturality, id_comp, eqToHom_refl]
        /-
          🎉 no goals
        -/
      /-
        case refine_1.refine_2
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
        ⊢ ∀ (i j : Nat), (ComplexShape.down Nat).Rel i j → Eq (CategoryTheory.Category …
      -/
    · rintro _ n (rfl : n + 1 = _)
      /-
        case refine_1.refine_2
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
        n : Nat
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.N₂.comp  …
      -/
      ext
      /-
        case refine_1.refine_2.h
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
        n : Nat
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.N₂.comp  …
      -/
      have h := (AlternatingFaceMapComplex.map P.p).comm (n + 1) n
      dsimp [N₂, karoubiChainComplexEquivalence,
        KaroubiHomologicalComplexEquivalence.Functor.obj] at h ⊢
      simp only [assoc, Karoubi.eqToHom_f, eqToHom_refl, comp_id,
        karoubi_alternatingFaceMapComplex_d, karoubi_PInfty_f,
        ← HomologicalComplex.Hom.comm_assoc, ← h, app_idem_assoc]
    /-
      case refine_2
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      f : Quiver.Hom P Q
      ⊢ Eq ((AlgebraicTopology.DoldKan.N₂.comp (CategoryTheory.Idempotents.karoubiCh …
    -/
  · ext n
    /-
      case refine_2.h.h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      f : Quiver.Hom P Q
      n : Nat
      ⊢ Eq (((AlgebraicTopology.DoldKan.N₂.comp (CategoryTheory.Idempotents.karoubiC …
    -/
    dsimp [KaroubiKaroubi.inverse, Functor.mapHomologicalComplex]
    simp only [karoubi_PInfty_f, HomologicalComplex.eqToHom_f, Karoubi.eqToHom_f,
      assoc, comp_id, PInfty_f_naturality, app_p_comp,
      karoubiChainComplexEquivalence_functor_obj_X_p, N₂_obj_p_f, eqToHom_refl,
      PInfty_f_naturality_assoc, app_comp_p, PInfty_f_idem_assoc]


/-- We deduce that `N₂ : Karoubi (SimplicialObject C) ⥤ Karoubi (ChainComplex C ℕ))`
reflects isomorphisms from the fact that
`N₁ : SimplicialObject (Karoubi C) ⥤ Karoubi (ChainComplex (Karoubi C) ℕ)` does. -/
instance : (N₂ : Karoubi (SimplicialObject C) ⥤ Karoubi (ChainComplex C ℕ)).ReflectsIsomorphisms :=
  ⟨fun f => by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      A✝ B✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      f : Quiver.Hom A✝ B✝
      ⊢ ∀ [inst : CategoryTheory.IsIso (AlgebraicTopology.DoldKan.N₂.map f)], Catego …
    -/
    intro
    -- The following functor `F` reflects isomorphism because it is
    -- a composition of four functors which reflects isomorphisms.
    -- Then, it suffices to show that `F.map f` is an isomorphism.
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      A✝ B✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      f : Quiver.Hom A✝ B✝
      inst✝ : CategoryTheory.IsIso (AlgebraicTopology.DoldKan.N₂.map f)
      ⊢ CategoryTheory.IsIso f
    -/
    let F₁ := karoubiFunctorCategoryEmbedding SimplexCategoryᵒᵖ C
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      A✝ B✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      f : Quiver.Hom A✝ B✝
      inst✝ : CategoryTheory.IsIso (AlgebraicTopology.DoldKan.N₂.map f)
      F₁ : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi (CategoryTheor …
      ⊢ CategoryTheory.IsIso f
    -/
    let F₂ : SimplicialObject (Karoubi C) ⥤ _ := N₁
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      A✝ B✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      f : Quiver.Hom A✝ B✝
      inst✝ : CategoryTheory.IsIso (AlgebraicTopology.DoldKan.N₂.map f)
      F₁ : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi (CategoryTheor …
      F₂ : CategoryTheory.Functor (CategoryTheory.SimplicialObject (CategoryTheory.I …
      ⊢ CategoryTheory.IsIso f
    -/
    let F₃ := (karoubiChainComplexEquivalence (Karoubi C) ℕ).functor
    let F₄ := Functor.mapHomologicalComplex (KaroubiKaroubi.equivalence C).inverse
      (ComplexShape.down ℕ)
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      A✝ B✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      f : Quiver.Hom A✝ B✝
      inst✝ : CategoryTheory.IsIso (AlgebraicTopology.DoldKan.N₂.map f)
      F₁ : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi (CategoryTheor …
      F₂ : CategoryTheory.Functor (CategoryTheory.SimplicialObject (CategoryTheory.I …
      F₃ : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi (ChainComplex  …
      F₄ : CategoryTheory.Functor (HomologicalComplex (CategoryTheory.Idempotents.Ka …
      ⊢ CategoryTheory.IsIso f
    -/
    let F := F₁ ⋙ F₂ ⋙ F₃ ⋙ F₄
    -- Porting note: we have to help Lean4 find the `ReflectsIsomorphisms` instances
    -- could this be fixed by setting better instance priorities?
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      A✝ B✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      f : Quiver.Hom A✝ B✝
      inst✝ : CategoryTheory.IsIso (AlgebraicTopology.DoldKan.N₂.map f)
      F₁ : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi (CategoryTheor …
      F₂ : CategoryTheory.Functor (CategoryTheory.SimplicialObject (CategoryTheory.I …
      F₃ : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi (ChainComplex  …
      F₄ : CategoryTheory.Functor (HomologicalComplex (CategoryTheory.Idempotents.Ka …
      F : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi (CategoryTheory …
      ⊢ CategoryTheory.IsIso f
    -/
    haveI : F₁.ReflectsIsomorphisms := reflectsIsomorphisms_of_full_and_faithful _
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      A✝ B✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      f : Quiver.Hom A✝ B✝
      inst✝ : CategoryTheory.IsIso (AlgebraicTopology.DoldKan.N₂.map f)
      F₁ : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi (CategoryTheor …
      F₂ : CategoryTheory.Functor (CategoryTheory.SimplicialObject (CategoryTheory.I …
      F₃ : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi (ChainComplex  …
      F₄ : CategoryTheory.Functor (HomologicalComplex (CategoryTheory.Idempotents.Ka …
      F : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi (CategoryTheory …
      this : F₁.ReflectsIsomorphisms
      ⊢ CategoryTheory.IsIso f
    -/
    haveI : F₂.ReflectsIsomorphisms := by infer_instance
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      A✝ B✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      f : Quiver.Hom A✝ B✝
      inst✝ : CategoryTheory.IsIso (AlgebraicTopology.DoldKan.N₂.map f)
      F₁ : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi (CategoryTheor …
      F₂ : CategoryTheory.Functor (CategoryTheory.SimplicialObject (CategoryTheory.I …
      F₃ : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi (ChainComplex  …
      F₄ : CategoryTheory.Functor (HomologicalComplex (CategoryTheory.Idempotents.Ka …
      F : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi (CategoryTheory …
      this✝ : F₁.ReflectsIsomorphisms
      this : F₂.ReflectsIsomorphisms
      ⊢ CategoryTheory.IsIso f
    -/
    haveI : F₃.ReflectsIsomorphisms := reflectsIsomorphisms_of_full_and_faithful _
    haveI : ((KaroubiKaroubi.equivalence C).inverse).ReflectsIsomorphisms :=
      reflectsIsomorphisms_of_full_and_faithful _
    have : IsIso (F.map f) := by
      simp only [F]
      rw [← compatibility_N₂_N₁_karoubi, Functor.comp_map]
      apply Functor.map_isIso
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      A✝ B✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      f : Quiver.Hom A✝ B✝
      inst✝ : CategoryTheory.IsIso (AlgebraicTopology.DoldKan.N₂.map f)
      F₁ : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi (CategoryTheor …
      F₂ : CategoryTheory.Functor (CategoryTheory.SimplicialObject (CategoryTheory.I …
      F₃ : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi (ChainComplex  …
      F₄ : CategoryTheory.Functor (HomologicalComplex (CategoryTheory.Idempotents.Ka …
      F : CategoryTheory.Functor (CategoryTheory.Idempotents.Karoubi (CategoryTheory …
      this✝³ : F₁.ReflectsIsomorphisms
      this✝² : F₂.ReflectsIsomorphisms
      this✝¹ : F₃.ReflectsIsomorphisms
      this✝ : (CategoryTheory.Idempotents.KaroubiKaroubi.equivalence C).inverse.Refl …
      this : CategoryTheory.IsIso (F.map f)
      ⊢ CategoryTheory.IsIso f
    -/
    exact isIso_of_reflects_iso f F⟩
    /-
      🎉 no goals
    -/


