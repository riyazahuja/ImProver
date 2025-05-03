/-- The morphism `snakeInput hS₁ i j hij ⟶ snakeInput hS₂ i j hij` induced by
a morphism `φ : S₁ ⟶ S₂` of short complexes of homological complexes, that
are short exact (`hS₁ : S₁.ShortExact` and `hS₂ : S₁.ShortExact`). -/
@[simps]
noncomputable def mapSnakeInput (i j : ι) (hij : c.Rel i j) :
    snakeInput hS₁ i j hij ⟶ snakeInput hS₂ i j hij where
  f₀ := (homologyFunctor C c i).mapShortComplex.map φ
  f₁ := (opcyclesFunctor C c i).mapShortComplex.map φ
  f₂ := (cyclesFunctor C c j).mapShortComplex.map φ
  f₃ := (homologyFunctor C c j).mapShortComplex.map φ


@[reassoc]
lemma δ_naturality (i j : ι) (hij : c.Rel i j) :
    hS₁.δ i j hij ≫ HomologicalComplex.homologyMap φ.τ₁ _ =
      HomologicalComplex.homologyMap φ.τ₃ _ ≫ hS₂.δ i j hij :=
  ShortComplex.SnakeInput.naturality_δ (mapSnakeInput φ hS₁ hS₂ i j hij)


/-- The (exact) sequence `S.X₁.homology i ⟶ S.X₂.homology i ⟶ S.X₃.homology i` -/
@[simp]
noncomputable def composableArrows₂ (i : ι) : ComposableArrows C 2 :=
  mk₂ (homologyMap S.f i) (homologyMap S.g i)


lemma composableArrows₂_exact (hS₁ : S₁.ShortExact) (i : ι) :
    (composableArrows₂ S₁ i).Exact :=
  (hS₁.homology_exact₂ i).exact_toComposableArrows


/-- The (exact) sequence
`H_i(S.X₁) ⟶ H_i(S.X₂) ⟶ H_i(S.X₃) ⟶ H_j(S.X₁) ⟶ H_j(S.X₂) ⟶ H_j(S.X₃)` when `c.Rel i j`
and `S` is a short exact short complex of homological complexes in an abelian category. -/
@[simp]
noncomputable def composableArrows₅ (i j : ι) (hij : c.Rel i j) : ComposableArrows C 5 :=
  mk₅ (homologyMap S₁.f i) (homologyMap S₁.g i) (hS₁.δ i j hij)
    (homologyMap S₁.f j) (homologyMap S₁.g j)


lemma composableArrows₅_exact (i j : ι) (hij : c.Rel i j) :
    (composableArrows₅ hS₁ i j hij).Exact :=
  exact_of_δ₀ (hS₁.homology_exact₂ i).exact_toComposableArrows
    (exact_of_δ₀ (hS₁.homology_exact₃ i j hij).exact_toComposableArrows
      (exact_of_δ₀ (hS₁.homology_exact₁ i j hij).exact_toComposableArrows
        (hS₁.homology_exact₂ j).exact_toComposableArrows))


/-- The map between the exact sequences `S₁.X₁.homology i ⟶ S₁.X₂.homology i ⟶ S₁.X₃.homology i`
and `S₂.X₁.homology i ⟶ S₂.X₂.homology i ⟶ S₂.X₃.homology i` that is induced by `φ : S₁ ⟶ S₂`. -/
@[simp]
noncomputable def mapComposableArrows₂ (i : ι) : composableArrows₂ S₁ i ⟶ composableArrows₂ S₂ i :=
  homMk₂ (homologyMap φ.τ₁ i) (homologyMap φ.τ₂ i) (homologyMap φ.τ₃ i) (by
    /-
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.29664, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
      φ : Quiver.Hom S₁ S₂
      hS₁ : S₁.ShortExact
      hS₂ : S₂.ShortExact
      i : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomologicalComplex.HomologySequence …
    -/
    dsimp
    /-
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.29664, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
      φ : Quiver.Hom S₁ S₂
      hS₁ : S₁.ShortExact
      hS₂ : S₂.ShortExact
      i : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homologyMap S₁.f  …
    -/
    simp only [← homologyMap_comp, φ.comm₁₂]) (by
    /-
      🎉 no goals
    -/
    /-
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.29664, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
      φ : Quiver.Hom S₁ S₂
      hS₁ : S₁.ShortExact
      hS₂ : S₂.ShortExact
      i : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomologicalComplex.HomologySequence …
    -/
    dsimp [Precomp.map]
    /-
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.29664, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
      φ : Quiver.Hom S₁ S₂
      hS₁ : S₁.ShortExact
      hS₂ : S₂.ShortExact
      i : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homologyMap S₁.g  …
    -/
    simp only [← homologyMap_comp, φ.comm₂₃])
    /-
      🎉 no goals
    -/


/-- The map `composableArrows₅ hS₁ i j hij ⟶ composableArrows₅ hS₂ i j hij` of exact
sequences induced by a morphism `φ : S₁ ⟶ S₂` between short exact short complexes of
homological complexes. -/
@[simp]
noncomputable def mapComposableArrows₅ (i j : ι) (hij : c.Rel i j) :
    composableArrows₅ hS₁ i j hij ⟶ composableArrows₅ hS₂ i j hij :=
  homMk₅ (homologyMap φ.τ₁ i) (homologyMap φ.τ₂ i) (homologyMap φ.τ₃ i)
    (homologyMap φ.τ₁ j) (homologyMap φ.τ₂ j) (homologyMap φ.τ₃ j)
     /-
       C : Type u_1
       ι : Type u_2
       inst✝¹ : CategoryTheory.Category.{?u.33903, u_1} C
       inst✝ : CategoryTheory.Abelian C
       c : ComplexShape ι
       S S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
       φ : Quiver.Hom S₁ S₂
       hS₁ : S₁.ShortExact
       hS₂ : S₂.ShortExact
       i j : ι
       hij : c.Rel i j
       ⊢ LE.le 0 1
     -/
     /-
       🎉 no goals
     -/
    (naturality' (mapComposableArrows₂ φ i) 0 1)
     /-
       🎉 no goals
     -/
     /-
       C : Type u_1
       ι : Type u_2
       inst✝¹ : CategoryTheory.Category.{?u.33903, u_1} C
       inst✝ : CategoryTheory.Abelian C
       c : ComplexShape ι
       S S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
       φ : Quiver.Hom S₁ S₂
       hS₁ : S₁.ShortExact
       hS₂ : S₂.ShortExact
       i j : ι
       hij : c.Rel i j
       ⊢ LE.le 1 2
     -/
     /-
       🎉 no goals
     -/
    (naturality' (mapComposableArrows₂ φ i) 1 2)
     /-
       🎉 no goals
     -/
    (δ_naturality φ hS₁ hS₂ i j hij)
     /-
       C : Type u_1
       ι : Type u_2
       inst✝¹ : CategoryTheory.Category.{?u.33903, u_1} C
       inst✝ : CategoryTheory.Abelian C
       c : ComplexShape ι
       S S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
       φ : Quiver.Hom S₁ S₂
       hS₁ : S₁.ShortExact
       hS₂ : S₂.ShortExact
       i j : ι
       hij : c.Rel i j
       ⊢ LE.le 0 1
     -/
     /-
       🎉 no goals
     -/
    (naturality' (mapComposableArrows₂ φ j) 0 1)
     /-
       🎉 no goals
     -/
     /-
       C : Type u_1
       ι : Type u_2
       inst✝¹ : CategoryTheory.Category.{?u.33903, u_1} C
       inst✝ : CategoryTheory.Abelian C
       c : ComplexShape ι
       S S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
       φ : Quiver.Hom S₁ S₂
       hS₁ : S₁.ShortExact
       hS₂ : S₂.ShortExact
       i j : ι
       hij : c.Rel i j
       ⊢ LE.le 1 2
     -/
     /-
       🎉 no goals
     -/
    (naturality' (mapComposableArrows₂ φ j) 1 2)
     /-
       🎉 no goals
     -/


lemma mono_homologyMap_τ₃ (i : ι)
    (h₁ : Epi (homologyMap φ.τ₁ i))
    (h₂ : Mono (homologyMap φ.τ₂ i))
    (h₃ : ∀ j, c.Rel i j → Mono (homologyMap φ.τ₁ j)) :
    Mono (homologyMap φ.τ₃ i) := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    c : ComplexShape ι
    S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
    φ : Quiver.Hom S₁ S₂
    hS₁ : S₁.ShortExact
    hS₂ : S₂.ShortExact
    i : ι
    h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₁ i)
    h₂ : CategoryTheory.Mono (HomologicalComplex.homologyMap φ.τ₂ i)
    h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
    ⊢ CategoryTheory.Mono (HomologicalComplex.homologyMap φ.τ₃ i)
  -/
  by_cases hi : ∃ j, c.Rel i j
    /-
      case pos
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
      φ : Quiver.Hom S₁ S₂
      hS₁ : S₁.ShortExact
      hS₂ : S₂.ShortExact
      i : ι
      h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₁ i)
      h₂ : CategoryTheory.Mono (HomologicalComplex.homologyMap φ.τ₂ i)
      h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
      hi : Exists fun j => c.Rel i j
      ⊢ CategoryTheory.Mono (HomologicalComplex.homologyMap φ.τ₃ i)
    -/
  · obtain ⟨j, hij⟩ := hi
    apply mono_of_epi_of_mono_of_mono
      ((δlastFunctor ⋙ δlastFunctor).map (mapComposableArrows₅ φ hS₁ hS₂ i j hij))
      /-
        case pos.intro.hR₁
        C : Type u_1
        ι : Type u_2
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
        inst✝ : CategoryTheory.Abelian C
        c : ComplexShape ι
        S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
        φ : Quiver.Hom S₁ S₂
        hS₁ : S₁.ShortExact
        hS₂ : S₂.ShortExact
        i : ι
        h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₁ i)
        h₂ : CategoryTheory.Mono (HomologicalComplex.homologyMap φ.τ₂ i)
        h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
        j : ι
        hij : c.Rel i j
        ⊢ ((CategoryTheory.ComposableArrows.δlastFunctor.comp CategoryTheory.Composabl …
      -/
    · exact (composableArrows₅_exact hS₁ i j hij).δlast.δlast
      /-
        🎉 no goals
      -/
      /-
        case pos.intro.hR₂
        C : Type u_1
        ι : Type u_2
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
        inst✝ : CategoryTheory.Abelian C
        c : ComplexShape ι
        S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
        φ : Quiver.Hom S₁ S₂
        hS₁ : S₁.ShortExact
        hS₂ : S₂.ShortExact
        i : ι
        h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₁ i)
        h₂ : CategoryTheory.Mono (HomologicalComplex.homologyMap φ.τ₂ i)
        h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
        j : ι
        hij : c.Rel i j
        ⊢ ((CategoryTheory.ComposableArrows.δlastFunctor.comp CategoryTheory.Composabl …
      -/
    · exact (composableArrows₅_exact hS₂ i j hij).δlast.δlast
      /-
        🎉 no goals
      -/
      /-
        case pos.intro.h₀
        C : Type u_1
        ι : Type u_2
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
        inst✝ : CategoryTheory.Abelian C
        c : ComplexShape ι
        S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
        φ : Quiver.Hom S₁ S₂
        hS₁ : S₁.ShortExact
        hS₂ : S₂.ShortExact
        i : ι
        h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₁ i)
        h₂ : CategoryTheory.Mono (HomologicalComplex.homologyMap φ.τ₂ i)
        h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
        j : ι
        hij : c.Rel i j
        ⊢ CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' ((CategoryTheory.Co …
      -/
    · exact h₁
      /-
        🎉 no goals
      -/
      /-
        case pos.intro.h₁
        C : Type u_1
        ι : Type u_2
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
        inst✝ : CategoryTheory.Abelian C
        c : ComplexShape ι
        S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
        φ : Quiver.Hom S₁ S₂
        hS₁ : S₁.ShortExact
        hS₂ : S₂.ShortExact
        i : ι
        h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₁ i)
        h₂ : CategoryTheory.Mono (HomologicalComplex.homologyMap φ.τ₂ i)
        h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
        j : ι
        hij : c.Rel i j
        ⊢ CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' ((CategoryTheory.C …
      -/
    · exact h₂
      /-
        🎉 no goals
      -/
      /-
        case pos.intro.h₃
        C : Type u_1
        ι : Type u_2
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
        inst✝ : CategoryTheory.Abelian C
        c : ComplexShape ι
        S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
        φ : Quiver.Hom S₁ S₂
        hS₁ : S₁.ShortExact
        hS₂ : S₂.ShortExact
        i : ι
        h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₁ i)
        h₂ : CategoryTheory.Mono (HomologicalComplex.homologyMap φ.τ₂ i)
        h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
        j : ι
        hij : c.Rel i j
        ⊢ CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' ((CategoryTheory.C …
      -/
    · exact h₃ _ hij
      /-
        🎉 no goals
      -/
  · refine mono_of_epi_of_epi_of_mono (mapComposableArrows₂ φ i)
      (composableArrows₂_exact hS₁ i) (composableArrows₂_exact hS₂ i) ?_ h₁ h₂
    /-
      case neg
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
      φ : Quiver.Hom S₁ S₂
      hS₁ : S₁.ShortExact
      hS₂ : S₂.ShortExact
      i : ι
      h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₁ i)
      h₂ : CategoryTheory.Mono (HomologicalComplex.homologyMap φ.τ₂ i)
      h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
      hi : Not (Exists fun j => c.Rel i j)
      ⊢ CategoryTheory.Epi ((HomologicalComplex.HomologySequence.composableArrows₂ S …
    -/
    have := hS₁.epi_g
    /-
      case neg
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
      φ : Quiver.Hom S₁ S₂
      hS₁ : S₁.ShortExact
      hS₂ : S₂.ShortExact
      i : ι
      h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₁ i)
      h₂ : CategoryTheory.Mono (HomologicalComplex.homologyMap φ.τ₂ i)
      h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
      hi : Not (Exists fun j => c.Rel i j)
      this : CategoryTheory.Epi S₁.g
      ⊢ CategoryTheory.Epi ((HomologicalComplex.HomologySequence.composableArrows₂ S …
    -/
    apply epi_homologyMap_of_epi_of_not_rel
    /-
      case neg.hi
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
      φ : Quiver.Hom S₁ S₂
      hS₁ : S₁.ShortExact
      hS₂ : S₂.ShortExact
      i : ι
      h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₁ i)
      h₂ : CategoryTheory.Mono (HomologicalComplex.homologyMap φ.τ₂ i)
      h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
      hi : Not (Exists fun j => c.Rel i j)
      this : CategoryTheory.Epi S₁.g
      ⊢ ∀ (j : ι), Not (c.Rel i j)
    -/
    simpa using hi
    /-
      🎉 no goals
    -/


lemma epi_homologyMap_τ₃ (i : ι)
    (h₁ : Epi (homologyMap φ.τ₂ i))
    (h₂ : ∀ j, c.Rel i j → Epi (homologyMap φ.τ₁ j))
    (h₃ : ∀ j, c.Rel i j → Mono (homologyMap φ.τ₂ j)) :
    Epi (homologyMap φ.τ₃ i) := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    c : ComplexShape ι
    S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
    φ : Quiver.Hom S₁ S₂
    hS₁ : S₁.ShortExact
    hS₂ : S₂.ShortExact
    i : ι
    h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₂ i)
    h₂ : ∀ (j : ι), c.Rel i j → CategoryTheory.Epi (HomologicalComplex.homologyMap …
    h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
    ⊢ CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₃ i)
  -/
  by_cases hi : ∃ j, c.Rel i j
    /-
      case pos
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
      φ : Quiver.Hom S₁ S₂
      hS₁ : S₁.ShortExact
      hS₂ : S₂.ShortExact
      i : ι
      h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₂ i)
      h₂ : ∀ (j : ι), c.Rel i j → CategoryTheory.Epi (HomologicalComplex.homologyMap …
      h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
      hi : Exists fun j => c.Rel i j
      ⊢ CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₃ i)
    -/
  · obtain ⟨j, hij⟩ := hi
    apply epi_of_epi_of_epi_of_mono
      ((δ₀Functor ⋙ δlastFunctor).map (mapComposableArrows₅ φ hS₁ hS₂ i j hij))
      /-
        case pos.intro.hR₁
        C : Type u_1
        ι : Type u_2
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
        inst✝ : CategoryTheory.Abelian C
        c : ComplexShape ι
        S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
        φ : Quiver.Hom S₁ S₂
        hS₁ : S₁.ShortExact
        hS₂ : S₂.ShortExact
        i : ι
        h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₂ i)
        h₂ : ∀ (j : ι), c.Rel i j → CategoryTheory.Epi (HomologicalComplex.homologyMap …
        h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
        j : ι
        hij : c.Rel i j
        ⊢ ((CategoryTheory.ComposableArrows.δ₀Functor.comp CategoryTheory.ComposableAr …
      -/
    · exact (composableArrows₅_exact hS₁ i j hij).δ₀.δlast
      /-
        🎉 no goals
      -/
      /-
        case pos.intro.hR₂
        C : Type u_1
        ι : Type u_2
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
        inst✝ : CategoryTheory.Abelian C
        c : ComplexShape ι
        S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
        φ : Quiver.Hom S₁ S₂
        hS₁ : S₁.ShortExact
        hS₂ : S₂.ShortExact
        i : ι
        h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₂ i)
        h₂ : ∀ (j : ι), c.Rel i j → CategoryTheory.Epi (HomologicalComplex.homologyMap …
        h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
        j : ι
        hij : c.Rel i j
        ⊢ ((CategoryTheory.ComposableArrows.δ₀Functor.comp CategoryTheory.ComposableAr …
      -/
    · exact (composableArrows₅_exact hS₂ i j hij).δ₀.δlast
      /-
        🎉 no goals
      -/
      /-
        case pos.intro.h₀
        C : Type u_1
        ι : Type u_2
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
        inst✝ : CategoryTheory.Abelian C
        c : ComplexShape ι
        S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
        φ : Quiver.Hom S₁ S₂
        hS₁ : S₁.ShortExact
        hS₂ : S₂.ShortExact
        i : ι
        h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₂ i)
        h₂ : ∀ (j : ι), c.Rel i j → CategoryTheory.Epi (HomologicalComplex.homologyMap …
        h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
        j : ι
        hij : c.Rel i j
        ⊢ CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' ((CategoryTheory.Co …
      -/
    · exact h₁
      /-
        🎉 no goals
      -/
      /-
        case pos.intro.h₂
        C : Type u_1
        ι : Type u_2
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
        inst✝ : CategoryTheory.Abelian C
        c : ComplexShape ι
        S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
        φ : Quiver.Hom S₁ S₂
        hS₁ : S₁.ShortExact
        hS₂ : S₂.ShortExact
        i : ι
        h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₂ i)
        h₂ : ∀ (j : ι), c.Rel i j → CategoryTheory.Epi (HomologicalComplex.homologyMap …
        h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
        j : ι
        hij : c.Rel i j
        ⊢ CategoryTheory.Epi (CategoryTheory.ComposableArrows.app' ((CategoryTheory.Co …
      -/
    · exact h₂ j hij
      /-
        🎉 no goals
      -/
      /-
        case pos.intro.h₃
        C : Type u_1
        ι : Type u_2
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
        inst✝ : CategoryTheory.Abelian C
        c : ComplexShape ι
        S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
        φ : Quiver.Hom S₁ S₂
        hS₁ : S₁.ShortExact
        hS₂ : S₂.ShortExact
        i : ι
        h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₂ i)
        h₂ : ∀ (j : ι), c.Rel i j → CategoryTheory.Epi (HomologicalComplex.homologyMap …
        h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
        j : ι
        hij : c.Rel i j
        ⊢ CategoryTheory.Mono (CategoryTheory.ComposableArrows.app' ((CategoryTheory.C …
      -/
    · exact h₃ j hij
      /-
        🎉 no goals
      -/
    /-
      case neg
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
      φ : Quiver.Hom S₁ S₂
      hS₁ : S₁.ShortExact
      hS₂ : S₂.ShortExact
      i : ι
      h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₂ i)
      h₂ : ∀ (j : ι), c.Rel i j → CategoryTheory.Epi (HomologicalComplex.homologyMap …
      h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
      hi : Not (Exists fun j => c.Rel i j)
      ⊢ CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₃ i)
    -/
  · have := hS₂.epi_g
    /-
      case neg
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
      φ : Quiver.Hom S₁ S₂
      hS₁ : S₁.ShortExact
      hS₂ : S₂.ShortExact
      i : ι
      h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₂ i)
      h₂ : ∀ (j : ι), c.Rel i j → CategoryTheory.Epi (HomologicalComplex.homologyMap …
      h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
      hi : Not (Exists fun j => c.Rel i j)
      this : CategoryTheory.Epi S₂.g
      ⊢ CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₃ i)
    -/
    have eq := (homologyFunctor C _ i).congr_map φ.comm₂₃
    /-
      case neg
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
      φ : Quiver.Hom S₁ S₂
      hS₁ : S₁.ShortExact
      hS₂ : S₂.ShortExact
      i : ι
      h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₂ i)
      h₂ : ∀ (j : ι), c.Rel i j → CategoryTheory.Epi (HomologicalComplex.homologyMap …
      h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
      hi : Not (Exists fun j => c.Rel i j)
      this : CategoryTheory.Epi S₂.g
      eq : Eq ((HomologicalComplex.homologyFunctor C c i).map (CategoryTheory.Catego …
      ⊢ CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₃ i)
    -/
    dsimp at eq
    /-
      case neg
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
      φ : Quiver.Hom S₁ S₂
      hS₁ : S₁.ShortExact
      hS₂ : S₂.ShortExact
      i : ι
      h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₂ i)
      h₂ : ∀ (j : ι), c.Rel i j → CategoryTheory.Epi (HomologicalComplex.homologyMap …
      h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
      hi : Not (Exists fun j => c.Rel i j)
      this : CategoryTheory.Epi S₂.g
      eq : Eq (HomologicalComplex.homologyMap (CategoryTheory.CategoryStruct.comp φ. …
      ⊢ CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₃ i)
    -/
    simp only [homologyMap_comp] at eq
    /-
      case neg
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
      φ : Quiver.Hom S₁ S₂
      hS₁ : S₁.ShortExact
      hS₂ : S₂.ShortExact
      i : ι
      h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₂ i)
      h₂ : ∀ (j : ι), c.Rel i j → CategoryTheory.Epi (HomologicalComplex.homologyMap …
      h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
      hi : Not (Exists fun j => c.Rel i j)
      this : CategoryTheory.Epi S₂.g
      eq : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homologyMap φ. …
      ⊢ CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₃ i)
    -/
    have := epi_homologyMap_of_epi_of_not_rel S₂.g i (by simpa using hi)
    /-
      case neg
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
      φ : Quiver.Hom S₁ S₂
      hS₁ : S₁.ShortExact
      hS₂ : S₂.ShortExact
      i : ι
      h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₂ i)
      h₂ : ∀ (j : ι), c.Rel i j → CategoryTheory.Epi (HomologicalComplex.homologyMap …
      h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
      hi : Not (Exists fun j => c.Rel i j)
      this✝ : CategoryTheory.Epi S₂.g
      eq : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homologyMap φ. …
      this : CategoryTheory.Epi (HomologicalComplex.homologyMap S₂.g i)
      ⊢ CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₃ i)
    -/
    exact epi_of_epi_fac eq.symm
    /-
      🎉 no goals
    -/


lemma isIso_homologyMap_τ₃ (i : ι)
    (h₁ : Epi (homologyMap φ.τ₁ i))
    (h₂ : IsIso (homologyMap φ.τ₂ i))
    (h₃ : ∀ j, c.Rel i j → IsIso (homologyMap φ.τ₁ j))
    (h₄ : ∀ j, c.Rel i j → Mono (homologyMap φ.τ₂ j)) :
    IsIso (homologyMap φ.τ₃ i) := by
  have := mono_homologyMap_τ₃ φ hS₁ hS₂ i h₁ (IsIso.mono_of_iso _) (fun j hij => by
    have := h₃ j hij
    infer_instance)
  have := epi_homologyMap_τ₃ φ hS₁ hS₂ i inferInstance (fun j hij => by
    have := h₃ j hij
    infer_instance) h₄
  /-
    C : Type u_1
    ι : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    c : ComplexShape ι
    S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
    φ : Quiver.Hom S₁ S₂
    hS₁ : S₁.ShortExact
    hS₂ : S₂.ShortExact
    i : ι
    h₁ : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₁ i)
    h₂ : CategoryTheory.IsIso (HomologicalComplex.homologyMap φ.τ₂ i)
    h₃ : ∀ (j : ι), c.Rel i j → CategoryTheory.IsIso (HomologicalComplex.homologyM …
    h₄ : ∀ (j : ι), c.Rel i j → CategoryTheory.Mono (HomologicalComplex.homologyMa …
    this✝ : CategoryTheory.Mono (HomologicalComplex.homologyMap φ.τ₃ i)
    this : CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₃ i)
    ⊢ CategoryTheory.IsIso (HomologicalComplex.homologyMap φ.τ₃ i)
  -/
  apply isIso_of_mono_of_epi
  /-
    🎉 no goals
  -/


lemma quasiIso_τ₃ (h₁ : QuasiIso φ.τ₁) (h₂ : QuasiIso φ.τ₂) :
    QuasiIso φ.τ₃ := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    c : ComplexShape ι
    S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
    φ : Quiver.Hom S₁ S₂
    hS₁ : S₁.ShortExact
    hS₂ : S₂.ShortExact
    h₁ : QuasiIso φ.τ₁
    h₂ : QuasiIso φ.τ₂
    ⊢ QuasiIso φ.τ₃
  -/
  rw [quasiIso_iff]
  /-
    C : Type u_1
    ι : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    c : ComplexShape ι
    S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
    φ : Quiver.Hom S₁ S₂
    hS₁ : S₁.ShortExact
    hS₂ : S₂.ShortExact
    h₁ : QuasiIso φ.τ₁
    h₂ : QuasiIso φ.τ₂
    ⊢ ∀ (i : ι), QuasiIsoAt φ.τ₃ i
  -/
  intro i
  /-
    C : Type u_1
    ι : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    c : ComplexShape ι
    S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
    φ : Quiver.Hom S₁ S₂
    hS₁ : S₁.ShortExact
    hS₂ : S₂.ShortExact
    h₁ : QuasiIso φ.τ₁
    h₂ : QuasiIso φ.τ₂
    i : ι
    ⊢ QuasiIsoAt φ.τ₃ i
  -/
  rw [quasiIsoAt_iff_isIso_homologyMap]
  /-
    C : Type u_1
    ι : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    c : ComplexShape ι
    S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
    φ : Quiver.Hom S₁ S₂
    hS₁ : S₁.ShortExact
    hS₂ : S₂.ShortExact
    h₁ : QuasiIso φ.τ₁
    h₂ : QuasiIso φ.τ₂
    i : ι
    ⊢ CategoryTheory.IsIso (HomologicalComplex.homologyMap φ.τ₃ i)
  -/
  apply isIso_homologyMap_τ₃ φ hS₁ hS₂
  /-
    case h₁
    C : Type u_1
    ι : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    c : ComplexShape ι
    S₁ S₂ : CategoryTheory.ShortComplex (HomologicalComplex C c)
    φ : Quiver.Hom S₁ S₂
    hS₁ : S₁.ShortExact
    hS₂ : S₂.ShortExact
    h₁ : QuasiIso φ.τ₁
    h₂ : QuasiIso φ.τ₂
    i : ι
    ⊢ CategoryTheory.Epi (HomologicalComplex.homologyMap φ.τ₁ i)
  -/
  all_goals infer_instance
  /-
    🎉 no goals
  -/


